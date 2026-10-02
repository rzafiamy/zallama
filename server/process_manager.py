"""
process_manager.py — Zallama Process Manager

Manages inference subprocess lifecycle (backend-agnostic):
  - Spawning a server instance per model via its Backend
  - Health-checking until ready
  - Port assignment (with OS bind-check)
  - LRU eviction on idle timeout and on max-loaded cap
  - Graceful shutdown

What it does NOT know: how to build the command line or which binary to run for
a given model. That lives in backends.py, keyed off the model's `backend` field,
so adding TTS/ASR/image backends does not touch this file.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import socket
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from .backends import ASR, EMBEDDING, IMAGE, NORMALIZATION, RERANK, TEXT, TRANSLATION, TTS, Backend, get_backend
from .config import resolve_binary

logger = logging.getLogger("zallama.process_manager")

# Substrings (lowercased) that mark an allocation failure in a backend's startup
# log: CUDA/ggml ("out of memory", "cudaMalloc failed"), ONNX Runtime's BFC
# arena ("Failed to allocate memory"), torch/stable-diffusion variants.
_OOM_MARKERS = (
    "out of memory",
    "failed to allocate",
    "unable to allocate",
    "cudamalloc failed",
    "cuda_error_out_of_memory",
    "error_out_of_device_memory",
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class BackendOOMError(RuntimeError):
    """A backend died during startup because it could not allocate memory.

    Retrying the same request is pointless until memory is freed, so the
    message spells out what is resident and what the caller can do instead.
    """


# ---------------------------------------------------------------------------
# Data model for a running model instance
# ---------------------------------------------------------------------------
class ModelInstance:
    def __init__(
        self,
        name: str,
        port: int,
        process: asyncio.subprocess.Process,
        log_file: Path,
        entry: dict,
        backend: Backend,
        mem_gb: float = 0.0,
    ):
        self.name = name
        self.port = port
        self.process = process
        self.log_file = log_file
        self.entry = entry
        self.backend = backend
        self.mem_gb = mem_gb  # declared/estimated memory cost
        self.started_at = time.time()
        self.last_used = time.time()
        self.base_url = f"http://127.0.0.1:{port}"
        # In-flight request accounting. `active` is the number of proxied
        # requests currently reading from this backend; `_idle_event` is set
        # whenever `active` drops to 0. Eviction consults these so a backend is
        # not killed out from under a request being streamed through it (which
        # would 502 the client mid-response), and idle-sweep skips a backend
        # that is still serving a long generation.
        self.active = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    def touch(self):
        self.last_used = time.time()

    def acquire(self):
        """Mark one proxied request as in-flight against this backend."""
        self.active += 1
        self._idle_event.clear()
        self.last_used = time.time()

    def release(self):
        """Release an in-flight request; wake eviction waiters when idle."""
        self.active = max(0, self.active - 1)
        self.last_used = time.time()
        if self.active == 0:
            self._idle_event.set()

    async def wait_idle(self, timeout: float) -> bool:
        """Block until no request is in-flight, or `timeout` seconds elapse.

        Returns True if the backend went idle, False on timeout.
        """
        if self.active == 0:
            return True
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def is_alive(self) -> bool:
        return self.process.returncode is None


# ---------------------------------------------------------------------------
# Process Manager
# ---------------------------------------------------------------------------
class ProcessManager:
    def __init__(self, cfg: dict, logs_dir: str):
        self.cfg = cfg
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._instances: OrderedDict[str, ModelInstance] = OrderedDict()
        self._global_lock = asyncio.Lock()
        # Capacity admission covers eviction, shutdown, and startup. Keeping it
        # separate from _global_lock lets normal manager operations proceed
        # while a backend is becoming healthy, without allowing concurrent
        # starts to over-commit the configured limits.
        self._admission_lock = asyncio.Lock()
        # Per-model locks so booting one model never blocks requests to another.
        self._model_locks: dict[str, asyncio.Lock] = {}
        # Starts in progress, one per model, awaited by every caller (see
        # get_or_start): a caller that gets cancelled mid-start must not abort
        # a start that other callers, or the next request, still need.
        self._starting: dict[str, asyncio.Task] = {}
        self._binary_cache: dict[str, str] = {}

        ls = cfg["llama_server"]
        self._port_counter = ls["port_start"]
        self._idle_timeout: int = ls.get("idle_timeout", 300)
        self._startup_timeout: int = ls.get("startup_timeout", 60)
        self._max_loaded: int = ls.get("max_loaded_models", 0)  # 0 = unlimited
        self._mem_budget_gb: float = float(ls.get("mem_budget_gb", 0))  # 0 = unlimited
        # Live + recent request metrics for `zallama monitor` (/api/requests).
        from .request_log import RequestLog
        self.request_log = RequestLog()
        # Lifecycle counters for /metrics, keyed by event then model name. They
        # deliberately outlive the instances they describe: the interesting
        # question ("how many times did this model crash today?") is asked
        # after the instance is gone. Events: start, start_failure (died or
        # timed out before /health), crash (found dead on the next request),
        # evict_capacity (LRU eviction to admit another model), evict_idle
        # (idle sweep), unload (explicit stop via API/CLI).
        from collections import Counter, defaultdict
        self.lifecycle: dict[str, Counter] = defaultdict(Counter)
        self._mem_init_gb: float = float(ls.get("mem_init_gb", 2))      # fallback cost
        # How long eviction waits for a victim backend's in-flight requests to
        # finish before killing it anyway. 0 disables the wait (old behavior:
        # evict immediately, 502-ing whatever was mid-flight). The wait only
        # happens when the LRU victim is actually busy *and* no idle victim
        # exists, so it does not slow the common case.
        self._evict_drain_timeout: float = float(ls.get("evict_drain_timeout", 30))
        # Per-evict_group memory budget (GB), e.g. {"services": 1.5} — mirrors
        # mem_budget_gb but scoped to one group, so ASR/embedding/autocomplete
        # (or whatever shares "services") can freely coexist as long as their
        # combined mem_gb fits the group's own budget, and only evict their LRU
        # member once a new arrival would push the group over it. This is
        # deliberately a memory cap, not an instance-count cap: capping by count
        # would block a second small model from loading even when VRAM is
        # plentiful, which isn't the goal — the goal is just to stop the group
        # from growing past what its budget allows. It's checked independent of
        # the global max_loaded_models/mem_budget_gb, which only pressure
        # eviction once the *global* count/budget is hit — leaving slack for
        # several same-group models to coexist unchecked whenever fewer
        # "primary" models are resident than the global cap assumes.
        self._group_mem_budgets: dict[str, float] = {
            k: float(v) for k, v in (ls.get("evict_group_mem_budgets") or {}).items()
        }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def _lock_for(self, model_name: str) -> asyncio.Lock:
        lock = self._model_locks.get(model_name)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_name] = lock
        return lock

    @asynccontextmanager
    async def serving(self, inst: ModelInstance, endpoint: str | None = None,
                      stream: bool = False):
        """Scope a proxied request against `inst` so eviction won't kill the
        backend while the request is still reading from it.

        Routes wrap their upstream call in this; for streaming responses the
        scope must span the generator's lifetime, not just the handler.

        With `endpoint` given, the request is also logged for `zallama monitor`
        (`/api/requests`): registered as in flight on entry, finalised on exit
        with the status the block ended in. The yielded value is then the
        RequestRecord, so a route can attach llama.cpp's timings to it.
        """
        inst.acquire()
        rec = self.request_log.start(inst.name, endpoint, stream) if endpoint else None
        try:
            yield rec if rec is not None else inst
        except BaseException as e:
            if rec is not None:
                self.request_log.finish(rec, error=f"{type(e).__name__}: {e}")
            raise
        else:
            if rec is not None:
                self.request_log.finish(rec)
        finally:
            inst.release()

    # A declared mem_gb below this is a model that holds no VRAM (a CPU
    # backend, e.g. pocket-tts or parakeet with `device: cpu`, declared 0.01).
    _FREE_MEM_GB = 0.05

    @classmethod
    def _is_lightweight(cls, entry: dict) -> bool:
        """A lightweight model sits outside capacity accounting: it never
        counts toward max_loaded_models or any memory budget, so loading it
        can't evict anything. That is a lightweight backend (tn-server: a few
        MB of CPU memory, ~1 ms per request) or an entry declaring a mem_gb
        under 0.05 GB (a CPU backend, no VRAM). Counting those made the voice
        stack (LLM + CPU ASR + CPU TTS) hit max_loaded_models: 4 and evict its
        own LLM on every TTS load."""
        from .model_registry import ModelRegistry  # local import to avoid cycle

        try:
            declared = float(entry.get("mem_gb") or 0)
        except (TypeError, ValueError):
            declared = 0.0
        if 0 < declared < cls._FREE_MEM_GB:
            return True
        try:
            return bool(getattr(get_backend(ModelRegistry.backend_of(entry)), "lightweight", False))
        except Exception:
            return False

    @staticmethod
    def _is_pinned(entry: dict) -> bool:
        """A pinned model is pre-warmed at startup and never evicted.

        Pinning exists for backends whose cold load is slow and not GPU-bound
        (e.g. kokoro-server TTS loads ONNX models on CPU in tens of seconds). For
        those, paying the load once at startup and keeping the instance resident
        turns a 50s first-request into a sub-second one — at the cost of holding
        the model's memory for the process lifetime, which is the intended trade.
        """
        return bool(entry.get("pinned"))

    # A model with no explicit `evict_group` in registry.yaml falls back to
    # this default, keyed by modality. text/image are large and slow to
    # reload, so they get their own group ("primary") that only they can
    # evict from; asr/embedding/rerank are small services that trade a
    # shared slot with each other ("services") without ever reaching into
    # "primary". tts defaults into "services" too, though in practice it's
    # almost always `pinned`, which exempts it from eviction entirely.
    _DEFAULT_EVICT_GROUP = {
        TEXT: "primary",
        IMAGE: "primary",
        ASR: "services",
        EMBEDDING: "services",
        RERANK: "services",
        TTS: "services",
        TRANSLATION: "services",
        NORMALIZATION: "services",
    }

    @classmethod
    def evict_group_of(cls, entry: dict) -> str | None:
        """Effective eviction group for an entry: explicit `evict_group`,
        else a modality-based default (see `_DEFAULT_EVICT_GROUP`).

        Entries sharing a group can evict each other but never reach outside
        the group — e.g. asr/embedding trade a shared slot without ever
        evicting a text/image model. Set `evict_group` explicitly to opt out
        of the default (e.g. group a specific model with "primary" even
        though its modality would default elsewhere) or to invent a new,
        unrelated group.

        Public (not just used internally by `_make_room_locked`) so callers
        like the model-info/list route can show the group actually in effect
        — the raw `entry.get("evict_group")` is None for the vast majority of
        entries that just take the modality default, which isn't useful to
        display on its own.
        """
        if "evict_group" in entry:
            # Explicitly set (including "" or null): honor it as-is, even to
            # opt out of the modality default (empty/null -> no group, the
            # old any-non-pinned-victim behavior).
            g = entry.get("evict_group")
            return str(g) if g else None
        modality = (entry.get("modality") or TEXT).strip().lower()
        return cls._DEFAULT_EVICT_GROUP.get(modality)

    async def get_or_start(self, model_name: str, entry: dict, model_path: Path) -> ModelInstance:
        """Return running instance for model, starting it if necessary.

        The registry's canonical name is the instance identity. This keeps an
        alias and its canonical name from acquiring separate locks or starting
        separate backend processes. Starts are serialized only when a capacity
        limit is configured, so admission cannot oversubscribe the limit.
        """
        model_name = entry.get("name") or model_name
        # Fast path: already running.
        async with self._global_lock:
            inst = self._instances.get(model_name)
            if inst is not None and inst.is_alive():
                inst.touch()
                self._instances.move_to_end(model_name)
                return inst

        # Slow path, in a task shared by every caller and shielded from their
        # cancellation. A realtime turn dropped mid-start (the user spoke
        # again) used to cancel the start after the process was spawned but
        # before it was registered: the process was orphaned, still holding
        # its VRAM, and the next request spawned a second one.
        task = self._starting.get(model_name)
        if task is None:
            task = asyncio.ensure_future(self._start(model_name, entry, model_path))
            self._starting[model_name] = task

            def _done(t: asyncio.Task, name: str = model_name) -> None:
                if self._starting.get(name) is t:
                    del self._starting[name]
                if not t.cancelled() and t.exception() is not None:
                    logger.debug(f"Start of '{name}' failed: {t.exception()}")
            task.add_done_callback(_done)
        return await asyncio.shield(task)

    async def _start(self, model_name: str, entry: dict, model_path: Path) -> ModelInstance:
        """get_or_start's slow path: start the model unless it got started
        meanwhile. Serialized per model."""
        async with self._lock_for(model_name):
            async with self._global_lock:
                inst = self._instances.get(model_name)
                if inst is not None:
                    if inst.is_alive():
                        inst.touch()
                        self._instances.move_to_end(model_name)
                        return inst
                    logger.warning(f"Instance {model_name} died unexpectedly, restarting...")
                    self.lifecycle["crash"][model_name] += 1
                    del self._instances[model_name]

            incoming_cost = self._estimate_cost(entry, model_path)
            incoming_group = self.evict_group_of(entry)
            has_group_budget = incoming_group is not None and incoming_group in self._group_mem_budgets
            has_capacity_limit = (
                self._max_loaded > 0 or self._mem_budget_gb > 0 or has_group_budget
            ) and not self._is_lightweight(entry)

            if has_capacity_limit:
                # Keep the reservation from capacity check through registration.
                # The global lock remains short-lived; killing and health checks
                # happen without it.
                async with self._admission_lock:
                    async with self._global_lock:
                        evicted = self._make_room_locked(incoming_cost, entry)
                    for victim in evicted:
                        await self._kill_instance(
                            victim, drain_timeout=self._evict_drain_timeout
                        )
                    inst = await self._spawn(model_name, entry, model_path, incoming_cost)
                    await self._register(model_name, inst)
                    return inst

            inst = await self._spawn(model_name, entry, model_path, incoming_cost)
            await self._register(model_name, inst)
            return inst

    async def _register(self, model_name: str, inst: ModelInstance) -> None:
        """Record a started instance. Never overwrite a live one silently: the
        overwritten process would keep running, unaccounted and unevictable."""
        async with self._global_lock:
            old = self._instances.get(model_name)
            self._instances[model_name] = inst
        if old is not None and old is not inst and old.is_alive():
            logger.warning(f"'{model_name}' was already running on port {old.port}; "
                           f"stopping that duplicate (now on port {inst.port})")
            await self._kill_instance(old)

    async def prewarm_pinned(self, registry) -> None:
        """Start every pinned model so its slow cold load happens at boot.

        Pinned models (e.g. kokoro-server TTS, whose ONNX load is a slow CPU
        operation) are loaded here once, off the request path, and then kept
        resident by the eviction exemptions. A failure to warm one model is
        logged and skipped — it must not block startup or the other models.
        """
        pinned = [e for e in registry.list_models() if self._is_pinned(e)]
        if not pinned:
            return
        logger.info(f"Pre-warming {len(pinned)} pinned model(s)...")
        for entry in pinned:
            name = entry["name"]
            try:
                model_path = registry.resolve_path(entry)
                await self.get_or_start(name, entry, model_path)
                logger.info(f"Pinned model '{name}' is warm")
            except Exception as e:
                logger.warning(f"Failed to pre-warm pinned model '{name}': {e}")

    async def stop(self, model_name: str) -> bool:
        """Stop a running model instance."""
        async with self._global_lock:
            if model_name not in self._instances:
                return False
            inst = self._instances.pop(model_name)
        self.lifecycle["unload"][model_name] += 1
        await self._kill_instance(inst)
        return True

    def is_running(self, model_name: str) -> bool:
        """True if the model has a live instance."""
        inst = self._instances.get(model_name)
        return inst is not None and inst.is_alive()

    def list_running(self) -> list[dict]:
        """Return info about all running instances."""
        by_pid = self._gpu_used_by_pid()
        result = []
        for name, inst in self._instances.items():
            result.append({
                "name": name,
                "port": inst.port,
                "base_url": inst.base_url,
                "modality": inst.entry.get("modality", "text"),
                "backend": inst.backend.name,
                "mem_gb": round(inst.mem_gb, 2),
                # Measured GPU VRAM via nvidia-smi; None for CPU-only backends
                # or when nvidia-smi is unavailable. This is the *real*
                # footprint, vs. mem_gb which is the launch-time estimate.
                "vram_gb": self._vram_for_instance(inst, by_pid),
                "started_at": inst.started_at,
                "last_used": inst.last_used,
                "alive": inst.is_alive(),
                # Proxied requests currently reading from this backend.
                "active": inst.active,
            })
        return result

    def memory_status(self) -> dict:
        """Loaded memory vs. configured budget (GB)."""
        used = round(self._loaded_mem_gb(), 2)
        by_pid = self._gpu_used_by_pid()
        measured = [self._vram_for_instance(i, by_pid) for i in self._instances.values()]
        vram_total = round(sum(v for v in measured if v is not None), 2) if by_pid else None
        return {
            "loaded_gb": used,
            "vram_used_gb": vram_total,  # measured GPU total; None if unavailable
            "budget_gb": self._mem_budget_gb,
            "headroom_gb": round(self._mem_budget_gb - used, 2) if self._mem_budget_gb > 0 else None,
            "max_loaded_models": self._max_loaded,
            "loaded_count": len(self._instances),
        }

    async def shutdown_all(self):
        """Gracefully stop all running instances."""
        async with self._global_lock:
            instances = list(self._instances.values())
            self._instances.clear()
        for inst in instances:
            await self._kill_instance(inst)

    async def sweep_idle(self):
        """Background task: evict models idle longer than idle_timeout."""
        if self._idle_timeout <= 0:
            return
        async with self._global_lock:
            now = time.time()
            to_evict = [
                name for name, inst in self._instances.items()
                if not self._is_pinned(inst.entry)
                and inst.active == 0
                and (now - inst.last_used) > self._idle_timeout
            ]
            evicted = [self._instances.pop(name) for name in to_evict]
        for inst in evicted:
            logger.info(f"Evicting idle model: {inst.name}")
            self.lifecycle["evict_idle"][inst.name] += 1
            await self._kill_instance(inst)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _estimate_cost(self, entry: dict, model_path: Path) -> float:
        """Estimated memory cost (GB) of running a model.

        Prefers the declared `mem_gb`; otherwise approximates from the GGUF file
        size (a loaded GGUF roughly occupies its on-disk size plus KV-cache
        overhead, so file_size * 1.2 is a reasonable default); falls back to
        the configured mem_init_gb when the size is unknown.
        """
        declared = entry.get("mem_gb")
        if declared:
            try:
                return float(declared)
            except (TypeError, ValueError):
                pass
        try:
            size_gb = model_path.stat().st_size / 1e9
            if size_gb > 0:
                return round(size_gb * 1.2, 2)
        except OSError:
            pass
        return self._mem_init_gb

    def _loaded_mem_gb(self) -> float:
        return sum(inst.mem_gb for inst in self._instances.values())

    @staticmethod
    def _gpu_used_by_pid() -> dict[int, float]:
        """Map of {pid: VRAM GB} for GPU compute processes, via nvidia-smi.

        Returns the *measured* per-process VRAM footprint so callers can show
        real usage instead of (or alongside) the static launch-time estimate.
        Empty dict if nvidia-smi is unavailable or no NVIDIA GPU is present —
        CPU-only backends (e.g. kokoro ONNX) simply won't appear.
        """
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return {}
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            )
        except Exception:
            return {}
        result: dict[int, float] = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
                mib = float(parts[1])
            except ValueError:
                continue
            result[pid] = mib / 1024.0  # MiB -> GiB
        return result

    def _vram_for_instance(self, inst: ModelInstance, by_pid: dict[int, float]) -> float | None:
        """Measured VRAM (GB) for one instance, or None if it has no GPU usage.

        Matches the backend's own PID first; falls back to summing any GPU
        process living in the instance's process group (a backend may spawn
        worker PIDs under the same group created by os.setsid).
        """
        if not by_pid:
            return None
        try:
            pid = inst.process.pid
        except Exception:
            return None
        if pid in by_pid:
            return round(by_pid[pid], 2)
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            return round(by_pid[pid], 2) if pid in by_pid else None
        total = 0.0
        matched = False
        for gpu_pid, gb in by_pid.items():
            try:
                if os.getpgid(gpu_pid) == pgid:
                    total += gb
                    matched = True
            except (ProcessLookupError, PermissionError):
                continue
        return round(total, 2) if matched else None

    def _make_room_locked(self, incoming_cost: float, entry: dict) -> list[ModelInstance]:
        """Evict LRU instances until an incoming model fits both budgets.

        Caller holds the global lock. Count budget: keep loaded count below
        max_loaded. Memory budget: keep loaded + incoming within mem_budget_gb.
        Group budget: keep the incoming model's own evict_group's *own* mem_gb
        total (not the global total) within its configured
        `evict_group_mem_budgets` entry, if any — checked regardless of the
        global count/budget, so e.g. a 1.5GB "services" budget is enforced even
        when max_loaded_models has slack (only one "primary" model loaded,
        leaving multiple global slots free that would otherwise let several
        "services" models coexist without ever being pressured to evict each
        other). This is a memory cap, not an instance-count cap: two small
        models that both fit the group's budget are free to coexist — eviction
        only fires once an arrival would actually push the group's own total
        over its own budget, never just because a second one showed up.
        Eviction always targets the least-recently-used instance first. Victims
        are removed here but killed by the caller after releasing the lock.

        If the incoming entry declares `evict_group`, eviction is restricted to
        other loaded instances in that same group — e.g. an asr/embedding pair
        sharing a group can trade a slot back and forth without ever reaching
        into a text/image model's group. Ungrouped entries keep the old
        behavior of evicting (and being evicted by) anything non-pinned.
        """
        evicted: list[ModelInstance] = []
        incoming_group = self.evict_group_of(entry)
        group_budget = self._group_mem_budgets.get(incoming_group) if incoming_group is not None else None

        def _group_mem_gb() -> float:
            return sum(
                inst.mem_gb for inst in self._instances.values()
                if self.evict_group_of(inst.entry) == incoming_group
            )

        def over_count() -> bool:
            counted = sum(1 for i in self._instances.values() if not self._is_lightweight(i.entry))
            return self._max_loaded > 0 and counted >= self._max_loaded

        def over_mem() -> bool:
            return (
                self._mem_budget_gb > 0
                and (self._loaded_mem_gb() + incoming_cost) > self._mem_budget_gb
            )

        def over_group() -> bool:
            return group_budget is not None and (_group_mem_gb() + incoming_cost) > group_budget

        def next_victim() -> str | None:
            # LRU is first; skip pinned models — they are never evicted, even
            # under capacity pressure. If only pinned models remain, give up and
            # let the incoming model exceed the budget rather than killing a
            # warm-pinned instance.
            #
            # Among eligible victims, prefer one with no in-flight requests so a
            # backend that is mid-response is only chosen when it is the sole
            # candidate (and even then the caller drains it before killing).
            busy_fallback: str | None = None
            for name, inst in self._instances.items():
                if self._is_pinned(inst.entry) or self._is_lightweight(inst.entry):
                    continue
                if incoming_group is not None and self.evict_group_of(inst.entry) != incoming_group:
                    continue
                if inst.active > 0:
                    if busy_fallback is None:
                        busy_fallback = name
                    continue
                return name
            return busy_fallback

        while self._instances and (over_count() or over_mem() or over_group()):
            victim_name = next_victim()
            if victim_name is None:
                reason = (
                    f"all loaded models are pinned or outside group '{incoming_group}'"
                    if incoming_group is not None
                    else "all loaded models are pinned"
                )
                logger.warning(
                    f"Capacity reached but {reason} — "
                    f"admitting incoming {incoming_cost:.1f}GB over budget."
                )
                break
            # Decide the reason before popping: afterwards the check may no longer hold.
            reason = "count" if over_count() else ("memory" if over_mem() else f"group '{incoming_group}' budget {group_budget}GB")
            victim = self._instances.pop(victim_name)
            logger.info(
                f"Capacity ({reason}) reached — evicting LRU model "
                f"'{victim_name}' ({victim.mem_gb:.1f}GB) to make room "
                f"for incoming {incoming_cost:.1f}GB"
            )
            self.lifecycle["evict_capacity"][victim_name] += 1
            evicted.append(victim)
        return evicted

    def _binary_for(self, backend: Backend) -> str:
        cached = self._binary_cache.get(backend.name)
        if cached:
            return cached
        binary = resolve_binary(self.cfg, backend.binary_name)
        self._binary_cache[backend.name] = binary
        return binary

    def _next_port(self) -> int:
        """Pick the next free port, skipping ours and anything the OS holds."""
        used_ports = {inst.port for inst in self._instances.values()}
        port = self._port_counter
        while port in used_ports or not self._port_free(port):
            port += 1
        self._port_counter = port + 1
        return port

    @staticmethod
    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def merged_params(self, entry: dict) -> dict:
        """Registry params layered over llama_server.default_params.

        Public so callers outside the spawn path (e.g. the model-info route)
        can see the effective ctx_size etc. without duplicating the merge.
        """
        default_params = self.cfg["llama_server"]["default_params"].copy()
        model_params = entry.get("params", {})
        return {**default_params, **model_params}

    async def _spawn(
        self, model_name: str, entry: dict, model_path: Path, mem_gb: float = 0.0
    ) -> ModelInstance:
        """Spawn a new inference server process for the model's backend."""
        from .dependencies import get_registry  # local import to avoid cycle
        from .model_registry import ModelRegistry  # local import to avoid cycle

        # Resolve via backend_of so legacy embedding entries (text +
        # params.embedding=true) map onto the embedding-server backend.
        backend = get_backend(ModelRegistry.backend_of(entry))
        binary = self._binary_for(backend)
        artifacts = get_registry().resolve_artifacts(entry)
        merged = self.merged_params(entry)

        port = self._next_port()
        log_path = self.logs_dir / f"{model_name.replace(':', '_')}.log"
        # The log is appended across spawns; remember where this run starts so
        # a startup failure is diagnosed from this run's output only.
        try:
            log_offset = log_path.stat().st_size
        except OSError:
            log_offset = 0
        args = backend.build_args(binary, port, model_path, entry, merged, artifacts)

        logger.info(f"Spawning {backend.name} for '{model_name}' on port {port}")
        logger.debug(f"Command: {' '.join(args)}")

        log_file = open(log_path, "ab")
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=log_file,
                stderr=log_file,
                preexec_fn=os.setsid,
            )
        finally:
            log_file.close()

        inst = ModelInstance(
            name=model_name,
            port=port,
            process=process,
            log_file=log_path,
            entry=entry,
            backend=backend,
            mem_gb=mem_gb,
        )

        try:
            await self._wait_healthy(inst, log_offset)
        except BaseException:
            # Whatever stopped the start (failure, timeout, cancellation), the
            # process must not outlive it unregistered.
            self.lifecycle["start_failure"][model_name] += 1
            if inst.process.returncode is None:
                try:
                    os.killpg(os.getpgid(inst.process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            raise
        self.lifecycle["start"][model_name] += 1
        logger.info(f"Model '{model_name}' is ready on port {port}")
        return inst

    async def _wait_healthy(self, inst: ModelInstance, log_offset: int = 0):
        """Poll the backend's health path until the server is ready."""
        health_url = f"{inst.base_url}{inst.backend.health_path()}"
        timeout = self._startup_timeout * getattr(
            inst.backend, "startup_timeout_factor", 1.0
        )
        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if not inst.is_alive():
                    raise self._startup_failure(inst, log_offset)
                try:
                    r = await client.get(health_url)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        # Time out: kill the half-started process so it doesn't linger.
        await self._kill_instance(inst)
        raise TimeoutError(
            f"{inst.backend.name} for '{inst.name}' did not become healthy within "
            f"{timeout:.0f}s. Check logs: {inst.log_file}"
        )

    @staticmethod
    def _startup_log_lines(inst: ModelInstance, log_offset: int) -> list[str]:
        """Non-empty, ANSI-stripped lines this spawn wrote to its log (last 64KB)."""
        try:
            with open(inst.log_file, "rb") as f:
                f.seek(max(log_offset, inst.log_file.stat().st_size - 65536))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        return [ln for ln in (_ANSI_RE.sub("", l).strip() for l in text.splitlines()) if ln]

    @staticmethod
    def _gpu_free_total_gb() -> tuple[float, float] | None:
        """(free, total) GB on the first GPU via nvidia-smi, or None."""
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return None
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            )
            free, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        except Exception:
            return None
        return free / 1024.0, total / 1024.0

    def _startup_failure(self, inst: ModelInstance, log_offset: int) -> RuntimeError:
        """Build the error for a backend that exited before becoming healthy.

        Callers (often agents) only see this message in the 503 body, so it
        carries the backend's own last error line and, for an out-of-memory
        death, what is holding the memory and which alternatives exist —
        instead of a bare "check logs" that invites blind retries.
        """
        lines = self._startup_log_lines(inst, log_offset)
        oom_lines = [ln for ln in lines if any(m in ln.lower() for m in _OOM_MARKERS)]
        head = f"{inst.backend.name} for '{inst.name}' died during startup"
        if not oom_lines:
            last = f" Last log line: {lines[-1][:400]}" if lines else ""
            return RuntimeError(f"{head}.{last} Full log: {inst.log_file}")

        on_cpu = str(self.merged_params(inst.entry).get("device", "")).lower() == "cpu"
        where = "system RAM" if on_cpu else "GPU memory"
        parts = [f"{head}: out of {where} (needs ~{inst.mem_gb:.1f} GB)."]
        gpu = None if on_cpu else self._gpu_free_total_gb()
        if gpu is not None:
            parts.append(f"GPU has {gpu[0]:.1f} GB free of {gpu[1]:.1f} GB.")
        err = oom_lines[-1]
        # Long ONNX/CUDA lines put the useful part ("Failed to allocate ...") at the end.
        parts.append(f"Backend error: {err if len(err) <= 300 else '...' + err[-300:]}")

        group = self.evict_group_of(inst.entry)
        by_pid = self._gpu_used_by_pid()
        resident = []
        for name, other in self._instances.items():
            if name == inst.name or not other.is_alive():
                continue
            other_group = self.evict_group_of(other.entry)
            pinned = self._is_pinned(other.entry)
            if group is not None and other_group == group and not pinned:
                continue  # same group: was evictable, so not what blocked it
            vram = self._vram_for_instance(other, by_pid)
            gb = vram if vram is not None else other.mem_gb
            why = "pinned" if pinned else f"group '{other_group}'"
            resident.append((gb, name, f"'{name}' ({gb:.1f} GB, {why})"))
        if resident:
            resident.sort(reverse=True)
            parts.append(
                f"Loaded models it could not evict (it is in group '{group}'): "
                + ", ".join(d for _, _, d in resident) + "."
            )

        parts.append("Retrying the same request will fail the same way until memory is freed.")
        options = []
        try:
            from .dependencies import get_registry  # local import to avoid cycle
            from .model_registry import ModelRegistry  # local import to avoid cycle

            modality = ModelRegistry.modality_of(inst.entry)
            cpu_alts = [
                e["name"] for e in get_registry().list_models()
                if e.get("name") != inst.name
                and ModelRegistry.modality_of(e) == modality
                and str(self.merged_params(e).get("device", "")).lower() == "cpu"
            ]
        except Exception:
            cpu_alts = []
        if cpu_alts and not on_cpu:
            options.append("use a CPU model instead: " + ", ".join(f"'{n}'" for n in cpu_alts))
        if resident:
            admin_port = self.cfg.get("zallama", {}).get("admin_port")
            options.append(
                f"free memory by unloading a model via the admin API "
                f"(POST /api/models/{resident[0][1]}/unload on port {admin_port}), then retry"
            )
        if options:
            parts.append("Options: " + "; ".join(options) + ".")
        parts.append(f"Full log: {inst.log_file}")
        return BackendOOMError(" ".join(parts))

    async def _kill_instance(self, inst: ModelInstance, drain_timeout: float = 0.0):
        """Gracefully terminate a process group.

        When `drain_timeout` > 0 and the instance still has in-flight requests,
        wait up to that many seconds for them to finish first, so eviction does
        not cut a response off mid-stream. If they don't drain in time, evict
        anyway — a bounded wait beats an unbounded stall.
        """
        if drain_timeout > 0 and inst.active > 0:
            if not await inst.wait_idle(drain_timeout):
                logger.warning(
                    f"Evicting '{inst.name}' with {inst.active} in-flight "
                    f"request(s) still active after {drain_timeout:.0f}s drain wait"
                )
        try:
            if inst.process.returncode is None:
                os.killpg(os.getpgid(inst.process.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(inst.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    os.killpg(os.getpgid(inst.process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        logger.info(f"Stopped model '{inst.name}'")
