"""
server/metrics.py — Prometheus text exposition for GET /metrics

Renders the daemon's existing state (ProcessManager instances + lifecycle
counters, RequestLog aggregates, the whole-GPU numbers nvidia-smi reports)
in the Prometheus text format, by hand. There is nothing here that needs
prometheus_client: a handful of gauges, counters and two histograms, all of
which already exist as plain Python numbers — a dependency would only add a
registry to keep in sync with the objects that are the source of truth.

Conventions: base units (bytes, seconds, ratios 0–1) with the unit in the
metric name, `_total` on counters, one `model` label carrying the registry
name. Anything that is a launch-time *estimate* (mem_gb) is named
`_declared_` so it can't be confused with a measurement.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from . import __version__
from .backends import LlamaServerBackend, get_backend
from .model_registry import ModelRegistry

GIB = 1024 ** 3


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**kw) -> str:
    items = [(k, v) for k, v in kw.items() if v is not None]
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_esc(str(v))}"' for k, v in items) + "}"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


class _Out:
    """Collects samples per metric and emits each metric as one contiguous
    block (HELP, TYPE, samples) in the order the heads were declared — the
    text format requires that grouping, and it lets the render loops below
    walk the data once instead of once per metric."""

    def __init__(self):
        self.order: list[str] = []
        self.heads: dict[str, tuple[str, str]] = {}
        self.samples: dict[str, list[str]] = {}

    def head(self, name: str, kind: str, help_: str) -> None:
        if name not in self.heads:
            self.order.append(name)
            self.heads[name] = (kind, help_)
            self.samples[name] = []

    def sample(self, name: str, value, /, **labels) -> None:
        self.samples[name].append(f"{name}{_labels(**labels)} {_fmt(value)}")

    def text(self) -> str:
        lines: list[str] = []
        for name in self.order:
            kind, help_ = self.heads[name]
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} {kind}")
            lines.extend(self.samples[name])
        return "\n".join(lines) + "\n"

    def histogram(self, name: str, hist, /, **labels) -> None:
        out = self.samples[name]
        for edge, count in zip(hist.edges, hist.counts):
            out.append(f"{name}_bucket{_labels(**labels, le=_fmt(edge))} {count}")
        out.append(f"{name}_bucket{_labels(**labels, le='+Inf')} {hist.count}")
        out.append(f"{name}_sum{_labels(**labels)} {_fmt(hist.total)}")
        out.append(f"{name}_count{_labels(**labels)} {hist.count}")


def gpu_totals() -> list[dict]:
    """Whole-card numbers, one dict per GPU; [] without nvidia-smi.

    The per-process figures in /api/ps say what *our* backends hold; this is
    what the card as a whole has left, which is the number that decides
    whether the next model loads.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "used_bytes": int(float(parts[2]) * 1024 * 1024),
                "total_bytes": int(float(parts[3]) * 1024 * 1024),
                "util": _maybe_float(parts[4]),
                "temp_c": _maybe_float(parts[5]),
                "power_w": _maybe_float(parts[6]),
            })
        except ValueError:
            continue
    return gpus


def _maybe_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:  # "[N/A]" on some cards/drivers
        return None


def render(pm, registry) -> str:
    """Build the /metrics body from live daemon state."""
    o = _Out()
    now = time.time()
    rl = pm.request_log

    o.head("zallama_info", "gauge", "Daemon version (always 1).")
    o.sample("zallama_info", 1, version=__version__)
    o.head("zallama_uptime_seconds", "gauge", "Seconds since the daemon started.")
    o.sample("zallama_uptime_seconds", now - rl.started)

    # -- registry / capacity ---------------------------------------------
    entries = registry.list_models()
    running = pm.list_running()
    mem = pm.memory_status()
    o.head("zallama_models_registered", "gauge", "Models in the registry.")
    o.sample("zallama_models_registered", len(entries))
    o.head("zallama_models_loaded", "gauge", "Backend processes currently running.")
    o.sample("zallama_models_loaded", len(running))
    o.head("zallama_mem_budget_bytes", "gauge",
           "llama_server.mem_budget_gb (0 = unlimited); declared mem_gb of loaded models must fit it.")
    o.sample("zallama_mem_budget_bytes", int(mem["budget_gb"] * GIB))
    o.head("zallama_mem_declared_bytes", "gauge",
           "Sum of declared mem_gb over loaded models — the daemon's own accounting, not a measurement.")
    o.sample("zallama_mem_declared_bytes", int(mem["loaded_gb"] * GIB))
    if mem.get("vram_used_gb") is not None:
        o.head("zallama_vram_used_bytes", "gauge",
               "Measured VRAM held by zallama's backend processes (nvidia-smi per-process).")
        o.sample("zallama_vram_used_bytes", int(mem["vram_used_gb"] * GIB))
    o.head("zallama_group_mem_budget_bytes", "gauge",
           "Per-evict_group memory budget (llama_server.evict_group_mem_budgets).")
    for name, budget in getattr(pm, "_group_mem_budgets", {}).items():
        o.sample("zallama_group_mem_budget_bytes", int(budget * GIB), group=name)

    # -- registry: every model, loaded or not, with its effective params ----
    # `_info` carries the identity; `_param` is one sample per effective
    # launch parameter (registry `params` layered over default_params, i.e.
    # what `zallama show` prints), as {param, value} labels — so ctx_size,
    # cache_type_k/v, spec_type... can be joined onto any per-model series,
    # and a re-tune shows up in the graph as a label change. Numeric params
    # are additionally exposed as real gauges so they can be plotted.
    o.head("zallama_model_info", "gauge",
           "1 per registered model: file, declared mem_gb, modality, backend, eviction group, pinned.")
    o.head("zallama_model_param", "gauge",
           "1 per effective launch parameter (registry params over default_params), as param/value labels.")
    o.head("zallama_model_param_value", "gauge",
           "Numeric effective launch parameters (ctx_size, n_gpu_layers, spec_draft_n_max, ...) as values.")
    o.head("zallama_model_file_bytes", "gauge", "Size of the main model file on disk.")
    for e in entries:
        name = e.get("name")
        if not name:
            continue
        group = pm.evict_group_of(e)
        o.sample("zallama_model_info", 1, model=name,
                 modality=e.get("modality", "text"), backend=e.get("backend", "llama-server"),
                 group=group, pinned=_fmt(bool(e.get("pinned"))),
                 mem_gb=e.get("mem_gb"), file=e.get("file"),
                 aliases=",".join(e.get("aliases") or []) or None)
        try:
            o.sample("zallama_model_file_bytes", os.stat(e["file"]).st_size, model=name)
        except (KeyError, OSError, TypeError):
            pass
        # default_params are llama.cpp flags; only the llama-server family
        # inherits them. Other backends (parakeet, kokoro, sd) read just a
        # key or two from the merge, so for them the registry's own params
        # are the honest list.
        try:
            llama_family = isinstance(get_backend(ModelRegistry.backend_of(e)), LlamaServerBackend)
        except Exception:
            llama_family = False
        params = pm.merged_params(e) if llama_family else (e.get("params") or {})
        for param, value in sorted(params.items()):
            if value is None or value == "":
                continue
            o.sample("zallama_model_param", 1, model=name, param=param,
                     value=_fmt(value) if isinstance(value, bool) else value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                o.sample("zallama_model_param_value", value, model=name, param=param)
        for kind, path in (e.get("artifacts") or {}).items():
            o.sample("zallama_model_param", 1, model=name, param=f"artifact_{kind}", value=path)

    # -- per loaded model ------------------------------------------------
    by_name = {e.get("name"): e for e in entries}
    o.head("zallama_model_loaded", "gauge", "1 for each running backend, with its identity as labels.")
    o.head("zallama_model_mem_declared_bytes", "gauge", "Declared mem_gb the daemon budgets for this instance.")
    o.head("zallama_model_vram_bytes", "gauge", "Measured VRAM of this instance's process (absent for CPU backends).")
    o.head("zallama_model_uptime_seconds", "gauge", "Seconds since this instance became healthy.")
    o.head("zallama_model_idle_seconds", "gauge", "Seconds since this instance last served a request.")
    o.head("zallama_model_active_requests", "gauge", "Proxied requests currently reading from this backend.")
    for p in running:
        name = p["name"]
        entry = by_name.get(name, {})
        group = pm.evict_group_of(entry) if entry else None
        o.sample("zallama_model_loaded", 1, model=name, modality=p["modality"],
                 backend=p["backend"], group=group, port=p["port"])
        o.sample("zallama_model_mem_declared_bytes", int(p["mem_gb"] * GIB), model=name)
        if p.get("vram_gb") is not None:
            o.sample("zallama_model_vram_bytes", int(p["vram_gb"] * GIB), model=name)
        o.sample("zallama_model_uptime_seconds", now - p["started_at"], model=name)
        o.sample("zallama_model_idle_seconds", now - p["last_used"], model=name)
        o.sample("zallama_model_active_requests", p["active"], model=name)

    # -- lifecycle counters ----------------------------------------------
    events = {
        "start": "Successful backend starts (became healthy).",
        "start_failure": "Backend died or timed out before becoming healthy (the 503 'died during startup').",
        "crash": "Backend found dead on a later request (restarted on the spot).",
        "evict_capacity": "LRU evictions to make room for another model.",
        "evict_idle": "Evictions by the idle sweep.",
        "unload": "Explicit unloads via the API/CLI.",
    }
    for ev, help_ in events.items():
        metric = f"zallama_model_{ev}_total"
        o.head(metric, "counter", help_)
        for model, n in sorted(pm.lifecycle.get(ev, {}).items()):
            o.sample(metric, n, model=model)

    # -- request aggregates ----------------------------------------------
    o.head("zallama_requests_total", "counter", "Proxied /v1 requests, by model, endpoint and outcome.")
    o.head("zallama_requests_active", "gauge", "Proxied /v1 requests in flight.")
    o.head("zallama_prompt_tokens_total", "counter", "Prompt tokens processed (usage.prompt_tokens / timings.prompt_n).")
    o.head("zallama_completion_tokens_total", "counter", "Tokens generated.")
    o.head("zallama_cached_tokens_total", "counter", "Prompt tokens served from the KV cache instead of prefilled.")
    o.head("zallama_prefill_seconds_total", "counter",
           "Backend prefill wall time; rate(prompt_tokens)/rate(this) = prefill tok/s over a window.")
    o.head("zallama_decode_seconds_total", "counter",
           "Backend decode wall time; rate(completion_tokens)/rate(this) = decode tok/s over a window.")
    o.head("zallama_draft_tokens_total", "counter", "Speculative tokens drafted (draft-mtp/dflash/simple).")
    o.head("zallama_draft_accepted_tokens_total", "counter", "Speculative tokens accepted; divide by drafted for acceptance.")
    o.head("zallama_last_prefill_tokens_per_second", "gauge", "Prefill rate of the most recent request.")
    o.head("zallama_last_decode_tokens_per_second", "gauge", "Decode rate of the most recent request.")
    o.head("zallama_last_draft_acceptance_ratio", "gauge", "Draft acceptance of the most recent request (0–1).")
    o.head("zallama_request_ttft_seconds", "histogram", "Time to first content delta, streamed requests only.")
    o.head("zallama_request_duration_seconds", "histogram", "End-to-end request time as seen by the proxy.")

    active = rl.active_by_model()
    for model, agg in sorted(rl.aggregates.items()):
        for (endpoint, status), n in sorted(agg.requests.items()):
            o.sample("zallama_requests_total", n, model=model, endpoint=endpoint, status=status)
        o.sample("zallama_requests_active", active.pop(model, 0), model=model)
        o.sample("zallama_prompt_tokens_total", agg.prompt_tokens, model=model)
        o.sample("zallama_completion_tokens_total", agg.completion_tokens, model=model)
        o.sample("zallama_cached_tokens_total", agg.cached_tokens, model=model)
        o.sample("zallama_prefill_seconds_total", agg.prefill_seconds, model=model)
        o.sample("zallama_decode_seconds_total", agg.decode_seconds, model=model)
        o.sample("zallama_draft_tokens_total", agg.draft_generated, model=model)
        o.sample("zallama_draft_accepted_tokens_total", agg.draft_accepted, model=model)
        if agg.last_prefill_tps is not None:
            o.sample("zallama_last_prefill_tokens_per_second", agg.last_prefill_tps, model=model)
        if agg.last_decode_tps is not None:
            o.sample("zallama_last_decode_tokens_per_second", agg.last_decode_tps, model=model)
        if agg.last_draft_accept is not None:
            o.sample("zallama_last_draft_acceptance_ratio", agg.last_draft_accept, model=model)
        if agg.ttft.count:
            o.histogram("zallama_request_ttft_seconds", agg.ttft, model=model)
        for endpoint, hist in sorted(agg.duration.items()):
            o.histogram("zallama_request_duration_seconds", hist, model=model, endpoint=endpoint)
    # Models with an in-flight first request and no finished ones yet.
    for model, n in sorted(active.items()):
        o.sample("zallama_requests_active", n, model=model)

    # -- whole GPU -------------------------------------------------------
    gpus = gpu_totals()
    if gpus:
        o.head("zallama_gpu_memory_used_bytes", "gauge", "VRAM in use on the card, all processes (nvidia-smi).")
        o.head("zallama_gpu_memory_total_bytes", "gauge", "VRAM on the card.")
        o.head("zallama_gpu_utilization_ratio", "gauge", "GPU compute utilization (0–1).")
        o.head("zallama_gpu_temperature_celsius", "gauge", "GPU temperature.")
        o.head("zallama_gpu_power_watts", "gauge", "GPU power draw.")
        for g in gpus:
            lab = dict(gpu=g["index"], name=g["name"])
            o.sample("zallama_gpu_memory_used_bytes", g["used_bytes"], **lab)
            o.sample("zallama_gpu_memory_total_bytes", g["total_bytes"], **lab)
            if g["util"] is not None:
                o.sample("zallama_gpu_utilization_ratio", g["util"] / 100, **lab)
            if g["temp_c"] is not None:
                o.sample("zallama_gpu_temperature_celsius", g["temp_c"], **lab)
            if g["power_w"] is not None:
                o.sample("zallama_gpu_power_watts", g["power_w"], **lab)

    return o.text()
