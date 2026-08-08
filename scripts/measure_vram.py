#!/usr/bin/env python3
"""
measure_vram.py — Measure what each registered model actually costs in VRAM.

Zallama estimates an undeclared model's footprint as `gguf_size * 1.2`, which
ignores the KV cache, the artifacts (mmproj, t5xxl, clip, vae) and the compute
buffers. The error routinely exceeds 100% in both directions, so a memory
budget built on those estimates evicts the wrong models and still OOMs. This
script replaces the guess with a measurement.

For each model it builds the command line through Zallama's OWN
`backend.build_args()` — so what gets measured is exactly the process the daemon
would spawn — launches it outside the daemon on a free port, waits for /health,
reads the per-PID VRAM from nvidia-smi, and shuts it down.

Two backends are special-cased:
  * sd-server allocates nothing at load time (weights come up on the first
    generation), so a real 512x512 generation is run and the peak is sampled.
  * kokoro-server is ONNX on CPU and uses no VRAM; it is skipped.

Run this on an IDLE GPU. A resident model skews the reading, and large models
will fail to allocate their KV cache. Stop the daemon or unload everything
first.

Usage:
    python3 scripts/measure_vram.py                       # every model
    python3 scripts/measure_vram.py Qwen3.6-27B gemma-4   # substring match
    python3 scripts/measure_vram.py --write               # write mem_gb into registry.yaml
    python3 scripts/measure_vram.py --margin 0.3          # safety margin (default 0.15)

See docs/vram-planning.md for what to do with the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.backends import get_backend            # noqa: E402
from server.config import load_config              # noqa: E402
from server.model_registry import ModelRegistry    # noqa: E402

HEALTH_TIMEOUT = 420   # large models on a cold page cache can take minutes
SETTLE_SEC = 6         # let allocations settle before reading nvidia-smi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def free_port() -> int:
    """A port free right now. Never reuse a fixed one: a backend that is still
    shutting down keeps its socket open, and the next launch either fails to
    bind or — worse — answers the health check in its place."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def vram_of(pid: int) -> float | None:
    """VRAM held by one PID, in GiB. None when the process owns no GPU memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return round(float(parts[1]) / 1024.0, 2)
    return None


def gpu_free_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


def health_ok(port: int, path: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2):
            return True
    except Exception:
        return False


def terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=40)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def trigger_image_generation(port: int) -> None:
    """sd-server holds no VRAM until it generates; force one small image."""
    body = json.dumps({"prompt": "a lighthouse at dawn", "n": 1,
                       "size": "512x512", "response_format": "b64_json"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/generations",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=900).read()
    except Exception as e:
        print(f"    generation failed ({type(e).__name__}) — peak may be low", file=sys.stderr)


def measure(entry: dict, reg: ModelRegistry, defaults: dict, logs: Path) -> dict:
    name = entry["name"]
    backend = get_backend(ModelRegistry.backend_of(entry))
    result = {"name": name, "backend": backend.name}

    binary = str(REPO / "bin" / backend.binary_name)
    if not Path(binary).exists():
        binary = backend.binary_name  # fall back to PATH

    model_path = reg.resolve_path(entry)
    merged = dict(defaults) if backend.name in ("llama-server", "embedding-server",
                                                "rerank-server") else {}
    merged.update(entry.get("params") or {})

    # Refuse to launch something that cannot fit in what is free right now —
    # a half-allocated model reports a meaningless number.
    need = model_path.stat().st_size / 1e9 if model_path.is_file() else 2.0
    free = gpu_free_gb()
    if free < need + 0.3:
        return {**result, "status": "SKIP_VRAM",
                "note": f"needs ~{need:.1f}GB, only {free:.1f}GB free — free the GPU first"}

    port = free_port()
    args = backend.build_args(binary, port, model_path, entry, merged,
                              reg.resolve_artifacts(entry))
    log_path = logs / f"{re.sub(r'[^A-Za-z0-9._-]', '_', name)}.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)

    t0 = time.time()
    ready = False
    while time.time() - t0 < HEALTH_TIMEOUT:
        if proc.poll() is not None:
            break
        if health_ok(port, backend.health_path()):
            ready = True
            break
        time.sleep(1.0)
    result["load_s"] = round(time.time() - t0, 1)

    if not ready:
        result["status"] = "CRASH" if proc.poll() is not None else "TIMEOUT"
        result["note"] = f"see {log_path}"
        terminate(proc)
        log.close()
        return result

    if backend.name == "sd-server":
        trigger_image_generation(port)
    time.sleep(SETTLE_SEC)
    result["vram_gb"] = vram_of(proc.pid)
    result["status"] = "OK"

    terminate(proc)
    log.close()
    time.sleep(5)  # let the driver hand the memory back before the next model
    return result


# ---------------------------------------------------------------------------
# registry writing
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"^\s*(?:- )?name:\s*(.+?)\s*$")
_MEM_RE = re.compile(r"^\s+mem_gb:\s")
_ITEM_RE = re.compile(r"^\s*- ")


def write_mem_gb(registry_path: Path, values: dict[str, float]) -> int:
    """Set `mem_gb:` on the named entries, in place.

    Line-based on purpose: a YAML round-trip through safe_load/dump would strip
    every comment and reorder the whole file. We split the document into
    list-item blocks, and in each block we want to rewrite we drop any existing
    `mem_gb:` line and insert a fresh one after `name:`. Splitting into blocks
    first matters because `name:` is not necessarily an entry's first key —
    dropping the old line has to work whether it sits before or after it.
    """
    lines = registry_path.read_text().split("\n")

    # Split into [preamble, block, block, ...] on top-level "- " items.
    blocks: list[list[str]] = [[]]
    for line in lines:
        if _ITEM_RE.match(line) and not line.lstrip().startswith("#"):
            blocks.append([])
        blocks[-1].append(line)

    written = 0
    for block in blocks:
        name = None
        for line in block:
            m = _NAME_RE.match(line)
            if m and not line.lstrip().startswith("#"):
                name = m.group(1).strip().strip("'\"")
                break
        if name not in values:
            continue

        rewritten: list[str] = []
        for line in block:
            if _MEM_RE.match(line):
                continue                       # drop the stale value
            rewritten.append(line)
            if _NAME_RE.match(line) and not line.lstrip().startswith("#"):
                rewritten.append(f"  mem_gb: {values[name]}")
        block[:] = rewritten
        written += 1

    registry_path.write_text("\n".join(l for b in blocks for l in b))
    return written


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="substring filters; default = all")
    ap.add_argument("--write", action="store_true",
                    help="write the measured mem_gb into registry.yaml")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="safety margin added to each measured value (GB, default 0.15)")
    ap.add_argument("--json", metavar="PATH", help="also dump raw results as JSON")
    args = ap.parse_args()

    cfg = load_config()
    models_dir = cfg["zallama"]["models_dir"]
    registry_path = Path(models_dir) / "registry.yaml"
    reg = ModelRegistry(registry_path, models_dir)
    defaults = cfg["llama_server"].get("default_params", {})

    logs = Path(cfg["zallama"].get("logs_dir", "/tmp")) / "vram-measure"
    logs.mkdir(parents=True, exist_ok=True)

    entries = [e for e in reg.list_models()
               if not args.models or any(f.lower() in e["name"].lower() for f in args.models)]
    if not entries:
        print("No model matched.", file=sys.stderr)
        return 1

    print(f"GPU free: {gpu_free_gb():.2f} GB — measuring {len(entries)} model(s)\n")

    results, measured = [], {}
    for entry in entries:
        name = entry["name"]
        if ModelRegistry.modality_of(entry) == "tts":
            print(f"  {name:<34} skipped (CPU backend, no VRAM)")
            continue
        print(f"→ {name}", flush=True)
        r = measure(entry, reg, defaults, logs)
        results.append(r)
        if r["status"] == "OK" and r.get("vram_gb"):
            measured[name] = round(r["vram_gb"] + args.margin, 1)
            print(f"  {r['vram_gb']:.2f} GB   (load {r['load_s']}s)"
                  f"   -> mem_gb: {measured[name]}", flush=True)
        else:
            print(f"  {r['status']}: {r.get('note', '')}", flush=True)

    print("\n" + "=" * 64)
    for r in results:
        v = f"{r['vram_gb']:.2f} GB" if r.get("vram_gb") else r["status"]
        print(f"{r['name']:<34} {v:>12}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nRaw results: {args.json}")

    if args.write and measured:
        n = write_mem_gb(registry_path, measured)
        print(f"\nWrote mem_gb on {n} entr{'y' if n == 1 else 'ies'} in {registry_path}")
        print("The registry hot-reloads; running models keep their old params "
              "until `zallama reload <name>`.")
    elif measured:
        print("\nAdd these to registry.yaml (or re-run with --write):")
        for k, v in measured.items():
            print(f"  {k}: mem_gb: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
