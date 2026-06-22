<div align="center">

# 🦙 Zallama

**A simple, memory-aware, multimodal-ready local LLM server powered by `llama.cpp`.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![llama.cpp](https://img.shields.io/badge/powered%20by-llama.cpp-black.svg)](https://github.com/ggml-org/llama.cpp)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20compatible-412991.svg?logo=openai&logoColor=white)](#-openai-api-integration)
[![Changelog](https://img.shields.io/badge/changelog-keep%20a%20changelog-orange.svg)](CHANGELOG.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

</div>

> A self-contained local LLM ecosystem powered by `llama-server` (llama.cpp).

## 📚 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Reference](#-cli-reference)
- [Configuration Files](#-configuration-files)
- [Memory-Aware Eviction](#-memory-aware-eviction)
- [Vision (Multimodal) Models](#-vision-multimodal-models)
- [Backends & Modalities (Architecture)](#-backends--modalities-architecture)
- [OpenAI API Integration](#-openai-api-integration)
- [Deployment (systemd)](#-deployment-systemd)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

Zallama acts as a dynamic router and process manager for your local GGUF models. It exposes a single, unified endpoint that is fully OpenAI-compatible. When you request a model, Zallama starts the underlying backend (e.g. `llama-server`) in the background, routes your request, and automatically unloads the model after a period of inactivity to free up RAM/VRAM.

Zallama is built around a **pluggable backend abstraction**: each model declares a `modality` (`text`, and — by design — `asr`, `tts`, `image`) and a `backend`. Text/chat/embeddings and **vision** (via an `mmproj` projector) ship today on `llama-server`; the architecture is structured so additional modalities are added as new backends rather than as cross-cutting changes.

---

## ✨ Features

- **🚀 Simple CLI:** Run commands like `zallama serve`, `zallama run <model>`, `zallama add`, `zallama set`, and `zallama ps`.
- **⚡ High-Performance Downloads:** Accelerated download engine utilizing `aria2c` (with 8 concurrent connections), falling back to Python-based parallel HTTP range requests and standard stream decoders.
- **🧠 Reasoning Model Support:** Full support for thinking models (e.g., DeepSeek-R1, Qwen 3.5), rendering thinking/reasoning blocks in real-time with dim/gray coloring inside the interactive chat.
- **🔌 Full OpenAI /v1 API:** Full drop-in replacement for OpenAI endpoints (Chat, Completions, and Embeddings) with streaming supported via SSE.
- **👁️ Vision (Multimodal):** Run vision models by attaching an `mmproj` projector artifact — image input flows through `/v1/chat/completions`.
- **🧩 Pluggable Backends & Modalities:** Each model declares a `modality` and `backend`. A `Backend` abstraction isolates engine-specific logic, so new modalities (TTS, ASR, image generation) slot in as new backends. A modality guard returns a clear error if a model is used on an incompatible endpoint.
- **🌐 Sleek Embedded Web UI:** Access model management, registration, loading/unloading, and streaming chat at `http://localhost:11435`.
- **⚙️ Config-Driven Architecture:** Define global defaults and customize per-model parameters (context size, GPU layers offload, batching options) in simple YAML configurations.
- **🔄 Dynamic Process Management:** Per-model startup locking, OS-checked port assignment, server health checking, an optional concurrency cap (`max_loaded_models`), and automatic LRU model eviction/unloading when idle.
- **🧠 Memory-Aware Eviction:** Set a `mem_budget_gb` and Zallama evicts least-recently-used models to keep total declared/estimated memory within budget.
- **🔒 Production-Ready Defaults:** Binds to `127.0.0.1` by default, optional Bearer-token API key, and configurable request timeouts.

---

## 🛠️ Installation

**Requirements:** Python 3.10+, and a `llama-server` binary (built from [llama.cpp](https://github.com/ggml-org/llama.cpp) or placed in `./bin/llama-server`). `aria2c` is optional but recommended for fast downloads.

Run the one-shot installer:

```bash
chmod +x install.sh
./install.sh
```

The installer:
1. Verifies Python 3 and installs `requirements.txt` into a project-local **`.venv`** (required on modern Debian/Ubuntu, which block system-wide `pip` under [PEP 668](https://peps.python.org/pep-0668/)).
2. Checks for a `llama-server` binary (`./bin`, or on `PATH`).
3. Makes the `zallama` CLI executable and creates `~/.zallama/{models,logs,bin}`.
4. Symlinks `zallama` into `/usr/local/bin` (or `~/.local/bin`) when possible.
5. **Installs a systemd service — only when run as root** (`sudo ./install.sh`). See [Deployment](#-deployment-systemd).

> **No activation needed:** the `zallama` launcher automatically re-execs into `.venv`, so `zallama serve` and every other command just work. (Set `ZALLAMA_NO_VENV=1` to bypass and use the current interpreter.)

To run `zallama` from anywhere without the symlink, add the repo to your `PATH`:
```bash
export PATH="$PWD:$PATH"   # from the zallama checkout
```

---

## 🚀 Quick Start

### 1. Start the Daemon
```bash
zallama serve
```
*Starts the FastAPI controller and Web UI on `http://localhost:11435`.*

### 2. Pull a Model from HuggingFace
Accelerated high-speed model acquisition via `aria2c`:
```bash
# Pull using a simple shorthand (from Unsloth's repo)
zallama pull llama3.2:3b

# Or pull any GGUF file directly from HuggingFace
zallama pull unsloth/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
```

### 3. Configure Model Parameters
Dynamically set context size, GPU offloading, or turn reasoning (thinking blocks) on or off:
```bash
# Set context size to 8192 and disable thinking blocks
zallama set qwen3.5-4b-q4_k_m reasoning=false ctx_size=8192

# Enable full GPU layers offload
zallama set llama3.2:1b n_gpu_layers=99
```

### 4. Run Interactive Chat
```bash
zallama run llama3.2:3b
```

---

## 🖥️ CLI Reference

```
serve                  Start the Zallama daemon
list                   List registered models (alias: ls)
add <name> <file>      Register a local .gguf model
set <name> <k>=<v>...  Configure parameters for a registered model
pull <name>            Pull model from HF / Unsloth presets (uses aria2c)
remove <name>          Remove a model from registry (alias: rm)
run <name>             Interactive chat with a model (streams reasoning)
ps                     Show running model processes
load <name>            Pre-load a model (start llama-server)
unload <name>          Stop a running model (alias: stop)
reload <name>          Restart a running model to apply registry param changes
logs <name>            Tail logs for a model
health                 Show daemon health status
version                Show version info
```

---

## ⚙️ Configuration Files

### Global Settings (`config/config.yaml`)
```yaml
zallama:
  host: "127.0.0.1"      # localhost only; set "0.0.0.0" to expose on the network
  port: 11435
  models_dir: "~/.zallama/models"
  logs_dir: "~/.zallama/logs"
  api_key: ""            # if set, required as a Bearer token on /v1 and /api
  request_timeout: 600   # seconds for non-streaming upstream proxy calls

llama_server:
  binary: ""             # Auto-detects in ./bin/llama-server or PATH
  port_start: 8100       # Backends spawn on ports 8100, 8101, etc.
  startup_timeout: 60    # Seconds to wait for a backend's /health
  idle_timeout: 300      # Auto-unload model after 300s of inactivity (0 to disable)
  max_loaded_models: 0   # Max concurrently loaded models; evicts LRU (0 = unlimited)
  mem_budget_gb: 0       # Memory budget (GB); evicts LRU to fit a model's mem_gb (0 = unlimited)
  mem_init_gb: 2         # Fallback per-model cost when mem_gb is undeclared & file size unknown
  default_params:
    ctx_size: 8192
    n_gpu_layers: 99     # Attempt full GPU offload by default
    threads: 8
    flash_attn: true
    parallel: 1          # Single-session slot allocation (avoids context limits)
```

> **Security note:** Zallama binds to `127.0.0.1` by default. If you set `host: "0.0.0.0"` to expose it on your network, also set an `api_key` — the daemon has no auth otherwise.

### Model Registry (`models/registry.yaml`)
```yaml
models:
  # Standard text/chat model
  - name: "qwen3.5-4b-q4_k_m"
    file: "/home/cook/.zallama/models/Qwen3.5-4B-Q4_K_M.gguf"
    description: "Downloaded from unsloth/Qwen3.5-4B-GGUF"
    params:
      ctx_size: 8192
      n_gpu_layers: 99
      reasoning: false   # Bypasses thinking blocks for immediate responses

  # Vision (multimodal) model — attach an mmproj projector via `artifacts`
  - name: "qwen2-vl-7b"
    file: "Qwen2-VL-7B-Instruct-Q4_K_M.gguf"   # relative paths resolve under models_dir
    modality: "text"     # vision is served on /v1/chat/completions (default modality)
    backend: "llama-server"
    artifacts:
      mmproj: "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
    params:
      ctx_size: 8192
      n_gpu_layers: 99
```

> **Applying changes:** The registry reloads from disk automatically, so adding, editing, or removing an entry takes effect on the next request — no daemon restart. The one exception is a model that's **already running**: its `llama-server` keeps the params it launched with, so run `zallama reload <name>` to restart it with the new params. (Changes to `config.yaml` are read only at startup and do require `systemctl restart zallama`.)

Each entry may declare:
- **`modality`** — `text` (default), or the planned `asr` / `tts` / `image`. Determines which endpoints the model may serve. Requests to a mismatched endpoint return a clear `400`.
- **`backend`** — which engine runs the model (default `llama-server`). New backends resolve their own binary from `./bin/<name>`, `~/.zallama/bin/<name>`, or `PATH`.
- **`artifacts`** — extra files beyond the primary GGUF (e.g. `mmproj` for vision, and — for future backends — vocoders, etc.). Paths are absolute or relative to `models_dir`.
- **`mem_gb`** — declared memory footprint, used by memory-aware eviction (see below). If omitted, it's estimated from the GGUF file size.

---

## 🧠 Memory-Aware Eviction

Beyond the time-based idle sweep and the `max_loaded_models` count cap, Zallama can keep total loaded-model memory within a budget. Set `mem_budget_gb` and, before starting a model, Zallama evicts least-recently-used instances until the incoming model fits.

Each model's cost is taken from its declared `mem_gb`; if undeclared, it's estimated from the GGUF file size (≈ size × 1.2 for KV-cache overhead), falling back to `mem_init_gb` when the size is unknown. Declaring `mem_gb` is recommended for accuracy:

```bash
zallama set qwen3.5-4b-q4_k_m mem_gb=4
```

Inspect current usage and headroom any time with `zallama ps` (or `GET /api/ps`), which reports per-model memory and the budget:

```
NAME                      PORT     MEM      UPTIME       LAST USED
─────────────────────────────────────────────────────────────────────────
qwen3.5-4b-q4_k_m         8100     4.0GB    3m12s        8s ago

Memory: 4.0GB / 12.0GB used  •  8.0GB free  •  1 loaded
```

> **Note:** `mem_gb` is a declared/estimated budget for scheduling, not a hard GPU/VRAM measurement. For subprocess backends like `llama-server`, the OS — not Zallama — owns actual memory; the budget governs *how many* models Zallama keeps resident.

---

## 👁️ Vision (Multimodal) Models

Vision models run on `llama-server` with a multimodal projector (`mmproj`). Register the base GGUF and point an `mmproj` artifact at the projector file:

```bash
zallama add qwen2-vl-7b ~/.zallama/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf "Qwen2-VL 7B vision"
# then edit models/registry.yaml to add the `artifacts: { mmproj: ... }` block shown above
```

Zallama passes `--mmproj <file>` to the backend automatically, and image input flows through the standard OpenAI `/v1/chat/completions` endpoint (image_url message content).

---

## 🧩 Backends & Modalities (Architecture)

Zallama separates the **generic process lifecycle** (spawn, health-check, port assignment, LRU eviction, kill) from **engine-specific logic** (which binary to run, how to build its arguments, which health path to poll). The latter lives behind a `Backend` abstraction in [`server/backends.py`](server/backends.py).

This is the seam for new modalities. Today `LlamaServerBackend` covers text, chat, embeddings, and vision. Adding TTS, ASR, or image generation means adding a new `Backend` subclass and the matching endpoint proxy — no changes to the process manager or registry schema. The OpenAI endpoints for these modalities (`/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/images/generations`) are already mapped in the modality guard, awaiting their backends.

---

## 🌐 OpenAI API Integration

Zallama acts as a standard OpenAI-compatible API gateway. Specify the model you want to target in the request body, and Zallama will handle model loading and routing automatically:

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b-q4_k_m",
    "messages": [
      {"role": "user", "content": "Tell me a joke."}
    ],
    "stream": true
  }'
```

---

## 🚀 Deployment (systemd)

For an always-on daemon, install Zallama as a systemd service. Running the installer **as root** writes the unit automatically; otherwise install it manually:

```bash
sudo tee /etc/systemd/system/zallama.service >/dev/null <<EOF
[Unit]
Description=Zallama — Local LLM Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/zallama serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now zallama
systemctl status zallama
```

> **Note:** `./install.sh` only installs the unit when run as root (`sudo ./install.sh`), and it does **not** enable/start it for you — run `sudo systemctl enable --now zallama` afterwards. Tail logs with `journalctl -u zallama -f`.

---

## 🔐 Security

Zallama defaults to **localhost-only** (`host: 127.0.0.1`) with no authentication, which is safe for single-user local use. If you expose it:

- Set `host: "0.0.0.0"` **only together with** an `api_key`. With a key set, all `/v1` and `/api` calls require an `Authorization: Bearer <key>` header (the Web UI and health checks stay public).
  ```bash
  curl http://localhost:11435/v1/models -H "Authorization: Bearer $YOUR_KEY"
  ```
- Prefer running behind a reverse proxy (TLS termination, rate limiting) for any network-facing deployment.
- CORS allows all origins for the bundled Web UI but does **not** send credentials.

---

## 🔧 Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Cannot connect to Zallama at ...` | Daemon isn't running — start it with `zallama serve` (or `systemctl start zallama`). |
| `llama-server binary not found` | Build/place it at `./bin/llama-server`, or set `llama_server.binary` in `config/config.yaml`. |
| Model fails to start / startup timeout | Check `zallama logs <model>`. Often a bad GGUF path, too-high `n_gpu_layers`, or `ctx_size` exceeding VRAM. |
| Models keep getting unloaded | Increase `idle_timeout`, `max_loaded_models`, or `mem_budget_gb`. |
| `400` "modality ... cannot serve" | You called an endpoint the model's `modality` doesn't support (e.g. a vision-only flow on the wrong route). |
| `401 Invalid or missing API key` | `api_key` is set — pass `Authorization: Bearer <key>`. |
| systemd service missing after install | `install.sh` only installs it as root — run `sudo ./install.sh` then `sudo systemctl enable --now zallama`. |
| `error: externally-managed-environment` from pip | PEP 668 — don't `pip install` system-wide. Re-run `./install.sh` (it uses `.venv`). If venv creation fails: `sudo apt install python3-venv python3-full`. |
| `Import error: No module named 'fastapi'` on `serve` | The `.venv` is missing or incomplete — re-run `./install.sh`. |

---

## 🤝 Contributing

Contributions are welcome! A good shape for a PR:

1. Fork and branch from `main`.
2. Keep changes focused; match the surrounding code style.
3. For a new **backend/modality**, add a `Backend` subclass in [`server/backends.py`](server/backends.py) and the matching endpoint proxy — the process manager and registry schema should not need changes.
4. Update [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]`.
5. Open a PR describing the change and how you tested it.

---

## ⚖️ License

Released under the [MIT License](LICENSE). © 2026 Rija Z.
