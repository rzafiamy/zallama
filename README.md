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
- [Speech-to-Text (ASR)](#-speech-to-text-asr)
- [Backends & Modalities (Architecture)](#-backends--modalities-architecture)
- [RAG: Reranking & the zvec Vector Store](#-rag-reranking--the-zvec-vector-store)
- [OpenAI API Integration](#-openai-api-integration)
- [Deployment (systemd)](#-deployment-systemd)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

Zallama acts as a dynamic router and process manager for your local GGUF models. It exposes a single, unified endpoint that is fully OpenAI-compatible. When you request a model, Zallama starts the underlying backend (e.g. `llama-server`) in the background, routes your request, and automatically unloads the model after a period of inactivity to free up RAM/VRAM.

Zallama is built around a **pluggable backend abstraction**: each model declares a `modality` (`text`, `embedding`, `rerank`, `asr`, `tts`, and — by design — `image`) and a `backend`. Text/chat and **vision** (via an `mmproj` projector) ship on `llama-server`; **embeddings** and **reranking** run `llama-server` in `--embedding` / `--reranking` mode; **speech-to-text (ASR)** ships on `parakeet-server` ([parakeet.cpp](https://github.com/mudler/parakeet.cpp)) and **text-to-speech (TTS)** on `kokoro-server`. The architecture is structured so additional modalities are added as new backends rather than as cross-cutting changes.

---

## ✨ Features

- **🚀 Simple CLI:** Run commands like `zallama serve`, `zallama run <model>`, `zallama add`, `zallama set`, and `zallama ps`.
- **⚡ High-Performance Downloads:** Accelerated download engine utilizing `aria2c` (with 8 concurrent connections), falling back to Python-based parallel HTTP range requests and standard stream decoders.
- **🧠 Reasoning Model Support:** Full support for thinking models (e.g., DeepSeek-R1, Qwen 3.5), rendering thinking/reasoning blocks in real-time with dim/gray coloring inside the interactive chat.
- **🔌 Full OpenAI /v1 API:** Full drop-in replacement for OpenAI endpoints (Chat, Completions, and Embeddings) with streaming supported via SSE.
- **👁️ Vision (Multimodal):** Run vision models by attaching an `mmproj` projector artifact — image input flows through `/v1/chat/completions`.
- **🎙️ Speech-to-Text (ASR):** Transcribe audio via `/v1/audio/transcriptions` (OpenAI-compatible) on the `parakeet-server` backend. Any input format (mp3/m4a/webm/flac/…) is auto-transcoded to WAV with `ffmpeg`. Multilingual models (e.g. Parakeet TDT v3, 25 European languages incl. French) supported.
- **🔎 Built-in RAG:** Cross-encoder reranking at `/v1/rerank` plus **zvec**, an in-process HNSW vector store (the [`zvec`](https://zvec.org) library — no external DB server) with `/v1/zvec/*` ingest & semantic-search endpoints and a `zallama zvec` CLI.
- **🧩 Pluggable Backends & Modalities:** Each model declares a `modality` and `backend`. A `Backend` abstraction isolates engine-specific logic, so new modalities (TTS, image generation) slot in as new backends. A modality guard returns a clear error if a model is used on an incompatible endpoint.
- **🌐 Sleek Embedded Web UI:** Access model management, registration, loading/unloading, and streaming chat at `http://localhost:11435`.
- **⚙️ Config-Driven Architecture:** Define global defaults and customize per-model parameters (context size, GPU layers offload, batching options) in simple YAML configurations.
- **🔄 Dynamic Process Management:** Per-model startup locking, OS-checked port assignment, server health checking, an optional concurrency cap (`max_loaded_models`), and automatic LRU model eviction/unloading when idle.
- **🧠 Memory-Aware Eviction:** Set a `mem_budget_gb` and Zallama evicts least-recently-used models to keep total declared/estimated memory within budget.
- **🔒 Production-Ready Defaults:** Binds to `127.0.0.1` by default, optional Bearer-token API key, and configurable request timeouts.

---

## 🛠️ Installation

**Requirements:** Python 3.10+, and a `llama-server` binary (built from [llama.cpp](https://github.com/ggml-org/llama.cpp) or placed in `./bin/llama-server`). `aria2c` is optional but recommended for fast downloads. For **ASR** (speech-to-text) and **TTS** (speech synthesis), build their respective engines and have `ffmpeg` installed.

### 1. Clone the repository

```bash
git clone https://github.com/rzafiamy/zallama.git
cd zallama
```

### 2. Building the inference engines

Helper scripts build each engine and install the binaries into `./bin/` (the clone and build happen in a temporary directory, keeping your repository tree clean):

```bash
# llama.cpp (text / chat / embeddings / vision) — requires a release tag/branch name
./build-ggml-llama.cpp.sh b4600

# parakeet.cpp (ASR / speech-to-text) — requires a release tag/branch name
./build-ggml-parakeet.cpp.sh master

# kokoro.cpp (TTS / voice synthesis) — requires a release tag/branch name
./build-ggml-kokoro.cpp.sh v0.1.0
```

> All scripts default to a **CUDA** build. The parakeet script also copies the shared `libggml*.so` next to the binaries and sets their `RPATH` to `$ORIGIN` (via `patchelf`) so they resolve at runtime.

### 3. Run the installer

Install the global CLI launcher and register local configuration defaults:

```bash
sudo bash install.sh
```

The installer:
1. Verifies Python 3 and installs `requirements.txt` into a project-local **`.venv`** (required on modern Debian/Ubuntu under [PEP 668](https://peps.python.org/pep-0668/)).
2. Checks for a `llama-server` binary (`./bin`, or on `PATH`).
3. Makes the `zallama` CLI executable and creates `~/.zallama/{models,logs,bin}`.
4. Symlinks `zallama` into `/usr/local/bin` (or `~/.local/bin`) when possible.
5. **Installs a systemd service — only when run as root** (`sudo bash install.sh`). See [Deployment](#-deployment-systemd).

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

# Pull an ASR (speech-to-text) model — auto-registered as modality=asr
zallama pull parakeet:0.6b
# Multilingual (25 European languages incl. French):
zallama pull mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf
```

> Parakeet GGUF repos are detected automatically and registered with the ASR backend. For a raw HF path whose repo name gives no hint, force it with `zallama pull <repo>/<file>.gguf --type asr`.

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
pull <name> [--type T] Pull model from HF / Unsloth presets (uses aria2c).
                       --type sets modality (text|embedding|rerank|asr|tts) for raw HF paths.
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

### Tab Completion

`./install.sh` installs **bash** and **zsh** completion automatically when it can
write to the system completion directories. Once enabled, `<TAB>` completes both
subcommands and **registered model names**:

```bash
zallama r<TAB>          # → reload  remove  rm  run
zallama run <TAB>       # → completes from your registered models
```

Completion reads the model list straight from `registry.yaml`, so it works even
when the daemon isn't running. To install manually:

```bash
# bash
sudo cp completions/zallama.bash /usr/share/bash-completion/completions/zallama
# zsh — copy into a dir on your $fpath, then run compinit
cp completions/_zallama ~/.zsh/completions/_zallama
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

  # ASR (speech-to-text) model — runs on parakeet-server
  - name: "tdt-0.6b-v3-q8_0"
    file: "tdt-0.6b-v3-q8_0.gguf"
    modality: "asr"
    backend: "parakeet-server"
    description: "Parakeet TDT 0.6B v3 (multilingual ASR)"
    params:
      threads: 4
```

> **Applying changes:** The registry reloads from disk automatically, so adding, editing, or removing an entry takes effect on the next request — no daemon restart. The one exception is a model that's **already running**: its `llama-server` keeps the params it launched with, so run `zallama reload <name>` to restart it with the new params. (Changes to `config.yaml` are read only at startup and do require `systemctl restart zallama`.)

Each entry may declare:
- **`modality`** — `text` (default), `embedding`, `rerank`, `asr`, `tts`, or the planned `image`. Determines which endpoints the model may serve. Requests to a mismatched endpoint return a clear `400`. (Legacy embedding models registered as `text` with `params: embedding: true` are still treated as `embedding` at runtime.)
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

## 🎙️ Speech-to-Text (ASR)

Audio transcription runs on the **`parakeet-server`** backend ([parakeet.cpp](https://github.com/mudler/parakeet.cpp)) and is exposed at the OpenAI-compatible `POST /v1/audio/transcriptions` endpoint.

**1. Build the binary** (installs `parakeet-server` into `./bin/`):
```bash
./build-ggml-parakeet.cpp.sh
```

**2. Pull a model** (auto-registered as `modality: asr`, `backend: parakeet-server`):
```bash
zallama pull parakeet:0.6b                                   # English (TDT 0.6B v2)
zallama pull mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf  # Multilingual v3 (incl. French)
```

**3. Transcribe** — upload any audio format; Zallama transcodes it to WAV via `ffmpeg` before forwarding:
```bash
curl http://localhost:11435/v1/audio/transcriptions \
  -F model=tdt-0.6b-v3-q8_0 \
  -F file=@speech.mp3 \
  -F response_format=text
```

`response_format` accepts `text`, `json`, or `verbose_json` (with `-F 'timestamp_granularities[]=word'` for per-word timing).

> **Language support is a property of the model, not Zallama.** `ctc-0.6b` / `tdt-0.6b-v2` are **English-only**; for French and other languages use the multilingual **[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** (25 European languages, automatic language detection).
>
> **`ffmpeg`** is only needed for non-WAV uploads. Without it, WAV uploads still work and other formats return a clear `415`.

---

## 🧩 Backends & Modalities (Architecture)

Zallama separates the **generic process lifecycle** (spawn, health-check, port assignment, LRU eviction, kill) from **engine-specific logic** (which binary to run, how to build its arguments, which health path to poll). The latter lives behind a `Backend` abstraction in [`server/backends.py`](server/backends.py).

This is the seam for new modalities. `LlamaServerBackend` covers text, chat, and vision; `EmbeddingServerBackend` runs `llama-server --embedding` for `/v1/embeddings`; `RerankServerBackend` runs `llama-server --reranking` for `/v1/rerank`; `ParakeetServerBackend` covers ASR (`/v1/audio/transcriptions`); `KokoroServerBackend` covers TTS (`/v1/audio/speech`). Adding image generation means adding a new `Backend` subclass and the matching endpoint proxy — no changes to the process manager or registry schema. The `/v1/images/generations` endpoint is already mapped in the modality guard, awaiting its backend.

---

## 🔎 RAG: Reranking & the zvec Vector Store

Zallama ships everything needed for retrieval-augmented generation locally: an embedding model (already supported via `/v1/embeddings`), a **reranker**, and **zvec** — an in-process vector store. No external vector database server to run.

> **zvec** is backed by the [`zvec`](https://zvec.org) library (Alibaba, Apache-2.0), an embedded HNSW-indexed vector database. It runs inside the Zallama daemon — each collection is a directory under `rag.zvec_dir` (default `~/.zallama/zvec`), tracked by a small `collections.json` manifest. Install it with the rest of the deps (`pip install -r requirements.txt`).

### One-command setup

`zallama pull` has shorthands for an embedding model and a reranker that write the correct registry entries (the `embedding-server` and `rerank-server` backends) for you:

```bash
zallama pull nomic-embed:v1.5      # embedding model → /v1/embeddings
zallama pull bge-reranker:v2-m3    # reranker        → /v1/rerank
```

Then point the `rag` config block at them (or use the matching env vars):

```yaml
rag:
  embedding_model: "nomic-embed:v1.5"
  rerank_model: "bge-reranker:v2-m3"
```

### Reranking — `POST /v1/rerank`

Reranking scores how relevant each document is to a query using a cross-encoder model (e.g. `bge-reranker-v2-m3`). It runs on `llama-server` in `--reranking` mode via the `rerank-server` backend. Register a reranker with `modality: rerank, backend: rerank-server` in `models/registry.yaml`, then:

```bash
curl http://localhost:11435/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "How do I unload a model?",
    "documents": ["zallama unload <name> stops a model", "zallama pull fetches a model"],
    "top_n": 2,
    "return_documents": true
  }'
```

Returns a Cohere/Jina-style `{ "results": [{ "index", "relevance_score", "document"? }] }`, sorted by score.

### zvec vector store

zvec stores documents and their embeddings (HNSW index, persisted under `rag.zvec_dir`). It embeds and searches by calling Zallama's own `/v1/embeddings`, so it just needs a default embedding model — set `rag.embedding_model` (or `ZALLAMA_EMBEDDING_MODEL`). A query can optionally rerank its candidates.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/zvec/collections` | Create a collection (`name`, optional `embedding_model`, `dim`) |
| `GET /v1/zvec/collections` | List collections |
| `DELETE /v1/zvec/collections/{name}` | Delete a collection |
| `POST /v1/zvec/{name}/upsert` | Add/replace documents (auto-embedded) |
| `POST /v1/zvec/{name}/query` | Semantic search (`query`, `top_k`, `filter`, optional `rerank_model`) |
| `POST /v1/zvec/{name}/delete` | Delete documents by `ids` |

From the CLI:

```bash
zallama zvec create notes                       # uses rag.embedding_model
zallama zvec upsert notes ./docs.txt            # one document per line (or a JSON array)
zallama zvec query notes "how to unload a model" --top-k 3 --rerank bge-reranker-v2-m3
zallama zvec collections
```

Or over HTTP:

```bash
curl http://localhost:11435/v1/zvec/notes/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how to unload a model", "top_k": 3, "rerank_model": "bge-reranker-v2-m3"}'
```

Configure defaults in the `rag` block of `config/config.yaml` (`embedding_model`, `rerank_model`, `zvec_dir`, `default_top_k`).

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

> **Note:** `install.sh` only installs the unit when run as root (`sudo bash install.sh`), and it does **not** enable/start it for you — run `sudo systemctl enable --now zallama` afterwards. Tail logs with `journalctl -u zallama -f`.

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
| `parakeet-server binary not found` | Build it: `./build-ggml-parakeet.cpp.sh master` (installs into `./bin/`). |
| `libggml*.so: cannot open shared object file` | The parakeet binary can't find its shared libs. Re-run `./build-ggml-parakeet.cpp.sh master` (it copies the `.so` files and sets `RPATH=$ORIGIN`), or `apt install patchelf` and `patchelf --set-rpath '$ORIGIN' bin/parakeet-server bin/libggml*.so.*.*`. |
| ASR returns "accepts WAV uploads only" | Non-WAV upload and `ffmpeg` is missing — `apt install ffmpeg` (Zallama auto-transcodes once present). |
| ASR transcribes gibberish for non-English | The model is English-only (`ctc-0.6b` / `tdt-0.6b-v2`). Use multilingual `tdt-0.6b-v3` instead. |
| Model fails to start / startup timeout | Check `zallama logs <model>`. Often a bad GGUF path, too-high `n_gpu_layers`, or `ctx_size` exceeding VRAM. |
| Models keep getting unloaded | Increase `idle_timeout`, `max_loaded_models`, or `mem_budget_gb`. |
| `400` "modality ... cannot serve" | You called an endpoint the model's `modality` doesn't support (e.g. a vision-only flow on the wrong route). |
| `401 Invalid or missing API key` | `api_key` is set — pass `Authorization: Bearer <key>`. |
| systemd service missing after install | `install.sh` only installs it as root — run `sudo bash install.sh` then `sudo systemctl enable --now zallama`. |
| `error: externally-managed-environment` from pip | PEP 668 — don't `pip install` system-wide. Re-run `sudo bash install.sh` (it uses `.venv`). If venv creation fails: `sudo apt install python3-venv python3-full`. |
| `Import error: No module named 'fastapi'` on `serve` | The `.venv` is missing or incomplete — re-run `sudo bash install.sh`. |

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
