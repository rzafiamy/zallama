<div align="center">

# 🦙 Zallama

### Own your AI.

**A local, multimodal, memory-aware LLM server — your models, your hardware, your rules.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![llama.cpp](https://img.shields.io/badge/powered%20by-llama.cpp-black.svg)](https://github.com/ggml-org/llama.cpp)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20compatible-412991.svg?logo=openai&logoColor=white)](#-openai-api-integration)
[![Local First](https://img.shields.io/badge/data-never%20leaves%20your%20machine-2ea44f.svg)](#-security)
[![Changelog](https://img.shields.io/badge/changelog-keep%20a%20changelog-orange.svg)](CHANGELOG.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

</div>

---

## 🌱 Why Zallama exists

Every time you send a prompt to a cloud API, you're renting intelligence — and handing over your data, your context, and your habits along with it. Your hardware is idle. Your bill keeps growing. And the model behind the curtain can change, get deprecated, or get more expensive overnight.

Zallama flips that. It's a small daemon that turns any `.gguf` model into a real, standing service on **your** machine — chat, vision, speech, embeddings, reranking, retrieval — all speaking the OpenAI API dialect your tools already know. Point your existing app at `http://localhost:11435` instead of `api.openai.com`, and nothing else has to change.

You decide which models load, how much RAM/VRAM they get, when they sleep, and who's allowed to talk to them. Nobody else's usage policy, nobody else's outage, nobody else's log retention.

**This is what it looks like to own your AI.**

---

## 📚 Table of Contents

- [What You Get](#-what-you-get)
- [Installation](#️-installation)
- [Prebuilt Packages](#-prebuilt-packages)
- [Your First 5 Minutes](#-your-first-5-minutes)
- [Using Models You Already Have](#-using-models-you-already-have)
- [CLI Reference](#️-cli-reference)
- [Configuration](#️-configuration)
- [Registry Parameter Reference](CONFIG.md)
- [Benchmarking](#-benchmarking-zallama-bench)
- [Memory-Aware Eviction](#-memory-aware-eviction)
- [Fitting Your Models on One GPU](docs/vram-planning.md)
- [Doubling Decode Speed with MTP](docs/mtp-speculative-decoding.md)
- [What Qwen3-0.6B Can Actually Do in an Agentic Client](docs/qwen3-0.6b-agentic.md)
- [Model Tuning Log](docs/tuning-log.md)
- [Vision (Multimodal) Models](#️-vision-multimodal-models)
- [Speech-to-Text (ASR)](#-speech-to-text-asr)
- [Text-to-Speech (TTS)](#-text-to-speech-tts)
- [Image Generation (Stable Diffusion)](#-image-generation-stable-diffusion)
- [Backends & Modalities (Architecture)](#-backends--modalities-architecture)
- [RAG: Reranking & the zvec Vector Store](#-rag-reranking--the-zvec-vector-store)
- [OpenAI API Integration](#-openai-api-integration)
- [Deployment (systemd)](#-deployment-systemd)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#️-license)

---

## ✨ What You Get

| | |
|---|---|
| 🚀 **A CLI that gets out of your way** | `zallama serve`, `zallama pull`, `zallama run`, `zallama set`, `zallama ps` — five commands cover most of your day. |
| ⚡ **Fast model acquisition** | Downloads accelerate through `aria2c` (8 parallel connections), with a pure-Python parallel fallback. |
| 🧠 **Reasoning models, rendered live** | DeepSeek-R1, Qwen 3.5, and other thinking models stream their reasoning in dim/gray inside the interactive chat. |
| 🔌 **A real OpenAI `/v1` surface** | Chat, Completions, Embeddings — streaming included — so existing SDKs and tools just work. |
| 👁️ **Vision** | Attach an `mmproj` projector and send images straight through `/v1/chat/completions`. |
| 🎙️ **Speech-to-text** | `/v1/audio/transcriptions`, any input format auto-transcoded via `ffmpeg`, multilingual models supported. |
| 🎨 **Image Generation** | `/v1/images/generations` powered by `sd-server` (stable-diffusion.cpp), plus `zallama generate` CLI. |
| 🔎 **RAG, built in** | A reranker at `/v1/rerank` plus **zvec**, an embedded HNSW vector store — no external vector DB to run. |
| 🧩 **Pluggable backends** | Each model declares a `modality` + `backend`; new engines slot in without touching the core. |
| ⚙️ **Config, not code** | Context size, GPU offload, batching — all YAML, all per-model, all hot-reloadable. |
| 🔄 **A process manager that behaves** | Health-checked startup, port assignment, concurrency caps, LRU eviction when idle. |
| 🧠 **Memory awareness** | Set a `mem_budget_gb` and Zallama evicts least-recently-used models to make room — automatically. |
| 🔒 **Locked down by default** | Binds to `127.0.0.1`, optional Bearer-token auth, sane timeouts out of the box. |

Under the hood, Zallama is a **dynamic router and process manager** for your local GGUF models. Ask for a model, and it starts the right backend (`llama-server`, `parakeet-server`, `kokoro-server`, `sd-server`), routes your request to it, and unloads it after a period of inactivity so your RAM/VRAM goes back to you. Each model declares a `modality` (`text`, `embedding`, `rerank`, `asr`, `tts`, `image`); new modalities are added as new backends, not as changes scattered across the codebase.

---

## 🛠️ Installation

**Requirements:** Python 3.10+, and a `llama-server` binary (built from [llama.cpp](https://github.com/ggml-org/llama.cpp) or placed in `./bin/llama-server`). `aria2c` is optional but recommended for fast downloads. For **ASR** and **TTS**, build their respective engines and have `ffmpeg` installed.

### 1. Clone the repository

```bash
git clone https://github.com/rzafiamy/zallama.git
cd zallama
```

### 2. Build the inference engines

Helper scripts build each engine and install the binaries into `./bin/` (the clone and build happen in a temp directory, keeping your repo tree clean):

```bash
# llama.cpp (text / chat / embeddings / vision) — requires a release tag/branch name
./build-ggml-llama.cpp.sh b4600

# parakeet.cpp (ASR / speech-to-text) — requires a release tag/branch name
./build-ggml-parakeet.cpp.sh master

# kokoro.cpp (TTS / voice synthesis) — requires a release tag/branch name
./build-ggml-kokoro.cpp.sh v0.3.0

# stable-diffusion.cpp (Image generation) — requires a release tag/branch name
./build-ggml-stable-diffusion.cpp.sh master
```

> All scripts default to a **CUDA** build. The parakeet and stable-diffusion scripts also copy shared libraries next to the binaries and set their `RPATH` to `$ORIGIN` (via `patchelf`) so they resolve at runtime.
>
> `kokoro.cpp` requires **CMake 3.29+**. Ubuntu 24.04's apt package is 3.28 — install a newer CMake and run the script with `CMAKE_BIN=/path/to/cmake`.
>
> `kokoro.cpp` v0.2.0+ picks up **espeak-ng** at runtime (`dlopen`) for grapheme-to-phoneme and is ~2.5x faster with it than with the bundled ByT5 phonemizer. The build script installs it; on a machine that only *runs* the binary, install it yourself:
> `sudo apt install -y libespeak-ng1 espeak-ng-data libpcaudio0 libsonic0`.
> `kokoro-server` logs which phonemizer it selected at startup (`phonemizer: espeak-ng` vs `phonemizer: ByT5 (bundled)`).

### Optional: use prebuilt packages

To skip building from source, prebuilt packages are available from the shared Google Drive folder:

<https://drive.google.com/drive/folders/1B7AmE36r869kpMZbOatqMW-Dhedq2Sil?usp=sharing>

Download the package matching your platform/accelerator stack, copy the binaries into `./bin/`, then run the installer. At minimum, Zallama needs `llama-server` in `./bin/` or on your `PATH`. For speech-to-text, text-to-speech, or image generation, also copy `parakeet-server`, `kokoro-server`, or `sd-server`.

### 3. Run the installer

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

## 📦 Prebuilt Packages

If you'd rather not build the inference engines from source, download a prebuilt package from the shared Google Drive folder:

<https://drive.google.com/drive/folders/1B7AmE36r869kpMZbOatqMW-Dhedq2Sil?usp=sharing>

After extracting, place the engine binaries in the repository's `bin/` directory:

```bash
mkdir -p ./bin
cp /path/to/extracted/package/llama-server ./bin/
```

Optional modality backends can be copied the same way:

```bash
cp /path/to/extracted/package/parakeet-server ./bin/   # ASR
cp /path/to/extracted/package/kokoro-server ./bin/     # TTS
```

Then continue with the installer:

```bash
sudo bash install.sh
```

The installer checks `./bin/` first, so locally copied binaries are used automatically.

See [PREBUILT.md](PREBUILT.md) for package naming rules and the `prebuilt-cli.sh` helper, which detects your Ubuntu/CUDA setup, lists compatible packages, and installs the selected archive.

---

## 🚀 Your First 5 Minutes

This is the whole loop — start the daemon, pull a model, talk to it. Everything else in this README is a deeper look at one of these four steps.

### 1 · Wake the daemon

```bash
zallama serve
```
*Starts the FastAPI controller on `http://localhost:11435`. It sits quietly until a model is requested.*

### 2 · Pull a model — it's now yours

```bash
# A friendly shorthand (from Unsloth's repo)
zallama pull llama3.2:3b

# Or grab any GGUF straight from HuggingFace
zallama pull unsloth/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf

# Speech-to-text model, auto-registered as modality=asr
zallama pull parakeet:0.6b
zallama pull mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf   # multilingual
```

> Parakeet GGUF repos are auto-detected and registered on the ASR backend. For a raw HF path with an ambiguous name, force it with `zallama pull <repo>/<file>.gguf --type asr`.

### 3 · Tune it to your hardware

```bash
zallama set qwen3.5-4b-q4_k_m reasoning=false ctx_size=8192   # skip thinking blocks
zallama set llama3.2:1b n_gpu_layers=99                        # push everything to GPU
```

### 4 · Talk to it

```bash
zallama run llama3.2:3b
```

That's it. You now have a model running on hardware you control, answering only to you.

---

## 📦 Using Models You Already Have

Zallama can serve GGUF files you already downloaded, but it won't auto-discover files sitting in a directory — a model has to be **registered** before it shows up in `zallama list` or `/v1/models`. This is deliberate: nothing runs, and no memory is claimed, until you say so.

Check where Zallama looks for models:

```bash
zallama config
```

Default is `~/.zallama/models`. Copy a model there:

```bash
cp ~/Downloads/mistral-7b-q4.gguf ~/.zallama/models/
```

Register it:

```bash
zallama add mistral-7b ~/.zallama/models/mistral-7b-q4.gguf
```

Or register a file outside `models_dir` by passing its absolute path:

```bash
zallama add my-model /data/models/my-model-q4_k_m.gguf
```

For manual registry edits, relative paths resolve under `models_dir`:

```yaml
models:
  - name: "mistral-7b"
    file: "mistral-7b-q4.gguf"
    params:
      ctx_size: 8192
      n_gpu_layers: 99
```

Then run it like anything else you pulled:

```bash
zallama run mistral-7b
```

**Vision models** need one extra file — copy the base GGUF and its projector:

```bash
cp ~/Downloads/qwen2.5-vl-7b-q4_k_m.gguf ~/.zallama/models/
cp ~/Downloads/mmproj-qwen2.5-vl-7b-f16.gguf ~/.zallama/models/
```

Register both together:

```bash
zallama add qwen2.5-vl-7b \
  ~/.zallama/models/qwen2.5-vl-7b-q4_k_m.gguf \
  --mmproj ~/.zallama/models/mmproj-qwen2.5-vl-7b-f16.gguf
```

Or by hand:

```yaml
models:
  - name: "qwen2.5-vl-7b"
    file: "qwen2.5-vl-7b-q4_k_m.gguf"
    modality: "text"
    backend: "llama-server"
    artifacts:
      mmproj: "mmproj-qwen2.5-vl-7b-f16.gguf"
    params:
      ctx_size: 8192
      n_gpu_layers: 99
```

When the model loads, Zallama resolves both paths relative to `models_dir` and passes `--mmproj <file>` to `llama-server`. If `mmproj` is missing or not listed under `artifacts`, the model may still load for text, but image input won't work.

---

## 🖥️ CLI Reference

```
serve                  Start the Zallama daemon
list                   List registered models (alias: ls)
add <name> <file>      Register a local .gguf model
                       --mmproj attaches a vision projector file
set <name> <k>=<v>...  Configure parameters for a registered model
pull <name> [--type T] Pull model from HF / Unsloth presets (uses aria2c).
                       --type sets modality (text|embedding|rerank|asr|tts|image) for raw HF paths.
remove <name>          Remove a model from registry (alias: rm)
run <name>             Interactive chat with a model (streams reasoning)
generate <name> "<p>"  Generate an image with a diffusion model (alias: gen)
                       --output out.png --size 512x512 --steps 20
                       --cfg-scale 7.0 --negative-prompt "..."
ps                     Show running model processes
load <name>            Pre-load a model (start llama-server)
unload <name>          Stop a running model (alias: stop)
reload <name>          Restart a running model to apply registry param changes
calibrate <name>       Recommend max ctx_size + mem_gb from your VRAM (dry-run)
bench <name>           Measure tokens/sec, sweeping params like ctx_size
                       --sweep k=v1,v2 --prompt-tokens N --concurrency N --out f
logs <name>            Tail logs for a model
health                 Show daemon health status
version                Show version info
```

### Tab Completion

`./install.sh` installs **bash** and **zsh** completion automatically when it can write to the system completion directories. Once enabled, `<TAB>` completes both subcommands and **registered model names**:

```bash
zallama r<TAB>          # → reload  remove  rm  run
zallama run <TAB>       # → completes from your registered models
```

Completion reads the model list straight from `registry.yaml`, so it works even when the daemon isn't running. To install manually:

```bash
# bash
sudo cp completions/zallama.bash /usr/share/bash-completion/completions/zallama
# zsh — copy into a dir on your $fpath, then run compinit
cp completions/_zallama ~/.zsh/completions/_zallama
```

---

## ⚙️ Configuration

There are two layers, and they change on different timescales:

- **Global daemon config** (`config/config.yaml`) — host, port, timeouts, memory budgets, defaults. Read once at startup.
- **Per-model registry** (`models/registry.yaml`) — one entry per model, hot-reloaded from disk on every request.

### See what's active

```bash
zallama config
```

Shows the effective configuration, where it was loaded from, and the available keys.

### Change global settings

```bash
mkdir -p ~/.zallama
cp config/config.example.yaml ~/.zallama/config.yaml
$EDITOR ~/.zallama/config.yaml
```

Zallama looks for the first existing config file, in order:

1. `<repo>/config/config.yaml`
2. `~/.zallama/config.yaml`
3. `/etc/zallama/config.yaml`

Anything omitted falls back to built-in defaults. **Restart the daemon** after editing `config.yaml` — global config is read at startup:

```bash
systemctl restart zallama
```

### Change per-model settings

Prefer the CLI over hand-editing the registry:

```bash
zallama set <model> <key>=<value> [<key>=<value> ...]
zallama set llama3.2:1b ctx_size=8192 n_gpu_layers=99
zallama set my-reranker modality=rerank
zallama show llama3.2:1b
```

Per-model changes are saved to the registry and apply on the model's next load. If it's already running, its process keeps the params it launched with — reload it to pick up the change:

```bash
zallama reload <model>
```

### Environment overrides

For a one-off shell/session, without touching any file:

```bash
ZALLAMA_HOST=0.0.0.0 ZALLAMA_PORT=11435 zallama serve
ZALLAMA_MODELS_DIR=/data/models zallama serve
LLAMA_GPU_LAYERS=99 zallama serve
ZALLAMA_EMBEDDING_MODEL=my-embed zallama serve
```

### Global settings reference (`config/config.yaml`)

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

### Model registry reference (`models/registry.yaml`)

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
- **`modality`** — `text` (default), `embedding`, `rerank`, `asr`, `tts`, or the planned `image`. Determines which endpoints the model may serve; requests to a mismatched endpoint return a clear `400`. (Legacy embedding models registered as `text` with `params: embedding: true` are still treated as `embedding` at runtime.)
- **`backend`** — which engine runs the model (default `llama-server`). New backends resolve their own binary from `./bin/<name>`, `~/.zallama/bin/<name>`, or `PATH`.
- **`artifacts`** — extra files beyond the primary GGUF (e.g. `mmproj` for vision, and — for future backends — vocoders, etc.). Paths are absolute or relative to `models_dir`.
- **`mem_gb`** — declared memory footprint, used by memory-aware eviction (see below). If omitted, it's estimated from the GGUF file size — an estimate that is frequently off by 100% or more, so [measure it](docs/vram-planning.md#measuring-what-a-model-actually-costs).
- **`pinned`** — `true` keeps the model loaded for the daemon's lifetime: pre-warmed at startup, exempt from both idle sweep and eviction. Intended for small always-on services (ASR, TTS), not for large models.

Useful `params` for making a model fit on a busy GPU — `cache_type_k`/`cache_type_v` (quantize the KV cache), `ctx_size`, `n_cpu_moe` (keep the expert weights of the first N layers in system RAM, MoE models only) and `n_gpu_layers` — are covered with measured trade-offs in [Fitting Your Models on One GPU](docs/vram-planning.md#making-a-model-fit).

If your GGUF carries a baked-in MTP head — llama.cpp logs its tensors as `unused tensor blk.N.nextn.* -- ignoring` until you switch it on — `spec_type: draft-mtp` roughly doubles decode speed for about 1 GiB of VRAM. See [Doubling Decode Speed with MTP](docs/mtp-speculative-decoding.md). For a model with no baked-in head but a genuinely separate draft checkpoint, register it as the `draft` artifact (`--model-draft`) and set `spec_type: draft-simple` instead; `spec_draft_ngl` caps how much of the draft model rides on VRAM.

Sampling knobs — `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty` — are also settable per-model. They become the `llama-server` launch defaults, so a chat request that sets its own value always wins; the registry value only applies when the request leaves the field out. Full list of every `params` key per modality/backend: [Registry Parameter Reference](CONFIG.md).

---

## 📊 Benchmarking (`zallama bench`)

`zallama calibrate` tells you what *fits*. `zallama bench` tells you what it *costs* — in tokens/sec, VRAM, and latency — for any combination of parameters you care to compare.

```bash
zallama bench qwen3.5-4b-q4_k_m
```

```
┌───────────────────────────────────────────────────────────────────────┐
│ PROMPT │ GEN │ LOAD s │ VRAM GB │ TTFT ms │ PREFILL t/s │ DECODE t/s  │
├────────┼─────┼────────┼─────────┼─────────┼─────────────┼─────────────┤
│    469 │ 128 │    1.5 │     4.8 │  118 ±2 │    8925 ±20 │  179.8 ±0.2 │
└───────────────────────────────────────────────────────────────────────┘
```

Three numbers, three different questions:

| | |
|---|---|
| **PREFILL t/s** | How fast it *reads*. Dominates long-prompt / RAG workloads. Tuned by `batch_size`, `ubatch_size`, `n_gpu_layers`. |
| **DECODE t/s** | How fast it *writes*. The number people mean by "tokens/sec". Tuned by quantization, `n_gpu_layers`, `flash_attn`. |
| **TTFT ms** | How fast it *answers*. What an interactive user actually feels. |

`LOAD s` and `VRAM GB` come along for free — which is how you find out that doubling `ctx_size` cost you 400 MB of VRAM and bought nothing.

### Sweeping parameters

`--sweep <param>=<v1>,<v2>` is repeatable and builds a full grid. Each combination is written to the registry, the model is **restarted** so llama-server actually launches with it, and your original parameters are **restored when the run ends** — including on Ctrl-C.

```bash
# Does a bigger context slow generation down? What does it cost in VRAM?
zallama bench qwen3.5-4b-q4_k_m --sweep ctx_size=4096,16384,65536

# What does thinking cost, at short and long prompts?
zallama bench qwen3.5-4b-q4_k_m --sweep reasoning=on,off -p 512,8192

# Find the offload cliff on a small GPU
zallama bench qwen3.5-9b-q4_k_m --sweep n_gpu_layers=0,20,40,99

# Quantized KV cache: how much ctx does the VRAM buy back?
zallama bench qwen3.5-4b-q4_k_m --sweep cache_type_k=f16,q8_0,q4_0

# Compare models head to head, and keep the numbers
zallama bench --all --runs 5 --out bench.md
```

Anything llama-server takes is fair game: `ctx_size`, `reasoning`, `reasoning_effort`, `n_gpu_layers`, `threads`, `batch_size`, `ubatch_size`, `flash_attn`, `parallel`, `cache_type_k`, `cache_type_v`, `spec_type`, `spec_draft_n_max`, `spec_draft_ngl`, `image_min_tokens`, `image_max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`.

Measured results from past sweeps — settings tried, tok/s and VRAM they measured — are kept in the [Model Tuning Log](docs/tuning-log.md), so a config doesn't get re-benchmarked from scratch every session.

### Options

| Flag | Default | |
|---|---|---|
| `-p, --prompt-tokens N[,N]` | `512` | Prompt sizes to send |
| `-n, --max-tokens N` | `128` | Tokens to generate per request |
| `-c, --concurrency N[,N]` | `1` | Simultaneous requests — adds a **TOTAL t/s** throughput column |
| `-r, --runs N` | `3` | Measured runs per point (reported as mean ± stdev) |
| `--warmup N` | `1` | Discarded runs per point |
| `--natural` | off | Let the model stop on its own; by default `ignore_eos` pins every run to exactly `--max-tokens` so rates compare identical work |
| `--reuse-cache` | off | Measure the *warm* prefill path; by default each prompt is uniquely prefixed so llama-server can't reuse its KV cache |
| `--all` | | Bench every registered text model |
| `-o, --out FILE` | | Write `.json`, `.csv` or `.md` — format from the extension |
| `--label NAME` | GPU name | Names this machine in the export |
| `--keep` | off | Leave the last swept params in the registry |
| `--dry-run` | | Print the matrix and exit |

### Comparing two machines

A tokens/sec number means nothing without the GPU, driver and llama.cpp build behind it, so every `.json` export carries them. Run the same command on each box, then compare:

```bash
# on each machine — identical workload, one file each
zallama bench Qwen3.6-35B-A3B-UD-Q4_K_M \
  --sweep ctx_size=32768 --sweep reasoning=on,off \
  -p 32000 -n 256 --runs 3 --out $(hostname).json

zallama bench --compare rtx4090.json rtx3090.json
```

```
 Machines
│ LABEL   │ GPU                     │ VRAM GiB │ DRIVER     │ CPU                │ POINTS │
│ rtx4090 │ NVIDIA GeForce RTX 4090 │ 24.0     │ 570.172.08 │ Core(TM) i9-14900K │ 2      │
│ rtx3090 │ NVIDIA GeForce RTX 3090 │ 24.0     │ 570.172.08 │ Core(TM) i9-14900K │ 2      │

 DECODE t/s  (higher is better, % vs rtx4090)
│ POINT                                     │ rtx4090 (base) │ rtx3090      │
│ …· reasoning=on · prompt 32000 · gen 256  │          182.2 │ 113.0 (-38%) │
│ …· reasoning=off · prompt 32000 · gen 256 │          181.6 │ 112.6 (-38%) │
```

The first file is the baseline; every other column shows its delta. **Only points with identical model, swept params, prompt size, max-tokens and concurrency are put side by side** — anything unmatched is reported, never silently averaged in. If the exports came from different llama.cpp builds, you get a warning, because part of the gap is then the engine rather than the hardware.

`--metric` picks which tables to print: `prefill_tps`, `decode_tps`, `throughput_tps`, `ttft_ms`, `load_s`, `vram_gib`, `gen_tokens` (default: the first two plus `ttft_ms`).

> Comparing two GPUs **in the same box** needs the daemon pinned to one of them: set `CUDA_VISIBLE_DEVICES=0` in the service environment, restart, bench, then repeat with `=1`.

> **On `reasoning=on/off`:** it will not move decode tok/s — the chat template changes, the arithmetic doesn't. What it changes is *how many tokens* the model emits before answering. To measure that cost, add `--natural` and compare the `gen_tokens` metric; with the default fixed-length generation both rows are identical by construction.

### Reading the results

- **PREFILL / DECODE come off llama.cpp's own clock** (its `timings` block), so they measure the engine, not your network. **TTFT and TOTAL are measured at the client** and include queueing and proxy overhead — which is what you want when the question is "how does this feel".
- A combination that **fails to load** (a `ctx_size` that won't fit, say) is reported inline and skipped; the sweep carries on and the failure is recorded in the export with its error.
- `--concurrency` above the model's `parallel` slots just queues. Sweep them together — `--sweep parallel=1,4 -c 4` — to see what batching actually buys.

---

## 🧠 Memory-Aware Eviction

Beyond the time-based idle sweep and the `max_loaded_models` count cap, Zallama can keep total loaded-model memory within a budget. Set `mem_budget_gb` and, before starting a model, Zallama evicts least-recently-used instances until the incoming model fits.

Each model's cost is taken from its declared `mem_gb`; if undeclared, it's estimated from the GGUF file size (≈ size × 1.2 for KV-cache overhead), falling back to `mem_init_gb` when the size is unknown. Declaring `mem_gb` is recommended for accuracy:

```bash
zallama set qwen3.5-4b-q4_k_m mem_gb=4
```

For a starting number without loading anything, `zallama calibrate <model>` reads the GGUF's own dimensions — counting only the layers that actually hold a KV cache, which on hybrid and sliding-window architectures is a fraction of the block count — and reports the largest `ctx_size` that still fits alongside the weights, the artifacts and a compute reserve. It writes nothing; it prints the `zallama set` line.

> **Measure it, don't guess it.** The `size × 1.2` fallback ignores the KV cache, the artifacts (`mmproj`, and the text encoders of image models) and the compute buffers. Real-world errors exceed 100% in **both** directions — a 4B model at a long context measured 12.9 GB against an estimate of 4.2 GB. A budget built on those estimates evicts models that would have fit *and* admits models that then OOM, so measure before enabling `mem_budget_gb`:
>
> ```bash
> python3 scripts/measure_vram.py --write   # run it on an idle GPU
> ```

### Keeping a model resident: `pinned`

A model with `pinned: true` in its registry entry is **pre-loaded when the daemon starts and never evicted**. It exists for small, latency-sensitive backends that would otherwise be evicted by every chat request and reloaded on the next call — ASR and TTS in particular, where a ~1 GB model turns a 780 ms round trip into an 80 ms one.

```yaml
- name: tdt-0.6b-v3-q8_0
  modality: asr
  backend: parakeet-server
  mem_gb: 1.3
  pinned: true
```

Two things to know. Pinned models **still count against `max_loaded_models`**, so size that cap as *(pinned services) + (concurrent big models)* — with one pinned TTS and `max_loaded_models: 1`, the only slot is permanently taken and every other model thrashes. And when the cap is reached but everything loaded is pinned, Zallama logs a warning and admits the incoming model **over budget** rather than killing a warm pinned instance.

Pinning a small model can make your largest one unloadable. **[Fitting Your Models on One GPU](docs/vram-planning.md)** works through that arithmetic and the levers — KV quantization, `ctx_size`, MoE expert offload (`n_cpu_moe`), `mmproj` — with measured numbers for each.

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

Zallama passes `--mmproj <file>` to the backend automatically, and image input flows through the standard OpenAI `/v1/chat/completions` endpoint (`image_url` message content).

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

## 🗣️ Text-to-Speech (TTS)

Speech synthesis runs on the **`kokoro-server`** backend ([kokoro.cpp](https://github.com/rzafiamy/kokoro.cpp)) and is exposed at the OpenAI-compatible `POST /v1/audio/speech` endpoint.

**1. Build the binary** (installs `kokoro-server` and `kokoro-cli` into `./bin/`):
```bash
./build-ggml-kokoro.cpp.sh v0.3.0
```

**2. Pull a model** (auto-registered as `modality: tts`, `backend: kokoro-server`):
```bash
zallama pull kokoro:82m     # Kokoro-82M, 54 voices across 8 languages
```

**3. Synthesize** — the response is a WAV stream:
```bash
curl http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro:82m","input":"Bonjour, comment allez-vous ?"}' \
  -o speech.wav
```

### Voice selection

Kokoro takes no language argument — it phonemizes according to the **voice prefix** (`ff_siwis` → French, `af_heart` → American English, `if_sara` → Italian, …). Sending French text with an English voice therefore reads it with English sounds.

So when a request names **no** voice, Zallama guesses the language from the text and picks that language's voice ([`server/tts_lang.py`](server/tts_lang.py)). Precedence:

| | |
|---|---|
| 1. `voice` in the request | always wins — auto-selection never overrides an explicit choice |
| 2. detected language | `fr` → `ff_siwis`, `en` → `af_heart`, plus `es`/`it`/`pt`/`hi`/`ja`/`zh` |
| 3. the model's registry `voice` param | used when the language can't be determined |
| 4. kokoro's own default | when no registry default is set either |

Detection is a small built-in heuristic, not a language identifier: CJK and Devanagari are settled by script, the Latin languages by function-word frequency. It deliberately answers "unknown" for very short inputs — `"Merci"` and `"Mercy"` are not distinguishable in five letters — and falls through to your registry default there. Pin `voice` in the request whenever you need a guaranteed result.

```bash
# Explicit voice — no detection, no surprises
curl http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro:82m","input":"Bonjour !","voice":"ff_siwis"}' \
  -o speech.wav
```

> Install **espeak-ng** (see [Installation](#2-build-the-inference-engines)) — without it kokoro falls back to a bundled phonemizer that is ~2.5x slower.

---

## 🎨 Image Generation (Stable Diffusion)

Text-to-image runs on the **`sd-server`** backend ([stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)) and is exposed at the OpenAI-compatible `POST /v1/images/generations` endpoint.

**1. Build the binary** (installs `sd-server` and the `sd` CLI into `./bin/`):
```bash
./build-ggml-stable-diffusion.cpp.sh master
```
The script builds with CUDA when `nvcc` is found and falls back to a CPU build otherwise.

**2. Pull a model** (auto-registered as `modality: image`, `backend: sd-server`):
```bash
zallama pull sd:1.5        # Stable Diffusion v1.5
zallama pull sdxl:turbo    # SDXL Turbo — few-step, near real-time
zallama pull flux:klein    # FLUX Klein (Compact 4-bit, fast 4-step generation)
zallama pull qwen-image:20b # Qwen-Image 20B (MMDiT image generation)
```
Diffusion weights ship as `.safetensors` / `.ckpt` rather than GGUF, and the downloader accepts those extensions for image repos. Already have weights locally? Register them directly:
```bash
zallama add sd15 /path/to/v1-5-pruned-emaonly.safetensors --modality image
```

**3. Generate** — from the CLI:
```bash
zallama generate sd:1.5 "a lighthouse at dawn, cinematic" --output dawn.png --size 512x512 --steps 20
```
or over HTTP:
```bash
curl http://localhost:11435/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"sd:1.5","prompt":"a lighthouse at dawn","size":"512x512","response_format":"b64_json"}'
```

Generation knobs (`steps`, `cfg_scale`, `sampler`, `negative_prompt`) can be set once per model and reused for every request:
```bash
zallama set sd:1.5 steps=25 cfg_scale=7.5
```
The daemon applies those registry values to any request that does not specify them. Auxiliary weights (`vae`, `taesd`, `control_net`, `clip_l`, `clip_g`, `t5xxl`) are passed to `sd-server` at launch when registered as artifacts on the model.

**Large images and VRAM.** The VAE decode buffer grows with the square of the image size and is allocated *after* the whole stack is resident, so a generation can sample all its steps and then die at the final decode — a FLUX 1024×1024 decode asks for ~6.6 GB on top of ~16 GB of weights, which does not fit on a 24 GB card. Set `vae_tiling` to decode the latent in patches instead, which cuts that buffer to a few hundred MB for no visible seam:
```bash
zallama set flux:klein vae_tiling=true diffusion_fa=true
```
The other memory switches are `fa` (flash attention everywhere), `diffusion_fa` (diffusion model only), `vae_conv_direct` / `diffusion_conv_direct`, `offload_to_cpu` (weights live in RAM, streamed into VRAM per graph), and `backend` for per-component placement (e.g. `backend=te=cpu`). Tile geometry is tunable with `vae_tile_size` and `vae_tile_overlap`.

> Image models are not chat models: `zallama run <name>` refuses them and points you at `zallama generate`.

---

## 🧩 Backends & Modalities (Architecture)

Zallama separates the **generic process lifecycle** (spawn, health-check, port assignment, LRU eviction, kill) from **engine-specific logic** (which binary to run, how to build its arguments, which health path to poll). The latter lives behind a `Backend` abstraction in [`server/backends.py`](server/backends.py).

This is the seam for new modalities. `LlamaServerBackend` covers text, chat, and vision; `EmbeddingServerBackend` runs `llama-server --embedding` for `/v1/embeddings`; `RerankServerBackend` runs `llama-server --reranking` for `/v1/rerank`; `ParakeetServerBackend` covers ASR (`/v1/audio/transcriptions`); `KokoroServerBackend` covers TTS (`/v1/audio/speech`); `SdServerBackend` covers image generation (`/v1/images/generations`). Each one arrived as a new `Backend` subclass plus a matching endpoint proxy — no changes to the process manager or registry schema.

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

Zallama acts as a standard OpenAI-compatible API gateway. Specify the model you want in the request body, and Zallama handles loading and routing automatically — meaning any OpenAI SDK, LangChain, or existing integration can point here with a one-line base-URL change:

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

Owning your AI means owning the responsibility for it running safely. Zallama defaults to **localhost-only** (`host: 127.0.0.1`) with no authentication, which is safe for single-user local use — and gets more deliberate the further you push it outward.

### API key

Set `host: "0.0.0.0"` **only together with** an API key. The easiest and safest way is the CLI:

```bash
zallama apikey                     # generate a key valid 30 days (default)
zallama apikey --expires 90d      # or 12h / 45m / 2026-09-01 / never
zallama apikey hash <your-key>    # keep a key you chose yourself
zallama apikey clear              # disable auth
```

`zallama apikey` generates a 256-bit random key, writes **only its SHA-256 hash** (plus the expiry instant) into `config.yaml`, chmods the file to 600, and prints the key **once** — it cannot be recovered afterwards, so store it in your password manager. Restart the daemon to apply. Expired keys are rejected with a 401 until you issue a new one.

With a key set, everything except `/` and `/health` — including the API docs — requires an `Authorization: Bearer <key>` header, verified hash-to-hash with a constant-time comparison:

```bash
curl http://localhost:11435/v1/models -H "Authorization: Bearer $YOUR_KEY"
```

(A plaintext `api_key` config value is still honored for backward compatibility, but `api_key_sha256` is preferred and wins if both are set.)

CORS allows all origins but does **not** send credentials.

> ⚠️ The management API (`/api`) can register models pointing at arbitrary file paths and spawn backend processes. Treat the API key like a shell credential.

### Remote access: prefer a VPN

For personal remote access, the most secure option is to **not expose Zallama publicly at all**: keep `host: 127.0.0.1` and bind to a WireGuard/Tailscale interface, reaching it over the VPN. Zero public attack surface, no TLS or auth to configure.

### Public exposure: reverse proxy

If it must face the internet, keep Zallama on `127.0.0.1`, set an `api_key` anyway (defense in depth), and put a TLS-terminating proxy in front. Example with Caddy (automatic Let's Encrypt certificates), exposing **only** the OpenAI-compatible inference surface:

```caddyfile
llm.example.com {
    # Only /v1 passes through; management API, docs, and root stay private
    @api path /v1/*
    reverse_proxy @api 127.0.0.1:11435
    respond 403

    request_body {
        max_size 25MB   # enough for audio uploads; tune down if text-only
    }
}
```

Additional layers worth adding:

- **Firewall:** allow only 80/443 inbound (`ufw allow 80,443/tcp` then `ufw enable`).
- **Ban abusive clients:** fail2ban or CrowdSec keyed on repeated 401/403 responses.
- **mTLS:** if your clients support client certificates, require them at the proxy — anonymous scanners can't even complete a TLS handshake.
- **systemd sandboxing:** in the unit file, run as an unprivileged user with:
  ```ini
  [Service]
  User=zallama
  NoNewPrivileges=yes
  ProtectSystem=strict
  ProtectHome=read-only
  ReadWritePaths=/home/zallama/.zallama
  PrivateTmp=yes
  ```

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
