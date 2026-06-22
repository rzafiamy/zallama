# 🦙 Zallama

> An Ollama-like local LLM ecosystem powered by `llama-server` (llama.cpp).

Zallama acts as a dynamic router and process manager for your local GGUF models. It exposes a single, unified endpoint that is fully OpenAI-compatible. When you request a model, Zallama starts the underlying `llama-server` in the background, routes your request, and automatically unloads the model after a period of inactivity to free up RAM/VRAM.

---

## ✨ Features

- **🚀 Ollama-like CLI:** Run commands like `zallama serve`, `zallama run <model>`, `zallama add`, `zallama set`, and `zallama ps`.
- **⚡ High-Performance Downloads:** Accelerated download engine utilizing `aria2c` (with 8 concurrent connections), falling back to Python-based parallel HTTP range requests and standard stream decoders.
- **🧠 Reasoning Model Support:** Full support for thinking models (e.g., DeepSeek-R1, Qwen 3.5), rendering thinking/reasoning blocks in real-time with dim/gray coloring inside the interactive chat.
- **🔌 Full OpenAI /v1 API:** Full drop-in replacement for OpenAI endpoints (Chat, Completions, and Embeddings) with streaming supported via SSE.
- **🌐 Sleek Embedded Web UI:** Access model management, registration, loading/unloading, and streaming chat at `http://localhost:11435`.
- **⚙️ Config-Driven Architecture:** Define global defaults and customize per-model parameters (context size, GPU layers offload, batching options) in simple YAML configurations.
- **🔄 Dynamic Process Management:** Automatic port assignment, server health checking, and automatic LRU model eviction/unloading when idle.

---

## 🛠️ Installation

Simply run the one-shot installer:

```bash
chmod +x install.sh
./install.sh
```

To run the `zallama` command from anywhere, add it to your PATH:
```bash
export PATH="/home/cook/Documents/Dev/Dev-ai/zallama:$PATH"
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
logs <name>            Tail logs for a model
health                 Show daemon health status
version                Show version info
```

---

## ⚙️ Configuration Files

### Global Settings (`config/config.yaml`)
```yaml
zallama:
  host: "0.0.0.0"
  port: 11435
  models_dir: "~/.zallama/models"
  logs_dir: "~/.zallama/logs"

llama_server:
  binary: ""            # Auto-detects in ./bin/llama-server or PATH
  port_start: 8100      # Backends spawn on ports 8100, 8101, etc.
  idle_timeout: 300     # Auto-unload model after 300s of inactivity (0 to disable)
  default_params:
    ctx_size: 8192
    n_gpu_layers: 99    # Attempt full GPU offload by default
    threads: 8
    flash_attn: true
    parallel: 1         # Single-session slot allocation (avoids context limits)
```

### Model Registry (`models/registry.yaml`)
```yaml
models:
  - name: "qwen3.5-4b-q4_k_m"
    file: "/home/cook/.zallama/models/Qwen3.5-4B-Q4_K_M.gguf"
    description: "Downloaded from unsloth/Qwen3.5-4B-GGUF"
    params:
      ctx_size: 8192
      n_gpu_layers: 99
      reasoning: false  # Bypasses thinking blocks for immediate responses
```

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

## ⚖️ License
MIT
