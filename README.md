# 🦙 Zallama

> An Ollama-like local LLM ecosystem powered by `llama-server` (llama.cpp).

Zallama acts as a dynamic router and process manager for your local GGUF models. It exposes a single, unified endpoint that is fully OpenAI-compatible. When you request a model, Zallama starts the underlying `llama-server` in the background, routes your request, and automatically unloads the model after a period of inactivity to free up RAM/VRAM.

---

## ✨ Features

- **🚀 Ollama-like CLI:** Run commands like `zallama serve`, `zallama run <model>`, `zallama add`, and `zallama ps`.
- **🔌 Full OpenAI /v1 API:** Full drop-in replacement for OpenAI endpoints (Chat, Completions, and Embeddings) with streaming supported via SSE.
- **🌐 Sleek Embedded Web UI:** Access model management, registration, loading/unloading, and streaming chat at `http://localhost:11434`.
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
*Starts the FastAPI controller and Web UI on `http://localhost:11434`.*

### 2. Register a Local Model
```bash
zallama add qwen3:4b /path/to/qwen3-4b-instruct-q4_k_m.gguf "Qwen3 4B Instruct"
```

### 3. Run Interactive Chat
```bash
zallama run qwen3:4b
```

---

## 🖥️ CLI Reference

```
serve           Start the Zallama daemon
list            List registered models (alias: ls)
add <name> <file>  Register a local .gguf model
remove <name>   Remove a model from registry (alias: rm)
run <name>      Interactive chat with a model
ps              Show running model processes
load <name>     Pre-load a model (start llama-server)
unload <name>   Stop a running model (alias: stop)
logs <name>     Tail logs for a model
health          Show daemon health status
version         Show version info
```

---

## ⚙️ Configuration Files

### Global Settings (`config/config.yaml`)
```yaml
zallama:
  host: "0.0.0.0"
  port: 11434
  models_dir: "~/.zallama/models"
  logs_dir: "~/.zallama/logs"

llama_server:
  binary: ""            # Auto-detects in ./bin/llama-server or PATH
  port_start: 8100      # Backends spawn on ports 8100, 8101, etc.
  idle_timeout: 300     # Auto-unload model after 300s of inactivity (0 to disable)
  default_params:
    ctx_size: 4096
    n_gpu_layers: 99    # Attempt full GPU offload by default
    threads: 8
    flash_attn: true
```

### Model Registry (`models/registry.yaml`)
```yaml
models:
  - name: "qwen3:4b"
    file: "/path/to/qwen3-4b.gguf"
    description: "Qwen3 4B Model"
    params:
      ctx_size: 8192
      n_gpu_layers: 99
```

---

## 🌐 OpenAI API Integration

Zallama acts as a standard OpenAI-compatible API gateway. Specify the model you want to target in the request body, and Zallama will handle model loading and routing automatically:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [
      {"role": "user", "content": "Tell me a joke."}
    ],
    "stream": true
  }'
```

---

## ⚖️ License
MIT
