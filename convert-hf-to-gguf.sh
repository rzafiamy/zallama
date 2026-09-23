#!/usr/bin/env bash
# convert-hf-to-gguf.sh
# Convert a Hugging Face safetensors model into a GGUF that zallama's own
# llama-server can load, e.g. a fine-tune that nobody has published as GGUF.
#
# The converter is taken from llama.cpp at the *same commit* as ./bin/llama-server
# (read from `llama-server --version`), so the GGUF matches the runtime that will
# serve it. The architecture in config.json is checked against that converter
# before anything large is downloaded: an unsupported model fails in seconds,
# not after a 10 GB download.
#
# Usage:
#   ./convert-hf-to-gguf.sh <hf-repo | local-dir> [outtype] [out-dir]
#     outtype  q8_0 (default) | f16 | bf16 | f32 | auto
#     out-dir  default: zallama.models_dir from ~/.zallama/config.yaml
#   e.g. ./convert-hf-to-gguf.sh wayfind/metask-jev-4b-policy-mix q8_0
#
# Env overrides:
#   REVISION      HF revision to download (default: main)
#   KEEP_SOURCE=1 keep the downloaded safetensors (default: removed after success)
#   LLAMA_SERVER  llama-server whose commit pins the converter (default: ./bin/llama-server)
#   CACHE_DIR     converter checkout + venv (default: ~/.cache/zallama/convert)
#
# Only the text model is converted. A vision tower, if any, needs a separate
# `--mmproj` conversion and is not handled here.

set -euo pipefail

SRC="${1:?Usage: $0 <hf-repo | local-dir> [outtype] [out-dir]}"
OUTTYPE="${2:-q8_0}"
OUT_DIR="${3:-}"
REVISION="${REVISION:-main}"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$WORKDIR/bin/llama-server}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/zallama/convert}"

case "$OUTTYPE" in q8_0|f16|bf16|f32|auto) ;; *)
    echo "❌ outtype must be q8_0, f16, bf16, f32 or auto (got '$OUTTYPE')." >&2
    echo "   Smaller quants (Q4_K_M, …) need llama-quantize on the q8_0/f16 output." >&2
    exit 1 ;;
esac

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$(sed -n 's/^\s*models_dir:\s*"\{0,1\}\([^"#]*\)"\{0,1\}.*/\1/p' "$HOME/.zallama/config.yaml" 2>/dev/null | head -1 | xargs)"
    OUT_DIR="${ZALLAMA_MODELS_DIR:-${OUT_DIR:-$HOME/.zallama/models}}"
fi
OUT_DIR="${OUT_DIR/#\~/$HOME}"
mkdir -p "$OUT_DIR"

# --- Pin the converter to the runtime's llama.cpp commit ---
[[ -x "$LLAMA_SERVER" ]] || { echo "❌ llama-server not found at $LLAMA_SERVER (build it first)." >&2; exit 1; }
COMMIT="$("$LLAMA_SERVER" --version 2>&1 | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -1)"
[[ -n "$COMMIT" ]] || { echo "❌ Could not read the commit from '$LLAMA_SERVER --version'." >&2; exit 1; }

REPO_DIR="$CACHE_DIR/llama.cpp"
VENV="$CACHE_DIR/venv"
mkdir -p "$CACHE_DIR"
echo "🔗 llama-server commit: $COMMIT"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp "$REPO_DIR"
fi
if ! git -C "$REPO_DIR" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    git -C "$REPO_DIR" fetch -q origin
fi
git -C "$REPO_DIR" checkout -q "$COMMIT"

# --- Converter venv (CPU torch; rebuilt when the pinned requirements change) ---
REQ="$REPO_DIR/requirements/requirements-convert_hf_to_gguf.txt"
REQ_HASH="$(cat "$REQ" "$REPO_DIR"/requirements/requirements-convert_legacy_llama.txt | sha256sum | cut -c1-16)"
if [[ ! -x "$VENV/bin/python" || "$(cat "$VENV/.req-hash" 2>/dev/null)" != "$REQ_HASH" ]]; then
    echo "🐍 Preparing converter venv (CPU torch, one-time) in $VENV"
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    (cd "$REPO_DIR/requirements" && "$VENV/bin/pip" install -q -r "$(basename "$REQ")")
    echo "$REQ_HASH" > "$VENV/.req-hash"
fi
# The gguf writer must match the converter, not whatever PyPI has.
"$VENV/bin/pip" install -q --no-deps --force-reinstall "$REPO_DIR/gguf-py"
PY="$VENV/bin/python"

# --- Fetch config.json only, and check the architecture is supported ---
if [[ -d "$SRC" ]]; then
    CONFIG="$SRC/config.json"
    NAME="$(basename "$(cd "$SRC" && pwd)")"
else
    NAME="${SRC##*/}"
    CONFIG="$("$PY" -c 'import sys; from huggingface_hub import hf_hub_download as d; print(d(sys.argv[1], "config.json", revision=sys.argv[2]))' "$SRC" "$REVISION")"
fi
ARCH="$("$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print((c.get("architectures") or ["?"])[0])' "$CONFIG")"
if ! grep -rqs "\"$ARCH\"" "$REPO_DIR/conversion/" "$REPO_DIR/convert_hf_to_gguf.py"; then
    echo "❌ Architecture '$ARCH' is not supported by llama.cpp at $COMMIT." >&2
    echo "   Rebuild llama-server from a newer llama.cpp (./build-ggml-llama.cpp.sh <tag>) if support has landed since." >&2
    exit 2
fi
echo "✅ Architecture '$ARCH' is supported at $COMMIT"

OUTFILE="$OUT_DIR/${NAME}-${OUTTYPE^^}.gguf"
[[ "$OUTTYPE" == "auto" ]] && OUTFILE="$OUT_DIR/${NAME}.gguf"
if [[ -e "$OUTFILE" ]]; then
    echo "❌ $OUTFILE already exists; remove it first." >&2
    exit 1
fi

# --- Download weights next to the models (not on the system disk) ---
if [[ -d "$SRC" ]]; then
    MODEL_DIR="$SRC"
else
    MODEL_DIR="$OUT_DIR/.hf-staging/$NAME"
    echo "📥 Downloading $SRC@$REVISION into $MODEL_DIR"
    "$PY" - "$SRC" "$REVISION" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"])
EOF
fi

# --- A declared MTP head that the checkpoint does not ship ---
# Qwen3.5/3.6 configs declare `mtp_num_hidden_layers`, and the converter then
# announces one extra block. Fine-tunes routinely drop the `mtp.*` tensors
# (merging a LoRA through transformers does), which yields a GGUF that fails
# to load with "tensor 'blk.N.attn_norm.weight' not found". Read the tensor
# names from the safetensors headers and convert without MTP in that case.
EXTRA_ARGS=()
if "$PY" - "$MODEL_DIR" <<'EOF'
import json, struct, sys
from pathlib import Path
d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
declared = max(cfg.get("mtp_num_hidden_layers", 0), (cfg.get("text_config") or {}).get("mtp_num_hidden_layers", 0))
names = []
for f in d.glob("*.safetensors"):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        names += json.loads(fh.read(n)).keys()
sys.exit(0 if declared and not any(".mtp." in k or k.startswith("mtp.") for k in names) else 1)
EOF
then
    echo "ℹ️  config.json declares an MTP head but the weights have none: converting with --no-mtp"
    EXTRA_ARGS+=(--no-mtp)
fi

# --- Convert ---
echo "⚙️  Converting to $OUTTYPE → $OUTFILE"
TQDM_DISABLE=1 "$PY" "$REPO_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype "$OUTTYPE" --outfile "$OUTFILE" \
    "${EXTRA_ARGS[@]}" 2>&1 | { grep -vE "^INFO:hf-to-gguf:(gguf: |blk\.|output|token_embd)" || true; }
[[ -s "$OUTFILE" ]] || { echo "❌ Conversion produced no output file." >&2; exit 1; }

if [[ ! -d "$SRC" && "${KEEP_SOURCE:-0}" != "1" ]]; then
    rm -rf "$MODEL_DIR"
    rmdir "$OUT_DIR/.hf-staging" 2>/dev/null || true
fi

echo
echo "✅ Done: $OUTFILE ($(du -h "$OUTFILE" | cut -f1))"
echo "   Register it with: zallama add <name> $OUTFILE"
