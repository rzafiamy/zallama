#!/usr/bin/env bash
# build-voxtral-tts.sh
# Build voxtral-tts-server (TTS — Voxtral-4B-TTS-2603) for a given branch/tag
# of mudler/voxtral-tts.c and install it into ./bin/. Nothing is left behind
# in the repo: cloning and building happen in a temp dir.
#
# This applies our own patches/voxtral-tts-server.cpp (a thin, zallama-shaped
# --model/--host/--port HTTP server) onto a clean clone of the upstream
# inference engine, rather than compiling upstream's one-shot CLI (main.c).
# Two small single-header libraries are fetched at pinned versions to build
# it: cpp-httplib (HTTP) and nlohmann/json (request parsing).

set -euo pipefail

BRANCH_OR_TAG="${1:-main}"
HTTPLIB_VERSION="v0.56.0"
JSON_VERSION="v3.12.0"

echo "🗣️  Building voxtral-tts-server for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
apt install -y git build-essential libopenblas-dev curl

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"
PATCH_FILE="${WORKDIR}/patches/voxtral-tts-server.cpp"
if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "❌ ${PATCH_FILE} not found." >&2
    exit 1
fi

# --- Locate CUDA toolkit (same approach as the other build-*.sh scripts) ---
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$(command -v nvcc || true)"
if [[ -z "${NVCC}" ]]; then
    for c in "${CUDA_ROOT}/bin/nvcc" /usr/local/cuda-*/bin/nvcc; do
        [[ -x "$c" ]] && NVCC="$c" && break
    done
fi
if [[ -z "${NVCC}" ]]; then
    echo "❌ nvcc not found. Install the CUDA Toolkit compiler package or set CUDA_HOME." >&2
    exit 1
fi
CUDA_ROOT="$(dirname "$(dirname "$NVCC")")"
export PATH="${CUDA_ROOT}/bin:${PATH}"
echo "🔧 Using nvcc: ${NVCC} (CUDA root: ${CUDA_ROOT})"

# Modern CUDA toolkits (12.x) put the shared libs under
# <root>/targets/x86_64-linux/lib rather than <root>/lib64, and that path is
# usually not on the default linker search path — find it explicitly rather
# than hardcoding either layout.
CUDA_LIB_DIR="$(find -L "${CUDA_ROOT}" -maxdepth 4 -name 'libcudart.so' -printf '%h\n' 2>/dev/null | head -1)"
if [[ -z "${CUDA_LIB_DIR}" ]]; then
    echo "❌ libcudart.so not found under ${CUDA_ROOT}." >&2
    exit 1
fi
echo "🔧 Using CUDA libs: ${CUDA_LIB_DIR}"

# Build in a throwaway temp dir so the repo stays clean.
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/voxtral-tts.c"

# --- Clone the engine ---
echo "📦 Cloning mudler/voxtral-tts.c into temp dir..."
git clone https://github.com/mudler/voxtral-tts.c "$REPO_DIR"
cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"

# --- Apply our server on top ---
echo "📦 Applying voxtral-tts-server.cpp..."
cp "${PATCH_FILE}" .

echo "📦 Fetching httplib.h (${HTTPLIB_VERSION}) and json.hpp (${JSON_VERSION})..."
curl -sL -o httplib.h \
    "https://raw.githubusercontent.com/yhirose/cpp-httplib/${HTTPLIB_VERSION}/httplib.h"
curl -sL -o json.hpp \
    "https://raw.githubusercontent.com/nlohmann/json/${JSON_VERSION}/single_include/nlohmann/json.hpp"

# --- Compile the engine's own sources (everything but main.c) ---
echo "⚙️  Compiling engine sources (CUDA + OpenBLAS)..."
ENGINE_SRCS=(
    voxtral_tts.c
    voxtral_tts_safetensors.c
    voxtral_tts_kernels.c
    voxtral_tts_llm.c
    voxtral_tts_acoustic.c
    voxtral_tts_codec.c
    voxtral_tts_voice.c
    voxtral_tts_wav.c
    voxtral_tts_tokenizer.c
)
CFLAGS="-O3 -Wall -Wextra -Wno-unused-parameter -std=c11 -D_GNU_SOURCE -DUSE_CUDA -DUSE_BLAS"
OBJS=()
for src in "${ENGINE_SRCS[@]}"; do
    obj="${src%.c}.o"
    gcc ${CFLAGS} -c -o "${obj}" "${src}"
    OBJS+=("${obj}")
done
nvcc -O3 -arch="${CUDA_ARCH:-sm_80}" --use_fast_math -Xcompiler -fPIC \
    -DUSE_CUDA -c -o voxtral_tts_cuda.o voxtral_tts_cuda.cu
OBJS+=("voxtral_tts_cuda.o")

echo "🚀 Compiling voxtral-tts-server.cpp..."
g++ -O3 -std=c++17 -DUSE_CUDA -DUSE_BLAS -c -o voxtral-tts-server.o voxtral-tts-server.cpp

echo "🔗 Linking voxtral-tts-server..."
g++ -O3 -o voxtral-tts-server \
    "${OBJS[@]}" voxtral-tts-server.o \
    -L"${CUDA_LIB_DIR}" -Wl,-rpath,"${CUDA_LIB_DIR}" \
    -lopenblas -lcublas -lcudart -lstdc++ -lm -lpthread

# --- Install into ./bin ---
echo "📦 Copying binary into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"
cp voxtral-tts-server "$BIN_DIR/"

# --- Summary ---
echo "✅ Build complete! Binary installed to:"
echo "   → ${BIN_DIR}/voxtral-tts-server"
ls -l "${BIN_DIR}/voxtral-tts-server" 2>/dev/null || true
echo
echo "ℹ️  Register a Voxtral-4B-TTS model with backend: voxtral-tts-server,"
echo "   modality: tts — see CONFIG.md for the registry entry shape."
echo "   Model weights are CC BY-NC 4.0 (Mistral AI) — non-commercial use only."
