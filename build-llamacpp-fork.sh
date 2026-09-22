#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build-llamacpp-fork.sh — build ANY llama.cpp fork into a separate,
# non-production binary pair (llama-fork-server / llama-fork-cli).
#
# Purpose: test experimental forks (custom quant kernels, new backends, …)
# without ever touching the mainline llama-server that zallama's other
# registered models depend on in production. Installs into
# ~/.zallama/bin/, one of zallama's normal binary search paths, under the
# "llama-fork-*" name so it's picked up only by a model explicitly set to
# `backend=llama-fork-server` — every other model keeps using the regular
# llama-server untouched.
#
# Binaries are linked statically (BUILD_SHARED_LIBS=OFF): a shared build
# leaves llama-fork-server with a RUNPATH into BUILD_ROOT, so clearing that
# cache — or rebuilding the checkout mid-run — silently breaks the backend.
#
# Usage:
#   build-llamacpp-fork.sh <git-url> [branch] [install-dir]
#   e.g. ./build-llamacpp-fork.sh https://github.com/PrismML-Eng/llama.cpp prism
#
# Env overrides:
#   BUILD_ROOT   where sources/build trees live (default: ~/.cache/llamacpp-forks)
#   JOBS         parallel build jobs (default: nproc)
#   CUDA         ON|OFF (default: ON if nvidia-smi is present, else OFF)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${1:?Usage: $0 <git-url> [branch] [install-dir]}"
BRANCH="${2:-}"
INSTALL_DIR="${3:-$HOME/.zallama/bin}"
BUILD_ROOT="${BUILD_ROOT:-$HOME/.cache/llamacpp-forks}"
JOBS="${JOBS:-$(nproc)}"
if [[ -z "${CUDA:-}" ]]; then
    CUDA="OFF"
    command -v nvidia-smi >/dev/null 2>&1 && CUDA="ON"
fi

# nvcc is often not on PATH even when the toolkit is installed; probe the
# usual locations so CMake's CUDA language check can find it.
if [[ "$CUDA" == "ON" ]] && ! command -v nvcc >/dev/null 2>&1; then
    for cudadir in /usr/local/cuda /usr/local/cuda-*; do
        if [[ -x "$cudadir/bin/nvcc" ]]; then
            export PATH="$cudadir/bin:$PATH"
            export CUDACXX="$cudadir/bin/nvcc"
            break
        fi
    done
fi

REPO_NAME="$(basename "$REPO_URL" .git)"
SRC_DIR="$BUILD_ROOT/${REPO_NAME}${BRANCH:+-$BRANCH}"

mkdir -p "$BUILD_ROOT" "$INSTALL_DIR"

echo "==> Fork:      $REPO_URL${BRANCH:+  (branch: $BRANCH)}"
echo "==> Source:    $SRC_DIR"
echo "==> Install:   $INSTALL_DIR/llama-fork-server, llama-fork-cli"
echo "==> CUDA:      $CUDA"
echo

if [[ -d "$SRC_DIR/.git" ]]; then
    echo "==> Existing checkout found, fetching latest"
    git -C "$SRC_DIR" fetch --depth 1 origin "${BRANCH:-HEAD}"
    git -C "$SRC_DIR" reset --hard FETCH_HEAD
else
    echo "==> Cloning"
    if [[ -n "$BRANCH" ]]; then
        git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$SRC_DIR"
    else
        git clone --depth 1 "$REPO_URL" "$SRC_DIR"
    fi
fi

echo
echo "==> Configuring (cmake, GGML_CUDA=$CUDA)"
cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA="$CUDA" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF

echo
echo "==> Building (jobs=$JOBS) — this can take a while"
cmake --build "$SRC_DIR/build" -j "$JOBS" --target llama-server llama-cli

BIN_DIR="$SRC_DIR/build/bin"
if [[ ! -f "$BIN_DIR/llama-server" ]]; then
    # Some layouts drop binaries at build root instead of build/bin.
    BIN_DIR="$SRC_DIR/build"
fi

echo
echo "==> Installing as llama-fork-server / llama-fork-cli"
install -m 755 "$BIN_DIR/llama-server" "$INSTALL_DIR/llama-fork-server"
install -m 755 "$BIN_DIR/llama-cli" "$INSTALL_DIR/llama-fork-cli"

echo
echo "✔ Done: $INSTALL_DIR/llama-fork-server"
"$INSTALL_DIR/llama-fork-server" --version || true
