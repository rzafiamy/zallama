#!/usr/bin/env bash
# build-ggml-parakeet.cpp.sh
# Build parakeet.cpp for a given branch or tag and install binaries into ./bin/
# Nothing is left behind in the repo: cloning and building happen in a temp dir.

set -euo pipefail

BRANCH_OR_TAG="${1:-master}"

echo "🦜 Building parakeet.cpp for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
# patchelf: rewrite RPATH=$ORIGIN so the binaries find libggml*.so in ./bin.
apt install -y git cmake build-essential libcurl4-openssl-dev patchelf

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"

# Build in a throwaway temp dir so the repo stays clean
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/parakeet.cpp"

# --- Clone (recursive: pulls in third_party/ggml) ---
echo "📦 Cloning parakeet.cpp into temp dir..."
git clone --recursive https://github.com/mudler/parakeet.cpp "$REPO_DIR"

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"
# Make sure ggml (and any other) submodules match the checked-out revision
git submodule update --init --recursive

# --- Build ---
echo "⚙️ Configuring build..."
# parakeet links ggml as shared libs (libggml*.so). We install those into ./bin
# next to the binaries, so bake an $ORIGIN RPATH: the loader then finds them in
# the executable's own directory instead of the (deleted) temp build path.
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAKEET_GGML_CUDA=ON \
    -DPARAKEET_BUILD_CLI=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'

echo "🚀 Compiling..."
cmake --build build --config Release -j 16

# --- Install into ./bin ---
echo "📦 Copying binaries into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

cp build/examples/cli/parakeet-cli "$BIN_DIR/" || echo "⚠️ parakeet-cli not found!"
cp build/examples/server/parakeet-server "$BIN_DIR/" || echo "⚠️ parakeet-server not found!"

# Shared ggml libraries the binaries dlopen at runtime. Copy them (preserving
# the symlink chain libggml.so.0 -> libggml.so.0.X.Y) next to the binaries; the
# $ORIGIN RPATH baked in above makes the loader pick them up from here.
echo "📦 Copying shared ggml libraries..."
found_libs=$(find build -name 'libggml*.so*' -printf '%f\n' | sort -u)
if [ -z "$found_libs" ]; then
    echo "⚠️ No libggml*.so found — binary may be statically linked or build differs."
else
    # -a preserves symlinks so the SONAME (libggml.so.0) keeps pointing at the
    # versioned file. Copy from each dir that holds them.
    find build -name 'libggml*.so*' -exec cp -a -t "$BIN_DIR/" {} +
fi

# --- Normalize RPATH to $ORIGIN ---
# ggml's CMake bakes the (temp) build path into both the executables and the
# libggml*.so files as RUNPATH, and libggml.so itself depends on libggml-cuda.so
# which it resolves via *its own* RUNPATH — not the executable's. Once the temp
# build dir is deleted, every one of those paths is dangling, so we must rewrite
# RPATH = $ORIGIN on the binaries AND the libraries. The $ORIGIN at configure
# time only covers freshly-linked targets; patchelf guarantees it for all of
# them regardless of how ggml linked. Skipped (with a warning) if patchelf is
# absent — the configure-time RPATH still covers the common case.
if command -v patchelf >/dev/null 2>&1; then
    echo "🔧 Setting RPATH=\$ORIGIN on binaries and libraries..."
    for f in "$BIN_DIR"/parakeet-cli "$BIN_DIR"/parakeet-server "$BIN_DIR"/libggml*.so.*.*; do
        # Skip the symlinks (libggml.so.0); only patch real ELF files.
        [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null \
            && echo "   patched $(basename "$f")"
    done
else
    echo "⚠️ patchelf not found — install it (apt install patchelf) if parakeet-server"
    echo "   fails at runtime with 'libggml*.so: cannot open shared object file'."
fi

# --- Summary ---
echo "✅ Build complete! Binaries installed to:"
echo "   → ${BIN_DIR}"
ls -l "${BIN_DIR}/parakeet-"* 2>/dev/null || true
