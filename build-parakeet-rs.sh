#!/usr/bin/env bash
# build-parakeet-rs.sh
# Build parakeet-rs-server (ASR + NVIDIA Nemotron-3 speaker diarization, ONNX
# Runtime) from rzafiamy/parakeet-rs and install it into ./bin/:
#
#   bin/parakeet-rs-server        the server
#   bin/parakeet-rs-lib/          its runtime (GPU builds only): ONNX Runtime,
#                                 the CUDA execution provider and cuDNN 9.
#                                 The server finds this directory by itself.
#
# Usage: ./build-parakeet-rs.sh [branch-or-tag] [--cpu]
#   --cpu            CPU-only build (static ONNX Runtime, no parakeet-rs-lib/).
# Environment:
#   PARAKEET_RS_REPO   git URL           (default: https://github.com/rzafiamy/parakeet-rs)
#   ORT_VERSION        ONNX Runtime GPU  (default: 1.28.2; must be >= 1.28)
#   CUDA_MAJOR         12 or 13          (default: detected from the driver)
#   CUDNN_DIR          directory holding libcudnn*.so.9 to bundle instead of
#                      downloading NVIDIA's cuDNN wheel from PyPI
#
# No root needed: nothing is installed system-wide. Building needs cargo
# (https://rustup.rs); cloning and building happen in a temp dir.

set -euo pipefail

REF="master"
CPU_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=1 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) REF="$arg" ;;
    esac
done
REPO="${PARAKEET_RS_REPO:-https://github.com/rzafiamy/parakeet-rs}"
ORT_VERSION="${ORT_VERSION:-1.28.2}"

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"
LIB_DIR="${BIN_DIR}/parakeet-rs-lib"

command -v cargo >/dev/null || { echo "❌ cargo not found — install Rust from https://rustup.rs" >&2; exit 1; }
command -v curl >/dev/null || { echo "❌ curl not found" >&2; exit 1; }

BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT

echo "🦜 Building parakeet-rs-server from ${REPO} @ ${REF}"
git clone --quiet "${REPO}" "${BUILD_ROOT}/parakeet-rs"
git -C "${BUILD_ROOT}/parakeet-rs" checkout --quiet "${REF}"
MANIFEST="${BUILD_ROOT}/parakeet-rs/server/Cargo.toml"

if [[ "${CPU_ONLY}" == 1 ]]; then
    echo "⚙️  CPU-only build"
    cargo build --release --manifest-path "${MANIFEST}"
    mkdir -p "${BIN_DIR}"
    cp "${BUILD_ROOT}/parakeet-rs/server/target/release/parakeet-rs-server" "${BIN_DIR}/"
    rm -rf "${LIB_DIR}"
    echo "✅ Installed ${BIN_DIR}/parakeet-rs-server (CPU)"
    exit 0
fi

# --- CUDA major version -------------------------------------------------------
# Taken from the driver, not the toolkit: the driver decides which CUDA runtime
# can run (driver 570 tops out at CUDA 12.8; CUDA 13 needs driver >= 580).
if [[ -z "${CUDA_MAJOR:-}" ]]; then
    drv_cuda=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' || true)
    [[ -n "${drv_cuda}" ]] || { echo "❌ nvidia-smi not found; set CUDA_MAJOR or use --cpu" >&2; exit 1; }
    CUDA_MAJOR=$(( drv_cuda >= 13 ? 13 : 12 ))
fi
echo "🔧 CUDA ${CUDA_MAJOR} runtime, ONNX Runtime ${ORT_VERSION}"

echo "⚙️  Compiling (cuda + load-dynamic)..."
cargo build --release --manifest-path "${MANIFEST}" --no-default-features --features cuda,load-dynamic

# --- ONNX Runtime GPU (official Microsoft release) ------------------------------
ORT_NAME="onnxruntime-linux-x64-gpu_cuda${CUDA_MAJOR}-${ORT_VERSION}"
echo "📦 Fetching ${ORT_NAME}..."
curl -fsSL --retry 3 -o "${BUILD_ROOT}/ort.tgz" \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_NAME}.tgz"
tar -xzf "${BUILD_ROOT}/ort.tgz" -C "${BUILD_ROOT}"

STAGE="${BUILD_ROOT}/lib"
mkdir -p "${STAGE}"
cp -a "${BUILD_ROOT}/${ORT_NAME}/lib/"libonnxruntime.so* "${STAGE}/"
cp -a "${BUILD_ROOT}/${ORT_NAME}/lib/"libonnxruntime_providers_{shared,cuda}.so "${STAGE}/"

# --- cuDNN 9 ---------------------------------------------------------------------
# The CUDA provider dlopen()s cuDNN by soname; bundle it unless the system has it.
if [[ -n "${CUDNN_DIR:-}" ]]; then
    echo "📦 Bundling cuDNN from ${CUDNN_DIR}"
    cp -L "${CUDNN_DIR}"/libcudnn*.so.9 "${STAGE}/"
elif ldconfig -p 2>/dev/null | grep -q 'libcudnn.so.9 '; then
    echo "✓ cuDNN 9 found system-wide; not bundling it"
else
    pkg="nvidia-cudnn-cu${CUDA_MAJOR}"
    echo "📦 Fetching ${pkg} wheel from PyPI (large, one-off)..."
    url=$(python3 - "$pkg" <<'PY'
import json, sys, urllib.request
pkg = sys.argv[1]
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json"))
# Final 9.x releases only (skip .dev/rc builds), newest first, first with a Linux x86-64 wheel.
releases = [v for v in meta["releases"] if v.startswith("9.") and all(p.isdigit() for p in v.split("."))]
releases.sort(key=lambda v: [int(p) for p in v.split(".")], reverse=True)
for v in releases:
    wheels = [f["url"] for f in meta["releases"][v]
              if f["filename"].endswith(".whl") and "manylinux" in f["filename"] and "x86_64" in f["filename"]]
    if wheels:
        print(wheels[0]); break
PY
)
    [[ -n "${url}" ]] || { echo "❌ no ${pkg} wheel found; set CUDNN_DIR" >&2; exit 1; }
    curl -fsSL --retry 3 -o "${BUILD_ROOT}/cudnn.whl" "${url}"
    python3 - "${BUILD_ROOT}/cudnn.whl" "${STAGE}" <<'PY'
import sys, zipfile, os
whl, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if "/lib/libcudnn" in n and ".so" in n:
            with open(os.path.join(out, os.path.basename(n)), "wb") as f:
                f.write(z.read(n))
PY
fi

# The CUDA runtime/cuBLAS come from the CUDA toolkit or driver install.
if ! ldconfig -p 2>/dev/null | grep -q "libcublasLt.so.${CUDA_MAJOR}"; then
    echo "⚠️  libcublasLt.so.${CUDA_MAJOR} is not on the loader path; install the CUDA ${CUDA_MAJOR}"
    echo "   runtime libraries (cuBLAS, cudart, cuRAND, cuFFT): without them CUDA is unavailable"
    echo "   (--device cuda fails at start-up; --device auto runs on CPU)."
fi

# --- Install -------------------------------------------------------------------------
mkdir -p "${BIN_DIR}"
rm -rf "${LIB_DIR}"
mv "${STAGE}" "${LIB_DIR}"
cp "${BUILD_ROOT}/parakeet-rs/server/target/release/parakeet-rs-server" "${BIN_DIR}/"

echo "✅ Build complete:"
echo "   → ${BIN_DIR}/parakeet-rs-server"
echo "   → ${LIB_DIR}/ ($(du -sh "${LIB_DIR}" | cut -f1))"
"${BIN_DIR}/parakeet-rs-server" --version
