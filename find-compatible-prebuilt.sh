#!/usr/bin/env bash
# Find the most compatible prebuilt Zallama engine archive for this system.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./find-compatible-prebuilt.sh <package-dir> [engine]

Arguments:
  package-dir  Directory containing prebuilt *.tar.gz archives.
  engine       Optional engine filter, e.g. llama.cpp, parakeet.cpp, kokoro.cpp.

The script prints the best matching archive path and exits with 0.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

PACKAGE_DIR="${1:-.}"
ENGINE_FILTER="${2:-}"

if [[ ! -d "$PACKAGE_DIR" ]]; then
    echo "ERROR: Package directory not found: $PACKAGE_DIR" >&2
    exit 1
fi

OS_NAME="${ZALLAMA_PREBUILT_OS:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
if [[ "$OS_NAME" != "linux" ]]; then
    echo "ERROR: No prebuilt package rule for OS: $OS_NAME" >&2
    exit 1
fi

if [[ -n "${ZALLAMA_PREBUILT_UBUNTU_VERSION:-}" ]]; then
    ID="ubuntu"
    VERSION_ID="$ZALLAMA_PREBUILT_UBUNTU_VERSION"
elif [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
else
    echo "ERROR: Cannot detect Linux distribution: /etc/os-release is missing." >&2
    exit 1
fi

if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "ERROR: Prebuilt packages currently target Ubuntu; detected: ${ID:-unknown}" >&2
    exit 1
fi

UBUNTU_VERSION="${ZALLAMA_PREBUILT_UBUNTU_VERSION:-${VERSION_ID:-}}"
case "$UBUNTU_VERSION" in
    24.04)
        UBUNTU_CANDIDATES=("24.04" "22.04")
        ;;
    22.04)
        UBUNTU_CANDIDATES=("22.04")
        ;;
    *)
        echo "ERROR: Unsupported Ubuntu version: ${UBUNTU_VERSION:-unknown}" >&2
        echo "   Supported package baselines: ubuntu22.04, ubuntu24.04" >&2
        exit 1
        ;;
esac

if [[ -n "${ZALLAMA_PREBUILT_BACKEND:-}" ]]; then
    BACKEND_CANDIDATES=("$ZALLAMA_PREBUILT_BACKEND")
elif command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1 || [[ -d /usr/local/cuda ]]; then
    BACKEND_CANDIDATES=("cuda" "cpu")
else
    BACKEND_CANDIDATES=("cpu")
fi

shopt -s nullglob

matches_for() {
    local ubuntu_version="$1"
    local backend="$2"
    local pattern
    local files

    if [[ -n "$ENGINE_FILTER" ]]; then
        pattern="${ENGINE_FILTER}-*-linux-ubuntu${ubuntu_version}-${backend}.tar.gz"
    else
        pattern="*-linux-ubuntu${ubuntu_version}-${backend}.tar.gz"
    fi

    files=("$PACKAGE_DIR"/$pattern)
    if (( ${#files[@]} > 0 )); then
        printf '%s\n' "${files[@]}"
    fi
}

sort_packages() {
    if sort -V </dev/null >/dev/null 2>&1; then
        sort -V
    else
        sort
    fi
}

for ubuntu_version in "${UBUNTU_CANDIDATES[@]}"; do
    for backend in "${BACKEND_CANDIDATES[@]}"; do
        MATCHES=()
        while IFS= read -r match; do
            MATCHES+=("$match")
        done < <(matches_for "$ubuntu_version" "$backend" | sort_packages)
        if (( ${#MATCHES[@]} > 0 )); then
            printf '%s\n' "${MATCHES[$(( ${#MATCHES[@]} - 1 ))]}"
            exit 0
        fi
    done
done

echo "ERROR: No compatible prebuilt package found in: $PACKAGE_DIR" >&2
echo "   Detected: linux ubuntu${UBUNTU_VERSION}; backend candidates: ${BACKEND_CANDIDATES[*]}" >&2
if [[ -n "$ENGINE_FILTER" ]]; then
    echo "   Engine filter: $ENGINE_FILTER" >&2
fi
exit 1
