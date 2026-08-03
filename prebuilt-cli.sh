#!/usr/bin/env bash
# Interactive prebuilt package installer for Zallama.

set -euo pipefail

DEFAULT_DRIVE_URL="https://drive.google.com/drive/folders/1B7AmE36r869kpMZbOatqMW-Dhedq2Sil?usp=sharing"
DEFAULT_BIN_DIR="./bin"

usage() {
    cat <<EOF
Usage:
  ./prebuilt-cli.sh [options]

Options:
  --source <dir-or-url>  Package directory or public Google Drive folder URL.
                         Default: ${DEFAULT_DRIVE_URL}
  --bin-dir <dir>        Directory where selected package is extracted.
                         Default: ${DEFAULT_BIN_DIR}
  --engine <name>        Filter by engine: llama.cpp, parakeet.cpp, kokoro.cpp.
  --list-only            Show compatible packages without installing.
  -h, --help             Show this help.

The script detects your system, lists compatible packages, asks which archive to
download, then extracts the selected package.
EOF
}

SOURCE="${ZALLAMA_PREBUILT_SOURCE:-$DEFAULT_DRIVE_URL}"
BIN_DIR="${ZALLAMA_PREBUILT_BIN_DIR:-$DEFAULT_BIN_DIR}"
ENGINE_FILTER=""
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="${2:-}"
            shift 2
            ;;
        --bin-dir)
            BIN_DIR="${2:-}"
            shift 2
            ;;
        --engine)
            ENGINE_FILTER="${2:-}"
            shift 2
            ;;
        --list-only)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$SOURCE" ]]; then
    echo "ERROR: --source cannot be empty." >&2
    exit 1
fi

is_url() {
    case "$1" in
        http://*|https://*) return 0 ;;
        *) return 1 ;;
    esac
}

sort_packages() {
    if sort -V </dev/null >/dev/null 2>&1; then
        sort -V
    else
        sort
    fi
}

drive_folder_id() {
    local url="$1"
    local id
    id="$(printf '%s\n' "$url" | sed -n 's#.*drive/folders/\([^/?#]*\).*#\1#p')"
    if [[ -z "$id" ]]; then
        echo "ERROR: Could not parse Google Drive folder id from URL: $url" >&2
        exit 1
    fi
    printf '%s\n' "$id"
}

list_drive_folder() {
    local url="$1"
    local folder_id
    local list_url

    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 is required to list public Google Drive folders." >&2
        exit 1
    fi

    folder_id="$(drive_folder_id "$url")"
    list_url="https://drive.google.com/embeddedfolderview?id=${folder_id}#list"

    python3 - "$list_url" <<'PY'
import html
import re
import sys
import urllib.request

url = sys.argv[1]
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

pattern = re.compile(
    r'href="https://drive\.google\.com/file/d/([^/]+)/[^"]*"[^>]*>(.*?)</a>',
    re.S,
)
seen = set()
for file_id, label in pattern.findall(body):
    name = re.sub(r"<[^>]+>", "", label)
    name = html.unescape(name).strip()
    if not name.endswith(".tar.gz"):
        continue
    key = (file_id, name)
    if key in seen:
        continue
    seen.add(key)
    print(f"{file_id}\t{name}")
PY
}

download_drive_file() {
    local file_id="$1"
    local filename="$2"
    local cache_base
    local cache_dir
    local output

    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 is required to download Google Drive files." >&2
        exit 1
    fi

    cache_base="${ZALLAMA_PREBUILT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/zallama/prebuilt}"
    cache_dir="${cache_base}/downloads"
    mkdir -p "$cache_dir"
    output="${cache_dir}/${filename}"

    if [[ ! -f "$output" ]]; then
        echo "Downloading $filename..."
        python3 - "$file_id" "$output" <<'PY'
import html
import http.cookiejar
import re
import sys
import urllib.parse
import urllib.request

file_id, output = sys.argv[1], sys.argv[2]
cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

def open_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return opener.open(req, timeout=60)

def save_response(resp):
    total = resp.headers.get("Content-Length")
    done = 0
    with open(output, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                percent = min(100, int(done * 100 / int(total)))
                print(f"\r{percent:3d}% {done / (1024 * 1024):.1f} MiB", end="", file=sys.stderr)
            else:
                print(f"\r{done / (1024 * 1024):.1f} MiB", end="", file=sys.stderr)
    print(file=sys.stderr)

url = "https://drive.google.com/uc?" + urllib.parse.urlencode({"export": "download", "id": file_id})
resp = open_url(url)
ctype = resp.headers.get("Content-Type", "")

if "text/html" not in ctype:
    save_response(resp)
    raise SystemExit(0)

body = resp.read().decode("utf-8", "replace")
confirm = None
match = re.search(r"confirm=([0-9A-Za-z_]+)", body)
if match:
    confirm = html.unescape(match.group(1))

download_url = None
match = re.search(r'href="(/uc\?export=download[^"]+)"', body)
if match:
    download_url = "https://drive.google.com" + html.unescape(match.group(1)).replace("&amp;", "&")

if confirm:
    url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "confirm": confirm, "id": file_id}
    )
elif download_url:
    url = download_url
else:
    print("ERROR: Google Drive did not provide a downloadable response.", file=sys.stderr)
    print("The file may not be public, or Google changed the confirmation page.", file=sys.stderr)
    raise SystemExit(1)

save_response(open_url(url))
PY
    else
        echo "Using cached download: $output"
    fi
    printf '%s\n' "$output"
}

detect_system() {
    OS_NAME="${ZALLAMA_PREBUILT_OS:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
    if [[ "$OS_NAME" != "linux" ]]; then
        echo "ERROR: Prebuilt packages currently target Linux; detected: $OS_NAME" >&2
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
        24.04) UBUNTU_CANDIDATES=("24.04" "22.04") ;;
        22.04) UBUNTU_CANDIDATES=("22.04") ;;
        *)
            echo "ERROR: Unsupported Ubuntu version: ${UBUNTU_VERSION:-unknown}" >&2
            echo "Supported package baselines: ubuntu22.04, ubuntu24.04" >&2
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
}

parse_package() {
    local file="$1"
    local base
    base="$(basename "$file")"
    [[ "$base" =~ ^(.+)-([^-]+)-linux-ubuntu([0-9]+\.[0-9]+)-(cpu|cuda)\.tar\.gz$ ]] || return 1
    PKG_ENGINE="${BASH_REMATCH[1]}"
    PKG_REVISION="${BASH_REMATCH[2]}"
    PKG_UBUNTU="${BASH_REMATCH[3]}"
    PKG_BACKEND="${BASH_REMATCH[4]}"
}

is_in() {
    local needle="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$needle" == "$value" ]] && return 0
    done
    return 1
}

compatibility_rank() {
    local ubuntu="$1"
    local backend="$2"
    local version_index
    local backend_index

    for version_index in "${!UBUNTU_CANDIDATES[@]}"; do
        if [[ "$ubuntu" == "${UBUNTU_CANDIDATES[$version_index]}" ]]; then
            break
        fi
    done
    for backend_index in "${!BACKEND_CANDIDATES[@]}"; do
        if [[ "$backend" == "${BACKEND_CANDIDATES[$backend_index]}" ]]; then
            break
        fi
    done
    echo $(( version_index * 10 + backend_index ))
}

collect_compatible_packages() {
    local file
    local rank

    COMPATIBLE=()
    shopt -s nullglob
    for file in "$PACKAGE_DIR"/*.tar.gz; do
        if ! parse_package "$file"; then
            continue
        fi
        if [[ -n "$ENGINE_FILTER" && "$PKG_ENGINE" != "$ENGINE_FILTER" ]]; then
            continue
        fi
        if ! is_in "$PKG_UBUNTU" "${UBUNTU_CANDIDATES[@]}"; then
            continue
        fi
        if ! is_in "$PKG_BACKEND" "${BACKEND_CANDIDATES[@]}"; then
            continue
        fi
        rank="$(compatibility_rank "$PKG_UBUNTU" "$PKG_BACKEND")"
        COMPATIBLE+=("${rank}|${PKG_ENGINE}|${PKG_REVISION}|ubuntu${PKG_UBUNTU}|${PKG_BACKEND}|${file}")
    done
}

collect_compatible_drive_packages() {
    local line
    local file_id
    local name
    local rank

    COMPATIBLE=()
    while IFS=$'\t' read -r file_id name; do
        [[ -z "${file_id:-}" || -z "${name:-}" ]] && continue
        if ! parse_package "$name"; then
            continue
        fi
        if [[ -n "$ENGINE_FILTER" && "$PKG_ENGINE" != "$ENGINE_FILTER" ]]; then
            continue
        fi
        if ! is_in "$PKG_UBUNTU" "${UBUNTU_CANDIDATES[@]}"; then
            continue
        fi
        if ! is_in "$PKG_BACKEND" "${BACKEND_CANDIDATES[@]}"; then
            continue
        fi
        rank="$(compatibility_rank "$PKG_UBUNTU" "$PKG_BACKEND")"
        COMPATIBLE+=("${rank}|${PKG_ENGINE}|${PKG_REVISION}|ubuntu${PKG_UBUNTU}|${PKG_BACKEND}|drive:${file_id}:${name}")
    done < <(list_drive_folder "$SOURCE")
}

print_table() {
    local index=1
    local row
    printf '\nDetected system: linux ubuntu%s; backend candidates: %s\n' "$UBUNTU_VERSION" "${BACKEND_CANDIDATES[*]}"
    printf 'Package source: %s\n\n' "$PACKAGE_DIR"
    printf '%-4s %-14s %-12s %-12s %-8s %s\n' "No." "Engine" "Revision" "Ubuntu" "Backend" "Archive"
    printf '%-4s %-14s %-12s %-12s %-8s %s\n' "---" "------" "--------" "------" "-------" "-------"
    for row in "${COMPATIBLE[@]}"; do
        IFS='|' read -r _rank engine revision ubuntu backend file <<<"$row"
        printf '%-4s %-14s %-12s %-12s %-8s %s\n' "$index" "$engine" "$revision" "$ubuntu" "$backend" "$(display_name "$file")"
        index=$(( index + 1 ))
    done
    printf '\n'
}

install_selected() {
    local selected="$1"
    local file
    local file_id
    local filename
    local archive

    IFS='|' read -r _rank _engine _revision _ubuntu _backend file <<<"$selected"
    if [[ "$file" == drive:* ]]; then
        file_id="${file#drive:}"
        filename="${file_id#*:}"
        file_id="${file_id%%:*}"
        archive="$(download_drive_file "$file_id" "$filename")"
    else
        archive="$file"
    fi

    mkdir -p "$BIN_DIR"
    echo "Extracting $(basename "$archive") into $BIN_DIR..."
    tar -xzf "$archive" -C "$BIN_DIR"
    echo "Done."
}

display_name() {
    local value="$1"
    if [[ "$value" == drive:* ]]; then
        value="${value#drive:}"
        printf '%s\n' "${value#*:}"
    else
        basename "$value"
    fi
}

detect_system

if is_url "$SOURCE"; then
    PACKAGE_DIR="$SOURCE"
    collect_compatible_drive_packages
else
    PACKAGE_DIR="$SOURCE"
    if [[ ! -d "$PACKAGE_DIR" ]]; then
        echo "ERROR: Package directory not found: $PACKAGE_DIR" >&2
        exit 1
    fi
    collect_compatible_packages
fi
if (( ${#COMPATIBLE[@]} == 0 )); then
    echo "ERROR: No compatible prebuilt packages found." >&2
    echo "Detected: linux ubuntu${UBUNTU_VERSION}; backend candidates: ${BACKEND_CANDIDATES[*]}" >&2
    echo "Source: $PACKAGE_DIR" >&2
    exit 1
fi

IFS=$'\n' COMPATIBLE=($(printf '%s\n' "${COMPATIBLE[@]}" | sort_packages))
unset IFS

print_table

if (( LIST_ONLY == 1 )); then
    exit 0
fi

while true; do
    read -r -p "Choose package number to install [1-${#COMPATIBLE[@]}] or q to quit: " choice
    case "$choice" in
        q|Q)
            echo "Cancelled."
            exit 0
            ;;
        ''|*[!0-9]*)
            echo "Please enter a number from 1 to ${#COMPATIBLE[@]}, or q."
            ;;
        *)
            if (( choice >= 1 && choice <= ${#COMPATIBLE[@]} )); then
                install_selected "${COMPATIBLE[$(( choice - 1 ))]}"
                exit 0
            fi
            echo "Please enter a number from 1 to ${#COMPATIBLE[@]}, or q."
            ;;
    esac
done
