#!/usr/bin/env bash
# install.sh — Zallama one-shot installer
set -euo pipefail

ZALLAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[92m'; CYAN='\033[96m'; DIM='\033[2m'; RESET='\033[0m'; BOLD='\033[1m'

echo -e "${CYAN}${BOLD}"
echo "  ███████╗ █████╗ ██╗     ██╗      █████╗ ███╗   ███╗  █████╗"
echo "  ╚════██║██╔══██╗██║     ██║     ██╔══██╗████╗ ████║ ██╔══██╗"
echo "      ██╔╝███████║██║     ██║     ███████║██╔████╔██║ ███████║"
echo "     ██╔╝ ██╔══██║██║     ██║     ██╔══██║██║╚██╔╝██║ ██╔══██║"
echo "     ██║  ██║  ██║███████╗███████╗██║  ██║██║ ╚═╝ ██║ ██║  ██║"
echo "     ╚═╝  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═╝  ╚═╝"
echo -e "${RESET}${DIM}  Local LLM server powered by llama.cpp${RESET}"
echo ""

# ── 1. Python check ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 is required. Install it with: sudo apt install python3 python3-pip"
  exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓${RESET} Python $PYTHON_VERSION found"

# ── 2. Install Python dependencies (into a project-local venv) ────────────────
# Modern Debian/Ubuntu (PEP 668) refuse system-wide `pip install`. A venv is the
# supported path and keeps the install self-contained. The `zallama` launcher
# re-execs into this venv automatically, so no activation is ever needed.
VENV_DIR="$ZALLAMA_DIR/.venv"
echo "📦 Setting up virtual environment at .venv ..."

if [ ! -x "$VENV_DIR/bin/python" ]; then
  if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
    echo "❌ Failed to create venv. Install the venv module:"
    echo "   sudo apt install python3-venv python3-full"
    exit 1
  fi
fi

echo "📦 Installing Python dependencies..."
"$VENV_DIR/bin/python" -m pip install -q --upgrade pip
"$VENV_DIR/bin/python" -m pip install -q -r "$ZALLAMA_DIR/requirements.txt"
echo -e "${GREEN}✓${RESET} Dependencies installed into .venv"

# If the installer is run via sudo, the venv ends up owned by root; hand it back
# to the invoking user so `zallama serve` (run as that user) can use it.
if [ -n "${SUDO_USER:-}" ]; then
  chown -R "$SUDO_USER" "$VENV_DIR" 2>/dev/null || true
fi

# ── 3. Check llama-server binary ────────────────────────────────────────────
if [ -f "$ZALLAMA_DIR/bin/llama-server" ]; then
  echo -e "${GREEN}✓${RESET} llama-server binary found at ./bin/llama-server"
elif command -v llama-server &>/dev/null; then
  echo -e "${GREEN}✓${RESET} llama-server found in PATH"
else
  echo -e "⚠  llama-server not found."
  echo "   Build it with: bash build-ggml-llama.cpp.sh"
  echo "   Or install llama.cpp from: https://github.com/ggml-org/llama.cpp"
fi

# ── 4. Make CLI executable ───────────────────────────────────────────────────
chmod +x "$ZALLAMA_DIR/zallama"
echo -e "${GREEN}✓${RESET} zallama CLI is executable"

# ── 5. Create ~/.zallama dirs ────────────────────────────────────────────────
mkdir -p ~/.zallama/{models,logs,bin}
echo -e "${GREEN}✓${RESET} ~/.zallama directory structure created"

# ── 6. Symlink to /usr/local/bin (optional) ──────────────────────────────────
SYMLINK_TARGET="/usr/local/bin/zallama"
if [ -w "/usr/local/bin" ] || sudo -n true 2>/dev/null; then
  sudo ln -sf "$ZALLAMA_DIR/zallama" "$SYMLINK_TARGET" 2>/dev/null || \
    ln -sf "$ZALLAMA_DIR/zallama" "$HOME/.local/bin/zallama" 2>/dev/null || true
  echo -e "${GREEN}✓${RESET} zallama command installed to $SYMLINK_TARGET"
else
  echo -e "ℹ  To run zallama from anywhere, add to your PATH:"
  echo -e "   ${DIM}export PATH=\"$ZALLAMA_DIR:\$PATH\"${RESET}"
fi

# ── 7. Systemd service (optional) ────────────────────────────────────────────
SERVICE_FILE="/etc/systemd/system/zallama.service"
if command -v systemctl &>/dev/null && [ "$EUID" -eq 0 ]; then
  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Zallama — Local LLM Server
After=network.target

[Service]
Type=simple
User=$SUDO_USER
WorkingDirectory=$ZALLAMA_DIR
ExecStart=$ZALLAMA_DIR/zallama serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  echo -e "${GREEN}✓${RESET} systemd service installed: systemctl enable --now zallama"

  # If the service is already running (re-install), the new unit/code won't take
  # effect until a restart — do it so re-running the installer behaves as users
  # expect rather than silently leaving the old process in place.
  if systemctl is-active --quiet zallama; then
    systemctl restart zallama
    echo -e "${GREEN}✓${RESET} Restarted running zallama service"
  fi
fi

echo ""
echo -e "${CYAN}${BOLD}🎉 Zallama installed successfully!${RESET}"
echo ""
echo -e "  Start the server:    ${BOLD}zallama serve${RESET}"
echo -e "  Add a model:         ${BOLD}zallama add mymodel /path/to/model.gguf${RESET}"
echo -e "  Chat interactively:  ${BOLD}zallama run mymodel${RESET}"
echo -e "  Web UI:              ${BOLD}http://localhost:11435${RESET}"
echo -e "  API docs:            ${BOLD}http://localhost:11435/docs${RESET}"
echo ""
