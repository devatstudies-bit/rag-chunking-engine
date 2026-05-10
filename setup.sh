#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — bootstrap the chunking-engine development environment
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"
MIN_PYTHON_MINOR=11

# ── Color codes ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║     Chunking Engine — Environment Setup  ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo

# ── Check Python version ──────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ]; then
    error "Python 3.$MIN_PYTHON_MINOR+ required. Found: $PYTHON_VERSION"
fi
success "Python $PYTHON_VERSION detected"

# ── Create virtual environment ────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment '$VENV_DIR' already exists — skipping creation"
else
    info "Creating virtual environment in '$VENV_DIR'..."
    $PYTHON -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
info "Virtual environment activated"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip
success "pip upgraded"

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing production dependencies..."
pip install --quiet -r requirements.txt
success "Production dependencies installed"

# ── Install package in editable mode ─────────────────────────────────────────
info "Installing chunking-engine in editable mode..."
pip install --quiet -e .
success "Package installed"

# ── Copy .env if needed ───────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from .env.example — please fill in your credentials"
else
    info ".env already exists — skipping"
fi

echo
echo -e "${BOLD}${GREEN}Setup complete!${RESET}"
echo
echo -e "  Activate the environment:  ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  Start the API server:      ${CYAN}uvicorn api.main:app --reload${RESET}"
echo -e "  Run tests:                 ${CYAN}pytest${RESET}"
echo -e "  Run the demo:              ${CYAN}python examples/demo.py${RESET}"
echo -e "  Start Milvus (Docker):     ${CYAN}docker compose -f docker/docker-compose.yml up -d${RESET}"
echo
