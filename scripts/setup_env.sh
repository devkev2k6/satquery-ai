#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI - Environment Setup Script (Linux / macOS)
# Creates a Python virtual environment (.venv) and installs project dependencies
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "SatQuery AI - Virtual Environment Setup"
echo "Project root: ${PROJECT_ROOT}"
echo "=========================================="

# Find Python 3
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "[-] Error: Python 3 was not found on your PATH."
    echo "    Please install Python 3.9+ and ensure it is added to your PATH."
    exit 1
fi

PYTHON_VER=$(${PYTHON_BIN} --version 2>&1)
echo "[+] Detected Python: ${PYTHON_VER} (${PYTHON_BIN})"

# Virtual environment path
VENV_DIR="${PROJECT_ROOT}/.venv"

if [ -d "${VENV_DIR}" ]; then
    echo "[*] Virtual environment already exists at .venv"
else
    echo "[*] Creating virtual environment at .venv ..."
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
    echo "[+] Virtual environment created successfully."
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "[*] Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    echo "[*] Installing dependencies from requirements.txt..."
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    echo "[+] Dependencies installed successfully."
else
    echo "[-] Warning: requirements.txt not found at ${PROJECT_ROOT}/requirements.txt"
fi

echo "=========================================="
echo "Environment setup complete!"
echo "To activate this environment in your shell, run:"
echo "    source .venv/bin/activate"
echo "=========================================="
