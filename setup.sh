#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <uv|pip>"
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

if [ "$1" != "uv" ] && [ "$1" != "pip" ]; then
    echo "Error: unsupported argument '$1'"
    usage
    exit 1
fi

echo "=== Starting setup with setting $1 ==="

# Install the package in editable mode
echo "=== Installing package in editable mode ==="
if [ "$1" == "pip" ]; then
    pip install -e .
    PYTHON_CMD="python"
else
    uv sync
    PYTHON_CMD="uv run python"
fi
echo "=== Package installation completed ==="

# Remove ete3 cgi dependency - not needed anymore and require removal as of python 3.13
# Grep import cgi from .venv/lib/python3.13/site-packages/ete3/webplugin/webapp.py and replace the line with blank
echo "=== Removing ete3 cgi dependency ==="
ETE3_WEBAPP_PATH=$($PYTHON_CMD -c "import ete3.webplugin.webapp as webapp; print(webapp.__file__)")
sed -i '/import cgi/d' "$ETE3_WEBAPP_PATH"
echo "=== ete3 cgi dependency removal completed ==="

# Test import works
echo "=== Testing ete3 import ==="
$PYTHON_CMD -c "import ete3"
echo "=== ete3 import test completed ==="