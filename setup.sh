#!/usr/bin/env bash
# Run once to create the virtual environment and install dependencies.
set -e

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. Start the server with:"
echo "  .venv/bin/python app.py"
