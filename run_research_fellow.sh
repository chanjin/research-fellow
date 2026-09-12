#!/bin/sh
set -eu

PORT="${1:-8501}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"

exec streamlit run app.py --server.port "$PORT"
