#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

cd "${ROOT_DIR}"
"${UV_BIN}" sync --locked
exec "${UV_BIN}" run --no-sync uvicorn llmevals.web.app:app --host "${HOST}" --port "${PORT}" "$@"
