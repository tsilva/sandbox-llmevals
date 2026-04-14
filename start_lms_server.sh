#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"

cd "${ROOT_DIR}"
"${UV_BIN}" sync --locked
exec "${UV_BIN}" run --no-sync python -m llmevals.cli start-server "$@"

