#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_ett.sh" "$@"
"$SCRIPT_DIR/run_popular.sh" "$@"
