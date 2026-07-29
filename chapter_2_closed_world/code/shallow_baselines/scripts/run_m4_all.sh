#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pattern in Yearly Quarterly Monthly Weekly Daily Hourly; do
  "$SCRIPT_DIR/run_m4.sh" --seasonal-pattern "$pattern" "$@"
done
