#!/usr/bin/env bash
# Serve the evaluator UI (Laptop 2). Usage: scripts/run_ui.sh [port]
# Default port is 5500. The API address is set separately in ui/app.js (API_BASE).
PORT="${1:-5500}"
cd "$(dirname "$0")/../ui" || exit 1
echo "PathSnap UI on http://localhost:${PORT}  (Ctrl+C to stop)"
python3 -m http.server "$PORT"
