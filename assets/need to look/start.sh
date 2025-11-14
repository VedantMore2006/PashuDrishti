#!/usr/bin/env bash
set -euo pipefail

# Use provided PORT env or default
PORT="${PORT:-10000}"

# Optional GPU awareness (for local testing)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES is set. Using GPU if available."
fi

# Launch uvicorn
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info
