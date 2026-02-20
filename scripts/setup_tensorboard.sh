#!/usr/bin/env bash
# setup_tensorboard.sh — Start TensorBoard pointing at the local runs/ directory.
#
# Usage:
#   chmod +x scripts/setup_tensorboard.sh
#   ./scripts/setup_tensorboard.sh
#   ./scripts/setup_tensorboard.sh --port 6007   # custom port

set -euo pipefail

LOGDIR="${LOGDIR:-./runs}"
PORT="${1:-6006}"

# Allow --port=XXXX syntax
for arg in "$@"; do
    case $arg in
        --port=*) PORT="${arg#*=}" ;;
        --port)   shift; PORT="$1" ;;
    esac
done

echo "Starting TensorBoard..."
echo "  Log directory : $LOGDIR"
echo "  URL           : http://localhost:$PORT"
echo ""
echo "  Open your browser at http://localhost:$PORT"
echo "  Press Ctrl-C to stop."
echo ""

tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
