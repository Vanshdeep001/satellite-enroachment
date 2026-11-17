#!/bin/zsh
# activate.sh — Helper script to activate the project virtual environment
# with all required environment variables set.
#
# Usage:
#   source activate.sh
#
# DO NOT run as: ./activate.sh  (that would spawn a subshell and changes won't persist)

SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv311/bin/activate"

# Fix the matplotlib config directory warning (macOS SIP prevents ~/.matplotlib writes)
export MPLCONFIGDIR="$SCRIPT_DIR/.mpl_config"

echo "✅  Virtual environment activated: .venv311 (Python 3.11.14)"
echo "✅  MPLCONFIGDIR set to: $MPLCONFIGDIR"
echo ""
echo "To deactivate, run: deactivate"

