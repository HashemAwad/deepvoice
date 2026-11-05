#!/usr/bin/env bash
# One-command setup for macOS and Linux.
#   ./scripts/setup.sh         CPU stack (features, sklearn, metrics)
#   ./scripts/setup.sh --dl    also the deep-learning stack (torch, transformers, peft, optuna)
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=()
if [ "${1:-}" = "--dl" ]; then EXTRA=(--extra dl); fi
PY="$(tr -d '[:space:]' < .python-version)"

if ! command -v git >/dev/null 2>&1; then
  echo "git is missing. macOS: run 'xcode-select --install'. Linux: install git with your package manager." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (https://docs.astral.sh/uv/)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv python install "$PY"
uv sync --python "$PY" ${EXTRA[@]+"${EXTRA[@]}"}

# strip notebook outputs on commit (per-clone git setting)
uv run nbstripout --install

uv run pytest -q
echo
echo "Setup done. Run everything through 'uv run', e.g. 'uv run pytest'."
echo "If 'uv' is not found in a new terminal, open a new terminal or add ~/.local/bin to PATH."
