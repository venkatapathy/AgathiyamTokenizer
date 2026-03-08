#!/bin/bash
# Download Samanantar Tamil and train BPE + GPE (Agathiyam). Uses Python 3.10 venv with lzma/datasets.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Prefer Agathiyam/.venv310 (sibling folder, Python 3.10 + lzma/datasets)
VENV="$(cd "$ROOT/../Agathiyam" 2>/dev/null && pwd)/.venv310"
if [[ ! -d "$VENV" ]]; then
  VENV="$ROOT/.venv310"
fi
if [[ ! -d "$VENV" ]]; then
  echo "Create a Python 3.10 venv with: python3.10 -m venv .venv310 && . .venv310/bin/activate && pip install grapheme datasets"
  exit 1
fi

export SAMANANTAR_TAMIL_MAX_LINES="${SAMANANTAR_TAMIL_MAX_LINES:-50000}"
export MERGE_BUDGET="${MERGE_BUDGET:-2000}"

echo "Samanantar lines: $SAMANANTAR_TAMIL_MAX_LINES, merge budget: $MERGE_BUDGET"
"$VENV/bin/python" -u "$ROOT/train_models.py"
