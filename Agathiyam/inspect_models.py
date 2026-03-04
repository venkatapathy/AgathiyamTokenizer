#!/usr/bin/env python3
"""
Quick inspection of trained tokenizer models in models/.

Run from Agathiyam/:

    python inspect_models.py

Outputs, for each available model:
- vocab size (number of distinct tokens)
- number of merges
- a few sample tokens and merges
"""

import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def inspect_one(name: str, vocab_name: str, merges_name: str) -> None:
    vocab_path = MODELS_DIR / vocab_name
    merges_path = MODELS_DIR / merges_name

    if not vocab_path.exists() or not merges_path.exists():
        print(f"{name}: missing files ({vocab_path.name}, {merges_path.name})")
        return

    vocab = load_pickle(vocab_path)
    merges = load_pickle(merges_path)

    print(f"{name}")
    print(f"  vocab file : {vocab_path.name}")
    print(f"  merges file: {merges_path.name}")
    print(f"  vocab size : {len(vocab):,} tokens")
    print(f"  merges     : {len(merges):,} merge ops")

    # Show a few sample tokens and merges for sanity
    sample_tokens = list(vocab.keys())[:10]
    print(f"  sample tokens: {sample_tokens}")
    sample_merges = merges[:10]
    print(f"  first merges : {sample_merges}")
    print()


def main() -> None:
    print(f"Inspecting models in {MODELS_DIR}")
    print()

    inspect_one("BPE", "vocab_bpe.pkl", "merges_bpe.pkl")
    inspect_one("GPE", "vocab_gpe.pkl", "merges_gpe.pkl")
    # Agathiyam uses the sandhi-aware GPE model
    inspect_one("Agathiyam", "vocab_re.pkl", "merges.pkl")


if __name__ == "__main__":
    main()

