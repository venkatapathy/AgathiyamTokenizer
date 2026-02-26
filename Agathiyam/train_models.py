#!/usr/bin/env python3
"""
Train BPE, GPE, and Agathiyam (GPE) tokenizers and save to models/ for the webapp.
Run from Agathiyam directory: python train_models.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Merge budget; will be capped by corpus size (override with env MERGE_BUDGET)
MERGE_BUDGET = int(os.getenv("MERGE_BUDGET", "5000"))

# Max lines to use from Samanantar Tamil (override with env SAMANANTAR_TAMIL_MAX_LINES)
# Default 20k for faster training; use 100000 for full run (expect several hours for BPE+GPE+Agathiyam)
SAMANANTAR_TAMIL_MAX_LINES = int(os.getenv("SAMANANTAR_TAMIL_MAX_LINES", "100000"))


def _is_lfs_pointer(path: Path) -> bool:
    """True if file is a Git LFS pointer."""
    if not path.exists() or path.stat().st_size > 300:
        return False
    try:
        text = path.read_text(errors="ignore")
        return "git-lfs.github.com" in text and "oid sha256:" in text
    except Exception:
        return False


def load_samanantar_tamil(max_lines: int = SAMANANTAR_TAMIL_MAX_LINES) -> list[str]:
    """Load Tamil sentences from local file or Hugging Face ai4bharat/samanantar."""
    # 1) Local Samanantar Tamil file
    for name in ["samanantar_tamil.txt", "samanantar_ta.txt"]:
        p = DATA_DIR / name
        if p.exists() and not _is_lfs_pointer(p):
            raw = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            lines = [s.strip() for s in raw if s.strip()][:max_lines]
            if lines:
                print(f"  Samanantar Tamil (local {name}): {len(lines)} lines", flush=True)
                return lines
    # 2) Hugging Face ai4bharat/samanantar (streaming; src=English, tgt=Tamil)
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/samanantar", "ta", split="train", streaming=True)
        lines = []
        for row in ds:
            if len(lines) >= max_lines:
                break
            tgt = (row.get("tgt") or row.get("target") or "").strip()
            if tgt:
                lines.append(tgt)
                if len(lines) in (1, 1000, 5000, 10000, 50000, 100000):
                    print(f"  ... streamed {len(lines)} tamil lines", flush=True)
        if lines:
            print(f"  Samanantar Tamil (Hugging Face): {len(lines)} lines", flush=True)
            return lines
    except Exception as e:
        print(f"  Samanantar Tamil (HF skipped): {e}", flush=True)
    return []


def load_corpus():
    """Load training lines: trainn.txt, data/*.txt, and Samanantar Tamil."""
    lines = []

    # Primary: trainn.txt or train.txt
    for name in ["trainn.txt", "train.txt"]:
        p = ROOT / name
        if p.exists() and not _is_lfs_pointer(p):
            raw = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            lines = [s.strip() for s in raw if s.strip()]
            if lines:
                print(f"Using {name}: {len(lines)} lines", flush=True)
                break
    else:
        lines = []

    # Add Samanantar Tamil (local or HF)
    print("Samanantar Tamil:", flush=True)
    samanantar = load_samanantar_tamil()
    if samanantar:
        lines = list(lines) + samanantar
        print(f"  Total after Samanantar: {len(lines)} lines", flush=True)

    # If no primary, add all data/*.txt (skip LFS pointers)
    if not lines:
        for f in sorted(DATA_DIR.glob("*.txt")):
            if _is_lfs_pointer(f):
                continue
            lines.extend(
                f.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            )
        lines = [s.strip() for s in lines if s.strip()]
        if lines:
            print(f"Using data/*.txt: {len(lines)} lines", flush=True)

    if not lines:
        msg = (
            "No corpus found. Add one of:\n"
            "  • trainn.txt or train.txt (in Agathiyam/)\n"
            "  • data/samanantar_tamil.txt or data/samanantar_ta.txt\n"
            "  • any .txt files in data/\n"
            "If Hugging Face failed with 'No module named _lzma', use a local file above, "
            "or use a Python build with lzma (e.g. install liblzma-dev and reinstall Python)."
        )
        raise SystemExit(msg)
    return lines


def main():
    corpus = load_corpus()
    # Cap merges so we don't run out of pairs on small data
    budget = min(MERGE_BUDGET, max(500, len(corpus) * 50))
    print(f"Corpus: {len(corpus)} lines | Merge budget: {budget}", flush=True)

    print("Training BPE...")
    from core.bpe import train_bpe_with_merges, save_bpe
    token_to_id_bpe, merges_bpe = train_bpe_with_merges(corpus, num_merges=budget)
    save_bpe(
        token_to_id_bpe,
        merges_bpe,
        str(MODELS_DIR / "vocab_bpe.pkl"),
        str(MODELS_DIR / "merges_bpe.pkl"),
    )
    print(f"  BPE vocab size: {len(token_to_id_bpe)} -> models/vocab_bpe.pkl, merges_bpe.pkl")

    # GPE on *raw* corpus (no boundaries) — can merge across sandhi positions
    print("Training GPE (raw corpus, can cross boundaries)...")
    from core.gpe import train_gpe, save_gpe
    token_to_id_gpe, merges_gpe = train_gpe(corpus, num_merges=budget)
    save_gpe(
        token_to_id_gpe,
        merges_gpe,
        str(MODELS_DIR / "vocab_gpe.pkl"),
        str(MODELS_DIR / "merges_gpe.pkl"),
    )
    print(f"  GPE vocab size: {len(token_to_id_gpe)} -> models/vocab_gpe.pkl, merges_gpe.pkl")

    # Agathiyam on *marked* corpus with boundary_char — never merges across ⟂
    print("Training Agathiyam (sandhi-aware GPE, never cross boundaries)...")
    from core.sandhi import BOUND, _mark_mixed
    corpus_sandhi = [_mark_mixed(line) for line in corpus]
    token_to_id_ag, merges_ag = train_gpe(
        corpus_sandhi, num_merges=budget, boundary_char=BOUND
    )
    save_gpe(
        token_to_id_ag,
        merges_ag,
        str(MODELS_DIR / "vocab_re.pkl"),
        str(MODELS_DIR / "merges.pkl"),
    )
    print(f"  Agathiyam vocab size: {len(token_to_id_ag)} -> models/vocab_re.pkl, merges.pkl")

    print("Done. Restart the webapp to use the new models.")


if __name__ == "__main__":
    main()
