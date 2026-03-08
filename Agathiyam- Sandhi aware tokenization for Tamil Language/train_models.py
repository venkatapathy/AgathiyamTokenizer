#!/usr/bin/env python3
"""
Download Samanantar Tamil (Hugging Face or local file), then train BPE and GPE (Agathiyam)
and save to models/ for the webapp.
Run from this directory: python train_models.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Set OVERWRITE_MODELS=1 to allow overwriting existing .pkl files; otherwise leave models/ intact
OVERWRITE_MODELS = os.environ.get("OVERWRITE_MODELS", "").strip().lower() in ("1", "true", "yes")

MERGE_BUDGET = int(os.environ.get("MERGE_BUDGET", "2000"))
SAMANANTAR_MAX_LINES = int(os.environ.get("SAMANANTAR_TAMIL_MAX_LINES", "50000"))


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 300:
        return False
    try:
        text = path.read_text(errors="ignore")
        return "git-lfs.github.com" in text and "oid sha256:" in text
    except Exception:
        return False


def load_samanantar_tamil(max_lines: int = SAMANANTAR_MAX_LINES):
    """Load Tamil sentences: local data/samanantar_tamil.txt or Hugging Face ai4bharat/samanantar."""
    # 1) Local file
    for name in ("samanantar_tamil.txt", "samanantar_ta.txt"):
        p = DATA_DIR / name
        if p.exists() and not _is_lfs_pointer(p):
            lines = [s.strip() for s in p.read_text(encoding="utf-8", errors="ignore").strip().splitlines() if s.strip()][:max_lines]
            if lines:
                print(f"Samanantar Tamil (local {name}): {len(lines)} lines", flush=True)
                return lines

    # 2) Hugging Face (streaming)
    try:
        from datasets import load_dataset
        print("Samanantar Tamil: downloading from Hugging Face (streaming)...", flush=True)
        ds = load_dataset("ai4bharat/samanantar", "ta", split="train", streaming=True)
        lines = []
        for row in ds:
            tgt = (row.get("tgt") or row.get("target") or "").strip()
            if tgt:
                lines.append(tgt)
            if len(lines) >= max_lines:
                break
            if len(lines) in (1000, 5000, 10000, 25000, 50000) and len(lines) % 1000 == 0:
                print(f"  ... {len(lines)} lines", flush=True)
        if lines:
            print(f"Samanantar Tamil (Hugging Face): {len(lines)} lines", flush=True)
            return lines
    except Exception as e:
        print(f"Samanantar Tamil (HF skipped): {e}", flush=True)

    return []


def load_corpus():
    """Build corpus: trainn.txt / train.txt, then add Samanantar Tamil, then data/*.txt."""
    lines = []

    for name in ("trainn.txt", "train.txt"):
        p = ROOT / name
        if p.exists() and not _is_lfs_pointer(p):
            raw = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            lines = [s.strip() for s in raw if s.strip()]
            if lines:
                print(f"Using {name}: {len(lines)} lines", flush=True)
                break
    else:
        lines = []

    samanantar = load_samanantar_tamil()
    if samanantar:
        lines = list(lines) + samanantar
        print(f"Total after Samanantar: {len(lines)} lines", flush=True)

    if not lines and DATA_DIR.exists():
        for f in sorted(DATA_DIR.glob("*.txt")):
            if _is_lfs_pointer(f):
                continue
            lines.extend(f.read_text(encoding="utf-8", errors="ignore").strip().splitlines())
        lines = [s.strip() for s in lines if s.strip()]
        if lines:
            print(f"Using data/*.txt: {len(lines)} lines", flush=True)

    if not lines:
        raise SystemExit(
            "No corpus found. Add data/samanantar_tamil.txt, or set SAMANANTAR_TAMIL_MAX_LINES "
            "and install: pip install datasets (Python 3.10+ with lzma recommended)."
        )
    return lines


def _models_already_exist() -> bool:
    """True if key model files exist (skip overwriting unless OVERWRITE_MODELS=1)."""
    keys = [
        "vocab_bpe.pkl", "merges_bpe.pkl",
        "vocab_gpe.pkl", "merges_gpe.pkl",
        "vocab_re.pkl", "merges.pkl",
    ]
    return all((MODELS_DIR / k).exists() for k in keys)


def main():
    if not OVERWRITE_MODELS and _models_already_exist():
        print("models/ already has vocab and merges files; leaving intact.", flush=True)
        print("To retrain and overwrite, set OVERWRITE_MODELS=1", flush=True)
        return
    corpus = load_corpus()
    budget = min(MERGE_BUDGET, max(200, len(corpus) * 20))

    # BPE
    print("Training BPE...", flush=True)
    from core.bpe import train_bpe, save_bpe, BPETokenizer
    token_to_id_bpe, merges_bpe = train_bpe(corpus, num_merges=budget)
    save_bpe(
        token_to_id_bpe,
        merges_bpe,
        str(MODELS_DIR / "vocab_bpe.pkl"),
        str(MODELS_DIR / "merges_bpe.pkl"),
    )
    print(f"  BPE vocab size {len(token_to_id_bpe)} -> models/vocab_bpe.pkl, merges_bpe.pkl", flush=True)

    # GPE
    print("Training GPE (Agathiyam)...", flush=True)
    from core.gpe import train_gpe, save_gpe, GPETokenizer
    vocab_gpe, merges_gpe = train_gpe(corpus)
    tok_gpe = GPETokenizer(vocab_gpe, merges_gpe)
    save_gpe(tok_gpe, str(MODELS_DIR / "vocab_gpe.pkl"), str(MODELS_DIR / "merges_gpe.pkl"))
    print(f"  GPE vocab size {len(vocab_gpe)} -> models/vocab_gpe.pkl, merges_gpe.pkl", flush=True)

    # Agathiyam (same GPE saved under alternate names for webapp)
    save_gpe(tok_gpe, str(MODELS_DIR / "vocab_re.pkl"), str(MODELS_DIR / "merges.pkl"))
    print("  Agathiyam -> models/vocab_re.pkl, merges.pkl", flush=True)

    print("Done. Restart the webapp to use the new models.", flush=True)


if __name__ == "__main__":
    main()
