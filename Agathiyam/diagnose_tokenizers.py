#!/usr/bin/env python3
"""
Diagnose why Agathiyam vs GPE produce same token count.
Run from Agathiyam/: python diagnose_tokenizers.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

def main():
    # Exact user text (spaces as given)
    text = "முதலாம் தியோனீசியசு ( அண். கி.மு. 432–367 ) என்பவர் பண்டைய கிரேக்க சர்வாதிகாரி ஆவார்"
    print("Input text (repr):", repr(text[:50]) + "...")
    print()

    # 1) Sandhi marking
    from core.sandhi import _mark_mixed, BOUND
    marked = _mark_mixed(text)
    n_bound = marked.count(BOUND)
    print(f"1) Sandhi: {n_bound} boundaries inserted")
    if n_bound:
        # Show where
        parts = marked.split(BOUND)
        print(f"   Marked has {len(parts)} segments (split by BOUND)")
        for i, p in enumerate(parts[:5]):
            print(f"   segment {i}: {repr(p[:30])}...")
    print()

    # 2) How does encode() split input?
    words_raw = text.strip().split()
    words_marked = marked.strip().split()
    print(f"2) Word split: raw={len(words_raw)} words, marked={len(words_marked)} words")
    if words_raw != words_marked:
        for i, (r, m) in enumerate(zip(words_raw, words_marked)):
            if r != m:
                print(f"   word {i}: raw={repr(r[:25])}  marked={repr(m[:25])}")
    else:
        print("   Same word boundaries (space-split); words containing BOUND:")
        for i, w in enumerate(words_marked):
            if BOUND in w:
                print(f"   word {i}: {repr(w)}")
    print()

    # 3) Grapheme split of a word that contains BOUND
    try:
        import grapheme
        sample_word = "தி⟂யோனீசியசு"  # word with boundaries
        graphemes = list(grapheme.graphemes(sample_word))
        print(f"3) Graphemes for {repr(sample_word)}:")
        print(f"   {graphemes}")
        print(f"   BOUND in graphemes: {BOUND in graphemes}")
    except ImportError:
        print("3) grapheme module not found, skipping")
    print()

    # 4) Load both tokenizers and encode
    import pickle
    # GPE
    gpe_v = MODELS_DIR / "vocab_gpe.pkl"
    gpe_m = MODELS_DIR / "merges_gpe.pkl"
    ag_v = MODELS_DIR / "vocab_re.pkl"
    ag_m = MODELS_DIR / "merges.pkl"

    if not gpe_v.exists() or not gpe_m.exists():
        print("4) GPE model files missing, skipping encode comparison")
        return
    if not ag_v.exists() or not ag_m.exists():
        print("4) Agathiyam model files missing, skipping encode comparison")
        return

    from core.gpe import GPETokenizer
    with open(gpe_v, "rb") as f:
        gpe_vocab = pickle.load(f)
    with open(gpe_m, "rb") as f:
        gpe_merges = pickle.load(f)
    with open(ag_v, "rb") as f:
        ag_vocab = pickle.load(f)
    with open(ag_m, "rb") as f:
        ag_merges = pickle.load(f)

    tok_gpe = GPETokenizer(gpe_vocab, gpe_merges)
    tok_ag = GPETokenizer(ag_vocab, ag_merges, boundary_char=BOUND)

    gpe_tokens, gpe_ids = tok_gpe.encode(text)
    ag_tokens_full, ag_ids_full = tok_ag.encode(marked)
    ag_tokens = [t for t in ag_tokens_full if t != BOUND]
    ag_ids = [i for t, i in zip(ag_tokens_full, ag_ids_full) if t != BOUND]

    print("4) Token counts:")
    print(f"   GPE:       {len(gpe_tokens)} tokens")
    print(f"   Agathiyam (before filter): {len(ag_tokens_full)} tokens (of which BOUND: {sum(1 for t in ag_tokens_full if t == BOUND)})")
    print(f"   Agathiyam (after filter):  {len(ag_tokens)} tokens")
    print()

    # Per-word breakdown for words that have boundaries
    words_m = marked.strip().split()
    words_r = text.strip().split()
    print("5) Per-word token count (words with boundaries):")
    for idx, (wr, wm) in enumerate(zip(words_r, words_m)):
        if BOUND in wm or wr != wm:
            st_gpe = tok_gpe._apply_merges_to_word(wr)
            st_ag = tok_ag._apply_merges_to_word(wm)
            print(f"   Word {idx} marked={repr(wm[:35])}")
            print(f"      GPE: {len(st_gpe)} sub-tokens | Agathiyam: {len(st_ag)} sub-tokens")
            if st_gpe != st_ag:
                print(f"      GPE: {st_gpe}")
                print(f"      Ag:  {st_ag}")
    print()

    if gpe_tokens == ag_tokens:
        print("6) Token lists are IDENTICAL.")
    else:
        print("6) Token lists DIFFER")
        for i, (a, b) in enumerate(zip(gpe_tokens, ag_tokens)):
            if a != b:
                print(f"   First diff at index {i}: GPE={repr(a)} vs Agathiyam={repr(b)}")
                break
        print(f"   Lengths: GPE={len(gpe_tokens)}, Agathiyam={len(ag_tokens)}")

    # 7) Are GPE and Agathiyam using the same merges?
    print()
    print("7) Model comparison:")
    print(f"   GPE merges count: {len(gpe_merges)}")
    print(f"   Agathiyam merges count: {len(ag_merges)}")
    print(f"   Same merge list: {gpe_merges == ag_merges}")
    if gpe_merges == ag_merges:
        print("   >>> WARNING: GPE and Agathiyam use the SAME merges. Train GPE on raw corpus and Agathiyam on marked corpus to get different behavior.")

    # 8) Simulate app: what would the API return for each tokenizer?
    print()
    print("8) Simulated app response (same as /api/tokenize):")
    print(f"   GPE:       count={len(gpe_tokens)}")
    print(f"   Agathiyam: count={len(ag_tokens)} (sandhi_boundaries={n_bound})")
    try:
        from core.bpe import BPETokenizer
        bpe_v = MODELS_DIR / "vocab_bpe.pkl"
        bpe_m = MODELS_DIR / "merges_bpe.pkl"
        if bpe_v.exists() and bpe_m.exists():
            with open(bpe_v, "rb") as f:
                bpe_vocab = pickle.load(f)
            with open(bpe_m, "rb") as f:
                bpe_merges = pickle.load(f)
            tok_bpe = BPETokenizer(bpe_vocab, bpe_merges)
            bpe_tokens, _ = tok_bpe.encode(text)
            print(f"   BPE:       count={len(bpe_tokens)}")
        else:
            print("   BPE:       (model files not found)")
    except Exception as e:
        print(f"   BPE:       error={e}")
    print()
    print("If all three show the same count, retrain: run train_models.py so GPE is")
    print("trained on raw corpus (can merge across sandhi) and Agathiyam on marked")
    print("corpus with boundary_char (never merges across ⟂). Then restart the app.")

if __name__ == "__main__":
    main()
