"""
Compute grapheme-normalized fertility for Sandhi-GPE
BEFORE and AFTER Sandhi boundary removal.
"""

import numpy as np
import grapheme
from GPE.sandhi import sandhi_mark, remove_boundaries
from GPE.GPE_sandhi import load_tokenizer


# -------------------------------------------------
# Helper: grapheme-normalized fertility
# -------------------------------------------------

def fertility(text, num_tokens):
    L = grapheme.length(text)
    return num_tokens / L if L > 0 else 0.0


# -------------------------------------------------
# Encode with optional Sandhi guidance
# -------------------------------------------------

def sandhi_gpe_encode(text, tokenizer, apply_sandhi=False):
    """
    Returns subword tokens from Sandhi-GPE.
    Sandhi boundaries are applied ONLY as guidance
    and are removed before encoding.
    """
    if apply_sandhi:
        text = sandhi_mark(text, lang="ta")
        text = remove_boundaries(text)

    tokens, _ = tokenizer.encode(text)
    return tokens


# -------------------------------------------------
# Compute average fertility on a corpus
# -------------------------------------------------

def compute_avg_fertility(
    file_path,
    tokenizer,
    apply_sandhi=False,
    max_lines=None
):
    fertilities = []

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break

            text = line.strip()
            if not text:
                continue

            tokens = sandhi_gpe_encode(
                text,
                tokenizer,
                apply_sandhi=apply_sandhi
            )

            fertilities.append(fertility(text, len(tokens)))

    avg_fertility = np.mean(fertilities) if fertilities else 0.0

    label = "with Sandhi guidance" if apply_sandhi else "plain tokenizer"
    print(f"Average fertility ({label}): {avg_fertility:.4f}")

    return avg_fertility


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    file_path = (
        r"C:\Users\ROSHINI PRIYA\Downloads\tokenizers-coling2025-main (4)"
        r"\tokenizers-coling2025-main\data\train_3m.txt"
    )

    print("\nLoading Sandhi-GPE tokenizer...")
    sandhi_tok = load_tokenizer(
        "tokenizers-coling2025-main/vocab.pkl",
        "tokenizers-coling2025-main/merges.pkl"
    )

    print("\nComputing fertility on 3M-word corpus...\n")

    # 1️⃣ Fertility BEFORE Sandhi boundary guidance
    compute_avg_fertility(
        file_path=file_path,
        tokenizer=sandhi_tok,
        apply_sandhi=False
    )

    # 2️⃣ Fertility AFTER Sandhi boundary guidance
    compute_avg_fertility(
        file_path=file_path,
        tokenizer=sandhi_tok,
        apply_sandhi=True
    )