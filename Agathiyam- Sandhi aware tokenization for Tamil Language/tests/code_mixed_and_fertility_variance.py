"""
Evaluation script for Tamil tokenizers:

1. Code-Mixed Robustness:
   - Measures whether English words remain intact in Tamil–English code-mixed text.

2. Cross-Variant Fertility & Variance:
   - Measures average token count (fertility)
   - Measures variance across Sandhi variants of the same morpheme

IMPORTANT:
- This script assumes real tokenizers are plugged in.
- Dummy tokenizers are placeholders for structure validation only.
"""

import re
import numpy as np
# Remove incorrect import below:
# from GPE.GPE_sandhi import bpe_tokenize, gpe_tokenize, sandhi_gpe_tokenize
# Import real tokenizers
from GPE.bpe import load_bpe
from GPE.gpe import load_gpe
from GPE.sandhi import sandhi_split, sandhi_mark

# Load trained tokenizers
bpe_tokenizer = load_bpe()
gpe_tokenizer = load_gpe()

def sandhi_gpe_tokenize(text):
    # Use sandhi_split to tokenize, return only tokens (not offsets)
    return [token for token, _ in sandhi_split(text, lang="ta")]


# Wrapper functions for evaluation

def bpe_tokenize(text):
    tokens, _ = bpe_tokenizer.encode(text)
    return tokens

def gpe_tokenize(text):
    tokens, _ = gpe_tokenizer.encode(text)
    return tokens

# TODO: Replace with actual Sandhi-GPE tokenizer implementation
# def sandhi_gpe_tokenize(text):
#     tokens, _ = sandhi_gpe_tokenizer.encode(text)
#     return tokens


# ------------------------------------------------------------------
# 1. Code-Mixed Robustness Test
# ------------------------------------------------------------------

EN_WORD_PATTERN = re.compile(r"[A-Za-z]+")

def test_code_mixed_robustness(tokenizer, code_mixed_samples):
    """
    Counts how many English words remain whole after tokenization.
    """
    total_english = 0
    unaffected = 0

    for sample in code_mixed_samples:
        tokens = tokenizer(sample)
        english_words = EN_WORD_PATTERN.findall(sample)

        for word in english_words:
            total_english += 1
            if word in tokens:
                unaffected += 1

    print(
        f"Code-mixed robustness: "
        f"{unaffected}/{total_english} English words unaffected."
    )

# ------------------------------------------------------------------
# 2. Cross-Variant Fertility & Variance Test
# ------------------------------------------------------------------

def test_cross_variant_fertility(tokenizer, sandhi_variant_sets):
    """
    Computes:
    - Average fertility (tokens per word)
    - Cross-variant fertility variance
    """
    variances = []
    means = []

    for variants in sandhi_variant_sets:
        token_counts = [len(tokenizer(v)) for v in variants]

        if len(token_counts) > 1:
            means.append(np.mean(token_counts))
            variances.append(np.var(token_counts))

    avg_fertility = np.mean(means) if means else 0.0
    avg_variance = np.mean(variances) if variances else 0.0

    print(f"Average fertility: {avg_fertility:.2f}")
    print(f"Cross-variant fertility variance: {avg_variance:.4f}")

# ------------------------------------------------------------------
# Sandhi Boundary Violation Rate (SBVR) Metric
# ------------------------------------------------------------------
def sandhi_boundary_violation_rate(tokenizer, texts, sandhi_marker, sandhi_aware=False):
    total = 0
    violations = 0

    for text in texts:
        marked = sandhi_marker(text)
        boundaries = marked.count("⟂")
        if boundaries == 0:
            continue

        total += boundaries
        clean = marked.replace("⟂", "")
        tokens = tokenizer(clean)

        # If tokenizer is NOT sandhi-aware, any boundary is a violation
        if not sandhi_aware:
            violations += boundaries
            continue

        # Sandhi-aware tokenizer: check actual boundary crossing
        cursor = 0
        spans = []
        for t in tokens:
            spans.append((cursor, cursor + len(t)))
            cursor += len(t)

        clean_pos = 0
        boundary_positions = []
        for ch in marked:
            if ch == "⟂":
                boundary_positions.append(clean_pos)
            else:
                clean_pos += 1

        for b in boundary_positions:
            for s, e in spans:
                if s < b < e:
                    violations += 1
                    break

    rate = violations / total if total else 0.0
    print(f"Sandhi Boundary Violation Rate: {rate:.3f}")

sandhi_stress_texts = [
    "அவன் இங்கு வந்தான்",
    "மரம் இலை விழுந்தது",
    "கோழி கறி சுவையாக உள்ளது",
    "திரு அருள் பேசினார்",
    "பொன் குடம் உடைந்தது",
]

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def compute_fertility_on_file(file_path, tokenizer, remove_boundary=False, max_lines=None):
    from GPE.sandhi import remove_boundaries
    fertilities = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            text = line.strip()
            if not text:
                continue
            if remove_boundary:
                text = remove_boundaries(text)
            tokens = tokenizer(text)
            fertilities.append(len(tokens))
    avg_fertility = np.mean(fertilities) if fertilities else 0.0
    print(f"Average fertility ({'no boundary' if remove_boundary else 'with boundary'}): {avg_fertility:.4f}")

if __name__ == "__main__":

    # -------------------------
    # Code-mixed evaluation set
    # -------------------------
    code_mixed_samples = [
        "இந்த sentence is code mixed",
        "தமிழ் and English together works",
        "Sandhi rules apply only to தமிழ் words",
        "Tokenization should not break English words"
    ]

    # ------------------------------------------------
    # REALISTIC Sandhi variant surface forms (NO '+')
    # ------------------------------------------------
    sandhi_variant_sets = [
        # verb + particle
        ["படிக்கவும்", "படிக்க வும்", "படிக்கவும்"],

        # verb + suffix
        ["சென்றான்", "சென்று ஆன", "சென்றான்"],

        # noun + noun sandhi
        ["மரம் இலை", "மரமிலை", "மரத்தின் இலை"],

        # vowel sandhi
        ["அவன் இங்கு", "அவனிங்கு", "அவன் இங்கு"]
    ]

    print("\n====================")
    print("BPE")
    print("====================")
    test_code_mixed_robustness(bpe_tokenize, code_mixed_samples)
    test_cross_variant_fertility(bpe_tokenize, sandhi_variant_sets)

    print("\n====================")
    print("GPE")
    print("====================")
    test_code_mixed_robustness(gpe_tokenize, code_mixed_samples)
    test_cross_variant_fertility(gpe_tokenize, sandhi_variant_sets)

    print("\n====================")
    print("Sandhi-GPE")
    print("====================")
    test_code_mixed_robustness(sandhi_gpe_tokenize, code_mixed_samples)
    test_cross_variant_fertility(sandhi_gpe_tokenize, sandhi_variant_sets)


    # --- Fertility on 3M corpus ---
    print("\n====================")
    print("Sandhi-GPE Fertility on 3M corpus (data/train_3m.txt)")
    print("====================")
    compute_fertility_on_file(
        file_path="data/train_3m.txt",
        tokenizer=sandhi_gpe_tokenize,
        remove_boundary=False
    )
    compute_fertility_on_file(
        file_path="data/train_3m.txt",
        tokenizer=sandhi_gpe_tokenize,
        remove_boundary=True
    )
