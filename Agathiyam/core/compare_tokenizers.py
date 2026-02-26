# Utility: Load lines up to a word cap
def load_lines_with_word_cap(filename, max_words):
    lines = []
    total_words = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word_count = len(line.split())
            if total_words + word_count > max_words:
                # Only add up to the word cap
                allowed = max_words - total_words
                if allowed > 0:
                    words = line.split()[:allowed]
                    lines.append(" ".join(words))
                break
            lines.append(line)
            total_words += word_count
    return lines
# compare_samanantar_local_csv.py
import pickle
import grapheme
import regex as re
import csv
from GPE.bpe import load_bpe
from GPE.gpe import load_gpe
from GPE.GPE_sandhi import load_tokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------ Metrics ------------------
def compression_ratio(text, num_tokens):
    return len(text) / num_tokens if num_tokens > 0 else 0

def fertility_score(text, num_tokens):
    return num_tokens / len(text) if len(text) > 0 else 0

# ------------------ Evaluate a list of texts ------------------
def evaluate_texts(texts, tokenizers):
    results = {}
    for name, tok in tokenizers.items():
        total_cr, total_fs, total_tokens, count = 0, 0, 0, 0
        for text in texts:
            tokens, ids = tok.encode(text)
            num_tokens = len(tokens)
            if num_tokens == 0:
                continue
            total_cr += compression_ratio(text, num_tokens)
            total_fs += fertility_score(text, num_tokens)
            total_tokens += num_tokens
            count += 1
        avg_cr = total_cr / count if count > 0 else 0
        avg_fs = total_fs / count if count > 0 else 0
        avg_tokens = total_tokens / count if count > 0 else 0
        results[name] = (avg_cr, avg_fs, avg_tokens)
    return results

# ------------------ Main ------------------
if __name__ == "__main__":

    local_file = (
        r"C:\Users\ROSHINI PRIYA\Downloads\tokenizers-coling2025-main (4)"
        r"\tokenizers-coling2025-main\data\train_3m.txt"
    )

    print("Loading ~3M words...")
    lines = load_lines_with_word_cap(local_file, max_words=3_000_000)
    print(f"Loaded {len(lines)} lines")

    bpe_tok = load_bpe("vocab_bpe.pkl", "merges_bpe.pkl")
    gpe_tok = load_gpe("vocab_gpe.pkl", "merges_gpe.pkl")
    sandhi_tok = load_tokenizer("vocab.pkl", "merges.pkl")

    tokenizers = {
        "BPE": bpe_tok,
        "GPE": gpe_tok,
        "Sandhi-GPE": sandhi_tok
    }

    results = evaluate_texts(lines, tokenizers)

    for name, (cr, fs, avg_tokens) in results.items():
        print(
            f"{name:<12} "
            f"CR={cr:.4f} "
            f"Fertility={fs:.4f} "
            f"AvgTokens={avg_tokens:.2f}"
        )
