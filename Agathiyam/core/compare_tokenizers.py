# compare_samanantar_local_csv.py
import pickle
import grapheme
import regex as re
import csv
from bpe import load_bpe
from gpe import load_gpe
from GPE_sandhi import load_tokenizer

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
    # Load Samanantar Tamil dataset from local file
    local_file = "data/samanantar_eng_90_percent_cleaned1.txt"
    ws_pat = re.compile(r'\s+')
    with open(local_file, "r", encoding="utf-8") as f:
        lines = [ws_pat.sub(' ', line.strip()) for line in f if line.strip()]

    # Load pretrained tokenizers (trained on Samanantar)
    bpe_tok = load_bpe("models/vocab_bpe.pkl", "models/merges_bpe.pkl")
    gpe_tok = load_gpe("models/vocab_gpe.pkl", "models/merges_gpe.pkl")
    sandhi_tok = load_tokenizer("models/vocab_re.pkl", "models/merges.pkl")

    tokenizers = {
        "BPE": bpe_tok,
        "GPE": gpe_tok,
        "Sandhi-GPE": sandhi_tok
    }

    # CSV output
    csv_file = "evaluation_results.csv"
    header = ["Limit", "Tokenizer", "Avg CR", "Avg FS", "Avg Tokens"]
    rows = []

    # Evaluate on multiple limits
    for limit in [500, 1000, 2000, 3000]:
        subset = lines[:limit]
        results = evaluate_texts(subset, tokenizers)
        print(f"\n--- Evaluating {limit} lines ---")
        for name, (cr, fs, avg_tokens) in results.items():
            print(f"{name:<12} Avg CR={cr:.4f} Avg FS={fs:.4f} Avg Tokens={avg_tokens:.2f}")
            rows.append([limit, name, cr, fs, avg_tokens])

    # Save to CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ Results saved to {csv_file}")

    # ------------------ Print detailed tokenization for a test sentence ------------------
    test_sentence = "இது ஒரு சோதனை வாக்கியம்"
    print("\n--- Tokenization details for test sentence ---")
    for name, tok in tokenizers.items():
            tokens, ids = tok.encode(test_sentence)

            # Use tokens for decoding (safe fallback if decode() fails)
            try:
                decoded = tok.decode(ids)
            except Exception:
                decoded = "".join(tokens)

            print(f"\n{name} Results")
            print("-" * (len(name) + 10))
            print(f"Tokens       : {tokens}")
            print(f"Num tokens   : {len(tokens)}")
            print(f"Decoded text : {decoded}")
