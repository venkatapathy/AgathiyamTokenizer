#!/usr/bin/env python3
# oov_eval.py

import time
import csv
import os
import pandas as pd
from typing import List, Tuple


def load_text(filepath: str) -> List[str]:
    """Load a text file as a list of whitespace-tokenized words."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().split()


def compute_oov_word_level(train_file: str, test_file: str) -> Tuple[float, set]:
    """
    Compute word-level OOV coverage.
    Coverage = % of test tokens that appear in training vocab.
    """
    train_words = set(load_text(train_file))
    test_words = load_text(test_file)

    oov_tokens = [w for w in test_words if w not in train_words]
    oov_rate = len(oov_tokens) / len(test_words) if test_words else 0.0

    coverage = 1 - oov_rate
    return coverage, set(oov_tokens)


def compute_oov_token_level(test_file: str, tokenizer) -> float:
    """
    Compute OOV token ratio for subword tokenizers.
    Ratio = % of produced tokens that are <unk>.
    Assumes tokenizer implements `encode` returning token IDs,
    and has a defined unk_id.
    """
    with open(test_file, "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    total_tokens, unk_tokens = 0, 0
    for line in test_lines:
        ids = tokenizer.encode(line)
        total_tokens += len(ids)
        unk_tokens += sum(1 for i in ids if i == tokenizer.unk_id)

    return unk_tokens / total_tokens if total_tokens > 0 else 0.0


def measure_throughput(tokenizer, text_file: str, repeat: int = 3) -> float:
    """
    Measure throughput (tokens/sec) for tokenization.
    """
    lines = load_text(text_file)
    start = time.time()
    for _ in range(repeat):
        for line in lines:
            _ = tokenizer.encode(line)
    elapsed = time.time() - start
    tokens = sum(len(tokenizer.encode(line)) for line in lines) * repeat
    return tokens / elapsed if elapsed > 0 else 0.0


def run_oov_eval(train_file: str, test_file: str, tokenizer, exp_name: str,
                 matrix_csv: str = "experiment_matrix.csv"):
    # Word-level OOV
    word_coverage, _ = compute_oov_word_level(train_file, test_file)
    word_oov_rate = 1 - word_coverage

    # Token-level OOV
    token_oov_rate = compute_oov_token_level(test_file, tokenizer)

    # Throughput
    throughput = measure_throughput(tokenizer, test_file)

    # Merge with experiment matrix
    if os.path.exists(matrix_csv):
        df = pd.read_csv(matrix_csv)
    else:
        # create a new one if it doesn’t exist
        df = pd.DataFrame(columns=[
            "Experiment", "Fertility", "Compression", "PPL",
            "Word_OOV_Rate", "Token_OOV_Ratio", "Throughput"
        ])

    # Check if exp_name already exists in CSV
    if exp_name in df["Experiment"].values:
        # Update row
        df.loc[df["Experiment"] == exp_name, "Word_OOV_Rate"] = word_oov_rate
        df.loc[df["Experiment"] == exp_name, "Token_OOV_Ratio"] = token_oov_rate
        df.loc[df["Experiment"] == exp_name, "Throughput"] = throughput
    else:
        # Add new row (other cols left blank if not known yet)
        df = pd.concat([
            df,
            pd.DataFrame([{
                "Experiment": exp_name,
                "Word_OOV_Rate": word_oov_rate,
                "Token_OOV_Ratio": token_oov_rate,
                "Throughput": throughput
            }])
        ], ignore_index=True)

    # Save updated CSV
    df.to_csv(matrix_csv, index=False, encoding="utf-8")

    print(f"[{exp_name}] Word OOV Rate: {word_oov_rate:.4f}, "
          f"Token OOV Ratio: {token_oov_rate:.4f}, "
          f"Throughput: {throughput:.2f} tok/s")

    return word_oov_rate, token_oov_rate, throughput


if __name__ == "__main__":
    """
    Example usage:
    python oov_eval.py

    Assumes you have:
    - data/train.txt
    - data/test.txt
    - experiment_matrix.csv (optional, created if not found)
    - a tokenizer object (SentencePiece, BPE, Sandhi, etc.)
    """
    import sentencepiece as spm

    # Load a SentencePiece tokenizer (example)
    sp = spm.SentencePieceProcessor()
    sp.load("models/gemma-3-270m/tokenizer.model")

    # Add unk_id attribute if not present
    if not hasattr(sp, "unk_id"):
        sp.unk_id = sp.piece_to_id("<unk>")

#training and testing sets
run_oov_eval("data/samanantar_eng_90_percent_cleaned1.txt", "data/sample.txt", sp, exp_name="V0-Baseline")
