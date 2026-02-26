# gpe_vocab_correct.py

import grapheme
import pickle
import time
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def get_stats(ids, stats=None):
    stats = {} if stats is None else stats
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# -------------------------------------------------
# Load Tamil Corpus (LOCAL, FAST)
# -------------------------------------------------

def load_tamil_corpus(folder_path, max_lines=5000):
    corpus = []
    folder = Path(folder_path)

    for file in folder.glob("*.ta"):
        print(f"Reading {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    corpus.append(s)
                if len(corpus) >= max_lines:
                    return corpus
    return corpus


# -------------------------------------------------
# Convert text to grapheme IDs
# -------------------------------------------------

def build_initial_vocab(lines):
    vocab = {}
    vocab_re = {}

    for text in lines:
        for g in grapheme.graphemes(text):
            if g not in vocab_re:
                idx = len(vocab)
                vocab[idx] = g
                vocab_re[g] = idx
    return vocab, vocab_re


def convert_to_ids(lines, vocab_re):
    ids = []
    for text in lines:
        g_ids = [vocab_re[g] for g in grapheme.graphemes(text)]
        ids.append(g_ids)
    return ids


# -------------------------------------------------
# Train GPE (REAL MERGES)
# -------------------------------------------------

def train_gpe(lines, num_merges):
    vocab, vocab_re = build_initial_vocab(lines)
    ids = convert_to_ids(lines, vocab_re)

    merges = {}

    for i in tqdm(range(num_merges), desc=f"GPE merges ({num_merges})"):
        stats = {}
        for chunk in ids:
            get_stats(chunk, stats)

        if not stats:
            break

        best_pair = max(stats, key=stats.get)
        new_id = len(vocab)

        ids = [merge(chunk, best_pair, new_id) for chunk in ids]

        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id

    return vocab, merges


# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":

    with open('trainn.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} lines from local corpus")
   
    MERGE_BUDGETS = [10000, 20000, 30000,3000000]
    

    print("=" * 60)
    print("Grapheme Pair Encoding (GPE)")
    print("Vocabulary Size vs Merge Budget")
    print("=" * 60)

    for budget in MERGE_BUDGETS:
        start = time.time()
        vocab, merges = train_gpe(lines, budget)
        vocab_size = len(vocab)
        elapsed = int(time.time() - start)

        print(
            f"Merge Budget: {budget:>6} | "
            f"Vocabulary Size: {vocab_size:>6} | "
            f"Time: {elapsed}s"
        )

    print("=" * 60)
