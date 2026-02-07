# gpe_correct.py

import pickle
from collections import Counter
from typing import List, Tuple, Dict
from pathlib import Path
import grapheme   # pip install grapheme


# --------------------------------------------------
# Utilities for GPE training
# --------------------------------------------------

def get_vocab(corpus: List[str]) -> Counter:
    """
    Build initial vocabulary from corpus.
    Each word is represented as a tuple of graphemes with an end marker '</w>'.
    Returns Counter mapping tuple(graphemes...) -> frequency.
    """
    vocab = Counter()
    for line in corpus:
        for word in line.strip().split():
            if not word:
                continue
            symbols = tuple(list(grapheme.graphemes(word)) + ['</w>'])
            vocab[symbols] += 1
    return vocab


def get_pair_stats(vocab: Counter) -> Counter:
    """
    Count frequency of each adjacent grapheme pair.
    """
    pairs = Counter()
    for word_symbols, freq in vocab.items():
        for i in range(len(word_symbols) - 1):
            pairs[(word_symbols[i], word_symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Counter) -> Counter:
    """
    Merge a grapheme pair into a single symbol.
    """
    merged_symbol = pair[0] + pair[1]
    new_vocab = Counter()

    for word_symbols, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_symbols):
            if (
                i < len(word_symbols) - 1
                and word_symbols[i] == pair[0]
                and word_symbols[i + 1] == pair[1]
            ):
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word_symbols[i])
                i += 1
        new_vocab[tuple(new_word)] += freq

    return new_vocab


def train_gpe(
    corpus: List[str],
    num_merges: int
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train Grapheme Pair Encoding (GPE).
    Returns token_to_id and ordered merge list.
    """
    vocab = get_vocab(corpus)
    merges = []

    for _ in range(num_merges):
        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break
        best_pair = max(pair_stats, key=pair_stats.get)
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

    # Build final token inventory
    tokens = set()
    for word in vocab:
        tokens.update(word)

    token_to_id = {"<UNK>": 0, "<SPACE>": 1}
    idx = 2
    for tok in sorted(tokens):
        if tok not in token_to_id:
            token_to_id[tok] = idx
            idx += 1

    return token_to_id, merges


# --------------------------------------------------
# Local corpus loader (FAST)
# --------------------------------------------------

def load_tamil_corpus(folder_path: str, max_lines: int = 5000) -> List[str]:
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


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

if __name__ == "__main__":

    DATA_PATH = r"C:\Users\Subika M\Downloads\archive\final_data\en-ta"
      # reduce further if needed

    print("Loading Tamil corpus...")
    with open('trainn.txt', 'r', encoding='utf-8') as f:
        corpus = f.readlines()

    print(f"Loaded {len(corpus)} lines from local corpus")

    MERGE_BUDGETS = [10000, 20000, 30000,3000000]
    MAX_LINES = 5000 

    print("=" * 60)
    print("Grapheme Pair Encoding (GPE)")
    print("Vocabulary Size vs Merge Budget")
    print("=" * 60)

    for budget in MERGE_BUDGETS:
        token_to_id, merges = train_gpe(corpus, num_merges=budget)
        vocab_size = len(token_to_id)

        print(
            f"Merge Budget: {budget:>6} | "
            f"Vocabulary Size: {vocab_size}"
        )

    print("=" * 60)
