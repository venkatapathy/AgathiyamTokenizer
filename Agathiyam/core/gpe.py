# gpe_correct.py

import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional
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


def get_pair_stats(vocab: Counter, boundary_char: Optional[str] = None) -> Counter:
    """
    Count frequency of each adjacent grapheme pair.
    If boundary_char is set (e.g. "⟂"), never count pairs that include it,
    so merges never cross that boundary (sandhi-aware tokenization).
    """
    pairs = Counter()
    for word_symbols, freq in vocab.items():
        for i in range(len(word_symbols) - 1):
            a, b = word_symbols[i], word_symbols[i + 1]
            if boundary_char is not None and (a == boundary_char or b == boundary_char):
                continue
            pairs[(a, b)] += freq
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
    num_merges: int,
    boundary_char: Optional[str] = None,
    progress_interval: int = 500,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train Grapheme Pair Encoding (GPE).
    If boundary_char is set (e.g. "⟂" for sandhi boundaries), no merge will
    cross that character, so the tokenizer respects the boundary.
    Returns token_to_id and ordered merge list.
    progress_interval: log progress every N merges (0 = no logging).
    """
    vocab = get_vocab(corpus)
    merges = []

    for step in range(num_merges):
        pair_stats = get_pair_stats(vocab, boundary_char=boundary_char)
        if not pair_stats:
            break
        best_pair = max(pair_stats, key=pair_stats.get)
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)
        if progress_interval and (step + 1) % progress_interval == 0:
            label = "Agathiyam" if boundary_char else "GPE"
            print(f"  {label} merge {step + 1}/{num_merges}", flush=True)

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
# Tokenizer object (encode/decode with grapheme-level merges)
# --------------------------------------------------

class GPETokenizer:
    """Grapheme Pair Encoding tokenizer. encode(text) -> (tokens, ids)."""

    def __init__(
        self,
        token_to_id: Dict[str, int],
        merges: List[Tuple[str, str]],
        boundary_char: Optional[str] = None,
    ):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.merges = merges
        self.boundary_char = boundary_char

    def _apply_merges_to_word(self, word: str) -> List[str]:
        """Apply merges to a word (grapheme segments + '</w>'). Returns list of subword tokens.
        If boundary_char is set, never merge across it (sandhi boundary)."""
        symbols = list(grapheme.graphemes(word)) + ["</w>"]
        bound = self.boundary_char
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                    and (bound is None or (symbols[i] != bound and symbols[i + 1] != bound))
                ):
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(symbols[i])
                    i += 1
            symbols = new_tokens
        # Strip internal </w> marker from any merged token before returning
        return [t.replace("</w>", "") for t in symbols if t != "</w>"]

    def encode(self, text: str):
        """Encode text into GPE tokens and ids. Word boundaries get '<SPACE>'."""
        tokens: List[str] = []
        words = text.strip().split()
        for i, w in enumerate(words):
            sub_tokens = self._apply_merges_to_word(w)
            tokens.extend(sub_tokens)
            if i < len(words) - 1:
                tokens.append("<SPACE>")
        ids = [self.token_to_id.get(t, self.token_to_id.get("<UNK>", 0)) for t in tokens]
        return tokens, ids

    def decode(self, ids: List[int]) -> str:
        """Decode ids back to string."""
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        out_words = []
        current = []
        for t in tokens:
            if t == "<SPACE>":
                out_words.append("".join(current))
                current = []
            else:
                current.append(t)
        if current:
            out_words.append("".join(current))
        return " ".join(out_words)


def save_gpe(
    token_to_id: Dict[str, int],
    merges: List[Tuple[str, str]],
    vocab_file: str,
    merges_file: str,
):
    with open(vocab_file, "wb") as f:
        pickle.dump(token_to_id, f)
    with open(merges_file, "wb") as f:
        pickle.dump(merges, f)


def load_gpe(
    vocab_file: str,
    merges_file: str,
    boundary_char: Optional[str] = None,
) -> GPETokenizer:
    with open(vocab_file, "rb") as f:
        token_to_id = pickle.load(f)
    with open(merges_file, "rb") as f:
        merges = pickle.load(f)
    return GPETokenizer(token_to_id, merges, boundary_char=boundary_char)


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
