# bpe_correct.py
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple, Dict


# ---------------- utilities for BPE training ----------------
def get_vocab(corpus: List[str]) -> Counter:
    """
    Build initial vocabulary from corpus.
    Each word is represented as a tuple of characters with an end-of-word marker '</w>'.
    Returns Counter mapping tuple(symbols...) -> frequency.
    """
    vocab = Counter()
    for line in corpus:
        for word in line.strip().split():
            if not word:
                continue
            symbols = tuple(list(word) + ['</w>'])
            vocab[symbols] += 1
    return vocab


def get_pair_stats(vocab: Counter) -> Counter:
    """
    Count frequency of each adjacent pair (bigram of symbols) across the vocab,
    weighted by the word frequency.
    """
    pairs = Counter()
    for word_symbols, freq in vocab.items():
        for i in range(len(word_symbols) - 1):
            pair = (word_symbols[i], word_symbols[i + 1])
            pairs[pair] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Counter) -> Counter:
    """
    Replace all occurrences of the pair in vocabulary keys with the merged symbol.
    For instance pair=('a','b') => new symbol 'ab'.
    """
    merged_symbol = pair[0] + pair[1]
    new_vocab = Counter()
    find_left, find_right = pair

    for word_symbols, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_symbols):
            if i < len(word_symbols) - 1 and word_symbols[i] == find_left and word_symbols[i + 1] == find_right:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word_symbols[i])
                i += 1
        new_vocab[tuple(new_word)] += freq

    return new_vocab


def train_bpe_with_merges(
    corpus: List[str],
    num_merges: int,
    progress_interval: int = 500,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE on given corpus (list of lines) for a fixed number of merges.
    Returns token_to_id map and ordered merges list.
    progress_interval: log progress every N merges (0 = no logging).
    """
    vocab = get_vocab(corpus)
    merges: List[Tuple[str, str]] = []

    for step in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        most_freq = max(pairs, key=pairs.get)
        merges.append(most_freq)
        vocab = merge_vocab(most_freq, vocab)
        if progress_interval and (step + 1) % progress_interval == 0:
            print(f"  BPE merge {step + 1}/{num_merges}", flush=True)

    # Build token set from final vocab
    tokens = set()
    for word_symbols in vocab:
        tokens.update(word_symbols)

    # reserve special tokens
    token_to_id = {'<UNK>': 0, '<SPACE>': 1}
    idx = max(token_to_id.values()) + 1
    for t in sorted(tokens):
        if t not in token_to_id:
            token_to_id[t] = idx
            idx += 1

    return token_to_id, merges


# Backward compatibility
def train_bpe(corpus: List[str], num_merges: int = 100) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Train BPE (alias for train_bpe_with_merges)."""
    return train_bpe_with_merges(corpus, num_merges, progress_interval=0)


# ---------------- tokenizer object ----------------
class BPETokenizer:
    def __init__(self, token_to_id: Dict[str, int], merges: List[Tuple[str, str]]):
        """
        token_to_id: mapping token -> id (includes special tokens like '<UNK>' and '<SPACE>')
        merges: ordered list of merge pairs, e.g. [('a','b'), ('ab','c'), ...]
        """
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.merges = merges  # ordered list

    def _apply_merges_to_word(self, word: str) -> List[str]:
        """
        Apply learned merges (in order) to a single word.
        Returns list of BPE subword tokens for that word (without the '</w>' marker).
        """
        # initial token sequence: characters + end-of-word marker
        tokens = list(word) + ['</w>']
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        # drop the end-of-word marker and return
        return [t for t in tokens if t != '</w>']

    def encode(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Encode a full text string into BPE tokens and ids.
        Word boundaries are separated using the special token '<SPACE>' in the output token list.
        """
        tokens: List[str] = []
        for i, w in enumerate(text.strip().split()):
            sub_tokens = self._apply_merges_to_word(w)
            tokens.extend(sub_tokens)
            # append explicit space token between words
            if i < len(text.strip().split()) - 1:
                tokens.append('<SPACE>')

        ids = [self.token_to_id.get(t, self.token_to_id.get('<UNK>', 0)) for t in tokens]
        return tokens, ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of ids back to a string. '<SPACE>' id becomes a space.
        """
        tokens = [self.id_to_token.get(i, '<UNK>') for i in ids]
        out_words = []
        current = []
        for t in tokens:
            if t == '<SPACE>':
                out_words.append(''.join(current))
                current = []
            else:
                current.append(t)
        if current:
            out_words.append(''.join(current))
        # join words with spaces
        return ' '.join(out_words)


# ---------------- save / load ----------------
def save_bpe(token_to_id: Dict[str, int], merges: List[Tuple[str, str]],
             vocab_file: str = "vocab_bpe.pkl", merges_file: str = "merges_bpe.pkl"):
    with open(vocab_file, "wb") as f:
        pickle.dump(token_to_id, f)
    with open(merges_file, "wb") as f:
        pickle.dump(merges, f)


def load_bpe(vocab_file: str = "vocab_bpe.pkl", merges_file: str = "merges_bpe.pkl") -> BPETokenizer:
    with open(vocab_file, "rb") as f:
        token_to_id = pickle.load(f)
    with open(merges_file, "rb") as f:
        merges = pickle.load(f)
    return BPETokenizer(token_to_id, merges)


# ---------------- example usage ----------------
if __name__ == "__main__":
    # small test/training on first 500 lines of your file (like you had)
    with open('data/samanantar_eng_90_percent_cleaned1.txt', 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()][:500]

    # Train BPE (tune num_merges)
    token_to_id, merges = train_bpe(corpus_lines, num_merges=200)
    tokenizer = BPETokenizer(token_to_id, merges)

    # Save
    save_bpe(token_to_id, merges)

    # Test
    sample_text = "தமிழ் ஒரு செழுமையான மொழி"
    tokens, ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)

    print("BPE tokens:", tokens)
    print("BPE ids:", ids)
    print("Decoded:", decoded)
