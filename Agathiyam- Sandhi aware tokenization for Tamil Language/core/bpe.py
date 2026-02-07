# bpe_local_fast.py
import pickle
from collections import Counter
from typing import List, Tuple, Dict
from pathlib import Path

# ---------------- utilities for reading local corpus ----------------
def load_local_corpus(
    folder: str,
    pattern: str = "*.txt",
    max_sentences: int = 50000,
) -> List[str]:
    """
    Load text lines from all files matching 'pattern' in 'folder'.
    Stops after max_sentences non-empty lines for speed.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    lines: List[str] = []
    for file in folder_path.glob(pattern):
        # adjust encoding if needed
        text = file.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= max_sentences:
                return lines
    return lines


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
            if (
                i < len(word_symbols) - 1
                and word_symbols[i] == find_left
                and word_symbols[i + 1] == find_right
            ):
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
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE on given corpus (list of lines) for a fixed number of merges.
    Returns token_to_id map and ordered merges list.
    """
    vocab = get_vocab(corpus)
    merges: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        most_freq = max(pairs, key=pairs.get)
        merges.append(most_freq)
        vocab = merge_vocab(most_freq, vocab)

    # Build token set from final vocab
    tokens = set()
    for word_symbols in vocab:
        tokens.update(word_symbols)

    # reserve special tokens
    token_to_id: Dict[str, int] = {'<UNK>': 0, '<SPACE>': 1}
    idx = max(token_to_id.values()) + 1
    for t in sorted(tokens):
        if t not in token_to_id:
            token_to_id[t] = idx
            idx += 1

    return token_to_id, merges


def train_bpe_to_vocab_size(
    corpus: List[str],
    target_vocab_size: int,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE until (roughly) reaching target_vocab_size.
    For classic BPE:
        final_vocab_size ≈ initial_symbol_count + num_merges + 2 (special tokens).
    """
    # get initial character-level vocab once
    initial_vocab = get_vocab(corpus)
    base_symbols = set()
    for word_symbols in initial_vocab:
        base_symbols.update(word_symbols)
    # base_symbols includes '</w>'; +2 for <UNK>, <SPACE>
    base_size = len(base_symbols) + 2

    if target_vocab_size <= base_size:
        # no merges needed
        return train_bpe_with_merges(corpus, num_merges=0)

    num_merges = target_vocab_size - base_size
    return train_bpe_with_merges(corpus, num_merges=num_merges)


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
        tokens = list(word) + ['</w>']
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [t for t in tokens if t != '</w>']

    def encode(self, text: str):
        """
        Encode a full text string into BPE tokens and ids.
        Word boundaries are separated using the special token '<SPACE>' in the output token list.
        """
        tokens: List[str] = []
        words = text.strip().split()
        for i, w in enumerate(words):
            sub_tokens = self._apply_merges_to_word(w)
            tokens.extend(sub_tokens)
            if i < len(words) - 1:
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
        return ' '.join(out_words)


# ---------------- save / load ----------------
def save_bpe(
    token_to_id: Dict[str, int],
    merges: List[Tuple[str, str]],
    vocab_file: str,
    merges_file: str,
):
    with open(vocab_file, "wb") as f:
        pickle.dump(token_to_id, f)
    with open(merges_file, "wb") as f:
        pickle.dump(merges, f)


def load_bpe(vocab_file: str, merges_file: str) -> BPETokenizer:
    with open(vocab_file, "rb") as f:
        token_to_id = pickle.load(f)
    with open(merges_file, "rb") as f:
        merges = pickle.load(f)
    return BPETokenizer(token_to_id, merges)


# ---------------- main: train with 10k / 20k / 30k ----------------
if __name__ == "__main__":
    # 1) Load corpus from your local folder (change pattern if not .txt)
    
    with open('trainn.txt', 'r', encoding='utf-8') as f:
        corpus_lines = f.readlines()

    print(f"Loaded {len(corpus_lines)} lines from local corpus")

    # 2) Train for three vocab sizes
    merge_budget = [10000, 20000, 30000,3000000]

    for vs in merge_budget:
        token_to_id, merges = train_bpe_to_vocab_size(corpus_lines, target_vocab_size=vs)
        tokenizer = BPETokenizer(token_to_id, merges)
        print(f"Target vocab {vs}, actual vocab size: {len(token_to_id)}")

        # Save per vocab size
        save_bpe(
            token_to_id,
            merges,
            vocab_file=f"vocab_bpe_{vs}.pkl",
            merges_file=f"merges_bpe_{vs}.pkl",
        )

    # 3) Quick test on a Tamil sentence
    sample_text = "தமிழ் ஒரு செழுமையான மொழி"
    tokens, ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)
    print("BPE tokens:", tokens)
    print("BPE ids:", ids)
    print("Decoded:", decoded)
