import json
import regex as re
import grapheme
import os
import pickle
import unicodedata
import time
from tqdm.auto import tqdm
from sandhi import sandhi_split
from datasets import load_dataset

if __name__ == "__main__":
    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    # Use "mix" to enable Tamil + English + code-mix (Tanglish etc.)
    lang = "mix"
    DUMMY_PREFIX = " "
    checkpoint_path = "C:/Users/HP/Documents/vs code/tokenizers-coling2025-main/checkpoint.pkl"

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def calculate_elapsed_time(start_time):
        end_time = time.time()
        time_difference = end_time - start_time
        days, remainder = divmod(time_difference, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(days), int(hours), int(minutes), int(seconds)

    def get_stats(ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):  # consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def save_dict_to_pickle(dictionary, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(dictionary, file)

    # -------------------------------------------------------------------
    # Load training corpus (Tamil + English from Samanantar)
    # -------------------------------------------------------------------
    print("Loading Samanantar Tamil-English dataset...")
    # For the "ta" config, 'src' is typically English and 'tgt' is Tamil.
    ds = load_dataset("ai4bharat/samanantar", "ta")
    tamil_lines = [ex.get("tgt", "") for ex in ds["train"]]
    english_lines = [ex.get("src", "") for ex in ds["train"]]

    lines = []
    for ln in (tamil_lines + english_lines):
        s = re.sub(r'\s+', ' ', (ln or "").strip())
        if DUMMY_PREFIX is not None:
            s = DUMMY_PREFIX + s
        lines.append(s)

    # You can optionally include validation/test too (commented by default)
    # for split in ("validation", "test"):
    #     if split in ds:
    #         tamil_split = [ex.get("tgt", "") for ex in ds[split]]
    #         english_split = [ex.get("src", "") for ex in ds[split]]
    #         for ln in (tamil_split + english_split):
    #             s = re.sub(r'\s+', ' ', (ln or "").strip())
    #             if DUMMY_PREFIX is not None:
    #                 s = DUMMY_PREFIX + s
    #             lines.append(s)

    lines_limited = lines[:]  # use full set; slice here if you want to debug on a subset

    # -------------------------------------------------------------------
    # Build initial vocab (Tamil graphemes + English chars via grapheme)
    # -------------------------------------------------------------------
    intial_gh = []
    progress_bar = tqdm(range(len(lines_limited)), desc="Init vocab (graphemes)")
    for text in lines_limited:
        # lang="mix" applies Tamil sandhi only to Tamil spans; English is pass-through
        text_chunks = sandhi_split(text, lang=lang)  # [(tok,(s,e)),...]
        graphemed_ls = [list(grapheme.graphemes(tok)) for tok, _ in text_chunks]
        # NOTE: grapheme splits English into single letters; Tamil into GCs (with diacritics)
        flat_list = [x for ls in graphemed_ls for x in ls]
        intial_gh.extend(list(set(flat_list)))
        progress_bar.update()

    intial_gh = list(set(intial_gh))
    vocab = {idx: intial_gh[idx] for idx in range(len(intial_gh))}
    vocab_re = {intial_gh[idx]: idx for idx in range(len(intial_gh))}

    # target total vocab size (incl. initial graphemes)
    target_vocab_size = 6_000
    vocab_size_remaining = max(0, target_vocab_size - len(vocab))
    num_merges = vocab_size_remaining

    def covert_to_ids_train(texts):
        ids = []
        progress_bar = tqdm(range(len(texts)), desc="Encode to ids (train)")
        for text in texts:
            text_chunks = sandhi_split(text, lang=lang)
            graphemed_ls = [list(grapheme.graphemes(tok)) for tok, _ in text_chunks]
            ids_temp = [list(map(lambda x: vocab_re[x], ls)) for ls in graphemed_ls]
            ids.extend(ids_temp)
            progress_bar.update()
        return ids

    def covert_to_ids(text_chunk):
        graphemed_ls = list(grapheme.graphemes(text_chunk))
        ids = list(map(lambda x: vocab_re[x], graphemed_ls))
        return ids

    # -------------------------------------------------------------------
    # TRAIN BPE with Checkpointing
    # -------------------------------------------------------------------
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        vocab = state["vocab"]
        vocab_re = state["vocab_re"]
        merges = state["merges"]
        ids = state["ids"]
        start_iter = state["iteration"]
    else:
        print("Starting fresh training...")
        ids = covert_to_ids_train(lines_limited)
        merges = {}
        start_iter = 0

    startTime = time.time()
    for i in range(start_iter, num_merges):
        stats = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)
        if not stats:
            print("No more mergeable pairs found.")
            break
        pair = max(stats, key=stats.get)
        idx = len(vocab) + i
        ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save checkpoint every 100 merges
        if (i+1) % 100 == 0:
            state = {
                "vocab": vocab,
                "vocab_re": vocab_re,
                "merges": merges,
                "ids": ids,
                "iteration": i+1
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(state, f)
            print(f"Checkpoint saved at iteration {i+1}")

    days, hours, minutes, _ = calculate_elapsed_time(startTime)
    print(f"Time taken for training : {days} days {hours} hrs {minutes} mints")
    print("training finished")

    # Save final artifacts
    base = "C:/Users/HP/Documents/vs code/tokenizers-coling2025-main"
    save_dict_to_pickle(merges, os.path.join(base, "merges.pkl"))
    save_dict_to_pickle(vocab, os.path.join(base, "vocab.pkl"))
    save_dict_to_pickle(vocab_re, os.path.join(base, "vocab_re.pkl"))

# -------------------------------------------------------------------
# Tokenizer class
# -------------------------------------------------------------------
class SandhiBPETokenizer:
    def __init__(self, vocab, merges, lang="mix"):
        self.vocab = vocab                # maps id → token
        self.merges = merges
        self.lang = lang
        self.vocab_re = {v: k for k, v in vocab.items()}  # token → id
        self.id_to_token = vocab          # alias for clarity

    def encode(self, text):
        # Step 1: Apply sandhi split (lang-aware; "mix" is default)
        text_chunks = sandhi_split(text, self.lang)
        # Step 2: Convert split tokens to graphemes → IDs# Step 2: Convert split tokens to graphemes → IDs
        # Step 2: Convert split tokens to graphemes → IDs
        graphemes_ls = [list(grapheme.graphemes(tok)) for tok, _ in text_chunks]

        ids = []
        for g_list in graphemes_ls:
            for g in g_list:
                if g in self.vocab_re:
                    ids.append(self.vocab_re[g])
                else:
                    # Handle unseen graphemes (like \n, emojis, rare chars)
                    if "<UNK>" not in self.vocab_re:
                        # Determine numeric max id whether vocab maps id->token or token->id
                        max_id = None
                        # try numeric keys (id -> token)
                        try:
                            max_id = max(int(k) for k in self.vocab.keys())
                        except Exception:
                            pass
                        if max_id is None:
                            # try numeric values (token -> id)
                            try:
                                max_id = max(int(v) for v in self.vocab.values())
                            except Exception:
                                max_id = 0
                        unk_id = max_id + 1
                        self.vocab[unk_id] = "<UNK>"
                        self.vocab_re["<UNK>"] = unk_id
                    ids.append(self.vocab_re["<UNK>"])


        # Step 3: Apply BPE merges (greedy forward pass until convergence)
        changed = True
        while changed:
            changed = False
            i = 0
            new_ids = []
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) in self.merges:
                    new_ids.append(self.merges[(ids[i], ids[i+1])])
                    i += 2
                    changed = True
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        # Optionally return split tokens too (kept for compatibility)
        split_tokens = [tok for tok, _ in text_chunks]
        return split_tokens, ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in self.id_to_token:
                tokens.append(self.id_to_token[i])
            else:
                tokens.append("<UNK>")
        return ''.join(tokens)

def load_tokenizer(vocab_path, merges_path, lang="mix"):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    return SandhiBPETokenizer(vocab, merges, lang)
