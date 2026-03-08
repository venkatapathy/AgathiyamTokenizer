# Tokenizer comparison webapp

A small web UI to compare how different tokenizers split the same sentence.

## Run

From the `Agathiyam` directory:

```bash
# Create/activate venv and install deps
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask grapheme regex

# Start the app (listens on all interfaces so you can use any URL / IP)
python app.py
# Or: flask --app app run --host 0.0.0.0
```

Then open http://127.0.0.1:10010 (or http://&lt;your-machine-ip&gt;:10010) in a browser.

## Tokenizers shown

Only **BPE**, **GPE**, and **Agathiyam** are shown. Each is loaded from `models/` when the corresponding `.pkl` files exist and are valid.

## "Model not loaded" or Git LFS pointer

If the app says **Model file is a Git LFS pointer**, the `.pkl` files in `models/` are placeholders. Download the real files with:

```bash
git lfs pull
```

Run this in the repository root (e.g. `Agathiyam` or the repo containing `models/`). After that, restart the app.

To train and save your own BPE/GPE models, use `core/bpe.py` and `core/gpe.py`; the webapp expects the same format as `save_bpe` / `save_gpe`.
