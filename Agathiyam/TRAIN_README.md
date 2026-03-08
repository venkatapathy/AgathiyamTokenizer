# Training BPE and Agathiyam (GPE) with Samanantar Tamil

## Quick run

From this directory (`Agathiyam- Sandhi aware tokenization for Tamil Language`):

```bash
# Use Python 3.10 venv from sibling Agathiyam/ (has datasets + lzma)
../Agathiyam/.venv310/bin/python -u train_models.py
```

Or use the script (same venv):

```bash
chmod +x run_train.sh
./run_train.sh
```

## What it does

1. **Samanantar Tamil (what gets “downloaded”)**  
   - Tries **local** `data/samanantar_tamil.txt` (or `samanantar_ta.txt`) first. **No download** if these exist.  
   - If not found, it **streams** Tamil from Hugging Face **ai4bharat/samanantar** (config `ta`). That data is used in memory for training; nothing is written into `data/` by the script.

2. **Training**  
   - **BPE**: `core/bpe.py` → `models/vocab_bpe.pkl`, `models/merges_bpe.pkl`  
   - **GPE (Agathiyam)**: `core/gpe.py` → `models/vocab_gpe.pkl`, `models/merges_gpe.pkl`  
   - **Agathiyam** (same GPE): `models/vocab_re.pkl`, `models/merges.pkl`

3. **Keeping `data/` and `models/` intact**  
   - If all of the above `.pkl` files already exist in `models/`, the script **does not overwrite** them (it exits with a message).  
   - To force retraining and overwrite, set: `OVERWRITE_MODELS=1` when running.

## Environment variables

- `SAMANANTAR_TAMIL_MAX_LINES` – max Tamil lines to use (default `50000`). Lower for faster runs (e.g. `5000`).
- `MERGE_BUDGET` – BPE merge steps (default `2000`).
- `OVERWRITE_MODELS` – set to `1` to allow overwriting existing `models/*.pkl`; otherwise they are left intact.

Example (faster run):

```bash
SAMANANTAR_TAMIL_MAX_LINES=5000 MERGE_BUDGET=500 ../Agathiyam/.venv310/bin/python -u train_models.py
```

## Python / venv

Training needs **Python 3.10+** with `lzma` and `datasets` (for HF Samanantar).  
If `Agathiyam/.venv310` doesn’t exist, create it from repo root:

```bash
cd Agathiyam
python3.10 -m venv .venv310
. .venv310/bin/activate
pip install grapheme regex datasets
```

Then run `train_models.py` from this directory as above.
