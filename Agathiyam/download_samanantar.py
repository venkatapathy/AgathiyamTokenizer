#!/usr/bin/env python3
"""
Download ai4bharat/samanantar Tamil split from Hugging Face and save to data/samanantar_tamil.txt.
Run once so the data is local; train_models.py will then use it.

  cd Agathiyam && python download_samanantar.py

Uses Hugging Face parquet files (huggingface_hub + pyarrow) so it can work even when
Python was built without _lzma (unlike the full 'datasets' library).

Env:
  SAMANANTAR_TAMIL_MAX_LINES  Max lines to save (default 100000). Use 0 for full dataset.
  HF_HOME / HUGGINGFACE_HUB_CACHE  Where HF caches files (default ~/.cache/huggingface).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE = DATA_DIR / "samanantar_tamil.txt"

MAX_LINES = int(os.getenv("SAMANANTAR_TAMIL_MAX_LINES", "100000"))
REPO_ID = "ai4bharat/samanantar"
PARQUET_REVISION = "refs/convert/parquet"


def _download_via_parquet(max_lines: int) -> list[str]:
    """Download Tamil split using only huggingface_hub + pyarrow (no 'datasets', avoids _lzma)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"  Fallback needs: pip install huggingface_hub pyarrow  ({e})")
        return []

    try:
        files = list_repo_files(REPO_ID, repo_type="dataset", revision=PARQUET_REVISION)
    except Exception as e:
        print(f"  Could not list parquet files: {e}")
        return []

    ta_files = sorted(f for f in files if f.startswith("ta/") and f.endswith(".parquet"))
    if not ta_files:
        print("  No ta/*.parquet files found on parquet branch.")
        return []

    lines = []
    for filename in ta_files:
        if max_lines > 0 and len(lines) >= max_lines:
            break
        path = hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            revision=PARQUET_REVISION,
        )
        table = pq.read_table(path)
        if "tgt" in table.column_names:
            col = table.column("tgt")
        elif "target" in table.column_names:
            col = table.column("target")
        else:
            col = table.column(table.column_names[1])
        for i in range(col.length()):
            if max_lines > 0 and len(lines) >= max_lines:
                break
            s = col[i].as_py()
            if isinstance(s, str) and s.strip():
                lines.append(s.strip())
    return lines


def main():
    print("Downloading ai4bharat/samanantar (ta)...")
    lines = []

    # 1) Try 'datasets' (full API; needs _lzma on some Pythons)
    try:
        from datasets import load_dataset
        from tqdm import tqdm

        ds = load_dataset(REPO_ID, "ta", trust_remote_code=True, split="train")
        n_total = len(ds)
        cap = MAX_LINES if MAX_LINES > 0 else n_total
        n_take = min(n_total, cap)
        print(f"  Train split size: {n_total}. Saving up to {n_take} Tamil lines.")
        for i in tqdm(range(n_take), desc="Extracting Tamil", unit=" rows"):
            row = ds[i]
            tgt = (row.get("tgt") or row.get("target") or "").strip()
            if tgt:
                lines.append(tgt)
    except (ImportError, ModuleNotFoundError) as e:
        err = str(e)
        if "_lzma" in err:
            print("  'datasets' needs _lzma; using parquet fallback (huggingface_hub + pyarrow).")
            lines = _download_via_parquet(MAX_LINES)
        else:
            print(f"  Error: {e}")
            print("  Install: pip install datasets tqdm")
            sys.exit(1)
    except Exception as e:
        if "_lzma" in str(e):
            print("  'datasets' hit _lzma; using parquet fallback.")
            lines = _download_via_parquet(MAX_LINES)
        else:
            raise

    if not lines:
        print("  No Tamil lines obtained.")
        print("  If you see _lzma errors, try: pip install pyarrow  and run again.")
        print("  Or use a Python built with _lzma: /usr/bin/python3 download_samanantar.py")
        sys.exit(1)

    OUT_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"  Wrote {len(lines)} lines to {OUT_FILE}")
    print("  Run train_models.py; it will use this file.")


if __name__ == "__main__":
    main()
