# Getting LFS model files (for forks)

Your fork doesn’t have the Git LFS objects; they stay in the original repo. Fetch them from **upstream**.

## 1. Add the original repo as upstream

Use the **original** repo URL (the one you forked from), for example:

```bash
cd /home/venkat/AgathiyamTokenizer
git remote add upstream https://github.com/RoshiniPriya05/Agathiyam-Tamil.git
```

If the original is different (e.g. another AgathiyamTokenizer repo), replace the URL.

## 2. Fetch LFS objects from upstream

```bash
git lfs fetch upstream
git lfs checkout
```

Or, to pull the current branch from upstream including LFS:

```bash
git lfs fetch upstream
git lfs pull upstream main
```

(Use `master` instead of `main` if the original repo’s default branch is `master`.)

## 3. Check that real files are there

```bash
file Agathiyam/models/vocab_bpe.pkl
```

You want something like `data` or `Python pickle`, **not** `ASCII text`. If it still says ASCII text, LFS didn’t pull. Try:

```bash
git lfs pull upstream main --include="*.pkl"
```

---

**If you don’t have the original repo or LFS still 404s:** train the models yourself so the webapp has real `.pkl` files:

```bash
cd Agathiyam
# Put some training text in trainn.txt, then:
python -m core.bpe    # creates vocab_bpe_*.pkl, merges_bpe_*.pkl
python -m core.gpe    # creates vocab/merges; copy to models/ with the expected names
```

Then copy the generated `vocab_bpe_*.pkl` and `merges_bpe_*.pkl` into `models/` as `vocab_bpe.pkl` and `merges_bpe.pkl` (and similarly for GPE if needed).
