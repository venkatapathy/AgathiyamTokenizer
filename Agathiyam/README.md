
Companion repository for the **“Agathiyam – Sandhi-aware tokenization for Tamil Language.”**

This repository collects the **corpora, tokenizers, evaluation scripts, and analysis artifacts** that underpin the paper’s exploration of Tamil and code-mixed tokenization.

---

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research--prototype-yellow)
![License](https://img.shields.io/badge/license-pending-lightgrey)

---

## 🧩 Overview

* Provides **sandhi-aware**, **grapheme-aware**, and **byte-level** tokenizers tailored to Tamil and Tamil–English code-mixed text.
* Bundles curated datasets (`Agathiyam-Tamil`, `flores`, Samanantar-derived splits) and ready-to-use tokenizer checkpoints.
* Includes **evaluation pipelines** for compression, fertility, coverage, and downstream Gemma-270M fine-tuning experiments.
* Documents extensive **sandhi-rule engineering and testing**, with coverage dashboards and analysis notebooks.

---

## 📁 Repository Layout

| Directory / File                             | Description                                                                                                           |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `GPE/`                                       | Core tokenizer implementations (`bpe.py`, `gpe.py`, `GPE_sandhi.py`), sandhi utilities, evaluation scripts, and tests |
| `data/`                                      | Lightweight text corpora for quick experimentation (`train.txt`, `test.txt`, cached tokenizer dumps)                  |
| `Agathiyam-Tamil/`                           | Lexical resources derived from classical Tamil grammar used to seed sandhi rules                                      |
| `Pretokenization/`, `Byte_level_tokenizers/` | Jupyter notebooks and CSV summaries for reproducibility                                                               |
| `coverage_html/`                             | HTML coverage dashboard produced by `coverage.py`                                                                     |
| `results_gemma*`, `gemma270m_regimeB*`       | Checkpoints, configs, and metrics for Gemma-3 270M experiments                                                        |
| `*.csv`, `*.png`, `*_REPORT.md`              | Metric dumps, charts, and narrative reports referenced in the paper                                                   |

---

## ⚙️ Getting Started

### 1. Clone / Extract

```bash
git clone https://github.com/RoshiniPriya05/Agathiyam-Tamil.git
cd Agathiyam-Tamil
```

### 2. Create a Python Environment (≥3.10 recommended)

```bash
python -m venv .venv

# Windows
.\\.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install regex grapheme datasets tqdm pandas numpy matplotlib \
            torch transformers accelerate huggingface-hub
```

**Optional (for notebooks, linting, and coverage reports):**

```bash
pip install jupyter pytest pytest-cov coverage
```

### 4. Environment Setup

* Add the repository root (or `GPE/`) to your `PYTHONPATH` if you plan to run modules with `python -m`.
* Hugging Face datasets are downloaded on first use; ensure `HF_HOME` points to a location with sufficient space.

**Tested on:** Python 3.10–3.12, Ubuntu 22.04, and Windows 11.

---

## 🗂️ Datasets

| Dataset                    | Source                                | Usage                   | Location           |
| -------------------------- | ------------------------------------- | ----------------------- | ------------------ |
| Samanantar (Tamil–English) | Hugging Face (`ai4bharat/samanantar`) | Parallel text           | dynamic / cached   |
| Agathiyam–Tamil            | Curated lexical set                   | Sandhi rule induction   | `Agathiyam-Tamil/` |
| Flores                     | Hugging Face                          | Multilingual evaluation | `flores/`          |
| Supplementary corpora      | Local                                 | Quick experiments       | `data/`            |

> ⚠️ Large Samanantar-derived corpora are not checked into Git.
> Configure absolute paths in scripts (see `corpus_file`, `checkpoint_path`, or `--path` CLI flag).

---

## 🧠 Training Tokenizers

### 1. Baseline BPE (`GPE/bpe.py`)

```bash
cd GPE
python bpe.py
```

Outputs: learned vocabulary and merge rules (pickle format).

---

### 2. Grapheme-level Tokenizer (`GPE/gpe.py`)

```bash
cd GPE
python gpe.py
```

Outputs: grapheme vocabulary and merge rules.

---

### 3. Sandhi-aware Grapheme BPE (`GPE/GPE_sandhi.py`)

```bash
python -m GPE.GPE_sandhi
```

**Key parameters:**

* `lang`: `"mix"` for Tamil–English code-mixed handling (default)
* `checkpoint_path`: optional pickle to resume long runs
* `base`: output directory for saved model

Outputs: sandhi-aware vocabulary and merge files, optionally checkpointed mid-run.

---

## 📊 Boundary & Coverage Utilities

```bash
python -m GPE.run_samanantar_coverage --limit 50000 --path /path/to/corpus.txt
```

Generates boundary counts and split statistics.
Additional analyses via `sandhi_rules_analysis.py` and `enhanced_sandhi_analysis.py`.

---

## 📈 Evaluation & Metrics

### Compression / Fertility

```bash
python -m GPE.compare_tokenizers
```

Produces `evaluation_results.csv` and console summaries.

### Coverage Dashboards

Generated with `coverage.py` and `pytest` → see `coverage_html/`, `COVERAGE_REPORT.md`, `COVERAGE_SUMMARY_REPORT.md`.

### Distribution Reports

Files such as:

* `sandhi_distribution.csv` / `.png`
* `sandhi_rule_frequencies.txt`
* `SANDHI_TOKENIZATION_ANALYSIS_REPORT.md`

### Pretokenization Studies

CSV results in:

* `Pretokenization/results/`
* `Byte_level_tokenizers/results/`

These reproduce compression and parity metrics from the paper.

---

## 🤖 Language Model Experiments

* `lightweight_pretrain*.py` — fine-tune small GPT-style models (e.g., Gemma-3 270M) with custom tokenizers.
* Configure `MODEL_ID`, `train_file`, and tokenizer paths.
* Outputs under `results/`, `gemma270m_regimeB*/`, or `results_gemma*/`.

Utilities:

* `initialize_embeddings.py` and `oov.py` align tokenizer vocabularies with pretrained embeddings for Tamil tokens.

---

## 🧪 Testing & Coverage

```bash
pytest GPE/test_*.py -v
coverage run --source=GPE -m pytest GPE/test_*.py
coverage report -m
coverage html  # view report in coverage_html/
```

Key test modules:

* `test_coverage_analysis.py`
* `test_reverse_sandhi_check.py`
* `test_summarizer.py`

---

## 📓 Notebooks

Open notebooks under `Pretokenization/` or `Byte_level_tokenizers/` to reproduce plots:

```bash
jupyter notebook Pretokenization/testing_FLORES200_Multi-lingual.ipynb
```

---

## 🤝 Contributing

We welcome pull requests that:

* Improve sandhi coverage or add new Tamil dialectal rules.
* Extend evaluation scripts to other Indic languages.
* Add reproducible test suites.

Please open an issue to discuss major changes before submitting a PR.

---

## ⚖️ License

A license has not been formally selected yet.
Until finalized, this repository is shared **for academic and research evaluation only**.
Please contact the authors for reuse or redistribution beyond this purpose.

---

