"""
Simple webapp to compare how different tokenizers split a sentence.
Run: flask --app app run
"""
import sys
from pathlib import Path

# Ensure core modules are importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")

# ---------------------------------------------------------------------------
# Tokenizer registry: only BPE, GPE, Agathiyam (name -> encode(text) -> (tokens, ids))
# ---------------------------------------------------------------------------

# Exactly these three tokenizers are shown (order preserved)
TOKENIZER_NAMES = ["BPE", "GPE", "Agathiyam"]
TOKENIZERS = {}
LOAD_ERRORS = {}  # name -> optional hint (e.g. LFS message)

def _register(name, fn):
    TOKENIZERS[name] = fn

MODELS_DIR = ROOT / "models"

def _is_lfs_pointer(path: Path) -> bool:
    """True if file is a Git LFS pointer (not the actual binary)."""
    if not path.exists() or path.stat().st_size > 200:
        return False
    try:
        text = path.read_text(errors="ignore")
        return "git-lfs.github.com" in text and "oid sha256:" in text
    except Exception:
        return False

def _load_pickle_safe(path: Path):
    """Load pickle; return None if file is LFS pointer or invalid."""
    if _is_lfs_pointer(path):
        return None
    try:
        with open(path, "rb") as f:
            return __import__("pickle").load(f)
    except Exception:
        return None

LFS_HINT = "Model file is a Git LFS pointer. Run: git lfs pull"

def _load_tokenizers():
    """Load BPE, GPE, and Agathiyam (Sandhi-GPE). Each encode(text) -> (tokens, ids)."""
    # BPE
    try:
        from core.bpe import BPETokenizer
        v_path, m_path = MODELS_DIR / "vocab_bpe.pkl", MODELS_DIR / "merges_bpe.pkl"
        if v_path.exists() and m_path.exists():
            if _is_lfs_pointer(v_path) or _is_lfs_pointer(m_path):
                LOAD_ERRORS["BPE"] = LFS_HINT
            else:
                token_to_id = _load_pickle_safe(v_path)
                merges = _load_pickle_safe(m_path)
                if token_to_id is not None and merges is not None:
                    tok_bpe = BPETokenizer(token_to_id, merges)
                    _register("BPE", lambda t, _tok=tok_bpe: _tok.encode(t))
    except Exception:
        pass
    # GPE
    try:
        from core.gpe import GPETokenizer
        v_path, m_path = MODELS_DIR / "vocab_gpe.pkl", MODELS_DIR / "merges_gpe.pkl"
        if v_path.exists() and m_path.exists():
            if _is_lfs_pointer(v_path) or _is_lfs_pointer(m_path):
                LOAD_ERRORS["GPE"] = LFS_HINT
            else:
                token_to_id = _load_pickle_safe(v_path)
                merges = _load_pickle_safe(m_path)
                if token_to_id is not None and merges is not None:
                    tok_gpe = GPETokenizer(token_to_id, merges)
                    _register("GPE", lambda t, _tok=tok_gpe: _tok.encode(t))
    except Exception:
        pass
    # Agathiyam (Sandhi-GPE): preprocess with sandhi boundaries, then encode
    try:
        from core.gpe import GPETokenizer
        from core.sandhi import BOUND, _mark_mixed
        for v_name in ("vocab.pkl", "vocab_re.pkl"):
            v_path, m_path = MODELS_DIR / v_name, MODELS_DIR / "merges.pkl"
            if v_path.exists() and m_path.exists():
                if _is_lfs_pointer(v_path) or _is_lfs_pointer(m_path):
                    LOAD_ERRORS["Agathiyam"] = LFS_HINT
                else:
                    token_to_id = _load_pickle_safe(v_path)
                    merges = _load_pickle_safe(m_path)
                    if token_to_id is not None and merges is not None:
                        tok_ag = GPETokenizer(
                            token_to_id, merges, boundary_char=BOUND
                        )
                        def _agathiyam_encode(text, _tok=tok_ag):
                            marked = _mark_mixed(text)
                            n_boundaries = marked.count(BOUND)
                            tokens, ids = _tok.encode(marked)
                            # ⟂ is a structural boundary only — don't expose it as a token
                            filtered = [(t, i) for t, i in zip(tokens, ids) if t != BOUND]
                            tokens_out = [t for t, _ in filtered]
                            ids_out = [i for _, i in filtered]
                            return (tokens_out, ids_out, n_boundaries)
                        _register("Agathiyam", _agathiyam_encode)
                break
    except Exception:
        pass

_load_tokenizers()


@app.route("/")
def index():
    return render_template("tokenizer_compare.html", tokenizer_names=TOKENIZER_NAMES)


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = {}
    for name in TOKENIZER_NAMES:
        encode_fn = TOKENIZERS.get(name)
        if encode_fn is None:
            err = LOAD_ERRORS.get(name) or "Model not loaded"
            results[name] = {"tokens": [], "count": 0, "error": err}
            continue
        try:
            out = encode_fn(text)
            tokens = out[0] if isinstance(out[0], list) else list(out[0])
            results[name] = {"tokens": tokens, "count": len(tokens)}
            if name == "Agathiyam" and len(out) >= 3 and isinstance(out[2], int):
                results[name]["sandhi_boundaries"] = out[2]
        except Exception as e:
            results[name] = {"tokens": [], "count": 0, "error": str(e)}


    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=10010)
