

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    import networkx as nx

# ---------------------------------------------------------------------------
# Colab dependency installation (run once in Colab)
# ---------------------------------------------------------------------------

def install_dependencies():
    """Install required packages for Google Colab. Run this cell first."""
    import subprocess
    import sys
    packages = [
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "networkx",
        "pyvis",
        "numpy",
        "requests",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    except Exception:
        pass  # Gradio optional
    print("Dependencies installed.")


# ---------------------------------------------------------------------------
# 1) Classification Module — Human / GPT-3.5 / GPT-4 / GPT-5 via OpenRouter
# ---------------------------------------------------------------------------

# Final output classes (no unknown class in final prediction).
CLASS_LABELS = ["human", "gpt3.5", "gpt4", "gpt5"]
# Real-world display names for classroom output (GPT-3.5 Turbo, GPT-4, GPT-5)
LABEL_DISPLAY_NAMES = {
    "human": "Human",
    "gpt3.5": "GPT-3.5 Turbo",
    "gpt4": "GPT-4",
    "gpt5": "GPT-5",
}


def display_label(label: str) -> str:
    """Return real-world display name for a classification label (e.g. gpt3.5 -> GPT-3.5 Turbo)."""
    return LABEL_DISPLAY_NAMES.get(label, label)

# OpenRouter: real GPT models for Min-K. Set env OPENROUTER_API_KEY (recommended) or put key here.
OPENROUTER_API_KEY = "PUT YOUR API KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/completions"
# Map OpenRouter model id -> our label. GPT only: 3.5 Turbo -> gpt3.5, GPT-4 -> gpt4, GPT-4o/GPT-5 -> gpt5.
# Attribution is over GPT-3.5 Turbo, GPT-4, GPT-5 (plus Human).
OPENROUTER_MODEL_TO_LABEL = {
    "openai/gpt-3.5-turbo": "gpt3.5",
    "openai/gpt-3.5-turbo-instruct": "gpt3.5",
    "openai/gpt-3.5-turbo-0125": "gpt3.5",
    "openai/gpt-4": "gpt4",
    "openai/gpt-4-turbo": "gpt4",
    "openai/gpt-4-turbo-preview": "gpt4",
    "openai/gpt-4o-mini": "gpt4",
    "openai/gpt-4o": "gpt5",
    "openai/gpt-4o-2024-08-06": "gpt5",
    "openai/gpt-4o-2024-11-20": "gpt5",
    "openai/gpt-5-pro": "gpt5",
    "openai/gpt-5-pro-2025-10-06": "gpt5",
}


def _resolve_openrouter_label(model_id: str | None) -> str | None:
    """
    Resolve OpenRouter model id to one of {gpt3.5, gpt4, gpt5}.
    Returns None when label cannot be resolved reliably.
    """
    if not model_id:
        return None
    mid = str(model_id).strip().lower()
    if not mid:
        return None
    if mid in OPENROUTER_MODEL_TO_LABEL:
        return OPENROUTER_MODEL_TO_LABEL[mid]
    # Heuristic fallback for versioned IDs not explicitly listed.
    if "gpt-5" in mid or "gpt5" in mid:
        return "gpt5"
    if "gpt-4" in mid or "gpt4" in mid:
        return "gpt4"
    if "gpt-3.5" in mid or "gpt3.5" in mid or "gpt-35" in mid or "gpt35" in mid:
        return "gpt3.5"
    return None
# OpenRouter only: GPT-3.5 Turbo, GPT-4, GPT-5 (no Hugging Face).
OPENROUTER_DEFAULT_MODELS = [
    "openai/gpt-3.5-turbo-instruct",  # GPT-3.5 Turbo
    "openai/gpt-4o-mini",             # GPT-4
    "openai/gpt-4o",                  # GPT-4o / GPT-5 tier
    "openai/gpt-5-pro",               # GPT-5 Pro (remove if not on your account)
]
# Min-K human/LLM decision: DEFAULT TO LLM to avoid GPT-5 (and any LLM) code labeled as human.
# Only classify as human if best model's Min-K score is STRICTLY ABOVE this (positive = very surprising code).
MIN_K_HUMAN_THRESHOLD = -3.0
MIN_K_HUMAN_THRESHOLD_GPT5 = -2.6   # slightly stricter for GPT-5 winner
# If best score is <= this, NEVER say human. Set to 0 so any non-positive score → always attribute to best model.
MIN_K_NEVER_HUMAN_BELOW = 0.0
# Clear-winner rule: if best model is this much lower (more negative) than second-best, always attribute to that model
MIN_K_CLEAR_WINNER_GAP = 0.2
# Optional: set OPENROUTER_EXTRA_MODELS / MIN_K_HUMAN_THRESHOLD / MIN_K_NEVER_HUMAN_BELOW via env to tune


def _get_openrouter_key(api_key: str | None = None) -> str | None:
    """Resolve OpenRouter API key: argument, then env, then OPENROUTER_API_KEY constant. Treat placeholder as None."""
    import os
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not key or (isinstance(key, str) and key.strip() in ("", "PUT_API", "PUT YOUR API KEY")):
        return None
    return key.strip()


# Legacy (only for load_classifier_model/run_classifier/refine if called directly)
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.35
HUMAN_TIE_MARGIN = 0.03
MIN_K_LEAN_LLM = -8.0
MIN_K_LEAN_HUMAN = -6.0

@dataclass
class ClassificationResult:
    """Output of the code classifier."""
    predicted_label: str
    probability_distribution: dict[str, float]  # softmax over CLASS_LABELS
    confidence: float  # probability of predicted_label
    is_llm_generated: bool  # True if predicted_label != "human"
    suggested_model: str | None  # None if human, else e.g. "gpt4"
    decision_source: str = "unspecified"  # e.g. openrouter_min_k, openrouter_chat_fallback
    decision_notes: str | None = None


def _get_device():
    import torch
    return torch.device("cuda" if __import__("torch").cuda.is_available() else "cpu")


def load_classifier_model(model_name: str = "microsoft/codebert-base"):
    """
    Load a transformer model and a 5-way classification head.
    Uses CodeBERT (or RoBERTa) [CLS] embedding + linear layer.
    Head is randomly initialized for prototype; replace with fine-tuned weights for real use.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = backbone.config.hidden_size
    num_labels = len(CLASS_LABELS)
    head = torch.nn.Linear(hidden_size, num_labels)
    # Optional: load state dict from checkpoint here for fine-tuned model
    backbone.to(device)
    head.to(device)
    return {
        "tokenizer": tokenizer,
        "backbone": backbone,
        "head": head,
        "device": device,
        "model_name": model_name,
    }


def run_classifier(code_snippet: str, model_dict: dict[str, Any]) -> ClassificationResult:
    """
    Run multi-class classification on a code snippet.
    Returns predicted label, softmax distribution, confidence, and LLM flag.
    """
    import torch
    import torch.nn.functional as F

    tokenizer = model_dict["tokenizer"]
    backbone = model_dict["backbone"]
    head = model_dict["head"]
    device = model_dict["device"]
    max_length = 512

    backbone.eval()
    head.eval()
    encoding = tokenizer(
        code_snippet,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = backbone(**encoding)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = head(cls_embedding)[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
    pred_idx = int(probs.argmax())
    predicted_label = CLASS_LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    human_prob = prob_dict["human"]

    # Combine classifier with thresholds; Min-K refinement will correct when uncertain.
    # Only default to human when classifier is clearly uncertain (low max prob).
    if confidence < CLASSIFIER_CONFIDENCE_THRESHOLD:
        predicted_label = "human"
        pred_idx = CLASS_LABELS.index("human")
        confidence = human_prob
        is_llm = False
        suggested_model = None
    # Only prefer human in a true tie (human within 3% of top); else trust LLM and let Min-K refine.
    elif predicted_label != "human" and human_prob >= confidence - HUMAN_TIE_MARGIN:
        predicted_label = "human"
        pred_idx = CLASS_LABELS.index("human")
        confidence = human_prob
        is_llm = False
        suggested_model = None
    else:
        is_llm = predicted_label != "human"
        suggested_model = predicted_label if is_llm else None

    return ClassificationResult(
        predicted_label=predicted_label,
        probability_distribution=prob_dict,
        confidence=confidence,
        is_llm_generated=is_llm,
        suggested_model=suggested_model,
    )


def refine_classification_with_min_k(
    classification_result: ClassificationResult,
    min_k_result: MinKResult,
) -> ClassificationResult:
    """
    Refine human vs LLM using Min-K: code that looks "familiar" to the LM (low Min-K score)
    leans toward LLM; code that looks "surprising" (high Min-K score) leans toward human.
    """
    prob_dict = classification_result.probability_distribution
    raw_label = classification_result.predicted_label
    raw_conf = classification_result.confidence
    score = min_k_result.min_k_score

    # Classifier was confident LLM (>= threshold) but Min-K says code is surprising → human
    if raw_label != "human" and raw_conf >= CLASSIFIER_CONFIDENCE_THRESHOLD and score > MIN_K_LEAN_HUMAN:
        return ClassificationResult(
            predicted_label="human",
            probability_distribution=prob_dict,
            confidence=prob_dict["human"],
            is_llm_generated=False,
            suggested_model=None,
        )
    # Classifier was uncertain (we defaulted to human) but Min-K says code is very familiar → LLM
    if raw_label == "human" and score < MIN_K_LEAN_LLM:
        llm_labels = [c for c in CLASS_LABELS if c != "human"]
        best_llm = max(llm_labels, key=lambda c: prob_dict[c])
        return ClassificationResult(
            predicted_label=best_llm,
            probability_distribution=prob_dict,
            confidence=prob_dict[best_llm],
            is_llm_generated=True,
            suggested_model=best_llm,
        )
    return classification_result


# ---------------------------------------------------------------------------
# 2) Repository Similarity Module (with real API calls for provenance)
# ---------------------------------------------------------------------------

REPO_DISCLAIMER = (
    "⚠️ Repository similarity is similarity-based inference only. "
    "It does NOT constitute proof that this code was in the model's training data."
)

# Fallback when APIs are unavailable or rate-limited
SAMPLE_GITHUB = [
    "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
    "import requests; r = requests.get(url); return r.json()",
    "class Logger:\n    def log(self, msg): print(msg)",
    "df = pd.read_csv(path); return df.dropna()",
    "with open(fname) as f: return json.load(f)",
]
SAMPLE_STACKOVERFLOW = [
    "try:\n    result = int(s)\nexcept ValueError:\n    result = 0",
    "for i, x in enumerate(lst): print(i, x)",
    "sorted(items, key=lambda x: x[1])",
    "if not os.path.exists(path): os.makedirs(path)",
    "re.sub(r'\\s+', ' ', text)",
]
SAMPLE_OTHER_WEB = [
    "def hello(): print('Hello World')",
    "x = [1, 2, 3]; y = sum(x)",
    "return {k: v for k, v in d.items() if v}",
    "assert isinstance(x, str)",
    "logging.info('message')",
]


def _code_keyword_candidates(code_snippet: str) -> list[str]:
    """
    Extract meaningful query keywords from code (imports, classes, API calls, identifiers).
    This avoids first-line-only queries that often return unrelated posts.
    """
    import re

    stop = {
        "import", "from", "as", "def", "class", "return", "for", "while", "if", "elif", "else",
        "try", "except", "with", "in", "and", "or", "not", "is", "none", "true", "false",
        "python", "code", "model", "data", "train", "test", "fit", "predict", "print",
    }
    lines = [ln.strip() for ln in code_snippet.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    out: list[str] = []

    # 1) Imported modules/symbols (high value)
    for ln in lines:
        m_from = re.match(r"from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.+)", ln)
        if m_from:
            mod = m_from.group(1).strip()
            if mod and mod.lower() not in stop:
                out.append(mod)
            syms = [s.strip() for s in re.split(r",|\s+", m_from.group(2)) if s.strip() and s.strip() != "as"]
            for s in syms[:4]:
                if re.match(r"[A-Za-z_][A-Za-z0-9_]*$", s) and s.lower() not in stop:
                    out.append(s)
            continue
        m_imp = re.match(r"import\s+(.+)", ln)
        if m_imp:
            mods = [s.strip() for s in m_imp.group(1).split(",")]
            for s in mods[:4]:
                s = s.split(" as ")[0].strip()
                if s and s.lower() not in stop:
                    out.append(s)

    # 2) Class-like identifiers and important API calls
    for tok in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", joined):
        if tok.lower() not in stop:
            out.append(tok)
    for fn in re.findall(r"\b([a-zA-Z_][A-Za-z0-9_]*)\s*\(", joined):
        if fn.lower() not in stop and len(fn) > 2:
            out.append(fn)

    # 3) Key python/sklearn markers
    markers = ["sklearn", "numpy", "pandas", "scipy", "matplotlib", "xgboost", "pytorch", "tensorflow"]
    low = joined.lower()
    for m in markers:
        if m in low:
            out.append(m)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for x in out:
        x2 = x.strip()
        if not x2:
            continue
        k = x2.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x2)
    return uniq


def _search_query_from_code(code_snippet: str, max_words: int = 9) -> str:
    """Build a focused search query from code keywords for API calls."""
    kws = _code_keyword_candidates(code_snippet)[:max_words]
    return " ".join(kws) if kws else "python sklearn model"


def _snippet_overlap_score(code_snippet: str, snippet: str) -> float:
    """Simple lexical overlap score between code keywords and snippet tokens."""
    import re
    kw = {k.lower() for k in _code_keyword_candidates(code_snippet)}
    st = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", snippet)}
    if not kw or not st:
        return 0.0
    inter = len(kw.intersection(st))
    return inter / max(1, len(kw))


def _matched_tokens(code_snippet: str, snippet: str, max_items: int = 12) -> list[str]:
    """Return shared keyword tokens between submitted code and retrieved snippet."""
    import re
    kw = {k.lower() for k in _code_keyword_candidates(code_snippet)}
    st = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", snippet)}
    mt = sorted(kw.intersection(st))
    return mt[:max_items]


def _snippet_preview_with_matches(code_snippet: str, snippet: str, max_len: int = 240) -> tuple[str, list[str], float]:
    """
    Build a compact snippet preview centered around matched tokens.
    Returns (preview, matched_tokens, overlap_score).
    """
    import re
    matched = _matched_tokens(code_snippet, snippet, max_items=12)
    overlap = _snippet_overlap_score(code_snippet, snippet)
    text = re.sub(r"\s+", " ", snippet).strip()
    if not text:
        return "", matched, overlap
    if matched:
        # center preview around the first matched token
        tok = matched[0]
        idx = text.lower().find(tok.lower())
        if idx >= 0:
            start = max(0, idx - max_len // 3)
            end = min(len(text), start + max_len)
            prev = text[start:end]
            if start > 0:
                prev = "..." + prev
            if end < len(text):
                prev = prev + "..."
            return prev, matched, overlap
    # fallback to head preview
    return (text[:max_len] + ("..." if len(text) > max_len else "")), matched, overlap


def fetch_github_code_search(query: str, token: str | None = None, per_page: int = 10) -> list[tuple[str, str]]:
    """
    Call GitHub Search API for code. Returns list of (snippet_text, source_label).
    Set env GITHUB_TOKEN for higher rate limits. Requires 'requests'.
    """
    import os
    import re
    try:
        import requests
    except ImportError:
        return []
    token = token or os.environ.get("GITHUB_TOKEN")
    # GitHub code search: q must include language:python, and we need a short query
    q = f"{query} language:python"[:256]
    url = "https://api.github.com/search/code"
    headers = {"Accept": "application/vnd.github.v3.text-match+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"q": q, "per_page": min(per_page, 10)}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("items", [])
        out = []
        for item in items:
            added = False
            for tm in item.get("text_matches", [])[:1]:
                frag = (tm.get("fragment") or "").strip()
                frag = re.sub(r"\s+", " ", frag)[:500]
                if frag:
                    out.append((frag, "github"))
                    added = True
                    break
            if not added:
                path = item.get("path") or item.get("name") or ""
                if path:
                    out.append((f"{path} (GitHub)", "github"))
        return out[:per_page]
    except Exception:
        return []


def fetch_stackexchange_search(query: str, site: str = "stackoverflow", page_size: int = 5) -> list[tuple[str, str]]:
    """
    Call Stack Exchange API. Returns list of (snippet_text, source_label).
    site: 'stackoverflow' or 'codereview.stackexchange' etc.
    """
    try:
        import requests
        import re
    except ImportError:
        return []
    label = "stackoverflow" if "stackoverflow" in site else "other_web"
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "tagged": "python",
        "q": query[:200],
        "site": site,
        "pagesize": min(page_size, 10),
        "filter": "withbody",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("items", [])
        out = []
        for item in items:
            body = (item.get("body") or "")
            # Strip HTML tags for snippet
            body = re.sub(r"<[^>]+>", " ", body).strip()
            body = re.sub(r"\s+", " ", body)[:400]
            if body:
                out.append((body, label))
        return out
    except Exception:
        return []


def fetch_repository_snippets_via_api(
    code_snippet: str,
    github_token: str | None = None,
    use_fallback_if_empty: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Fetch code snippets from GitHub and Stack Exchange APIs using a query derived from user code.
    Returns (list of snippet strings, list of source labels: 'github'|'stackoverflow'|'other_web').
    If APIs fail or return nothing, falls back to simulated samples when use_fallback_if_empty=True.
    """
    query = _search_query_from_code(code_snippet)
    all_snippets: list[str] = []
    all_sources: list[str] = []

    gh = fetch_github_code_search(query, token=github_token, per_page=5)
    for snip, src in gh:
        all_snippets.append(snip)
        all_sources.append(src)

    so = fetch_stackexchange_search(query, site="stackoverflow", page_size=5)
    for snip, src in so:
        all_snippets.append(snip)
        all_sources.append(src)

    other = fetch_stackexchange_search(query, site="codereview.stackexchange.com", page_size=5)
    for snip, src in other:
        all_snippets.append(snip)
        all_sources.append(src)

    # Relevance filter: keep snippets that have lexical overlap with code keywords.
    # This reduces unrelated generic posts in final similarity output.
    if all_snippets:
        filtered_snippets: list[str] = []
        filtered_sources: list[str] = []
        for snip, src in zip(all_snippets, all_sources):
            overlap = _snippet_overlap_score(code_snippet, snip)
            if overlap >= 0.12:
                filtered_snippets.append(snip)
                filtered_sources.append(src)
        # If filter is too strict, keep original to avoid empty retrieval.
        if filtered_snippets:
            all_snippets, all_sources = filtered_snippets, filtered_sources

    if use_fallback_if_empty and not all_snippets:
        for s in SAMPLE_GITHUB:
            all_snippets.append(s)
            all_sources.append("github")
        for s in SAMPLE_STACKOVERFLOW:
            all_snippets.append(s)
            all_sources.append("stackoverflow")
        for s in SAMPLE_OTHER_WEB:
            all_snippets.append(s)
            all_sources.append("other_web")

    return all_snippets, all_sources


@dataclass
class SimilarityHit:
    source: str  # "github" | "stackoverflow" | "other_web"
    snippet: str
    cosine_similarity: float
    rank: int


def load_embedding_model():
    """Load sentence-transformers model for code/text embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


def get_embeddings(model, texts: list[str]):
    """Compute embeddings for a list of strings."""
    return model.encode(texts, show_progress_bar=False)


def build_repository_embeddings(embedding_model, all_snippets: list[str], all_sources: list[str]):
    """Build embeddings for a list of (snippet, source) from API or fallback."""
    if not all_snippets:
        all_snippets = SAMPLE_GITHUB + SAMPLE_STACKOVERFLOW + SAMPLE_OTHER_WEB
        all_sources = ["github"] * len(SAMPLE_GITHUB) + ["stackoverflow"] * len(SAMPLE_STACKOVERFLOW) + ["other_web"] * len(SAMPLE_OTHER_WEB)
    embeddings = get_embeddings(embedding_model, all_snippets)
    return embeddings, all_snippets, all_sources


def repository_similarity(
    code_snippet: str,
    embedding_model,
    top_k: int = 5,
    repo_embeddings=None,
    repo_snippets=None,
    repo_sources=None,
    use_api: bool = True,
    github_token: str | None = None,
) -> tuple[list[SimilarityHit], str]:
    """
    Compare user code to repository samples via cosine similarity.
    If use_api=True, fetches snippets from GitHub and Stack Exchange APIs first.
    Returns top-k similar samples and disclaimer text.
    """
    import numpy as np

    if use_api or repo_embeddings is None or repo_snippets is None or repo_sources is None:
        snippets, sources = fetch_repository_snippets_via_api(code_snippet, github_token=github_token, use_fallback_if_empty=True)
        repo_embeddings, repo_snippets, repo_sources = build_repository_embeddings(embedding_model, snippets, sources)

    query_emb = get_embeddings(embedding_model, [code_snippet])[0]
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    repo_norms = repo_embeddings / (np.linalg.norm(repo_embeddings, axis=1, keepdims=True) + 1e-9)
    cosines = np.dot(repo_norms, query_norm)
    order = np.argsort(-cosines)
    hits = []
    rank = 1
    min_cos = 0.20
    for idx in order:
        if len(hits) >= top_k:
            break
        score = float(cosines[idx])
        snip = repo_snippets[idx]
        overlap = _snippet_overlap_score(code_snippet, snip)
        # Keep high cosine results, or moderate cosine with good lexical overlap.
        if score < min_cos and not (score >= 0.14 and overlap >= 0.20):
            continue
        hits.append(SimilarityHit(
            source=repo_sources[idx],
            snippet=snip,
            cosine_similarity=score,
            rank=rank,
        ))
        rank += 1
    # If all were filtered out, keep the single best to avoid empty output.
    if not hits and len(order) > 0:
        idx = int(order[0])
        snip = repo_snippets[idx]
        hits.append(SimilarityHit(
            source=repo_sources[idx],
            snippet=snip,
            cosine_similarity=float(cosines[idx]),
            rank=1,
        ))
    return hits, REPO_DISCLAIMER


# ---------------------------------------------------------------------------
# 3) Min-K% Membership Inference Module
# ---------------------------------------------------------------------------

@dataclass
class MinKResult:
    min_k_score: float
    membership_probability: float
    decision_threshold: float
    k_percent: float
    num_tokens_used: int
    explanation: str


def run_min_k_membership(
    code_snippet: str,
    k_percent: float = 0.1,
    tau: float | None = None,
    lm_model_id: str = "gpt2",
    max_length: int = 128,
) -> MinKResult:
    """
    Min-K% membership inference:
    1. Compute token log-probabilities with the language model.
    2. Sort log-probabilities and take the bottom k%.
    3. Average those → Min-K score.
    4. Compare to threshold τ to estimate membership probability.
    """
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
    model = AutoModelForCausalLM.from_pretrained(lm_model_id)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        code_snippet,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"][0]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)

    log_probs_list = []
    for j in range(len(input_ids) - 1):
        token_id = input_ids[j + 1].item()
        lp = log_probs[0, j, token_id].item()
        if math.isfinite(lp):
            log_probs_list.append(lp)
        else:
            log_probs_list.append(-20.0)

    if not log_probs_list:
        min_k_score = -10.0
        num_tokens = 0
    else:
        sorted_lp = sorted(log_probs_list)
        k = max(1, int(len(sorted_lp) * k_percent))
        lowest_k = sorted_lp[:k]
        min_k_score = float(np.mean(lowest_k))
        num_tokens = len(log_probs_list)

    # Default threshold: heuristic (e.g. -8). Lower (more negative) min_k_score → more likely "member".
    if tau is None:
        tau = -8.0
    # Membership probability: sigmoid over (min_k_score - tau); higher score (less negative) → lower membership prob
    # Convention: more negative min_k → more likely training member. So membership_prob = sigmoid(-(min_k_score - tau))
    membership_prob = 1.0 / (1.0 + math.exp(-(-(min_k_score - tau))))

    explanation = (
        f"Min-K% analysis (k={k_percent*100:.0f}%, τ={tau}): "
        f"Min-K score = {min_k_score:.4f} (mean of lowest {k_percent*100:.0f}% token log-probs). "
        f"Membership probability = {membership_prob:.2%} (threshold τ={tau}). "
        f"Based on {num_tokens} tokens."
    )
    return MinKResult(
        min_k_score=min_k_score,
        membership_probability=membership_prob,
        decision_threshold=tau,
        k_percent=k_percent,
        num_tokens_used=num_tokens,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# 3b) OpenRouter Min-K (real models via API for classification)
# ---------------------------------------------------------------------------

def _openrouter_completion_logprobs(
    code_snippet: str,
    model_id: str,
    api_key: str,
    max_tokens: int = 256,
) -> list[float] | None:
    """
    Call OpenRouter completions API; return list of token log-probabilities for Min-K.
    Uses prompt=code and requests logprobs for generated tokens (or echo if supported).
    """
    import os
    try:
        import requests
    except ImportError:
        return None
    key = _get_openrouter_key(api_key)
    if not key:
        return None
    url = os.environ.get("OPENROUTER_API_URL", OPENROUTER_API_URL)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/deem-mink",
    }
    # Request logprobs for generated tokens (prompt=code, model completes; we use those logprobs for Min-K)
    payload = {
        "model": model_id,
        "prompt": code_snippet[:4000],
        "max_tokens": max(64, min(max_tokens, 256)),
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        logprobs = choice.get("logprobs")
        if not logprobs:
            return None
        out: list[float] = []
        # 1) OpenAI-style: token_logprobs list
        token_logprobs = logprobs.get("token_logprobs")
        if token_logprobs is not None:
            for lp in token_logprobs:
                if lp is None:
                    continue
                try:
                    x = float(lp)
                    out.append(x if math.isfinite(x) else -20.0)
                except (TypeError, ValueError):
                    out.append(-20.0)
            return out if out else None
        # 2) content array with per-token dicts (e.g. {"token": "...", "logprob": -0.1})
        content = logprobs.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "logprob" in item:
                    try:
                        x = float(item["logprob"])
                        out.append(x if math.isfinite(x) else -20.0)
                    except (TypeError, ValueError):
                        out.append(-20.0)
            return out if out else None
        # 3) top_logprobs: list of dicts like [{"token": "x", "logprob": -0.1}, ...]; take first (top) logprob per position
        if isinstance(logprobs, list):
            for elem in logprobs:
                if isinstance(elem, dict) and "logprob" in elem:
                    try:
                        x = float(elem["logprob"])
                        out.append(x if math.isfinite(x) else -20.0)
                    except (TypeError, ValueError):
                        out.append(-20.0)
            return out if out else None
        return None
    except Exception:
        return None


def _openrouter_chat_logprobs(
    code_snippet: str,
    model_id: str,
    api_key: str,
    max_tokens: int = 256,
) -> list[float] | None:
    """Try OpenRouter chat/completions with logprobs; some models only expose logprobs here."""
    import os
    try:
        import requests
    except ImportError:
        return None
    key = _get_openrouter_key(api_key)
    if not key:
        return None
    base = (os.environ.get("OPENROUTER_API_URL") or OPENROUTER_API_URL).rstrip("/").rsplit("/", 1)[0]
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/deem-mink",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": code_snippet[:4000]}],
        "max_tokens": max(64, min(max_tokens, 256)),
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        logprobs = choice.get("logprobs")
        if not logprobs:
            return None
        out: list[float] = []
        token_logprobs = logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
        if token_logprobs is not None:
            for lp in token_logprobs:
                if lp is None:
                    continue
                try:
                    x = float(lp)
                    out.append(x if math.isfinite(x) else -20.0)
                except (TypeError, ValueError):
                    out.append(-20.0)
            return out if out else None
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "logprob" in item:
                    try:
                        x = float(item["logprob"])
                        out.append(x if math.isfinite(x) else -20.0)
                    except (TypeError, ValueError):
                        out.append(-20.0)
            return out if out else None
        return None
    except Exception:
        return None


# Model used for fallback classification when Min-K logprobs are not returned by OpenRouter
OPENROUTER_FALLBACK_CLASSIFIER_MODEL = "openai/gpt-4o"
OPENROUTER_FALLBACK_JUDGE_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-5-pro",
]


def _classify_by_openrouter_chat(
    code_snippet: str,
    api_key: str,
    model_id: str | None = None,
) -> ClassificationResult | None:
    """
    Fallback: use OpenRouter chat to ask a model to classify code as human / gpt3.5 / gpt4 / gpt5.
    Used when Min-K logprobs are unavailable. Does not require logprobs in the response.
    """
    import os
    import re
    try:
        import requests
    except ImportError:
        return None
    key = _get_openrouter_key(api_key)
    if not key:
        return None
    model = model_id or os.environ.get("OPENROUTER_FALLBACK_CLASSIFIER_MODEL", OPENROUTER_FALLBACK_CLASSIFIER_MODEL)
    base = (os.environ.get("OPENROUTER_API_URL") or OPENROUTER_API_URL).rstrip("/").rsplit("/", 1)[0]
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/deem-mink",
    }
    code_preview = code_snippet[:3000].strip()
    if len(code_snippet) > 3000:
        code_preview += "\n..."
    prompt = (
        "You are a strict code-attribution classifier for classroom integrity checks.\n"
        "Task: classify the code as one label from this exact set: human, gpt3.5, gpt4, gpt5.\n"
        "Policy: choose the most likely label based on evidence; do not force a non-human answer.\n"
        "Return JSON only with keys: label, confidence, rationale.\n"
        "Example: {\"label\":\"gpt5\",\"confidence\":0.82,\"rationale\":\"brief\"}\n\n"
        "Code:\n" + code_preview
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        raw = (choices[0].get("message") or {}).get("content") or choices[0].get("text") or ""
        raw_lower = raw.strip().lower()
        label = None
        conf = 0.75
        rationale = "chat fallback"
        # Try JSON-first parse
        try:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            payload_txt = m.group(0) if m else raw
            parsed = json.loads(payload_txt)
            lbl_raw = str(parsed.get("label", "")).strip().lower()
            lbl_raw = re.sub(r"[-\s]+", "", lbl_raw)
            if lbl_raw in ("human", "gpt35", "gpt3.5", "gpt4", "gpt5"):
                label = "gpt3.5" if lbl_raw in ("gpt35", "gpt3.5") else lbl_raw
            conf = float(parsed.get("confidence", conf))
            conf = max(0.0, min(1.0, conf))
            rationale = str(parsed.get("rationale", rationale))[:240]
        except Exception:
            pass
        if label is None:
            txt = re.sub(r"[-\s]+", "", raw_lower)
            if "gpt5" in txt:
                label = "gpt5"
            elif "gpt4" in txt:
                label = "gpt4"
            elif "gpt35" in txt or "gpt3" in txt or "3.5" in raw_lower:
                label = "gpt3.5"
            elif txt == "human" or raw_lower.startswith("human"):
                # only accept explicit human, not substring ambiguity
                label = "human"
            else:
                # Do not inject GPT-4 bias on malformed judge output.
                return None
        prob_dict = {c: 0.0 for c in CLASS_LABELS}
        prob_dict[label] = max(0.6, conf)
        for c in CLASS_LABELS:
            if c != label:
                prob_dict[c] = (1.0 - prob_dict[label]) / (len(CLASS_LABELS) - 1) if len(CLASS_LABELS) > 1 else 0.0
        return ClassificationResult(
            predicted_label=label,
            probability_distribution=prob_dict,
            confidence=prob_dict[label],
            is_llm_generated=(label != "human"),
            suggested_model=None if label == "human" else label,
            decision_source="openrouter_chat_fallback",
            decision_notes=rationale,
        )
    except Exception:
        return None


def _classify_by_openrouter_chat_distribution(
    code_snippet: str,
    api_key: str,
    model_id: str | None = None,
) -> tuple[dict[str, float] | None, str]:
    """
    Ask a judge model for a full posterior over labels:
    {human, gpt3.5, gpt4, gpt5}.
    Returns (prob_dict or None, note).
    """
    import os
    import re
    try:
        import requests
    except ImportError:
        return None, "requests_missing"
    key = _get_openrouter_key(api_key)
    if not key:
        return None, "missing_api_key"
    model = model_id or os.environ.get("OPENROUTER_FALLBACK_CLASSIFIER_MODEL", OPENROUTER_FALLBACK_CLASSIFIER_MODEL)
    base = (os.environ.get("OPENROUTER_API_URL") or OPENROUTER_API_URL).rstrip("/").rsplit("/", 1)[0]
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/deem-mink",
    }
    code_preview = code_snippet[:3000].strip()
    if len(code_snippet) > 3000:
        code_preview += "\n..."
    prompt = (
        "You are a code-attribution judge.\n"
        "Return ONLY JSON with this exact schema:\n"
        "{\"human\": float, \"gpt3.5\": float, \"gpt4\": float, \"gpt5\": float, \"rationale\": string}\n"
        "Rules:\n"
        "- Probabilities must be between 0 and 1 and sum to 1.\n"
        "- human means human-authored.\n"
        "- Choose gpt3.5/gpt4/gpt5 based on likely capability/style level.\n"
        "- No markdown, no extra keys.\n\n"
        "Code:\n" + code_preview
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 160,
        "temperature": 0,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=40)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return None, "empty_choices"
        raw = (choices[0].get("message") or {}).get("content") or choices[0].get("text") or ""
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload_txt = m.group(0) if m else raw
        parsed = json.loads(payload_txt)
        out = {k: 0.0 for k in CLASS_LABELS}
        for k in ("human", "gpt3.5", "gpt4", "gpt5"):
            try:
                out[k] = float(parsed.get(k, 0.0))
            except Exception:
                out[k] = 0.0
        # Repair/normalize
        for k in out:
            out[k] = max(0.0, min(1.0, out[k]))
        s = sum(out.values())
        if s <= 0:
            return None, "zero_mass"
        out = {k: v / s for k, v in out.items()}
        return out, "ok"
    except Exception as e:
        return None, f"exception_{type(e).__name__}"


def _classify_by_openrouter_chat_ensemble(
    code_snippet: str,
    api_key: str,
    judge_models: list[str] | None = None,
) -> ClassificationResult | None:
    """
    More robust fallback: query multiple OpenRouter judge models and aggregate votes by confidence.
    Conservative tie-break: prefer non-human labels to reduce false "human" on GPT code.
    """
    import os
    models = list(judge_models) if judge_models else list(OPENROUTER_FALLBACK_JUDGE_MODELS)
    env_extra = os.environ.get("OPENROUTER_FALLBACK_JUDGE_MODELS", "").strip()
    if env_extra:
        models = [m.strip() for m in env_extra.split(",") if m.strip()]
    votes: dict[str, float] = {c: 0.0 for c in CLASS_LABELS}
    total = 0.0
    notes: list[str] = []
    for mid in models:
        dist, note = _classify_by_openrouter_chat_distribution(code_snippet, api_key, model_id=mid)
        if dist is not None:
            # Higher weight if judge is more decisive (low entropy)
            eps = 1e-12
            ent = -sum(p * math.log(p + eps) for p in dist.values() if p > 0)
            max_ent = math.log(max(2, len(CLASS_LABELS)))
            decisiveness = 1.0 - (ent / max_ent)
            w = max(0.5, min(1.2, 0.7 + decisiveness))
            for k in CLASS_LABELS:
                votes[k] += w * dist.get(k, 0.0)
            total += w
            notes.append(f"{mid}:dist,w={w:.2f}")
            continue
        # fallback to single-label parser
        res = _classify_by_openrouter_chat(code_snippet, api_key, model_id=mid)
        if res is not None:
            w = max(0.35, min(1.0, res.confidence))
            votes[res.predicted_label] += w
            total += w
            notes.append(f"{mid}:{res.predicted_label},w={w:.2f}")
        else:
            notes.append(f"{mid}:{note}")
    if total <= 0:
        return None
    # Normalize vote scores
    prob_dict = {k: (v / total) for k, v in votes.items()}
    # Balanced decision: allow human when support is clearly stronger.
    best = max(prob_dict, key=lambda k: prob_dict[k])
    if best == "human":
        best_llm = max((k for k in CLASS_LABELS if k != "human"), key=lambda k: prob_dict[k])
        if prob_dict["human"] < 0.54 or (prob_dict["human"] - prob_dict[best_llm]) < 0.06:
            best = best_llm
    conf = max(0.0, min(1.0, prob_dict.get(best, 0.0)))
    return ClassificationResult(
        predicted_label=best,
        probability_distribution=prob_dict,
        confidence=conf,
        is_llm_generated=(best != "human"),
        suggested_model=None if best == "human" else best,
        decision_source="openrouter_chat_ensemble",
        decision_notes=" | ".join(notes)[:500],
    )


def _classify_by_bayesian_features(code_snippet: str) -> ClassificationResult:
    """
    Bayesian-style heuristic fallback when API signals are unavailable.
    Uses observable code-style evidence to estimate posterior over labels.
    Conservative policy: avoid false-human on uncertainty.
    """
    import re

    lines = code_snippet.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    n_lines = max(1, len(non_empty))
    text = "\n".join(non_empty)

    def _count(pattern: str) -> int:
        return len(re.findall(pattern, text, flags=re.MULTILINE))

    # Observable evidence nodes
    has_docstring = ('"""' in code_snippet) or ("'''" in code_snippet)
    type_hints = _count(r"def\s+\w+\(.*\)\s*->\s*[\w\[\], ]+")
    typed_params = _count(r"def\s+\w+\([^)]*:\s*[\w\[\], ]+")
    exception_blocks = _count(r"\btry:\b") + _count(r"\bexcept\b")
    explanatory_comments = _count(r"^\s*#")
    structured_markers = sum(
        1 for k in ("Args:", "Returns:", "Raises:", "Example:", "Complexity:", "Parameters:")
        if k in code_snippet
    )
    advanced_api_usage = _count(r"\b(dataclass|typing|pathlib|contextmanager|Protocol|TypedDict|Enum|match\s+)\b")
    debug_prints = _count(r"\bprint\(")
    very_short = n_lines < 18

    # Priors (log-space)
    logp = {
        "human": math.log(0.28),
        "gpt3.5": math.log(0.25),
        "gpt4": math.log(0.21),
        "gpt5": math.log(0.20),
    }

    # Likelihood-style updates
    if has_docstring:
        logp["gpt4"] += 0.45
        logp["gpt5"] += 0.55
        logp["human"] -= 0.10
    if type_hints + typed_params >= 2:
        logp["gpt4"] += 0.50
        logp["gpt5"] += 0.70
        logp["gpt3.5"] += 0.15
    if structured_markers >= 1:
        logp["gpt4"] += 0.45
        logp["gpt5"] += 0.60
    if advanced_api_usage >= 1:
        logp["gpt5"] += 0.45
        logp["gpt4"] += 0.30
    if exception_blocks >= 1:
        logp["gpt4"] += 0.20
        logp["gpt5"] += 0.25
    if explanatory_comments >= max(4, n_lines // 6):
        logp["gpt3.5"] += 0.20
        logp["gpt4"] += 0.30
        logp["gpt5"] += 0.30
    if debug_prints >= 2:
        logp["human"] += 0.25
        logp["gpt3.5"] += 0.10
    if very_short:
        logp["human"] += 0.15
        logp["gpt3.5"] += 0.10
        logp["gpt5"] -= 0.15

    # Softmax posterior
    m = max(logp.values())
    expv = {k: math.exp(v - m) for k, v in logp.items()}
    s = sum(expv.values()) or 1.0
    post = {k: expv[k] / s for k in expv}

    # Balanced decision: choose LLM only when human evidence is weak.
    pred = max(post, key=lambda k: post[k])
    if pred == "human":
        best_llm = max(("gpt3.5", "gpt4", "gpt5"), key=lambda k: post[k])
        if post["human"] < 0.52 or (post["human"] - post[best_llm]) < 0.04:
            pred = best_llm

    conf = max(0.0, min(1.0, post[pred]))
    notes = (
        f"bayesian_features: lines={n_lines}, docstring={has_docstring}, "
        f"type_hints={type_hints + typed_params}, structured={structured_markers}, "
        f"exceptions={exception_blocks}, comments={explanatory_comments}, adv={advanced_api_usage}"
    )
    return ClassificationResult(
        predicted_label=pred,
        probability_distribution=post,
        confidence=conf,
        is_llm_generated=(pred != "human"),
        suggested_model=None if pred == "human" else pred,
        decision_source="bayesian_feature_fallback",
        decision_notes=notes,
    )


def _ml_signature_strength(code_snippet: str) -> float:
    """
    Heuristic score [0,1] for structured ML/sklearn-style code.
    Used as a calibration feature in final fusion to reduce GPT-5->human flips.
    """
    import re
    text = code_snippet
    patterns = [
        r"\bfrom\s+sklearn\b",
        r"\bimport\s+sklearn\b",
        r"\btrain_test_split\b",
        r"\b(StandardScaler|MinMaxScaler|RobustScaler)\b",
        r"\b(Pipeline|make_pipeline)\b",
        r"\b(GridSearchCV|RandomizedSearchCV)\b",
        r"\b(cross_val_score|cross_validate|StratifiedKFold|KFold)\b",
        r"\b(classification_report|confusion_matrix|roc_auc_score|f1_score|accuracy_score)\b",
        r"\b(LogisticRegression|RandomForestClassifier|XGBClassifier|SVC|SVR|LinearRegression)\b",
        r"\.fit\(",
        r"\.predict\(",
        r"\.transform\(",
    ]
    hits = sum(1 for p in patterns if re.search(p, text))
    # cap at 1.0 after enough signals
    return max(0.0, min(1.0, hits / 8.0))


def _naive_bayes_signature_strength(code_snippet: str) -> float:
    """
    Heuristic score [0,1] for sklearn Naive Bayes style pipelines.
    Used to prevent GPT NB/ML code from being mislabeled as human.
    """
    import re
    text = code_snippet
    patterns = [
        r"\bfrom\s+sklearn\.naive_bayes\s+import\b",
        r"\b(GaussianNB|MultinomialNB|BernoulliNB|ComplementNB|CategoricalNB)\b",
        r"\bnaive[_\-\s]?bayes\b",
        r"\bCountVectorizer\b",
        r"\bTfidfVectorizer\b",
        r"\bPipeline\b",
        r"\bfit\(",
        r"\bpredict\(",
        r"\bclassification_report\b",
        r"\bconfusion_matrix\b",
    ]
    hits = sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))
    return max(0.0, min(1.0, hits / 6.0))


def run_min_k_openrouter(
    code_snippet: str,
    model_id: str,
    api_key: str,
    k_percent: float = 0.1,
    tau: float | None = None,
    max_tokens: int = 256,
) -> MinKResult | None:
    """
    Run Min-K using OpenRouter: get token log-probs from the API, then compute Min-K score.
    Tries completions first, then chat/completions if logprobs not returned.
    Returns None if API fails or logprobs unavailable.
    """
    log_probs_list = _openrouter_completion_logprobs(code_snippet, model_id, api_key, max_tokens=max_tokens)
    if not log_probs_list:
        log_probs_list = _openrouter_chat_logprobs(code_snippet, model_id, api_key, max_tokens=max_tokens)
    if not log_probs_list:
        return None
    import numpy as np
    sorted_lp = sorted(log_probs_list)
    k = max(1, int(len(sorted_lp) * k_percent))
    lowest_k = sorted_lp[:k]
    min_k_score = float(np.mean(lowest_k))
    num_tokens = len(log_probs_list)
    if tau is None:
        tau = -8.0
    membership_prob = 1.0 / (1.0 + math.exp(-(-(min_k_score - tau))))
    explanation = (
        f"Min-K% (OpenRouter {model_id}, k={k_percent*100:.0f}%, τ={tau}): "
        f"score={min_k_score:.4f}, membership={membership_prob:.2%}, n_tokens={num_tokens}."
    )
    return MinKResult(
        min_k_score=min_k_score,
        membership_probability=membership_prob,
        decision_threshold=tau,
        k_percent=k_percent,
        num_tokens_used=num_tokens,
        explanation=explanation,
    )


def classify_by_min_k_openrouter(
    code_snippet: str,
    api_key: str,
    models: list[str] | None = None,
    k_percent: float = 0.1,
    tau: float | None = None,
) -> tuple[ClassificationResult, MinKResult | None]:
    """
    Classify code as human vs LLM using Min-K over OpenRouter only (GPT-3.5 Turbo, GPT-4, GPT-5).
    No Hugging Face models. Requires OPENROUTER_API_KEY. Returns (ClassificationResult, MinKResult or None).
    """
    import os
    key = _get_openrouter_key(api_key)
    model_list = list(models) if models else list(OPENROUTER_DEFAULT_MODELS)
    extra = os.environ.get("OPENROUTER_EXTRA_MODELS", "").strip()
    if extra:
        for m in extra.split(","):
            m = m.strip()
            if m and m not in model_list:
                model_list.append(m)
    results_per_model: list[tuple[str, MinKResult]] = []
    for model_id in model_list:
        mink = run_min_k_openrouter(code_snippet, model_id, key, k_percent=k_percent, tau=tau)
        if mink is not None:
            results_per_model.append((model_id, mink))
    # Always try API judge ensemble too (even when Min-K exists) and fuse both API signals.
    judge_result = _classify_by_openrouter_chat_ensemble(code_snippet, key)

    # If API signal is unavailable, still return a class via Bayesian fallback.
    if not results_per_model and judge_result is None:
        import warnings
        warnings.warn(
            "No OpenRouter signal available: Min-K logprobs missing and chat judges failed.",
            UserWarning,
            stacklevel=2,
        )
        bayes = _classify_by_bayesian_features(code_snippet)
        bayes.decision_source = "api_failed_bayesian_fallback"
        bayes.decision_notes = (
            "Min-K logprobs unavailable and chat judges unavailable; "
            + (bayes.decision_notes or "Bayesian fallback")
        )
        return (bayes, None)

    # Build Min-K distribution (API-derived) when available.
    min_k_prob: dict[str, float] | None = None
    best_mink: MinKResult | None = None
    best_model_id = None
    if results_per_model:
        sorted_by_score = sorted(results_per_model, key=lambda x: x[1].min_k_score)
        best_model_id, best_mink = sorted_by_score[0]
        scores = [m.min_k_score for _, m in results_per_model]
        min_score = min(scores)
        exp_neg = [math.exp(-(s - min_score)) for s in scores]
        total = sum(exp_neg)
        probs = [x / total for x in exp_neg] if total else [1.0 / len(scores)] * len(scores)
        min_k_prob = {c: 0.0 for c in CLASS_LABELS}
        for (mid, _), p in zip(results_per_model, probs):
            lbl = _resolve_openrouter_label(mid)
            if lbl is None:
                continue
            min_k_prob[lbl] = min_k_prob.get(lbl, 0.0) + p
        # Human posterior from Min-K score thresholds
        best_score = best_mink.min_k_score
        best_label = _resolve_openrouter_label(best_model_id) or "gpt4"
        llm_mass_post = sum(min_k_prob[c] for c in CLASS_LABELS if c != "human")
        if llm_mass_post <= 0:
            # If model IDs could not be mapped, keep a neutral LLM split (no GPT-4 default).
            min_k_prob["gpt3.5"] = 1.0 / 3.0
            min_k_prob["gpt4"] = 1.0 / 3.0
            min_k_prob["gpt5"] = 1.0 / 3.0
        thresh = float(os.environ.get("MIN_K_HUMAN_THRESHOLD", str(MIN_K_HUMAN_THRESHOLD)))
        if best_label == "gpt5":
            thresh = float(os.environ.get("MIN_K_HUMAN_THRESHOLD_GPT5", str(MIN_K_HUMAN_THRESHOLD_GPT5)))
        # Convert distance-from-threshold to soft human probability; clip so it stays probabilistic.
        p_human = 1.0 / (1.0 + math.exp(-(best_score - thresh)))
        p_human = max(0.01, min(0.60, p_human))
        llm_mass = sum(min_k_prob[c] for c in CLASS_LABELS if c != "human")
        if llm_mass > 0:
            scale = (1.0 - p_human) / llm_mass
            for c in CLASS_LABELS:
                if c != "human":
                    min_k_prob[c] *= scale
        min_k_prob["human"] = p_human

    # Choose source-specific distributions; if one is missing, use the other directly.
    if min_k_prob is None and judge_result is not None:
        out = judge_result
        out.decision_source = "api_chat_ensemble_only"
        return (out, None)
    if min_k_prob is not None and judge_result is None:
        # Pick max posterior from Min-K distribution.
        label = max(min_k_prob, key=lambda k: min_k_prob[k])
        conf = max(0.0, min(1.0, min_k_prob[label]))
        return (
            ClassificationResult(
                predicted_label=label,
                probability_distribution=min_k_prob,
                confidence=conf,
                is_llm_generated=(label != "human"),
                suggested_model=None if label == "human" else label,
                decision_source="api_min_k_only",
                decision_notes=f"best_model={best_model_id}, best_score={best_mink.min_k_score:.4f}" if best_mink else "min_k_only",
            ),
            best_mink,
        )

    # Fuse API Min-K and API chat ensemble probabilities.
    assert min_k_prob is not None and judge_result is not None
    judge_prob = judge_result.probability_distribution
    # Reliability-based weights: Min-K stronger when it has multiple model signals; judges stronger when highly confident.
    w_min_k = 0.62 if len(results_per_model) >= 2 else 0.50
    w_judge = 1.0 - w_min_k
    if judge_result.confidence >= 0.80:
        w_judge += 0.08
        w_min_k -= 0.08
    # If Min-K score is in an ambiguous high-likelihood range, rely more on judge distribution.
    if best_mink is not None and best_mink.min_k_score > -2.0:
        w_judge += 0.12
        w_min_k -= 0.12
    if best_mink is not None and best_mink.min_k_score > -1.2:
        w_judge += 0.08
        w_min_k -= 0.08
    w_min_k = max(0.35, min(0.75, w_min_k))
    w_judge = 1.0 - w_min_k

    fused = {c: 0.0 for c in CLASS_LABELS}
    for c in CLASS_LABELS:
        fused[c] = w_min_k * min_k_prob.get(c, 0.0) + w_judge * judge_prob.get(c, 0.0)

    # ML/sklearn calibration: structured ML code is frequently AI-assisted in this task setting.
    # Apply a mild, evidence-gated correction to reduce human false positives on GPT-5 ML code.
    ml_sig = _ml_signature_strength(code_snippet)
    nb_sig = _naive_bayes_signature_strength(code_snippet)
    if ml_sig >= 0.45 and (judge_prob.get("human", 0.0) < 0.50 or min_k_prob.get("human", 0.0) < 0.50):
        hum_scale = max(0.65, 1.0 - 0.35 * ml_sig)
        fused["human"] *= hum_scale
        fused["gpt5"] += 0.10 * ml_sig
        fused["gpt4"] += 0.05 * ml_sig
        s_fix = sum(fused.values()) or 1.0
        fused = {k: v / s_fix for k, v in fused.items()}
    # Stronger correction for explicit sklearn Naive Bayes style code.
    if nb_sig >= 0.45:
        fused["human"] *= max(0.45, 1.0 - 0.55 * nb_sig)
        fused["gpt5"] += 0.16 * nb_sig
        fused["gpt4"] += 0.10 * nb_sig
        fused["gpt3.5"] += 0.04 * nb_sig
        s_fix_nb = sum(fused.values()) or 1.0
        fused = {k: v / s_fix_nb for k, v in fused.items()}
    # Guardrail: only override to LLM on very strong judge confidence.
    best_llm_judge = max((k for k in CLASS_LABELS if k != "human"), key=lambda k: judge_prob.get(k, 0.0))
    if judge_prob.get(best_llm_judge, 0.0) >= 0.90 and fused.get("human", 0.0) < 0.40:
        fused[best_llm_judge] = max(fused[best_llm_judge], 0.65)
        rem = 1.0 - fused[best_llm_judge]
        others = [k for k in CLASS_LABELS if k != best_llm_judge]
        denom = sum(fused[k] for k in others) or 1.0
        for k in others:
            fused[k] = rem * (fused[k] / denom)

    # Human-vs-LLM from judge distribution (primary gate).
    p_human_min_k = min_k_prob.get("human", 0.0)
    p_human_judge = judge_prob.get("human", 0.0)
    p_human_fused = fused.get("human", 0.0)
    best_llm_judge = max(("gpt3.5", "gpt4", "gpt5"), key=lambda k: judge_prob.get(k, 0.0))
    llm_judge_top = judge_prob.get(best_llm_judge, 0.0)

    # Strong ML/NB signatures: require very strong judge-human to call human.
    if ml_sig >= 0.50 and nb_sig >= 0.45:
        human_allowed = p_human_judge >= 0.80 and (p_human_judge - llm_judge_top) >= 0.12
    else:
        llm_fused_top = max(fused.get("gpt3.5", 0.0), fused.get("gpt4", 0.0), fused.get("gpt5", 0.0))
        human_allowed = (
            (p_human_judge >= 0.50 and (p_human_judge - llm_judge_top) >= 0.03)
            or (p_human_fused >= 0.58 and p_human_min_k >= 0.30 and (p_human_fused - llm_fused_top) >= 0.05)
        )

    # Safety: if Min-K strongly disagrees with human, block human flip.
    if human_allowed:
        best_llm_min_k = max(("gpt3.5", "gpt4", "gpt5"), key=lambda k: min_k_prob.get(k, 0.0))
        if p_human_min_k < 0.20 and min_k_prob.get(best_llm_min_k, 0.0) >= 0.55:
            human_allowed = False

    # Consistency guard: if fused posterior itself clearly favors human, do not output GPT label.
    llm_fused_top = max(fused.get("gpt3.5", 0.0), fused.get("gpt4", 0.0), fused.get("gpt5", 0.0))
    if (
        p_human_fused >= 0.50
        and (p_human_fused - llm_fused_top) >= 0.03
        and not (ml_sig >= 0.50 and nb_sig >= 0.45 and p_human_judge < 0.80)
    ):
        human_allowed = True
    # Strong consistency: if human is the fused argmax with clear margin, keep human
    # unless strict ML/NB anti-human condition is active.
    best_label_fused = max(fused, key=lambda k: fused[k])
    if (
        best_label_fused == "human"
        and (p_human_fused - llm_fused_top) >= 0.06
        and not (ml_sig >= 0.50 and nb_sig >= 0.45 and p_human_judge < 0.80)
    ):
        human_allowed = True

    # Uncertainty calibration for classroom false positives:
    # when GPT vs Human is close and no strong ML/NB anti-human signal exists,
    # slightly favor human to reduce overcalling GPT-4 on human-authored code.
    gap_llm_over_human = llm_fused_top - p_human_fused
    near_tie_human_case = (
        not (ml_sig >= 0.50 and nb_sig >= 0.45 and p_human_judge < 0.80)
        and p_human_fused >= 0.30
        and p_human_judge >= 0.28
        and llm_fused_top < 0.58
        and 0.0 <= gap_llm_over_human <= 0.18
    )
    if near_tie_human_case:
        top_llm_label = max(("gpt3.5", "gpt4", "gpt5"), key=lambda k: fused.get(k, 0.0))
        # Shift a limited mass from top LLM to human in borderline cases.
        shift = max(0.0, min(0.12, gap_llm_over_human + 0.02))
        if shift > 0.0 and fused.get(top_llm_label, 0.0) > shift:
            fused[top_llm_label] = max(0.0, fused[top_llm_label] - shift)
            fused["human"] = fused.get("human", 0.0) + shift
            s_re = sum(fused.values()) or 1.0
            fused = {k: v / s_re for k, v in fused.items()}
            p_human_fused = fused.get("human", 0.0)
            llm_fused_top = max(fused.get("gpt3.5", 0.0), fused.get("gpt4", 0.0), fused.get("gpt5", 0.0))
        if p_human_fused >= llm_fused_top:
            human_allowed = True

    if human_allowed:
        pred = "human"
        conf = max(0.0, min(1.0, max(p_human_fused, p_human_judge)))
    else:
        # If LLM, choose GPT tier using fused LLM-only probabilities.
        llm_labels = ("gpt3.5", "gpt4", "gpt5")
        llm_pred = max(llm_labels, key=lambda k: fused.get(k, 0.0))
        # GPT-4 de-bias: if GPT-4 only wins narrowly, prefer stronger neighboring evidence.
        if llm_pred == "gpt4":
            g4 = fused.get("gpt4", 0.0)
            g5 = fused.get("gpt5", 0.0)
            h = fused.get("human", 0.0)
            close_g5 = (g4 - g5) <= 0.06 and judge_prob.get("gpt5", 0.0) >= (judge_prob.get("gpt4", 0.0) - 0.02)
            close_h = (g4 - h) <= 0.08 and p_human_judge >= 0.34 and p_human_min_k >= 0.24
            if close_h and not (ml_sig >= 0.50 and nb_sig >= 0.45 and p_human_judge < 0.80):
                pred = "human"
                conf = max(0.0, min(1.0, max(h, p_human_judge)))
                return (
                    ClassificationResult(
                        predicted_label=pred,
                        probability_distribution=fused,
                        confidence=conf,
                        is_llm_generated=False,
                        suggested_model=None,
                        decision_source="api_ensemble_fusion",
                        decision_notes=(
                            f"weights(min_k={w_min_k:.2f},judge={w_judge:.2f}); "
                            f"judge={judge_result.predicted_label}:{judge_result.confidence:.2f}; "
                            f"best_model={best_model_id}; "
                            f"p_human(min_k={p_human_min_k:.2f},judge={p_human_judge:.2f},fused={p_human_fused:.2f}); "
                            f"ml_sig={ml_sig:.2f},nb_sig={nb_sig:.2f}; gpt4_debias=human"
                        ),
                    ),
                    best_mink,
                )
            if close_g5:
                llm_pred = "gpt5"
        pred = llm_pred
        conf = max(0.0, min(1.0, fused.get(llm_pred, 0.0)))

    return (
        ClassificationResult(
            predicted_label=pred,
            probability_distribution=fused,
            confidence=conf,
            is_llm_generated=(pred != "human"),
            suggested_model=None if pred == "human" else pred,
            decision_source="api_ensemble_fusion",
            decision_notes=(
                f"weights(min_k={w_min_k:.2f},judge={w_judge:.2f}); "
                f"judge={judge_result.predicted_label}:{judge_result.confidence:.2f}; "
                f"best_model={best_model_id}; "
                f"p_human(min_k={p_human_min_k:.2f},judge={p_human_judge:.2f},fused={p_human_fused:.2f}); "
                f"ml_sig={ml_sig:.2f},nb_sig={nb_sig:.2f}"
            ),
        ),
        best_mink,
    )


# ---------------------------------------------------------------------------
# 4) W3C PROV-Style Data Provenance Graph
# ---------------------------------------------------------------------------

def _top_similarity_per_source(similarity_hits: list[SimilarityHit]) -> dict[str, float]:
    """Return max cosine similarity per source (github, stackoverflow, other_web)."""
    out: dict[str, float] = {}
    for h in similarity_hits:
        s = h.source
        if s not in out or h.cosine_similarity > out[s]:
            out[s] = h.cosine_similarity
    return out


def create_prov_graph(
    classification_result: ClassificationResult,
    similarity_hits: list[SimilarityHit],
    min_k_result: MinKResult | None,
    predicted_model: str | None,
) -> Any:
    """
    Build a W3C PROV-style directed graph with Entities, Activities, Agents.
    Links user code to repositories via similarity analysis (with scores).
    """
    import networkx as nx

    G = nx.DiGraph()
    top_sim = _top_similarity_per_source(similarity_hits)

    # Minimal, intuitive provenance graph
    G.add_node("user_code", node_type="entity", label="Submitted Code")
    G.add_node("repo_similarity", node_type="activity", label="Repository Similarity")
    G.add_node("github_repo", node_type="entity", label=f"GitHub (sim={top_sim.get('github', 0):.2f})")
    G.add_node("stackoverflow_repo", node_type="entity", label=f"StackOverflow (sim={top_sim.get('stackoverflow', 0):.2f})")
    G.add_node("other_web_repo", node_type="entity", label=f"Other Web (sim={top_sim.get('other_web', 0):.2f})")
    G.add_node("model_attribution", node_type="activity", label="Model Attribution")
    G.add_node("code_generation", node_type="activity", label="Code Generation")
    G.add_node("generated_code_output", node_type="entity", label="Attributed Output")
    G.add_node("final_attribution", node_type="entity", label="Final Attribution")
    # Keep Min-K node/edge in graph structure even when Min-K details are unavailable.
    min_k_label = f"Min-K = {min_k_result.min_k_score:.3f}" if min_k_result is not None else "Min-K = N/A"
    G.add_node("min_k_score", node_type="entity", label=min_k_label)

    agents = ["human", "gpt3.5", "gpt4", "gpt5"]
    probs = classification_result.probability_distribution or {}
    for a in agents:
        pa = float(probs.get(a, 0.0))
        G.add_node(a, node_type="agent", label=f"{display_label(a)}\nP={pa:.2f}")

    chosen = (predicted_model or classification_result.predicted_label or "human").lower()
    if chosen not in agents:
        chosen = "gpt5"
    G.nodes[chosen]["selected"] = True

    # Evidence flow
    G.add_edge("user_code", "repo_similarity", relation="analyzedBy")
    G.add_edge("repo_similarity", "github_repo", relation=f"similarTo p={top_sim.get('github', 0):.2f}")
    G.add_edge("repo_similarity", "stackoverflow_repo", relation=f"similarTo p={top_sim.get('stackoverflow', 0):.2f}")
    G.add_edge("repo_similarity", "other_web_repo", relation=f"similarTo p={top_sim.get('other_web', 0):.2f}")
    G.add_edge("user_code", "model_attribution", relation="analyzedBy")
    G.add_edge("min_k_score", "model_attribution", relation="used")
    # Bayesian-style posterior edges from all candidate models/agents
    for a in agents:
        pa = float(probs.get(a, 0.0))
        G.add_edge(a, "model_attribution", relation=f"posterior P={pa:.2f}")
    G.add_edge("model_attribution", "final_attribution", relation="wasGeneratedBy")
    G.add_edge(chosen, "code_generation", relation="wasAttributedTo")
    G.add_edge("code_generation", "generated_code_output", relation="wasGeneratedBy")
    G.add_edge("final_attribution", "user_code", relation="wasDerivedFrom")

    if chosen == "human":
        G.nodes["generated_code_output"]["label"] = "Human Written Code"
    else:
        G.nodes["generated_code_output"]["label"] = "LLM Generated Code"
    G.nodes["final_attribution"]["label"] = (
        f"{display_label(chosen)}\n"
        f"P={float(probs.get(chosen, classification_result.confidence)):.2f}, "
        f"confidence={classification_result.confidence:.1%}"
    )

    return G


def create_min_k_subgraph(min_k_result: MinKResult, predicted_label: str, confidence: float) -> Any:
    """
    Subgraph: User Code → Min-K Analysis → Probability Score → Final Attribution.
    Includes K value, threshold τ, Min-K score, and final confidence.
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_node("user_code", node_type="entity", label="User Code")
    G.add_node("min_k_analysis", node_type="activity", label="Min-K Analysis")
    G.add_node("probability_score", node_type="entity", label=f"Min-K Score = {min_k_result.min_k_score:.4f}")
    G.add_node(
        "final_attribution",
        node_type="entity",
        label=f"{display_label(predicted_label)} (cls {confidence:.1%}, Min-K {min_k_result.membership_probability:.1%})",
    )

    G.add_edge("user_code", "min_k_analysis", relation="analyzedBy")
    G.add_edge("min_k_analysis", "probability_score", relation="wasGeneratedBy")
    G.add_edge("probability_score", "final_attribution", relation="used")

    # Store metadata for visualization
    G.graph["k_percent"] = min_k_result.k_percent
    G.graph["threshold_tau"] = min_k_result.decision_threshold
    G.graph["min_k_score"] = min_k_result.min_k_score
    G.graph["final_confidence"] = confidence
    return G


def _ensure_pyvis() -> bool:
    """Try to import pyvis; if missing, attempt pip install and retry. Returns True if available."""
    try:
        from pyvis.network import Network  # type: ignore[import-untyped]
        return True
    except ImportError:
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis"], timeout=120)
            from pyvis.network import Network  # type: ignore[import-untyped]
            return True
        except Exception:
            return False


def print_prov_graph_structure(G: Any, title: str = "Data Provenance Graph (W3C PROV)") -> None:
    """Print the graph structure (nodes and edges) to the console so it's always visible."""
    import networkx as nx
    print("\n" + "=" * 60)
    print(f"[{title}]")
    print("=" * 60)
    print("  Nodes (type: label):")
    for n in G.nodes():
        attr = G.nodes[n]
        node_type = attr.get("node_type", "entity")
        label = attr.get("label", str(n))
        print(f"    - {n} [{node_type}]: {label}")
    print("  Edges (source -> target : relation):")
    for u, v in G.edges():
        rel = G.edges[u, v].get("relation", "")
        print(f"    - {u} -> {v} : {rel}")
    print("=" * 60)


def draw_prov_graph_pyvis(G: Any, title: str = "PROV Graph", height: str = "600px") -> Any:
    """Render graph with pyvis (interactive HTML). Installs pyvis if missing."""
    if not _ensure_pyvis():
        raise ImportError("pyvis could not be installed. Run: pip install pyvis")
    from pyvis.network import Network  # type: ignore[import-untyped]

    net = Network(height=height, directed=True)
    for n in G.nodes():
        attr = G.nodes[n]
        node_type = attr.get("node_type", "entity")
        label = attr.get("label", str(n))
        color = "#97C2FC" if node_type == "entity" else "#FB7E81" if node_type == "activity" else "#7BE141"
        net.add_node(n, label=label, color=color, title=f"{node_type}: {label}")
    import re
    for u, v in G.edges():
        rel = str(G.edges[u, v].get("relation", ""))
        m = re.search(r"(?:P|p)=([0-9]*\.?[0-9]+)", rel)
        prob = float(m.group(1)) if m else None
        if prob is None:
            color = "#666666"
            width = 1.2
        elif prob >= 0.66:
            color = "#2E7D32"  # strong: green
            width = 3.0
        elif prob >= 0.33:
            color = "#F9A825"  # medium: amber
            width = 2.2
        else:
            color = "#C62828"  # weak: red
            width = 1.6
        net.add_edge(u, v, title=rel, label=rel, color=color, width=width)
    # Add a compact legend node for edge color interpretation.
    legend_text = (
        "Edge color legend\\n"
        "Green: strong (>=0.66)\\n"
        "Amber: medium (0.33-0.66)\\n"
        "Red: weak (<0.33)\\n"
        "Gray: non-probabilistic"
    )
    net.add_node(
        "__edge_color_legend__",
        label=legend_text,
        color="#FFFFFF",
        shape="box",
        fixed=True,
        x=900,
        y=-450,
        physics=False,
        title=legend_text,
        font={"size": 12},
    )
    net.set_options("""
    var options = {
      "edges": {"arrows": "to", "smooth": false},
      "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -3000}}
    }
    """)
    return net


def _draw_prov_graph_matplotlib(G: Any, figsize: tuple[float, float] = (20, 12), title: str | None = None) -> None:
    """Draw PROV graph with networkx + matplotlib (shared logic for save and display)."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import textwrap

    # Prefer a fixed left-to-right layout for known PROV nodes so the flow is intuitive.
    base_positions = {
        # Repositories and similarity
        "github_repo": (-4.2, 1.8),
        "stackoverflow_repo": (-4.2, 0.2),
        "other_web_repo": (-4.2, -1.4),
        "repo_similarity": (-2.6, 0.2),
        # Input and agent choices
        "user_code": (-1.0, 2.4),
        "human": (0.8, 2.9),
        "gpt3.5": (0.8, 1.5),
        "gpt4": (0.8, 0.1),
        "gpt5": (0.8, -1.3),
        # Attribution flow
        "min_k_score": (2.2, -1.2),
        "model_attribution": (2.8, 0.8),
        "code_generation": (2.8, 2.2),
        "final_attribution": (4.3, 0.8),
        "generated_code_output": (4.3, 2.2),
    }

    if all(n in base_positions for n in G.nodes()):
        pos = {n: base_positions[n] for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, k=2.0, seed=42, iterations=50)
    fig, ax = plt.subplots(figsize=figsize)
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        t = G.nodes[n].get("node_type", "entity")
        selected = G.nodes[n].get("selected", False)
        if selected:
            node_colors.append("#FFD54F")  # highlight chosen LLM/human
            node_sizes.append(2400)
        else:
            if t == "entity":
                node_colors.append("#4A90D9")
            elif t == "activity":
                node_colors.append("#E85D75")
            else:
                node_colors.append("#2E7D32")
            node_sizes.append(1900)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    # Wrap long labels so nodes remain readable.
    wrapped_labels = {}
    for n in G.nodes():
        lbl = str(G.nodes[n].get("label", n))
        wrapped_labels[n] = textwrap.fill(lbl, width=22)
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=8, font_weight="bold", ax=ax)
    import re
    # Edge styling by probability strength when relation carries p/P value.
    edgelist = list(G.edges())
    edge_colors: list[str] = []
    edge_widths: list[float] = []
    for u, v in edgelist:
        rel = str(G.edges[u, v].get("relation", ""))
        m = re.search(r"(?:P|p)=([0-9]*\.?[0-9]+)", rel)
        prob = float(m.group(1)) if m else None
        if prob is None:
            edge_colors.append("#666666")
            edge_widths.append(1.2)
        elif prob >= 0.66:
            edge_colors.append("#2E7D32")  # strong
            edge_widths.append(2.8)
        elif prob >= 0.33:
            edge_colors.append("#F9A825")  # medium
            edge_widths.append(2.2)
        else:
            edge_colors.append("#C62828")  # weak
            edge_widths.append(1.6)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=26,
        alpha=0.85,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    # Draw all relation labels with staggered positions to reduce overlap.
    all_edge_labels = nx.get_edge_attributes(G, "relation")
    if all_edge_labels:
        items = list(all_edge_labels.items())
        label_positions = [0.30, 0.45, 0.60, 0.75]
        for i, ((u, v), rel) in enumerate(items):
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels={(u, v): rel},
                font_size=7,
                label_pos=label_positions[i % len(label_positions)],
                rotate=False,
                bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.75, pad=0.2),
                ax=ax,
            )
    # Edge-color legend in-graph (for probability-strength edges).
    edge_legend = (
        "Edge color legend:\n"
        "Green: strong probability (>= 0.66)\n"
        "Amber: medium probability (0.33 - 0.66)\n"
        "Red: weak probability (< 0.33)\n"
        "Gray: non-probabilistic provenance relation"
    )
    ax.text(
        0.99,
        0.01,
        edge_legend,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.85),
    )
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()


def display_prov_graph_inline(G: Any, title: str | None = None) -> None:
    """
    Draw and display the provenance graph directly in the notebook/Colab output (no file).
    Uses matplotlib so the figure appears in the cell output.
    """
    try:
        import matplotlib.pyplot as plt
        _draw_prov_graph_matplotlib(G, figsize=(20, 12), title=title or "Data Provenance Graph")
        plt.show()
    except Exception as e:
        print("Inline graph display failed:", e)


def export_prov_graph_png(G: Any, path: str = "prov_graph.png") -> None:
    """Export graph to PNG using networkx layout and matplotlib (no graphviz required)."""
    import matplotlib.pyplot as plt
    _draw_prov_graph_matplotlib(G, figsize=(12, 8))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 5) User code input
# ---------------------------------------------------------------------------

def get_user_code_input(code_from_arg: str | None = None) -> str | None:
    """
    Get Python code from the user. Use this so the pipeline never runs without explicit code.
    - If code_from_arg is non-empty, return it.
    - Else if sys.argv[1] is a file path, read and return its contents.
    - Else prompt for pasted code (multiline until empty line).
    Returns None if user provides no code.
    """
    import sys
    if code_from_arg and code_from_arg.strip():
        return code_from_arg.strip()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Could not read file {path}: {e}")
    print("Enter your Python code to analyze.")
    print("Paste the code below, then press Enter on an empty line to finish:")
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines:
                break
            if line == "" and not lines:
                continue
            lines.append(line)
    except EOFError:
        pass
    code = "\n".join(lines).strip()
    return code if code else None


# ---------------------------------------------------------------------------
# 6) Main pipeline and output
# ---------------------------------------------------------------------------

def run_full_pipeline(
    code_snippet: str,
    classifier_model_dict=None,
    embedding_model=None,
    repo_data=None,
    k_percent: float = 0.1,
    tau: float | None = None,
    top_k_similar: int = 5,
    use_repo_api: bool = True,
    github_token: str | None = None,
) -> dict[str, Any]:
    """
    Run the full DEEM_MinK pipeline and return structured results.
    When use_repo_api=True (default), repository similarity uses GitHub and Stack Exchange APIs.
    """
    results = {
        "input_code_preview": code_snippet[:500] + ("..." if len(code_snippet) > 500 else ""),
        "classification": None,
        "repository_similarity": None,
        "repo_disclaimer": REPO_DISCLAIMER,
        "min_k": None,
        "prov_graph": None,
        "min_k_subgraph": None,
        "explanations": [],
    }

    # 1) Classification via OpenRouter API ensemble (Min-K + chat judges)
    classification_result, min_k_result = classify_by_min_k_openrouter(
        code_snippet,
        api_key=None,
        k_percent=k_percent,
        tau=tau,
    )

    # 2) Repository similarity (API-based when use_repo_api=True)
    if embedding_model is None:
        embedding_model = load_embedding_model()
    if repo_data is not None:
        repo_emb, repo_snippets, repo_sources = repo_data
        similarity_hits, disclaimer = repository_similarity(
            code_snippet, embedding_model, top_k=top_k_similar,
            repo_embeddings=repo_emb, repo_snippets=repo_snippets, repo_sources=repo_sources,
            use_api=False,
        )
    else:
        similarity_hits, disclaimer = repository_similarity(
            code_snippet, embedding_model, top_k=top_k_similar,
            use_api=use_repo_api,
            github_token=github_token,
        )
    results["repository_similarity"] = [
        {"source": h.source, "snippet": h.snippet, "cosine_similarity": h.cosine_similarity, "rank": h.rank}
        for h in similarity_hits
    ]
    results["repo_disclaimer"] = disclaimer
    results["explanations"].append(disclaimer)

    # 3) Store classification (from Min-K) and Min-K result
    results["classification"] = {
        "predicted_label": classification_result.predicted_label,
        "predicted_label_display": display_label(classification_result.predicted_label),
        "probability_distribution": classification_result.probability_distribution,
        "confidence": classification_result.confidence,
        "is_llm_generated": classification_result.is_llm_generated,
        "suggested_model": classification_result.suggested_model,
        "suggested_model_display": display_label(classification_result.suggested_model) if classification_result.suggested_model else None,
        "decision_source": classification_result.decision_source,
        "decision_notes": classification_result.decision_notes,
    }
    if min_k_result is not None:
        results["min_k"] = {
            "min_k_score": min_k_result.min_k_score,
            "membership_probability": min_k_result.membership_probability,
            "decision_threshold": min_k_result.decision_threshold,
            "k_percent": min_k_result.k_percent,
            "num_tokens_used": min_k_result.num_tokens_used,
            "explanation": min_k_result.explanation,
        }
        results["explanations"].append(min_k_result.explanation)
    results["explanations"].insert(
        0,
        f"Classification: {display_label(classification_result.predicted_label)} (confidence {classification_result.confidence:.2%}, "
        f"source={classification_result.decision_source}). LLM-generated: {classification_result.is_llm_generated}."
    )

    # 4) PROV graph and Min-K subgraph (with repository similarity linked)
    G = create_prov_graph(
        classification_result,
        similarity_hits,
        min_k_result,
        classification_result.suggested_model,
    )
    results["prov_graph"] = G
    if min_k_result is not None:
        subG = create_min_k_subgraph(
            min_k_result,
            classification_result.predicted_label,
            classification_result.confidence,
        )
        results["min_k_subgraph"] = subG
    else:
        results["min_k_subgraph"] = None

    return results


def print_pipeline_output(results: dict[str, Any]) -> None:
    """Pretty-print pipeline results and explanations."""
    print("=" * 60)
    print("DEEM_MinK Pipeline Output")
    print("=" * 60)
    print("\n[Classification]")
    c = results["classification"]
    print(f"  Predicted: {c.get('predicted_label_display', c['predicted_label'])} (confidence: {c['confidence']:.2%})")
    print(f"  LLM-generated: {c['is_llm_generated']}")
    if c.get("decision_source"):
        print(f"  Decision source: {c['decision_source']}")
    if c.get("decision_notes"):
        print(f"  Decision notes: {c['decision_notes']}")
    print("  Probability distribution:", c["probability_distribution"])
    print("\n[Repository Similarity (top-k)]")
    print(f"  {results['repo_disclaimer']}")
    for h in results["repository_similarity"]:
        snip = h["snippet"]
        print(f"  #{h['rank']} [{h['source']}] sim={h['cosine_similarity']:.4f}  snippet: {(snip[:80] + '...') if len(snip) > 80 else snip}")
    if results.get("min_k"):
        print("\n[Min-K% Membership Inference]")
        for k, v in results["min_k"].items():
            if k != "explanation":
                print(f"  {k}: {v}")
        print(f"  Explanation: {results['min_k']['explanation']}")
    print("\n[Explanations]")
    for ex in results["explanations"]:
        print(f"  - {ex}")
    # Always print the provenance graph structure
    if results.get("prov_graph") is not None:
        print_prov_graph_structure(results["prov_graph"], title="Data Provenance Graph (W3C PROV)")
    if results.get("min_k_subgraph") is not None:
        print_prov_graph_structure(results["min_k_subgraph"], title="Min-K Explanation Subgraph")
    print("=" * 60)


def save_results_json(results: dict[str, Any], path: str = "deem_mink_results.json") -> None:
    """Save results to JSON (excluding non-serializable graph objects)."""
    out = {k: v for k, v in results.items() if k not in ("prov_graph", "min_k_subgraph")}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


def build_experiment_metadata(
    results: dict[str, Any],
    *,
    run_id: str | None = None,
    experiment_tag: str = "deem_mink_run",
    code_snippet: str | None = None,
    code_char_count: int | None = None,
    k_percent: float = 0.1,
    tau: float | None = None,
) -> dict[str, Any]:
    """
    Build metadata JSON in the requested schema:
    provenance_metadata, classification, repository_attribution, feature_analysis, governance.
    """
    import re

    cls = results.get("classification") or {}
    probs: dict[str, float] = dict(cls.get("probability_distribution") or {})
    pred_label = str(cls.get("predicted_label") or "human")
    conf = float(cls.get("confidence") or 0.0)
    code_for_stats = (code_snippet or results.get("input_code_preview") or "").strip()
    repo_hits = list(results.get("repository_similarity") or [])

    # -------- IDs / timestamps --------
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if run_id:
        analysis_id = run_id
    else:
        analysis_id = f"prov-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-001"

    # -------- model/provider mapping --------
    model_catalog = {
        "human": ("human", "Human", "Human", "Human"),
        "gpt3.5": ("gpt-3.5-turbo", "OpenAI", "GPT", "gpt-3.5-turbo"),
        "gpt4": ("gpt-4o", "OpenAI", "GPT", "gpt-4o"),
        "gpt5": ("gpt-5.2", "OpenAI", "GPT", "gpt-5.2"),
    }
    model_name, provider, family, canonical = model_catalog.get(pred_label, model_catalog["human"])

    # Alternative candidates: top-2 excluding predicted (including human when competitive)
    alts = []
    sorted_candidates = sorted(
        [(k, float(v)) for k, v in probs.items() if k != pred_label and k in model_catalog],
        key=lambda x: -x[1],
    )
    for k, p in sorted_candidates[:2]:
        m_name, prov, _, _ = model_catalog[k]
        alts.append(
            {
                "model_name": m_name,
                "provider": prov,
                "probability": round(float(p), 4),
            }
        )

    # -------- repository attribution --------
    matched_repositories = []
    stack_matches = []
    for h in repo_hits:
        src = str(h.get("source", ""))
        sim = float(h.get("cosine_similarity", 0.0))
        snippet = str(h.get("snippet", ""))
        overlap = _snippet_overlap_score(code_for_stats, snippet)
        match_type = "structural_pattern" if overlap >= 0.22 else "code_snippet_overlap"

        if src == "github":
            name_guess = "repository_match"
            toks = _matched_tokens(code_for_stats, snippet, max_items=3)
            if toks:
                name_guess = "-".join(toks).lower()
            matched_repositories.append(
                {
                    "repository_name": name_guess,
                    "platform": "GitHub",
                    "similarity_score": round(sim, 4),
                    "match_type": match_type,
                }
            )
        elif src == "stackoverflow":
            rank = int(h.get("rank", 0))
            qid = f"SO-{datetime.now(timezone.utc).strftime('%H%M%S')}{rank:02d}"
            stack_matches.append(
                {
                    "question_id": qid,
                    "similarity_score": round(sim, 4),
                }
            )
        else:
            # keep non-stack, non-github evidence as generic repository-style evidence
            matched_repositories.append(
                {
                    "repository_name": "external_web_match",
                    "platform": "Web",
                    "similarity_score": round(sim, 4),
                    "match_type": match_type,
                }
            )

    # -------- feature analysis --------
    lines = [ln for ln in code_for_stats.splitlines() if ln.strip()]
    n_lines = max(1, len(lines))
    func_blocks = re.findall(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", code_for_stats)
    avg_fn_len = (n_lines / max(1, len(func_blocks))) if func_blocks else float(n_lines)
    comment_lines = sum(1 for ln in lines if ln.strip().startswith("#"))
    comment_density = comment_lines / n_lines
    has_doc = ('"""' in code_for_stats) or ("'''" in code_for_stats)
    docstyle = "LLM-like" if has_doc or comment_density >= 0.20 else "minimal"
    var_tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code_for_stats)
    long_names = [v for v in var_tokens if len(v) >= 8]
    naming = "descriptive_generic" if len(long_names) >= max(3, len(var_tokens) // 12) else "compact_manual"

    has_try = "try:" in code_for_stats and "except" in code_for_stats
    err_style = "explicit try-except blocks" if has_try else "implicit/no explicit handling"
    imports = len(re.findall(r"^\s*(from|import)\s+", code_for_stats, flags=re.MULTILINE))
    defs = len(re.findall(r"^\s*def\s+", code_for_stats, flags=re.MULTILINE))
    modularity = max(0.0, min(1.0, 0.35 + 0.08 * defs + 0.04 * imports))
    boilerplate = max(0.0, min(1.0, 0.25 + 0.45 * (1.0 - min(1.0, comment_density + 0.15)) + 0.2 * float(has_doc)))

    # -------- governance --------
    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",                 # SSN-like
        r"\b[0-9]{10,}\b",                        # long numeric ids/phones
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # emails
    ]
    pii_detected = any(re.search(p, code_for_stats) for p in pii_patterns)
    license_risk = any(x in code_for_stats.lower() for x in ("copyright", "all rights reserved", "proprietary"))
    audit_log_id = f"audit-{datetime.now(timezone.utc).strftime('%H%M%S')}{len(code_for_stats):04d}"

    metadata = {
        "provenance_metadata": {
            "analysis_id": analysis_id,
            "timestamp": ts,
            "input_type": "source_code",
            "language_detected": "python",
        },
        "classification": {
            "is_llm_generated": bool(cls.get("is_llm_generated", pred_label != "human")),
            "confidence_score": round(conf, 4),
            "predicted_model": {
                "model_name": model_name,
                "provider": provider,
                "model_family": family,
                "generation_probability": round(float(probs.get(pred_label, conf)), 4),
            },
            "alternative_candidates": alts,
        },
        "repository_attribution": {
            "matched_repositories": matched_repositories[:5],
            "stack_overflow_matches": stack_matches[:5],
        },
        "feature_analysis": {
            "stylometric_features": {
                "average_function_length": round(float(avg_fn_len), 2),
                "comment_density": round(float(comment_density), 2),
                "docstring_style": docstyle,
                "variable_naming_pattern": naming,
            },
            "semantic_patterns": {
                "error_handling_style": err_style,
                "modularity_score": round(float(modularity), 2),
                "boilerplate_probability": round(float(boilerplate), 2),
            },
        },
        "governance": {
            "pii_detected": bool(pii_detected),
            "license_risk_flag": bool(license_risk),
            "audit_log_id": audit_log_id,
            "compliance_status": "research_use_only",
        },
    }
    return metadata


def save_experiment_metadata_json(
    metadata: dict[str, Any],
    path: str = "deem_mink_metadata.json",
) -> None:
    """Save experiment metadata (meta-dataset) to a downloadable JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {path}")


# ---------------------------------------------------------------------------
# Example and Colab entry point
# ---------------------------------------------------------------------------

EXAMPLE_CODE = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    for i in range(10):
        print(fibonacci(i))
'''


def run_example(
    code_snippet: str | None = None,
    export_png: bool = True,
    save_json: bool = True,
    show_pyvis: bool = True,
    use_repo_api: bool = True,
    github_token: str | None = None,
) -> dict[str, Any] | None:
    """
    Run the full pipeline on user-provided code. Prompts for code if code_snippet is None.
    Returns None if no code is provided.
    """
    print("Installing / loading dependencies...")
    try:
        import torch
        import transformers
        import networkx
    except ImportError:
        install_dependencies()
        import torch
        import transformers
        import networkx

    code = get_user_code_input(code_snippet) if code_snippet is None else code_snippet.strip()
    if not code:
        print("No code provided. Exiting. Run again and paste code or pass code_snippet=... or a file path as sys.argv[1].")
        return None

    print("Running pipeline on your code...")
    results = run_full_pipeline(
        code,
        k_percent=0.1,
        top_k_similar=5,
        use_repo_api=use_repo_api,
        github_token=github_token,
    )
    print_pipeline_output(results)

    # Show provenance graph(s) directly in notebook/Colab (no download)
    if _in_notebook():
        print("\n--- Data Provenance Graph (displayed below) ---\n")
        try:
            display_prov_graph_inline(results["prov_graph"], title="Data Provenance Graph (W3C PROV)")
        except Exception as e:
            print("Provenance graph display failed:", e)
        if results.get("min_k_subgraph") is not None:
            print("\n--- Min-K Explanation Subgraph (displayed below) ---\n")
            try:
                display_prov_graph_inline(results["min_k_subgraph"], title="Min-K Explanation Subgraph")
            except Exception as e:
                print("Min-K subgraph display failed:", e)
    else:
        # Not in notebook: save HTML and optionally open in browser
        if show_pyvis:
            try:
                net = draw_prov_graph_pyvis(results["prov_graph"], title="DEEM_MinK PROV Graph")
                net.save_graph("prov_graph.html")
                print("Interactive graph saved to prov_graph.html")
                show_graph_graphical(results, html_path="prov_graph.html", open_browser=True)
            except Exception as e:
                print("Pyvis HTML save skipped:", e)
            if results.get("min_k_subgraph") is not None:
                try:
                    net = draw_prov_graph_pyvis(results["min_k_subgraph"], title="Min-K Subgraph", height="400px")
                    net.save_graph("min_k_subgraph.html")
                except Exception:
                    pass

    if export_png:
        try:
            export_prov_graph_png(results["prov_graph"], "prov_graph.png")
        except Exception as e:
            print("PNG export skipped:", e)
    if save_json:
        save_results_json(results)
        try:
            metadata = build_experiment_metadata(
                results,
                experiment_tag="deem_mink_run",
                code_snippet=code,
                code_char_count=len(code),
                k_percent=0.1,
                tau=None,
            )
            save_experiment_metadata_json(metadata, path="deem_mink_metadata.json")
        except Exception as e:
            print("Metadata JSON export skipped:", e)
    return results


# ---------------------------------------------------------------------------
# Optional: Gradio interface
# ---------------------------------------------------------------------------

def launch_gradio():
    """Launch a simple Gradio UI for code input and pipeline output."""
    try:
        import gradio as gr  # type: ignore[import-untyped]
    except ImportError:
        install_dependencies()
        import gradio as gr  # type: ignore[import-untyped]

    embedding_model = None

    def load_models():
        nonlocal embedding_model
        if embedding_model is None:
            embedding_model = load_embedding_model()
        return "Models loaded (classification uses OpenRouter API ensemble)."

    def analyze(code: str):
        if not code.strip():
            return "Please enter Python code.", None, None
        load_models()
        results = run_full_pipeline(
            code,
            embedding_model=embedding_model,
            repo_data=None,
            use_repo_api=True,
        )
        text = "\n".join(results["explanations"])
        c = results["classification"]
        summary = f"Predicted: {c.get('predicted_label_display', c['predicted_label'])} ({c['confidence']:.2%}) | LLM: {c['is_llm_generated']}"
        try:
            net = draw_prov_graph_pyvis(results["prov_graph"])
            html = net.generate_html()
        except Exception:
            html = "<p>Graph could not be generated.</p>"
        return summary + "\n\n" + text, results.get("min_k", {}), html

    with gr.Blocks(title="DEEM_MinK") as demo:
        gr.Markdown("# DEEM_MinK: Code Attribution & Min-K Analysis")
        gr.Markdown(REPO_DISCLAIMER)
        with gr.Row():
            code_in = gr.Textbox(
                label="Python code snippet",
                placeholder="Paste your Python code here...",
                lines=10,
            )
        btn = gr.Button("Analyze")
        out_text = gr.Textbox(label="Result", lines=12)
        out_min_k = gr.JSON(label="Min-K result (if LLM)")
        out_graph = gr.HTML(label="Provenance graph")
        btn.click(fn=analyze, inputs=code_in, outputs=[out_text, out_min_k, out_graph])
    demo.launch()


# ---------------------------------------------------------------------------
# Graphical display: browser + Colab inline
# ---------------------------------------------------------------------------

def _in_notebook() -> bool:
    """True if running inside Jupyter/Colab (IPython)."""
    try:
        get_ipython = __import__("IPython", fromlist=["get_ipython"]).get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def open_graph_in_browser(html_path: str = "prov_graph.html") -> None:
    """Open the saved HTML graph in the default system browser."""
    import os
    import webbrowser
    path = os.path.abspath(html_path)
    if os.path.isfile(path):
        url = "file:///" + path.replace("\\", "/")
        try:
            webbrowser.open(url)
            print("Opening graph in your default browser:", path)
        except Exception as e:
            print("Could not open browser:", e)
            print("  Open the file manually:", path)
    else:
        print("Graph file not found:", path)


def display_prov_in_colab(results: dict[str, Any]) -> None:
    """In Google Colab / Jupyter, display the PROV graph inline so you see it in the notebook."""
    try:
        from IPython.display import HTML, display
        net = draw_prov_graph_pyvis(results["prov_graph"], title="DEEM_MinK PROV Graph")
        html = net.generate_html()
        display(HTML(html))
        if results.get("min_k_subgraph") is not None:
            net2 = draw_prov_graph_pyvis(results["min_k_subgraph"], title="Min-K Subgraph", height="400px")
            display(HTML(net2.generate_html()))
    except Exception as e:
        print("Inline display skipped (not in Colab or pyvis missing):", e)


def show_graph_graphical(results: dict[str, Any], html_path: str = "prov_graph.html", open_browser: bool = True) -> None:
    """
    Show the provenance graph in a graphical way:
    - In Jupyter/Colab: display inline in the notebook.
    - Otherwise: open the saved HTML file in the default browser.
    """
    if _in_notebook():
        display_prov_in_colab(results)
    elif open_browser:
        open_graph_in_browser(html_path)


# ---------------------------------------------------------------------------
# Human code crawling from GitHub (pre-2021 snapshots) for dataset building
# ---------------------------------------------------------------------------

def _github_api_get(url: str, token: str | None = None, params: dict[str, Any] | None = None) -> Any:
    """GET request helper for GitHub REST API with optional token."""
    import requests

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "DEEM-MinK-HumanCodeCrawler",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, params=params, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text[:300]}")
    return response.json()


def _decode_github_content(content_obj: dict[str, Any]) -> str:
    """Decode base64 content payload from GitHub contents API."""
    import base64

    raw = content_obj.get("content", "")
    if not raw:
        return ""
    return base64.b64decode(raw).decode("utf-8", errors="ignore")


def _get_file_commit_before_cutoff(
    repo_full_name: str,
    file_path: str,
    cutoff_iso: str,
    token: str | None = None,
) -> dict[str, Any] | None:
    """
    Return the latest commit for file_path in repo_full_name at/before cutoff_iso.
    cutoff_iso example: '2020-12-31T23:59:59Z'
    """
    commits_url = f"https://api.github.com/repos/{repo_full_name}/commits"
    params = {"path": file_path, "until": cutoff_iso, "per_page": 1}
    commits = _github_api_get(commits_url, token=token, params=params)
    if not commits:
        return None
    return commits[0]


def _fetch_file_content_at_commit(
    repo_full_name: str,
    file_path: str,
    commit_sha: str,
    token: str | None = None,
) -> str:
    """Fetch repository file content at a specific commit SHA."""
    contents_url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
    content_obj = _github_api_get(contents_url, token=token, params={"ref": commit_sha})
    return _decode_github_content(content_obj)


def _looks_like_assignment_python(code: str, min_lines: int = 15, max_lines: int = 120) -> bool:
    """Heuristic filter for short assignment-like Python programs."""
    import re

    if not code.strip():
        return False
    if "```" in code:
        return False

    lines = [ln for ln in code.splitlines() if ln.strip()]
    if len(lines) < min_lines or len(lines) > max_lines:
        return False

    # Avoid likely non-source payloads.
    bad_tokens = ["<html", "{", "package.json", "import pandas as pd"]
    lc = code.lower()
    if any(tok in lc for tok in bad_tokens):
        return False

    # Must contain code-like structures common in assignments.
    if not re.search(r"\b(def|class|for|while|if|print|return)\b", code):
        return False

    return True


def crawl_human_python_pre2021_from_github(
    target_count: int = 55,
    output_dir: str = "older_human_pre2021_55_programs",
    cutoff_iso: str = "2020-12-31T23:59:59Z",
    github_token: str | None = None,
    sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    """
    Crawl assignment-style human Python programs from GitHub with pre-2021 commit snapshots.

    Notes:
    - This cannot mathematically guarantee "human-written", but pre-2021 snapshots and public repo history
      provide strong practical evidence for human authorship.
    - Requires network access in Colab.
    """
    import csv
    import hashlib
    import os
    import time
    import zipfile
    from pathlib import Path

    keyword_queries = [
        "longest common subsequence lcs",
        "palindrome string",
        "string reverse words",
        "character frequency python",
        "recursion factorial",
        "fibonacci recursion memoization",
        "tower of hanoi",
        "binary search iterative recursive",
        "stack implementation python",
        "queue implementation python",
        "postfix expression evaluation",
        "balanced parentheses stack",
        "linked list implementation",
        "merge sort python",
        "quick sort python",
        "anagram string",
        "first non repeating character",
        "lru cache ordereddict",
        "infix to postfix",
        "bfs graph traversal",
        "dfs graph traversal",
        "shortest path grid bfs",
        "distinct substrings string",
        "two sum python",
        "recursion backtracking subset",
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    collected = 0
    seen_hashes: set[str] = set()
    seen_repo_path: set[str] = set()
    metadata_rows: list[dict[str, Any]] = []

    # Search multiple pages and multiple keywords to gather enough candidates.
    max_pages_per_query = 5
    per_page = 50

    for kw in keyword_queries:
        if collected >= target_count:
            break

        for page in range(1, max_pages_per_query + 1):
            if collected >= target_count:
                break

            search_url = "https://api.github.com/search/code"
            q = f'{kw} language:Python extension:py size:20..20000 -generated -vendor'
            params = {"q": q, "per_page": per_page, "page": page}

            try:
                result = _github_api_get(search_url, token=github_token, params=params)
            except Exception as e:
                print(f"Search failed for '{kw}' page {page}: {e}")
                time.sleep(sleep_seconds)
                continue

            items = result.get("items", [])
            if not items:
                break

            for item in items:
                if collected >= target_count:
                    break

                repo_full_name = item.get("repository", {}).get("full_name", "")
                file_path = item.get("path", "")
                html_url = item.get("html_url", "")
                if not repo_full_name or not file_path:
                    continue

                repo_key = f"{repo_full_name}:{file_path}"
                if repo_key in seen_repo_path:
                    continue
                seen_repo_path.add(repo_key)

                try:
                    commit = _get_file_commit_before_cutoff(
                        repo_full_name=repo_full_name,
                        file_path=file_path,
                        cutoff_iso=cutoff_iso,
                        token=github_token,
                    )
                    if not commit:
                        continue

                    commit_sha = commit.get("sha", "")
                    commit_info = commit.get("commit", {})
                    commit_date = commit_info.get("author", {}).get("date", "")
                    author_name = commit_info.get("author", {}).get("name", "")
                    if not commit_sha:
                        continue

                    code = _fetch_file_content_at_commit(
                        repo_full_name=repo_full_name,
                        file_path=file_path,
                        commit_sha=commit_sha,
                        token=github_token,
                    )
                    if not _looks_like_assignment_python(code):
                        continue

                    # Deduplicate by normalized content hash.
                    normalized = "\n".join(line.rstrip() for line in code.splitlines()).strip()
                    code_hash = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
                    if code_hash in seen_hashes:
                        continue
                    seen_hashes.add(code_hash)

                    collected += 1
                    filename = f"{collected}.py"
                    out_file = output_path / filename
                    out_file.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")

                    metadata_rows.append(
                        {
                            "id": collected,
                            "file_name": filename,
                            "repo_full_name": repo_full_name,
                            "file_path": file_path,
                            "source_html_url": html_url,
                            "commit_sha": commit_sha,
                            "commit_author_name": author_name,
                            "commit_date_utc": commit_date,
                            "cutoff_iso": cutoff_iso,
                            "content_sha256": code_hash,
                            "keyword_query": kw,
                        }
                    )
                    print(f"[{collected}/{target_count}] saved {filename} from {repo_full_name}/{file_path}")
                except Exception as e:
                    print(f"Skip {repo_full_name}/{file_path}: {e}")

                time.sleep(sleep_seconds)

    # Save metadata CSV and JSON.
    metadata_csv = output_path / "metadata.csv"
    metadata_json = output_path / "metadata.json"
    fieldnames = [
        "id",
        "file_name",
        "repo_full_name",
        "file_path",
        "source_html_url",
        "commit_sha",
        "commit_author_name",
        "commit_date_utc",
        "cutoff_iso",
        "content_sha256",
        "keyword_query",
    ]
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    metadata_json.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")

    # Zip for easy Colab download.
    zip_path = Path(f"{output_path.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(output_path.parent))

    print("\nCrawling complete.")
    print(f"Collected files: {collected}")
    print(f"Folder: {output_path.resolve()}")
    print(f"Metadata: {metadata_csv.resolve()} and {metadata_json.resolve()}")
    print(f"Zip: {zip_path.resolve()}")

    # Auto-download in Colab if available.
    try:
        from google.colab import files  # type: ignore

        files.download(str(zip_path))
    except Exception:
        pass

    # Optional warning when target is not reached.
    if collected < target_count:
        print(
            f"Warning: only collected {collected}/{target_count}. "
            "Use a GitHub token and/or increase max pages/queries if needed."
        )

    return {
        "collected": collected,
        "target_count": target_count,
        "output_dir": str(output_path.resolve()),
        "metadata_csv": str(metadata_csv.resolve()),
        "metadata_json": str(metadata_json.resolve()),
        "zip_path": str(zip_path.resolve()),
    }


# ---------------------------------------------------------------------------
# How to run in Google Colab
# ---------------------------------------------------------------------------
# 1. Upload this file. In a Colab cell, run:
#    from DEEM_MinK import install_dependencies, run_example, display_prov_in_colab
#    install_dependencies()
# 2. Run the pipeline (you will be prompted to paste code, or pass code explicitly):
#    results = run_example(code_snippet=None)   # prompts for code
#    # OR: results = run_example(code_snippet="def foo(): return 1")
#    # Optional: set os.environ["GITHUB_TOKEN"] for higher GitHub API rate limits
#    if results: display_prov_in_colab(results)
# 3. Optional Gradio UI (no prompt; paste code in the web form):
#    from DEEM_MinK import launch_gradio
#    launch_gradio()
# 4. Crawl human-written Python dataset from GitHub (pre-2021 snapshots):
#    import os
#    from DEEM_MinK import crawl_human_python_pre2021_from_github
#    # Optional but recommended to avoid strict rate limits:
#    # os.environ["GITHUB_TOKEN"] = "ghp_xxx"
#    out = crawl_human_python_pre2021_from_github(
#        target_count=55,
#        output_dir="older_human_pre2021_55_programs",
#        cutoff_iso="2020-12-31T23:59:59Z",
#        github_token=os.environ.get("GITHUB_TOKEN"),
#    )
#    out
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Prompts for code (or use file path as sys.argv[1], or pass code_snippet=... in code)
    results = run_example(export_png=True, save_json=True, show_pyvis=True)
    if results:
        pass  # Optional: display_prov_in_colab(results) in Colab

    # Uncomment to launch Gradio in Colab:
    # launch_gradio()
