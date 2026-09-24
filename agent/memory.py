"""
Case memory: embeds case text (closed-case narratives + agent-written
summaries) and retrieves similar past cases through
GraphClient.find_similar_cases (cosine similarity, either TigerGraph-side
or client-side depending on backend — see graph/client.py).

Embedding choice: TF-IDF over the case-narrative corpus (scikit-learn),
NOT a downloaded transformer model. This was a deliberate switch during
development: a sentence-transformers model has to be fetched from
huggingface.co on first use, and this development sandbox's network
egress can't reach it (same restriction documented in docs/DECISIONS.md
for tgcloud.io/drive.google.com) — so a network-dependent embedder would
be untestable here and would silently break in any similarly locked-down
deployment. TF-IDF is fully offline, deterministic, reproducible for
judges without any model download, and fits the actual content well:
case narratives are short structured strings (pattern, outcome, exposure,
notes), which lexical overlap handles fine. Kept deliberately simple per
the hackathon spec ("do not overengineer memory") — one fitted vectorizer,
one similarity metric, results always carry their score and are explained
by the caller (never "blindly copy their decision"), never a black box.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

EMBED_DIM = 256
_VECTORIZER_CACHE: Dict[str, Any] = {}


def _vectorizer_path() -> str:
    return os.getenv("TFIDF_VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "..", "graph", "tfidf_vectorizer.pkl"))


def _get_vectorizer():
    path = _vectorizer_path()
    if path in _VECTORIZER_CACHE:
        return _VECTORIZER_CACHE[path]

    if os.path.exists(path):
        with open(path, "rb") as f:
            vec = pickle.load(f)
        _VECTORIZER_CACHE[path] = vec
        return vec

    vec = _fit_vectorizer_from_closed_cases()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vec, f)
    _VECTORIZER_CACHE[path] = vec
    return vec


def _fit_vectorizer_from_closed_cases():
    from sklearn.feature_extraction.text import TfidfVectorizer

    data_dir = os.getenv("DATA_DIR", "./data")
    cc_path = os.path.join(data_dir, "closed_cases_history.csv")
    corpus: List[str] = []
    if os.path.exists(cc_path):
        import pandas as pd

        df = pd.read_csv(cc_path, low_memory=False)
        for _, r in df.iterrows():
            corpus.append(
                case_narrative_text(
                    pattern=r["pattern"],
                    outcome_or_verdict=r["outcome"],
                    exposure_usd=float(r["exposure_usd"]),
                    n_txns=int(r["n_txns"]),
                    analyst_notes=str(r.get("analyst_notes", "")),
                )
            )
    if not corpus:
        # bootstrap corpus so the vectorizer is never empty (test/dev without data/)
        corpus = [
            "pattern: card_testing outcome: confirmed_fraud small authorizations then larger purchase",
            "pattern: out_of_region_use outcome: confirmed_fraud billing region new for this card",
            "pattern: account_takeover outcome: confirmed_fraud mixed channel activity",
            "pattern: none outcome: cleared legitimate recurring purchase",
        ]
    vec = TfidfVectorizer(max_features=EMBED_DIM, stop_words="english")
    vec.fit(corpus)
    return vec


def embed_text(text: str) -> List[float]:
    vec = _get_vectorizer()
    arr = vec.transform([text]).toarray()[0]
    return arr.tolist()


def case_narrative_text(
    pattern: str,
    outcome_or_verdict: str,
    exposure_usd: float,
    n_txns: int,
    analyst_notes: str,
    channel: str = "",
) -> str:
    """Builds the text that gets embedded for a closed/agent case, so
    retrieval is driven by what actually happened, not just IDs."""
    parts = [
        f"pattern: {pattern}",
        f"outcome: {outcome_or_verdict}",
        f"exposure: ${exposure_usd:.2f}",
        f"{n_txns} transaction(s)",
    ]
    if channel:
        parts.append(f"channel: {channel}")
    if analyst_notes:
        parts.append(analyst_notes)
    return " | ".join(parts)


def retrieve_similar_cases(
    graph_client, query_text: str, top_k: int = 5, case_source_filter: str = ""
) -> List[Dict[str, Any]]:
    embedding = embed_text(query_text)
    return graph_client.find_similar_cases(embedding, top_k=top_k, case_source_filter=case_source_filter)
