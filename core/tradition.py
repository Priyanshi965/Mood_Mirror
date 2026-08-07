"""
Traditional face-reading retrieval (the RAG step, plan §4).

Joins the geometric features from Layer 2 (core/features.py) to claims in the
folklore corpus (data/face_reading_corpus.json). Keyword lookup is deliberate --
the plan calls a vector store a stretch upgrade; with a tiny curated corpus,
an exact (feature, value) match is both sufficient and fully explainable.

Everything here is FOLKLORE reported as tradition, never asserted as true.
"""

from __future__ import annotations

import json
from typing import Any

import config


def _load_corpus() -> list[dict[str, Any]]:
    """Load the corpus entries. Read per call (the file is tiny) so hand-edits
    take effect without a restart and a transient read error never gets cached
    as a permanently-blank corpus."""
    try:
        data = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
        return [e for e in data.get("entries", []) if "feature" in e and "value" in e]
    except Exception:
        return []


def retrieve(features: dict[str, str]) -> list[dict[str, Any]]:
    """
    features: e.g. {"face_shape": "round", "lips": "full", ...}
    Returns the corpus entries whose (feature, value) match a detected feature.
    Order follows the features dict so the reading reads consistently.
    """
    entries = _load_corpus()
    if not entries:
        return []
    index: dict[tuple[str, str], dict] = {(e["feature"], e["value"]): e for e in entries}
    hits = []
    for feat, val in features.items():
        entry = index.get((feat, str(val)))
        if entry:
            hits.append(entry)
    return hits
