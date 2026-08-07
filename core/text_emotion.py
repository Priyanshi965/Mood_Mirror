"""
Layer 3 -- Language understanding (transformers): text emotion / sentiment.

One of the two graded cores (plan §5). PHASE 2: real. Runs a GoEmotions
classifier on the typed line so the text emotion vocabulary lines up with the
face emotion for the congruence step.

Model: SamLowe/roberta-base-go_emotions (28 emotions, multi-label sigmoid).

Two rules learned from the DeepFace phase, applied here:
  * GATED DOWNLOAD. The model is ~500MB. We load it with local_files_only=True
    and, if it isn't cached yet, return available=False with a note -- we never
    let a UI click trigger a 500MB blocking download. Pre-download deliberately
    with scripts/download_weights.py.
  * CACHED PIPELINE. The loaded pipeline is memoized at module level, so we pay
    the load cost once, not on every Reflect click.

transformers/torch are imported inside the loader, never at module top, so
importing this module stays instant.
"""

from __future__ import annotations

import threading

from state import TextState

_MODEL = "SamLowe/roberta-base-go_emotions"
# Streaming can call this concurrently with a Reflect click; serialize inference.
_infer_lock = threading.Lock()

# Module-level cache. None = not tried; False = tried and unavailable; else the
# loaded pipeline. Keeps 500MB out of the per-request path.
_clf: object | None = None
_load_failed = False


def _get_pipeline():
    """Return a cached text-classification pipeline, or None if not downloaded."""
    global _clf, _load_failed
    if _clf is not None:
        return _clf
    if _load_failed:
        return None
    try:
        from transformers import pipeline
        # local_files_only: never reach out to the network from here. If the
        # model isn't cached, this raises and we degrade gracefully.
        _clf = pipeline("text-classification", model=_MODEL,
                        top_k=None, local_files_only=True)
        return _clf
    except Exception:
        _load_failed = True
        return None


def analyze_text(text: str) -> TextState:
    """
    text: the line the user typed.
    Returns a populated TextState with a GoEmotions label + confidence.
    """
    if not text or not text.strip():
        return TextState(available=False, note="No text entered.")

    clf = _get_pipeline()
    if clf is None:
        return TextState(
            available=False,
            text=text.strip(),
            note=("Text-emotion model not downloaded. "
                  "Run: python scripts/download_weights.py"),
        )

    try:
        with _infer_lock:
            out = clf(text.strip())
    except Exception as e:  # noqa: BLE001
        return TextState(available=False, text=text.strip(),
                         note=f"Text-emotion model failed. ({type(e).__name__})")

    # top_k=None returns a list of {label, score}. Depending on version it may be
    # wrapped one level for a single input -- normalize to a flat list.
    if out and isinstance(out[0], list):
        out = out[0]
    scores = {d["label"]: round(float(d["score"]), 4) for d in out}
    # GoEmotions is multi-label (sigmoid) -- scores don't sum to 1; the top one
    # is our dominant label + confidence.
    top = max(out, key=lambda d: d["score"])
    return TextState(
        available=True,
        text=text.strip(),
        emotion=str(top["label"]),
        emotion_confidence=round(float(top["score"]), 4),
        emotion_scores=scores,
    )
