"""
Layer 3 -- Language understanding (transformers): text emotion / sentiment.

This is one of the two graded cores (plan §5). PHASE 0 STATUS: stub. Phase 2
replaces the body with a real HuggingFace pipeline -- GoEmotions is preferred so
the text labels line up with the face labels for the congruence step.

Rules: transformers/torch imported inside the function (they're slow to load);
empty text degrades to available=False; the loaded pipeline should be cached at
module level in Phase 2 so we don't reload the model every call.
"""

from __future__ import annotations

from state import TextState


def analyze_text(text: str) -> TextState:
    """
    text: the line the user typed.
    Returns a populated TextState with an emotion label + confidence.
    """
    if not text or not text.strip():
        return TextState(available=False, note="No text entered.")

    # ----- PHASE 0 STUB ---------------------------------------------------- #
    # Real Phase 2 body (sketch):
    #   from transformers import pipeline
    #   global _clf
    #   if _clf is None:
    #       _clf = pipeline("text-classification",
    #                       model="SamLowe/roberta-base-go_emotions",
    #                       top_k=None)
    #   scores = _clf(text)[0]        # list of {label, score}
    #   pick the top label, normalize to a small shared label set
    return TextState(
        available=True,
        note="stub",
        text=text.strip(),
        emotion="nervousness",
        emotion_confidence=0.61,
        emotion_scores={"nervousness": 0.61, "sadness": 0.22, "neutral": 0.17},
    )
