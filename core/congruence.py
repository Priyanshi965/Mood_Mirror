"""
Layer 4 -- Congruence: do the face, the words, and the tradition agree?

This is the standout, original idea of the project (plan §8.1): detecting when
what someone shows, says, and is "told by tradition" pull in different
directions -- e.g. face reads tired, words read stressed, folklore says
strong-willed. The tension is the whole personality of the app.

PHASE 0 STATUS: real-but-simple. Unlike the model layers, congruence is pure
logic over state that's already computed, so we can implement a genuine first
version now: compare the face and text emotion labels, gate on confidence.

Confidence-gating matters: if either signal is weak, we should say "insufficient"
rather than confidently declaring a mismatch. This is why the state contract
carries confidences everywhere.
"""

from __future__ import annotations

from state import FaceState, TextState, CongruenceState

# Valence grouping covering BOTH label vocabularies we compare:
#   * DeepFace face emotions (7): angry, disgust, fear, happy, sad, surprise, neutral
#   * GoEmotions text emotions (28), using the official sentiment grouping from
#     Demszky et al. (positive / negative / ambiguous / neutral).
# Every label a model can emit MUST appear here; anything unlisted falls to
# neutral in _valence(), which would silently turn real incongruence into
# "flat". "Ambiguous" GoEmotions labels (curiosity, surprise, ...) are treated
# as neutral valence on purpose -- they don't carry a clear +/- direction.
_POSITIVE = {
    # GoEmotions positive
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
    # DeepFace
    "happy",
}
_NEGATIVE = {
    # GoEmotions negative
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
    # DeepFace
    "angry", "sad",
}
_NEUTRAL = {
    # GoEmotions neutral + ambiguous, and DeepFace neutral/surprise
    "neutral", "confusion", "curiosity", "realization", "surprise",
}

_MIN_CONF = 0.35  # below this on either side, we don't trust a verdict


def _valence(label: str | None) -> str:
    if label in _POSITIVE:
        return "positive"
    if label in _NEGATIVE:
        return "negative"
    return "neutral"


def assess(face: FaceState, text: TextState) -> CongruenceState:
    if not (face.available and text.available):
        return CongruenceState(
            available=False,
            verdict="insufficient",
            note="Need both a face and typed text to compare.",
        )

    conf = min(face.emotion_confidence, text.emotion_confidence)
    if conf < _MIN_CONF:
        return CongruenceState(
            available=True,
            verdict="insufficient",
            confidence=conf,
            explanation=(
                "Signals are too weak to call -- reading gently rather than "
                "asserting a mismatch."
            ),
        )

    fv, tv = _valence(face.emotion), _valence(text.emotion)

    if fv == "neutral" and tv == "neutral":
        verdict = "flat"
        explanation = "Face and words both read muted, with little emotional signal."
    elif fv == tv:
        verdict = "congruent"
        explanation = (
            f"Face ({face.emotion}) and words ({text.emotion}) point the same "
            f"way -- what's shown matches what's said."
        )
    else:
        verdict = "incongruent"
        explanation = (
            f"Face reads {face.emotion} but the words read {text.emotion} -- a "
            f"gap between what's shown and what's said (possible masking)."
        )

    return CongruenceState(
        available=True,
        verdict=verdict,
        confidence=conf,
        explanation=explanation,
    )
