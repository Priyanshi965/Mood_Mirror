"""
Layer 5 -- Reasoning & generation (LLM): the three readings + a rec spec.

The second graded core (plan §5). The LLM receives the FULL state -- age, face
emotion + confidence, features + retrieved tradition, text, text sentiment,
congruence -- and returns three things:
  1. the emotional reading (unlimited, hedged by confidence)
  2. the traditional reading (folklore-labeled, woven from retrieved corpus)
  3. a structured recommendation spec (categories + search params)

PHASE 0 STATUS: stub. `generate` returns a deterministic placeholder built from
the state so the panels show something coherent and the confidence-aware hedging
is visible even before an LLM is wired. Phase 2/3 replace the body with a real
call via llm_client (Groq / Gemini / Ollama), keeping this signature.

Retrieval of the folklore corpus (the RAG step, plan §4) is a Phase 3 concern
and will populate readings.retrieved_tradition before this generation step.
"""

from __future__ import annotations

from state import MirrorState, Readings


def _hedge(confidence: float) -> str:
    """Confidence-aware hedging (plan §8.3): softer language when unsure."""
    if confidence >= 0.66:
        return "It looks like"
    if confidence >= 0.4:
        return "You might be"
    return "If I had to guess, maybe"


def generate(state: MirrorState) -> Readings:
    face, text, cong = state.face, state.text, state.congruence

    if not face.available and not text.available:
        return Readings(available=False, note="Nothing to read yet -- add a face or a line of text.")

    # ----- PHASE 0 STUB ---------------------------------------------------- #
    # Real body (sketch): build a prompt from state.to_dict(), inject any
    # feedback-guided few-shot examples (Phase 5), call the LLM, parse into the
    # three fields below.

    # Emotional reading -- hedged by the strongest available signal.
    conf = max(face.emotion_confidence, text.emotion_confidence)
    lead = _hedge(conf)
    shown = face.emotion if face.available else "unclear"
    said = text.emotion if text.available else None
    emotional = f"{lead} feeling {shown} right now."
    if said:
        emotional += f" Your words carry a note of {said}."
    if cong.available and cong.verdict != "insufficient":
        emotional += " " + cong.explanation

    # Traditional reading -- folklore-labeled placeholder. Real version weaves
    # readings.retrieved_tradition (RAG output) into prose.
    if state.features.available:
        feats = ", ".join(f"{k}: {v}" for k, v in state.features.features.items())
        traditional = (
            "Traditional face reading -- for fun, not science. "
            f"Tradition would note your {feats}. (Folklore claims get woven in "
            "here once the corpus retrieval lands in Phase 3.)"
        )
    else:
        traditional = ""

    spec = {
        "mood": _valence_hint(shown, said),
        "categories": ["music", "movies", "papers"],
    }

    return Readings(
        available=True,
        note="stub",
        emotional_reading=emotional,
        traditional_reading=traditional,
        retrieved_tradition=[],
        recommendation_spec=spec,
    )


def _valence_hint(shown: str, said: str | None) -> str:
    """Very rough mood hint for the rec spec -- Layer 6 refines it."""
    neg = {"sad", "angry", "fear", "nervousness", "disgust", "sadness"}
    if shown in neg or (said in neg):
        return "low"
    return "steady"
