"""
Layer 5 -- Reasoning & generation (LLM): the three readings + a rec spec.

The second graded core (plan §5). The LLM receives the full state -- age, face
emotion + confidence, text + text emotion + confidence, congruence -- and writes
the emotional reading (unlimited, hedged by confidence).

PHASE 2: real for the emotional reading. It goes through config.active_llm()
(OpenRouter by default), reusing the exact call shape scripts/smoke_llm.py
proved. Two hard-won guards from that testing survive here:
  * generous max_tokens -- reasoning models return empty content when starved;
  * message.content can be None -- we guard it instead of calling .strip().
On any failure, or when no LLM key is configured, we fall back to a deterministic
hedged template so the app stays demoable even if the free model 404s.

Still Phase 3 (unchanged here): the traditional/folklore reading is woven from
retrieved corpus entries; for now it stays on its labeled placeholder.
"""

from __future__ import annotations

import config
from state import MirrorState, Readings
from . import tradition

_TIMEOUT = 45
_MAX_TOKENS = 320  # headroom -- reasoning models emit nothing if starved

# Non-negotiable folklore disclaimer (plan §10). Prepended in CODE so it's always
# present regardless of what the LLM writes.
_FOLKLORE_LABEL = "**Traditional face reading — for fun, not science.** "


def _hedge(confidence: float) -> str:
    """Confidence-aware hedging (plan §8.3): softer language when unsure."""
    if confidence >= 0.66:
        return "It looks like"
    if confidence >= 0.4:
        return "You might be"
    return "If I had to guess, maybe"


def _build_messages(state: MirrorState) -> list[dict]:
    """Turn the state into a chat prompt for the emotional reading."""
    face, text, cong = state.face, state.text, state.congruence

    facts = []
    if face.available and face.emotion:
        facts.append(f"- Face shows: {face.emotion} "
                     f"(confidence {face.emotion_confidence:.0%})")
    if face.available and face.age_low is not None:
        facts.append(f"- Rough age band (a fuzzy guess): {face.age_low}-{face.age_high}")
    if text.available and text.text:
        facts.append(f'- They typed: "{text.text}"')
        if text.emotion:
            facts.append(f"- Text emotion: {text.emotion} "
                         f"(confidence {text.emotion_confidence:.0%})")
    if cong.available and cong.verdict != "insufficient":
        facts.append(f"- Face vs words: {cong.verdict} -- {cong.explanation}")

    system = (
        "You are MoodMirror, a warm, gentle reflective companion. You read a "
        "person's MOMENTARY expressed emotion -- never their character, "
        "personality, or worth. Write 2-3 short sentences, second person, kind "
        "and grounded. HEDGE in proportion to confidence: low confidence means "
        "softer, more tentative language. If face and words disagree, name that "
        "gap gently as something worth noticing, not a diagnosis. No lists, no "
        "clinical tone, no advice unless it flows naturally."
    )
    user = "Here is what I sense right now:\n" + "\n".join(facts) + \
           "\n\nWrite the reflection."
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _llm_reading(state: MirrorState) -> str | None:
    """Call the configured LLM. Returns the reading, or None to fall back."""
    if not config.llm_ready():
        return None
    llm = config.active_llm()
    try:
        import requests
        resp = requests.post(
            f"{llm['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {llm['api_key']}", **llm["headers"]},
            json={"model": llm["model"], "messages": _build_messages(state),
                  "max_tokens": _MAX_TOKENS, "temperature": 0.8},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content")  # can be None for reasoning models
        return content.strip() if content else None
    except Exception:  # noqa: BLE001 -- any failure -> template fallback
        return None


def _traditional_messages(retrieved: list[dict]) -> list[dict]:
    claims = "\n".join(
        f"- {e['tradition']}: {e['claim']} (feature: {e['feature'].replace('_',' ')} = {e['value']})"
        for e in retrieved
    )
    system = (
        "You narrate TRADITIONAL face reading purely as folklore/entertainment, "
        "like a horoscope. You report what named traditions CLAIM about facial "
        "features -- you NEVER assert these claims are true, and you never read "
        "psychology or character as fact. Weave the given claims into 2-3 warm, "
        "playful sentences. Name the tradition(s). Keep the tone light and "
        "clearly non-scientific. No lists."
    )
    user = ("Weave these traditional claims into a short reading:\n" + claims +
            "\n\nRemember: report them as what the tradition says, not as truth.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _template_traditional(retrieved: list[dict]) -> str:
    """Deterministic folklore reading -- fallback when the LLM is unavailable."""
    parts = [f"{e['tradition']} holds that {e['claim']}." for e in retrieved]
    return " ".join(parts)


def _traditional_reading(state: MirrorState) -> tuple[str, list[dict]]:
    """Retrieve folklore for the detected features and weave it (LLM or template)."""
    if not state.features.available:
        return "", []
    retrieved = tradition.retrieve(state.features.features)
    if not retrieved:
        return "", []
    body = None
    if config.llm_ready():
        llm = config.active_llm()
        try:
            import requests
            resp = requests.post(
                f"{llm['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {llm['api_key']}", **llm["headers"]},
                json={"model": llm["model"], "messages": _traditional_messages(retrieved),
                      "max_tokens": _MAX_TOKENS, "temperature": 0.9},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content")
            body = content.strip() if content else None
        except Exception:  # noqa: BLE001
            body = None
    if not body:
        body = _template_traditional(retrieved)
    # Disclaimer guaranteed by code, not by the model.
    return _FOLKLORE_LABEL + body, retrieved


def _template_reading(state: MirrorState) -> str:
    """Deterministic hedged reading -- the fallback when the LLM is unavailable."""
    face, text, cong = state.face, state.text, state.congruence
    conf = max(face.emotion_confidence, text.emotion_confidence)
    lead = _hedge(conf)
    shown = face.emotion if face.available else "unclear"
    said = text.emotion if text.available else None
    out = f"{lead} feeling {shown} right now."
    if said:
        out += f" Your words carry a note of {said}."
    if cong.available and cong.verdict != "insufficient":
        out += " " + cong.explanation
    return out


def generate(state: MirrorState) -> Readings:
    face, text = state.face, state.text
    if not (face.available or text.available or state.features.available):
        return Readings(available=False,
                        note="Nothing to read yet -- add a face or a line of text.")

    # Emotional reading: LLM first, deterministic template as fallback.
    reading = _llm_reading(state)
    source = "llm"
    if not reading:
        reading = _template_reading(state)
        source = "template"

    # Traditional reading -- folklore woven from retrieved corpus entries (RAG).
    traditional, retrieved = _traditional_reading(state)

    shown = face.emotion if face.available else "unclear"
    said = text.emotion if text.available else None
    spec = {"mood": _valence_hint(shown, said),
            "categories": ["music", "movies", "papers"]}

    return Readings(
        available=True,
        note=source,
        emotional_reading=reading,
        traditional_reading=traditional,
        retrieved_tradition=retrieved,
        recommendation_spec=spec,
    )


def _valence_hint(shown: str, said: str | None) -> str:
    """Very rough mood hint for the rec spec -- Layer 6 refines it."""
    neg = {"sad", "angry", "fear", "nervousness", "disgust", "sadness",
           "anger", "grief", "disappointment", "remorse"}
    if shown in neg or (said in neg):
        return "low"
    return "steady"
