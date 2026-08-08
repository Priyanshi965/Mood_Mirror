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

_TIMEOUT = 60
_MAX_TOKENS = 430       # emotional reading (a rich short paragraph)
_MAX_TOKENS_TRAD = 800  # folklore reading (intro + every feature + synthesis)
_MAX_TOKENS_CHAR = 430  # "most like you" matches

# Generic phrases that make writing read as AI-generated -- banned in prompts.
_BANNED = ("shimmer, radiate, essence, tapestry, journey, testament, delve, "
           "beacon, symphony, dance of, whisper of, tell a story, weave, "
           "in conclusion, it's important to note")

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
    if face.available and face.age_estimate is not None:
        facts.append(f"- Rough age impression (fuzzy): about {face.age_estimate}")
    if text.available and text.text:
        facts.append(f'- They typed: "{text.text}"')
        if text.emotion:
            facts.append(f"- Text emotion: {text.emotion} "
                         f"(confidence {text.emotion_confidence:.0%})")
    if cong.available and cong.verdict != "insufficient":
        facts.append(f"- Face vs words: {cong.verdict} -- {cong.explanation}")

    system = (
        "You are MoodMirror, a warm, perceptive reflective companion. You read a "
        "person's MOMENTARY expressed emotion -- never their character or worth. "
        "Write a rich reflection of TWO short paragraphs (about 5-7 sentences "
        "total), second person, confident and specific. Paragraph one: name what "
        "their face and words show and how those two line up or pull apart -- if "
        "they disagree, say so plainly as real, worth-noticing masking. Paragraph "
        "two: go a little deeper into the texture of this moment and end on a "
        "grounded, kind note. Do NOT hedge ('maybe/might/perhaps/it seems/if'). "
        f"Do NOT use these worn AI phrases: {_BANNED}. Write like a thoughtful "
        "human, not a chatbot -- concrete, unfussy, a little vivid. No lists, no "
        "headings, no preamble."
    )
    user = ("Here is what I read right now:\n" + "\n".join(facts) +
            "\n\nWrite the two-paragraph reflection, confidently.")
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
        f"- {e['feature'].replace('_',' ')} ({e['value']}) — {e['tradition']}: {e['claim']}"
        for e in retrieved
    )
    system = (
        "You are a sharp, charming traditional face reader giving a fun folklore "
        "reading. Entertainment only: you report what named traditions (Chinese "
        "mian xiang, Western physiognomy) SAY about a face, never as scientific "
        "truth. This person's face HAS the listed features -- speak DIRECTLY and "
        "with confidence, never 'if/maybe/might/those with…'.\n"
        "STRUCTURE your answer as markdown, and make it substantial:\n"
        "1. A vivid one-line opening that sets the scene.\n"
        "2. Then, for EACH listed feature, a short bolded sub-heading like "
        "'**The eyes**' followed by 2-3 sentences: what the tradition claims, "
        "plus a concrete, colourful elaboration. Cover every feature, none skipped.\n"
        "3. A final '**Put together**' paragraph synthesising what this "
        "combination of features suggests as a whole.\n"
        f"Do NOT use these worn AI phrases: {_BANNED}. No stage directions in "
        "asterisks. Confident, specific, a little witty. No disclaimers (added "
        "elsewhere)."
    )
    user = ("Read this face. It has these features, and here is what each tradition "
            "says -- write the full structured reading covering ALL of them:\n" + claims)
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
                      "max_tokens": _MAX_TOKENS_TRAD, "temperature": 0.9},
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


def _character_messages(state: MirrorState, retrieved: list[dict]) -> list[dict]:
    # Base the matches on the FACE READING (the folklore personality traits),
    # not the momentary mood -- "which character are you" is about who you are.
    if retrieved:
        traits = "; ".join(e["claim"] for e in retrieved)
        profile = f"their traditional face reading says: {traits}"
    elif state.features.available and state.features.features:
        profile = "face-reading features: " + ", ".join(
            f"{k} {v}" for k, v in state.features.features.items())
    else:
        profile = "a calm, balanced, adaptable presence"

    gender = state.face.gender if state.face.available else None
    gender_rule = ""
    if gender:
        gender_rule = (f" This person presents as a {gender}; the Character, "
                       f"Cartoon/Disney, and Star MUST be {gender} — do not pick "
                       f"the opposite gender.")
    system = (
        "You are a playful, culturally-savvy 'which character are you' matcher, "
        "like a fun personality quiz. Given someone's current mood and "
        "face-reading, name who they are most like RIGHT NOW. Output markdown with "
        "EXACTLY these five lines, each a bold label then a full vivid, confident "
        "sentence of why (name specific, real titles/people):\n"
        "**🎭 Character:** <a fictional film or book character> — <why>\n"
        "**🌈 Cartoon / Disney:** <an animated or Disney character> — <why>\n"
        "**🎵 Song:** <a real song, 'Title' by Artist> — <why>\n"
        "**🌟 Star:** <a real actor or actress> — <why>\n"
        "**✨ Your vibe in a line:** <one punchy sentence capturing them overall>\n"
        f"Confident and fun, for entertainment. Do NOT use worn AI phrases: "
        f"{_BANNED}. No intro or outro -- just the five lines in that format."
        + gender_rule
    )
    user = f"Match this person based on their face reading: {profile}."
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _character_match(state: MirrorState, retrieved: list[dict]) -> str:
    """Playful 'most like you' match, based on the face reading. Empty on failure."""
    if not config.llm_ready():
        return ""
    llm = config.active_llm()
    try:
        import requests
        resp = requests.post(
            f"{llm['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {llm['api_key']}", **llm["headers"]},
            json={"model": llm["model"], "messages": _character_messages(state, retrieved),
                  "max_tokens": _MAX_TOKENS_CHAR, "temperature": 0.95},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
        return content.strip() if content else ""
    except Exception:  # noqa: BLE001
        return ""


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

    # "Most like you" -- playful match, based on the face reading (folklore traits).
    character = _character_match(state, retrieved)

    shown = face.emotion if face.available else None
    said = text.emotion if text.available else None
    spec = {"mood": _valence_hint(shown or "unclear", said),
            # Specific dominant emotion drives varied, personal recommendations.
            "emotion": (said or shown or "neutral"),
            "categories": ["music", "movies", "papers"]}

    return Readings(
        available=True,
        note=source,
        emotional_reading=reading,
        traditional_reading=traditional,
        retrieved_tradition=retrieved,
        character_match=character,
        recommendation_spec=spec,
    )


def _valence_hint(shown: str, said: str | None) -> str:
    """Very rough mood hint for the rec spec -- Layer 6 refines it."""
    neg = {"sad", "angry", "fear", "nervousness", "disgust", "sadness",
           "anger", "grief", "disappointment", "remorse"}
    if shown in neg or (said in neg):
        return "low"
    return "steady"
