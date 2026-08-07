"""
MoodMirror -- Gradio app (Phase 0 skeleton).

A reflective companion: it looks at your face and reads a line you type, then
offers an expressed-emotion reading, a clearly-labeled traditional "face
reading" (folklore, for fun), and live recommendations -- and it notices when
your face, your words, and the tradition disagree.

PHASE 0: the UI and the full pipeline wiring exist; every model layer is a stub
(see core/*.py). You can run the app right now, upload a photo or grab a webcam
frame, type a line, and watch state flow through all six layers into the panels.

IMPORTANT -- launch speed: this module imports ONLY gradio + stdlib + our own
light modules at the top. The heavy libraries (tensorflow, mediapipe,
transformers) are imported lazily inside core/*.py functions, so the window
appears instantly instead of hanging on a 30s TensorFlow import.
"""

from __future__ import annotations

import gradio as gr

from state import MirrorState
from core import perception, features, text_emotion, congruence, generation, recommend
from feedback import store

CONSENT = (
    "**Privacy:** MoodMirror processes your face for signals only. "
    "No image is stored -- only the derived reading. The traditional face "
    "reading below is **folklore, for fun, not science.**"
)


def run_pipeline(image, text: str):
    """
    The whole six-layer flow (plan §5), on one button press.
    Returns strings for each panel. `image` is a numpy array from Gradio, or None.
    """
    state = MirrorState()

    # L1 + L2 -- perception & features (from the image, if any)
    state.face = perception.analyze_face(image)
    state.features = features.extract_features(image)

    # L3 -- text emotion (from the typed line, if any)
    state.text = text_emotion.analyze_text(text)

    # L4 -- congruence (pure logic over what we have)
    state.congruence = congruence.assess(state.face, state.text)

    # L5 -- reasoning & generation (the three readings + a rec spec)
    state.readings = generation.generate(state)

    # L6 -- live recommendations (from the spec)
    state.recommendations = recommend.recommend(state.readings.recommendation_spec)

    # Log derived state only (no image). Best-effort; never break the UI --
    # but surface the reason to stderr so a broken Phase 5 log is debuggable
    # rather than silently doing nothing.
    try:
        store.log_interaction(state.to_dict())
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"[MoodMirror] feedback log failed: {type(e).__name__}: {e}",
              file=sys.stderr)

    return _render(state)


def _render(s: MirrorState):
    # --- Emotion + age ---
    if s.face.available:
        emo = (
            f"**Expressed emotion:** {s.face.emotion} "
            f"(confidence {s.face.emotion_confidence:.0%})"
        )
        if s.face.age_low is not None and s.face.age_high is not None:
            emo += f"\n\n**Age (a fuzzy guess):** roughly {s.face.age_low}–{s.face.age_high}"
        elif s.face.note:
            emo += f"\n\n_{s.face.note}_"
    else:
        emo = f"_No face read._ {s.face.note}"

    # --- Emotional reading ---
    reading = s.readings.emotional_reading or f"_{s.readings.note}_"

    # --- Congruence ---
    if s.congruence.available and s.congruence.verdict != "insufficient":
        cong = f"**{s.congruence.verdict.title()}** — {s.congruence.explanation}"
    else:
        cong = f"_{s.congruence.explanation or s.congruence.note}_"

    # --- Traditional (folklore) ---
    if s.readings.traditional_reading:
        trad = s.readings.traditional_reading
    elif not s.features.available and s.features.note:
        # e.g. model not downloaded, or no face found -- say why, don't tell
        # someone who uploaded a photo to "add a photo".
        trad = f"_{s.features.note}_"
    else:
        trad = "_Add a photo for a traditional reading._"

    # --- Recommendations ---
    r = s.recommendations
    if r.available:
        def _link(text, url):
            return f"[{text}]({url})" if url else text

        parts = []
        if r.music:
            parts.append("**🎵 Music:** " + " · ".join(
                _link(f"{m['title']} — {m.get('artist','')}", m.get("url", ""))
                for m in r.music))
        if r.movies:
            parts.append("**🎬 Movies:** " + " · ".join(
                f"{m['title']} ({m.get('year','')})" for m in r.movies))
        if r.papers:
            parts.append("**📄 Papers:** " + " · ".join(
                _link(p["title"], p.get("url", "")) for p in r.papers))
        recs = "\n\n".join(parts) if parts else "_No recommendations available right now._"
        # Surface provider failures (e.g. missing key / timeout), not the "live" note.
        if r.note and not r.note.startswith("live"):
            recs += f"\n\n_{r.note}_"
    else:
        recs = f"_{r.note}_"

    return emo, reading, cong, trad, recs


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MoodMirror") as demo:
        gr.Markdown("# MoodMirror")
        gr.Markdown(CONSENT)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Look & tell")
                # Upload is the primary demo path (plan §3); webcam optional.
                image_in = gr.Image(sources=["upload", "webcam"], type="numpy",
                                    label="Photo or webcam frame")
                text_in = gr.Textbox(label="Say a line about how you feel",
                                     placeholder="Type anything...", lines=2)
                go = gr.Button("Reflect", variant="primary")

            with gr.Column(scale=1):
                out_emotion = gr.Markdown(label="Emotion & age")
                out_reading = gr.Markdown(label="Emotional reading")
                out_congruence = gr.Markdown(label="Congruence")
                out_traditional = gr.Markdown(label="Traditional face reading")
                out_recs = gr.Markdown(label="For you right now")

        go.click(
            run_pipeline,
            inputs=[image_in, text_in],
            outputs=[out_emotion, out_reading, out_congruence, out_traditional, out_recs],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
