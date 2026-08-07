"""
MoodMirror -- Gradio app (live video).

A reflective companion: it watches your face on a live webcam feed and reads a
line you type. The fast, local signals -- expressed emotion and congruence
(face vs words) -- update continuously in real time. The heavier, deliberate
outputs -- a fuzzy age guess, a hedged LLM reading, a folklore "face reading",
and live recommendations -- run when you press Reflect.

Why the split: emotion + congruence are local CPU work and can stream a few
times a second. The LLM readings and recommendation APIs take seconds each and
would be nonsensical (and, on a metered LLM, ruinous) to run per video frame.
So the live overlay is continuous; the deep read is on demand.

Launch speed: this module imports only gradio + stdlib + our light modules. The
heavy libraries (tensorflow, mediapipe, transformers) are imported lazily inside
core/*.py, so the window appears instantly.
"""

from __future__ import annotations

import sys
import threading

import gradio as gr

from state import MirrorState, FaceState, TextState, CongruenceState, Readings, Recommendations
from core import perception, features, text_emotion, congruence, generation, recommend
from feedback import store

CONSENT = (
    "**Privacy:** MoodMirror watches your face for signals only. **No image or "
    "video is stored** -- only the derived reading. The traditional face reading "
    "is **folklore, for fun, not science.**"
)

_EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
          "surprise": "😮", "disgust": "😖", "neutral": "😐"}

# Set while Reflect is running. The webcam stream checks this and skips its heavy
# DeepFace work so the CPU is free for the (much slower) local LLM calls --
# otherwise the continuous stream starves Reflect and it never finishes.
_reflecting = threading.Event()


# --------------------------------------------------------------------------- #
# LIVE path -- fast, local, runs continuously on the webcam stream.
# No age (skip the 539MB model per frame), no LLM, no recs, no logging.
# --------------------------------------------------------------------------- #
def live_update(image, text: str):
    # While Reflect runs, don't burn CPU on per-frame DeepFace -- leave the last
    # live panels as-is and just keep the latest frame current.
    if _reflecting.is_set():
        return gr.update(), gr.update(), image
    face = perception.analyze_face(image, with_age=False)
    txt = text_emotion.analyze_text(text)
    cong = congruence.assess(face, txt)
    # Third return value is stashed in a gr.State so Reflect has a real frame.
    return _live_emotion_md(face, txt), _congruence_md(cong), image


# --------------------------------------------------------------------------- #
# REFLECT path -- the full six-layer flow, on a deliberate button press.
# Adds age, both LLM readings, recommendations, and the (only) SQLite log write.
# --------------------------------------------------------------------------- #
def reflect(image, text: str):
    _reflecting.set()   # tell the stream to back off the CPU
    try:
        state = MirrorState()
        state.face = perception.analyze_face(image, with_age=True)   # full: age too
        state.features = features.extract_features(image)
        state.text = text_emotion.analyze_text(text)
        state.congruence = congruence.assess(state.face, state.text)
        state.readings = generation.generate(state)
        state.recommendations = recommend.recommend(state.readings.recommendation_spec)

        # Log derived state only (no image), and ONLY on Reflect -- logging the
        # stream would flood the Phase 5 preference table with duplicate noise.
        try:
            store.log_interaction(state.to_dict())
        except Exception as e:  # noqa: BLE001
            print(f"[MoodMirror] feedback log failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

        return (_age_md(state.face), _reading_md(state.readings),
                _traditional_md(state), _recs_md(state.recommendations))
    finally:
        _reflecting.clear()


# --------------------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------------------- #
def _live_emotion_md(face: FaceState, txt: TextState) -> str:
    if not face.available:
        return f"_Looking for a face…_ {face.note}".strip()
    emoji = _EMOJI.get(face.emotion or "", "")
    out = f"### {emoji} {face.emotion}  \n{face.emotion_confidence:.0%} confidence"
    if txt.available and txt.emotion:
        out += f"\n\n_Your words read:_ **{txt.emotion}** ({txt.emotion_confidence:.0%})"
    return out


def _congruence_md(cong: CongruenceState) -> str:
    if cong.available and cong.verdict != "insufficient":
        return f"**{cong.verdict.title()}** — {cong.explanation}"
    return f"_{cong.explanation or cong.note}_"


def _age_md(face: FaceState) -> str:
    if not face.available:
        return f"_No face read._ {face.note}"
    if face.age_estimate is not None:
        if face.has_glasses:
            tail = "blame the camera and the glasses — they always add a few years"
        else:
            tail = "blame the camera and the lighting — these models always guess a little old"
        return (f"**Age guess:** around **{face.age_estimate}**  \n"
                f"_…give or take. If that feels off, {tail}._")
    return f"_{face.note}_" if face.note else ""


def _reading_md(readings: Readings) -> str:
    return readings.emotional_reading or f"_{readings.note}_"


def _traditional_md(s: MirrorState) -> str:
    if s.readings.traditional_reading:
        return s.readings.traditional_reading
    if not s.features.available and s.features.note:
        return f"_{s.features.note}_"
    return "_No distinctive features to read right now._"


def _recs_md(r: Recommendations) -> str:
    if not r.available:
        return f"_{r.note}_"

    def _link(t, url):
        return f"[{t}]({url})" if url else t

    parts = []
    if r.music:
        parts.append("**🎵 Music:** " + " · ".join(
            _link(f"{m['title']} — {m.get('artist','')}", m.get("url", "")) for m in r.music))
    if r.movies:
        parts.append("**🎬 Movies:** " + " · ".join(
            _link(f"{m['title']} ({m.get('year','')})", m.get("url", "")) for m in r.movies))
    if r.papers:
        parts.append("**📄 Papers:** " + " · ".join(
            _link(p["title"], p.get("url", "")) for p in r.papers))
    out = "\n\n".join(parts) if parts else "_No recommendations right now._"
    if r.note and not r.note.startswith("live"):
        out += f"\n\n_{r.note}_"
    return out


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MoodMirror") as demo:
        gr.Markdown("# MoodMirror")
        gr.Markdown(CONSENT)

        latest_frame = gr.State(None)   # most recent webcam/upload frame

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Look & tell")
                # Streaming requires a webcam-only source; upload is a separate
                # component (Gradio forbids streaming with multiple sources).
                cam = gr.Image(sources=["webcam"], type="numpy",
                               streaming=True, label="Live webcam")
                gr.Markdown(
                    "▶ **Click the round record button on the webcam to start the "
                    "live feed.** The Live panel updates while it's recording.")
                upload = gr.Image(sources=["upload"], type="numpy",
                                  label="…or upload a photo")
                text_in = gr.Textbox(label="Say a line about how you feel",
                                     placeholder="Type anything…", lines=2)
                go = gr.Button("Reflect on this moment", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### Live")
                live_emotion = gr.Markdown("_Looking for a face…_")
                live_congruence = gr.Markdown()
                gr.Markdown("---\n#### On reflection")
                out_age = gr.Markdown()
                out_reading = gr.Markdown()
                out_traditional = gr.Markdown()
                out_recs = gr.Markdown()

        # Live overlay: stream webcam frames through the fast path (throttled,
        # non-overlapping). Uploading a photo runs the same fast path once.
        cam.stream(live_update, inputs=[cam, text_in],
                   outputs=[live_emotion, live_congruence, latest_frame],
                   stream_every=0.4, concurrency_limit=1, show_progress="hidden")
        upload.upload(live_update, inputs=[upload, text_in],
                      outputs=[live_emotion, live_congruence, latest_frame],
                      show_progress="hidden")

        # Deliberate deep read on the latest frame. Show an immediate "thinking"
        # message first (the local LLM takes ~15-25s) so it never looks idle.
        def _thinking():
            msg = "_Reflecting… the local model takes ~15-25s._"
            return msg, "", "", ""

        go.click(_thinking, inputs=None,
                 outputs=[out_age, out_reading, out_traditional, out_recs]).then(
            reflect, inputs=[latest_frame, text_in],
            outputs=[out_age, out_reading, out_traditional, out_recs])

    return demo


def warmup() -> None:
    """
    Load every heavy model ONCE in the main thread, before Gradio starts.

    Two problems this solves at once:
      * On Windows, TensorFlow's DLL init crashes (0x45A) if two worker threads
        (a stream frame + a Reflect click) trigger first-time init concurrently.
        Initializing here, in the main thread, means workers only ever *use*
        already-loaded models.
      * It removes the ~20-30s first-interaction lag -- the models are warm by
        the time the page opens.
    """
    import numpy as np
    print("[MoodMirror] warming models (first run downloads/loads may take a bit)…",
          file=sys.stderr)
    dummy = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    for name, fn in (
        ("perception", lambda: perception.analyze_face(dummy, with_age=True)),
        ("text", lambda: text_emotion.analyze_text("warming up")),
        ("features", lambda: features.extract_features(dummy)),
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[MoodMirror] warmup {name}: {type(e).__name__}: {e}", file=sys.stderr)
    print("[MoodMirror] models ready.", file=sys.stderr)


if __name__ == "__main__":
    warmup()
    build_ui().launch()
