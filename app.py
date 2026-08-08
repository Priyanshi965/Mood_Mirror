"""
MoodMirror -- Gradio app (live video, cosmic themed).

Left column: the live webcam / upload / text + a compact always-on live status
(emotion + congruence, streamed a few times a second). Right column: your
"reflection" as tabs (Emotion, Face Reading, Congruence, For You, Most Like You),
filled on demand when you press Reflect.

Split rationale: emotion + congruence are fast local work and stream live; the
LLM readings + recommendations take seconds each and run only on Reflect.

Launch speed: only gradio + stdlib + our light modules are imported here; the
heavy libraries load lazily in core/*.py and are warmed once at startup.
"""

from __future__ import annotations

import sys
import threading

import gradio as gr

from state import MirrorState, FaceState, TextState, CongruenceState, Readings, Recommendations
from core import perception, features, text_emotion, congruence, generation, recommend
from feedback import store

CONSENT = (
    "MoodMirror watches your face for signals only — **no image or video is "
    "stored**, just the derived reading. The traditional face reading is "
    "**folklore, for fun, not science.**"
)

_EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
          "surprise": "😮", "disgust": "😖", "neutral": "😐"}

# Set while Reflect runs so the webcam stream backs off the CPU (see live_update).
_reflecting = threading.Event()

_PLACEHOLDER = "_Press **Reflect on this moment** to read this._"
_PLACEHOLDER_HTML = ("<p style='color:#9aa0b8;font-size:1.08rem'>Press "
                     "<b style='color:#c9bbff'>Reflect on this moment</b> to read "
                     "your emotion.</p>")


# --------------------------------------------------------------------------- #
# LIVE path -- fast, local, continuous. No age / LLM / recs / logging.
# --------------------------------------------------------------------------- #
def live_update(image, text: str):
    if _reflecting.is_set():
        return gr.update(), image
    face = perception.analyze_face(image, with_age=False)
    txt = text_emotion.analyze_text(text)
    cong = congruence.assess(face, txt)
    return _live_card_md(face, txt, cong), image


# --------------------------------------------------------------------------- #
# REFLECT path -- full pipeline on the latest frame (age + 3 LLM calls + recs).
# --------------------------------------------------------------------------- #
def reflect(image, text: str):
    _reflecting.set()   # pause the stream's CV so the local LLM isn't starved
    try:
        state = MirrorState()
        state.face = perception.analyze_face(image, with_age=True)
        state.features = features.extract_features(image)
        state.text = text_emotion.analyze_text(text)
        state.congruence = congruence.assess(state.face, state.text)
        state.readings = generation.generate(state)   # 3 LLM calls
        state.recommendations = recommend.recommend(state.readings.recommendation_spec)

        try:
            store.log_interaction(state.to_dict())   # log only on Reflect
        except Exception as e:  # noqa: BLE001
            print(f"[MoodMirror] feedback log failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

        return (_emotion_tab_md(state.face, state.text),
                _congruence_tab_md(state.congruence),
                _traditional_md(state),
                _recs_md(state.recommendations),
                _character_md(state.readings))
    finally:
        _reflecting.clear()


# --------------------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------------------- #
def _live_card_md(face: FaceState, txt: TextState, cong: CongruenceState) -> str:
    if not face.available:
        return f"### 🔍 Looking for a face…\n_{face.note}_"
    emoji = _EMOJI.get(face.emotion or "", "🙂")
    lines = [f"### {emoji}  {face.emotion} · {face.emotion_confidence:.0%}"]
    if txt.available and txt.emotion:
        lines.append(f"Your words read **{txt.emotion}** ({txt.emotion_confidence:.0%})")
    if cong.available and cong.verdict != "insufficient":
        lines.append(f"**{cong.verdict.title()}** — {cong.explanation}")
    return "  \n".join(lines)


def _emotion_tab_md(face: FaceState, txt: TextState) -> str:
    if not face.available:
        return f"_No face read._ {face.note}"
    emoji = _EMOJI.get(face.emotion or "", "🙂")
    pct = round(face.emotion_confidence * 100)

    ring = (f"<div class='mm-emotion-wrap'>"
            f"<div class='mm-ring' style='--pct:{pct}'>"
            f"<div class='mm-ring-in'><div class='mm-ring-num'>{pct}%</div>"
            f"<div class='mm-ring-lbl'>confidence</div></div></div>"
            f"<div class='mm-emotion-name'>{emoji} {face.emotion}</div>"
            f"<div class='mm-emotion-sub'>primary expressed emotion</div></div>")

    scores = sorted((face.emotion_scores or {}).items(), key=lambda x: -x[1])[:4]
    bars = "".join(
        f"<div class='mm-bar'><span class='mm-bar-l'>{k}</span>"
        f"<div class='mm-track'><i style='width:{round(v*100)}%'></i></div>"
        f"<span class='mm-bar-p'>{round(v*100)}%</span></div>"
        for k, v in scores)
    body = ring + f"<div class='mm-bars'>{bars}</div>"

    extra = []
    if txt.available and txt.emotion:
        extra.append(f"<b>In your words:</b> {txt.emotion} "
                     f"({txt.emotion_confidence:.0%})")
    if face.age_estimate is not None:
        tail = ("blame the camera and the glasses" if face.has_glasses
                else "blame the camera and the lighting")
        extra.append(f"<b>Age guess:</b> around <b>{face.age_estimate}</b> — give "
                     f"or take; if it feels off, {tail}.")
    if extra:
        body += "<div class='mm-extra'>" + "".join(f"<p>{e}</p>" for e in extra) + "</div>"
    return body


_CONG_DETAIL = {
    "congruent": ("Your outward expression and your words are singing the same "
                  "note. There's an ease here — what you feel and what you show "
                  "aren't fighting each other, which usually reads as being "
                  "settled and unguarded in the moment."),
    "incongruent": ("Here's the interesting part: your face and your words are "
                    "telling slightly different stories. That gap is what people "
                    "call masking — the expression on the surface doesn't fully "
                    "match the feeling underneath. It's completely human, and "
                    "often the most telling signal of all."),
    "flat": ("Both your face and your words are running quiet right now — little "
             "emotional charge in either channel. That evenness can mean calm, "
             "focus, or simply a guarded, low-key moment."),
}


def _congruence_tab_md(cong: CongruenceState) -> str:
    if cong.available and cong.verdict != "insufficient":
        detail = _CONG_DETAIL.get(cong.verdict, "")
        return (f"## {cong.verdict.title()}\n\n{cong.explanation}\n\n{detail}\n\n"
                f"_Congruence maps whether what your face shows, what your words "
                f"say, and what tradition suggests all point the same way — or "
                f"pull apart into something more interesting._")
    return f"_{cong.explanation or cong.note}_"


def _traditional_md(s: MirrorState) -> str:
    if s.readings.traditional_reading:
        return s.readings.traditional_reading
    if not s.features.available and s.features.note:
        return f"_{s.features.note}_"
    return "_No distinctive features to read right now._"


def _character_md(readings: Readings) -> str:
    if readings.character_match:
        return "### ✨ Right now, you're most like…\n\n" + readings.character_match
    return "_Needs the local model to conjure your matches — press Reflect._"


def _recs_md(r: Recommendations) -> str:
    if not r.available:
        return f"_{r.note}_"

    def _link(t, url):
        return f"[{t}]({url})" if url else t

    parts = []
    if r.music:
        parts.append("**🎵 Music for your mood**\n\n" + "  \n".join(
            "• " + _link(f"{m['title']} — {m.get('artist','')}", m.get("url", ""))
            for m in r.music))
    if r.movies:
        parts.append("**🎬 Films to reflect with**\n\n" + "  \n".join(
            "• " + _link(f"{m['title']} ({m.get('year','')})", m.get("url", ""))
            for m in r.movies))
    if r.papers:
        parts.append("**📄 Feed your curiosity**\n\n" + "  \n".join(
            "• " + _link(p["title"], p.get("url", "")) for p in r.papers))
    out = "\n\n".join(parts) if parts else "_No recommendations right now._"
    if r.note and not r.note.startswith("live"):
        out += f"\n\n_{r.note}_"
    return out


# --------------------------------------------------------------------------- #
# Theme + CSS (cosmic dark, purple/teal, rounded glass cards, pill tabs)
# --------------------------------------------------------------------------- #
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
)

_CSS = """
.gradio-container {
  background:
    radial-gradient(1.6px 1.6px at 8% 12%, rgba(255,255,255,0.85), transparent),
    radial-gradient(1.4px 1.4px at 22% 38%, rgba(200,220,255,0.7), transparent),
    radial-gradient(1.2px 1.2px at 35% 8%, rgba(255,255,255,0.6), transparent),
    radial-gradient(1.5px 1.5px at 48% 62%, rgba(210,200,255,0.6), transparent),
    radial-gradient(1.3px 1.3px at 63% 22%, rgba(255,255,255,0.7), transparent),
    radial-gradient(1.6px 1.6px at 74% 55%, rgba(180,230,255,0.6), transparent),
    radial-gradient(1.2px 1.2px at 86% 30%, rgba(255,255,255,0.7), transparent),
    radial-gradient(1.4px 1.4px at 92% 74%, rgba(210,200,255,0.6), transparent),
    radial-gradient(1.3px 1.3px at 15% 78%, rgba(255,255,255,0.55), transparent),
    radial-gradient(1.5px 1.5px at 55% 88%, rgba(200,220,255,0.6), transparent),
    radial-gradient(1.2px 1.2px at 30% 92%, rgba(255,255,255,0.5), transparent),
    radial-gradient(1.4px 1.4px at 79% 90%, rgba(210,210,255,0.55), transparent),
    radial-gradient(1200px 820px at 12% -12%, rgba(124,92,255,0.22), transparent 60%),
    radial-gradient(1000px 700px at 105% 8%, rgba(45,212,191,0.14), transparent 55%),
    radial-gradient(900px 900px at 50% 122%, rgba(139,92,246,0.18), transparent 60%),
    #07060f !important;
  background-attachment: fixed !important;
  color: #e7e7f2 !important;
}
.mm-title {
  font-weight: 800; font-size: 2.5rem; line-height: 1.1; margin-bottom: 2px;
  background: linear-gradient(92deg,#e9e3ff, #a78bfa 30%, #5eead4 65%, #bef264);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.mm-sub { color:#9aa0b8 !important; margin-top: 0; }
.mm-eyebrow { color:#a78bfa !important; letter-spacing:.18em; font-size:.72rem;
  text-transform:uppercase; font-weight:700; }
.mm-card {
  background: rgba(19,17,38,0.55) !important;
  border: 1px solid rgba(139,92,246,0.18) !important;
  border-radius: 20px !important;
  box-shadow: 0 8px 40px rgba(80,40,180,0.10) !important;
  backdrop-filter: blur(8px);
  padding: 14px 26px !important;
}
/* Readable, editorial typography for the written readings */
.mm-card p, .mm-card li {
  font-size: 1.14rem !important; line-height: 1.85 !important;
  color: #e2e2f2 !important; margin: 0.55rem 0 !important;
}
.mm-card h2 { font-size: 1.7rem !important; margin: .2rem 0 .4rem !important; }
.mm-card h3 { font-size: 1.28rem !important; margin: 1rem 0 .3rem !important;
  color: #ded7ff !important; }
.mm-card strong { color: #c9bbff !important; font-weight: 700; }
.mm-card em { color: #9aa0b8 !important; }
.mm-card a { color: #8fe3d3 !important; text-decoration: none; }
.mm-card a:hover { text-decoration: underline; }
.mm-live { min-height: 96px; }
.mm-live h3 { color:#fff !important; }
/* pill tabs, scoped to our container */
#mm-tabs .tab-nav { border: none !important; gap: 8px; }
#mm-tabs .tab-nav button {
  border-radius: 999px !important; border: 1px solid rgba(139,92,246,0.20) !important;
  background: rgba(255,255,255,0.03) !important; color:#c7c9db !important;
  padding: 6px 16px !important; font-weight: 600;
}
#mm-tabs .tab-nav button.selected {
  background: linear-gradient(92deg, rgba(139,92,246,0.35), rgba(45,212,191,0.20)) !important;
  color:#fff !important; border-color: rgba(167,139,250,0.55) !important;
  box-shadow: 0 0 18px rgba(139,92,246,0.35);
}
/* primary button glow */
button.primary, .mm-reflect button {
  background: linear-gradient(92deg,#8b5cf6,#7c3aed) !important; border:none !important;
  border-radius: 999px !important; box-shadow: 0 6px 24px rgba(124,58,237,0.45) !important;
  font-weight: 700 !important;
}
/* Emotion tab: confidence ring */
.mm-emotion-wrap { text-align:center; padding: 6px 0 2px; }
.mm-ring {
  width:176px; height:176px; border-radius:50%; margin: 6px auto 12px;
  background: conic-gradient(#a78bfa 0deg,
    #8b5cf6 calc(var(--pct)*1.8deg), #5eead4 calc(var(--pct)*3.6deg),
    rgba(255,255,255,0.06) calc(var(--pct)*3.6deg));
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 0 34px rgba(139,92,246,0.40);
}
.mm-ring-in {
  width:140px; height:140px; border-radius:50%; background:#0d0b1a;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.mm-ring-num { font-size:2.2rem; font-weight:800; color:#fff; line-height:1; }
.mm-ring-lbl { font-size:.68rem; letter-spacing:.16em; text-transform:uppercase;
  color:#9aa0b8; margin-top:4px; }
.mm-emotion-name { font-size:2rem; font-weight:800; text-transform:capitalize;
  background:linear-gradient(92deg,#c4b5fd,#5eead4); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.mm-emotion-sub { font-size:.7rem; letter-spacing:.16em; text-transform:uppercase;
  color:#9aa0b8; }
/* Emotion tab: score bars */
.mm-bars { margin-top:18px; }
.mm-bar { display:flex; align-items:center; gap:12px; margin:10px 0; }
.mm-bar-l { width:96px; text-transform:capitalize; color:#dcdcec; font-size:1rem; }
.mm-bar-p { width:46px; text-align:right; color:#a78bfa; font-weight:700; }
.mm-track { flex:1; height:9px; border-radius:99px;
  background:rgba(255,255,255,0.07); overflow:hidden; }
.mm-track i { display:block; height:100%; border-radius:99px;
  background:linear-gradient(90deg,#8b5cf6,#5eead4); }
.mm-extra { margin-top:16px; border-top:1px solid rgba(139,92,246,0.15); padding-top:10px; }
.mm-extra p { font-size:1.08rem !important; margin:.4rem 0 !important; }
.mm-extra b { color:#c9bbff; }
footer { display:none !important; }
"""


def build_ui() -> gr.Blocks:
    # NOTE: theme/css are passed to .launch() (Gradio 6 moved them there); passing
    # them to Blocks() is ignored, which silently drops all styling.
    with gr.Blocks(title="MoodMirror") as demo:
        latest_frame = gr.State(None)

        gr.HTML("<span class='mm-eyebrow'>● live session</span>")
        gr.HTML("<span class='mm-title'>MoodMirror</span>")
        gr.HTML(f"<span class='mm-sub'>{CONSENT}</span>")

        with gr.Row(equal_height=False):
            # ---------- LEFT: capture + live status ----------
            with gr.Column(scale=5):
                cam = gr.Image(sources=["webcam"], type="numpy", streaming=True,
                               label="Live webcam", height=300)
                gr.HTML("<span class='mm-sub'>▶ Click the round record button "
                        "to start the live feed.</span>")
                with gr.Group(elem_classes=["mm-card", "mm-live"]):
                    live_card = gr.Markdown("### 🔍 Looking for a face…")
                upload = gr.Image(sources=["upload"], type="numpy",
                                  label="…or upload a photo", height=170)
                text_in = gr.Textbox(label="Say a line about how you feel",
                                     placeholder="Type anything…", lines=2)
                go = gr.Button("Reflect on this moment", variant="primary",
                               elem_classes=["mm-reflect"])

            # ---------- RIGHT: reflection tabs ----------
            with gr.Column(scale=6):
                gr.HTML("<span class='mm-title' style='font-size:1.7rem'>"
                        "Your Reflection</span>")
                with gr.Tabs(elem_id="mm-tabs"):
                    with gr.Tab("😊 Emotion"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_emotion = gr.HTML(_PLACEHOLDER_HTML)
                    with gr.Tab("🔮 Face Reading"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_traditional = gr.Markdown(_PLACEHOLDER)
                    with gr.Tab("⚖️ Congruence"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_congruence = gr.Markdown(_PLACEHOLDER)
                    with gr.Tab("✨ For You"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_recs = gr.Markdown(_PLACEHOLDER)
                    with gr.Tab("🎭 Most Like You"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_character = gr.Markdown(_PLACEHOLDER)

        # Live overlay: throttled, non-overlapping (keep these three verbatim).
        cam.stream(live_update, inputs=[cam, text_in],
                   outputs=[live_card, latest_frame],
                   stream_every=0.4, concurrency_limit=1, show_progress="hidden")
        upload.upload(live_update, inputs=[upload, text_in],
                      outputs=[live_card, latest_frame], show_progress="hidden")

        # Reflect: immediate "reading…" on all five tabs, then the full pipeline.
        reading_tabs = [t_emotion, t_congruence, t_traditional, t_recs, t_character]

        def _thinking():
            # Plain text so it renders in both the HTML (Emotion) and Markdown tabs.
            msg = "⏳ Reading this moment… the local model writes it by hand (~15-40s)."
            return (msg,) * 5

        go.click(_thinking, inputs=None, outputs=reading_tabs).then(
            reflect, inputs=[latest_frame, text_in], outputs=reading_tabs)

    return demo


def warmup() -> None:
    """Load every heavy model once in the MAIN thread before Gradio starts --
    avoids a Windows TF DLL init crash under worker threads and the first-click
    lag. Workers then only *use* already-loaded models."""
    import numpy as np
    print("[MoodMirror] warming models (first run may take a bit)…", file=sys.stderr)
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
    build_ui().launch(theme=_THEME, css=_CSS)
