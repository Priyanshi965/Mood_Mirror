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
_PLACEHOLDER_HTML = (
    "<div class='mm-empty'><span></span><p>Awaiting reflection capture</p></div>"
)


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
        return f"### Scanning for a face\n_{face.note}_"
    emoji = _EMOJI.get(face.emotion or "", "🙂")
    lines = [f"### {emoji} {face.emotion} · {face.emotion_confidence:.0%}"]
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
        return "### Right now, you're most like...\n\n" + readings.character_match
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
# Theme + CSS (futuristic dark console with luminous cyan/violet accents)
# --------------------------------------------------------------------------- #
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
)

_CSS = """
:root {
  --mm-bg: #050713;
  --mm-panel: rgba(9, 14, 31, 0.74);
  --mm-panel-strong: rgba(14, 21, 44, 0.88);
  --mm-line: rgba(125, 249, 255, 0.18);
  --mm-line-hot: rgba(185, 147, 255, 0.36);
  --mm-text: #edf6ff;
  --mm-muted: #93a4bd;
  --mm-cyan: #67e8f9;
  --mm-violet: #b993ff;
  --mm-green: #8cffc1;
  --mm-red: #ff6b9f;
}
.gradio-container {
  background:
    linear-gradient(rgba(103,232,249,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(103,232,249,0.035) 1px, transparent 1px),
    radial-gradient(920px 640px at 8% -10%, rgba(103,232,249,0.18), transparent 62%),
    radial-gradient(820px 620px at 94% 0%, rgba(185,147,255,0.18), transparent 58%),
    radial-gradient(860px 760px at 58% 115%, rgba(140,255,193,0.10), transparent 55%),
    linear-gradient(140deg, #03040b 0%, var(--mm-bg) 45%, #080816 100%) !important;
  background-size: 42px 42px, 42px 42px, auto, auto, auto, auto !important;
  background-attachment: fixed !important;
  color: var(--mm-text) !important;
}
.gradio-container::before {
  content:"";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(180deg, transparent 0%, rgba(103,232,249,0.05) 50%, transparent 100%);
  background-size: 100% 7px;
  opacity: .22;
  mix-blend-mode: screen;
  z-index: 0;
}
.gradio-container > .main, .contain {
  position: relative;
  z-index: 1;
}
.gradio-container .main {
  max-width: 1260px !important;
  margin: 0 auto !important;
}
.mm-shell {
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:24px;
  padding: 22px 2px 18px;
  border-bottom: 1px solid var(--mm-line);
  margin-bottom: 18px;
}
.mm-kicker {
  display:inline-flex;
  align-items:center;
  gap:8px;
  color: var(--mm-cyan);
  letter-spacing:.22em;
  font-size:.68rem;
  text-transform:uppercase;
  font-weight:800;
}
.mm-kicker::before {
  content:"";
  width:8px;
  height:8px;
  border-radius:50%;
  background: var(--mm-green);
  box-shadow:0 0 18px var(--mm-green);
}
.mm-title {
  display:block;
  font-weight: 900;
  font-size: clamp(2.3rem, 4.4vw, 5.2rem);
  line-height: .9;
  margin: 9px 0 8px;
  letter-spacing: 0;
  background: linear-gradient(92deg,#ffffff 0%, var(--mm-cyan) 38%, var(--mm-violet) 78%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  text-shadow: 0 0 34px rgba(103,232,249,.16);
}
.mm-sub {
  color: var(--mm-muted) !important;
  margin: 0;
  max-width: 760px;
  font-size: .98rem;
  line-height: 1.7;
}
.mm-status-grid {
  display:grid;
  grid-template-columns: repeat(3, minmax(76px, 1fr));
  gap:10px;
  min-width: 320px;
}
.mm-stat {
  background: linear-gradient(180deg, rgba(103,232,249,0.11), rgba(185,147,255,0.05));
  border: 1px solid var(--mm-line);
  border-radius: 8px;
  padding: 10px 12px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 0 22px rgba(103,232,249,.07);
}
.mm-stat b {
  display:block;
  color: var(--mm-text);
  font-size: 1.05rem;
  letter-spacing: 0;
}
.mm-stat span {
  display:block;
  margin-top: 3px;
  color: var(--mm-muted);
  font-size: .63rem;
  text-transform: uppercase;
  letter-spacing: .14em;
}
.mm-panel-title {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 16px;
  margin: 0 0 12px;
  color: var(--mm-text);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.mm-panel-title::after {
  content:"";
  flex:1;
  height: 1px;
  background: linear-gradient(90deg, var(--mm-line), transparent);
}
.mm-card {
  position: relative;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(15,23,49,0.82), rgba(7,11,25,0.78)) !important;
  border: 1px solid var(--mm-line) !important;
  border-radius: 8px !important;
  box-shadow: 0 18px 60px rgba(0,0,0,0.28), 0 0 34px rgba(103,232,249,0.08) !important;
  backdrop-filter: blur(8px);
  padding: 18px 22px !important;
}
.mm-card::before {
  content:"";
  position:absolute;
  left:0;
  right:0;
  top:0;
  height:1px;
  background: linear-gradient(90deg, transparent, var(--mm-cyan), var(--mm-violet), transparent);
  opacity:.75;
}
.mm-card::after {
  content:"";
  position:absolute;
  width:72px;
  height:72px;
  right:-36px;
  top:-36px;
  border:1px solid rgba(103,232,249,.22);
  transform: rotate(45deg);
}
/* Readable, editorial typography for the written readings */
.mm-card p, .mm-card li {
  font-size: 1.05rem !important; line-height: 1.78 !important;
  color: #dce8f8 !important; margin: 0.55rem 0 !important;
}
.mm-card h2 { font-size: 1.55rem !important; margin: .2rem 0 .4rem !important; }
.mm-card h3 { font-size: 1.28rem !important; margin: 1rem 0 .3rem !important;
  color: #e8fbff !important; }
.mm-card strong { color: var(--mm-cyan) !important; font-weight: 800; }
.mm-card em { color: var(--mm-muted) !important; }
.mm-card a { color: var(--mm-green) !important; text-decoration: none; }
.mm-card a:hover { text-decoration: underline; }
.mm-live { min-height: 118px; }
.mm-live h3 { color:#fff !important; }
.mm-empty {
  display:flex;
  align-items:center;
  gap:14px;
  color: var(--mm-muted);
  min-height: 116px;
}
.mm-empty span {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  border: 1px solid var(--mm-line);
  box-shadow: 0 0 22px rgba(103,232,249,.15), inset 0 0 20px rgba(103,232,249,.06);
}
.mm-empty p {
  margin: 0 !important;
  letter-spacing: .04em;
  text-transform: uppercase;
  font-size: .76rem !important;
}
.mm-input-panel {
  display:block;
}
/* pill tabs, scoped to our container */
#mm-tabs .tab-nav {
  border: none !important;
  gap: 8px;
  border-bottom: 1px solid var(--mm-line) !important;
  padding-bottom: 10px;
}
#mm-tabs .tab-nav button {
  border-radius: 7px !important; border: 1px solid rgba(103,232,249,0.14) !important;
  background: rgba(8,13,29,0.72) !important; color:#b8c8dd !important;
  padding: 8px 15px !important; font-weight: 800;
  letter-spacing: .02em;
}
#mm-tabs .tab-nav button.selected {
  background: linear-gradient(92deg, rgba(103,232,249,0.18), rgba(185,147,255,0.18)) !important;
  color:#fff !important; border-color: rgba(103,232,249,0.42) !important;
  box-shadow: 0 0 20px rgba(103,232,249,0.18);
}
/* primary button glow */
button.primary, .mm-reflect button {
  background: linear-gradient(92deg, var(--mm-cyan), var(--mm-violet)) !important;
  color: #06101d !important;
  border:none !important;
  border-radius: 7px !important;
  box-shadow: 0 0 26px rgba(103,232,249,0.34) !important;
  font-weight: 900 !important;
  letter-spacing: .02em;
}
button.primary:hover, .mm-reflect button:hover {
  filter: brightness(1.08);
  box-shadow: 0 0 34px rgba(185,147,255,0.36) !important;
}
input, textarea, .wrap, .block, .form, .panel {
  border-radius: 8px !important;
}
label, .label-wrap span {
  color: #c8d6e8 !important;
  font-weight: 700 !important;
}
textarea, input {
  background: rgba(5,9,22,0.78) !important;
  color: var(--mm-text) !important;
  border-color: rgba(103,232,249,0.18) !important;
}
.image-container, .image-frame, .upload-container {
  border-radius: 8px !important;
}
/* Emotion tab: confidence ring */
.mm-emotion-wrap { text-align:center; padding: 6px 0 2px; }
.mm-ring {
  width:176px; height:176px; border-radius:50%; margin: 6px auto 12px;
  background: conic-gradient(var(--mm-cyan) 0deg,
    var(--mm-violet) calc(var(--pct)*1.8deg), var(--mm-green) calc(var(--pct)*3.6deg),
    rgba(255,255,255,0.06) calc(var(--pct)*3.6deg));
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 0 38px rgba(103,232,249,0.28);
}
.mm-ring-in {
  width:140px; height:140px; border-radius:50%; background:#07101f;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  border: 1px solid rgba(103,232,249,0.14);
}
.mm-ring-num { font-size:2.2rem; font-weight:800; color:#fff; line-height:1; }
.mm-ring-lbl { font-size:.68rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--mm-muted); margin-top:4px; }
.mm-emotion-name { font-size:2rem; font-weight:800; text-transform:capitalize;
  background:linear-gradient(92deg,var(--mm-cyan),var(--mm-violet)); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.mm-emotion-sub { font-size:.7rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--mm-muted); }
/* Emotion tab: score bars */
.mm-bars { margin-top:18px; }
.mm-bar { display:flex; align-items:center; gap:12px; margin:10px 0; }
.mm-bar-l { width:96px; text-transform:capitalize; color:#dce8f8; font-size:1rem; }
.mm-bar-p { width:46px; text-align:right; color:var(--mm-cyan); font-weight:800; }
.mm-track { flex:1; height:9px; border-radius:99px;
  background:rgba(255,255,255,0.07); overflow:hidden; }
.mm-track i { display:block; height:100%; border-radius:99px;
  background:linear-gradient(90deg,var(--mm-cyan),var(--mm-violet)); }
.mm-extra { margin-top:16px; border-top:1px solid var(--mm-line); padding-top:10px; }
.mm-extra p { font-size:1.08rem !important; margin:.4rem 0 !important; }
.mm-extra b { color:var(--mm-cyan); }
@media (max-width: 900px) {
  .mm-shell {
    display:block;
  }
  .mm-status-grid {
    min-width: 0;
    margin-top: 16px;
  }
}
footer { display:none !important; }
"""


def build_ui() -> gr.Blocks:
    # NOTE: theme/css are passed to .launch() (Gradio 6 moved them there); passing
    # them to Blocks() is ignored, which silently drops all styling.
    with gr.Blocks(title="MoodMirror") as demo:
        latest_frame = gr.State(None)

        gr.HTML(
            "<div class='mm-shell'>"
            "<div><span class='mm-kicker'>Live session</span>"
            "<span class='mm-title'>MoodMirror</span>"
            f"<p class='mm-sub'>{CONSENT}</p></div>"
            "<div class='mm-status-grid'>"
            "<div class='mm-stat'><b>LIVE</b><span>Vision</span></div>"
            "<div class='mm-stat'><b>LOCAL</b><span>Signals</span></div>"
            "<div class='mm-stat'><b>ZERO</b><span>Media stored</span></div>"
            "</div></div>"
        )

        with gr.Row(equal_height=False):
            # ---------- LEFT: capture + live status ----------
            with gr.Column(scale=5):
                gr.HTML("<div class='mm-panel-title'>Capture Deck</div>")
                cam = gr.Image(sources=["webcam"], type="numpy", streaming=True,
                               label="Live webcam", height=300)
                gr.HTML("<p class='mm-sub'>Start the camera feed, or load a still frame.</p>")
                with gr.Group(elem_classes=["mm-card", "mm-live"]):
                    live_card = gr.Markdown("### Scanning for a face")
                upload = gr.Image(sources=["upload"], type="numpy",
                                  label="…or upload a photo", height=170)
                text_in = gr.Textbox(label="Say a line about how you feel",
                                     placeholder="Type anything…", lines=2)
                go = gr.Button("Reflect on this moment", variant="primary",
                               elem_classes=["mm-reflect"])

            # ---------- RIGHT: reflection tabs ----------
            with gr.Column(scale=6):
                gr.HTML("<div class='mm-panel-title'>Reflection Console</div>")
                with gr.Tabs(elem_id="mm-tabs"):
                    with gr.Tab("Emotion"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_emotion = gr.HTML(_PLACEHOLDER_HTML)
                    with gr.Tab("Face Reading"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_traditional = gr.Markdown(_PLACEHOLDER)
                    with gr.Tab("Congruence"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_congruence = gr.Markdown(_PLACEHOLDER)
                    with gr.Tab("For You"):
                        with gr.Group(elem_classes=["mm-card"]):
                            t_recs = gr.Markdown(_PLACEHOLDER)
                    with gr.Tab("Most Like You"):
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
