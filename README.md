# MoodMirror

A reflective companion. It looks at your face and reads a line you type, then gives you:

1. your current **expressed emotional state** (in the moment, never "character"),
2. a clearly-labeled **traditional face reading** — folklore, for fun, not science, and
3. **live recommendations** (music, movies, research) tuned to how you seem.

It also makes a fuzzy age guess, and — the idea that makes it more than a toy — it
notices when your **face, your words, and the tradition disagree**.

> **Not science.** The emotion read is *momentary expressed emotion*. The face-reading
> module reports what cultural traditions *claim* about features, cited as tradition —
> it never asserts those claims are true. See the project plan, §2 and §4.

---

## Status: Phase 0 (scaffold) ✅

The app runs end to end **right now**, but every model layer is a stub that returns
placeholder data. This proves the architecture, the UI, and the state contract before
any heavy logic goes in. Later phases replace stub bodies without changing signatures.

| Layer | File | Phase | State |
|------|------|-------|-------|
| 1 Perception (emotion / age / gender) | `core/perception.py` | 1 | **real** (DeepFace emotion + InsightFace age & gender) |
| 2 Facial features (MediaPipe) | `core/features.py` | 3 | **real** (Tasks FaceLandmarker; needs one-time model download) |
| 3 Text emotion (transformers) | `core/text_emotion.py` | 2 | **real** (GoEmotions; needs one-time model download) |
| 4 Congruence (face vs words) | `core/congruence.py` | 2 | **real** (full GoEmotions+DeepFace valence coverage) |
| 5 Reasoning & generation (LLM) | `core/generation.py` | 2/3 | **real** — emotional reading + folklore reading (RAG) via OpenRouter |
| 6 Recommendations (Last.fm/curated/arXiv) | `core/recommend.py` | 4 | **real** (emotion → varied music/films/papers) |
| Feedback log (SQLite) | `feedback/store.py` | 5 | **real (v1)** |

The single source of truth for what flows between layers is **`state.py`** — read it first.

## Run it

```bash
pip install -r requirements-base.txt      # gradio + dotenv + requests -- enough to launch
python app.py
```

The heavy CV/NLP stack (`requirements-heavy.txt`: deepface, mediapipe, tensorflow,
transformers, torch) is only needed once the model layers are real. Those libraries are
imported **lazily inside functions**, so the app launches instantly without them and each
layer degrades gracefully if one is missing.

**Enable age + text-emotion (one-time model download, ~1GB):**

```bash
python scripts/download_weights.py
```

The app runs without this — the age band and text-emotion layer just report themselves
"unavailable" until the models are on disk. The download is gated deliberately so a UI
click never triggers a multi-hundred-MB blocking download. An LLM key (`OPENROUTER_API_KEY`
in `.env`) enables the generated reading; without it, generation falls back to a
deterministic hedged template.

## Configure (optional)

```bash
cp .env.example .env        # then paste in any keys you have
python scripts/smoke_llm.py # week-1 de-risk: one Groq round-trip (no-op without a key)
```

No keys are required to launch. Missing keys just mark the matching layer "unavailable".

## Privacy (by design)

- Faces are processed for **signals only**. **No image is stored** — only the derived
  reading. The state contract (`state.py`) has nowhere to put an image, and `.gitignore`
  blocks image files as a backstop.
- The consent + "for fun, not science" notice is shown in the UI before anything runs.

## Layout

> **Import layout note:** `state.py` and `config.py` live at the project root and
> are imported as bare top-level modules (`from state import ...`). This is a
> deliberate Phase 0 simplification — run everything from the project root. If the
> project grows, promote these into a package (`moodmirror/`) to avoid name
> collisions on `sys.path`.

```
app.py                 Gradio UI + pipeline wiring (lazy imports only)
state.py               THE state contract -- start here
config.py              settings & API keys from .env
core/                  the six processing layers (stubs -> real, phase by phase)
data/                  face_reading_corpus.json (folklore, cited as tradition)
feedback/              SQLite preference log (Phase 5)
scripts/smoke_llm.py   one-shot LLM connectivity check
```
