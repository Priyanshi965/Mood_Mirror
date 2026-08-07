"""
Central configuration. All secrets come from the environment (a local .env,
loaded here), never from source. See .env.example for the full list.

Nothing in here is required for the app to *launch* -- missing keys just mean
the layers that need them report themselves unavailable. This keeps the Phase 0
skeleton runnable before you've signed up for a single API.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from the project root if present
except ImportError:
    # python-dotenv is a base dep, but never let its absence break import.
    pass


# --- Paths ----------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "face_reading_corpus.json"
FEEDBACK_DB = ROOT / "feedback" / "moodmirror.db"   # gitignored

# MediaPipe Face Mesh model (~3.6MB). Lives outside the repo; download once via
# scripts/download_weights.py. core/features.py gates on this file existing.
FACE_LANDMARKER_PATH = os.getenv(
    "FACE_LANDMARKER_PATH",
    str(Path(os.path.expanduser("~")) / ".mediapipe" / "face_landmarker.task"),
)


# --- LLM (OpenRouter is the chosen provider; others kept as easy fallbacks) - #
# All the OpenAI-compatible providers here share one call path -- see
# active_llm() below. Swapping providers is just an .env change.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # openrouter | groq | gemini | ollama

# OpenRouter -- free daily allowance, ~28 free models. Get a key at
# https://openrouter.ai/keys  (no card needed for the :free models).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# NOTE: OpenRouter rotates which models are free and rate-limits popular ones.
# Verified working Aug 2026. If it starts 404-ing ("unavailable for free") or
# 429-ing, swap to another free model -- `openrouter/free` is a resilient
# auto-router fallback. List current free models:
#   curl https://openrouter.ai/api/v1/models | grep -o '"id":"[^"]*:free"'
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# Groq (fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Gemini / Ollama (fallbacks; Gemini needs its own SDK path, not wired yet)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


# --- Recommendation APIs (plan §5, Layer 6) -------------------------------- #
# Music: Last.fm (free key, no OAuth). Mood -> tags -> tag.getTopTracks /
# track.getSimilar. Get a key at https://www.last.fm/api/account/create
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
# Movies: TMDB (free key, non-commercial). /discover/movie by genre.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
# arXiv needs no key.


# --- Behavior flags -------------------------------------------------------- #
# Quiet TensorFlow's oneDNN chatter unless the user wants it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def active_llm() -> dict:
    """
    Resolve the currently-selected provider to one uniform shape that any
    OpenAI-compatible caller (scripts/smoke_llm.py now, core/generation.py in
    Phase 2) can use without knowing which provider it is:

        {base_url, model, api_key, headers, ready}

    `ready` is False when the provider needs a key it doesn't have, so callers
    degrade instead of firing a doomed request.
    """
    if LLM_PROVIDER == "openrouter":
        return {
            "base_url": OPENROUTER_BASE_URL,
            "model": OPENROUTER_MODEL,
            "api_key": OPENROUTER_API_KEY,
            # Optional OpenRouter attribution headers -- harmless if the app
            # isn't public. They help with OpenRouter's model rankings.
            "headers": {
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "MoodMirror",
            },
            "ready": bool(OPENROUTER_API_KEY),
        }
    if LLM_PROVIDER == "groq":
        return {
            "base_url": GROQ_BASE_URL,
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "headers": {},
            "ready": bool(GROQ_API_KEY),
        }
    if LLM_PROVIDER == "ollama":
        return {
            "base_url": f"{OLLAMA_HOST}/v1",
            "model": OLLAMA_MODEL,
            "api_key": "ollama",  # local server ignores it
            "headers": {},
            "ready": True,  # assumes a running local server; checked at call time
        }
    # gemini or unknown -- not on the OpenAI-compatible path here.
    return {"base_url": "", "model": "", "api_key": GEMINI_API_KEY,
            "headers": {}, "ready": False}


def llm_ready() -> bool:
    return bool(active_llm().get("ready"))


def lastfm_ready() -> bool:
    return bool(LASTFM_API_KEY)


def tmdb_ready() -> bool:
    return bool(TMDB_API_KEY)
