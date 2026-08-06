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


# --- LLM (pick one; Groq is the recommended week-1 choice, plan §7) -------- #
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")     # groq | gemini | ollama
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


# --- Recommendation APIs (plan §5, Layer 6) -------------------------------- #
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
# arXiv needs no key.


# --- Behavior flags -------------------------------------------------------- #
# Quiet TensorFlow's oneDNN chatter unless the user wants it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def llm_ready() -> bool:
    if LLM_PROVIDER == "groq":
        return bool(GROQ_API_KEY)
    if LLM_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    if LLM_PROVIDER == "ollama":
        return True  # assumes a local server; checked at call time
    return False


def spotify_ready() -> bool:
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)


def tmdb_ready() -> bool:
    return bool(TMDB_API_KEY)
