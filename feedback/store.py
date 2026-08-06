"""
Feedback + preference log (plan §6).

Every interaction logs its derived state, the readings shown, and (later) a
thumbs up/down. Phase 5 uses the liked rows for feedback-guided few-shot:
retrieve past liked readings for similar states and inject them as prompt
examples. This is preference-conditioned generation / RAG -- honestly NOT
reinforcement learning, and the plan is careful never to call it that.

PHASE 0 STATUS: minimal-but-real. We create the SQLite schema and expose
`log_interaction`. Thumbs feedback and similarity retrieval come in Phase 5.

PRIVACY: we store DERIVED STATE ONLY -- emotion labels, text, readings. Never an
image, never a file path to one. The state contract (state.py) makes that easy
because there's nowhere in it to put an image in the first place. (plan §10)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import config


def _connect() -> sqlite3.Connection:
    config.FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.FEEDBACK_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT DEFAULT CURRENT_TIMESTAMP,
            state_json  TEXT NOT NULL,
            rating      INTEGER,          -- +1 / -1 / NULL, set later
            correction  TEXT              -- optional user correction
        )
        """
    )
    return conn


def log_interaction(state_dict: dict[str, Any]) -> int:
    """Persist one interaction's derived state. Returns the row id."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO interactions (state_json) VALUES (?)",
            (json.dumps(state_dict, default=str),),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def set_rating(row_id: int, rating: int, correction: str | None = None) -> None:
    """Phase 5: record a thumbs up (+1) / down (-1) on a past interaction."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE interactions SET rating = ?, correction = ? WHERE id = ?",
            (rating, correction, row_id),
        )
        conn.commit()
    finally:
        conn.close()
