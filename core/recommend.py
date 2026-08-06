"""
Layer 6 -- Recommendations (Last.fm / TMDB / arXiv): live content.

Takes the recommendation_spec produced by Layer 5 and turns it into live
results. PHASE 0 STATUS: stub. Phase 4 wires the real APIs:
  * Music  -> Last.fm API (map mood -> tags -> tag.getTopTracks / getSimilar)
  * Movies -> TMDB API (free key; /discover/movie by genre)
  * Papers -> arXiv API (no key), for the "curious/creative" state

Rules: each provider imported / called inside its own try block; a provider
that has no key or errors returns an empty list with a note, never raises.
Cache results and have fallbacks (plan §10, rate limits).
"""

from __future__ import annotations

from typing import Any

import config
from state import Recommendations


def recommend(spec: dict[str, Any]) -> Recommendations:
    if not spec:
        return Recommendations(available=False, note="No recommendation spec.")

    # ----- PHASE 0 STUB ---------------------------------------------------- #
    # Real Phase 4 body calls _music(spec), _movies(spec), _papers(spec),
    # each guarded so one dead API doesn't sink the panel.
    notes = []
    if not config.lastfm_ready():
        notes.append("Last.fm key not set")
    if not config.tmdb_ready():
        notes.append("TMDB key not set")

    return Recommendations(
        available=True,
        note="stub" + (f" ({'; '.join(notes)})" if notes else ""),
        music=[{"title": "(sample track)", "artist": "wired up in Phase 4"}],
        movies=[{"title": "(sample film)", "year": "Phase 4"}],
        papers=[{"title": "(sample paper)", "source": "arXiv, Phase 4"}],
    )
