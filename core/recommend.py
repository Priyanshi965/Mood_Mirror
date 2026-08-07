"""
Layer 6 -- Recommendations (Last.fm / TMDB / arXiv): live content.

PHASE 4: real. Turns the recommendation_spec from Layer 5 into live results:
  * Music  -> Last.fm  (mood -> tags -> tag.getTopTracks)     [LASTFM_API_KEY]
  * Movies -> TMDB     (mood -> genre -> /discover/movie)      [TMDB_API_KEY]
  * Papers -> arXiv    (mood -> query -> Atom feed)            [no key]

Rules (plan §10): each provider runs in its own try/except; a provider with no
key, a timeout, or an error returns an empty list plus a short note -- it never
raises and never sinks the other two. Results are cached in-process to be kind
to rate limits. Errors never echo the request URL/key.
"""

from __future__ import annotations

from typing import Any

import config
from state import Recommendations

_TIMEOUT = 12
_N = 4  # results per category

# Mood (from generation._valence_hint) -> query params for each provider.
_MOOD_MAP = {
    "low": {
        "tags": ["calm", "mellow", "chill"],
        "genre": (35, "Comedy"),            # a gentle lift
        "arxiv": "emotion and wellbeing",
    },
    "steady": {
        "tags": ["feel good", "indie", "happy"],
        "genre": (12, "Adventure"),
        "arxiv": "creativity and cognition",
    },
}

# Tiny in-process cache: {(provider, key) -> results list}
_cache: dict[tuple, list] = {}


def _get(url: str, params: dict) -> Any:
    import requests
    return requests.get(url, params=params, timeout=_TIMEOUT)


def _music(tags: list[str]) -> tuple[list[dict], str]:
    if not config.lastfm_ready():
        return [], "Last.fm key not set"
    tag = tags[0]
    ck = ("music", tag)
    if ck in _cache:
        return _cache[ck], ""
    try:
        r = _get("http://ws.audioscrobbler.com/2.0/", {
            "method": "tag.gettoptracks", "tag": tag,
            "api_key": config.LASTFM_API_KEY, "format": "json", "limit": _N})
        data = r.json()
        if "error" in data:
            return [], "Last.fm error"
        out = [{"title": t["name"], "artist": t["artist"]["name"],
                "url": t.get("url", "")}
               for t in data.get("tracks", {}).get("track", [])[:_N]]
        _cache[ck] = out
        return out, ""
    except Exception as e:  # noqa: BLE001 -- never leak URL/key
        return [], f"Last.fm unavailable ({type(e).__name__})"


def _movies(genre: tuple[int, str]) -> tuple[list[dict], str]:
    if not config.tmdb_ready():
        return [], "TMDB key not set"
    gid, _gname = genre
    ck = ("movies", gid)
    if ck in _cache:
        return _cache[ck], ""
    try:
        r = _get("https://api.themoviedb.org/3/discover/movie", {
            "api_key": config.TMDB_API_KEY, "with_genres": gid,
            "sort_by": "vote_average.desc", "vote_count.gte": 800,
            "language": "en-US", "page": 1})
        data = r.json()
        if data.get("status_code"):
            return [], "TMDB error"
        out = [{"title": m["title"], "year": (m.get("release_date") or "")[:4],
                "score": m.get("vote_average")}
               for m in data.get("results", [])[:_N]]
        _cache[ck] = out
        return out, ""
    except Exception as e:  # noqa: BLE001
        return [], f"TMDB unavailable ({type(e).__name__})"


def _papers(query: str) -> tuple[list[dict], str]:
    ck = ("papers", query)
    if ck in _cache:
        return _cache[ck], ""
    try:
        import xml.etree.ElementTree as ET
        r = _get("http://export.arxiv.org/api/query", {
            "search_query": f"all:{query}", "max_results": _N,
            "sortBy": "relevance"})
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        out = []
        for e in root.findall("a:entry", ns)[:_N]:
            title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
            link = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
            author = e.findtext("a:author/a:name", default="", namespaces=ns)
            out.append({"title": " ".join(title.split()),
                        "author": author, "url": link})
        _cache[ck] = out
        return out, ""
    except Exception as e:  # noqa: BLE001
        return [], f"arXiv unavailable ({type(e).__name__})"


def recommend(spec: dict[str, Any]) -> Recommendations:
    if not spec:
        return Recommendations(available=False, note="No recommendation spec.")

    mood = spec.get("mood", "steady")
    params = _MOOD_MAP.get(mood, _MOOD_MAP["steady"])

    music, m_note = _music(params["tags"])
    movies, v_note = _movies(params["genre"])
    papers, p_note = _papers(params["arxiv"])

    notes = [n for n in (m_note, v_note, p_note) if n]
    return Recommendations(
        available=True,
        note=("; ".join(notes) if notes else f"live ({mood})"),
        music=music,
        movies=movies,
        papers=papers,
    )
