"""
Layer 6 -- Recommendations (Last.fm / TMDB-free / arXiv): live content.

PHASE 4: real. Turns the emotion in the recommendation_spec into live picks:
  * Music  -> Last.fm  (emotion -> a vibe -> a tag -> tag.getTopTracks)  [LASTFM_API_KEY]
  * Movies -> curated mood pools linked to Letterboxd (TMDB is ISP-blocked)
  * Papers -> arXiv    (emotion -> a topic -> Atom feed)                 [no key]

Variety: recommendations are keyed on the SPECIFIC emotion (mapped to a "vibe"),
and we randomly sample from larger pools / result sets, so different people -- and
repeat runs -- get different picks rather than the same list every time.

Rules (plan §10): each provider runs in its own try/except; a provider with no
key, a timeout, or an error returns an empty list plus a short note -- it never
raises and never sinks the other two. Errors never echo the request URL/key.
"""

from __future__ import annotations

import random
from typing import Any

import config
from state import Recommendations

_TIMEOUT = 12
_N = 4  # picks shown per category

# GoEmotions (28) + DeepFace (7) labels -> a small set of "vibes".
_EMOTION_VIBE = {
    # up
    "happy": "up", "joy": "up", "amusement": "up", "excitement": "up",
    "optimism": "up", "pride": "up", "gratitude": "up", "approval": "up",
    # tender
    "love": "tender", "caring": "tender", "admiration": "tender",
    "desire": "tender", "relief": "tender",
    # down
    "sad": "down", "sadness": "down", "grief": "down", "disappointment": "down",
    "remorse": "down", "nervousness": "down", "fear": "down",
    "embarrassment": "down",
    # angry
    "angry": "angry", "anger": "angry", "annoyance": "angry", "disgust": "angry",
    "disapproval": "angry",
    # curious
    "curiosity": "curious", "surprise": "curious", "realization": "curious",
    "confusion": "curious",
    # neutral
    "neutral": "neutral",
}

_VIBES = {
    "up": {
        "tags": ["feel good", "happy", "upbeat", "indie pop", "funk", "dance"],
        "arxiv": ["creativity and cognition", "music and emotion", "computational creativity"],
        "movies": [("Everything Everywhere All at Once", 2022),
                   ("Spider-Man: Into the Spider-Verse", 2018), ("Sing Street", 2016),
                   ("La La Land", 2016), ("The Princess Bride", 1987),
                   ("School of Rock", 2003), ("Guardians of the Galaxy", 2014),
                   ("Paddington 2", 2017)],
    },
    "tender": {
        "tags": ["romantic", "soul", "acoustic", "dream pop", "r&b"],
        "arxiv": ["empathy and computing", "human connection online", "affective computing"],
        "movies": [("Before Sunrise", 1995), ("Past Lives", 2023),
                   ("Call Me by Your Name", 2017), ("Lady Bird", 2017),
                   ("Little Women", 2019), ("Brooklyn", 2015), ("Her", 2013),
                   ("The Farewell", 2019)],
    },
    "down": {
        "tags": ["calm", "mellow", "chill", "melancholy", "ambient", "sad"],
        "arxiv": ["emotion and wellbeing", "mental health and technology", "affective computing"],
        "movies": [("Amélie", 2001), ("Up", 2009), ("My Neighbor Totoro", 1988),
                   ("Little Miss Sunshine", 2006), ("Chef", 2014), ("About Time", 2013),
                   ("Ratatouille", 2007), ("The Grand Budapest Hotel", 2014)],
    },
    "angry": {
        "tags": ["rock", "punk", "grunge", "hip-hop", "metal"],
        "arxiv": ["stress detection", "conflict and emotion", "sentiment analysis"],
        "movies": [("Mad Max: Fury Road", 2015), ("Whiplash", 2014),
                   ("John Wick", 2014), ("Kill Bill: Vol. 1", 2003),
                   ("Nightcrawler", 2014), ("Baby Driver", 2017),
                   ("Edge of Tomorrow", 2014), ("Snowpiercer", 2013)],
    },
    "curious": {
        "tags": ["experimental", "electronic", "psychedelic", "post-rock", "indie"],
        "arxiv": ["cognitive science", "complexity science", "artificial curiosity",
                  "consciousness and computation"],
        "movies": [("Arrival", 2016), ("Interstellar", 2014), ("Inception", 2010),
                   ("Primer", 2004), ("Coherence", 2013), ("Annihilation", 2018),
                   ("The Prestige", 2006), ("Spirited Away", 2001)],
    },
    "neutral": {
        "tags": ["chillout", "lo-fi", "instrumental", "jazz", "downtempo"],
        "arxiv": ["mindfulness and attention", "perception and cognition", "affective computing"],
        "movies": [("Lost in Translation", 2003), ("Big Fish", 2003),
                   ("The Secret Life of Walter Mitty", 2013), ("Paterson", 2016),
                   ("Columbus", 2017), ("Perfect Days", 2023),
                   ("A Ghost Story", 2017), ("Soul", 2020)],
    },
}

# Cache the RAW fetch (a larger set) per key; we sample from it each call so
# results stay varied run-to-run while being kind to rate limits.
_cache: dict[tuple, list] = {}


def _get(url: str, params: dict):
    import requests
    return requests.get(url, params=params, timeout=_TIMEOUT)


def _music(vibe: dict) -> tuple[list[dict], str]:
    if not config.lastfm_ready():
        return [], "Last.fm key not set"
    tag = random.choice(vibe["tags"])
    ck = ("music", tag)
    try:
        if ck not in _cache:
            r = _get("http://ws.audioscrobbler.com/2.0/", {
                "method": "tag.gettoptracks", "tag": tag,
                "api_key": config.LASTFM_API_KEY, "format": "json", "limit": 60})
            data = r.json()
            if "error" in data:
                return [], "Last.fm error"
            _cache[ck] = [{"title": t["name"], "artist": t["artist"]["name"],
                           "url": t.get("url", "")}
                          for t in data.get("tracks", {}).get("track", [])]
        pool = _cache[ck]
        return (random.sample(pool, min(_N, len(pool))) if pool else []), ""
    except Exception as e:  # noqa: BLE001 -- never leak URL/key
        return [], f"Last.fm unavailable ({type(e).__name__})"


def _movies(vibe: dict) -> tuple[list[dict], str]:
    from urllib.parse import quote
    pool = list(vibe["movies"])
    picks = random.sample(pool, min(_N, len(pool)))
    return ([{"title": t, "year": str(y),
              "url": f"https://letterboxd.com/search/{quote(t)}/"}
             for t, y in picks], "")


def _papers(vibe: dict) -> tuple[list[dict], str]:
    query = random.choice(vibe["arxiv"])
    ck = ("papers", query)
    try:
        import xml.etree.ElementTree as ET
        if ck not in _cache:
            r = _get("http://export.arxiv.org/api/query", {
                "search_query": f"all:{query}", "max_results": 12,
                "sortBy": "relevance"})
            ns = {"a": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(r.text)
            out = []
            for e in root.findall("a:entry", ns):
                title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
                link = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
                author = e.findtext("a:author/a:name", default="", namespaces=ns)
                out.append({"title": " ".join(title.split()),
                            "author": author, "url": link})
            _cache[ck] = out
        pool = _cache[ck]
        return (random.sample(pool, min(_N, len(pool))) if pool else []), ""
    except Exception as e:  # noqa: BLE001
        return [], f"arXiv unavailable ({type(e).__name__})"


def recommend(spec: dict[str, Any]) -> Recommendations:
    if not spec:
        return Recommendations(available=False, note="No recommendation spec.")

    emotion = str(spec.get("emotion", "neutral")).lower()
    vibe = _VIBES[_EMOTION_VIBE.get(emotion, "neutral")]

    music, m_note = _music(vibe)
    movies, v_note = _movies(vibe)
    papers, p_note = _papers(vibe)

    notes = [n for n in (m_note, v_note, p_note) if n]
    return Recommendations(
        available=True,
        note=("; ".join(notes) if notes else f"live ({emotion})"),
        music=music, movies=movies, papers=papers,
    )
