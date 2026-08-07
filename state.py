"""
The MoodMirror state contract.

This is the single most important file in Phase 0. Every one of the six layers
(see the project plan, §5) reads and writes pieces of one shared object as it
flows through the pipeline. Layer 5's LLM prompt consumes the *union* of all of
them, so if any layer drops a field the whole downstream generation degrades
silently.

Two rules keep the app honest, and they live here as structure:

1. CONFIDENCE TRAVELS. Every perceived signal carries a confidence in [0, 1].
   Congruence (Layer 4) and confidence-aware hedging (§8.3) both break the
   moment a confidence gets lost between layers, so confidence is never
   optional and never implicit.

2. NO IMAGES. This object holds derived signals only -- emotion labels,
   landmark ratios, text -- never pixels, never a file path to an image.
   The §10 privacy promise ("store no images") is enforced by the fact that
   there is nowhere in this contract to put one.

A layer that cannot run (missing model, no face, empty text) does not raise.
It sets its `*_available` flag to False and fills a `*_note` explaining why,
so the UI can render "DeepFace not installed" instead of the app crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# --------------------------------------------------------------------------- #
# Layer 1 -- Perception (DeepFace): expressed face emotion + fuzzy age
# --------------------------------------------------------------------------- #
@dataclass
class FaceState:
    available: bool = False
    note: str = ""                      # why it's unavailable, if it is

    # Expressed emotion in the moment -- NEVER character. See plan §2.
    emotion: Optional[str] = None       # e.g. "happy", "sad", "neutral"
    emotion_confidence: float = 0.0     # [0, 1]
    emotion_scores: dict[str, float] = field(default_factory=dict)  # full dist

    # Age is a fuzzy guess, presented as a range (plan §10). We keep the raw
    # point estimate but the UI must show a hedged band around it.
    age_estimate: Optional[int] = None
    age_low: Optional[int] = None
    age_high: Optional[int] = None

    # Rough glasses guess (Reflect only) so the age caveat can adapt to the
    # actual face instead of always blaming glasses. None = not checked.
    has_glasses: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Layer 2 -- Facial features (MediaPipe Face Mesh): geometry for folklore
# --------------------------------------------------------------------------- #
@dataclass
class FeatureState:
    available: bool = False
    note: str = ""

    # Simple geometric ratios derived from 468 landmarks. Angle/lighting
    # sensitive (plan §10) -- treat as approximate. Keys are stable feature
    # names that the corpus (data/face_reading_corpus.json) keys off of.
    features: dict[str, str] = field(default_factory=dict)   # e.g. {"eyebrows": "thick"}
    raw_ratios: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Layer 3 -- Language understanding (transformers): text emotion / sentiment
# --------------------------------------------------------------------------- #
@dataclass
class TextState:
    available: bool = False
    note: str = ""

    text: str = ""                      # the line the user typed
    emotion: Optional[str] = None       # GoEmotions-style label
    emotion_confidence: float = 0.0     # [0, 1]
    emotion_scores: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Layer 4 -- Congruence: do face, words, (and tradition) agree?  The standout.
# --------------------------------------------------------------------------- #
@dataclass
class CongruenceState:
    available: bool = False
    note: str = ""

    # "congruent" | "incongruent" | "flat" | "insufficient"
    verdict: str = "insufficient"
    confidence: float = 0.0             # how sure we are about the verdict
    explanation: str = ""              # human-readable, feeds the LLM prompt


# --------------------------------------------------------------------------- #
# Layer 5 -- Retrieval + generation output (the three readings + a rec spec)
# --------------------------------------------------------------------------- #
@dataclass
class Readings:
    available: bool = False
    note: str = ""

    emotional_reading: str = ""        # unlimited, hedged (plan §8.3)
    traditional_reading: str = ""      # folklore-labeled, from retrieved corpus
    retrieved_tradition: list[dict[str, Any]] = field(default_factory=list)

    # Structured spec Layer 6 turns into live queries. Not the content itself.
    recommendation_spec: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Layer 6 -- Recommendations (Last.fm / TMDB / arXiv): live content
# --------------------------------------------------------------------------- #
@dataclass
class Recommendations:
    available: bool = False
    note: str = ""

    music: list[dict[str, Any]] = field(default_factory=list)
    movies: list[dict[str, Any]] = field(default_factory=list)
    papers: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The whole thing -- one object, passed layer to layer.
# --------------------------------------------------------------------------- #
@dataclass
class MirrorState:
    face: FaceState = field(default_factory=FaceState)
    features: FeatureState = field(default_factory=FeatureState)
    text: TextState = field(default_factory=TextState)
    congruence: CongruenceState = field(default_factory=CongruenceState)
    readings: Readings = field(default_factory=Readings)
    recommendations: Recommendations = field(default_factory=Recommendations)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for logging / feeding the LLM prompt / SQLite."""
        return asdict(self)
