"""
Layer 1 -- Perception (DeepFace): expressed face emotion + fuzzy age.

PHASE 0 STATUS: stub. Returns a placeholder FaceState so the pipeline and UI
work end to end. Phase 1 replaces the body of `analyze_face` with a real
DeepFace call. The signature and the returned contract do not change.

Design rules honored here (and to keep honoring in Phase 1):
  * DeepFace / tensorflow are imported INSIDE the function, never at module top,
    so importing this module is instant and app.py can launch without them.
  * On any failure (not installed, no face detected) we return available=False
    with a note -- we never raise into the UI.
  * We read pixels only to derive labels. The incoming `image` is a numpy array
    held in memory; we never write it to disk. (plan §10)
"""

from __future__ import annotations

from typing import Any

from state import FaceState


def analyze_face(image: Any) -> FaceState:
    """
    image: an RGB numpy array (H, W, 3), or None.
    Returns a populated FaceState.
    """
    if image is None:
        return FaceState(available=False, note="No image provided.")

    # ----- PHASE 0 STUB ---------------------------------------------------- #
    # Real Phase 1 body (sketch):
    #   from deepface import DeepFace
    #   res = DeepFace.analyze(image, actions=["emotion", "age"],
    #                          enforce_detection=False)[0]
    #   emotion = res["dominant_emotion"]
    #   scores  = {k: v/100 for k, v in res["emotion"].items()}
    #   age     = int(res["age"])
    #   -> map to FaceState with a +/-6yr band, confidence = scores[emotion]
    return FaceState(
        available=True,
        note="stub",
        emotion="neutral",
        emotion_confidence=0.42,
        emotion_scores={"neutral": 0.42, "happy": 0.30, "sad": 0.28},
        age_estimate=27,
        age_low=22,       # fuzzy band -- age is a guess, off 5-10yr (plan §10)
        age_high=33,
    )
