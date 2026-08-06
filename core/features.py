"""
Layer 2 -- Facial features (MediaPipe Face Mesh): geometry for the folklore layer.

PHASE 0 STATUS: stub. Returns placeholder feature labels. Phase 3 replaces the
body with a real MediaPipe Face Mesh pass (468 landmarks) and computes geometric
ratios: eyebrow thickness, jaw width, face shape, lip fullness, eye size.

The `features` dict keys must match the `feature` field in
data/face_reading_corpus.json -- that's how retrieval (also Phase 3) joins a
detected feature to a traditional claim.

Rules: mediapipe imported inside the function; failures degrade, never raise;
no image is written to disk.
"""

from __future__ import annotations

from typing import Any

from state import FeatureState


def extract_features(image: Any) -> FeatureState:
    """
    image: an RGB numpy array (H, W, 3), or None.
    Returns a populated FeatureState whose `features` keys align with the corpus.
    """
    if image is None:
        return FeatureState(available=False, note="No image provided.")

    # ----- PHASE 0 STUB ---------------------------------------------------- #
    # Real Phase 3 body (sketch):
    #   import mediapipe as mp
    #   mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, ...)
    #   landmarks = mesh.process(image)  -> 468 points
    #   ratios = compute_ratios(landmarks)          # eyebrow/jaw/eye/lip/shape
    #   features = bucketize(ratios)                # -> "thick"/"thin" etc.
    return FeatureState(
        available=True,
        note="stub",
        features={
            "eyebrows": "thick",
            "face_shape": "oval",
            "lips": "full",
        },
        raw_ratios={
            "eyebrow_thickness_ratio": 0.13,
            "jaw_width_ratio": 0.78,
            "lip_fullness_ratio": 0.34,
        },
    )
