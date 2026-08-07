"""
Layer 2 -- Facial features (MediaPipe Face Mesh): geometry for the folklore layer.

PHASE 3: real. Runs MediaPipe's FaceLandmarker (478 landmarks) on an in-memory
frame and derives a handful of coarse geometric features that key into the
traditional face-reading corpus (data/face_reading_corpus.json + core/tradition.py).

Honesty note (matches plan §4/§10): these ratios are ANGLE- AND LIGHTING-
SENSITIVE and the bucket thresholds are calibrated from very little data. That's
fine -- this feeds a for-fun folklore layer, never a factual claim. The features
are deterministic and plausible, which is all the folklore layer needs.

This mediapipe build (0.10.33) ships only the Tasks API (mp.tasks), not the old
mp.solutions, so we use FaceLandmarker with a downloaded .task model. Rules:
  * mediapipe imported inside the loader; landmarker cached at module level.
  * gated on the model file existing (config.FACE_LANDMARKER_PATH) so we never
    trigger a surprise download; degrade to available=False otherwise.
  * no image written to disk.

The `features` dict keys/values must match the corpus entries so retrieval joins.
"""

from __future__ import annotations

from typing import Any

import config
from state import FeatureState

# Canonical MediaPipe Face Mesh landmark indices.
_CHIN, _FOREHEAD = 152, 10
_CHEEK_L, _CHEEK_R = 234, 454
_JAW_L, _JAW_R = 172, 397
_EYE_L_OUT, _EYE_L_IN = 33, 133
_EYE_R_IN, _EYE_R_OUT = 362, 263
_EYE_L_TOP, _EYE_R_TOP = 159, 386
_BROW_L, _BROW_R = 105, 334        # mid-eyebrow points
_LIP_TOP, _LIP_BOT = 0, 17
_MOUTH_L, _MOUTH_R = 61, 291

import threading

_landmarker: object | None = None
_load_failed = False
# Streaming means concurrent requests can hit the cached landmarker. MediaPipe's
# FaceLandmarker.detect() in IMAGE mode isn't documented thread-safe, so serialize.
_detect_lock = threading.Lock()


def _close_landmarker() -> None:
    """Close the cached landmarker while the interpreter is still healthy.

    MediaPipe's FaceLandmarker.__del__ raises noisily if it runs during final
    interpreter teardown (its C++ deps are already gone). Closing at atexit --
    before module teardown -- and dropping the reference avoids that.
    """
    global _landmarker
    if _landmarker is not None:
        try:
            _landmarker.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        _landmarker = None


import atexit  # noqa: E402
atexit.register(_close_landmarker)


def _get_landmarker():
    """Cached FaceLandmarker, or None if the model file isn't present."""
    global _landmarker, _load_failed
    if _landmarker is not None:
        return _landmarker
    if _load_failed:
        return None
    import os
    if not os.path.isfile(config.FACE_LANDMARKER_PATH):
        _load_failed = True
        return None
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=config.FACE_LANDMARKER_PATH),
            num_faces=1, running_mode=vision.RunningMode.IMAGE)
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
        return _landmarker
    except Exception:
        _load_failed = True
        return None


def _bucket(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    """value < low -> labels[0]; value > high -> labels[2]; else labels[1]."""
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def extract_features(image: Any) -> FeatureState:
    """
    image: an RGB numpy array (H, W, 3), or None.
    Returns a FeatureState whose `features` keys align with the corpus.
    """
    if image is None:
        return FeatureState(available=False, note="No image provided.")

    landmarker = _get_landmarker()
    if landmarker is None:
        return FeatureState(
            available=False,
            note=("Face-mesh model not downloaded. "
                  "Run: python scripts/download_weights.py"),
        )

    try:
        import numpy as np
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(image))
        with _detect_lock:
            result = landmarker.detect(mp_img)
    except Exception as e:  # noqa: BLE001
        return FeatureState(available=False,
                            note=f"Face-mesh failed. ({type(e).__name__})")

    if not result.face_landmarks:
        return FeatureState(available=False, note="No face detected for feature reading.")

    p = result.face_landmarks[0]

    def dist(a: int, b: int) -> float:
        return ((p[a].x - p[b].x) ** 2 + (p[a].y - p[b].y) ** 2) ** 0.5

    face_w = dist(_CHEEK_L, _CHEEK_R)
    face_h = dist(_FOREHEAD, _CHIN)
    if face_w == 0 or face_h == 0:
        return FeatureState(available=False, note="Degenerate landmarks; skip reading.")

    ratios = {
        "face_wh": face_w / face_h,
        "jaw_taper": dist(_JAW_L, _JAW_R) / face_w,
        "eye_width": ((dist(_EYE_L_OUT, _EYE_L_IN) + dist(_EYE_R_IN, _EYE_R_OUT)) / 2) / face_w,
        "lip_fullness": dist(_LIP_TOP, _LIP_BOT) / dist(_MOUTH_L, _MOUTH_R),
        "brow_height": ((dist(_BROW_L, _EYE_L_TOP) + dist(_BROW_R, _EYE_R_TOP)) / 2) / face_h,
    }

    # Bucket thresholds -- heuristic, calibrated against a single reference face.
    # Treated as folklore inputs, not measurements. Every feature always resolves
    # to one of three named values, and the corpus has an entry for all of them,
    # so a detected face always gets a claim for each aspect.
    features = {
        "face_shape": _bucket(ratios["face_wh"], 0.75, 0.82, ("long", "oval", "round")),
        "brows":      _bucket(ratios["brow_height"], 0.10, 0.135, ("low", "balanced", "high")),
        "eyes":       _bucket(ratios["eye_width"], 0.195, 0.215, ("small", "average", "large")),
        "jaw":        _bucket(ratios["jaw_taper"], 0.80, 0.86, ("refined", "balanced", "strong")),
        "lips":       _bucket(ratios["lip_fullness"], 0.30, 0.38, ("thin", "average", "full")),
    }

    return FeatureState(
        available=True,
        features=features,
        raw_ratios={k: round(v, 4) for k, v in ratios.items()},
    )
