"""
Layer 1 -- Perception: expressed face emotion (DeepFace) + age & gender (InsightFace).

Runs on an in-memory RGB frame and returns a FaceState.
  * Emotion: DeepFace's small (~6MB) expression model.
  * Age + gender: InsightFace buffalo_l genderage -- chosen over DeepFace's age
    model because it is age-DISCRIMINATIVE (DeepFace collapsed different ages to
    ~30; InsightFace separates a 20yo from a 45yo). It reads a touch old, so a
    small correction is applied and the raw value is logged for tuning.
  * Glasses: a light edge-density heuristic, so the age caveat can adapt.

Design rules:
  * Heavy libs imported INSIDE functions; models cached at module level; every
    heavy call serialized with a lock (the live stream + Reflect can collide).
  * Any failure -> available=False with a note; we never raise into the UI.
  * Pixels are read only to derive labels; the frame is never written to disk.

Windows gotcha: DeepFace's logger prints emoji; cp1252 can't encode it and
crashes. We reconfigure stdout/stderr to UTF-8 once at import.
"""

from __future__ import annotations

import threading
from typing import Any

from state import FaceState

# Serialize DeepFace/TensorFlow calls. Under the live stream, a Reflect click and
# a stream frame can hit TF from two threads at once; concurrent first-time DLL
# init crashes on Windows (0x45A), and warmup() below inits it in the main thread.
_tf_lock = threading.Lock()

# Below this DeepFace face_confidence we treat the frame as "no clear face".
# A detected face scores ~0.9+; when nothing is found the whole frame is used
# and confidence collapses toward 0.
_FACE_CONF_MIN = 0.10

# Age + gender come from InsightFace's buffalo_l genderage model, which -- unlike
# DeepFace's age model -- is age-DISCRIMINATIVE (it won't collapse a 20yo and a
# 45yo to the same number). It reads slightly old, so we apply a small correction.
# Raw value is logged so the offset can be tuned from real faces.
_AGE_CORRECTION = 5
_AGE_MIN = 10

_face_app = None            # cached InsightFace FaceAnalysis
_face_app_failed = False
_ig_lock = threading.Lock()  # serialize InsightFace inference across threads


def _get_face_app():
    """Cached InsightFace age+gender analyzer, or None if unavailable."""
    global _face_app, _face_app_failed
    if _face_app is not None:
        return _face_app
    if _face_app_failed:
        return None
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection", "genderage"])
        app.prepare(ctx_id=-1, det_size=(640, 640))   # -1 = CPU
        _face_app = app
        return _face_app
    except Exception:
        _face_app_failed = True
        return None


def _analyze_age_gender(image: Any) -> "tuple[int | None, str | None]":
    """(corrected_age, 'man'/'woman') from InsightFace. (None, None) on failure."""
    app = _get_face_app()
    if app is None:
        return None, None
    try:
        import sys
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        with _ig_lock:
            faces = app.get(bgr)
        if not faces:
            return None, None
        f = max(faces, key=lambda ff: (ff.bbox[2] - ff.bbox[0]) * (ff.bbox[3] - ff.bbox[1]))
        raw_age = int(f.age)
        age = max(_AGE_MIN, raw_age - _AGE_CORRECTION)
        gender = "man" if int(f.gender) == 1 else "woman"
        print(f"[MoodMirror] insightface raw_age={raw_age} -> {age}, gender={gender}",
              file=sys.stderr)
        return age, gender
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"[MoodMirror] age/gender failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None, None

# Glasses heuristic: frames add strong edges across the eye band. Above this
# Canny edge density we call it glasses. Calibrated so a bare face (~0.22 on the
# reference) stays below it; tune from the logged values if it mis-fires.
_GLASSES_EDGE_THRESHOLD = 0.30


def _detect_glasses(image: Any, region: dict) -> "bool | None":
    """Rough glasses guess from eye-band edge density. None if it can't tell."""
    try:
        import sys
        import cv2
        x, y, w, h = region.get("x"), region.get("y"), region.get("w"), region.get("h")
        if not all(isinstance(v, (int, float)) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            return None
        y0, y1 = int(y + 0.22 * h), int(y + 0.50 * h)   # eye band
        x0, x1 = int(x + 0.10 * w), int(x + 0.90 * w)
        crop = image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        density = float(cv2.Canny(gray, 50, 150).mean()) / 255.0
        print(f"[MoodMirror] glasses eye-band density={density:.3f} "
              f"(threshold {_GLASSES_EDGE_THRESHOLD})", file=sys.stderr)
        return density > _GLASSES_EDGE_THRESHOLD
    except Exception:
        return None


def _make_console_utf8() -> None:
    """Idempotent: stop DeepFace's emoji download logs from crashing on Windows."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            if getattr(stream, "encoding", "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


# Run once at import (before Gradio starts logging), not per-request, so we
# never swap the streams Gradio is writing to mid-flight.
_make_console_utf8()


def analyze_face(image: Any, with_age: bool = True) -> FaceState:
    """
    image: an RGB numpy array (H, W, 3), or None.
    with_age: pass False on the live/streaming path -- age is a fuzzy guess and
        recomputing the 539MB age model several times a second is pure waste.
        Reflect (deliberate) keeps with_age=True.
    Returns a populated FaceState.
    """
    if image is None:
        return FaceState(available=False, note="No image provided.")

    try:
        from deepface import DeepFace
    except ImportError:
        return FaceState(available=False, note="DeepFace not installed.")

    # Emotion via DeepFace (tiny ~6MB model). Age + gender come from InsightFace.
    try:
        with _tf_lock:
            res = DeepFace.analyze(image, actions=("emotion",),
                                   enforce_detection=False, silent=True)
    except Exception as e:  # noqa: BLE001
        return FaceState(available=False,
                         note=f"DeepFace could not run. ({type(e).__name__})")

    r = res[0] if isinstance(res, list) else res

    # No clear face -> the emotion read would be meaningless.
    face_conf = float(r.get("face_confidence", 0.0) or 0.0)
    if face_conf < _FACE_CONF_MIN:
        return FaceState(available=False,
                         note="No clear face detected in the image.")

    # Emotion: DeepFace scores are 0-100 and sum to 100; normalize to [0, 1].
    raw = r.get("emotion", {}) or {}
    scores = {str(k): round(float(v) / 100.0, 4) for k, v in raw.items()}
    emotion = str(r.get("dominant_emotion", "")) or None
    emotion_conf = scores.get(emotion, 0.0) if emotion else 0.0

    face = FaceState(
        available=True,
        emotion=emotion,
        emotion_confidence=emotion_conf,
        emotion_scores=scores,
    )

    # Age + gender (deliberate/Reflect path only -- too heavy for the live stream).
    if with_age:
        face.age_estimate, face.gender = _analyze_age_gender(image)
        face.has_glasses = _detect_glasses(image, r.get("region", {}) or {})

    return face
