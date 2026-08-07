"""
Layer 1 -- Perception (DeepFace): expressed face emotion + fuzzy age.

PHASE 1: real. Runs DeepFace on an in-memory RGB frame and returns a FaceState.

Design rules honored here:
  * DeepFace / tensorflow are imported INSIDE the function, never at module top,
    so importing this module is instant and app.py launches without them.
  * On any failure (weights not downloaded, no face, bad frame) we return
    available=False with a note -- we never raise into the UI.
  * We read pixels only to derive labels. The frame stays in memory; we never
    write it to disk. (plan §10)

Two Windows-specific gotchas this file handles so you don't have to:
  1. DeepFace's logger prints emoji while downloading weights; the Windows
     cp1252 console can't encode them and crashes the download. We reconfigure
     stdout/stderr to UTF-8 (errors="replace") once before calling DeepFace.
  2. The age model is a ~500MB one-time download. If it isn't present yet, the
     combined emotion+age call raises -- so we fall back to emotion-only (a tiny
     ~6MB model) and mark age unavailable, rather than losing the emotion read.
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

# DeepFace's age model skews OLD and spreads its guesses WIDE. We COMPRESS the
# raw estimate with an affine calibration (scale < 1, then a small shift) so
# predictions pull toward a tighter, more accurate range -- high guesses get
# reeled in more than low ones. Calibrated so a raw ~30 lands near ~21 (typical
# young user). Heuristic, not a trained calibration -- hence the playful caveat.
_AGE_SCALE = 0.72
_AGE_SHIFT = 1
_AGE_MIN = 12

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


def _age_model_present() -> bool:
    """
    True only if the ~500MB age weights are fully on disk. We check this BEFORE
    requesting age, because DeepFace would otherwise start a huge, blocking
    download on the first call and hang the UI. Pre-download deliberately with
    scripts/download_weights.py instead.
    """
    import os
    home = os.environ.get("DEEPFACE_HOME", os.path.expanduser("~"))
    path = os.path.join(home, ".deepface", "weights", "age_model_weights.h5")
    return os.path.isfile(path)


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

    # Only request age if asked AND its weights are already on disk. Otherwise
    # DeepFace would launch a ~500MB blocking download and hang the UI. Emotion
    # is the graded core signal and always runs (its model is a tiny ~6MB).
    got_age = with_age and _age_model_present()
    actions = ("emotion", "age") if got_age else ("emotion",)
    try:
        with _tf_lock:
            res = DeepFace.analyze(image, actions=actions,
                                   enforce_detection=False, silent=True)
    except Exception as e:  # noqa: BLE001
        # Backstop: if the combined call still fails, retry emotion-only.
        if got_age:
            got_age = False
            try:
                with _tf_lock:
                    res = DeepFace.analyze(image, actions=("emotion",),
                                           enforce_detection=False, silent=True)
            except Exception as e2:  # noqa: BLE001
                return FaceState(available=False,
                                 note=f"DeepFace could not run. ({type(e2).__name__})")
        else:
            return FaceState(
                available=False,
                note=("DeepFace could not run -- emotion weights may still be "
                      f"downloading. ({type(e).__name__})"),
            )

    r = res[0] if isinstance(res, list) else res

    # No clear face -> the emotion read would be meaningless.
    face_conf = float(r.get("face_confidence", 0.0) or 0.0)
    if face_conf < _FACE_CONF_MIN:
        return FaceState(
            available=False,
            note="No clear face detected in the image.",
        )

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

    # Age: single compressed/de-biased number (no band), only if age model ran.
    if got_age and r.get("age") is not None:
        raw = float(r["age"])
        face.age_estimate = max(_AGE_MIN, int(round(raw * _AGE_SCALE - _AGE_SHIFT)))
        # age_low/high stay None -- we present one playful number, not a range.
    else:
        face.note = ("Emotion only -- age model not downloaded. "
                     "Run: python scripts/download_weights.py")

    # Glasses guess only on the deliberate (Reflect) path -- lets the age caveat
    # adapt. Skipped on the live stream to keep per-frame cost down.
    if with_age:
        face.has_glasses = _detect_glasses(image, r.get("region", {}) or {})

    return face
