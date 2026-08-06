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

from typing import Any

from state import FaceState

# Below this DeepFace face_confidence we treat the frame as "no clear face".
# A detected face scores ~0.9+; when nothing is found the whole frame is used
# and confidence collapses toward 0.
_FACE_CONF_MIN = 0.10

# Age is a guess (off 5-10yr, worse in poor light) -- present it as a band.
_AGE_BAND = 6


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


def analyze_face(image: Any) -> FaceState:
    """
    image: an RGB numpy array (H, W, 3), or None.
    Returns a populated FaceState.
    """
    if image is None:
        return FaceState(available=False, note="No image provided.")

    try:
        from deepface import DeepFace
    except ImportError:
        return FaceState(available=False, note="DeepFace not installed.")

    # Only request age if its weights are already on disk. Otherwise DeepFace
    # would launch a ~500MB blocking download and hang the UI. Emotion is the
    # graded core signal and always runs (its model is a tiny ~6MB).
    got_age = _age_model_present()
    actions = ("emotion", "age") if got_age else ("emotion",)
    try:
        res = DeepFace.analyze(image, actions=actions,
                               enforce_detection=False, silent=True)
    except Exception as e:  # noqa: BLE001
        # Backstop: if the combined call still fails, retry emotion-only.
        if got_age:
            got_age = False
            try:
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

    # Age (fuzzy band), only if the age model actually ran.
    if got_age and r.get("age") is not None:
        age = int(round(float(r["age"])))
        face.age_estimate = age
        face.age_low = max(0, age - _AGE_BAND)
        face.age_high = age + _AGE_BAND
    else:
        face.note = ("Emotion only -- age model not downloaded. "
                     "Run: python scripts/download_weights.py")

    return face
