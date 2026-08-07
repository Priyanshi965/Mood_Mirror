"""
One-time, deliberate download of the model weights the app gates on.

Why this exists: two models are large enough that auto-downloading them from a
UI click would hang the app, so the app only uses them if they're already on
disk (core/perception.py for age, core/text_emotion.py for text emotion). Run
this once, on a good connection, to enable both:

    python scripts/download_weights.py

Downloads:
  * DeepFace emotion (~6MB)   -- tiny; also downloads on first use anyway
  * DeepFace age (~540MB)     -- enables the fuzzy age band
  * GoEmotions text (~500MB)  -- enables real text-emotion (Layer 3)

Safe to re-run -- each library skips files it already has. The app still runs
without this; the gated layers just stay "unavailable" until you do it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows: DeepFace logs download progress with emoji; make the console UTF-8
# so cp1252 doesn't crash mid-download.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    import numpy as np
    try:
        from deepface import DeepFace
    except ImportError:
        print("DeepFace not installed. Run: pip install -r requirements-heavy.txt")
        return 1

    # A tiny throwaway frame; enforce_detection=False so it runs on anything.
    dummy = (np.random.rand(224, 224, 3) * 255).astype("uint8")

    print("Downloading emotion weights (~6MB) if missing ...")
    DeepFace.analyze(dummy, actions=("emotion",), enforce_detection=False, silent=True)
    print("  emotion: OK")

    print("Downloading age weights (~540MB) if missing -- this can take a while ...")
    DeepFace.analyze(dummy, actions=("age",), enforce_detection=False, silent=True)
    print("  age: OK")

    print("Downloading GoEmotions text-emotion model (~500MB) if missing ...")
    try:
        from transformers import pipeline
        pipeline("text-classification", model="SamLowe/roberta-base-go_emotions",
                 top_k=None)
        print("  text emotion: OK")
    except Exception as e:  # noqa: BLE001
        print(f"  text emotion: FAILED ({type(e).__name__}: {e})")

    print("\nDone. Age + text-emotion are now enabled in the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
