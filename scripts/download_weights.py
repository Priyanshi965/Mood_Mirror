"""
One-time, deliberate download of DeepFace model weights.

Why this exists: the emotion model is tiny (~6MB) and downloads on first use,
but the AGE model is ~500MB. We never want that to kick off from a UI click and
hang the app, so core/perception.py only uses age if these weights are already
on disk. Run this once, on a good connection, to enable age estimation:

    python scripts/download_weights.py

It's safe to re-run -- DeepFace skips files it already has. Emotion still works
without ever running this; age simply stays "unavailable" until you do.
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

    print("Downloading age weights (~500MB) if missing -- this can take a while ...")
    DeepFace.analyze(dummy, actions=("age",), enforce_detection=False, silent=True)
    print("  age: OK")

    print("\nDone. Age estimation is now enabled in the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
