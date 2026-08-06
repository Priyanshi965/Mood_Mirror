"""
Week-1 de-risk (plan §11): prove one round-trip to the chosen LLM works.

Run it directly:  python scripts/smoke_llm.py

It reads GROQ_API_KEY from your .env. If the key is missing it prints how to
get one and exits cleanly (code 2) -- the scaffold must never DEPEND on a key
being present. Nothing else in the app imports this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a loose script: make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def main() -> int:
    if config.LLM_PROVIDER != "groq":
        print(f"LLM_PROVIDER is '{config.LLM_PROVIDER}'. This smoke test targets Groq.")
        print("Set LLM_PROVIDER=groq in .env to use it, or adapt this script.")
        return 2

    if not config.GROQ_API_KEY:
        print("No GROQ_API_KEY found.")
        print("  1. Get a free key at https://console.groq.com/keys")
        print("  2. Copy .env.example to .env and paste it in.")
        print("  3. Re-run: python scripts/smoke_llm.py")
        return 2

    try:
        import requests
    except ImportError:
        print("requests not installed:  pip install requests")
        return 1

    print(f"Calling Groq ({config.GROQ_MODEL}) ...")
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json={
                "model": config.GROQ_MODEL,
                "messages": [
                    {"role": "user",
                     "content": "In one short sentence, say hello as a reflective mirror."}
                ],
                "max_tokens": 60,
            },
            timeout=30,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]["content"].strip()
        print("\nOK. Model replied:\n  " + msg)
        return 0
    except Exception as e:  # noqa: BLE001 -- smoke test wants the raw reason
        print(f"\nFAILED: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
