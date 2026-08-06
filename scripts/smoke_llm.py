"""
Week-1 de-risk (plan §11): prove one round-trip to the chosen LLM works.

Run it directly:  python scripts/smoke_llm.py

It reads whatever provider LLM_PROVIDER points at (default: OpenRouter) via
config.active_llm(), so the same script works for OpenRouter / Groq / Ollama --
they're all OpenAI-compatible. If the required key is missing it prints how to
get one and exits cleanly (code 2). The scaffold never DEPENDS on a key.
Nothing else in the app imports this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a loose script: make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def main() -> int:
    llm = config.active_llm()

    if not llm["base_url"]:
        print(f"LLM_PROVIDER='{config.LLM_PROVIDER}' isn't on the OpenAI-compatible "
              "path this script uses (OpenRouter / Groq / Ollama). Set LLM_PROVIDER "
              "to one of those in .env.")
        return 2

    if not llm["ready"]:
        print(f"No API key for provider '{config.LLM_PROVIDER}'.")
        if config.LLM_PROVIDER == "openrouter":
            print("  1. Get a free key at https://openrouter.ai/keys")
            print("  2. Copy .env.example to .env and set OPENROUTER_API_KEY=...")
            print("  3. Re-run: python scripts/smoke_llm.py")
        else:
            print("  Set the matching *_API_KEY in your .env, then re-run.")
        return 2

    try:
        import requests
    except ImportError:
        print("requests not installed:  pip install requests")
        return 1

    print(f"Calling {config.LLM_PROVIDER} ({llm['model']}) ...")
    try:
        resp = requests.post(
            f"{llm['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {llm['api_key']}", **llm["headers"]},
            json={
                "model": llm["model"],
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
        # OpenRouter returns a helpful JSON error body -- surface it if present.
        body = getattr(getattr(e, "response", None), "text", None)
        if body:
            print("  response body: " + body[:300])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
