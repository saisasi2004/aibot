"""The OpenRouter key: from the environment only, optionally filled from a git-ignored .env file.

The .env file is parsed, never sourced: a stray space after "=" would make a shell run the key as
a command and print it. Nothing here logs or returns the value; callers get "present / missing".
"""
from __future__ import annotations

import os
from pathlib import Path

KEY_NAME = "OPENROUTER_API_KEY"


def parse_env(text: str) -> dict[str, str]:
    """KEY=value lines; blanks and comments skipped, whitespace and surrounding quotes removed."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        out[name] = value.strip().strip("'\"").strip()
    return out


def load_dotenv(path: str | Path) -> bool:
    """Fill the environment from a .env file without overriding what is already set."""
    p = Path(path)
    if not p.is_file():
        return False
    for name, value in parse_env(p.read_text()).items():
        if value and not os.environ.get(name):
            os.environ[name] = value
    return True


def api_key() -> str:
    return os.environ.get(KEY_NAME, "").strip()


def key_status() -> str:
    """For logs and startup checks: never the key itself."""
    return f"{KEY_NAME}: present" if api_key() else f"{KEY_NAME}: missing"
