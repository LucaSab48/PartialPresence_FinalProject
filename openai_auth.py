import os
import sys
from pathlib import Path
from typing import Any


TOKEN_FILE = Path("openai_token.txt")
PLACEHOLDER_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def load_openai_settings(required: bool = True) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "api_key": "",
        "model": DEFAULT_OPENAI_MODEL,
        "base_url": "https://api.openai.com/v1",
        "timeout_seconds": 25,
        "enabled": True,
    }

    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token and token != PLACEHOLDER_API_KEY:
            settings["api_key"] = token

    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_api_key:
        settings["api_key"] = env_api_key

    env_model = os.getenv("OPENAI_MODEL", "").strip()
    if env_model:
        settings["model"] = env_model

    if settings["api_key"]:
        return settings

    if required:
        print(
            "OpenAI API key not found. Put your key on the first line of openai_token.txt.",
            file=sys.stderr,
        )
        print(
            "Example: open openai_token.txt and replace PASTE_YOUR_OPENAI_API_KEY_HERE with your real key.",
            file=sys.stderr,
        )
        sys.exit(1)

    return settings


def save_openai_token(api_key: str) -> None:
    token = api_key.strip() or PLACEHOLDER_API_KEY
    TOKEN_FILE.write_text(f"{token}\n", encoding="utf-8")
