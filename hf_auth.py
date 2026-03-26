import os
import sys
from pathlib import Path


TOKEN_FILE = Path("hf_token.txt")
PLACEHOLDER_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"


def load_hf_token():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token and token != PLACEHOLDER_TOKEN:
            return token

    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token

    print(
        "Hugging Face token not found. Put your token on the first line of hf_token.txt.",
        file=sys.stderr,
    )
    print(
        "Example: open hf_token.txt and replace PASTE_YOUR_HF_TOKEN_HERE with your real token.",
        file=sys.stderr,
    )
    sys.exit(1)
