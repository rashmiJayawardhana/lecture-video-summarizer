"""
Ollama Cloud client for Module 4's content/data-loss check step.

Uses ollama.com's hosted API (free tier, open models) instead of a local
Ollama install — the deployment machine has no GPU/spec to run models
locally, so nothing in this pipeline may depend on a local Ollama server.
"""

import json
import os

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

DEFAULT_MODEL = os.environ.get("OLLAMA_CLOUD_MODEL", "gpt-oss:20b-cloud")


def get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY not set. Sign up at https://ollama.com (no card needed), "
            "create a key under Settings > API Keys, and add it to .env."
        )
    return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})


def chat_json(prompt: str, system: str = "", model: str = DEFAULT_MODEL) -> dict:
    """Send a prompt to Ollama Cloud and parse the reply as JSON."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(model=model, messages=messages, format="json")
    return json.loads(response["message"]["content"])
