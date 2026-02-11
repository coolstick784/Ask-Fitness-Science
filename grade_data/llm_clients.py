"""
llm_clients.py
Call Ollama
"""
import requests
from config import OLLAMA_API_BASE


def call_ollama(model: str, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": int(max_tokens)},
    }
    resp = requests.post(
        f"{OLLAMA_API_BASE}/api/generate",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return str(resp.json().get("response", "")).strip()
