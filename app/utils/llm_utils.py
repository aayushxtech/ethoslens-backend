import os
from groq import Groq
from typing import Optional
from app.config import settings

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is not None:
        return _client
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to your .env or export GROQ_API_KEY in the environment."
        )
    _client = Groq(api_key=api_key)
    return _client


def call_llm(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> str:
    """
    Calls a Groq LLM with the given prompt and returns the response text.
    Client is initialized lazily using app.config.settings.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
