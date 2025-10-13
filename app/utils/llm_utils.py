import os
from groq import Groq

# Initialize Groq client once (recommended)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> str:
    """
    Calls a Groq LLM with the given prompt and returns the response text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content.strip()
