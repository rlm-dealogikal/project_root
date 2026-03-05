# /app/llm/openrouter_client.py
import requests
from config.settings import OPENROUTER_API_KEY, MISTRAL_MODEL_ID

def query_llm(prompt):

    if OPENROUTER_API_KEY is None:
        return "API Key Not Configured"

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MISTRAL_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "You are a compliance governance AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]