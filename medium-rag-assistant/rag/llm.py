import requests

from config import (
    LLMOD_BASE_URL,
    LLMOD_API_KEY,
    CHAT_MODEL
)


def generate_answer(system_prompt, user_prompt):
    response = requests.post(
        f"{LLMOD_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLMOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        },
        timeout=90
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]