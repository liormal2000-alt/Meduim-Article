import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai")
API_KEY = os.getenv("LLMOD_API_KEY")

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"


def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }


def check_models():
    response = requests.get(
        f"{BASE_URL}/models",
        headers=get_headers(),
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    model_ids = [model["id"] for model in data.get("data", [])]

    print("Available models:")
    for model_id in model_ids:
        print("-", model_id)

    print("\nRequired models:")
    print(f"{CHAT_MODEL}: {CHAT_MODEL in model_ids}")
    print(f"{EMBEDDING_MODEL}: {EMBEDDING_MODEL in model_ids}")


def check_embedding():
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=get_headers(),
        json={
            "model": EMBEDDING_MODEL,
            "input": ["This is a small test sentence."]
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    embedding = data["data"][0]["embedding"]

    print("\nEmbedding test:")
    print("Embedding dimension:", len(embedding))


def check_chat():
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=get_headers(),
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: API connection works"
                }
            ]
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    answer = data["choices"][0]["message"]["content"]

    print("\nChat test:")
    print(answer)


def main():
    if not API_KEY:
        print("Missing LLMOD_API_KEY in .env file")
        sys.exit(1)

    try:
        check_models()
        check_embedding()
        check_chat()
    except requests.RequestException as error:
        print("API request failed:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()