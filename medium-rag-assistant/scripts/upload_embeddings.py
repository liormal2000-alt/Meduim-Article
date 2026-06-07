import os
import sys
import time

import pandas as pd
import requests
from pinecone import Pinecone

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_DIR)

from config import (
    LLMOD_BASE_URL,
    LLMOD_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL
)


INPUT_PATH = "data/chunks_final.csv"

# Smaller batches are slightly slower but more stable.
BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "30")
)

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = 3


def safe_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_embedding_text(row):
    title = safe_text(row.get("title"))
    tags = safe_text(row.get("tags"))
    chunk = safe_text(row.get("chunk"))

    return (
        f"Title: {title}\n"
        f"Tags: {tags}\n"
        f"Content: {chunk}"
    )


def get_embedding_batch(texts):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{LLMOD_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {LLMOD_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts
                },
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            embeddings = [
                item["embedding"]
                for item in data["data"]
            ]

            if len(embeddings) != len(texts):
                raise ValueError(
                    "Embedding API returned an unexpected "
                    "number of vectors"
                )

            return embeddings

        except (
            requests.RequestException,
            ValueError
        ) as error:
            print(
                f"Embedding request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(RETRY_WAIT_SECONDS * attempt)


def get_existing_vector_count(index):
    stats = index.describe_index_stats()

    namespace_stats = stats.get(
        "namespaces",
        {}
    ).get(
        PINECONE_NAMESPACE,
        {}
    )

    return int(
        namespace_stats.get(
            "vector_count",
            0
        )
    )


def main():
    if not os.path.exists(INPUT_PATH):
        print("Chunks file not found:", INPUT_PATH)
        return

    if not LLMOD_API_KEY:
        print("Missing LLMOD_API_KEY")
        return

    if not PINECONE_API_KEY:
        print("Missing PINECONE_API_KEY")
        return

    df = pd.read_csv(INPUT_PATH)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    total = len(df)
    existing_count = get_existing_vector_count(
        index
    )

    print("Namespace:", PINECONE_NAMESPACE)
    print("Chunks in file:", total)
    print("Vectors already uploaded:", existing_count)

    if existing_count >= total:
        print("All chunks are already uploaded")
        return

    print("Resuming from row:", existing_count)

    for start in range(
        existing_count,
        total,
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = df.iloc[start:end]
        records = batch.to_dict("records")

        embedding_inputs = [
            build_embedding_text(row)
            for row in records
        ]

        embeddings = get_embedding_batch(
            embedding_inputs
        )

        vectors = []

        for row, embedding in zip(
            records,
            embeddings
        ):
            vectors.append({
                "id": safe_text(row.get("id")),
                "values": embedding,
                "metadata": {
                    "article_id": safe_text(
                        row.get("article_id")
                    ),
                    "dedup_group_id": safe_text(
                        row.get("dedup_group_id")
                    ),
                    "title": safe_text(
                        row.get("title")
                    ),
                    "authors": safe_text(
                        row.get("authors")
                    ),
                    "url": safe_text(
                        row.get("url")
                    ),
                    "timestamp": safe_text(
                        row.get("timestamp")
                    ),
                    "tags": safe_text(
                        row.get("tags")
                    ),
                    "chunk": safe_text(
                        row.get("chunk")
                    ),
                    "chunk_index": safe_int(
                        row.get("chunk_index")
                    )
                }
            })

        index.upsert(
            vectors=vectors,
            namespace=PINECONE_NAMESPACE
        )

        print(f"Uploaded {end}/{total}")

        time.sleep(0.5)

    print("Done uploading chunks to Pinecone")


if __name__ == "__main__":
    main()