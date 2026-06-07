import os
import sys
import statistics
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from rag.chunking import chunk_text, count_tokens


DATA_PATH = "data/medium-english-50mb.csv"

MAX_ARTICLES = 300

CONFIGURATIONS = [
    {
        "chunk_size": 512,
        "overlap_ratio": 0.15
    },
    {
        "chunk_size": 600,
        "overlap_ratio": 0.15
    },
    {
        "chunk_size": 700,
        "overlap_ratio": 0.15
    }
]


def safe_text(value):
    if pd.isna(value):
        return ""

    return str(value)


def profile_configuration(df, chunk_size, overlap_ratio):
    all_chunk_lengths = []
    chunks_per_article = []

    for _, row in df.iterrows():
        text = safe_text(row.get("text"))

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio
        )

        chunks_per_article.append(len(chunks))

        for chunk in chunks:
            all_chunk_lengths.append(
                count_tokens(chunk)
            )

    return {
        "articles": len(df),
        "total_chunks": len(all_chunk_lengths),
        "avg_chunks_per_article": statistics.mean(chunks_per_article),
        "median_chunks_per_article": statistics.median(chunks_per_article),
        "avg_chunk_tokens": statistics.mean(all_chunk_lengths),
        "median_chunk_tokens": statistics.median(all_chunk_lengths),
        "min_chunk_tokens": min(all_chunk_lengths),
        "max_chunk_tokens": max(all_chunk_lengths)
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.head(MAX_ARTICLES)

    print("Articles tested:", len(df))

    for config in CONFIGURATIONS:
        chunk_size = config["chunk_size"]
        overlap_ratio = config["overlap_ratio"]

        result = profile_configuration(
            df=df,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio
        )

        print("\n" + "=" * 70)
        print("CHUNK SIZE:", chunk_size)
        print("OVERLAP RATIO:", overlap_ratio)

        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()