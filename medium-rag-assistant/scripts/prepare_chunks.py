import ast
import hashlib
import os
import sys
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_DIR)

from config import CHUNK_SIZE, OVERLAP_RATIO
from rag.chunking import chunk_text, count_tokens


DATA_PATH = "data/medium-english-clean.csv"
OUTPUT_PATH = "data/chunks_final.csv"

# Keep this small during the smoke test.
# Later we will set it to None for the full ingest.
MAX_ARTICLES = None


def safe_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_list_field(value):
    """
    The CSV stores fields such as authors and tags as strings
    that look like Python lists.

    Example:
    "['writing', 'productivity']"
    """
    text = safe_text(value)

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [text]


def normalize_url(url):
    """
    Remove URL fragments so that:
    article-url
    article-url#section
    are treated as the same article.
    """
    url = safe_text(url)

    if not url:
        return ""

    try:
        parts = urlsplit(url)

        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            parts.query,
            ""
        ))

    except ValueError:
        return url.lower()


def normalize_for_hash(text):
    return " ".join(
        safe_text(text).lower().split()
    )


def create_dedup_group_id(url, title, authors):
    """
    Prefer the normalized URL.
    If URL is missing, fall back to normalized title + authors.
    """
    normalized_url = normalize_url(url)

    if normalized_url:
        source = f"url:{normalized_url}"
    else:
        normalized_title = normalize_for_hash(title)
        normalized_authors = normalize_for_hash(authors)

        source = (
            f"metadata:{normalized_title}|"
            f"{normalized_authors}"
        )

    return hashlib.sha1(
        source.encode("utf-8")
    ).hexdigest()


def main():
    if not os.path.exists(DATA_PATH):
        print("CSV file was not found here:")
        print(DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)

    print("Original rows:", len(df))

    if MAX_ARTICLES is not None:
        df = df.head(MAX_ARTICLES)

    all_chunks = []

    for article_index, row in df.iterrows():
        article_id = safe_text(
            row.get("source_row_id")
        ) or str(article_index)

        title = safe_text(row.get("title"))
        text = safe_text(row.get("text"))
        raw_url = safe_text(row.get("url"))

        url = safe_text(
            row.get("normalized_url")
        ) or raw_url
        timestamp = safe_text(row.get("timestamp"))

        authors_list = parse_list_field(
            row.get("authors")
        )

        tags_list = parse_list_field(
            row.get("tags")
        )

        authors = ", ".join(authors_list)
        tags = ", ".join(tags_list)

        dedup_group_id = create_dedup_group_id(
            url=url,
            title=title,
            authors=authors
        )

        chunks = chunk_text(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap_ratio=OVERLAP_RATIO
        )

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{article_id}-{chunk_index}"

            all_chunks.append({
                "id": chunk_id,
                "article_id": article_id,
                "dedup_group_id": dedup_group_id,
                "chunk_index": chunk_index,
                "title": title,
                "authors": authors,
                "url": url,
                "timestamp": timestamp,
                "tags": tags,
                "chunk": chunk,
                "chunk_tokens": count_tokens(chunk)
            })

    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(OUTPUT_PATH, index=False)

    print("Articles used:", len(df))
    print("Chunks created:", len(chunks_df))
    print("Saved to:", OUTPUT_PATH)

    if not chunks_df.empty:
        print(
            "Maximum chunk tokens:",
            chunks_df["chunk_tokens"].max()
        )

        print(
            "Average chunk tokens:",
            round(
                chunks_df["chunk_tokens"].mean(),
                2
            )
        )


if __name__ == "__main__":
    main()