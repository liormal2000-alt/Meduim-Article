import ast
import hashlib
import json
import os
import re
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


RAW_PATH = "data/medium-english-50mb.csv"
CLEAN_PATH = "data/medium-english-clean.csv"
REPORT_PATH = "data/cleaning_report.txt"

EXPECTED_COLUMNS = [
    "title",
    "text",
    "url",
    "authors",
    "timestamp",
    "tags"
]


def safe_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_article_text(value):
    """
    Light text cleanup only.
    Preserve paragraph structure for the chunking stage.
    """
    text = safe_text(value)

    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove repeated spaces and tabs, but keep newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Keep at most one empty line between paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_list_field(value):
    """
    Convert strings such as:
    "['writing', 'productivity']"
    into a clean Python list.
    """
    text = safe_text(value)

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            items = parsed
        else:
            items = [parsed]

    except (ValueError, SyntaxError):
        items = [text]

    cleaned_items = []

    for item in items:
        item = safe_text(item)

        if item and item not in cleaned_items:
            cleaned_items.append(item)

    return cleaned_items


def normalize_url(value):
    """
    Remove URL fragments and normalize casing where appropriate.
    """
    url = safe_text(value)

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


def normalize_for_hash(value):
    return " ".join(
        safe_text(value).lower().split()
    )


def create_text_hash(text):
    normalized_text = normalize_for_hash(text)

    return hashlib.sha1(
        normalized_text.encode("utf-8")
    ).hexdigest()


def main():
    if not os.path.exists(RAW_PATH):
        print("Raw dataset was not found:", RAW_PATH)
        return

    df = pd.read_csv(RAW_PATH)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    raw_rows = len(df)

    # Preserve the original row number as a stable article ID.
    df.insert(
        0,
        "source_row_id",
        df.index.astype(str)
    )

    df["title"] = df["title"].apply(safe_text)
    df["text"] = df["text"].apply(clean_article_text)
    df["url"] = df["url"].apply(safe_text)
    df["normalized_url"] = df["url"].apply(normalize_url)
    df["timestamp"] = df["timestamp"].apply(safe_text)

    # Save list-like fields in a consistent JSON format.
    df["authors"] = df["authors"].apply(
        lambda value: json.dumps(
            parse_list_field(value),
            ensure_ascii=False
        )
    )

    df["tags"] = df["tags"].apply(
        lambda value: json.dumps(
            parse_list_field(value),
            ensure_ascii=False
        )
    )

    # Remove rows without article content.
    empty_text_mask = df["text"].eq("")
    removed_empty_text = int(empty_text_mask.sum())
    df = df[~empty_text_mask].copy()

    # Remove duplicated normalized URLs, but do not treat empty URLs
    # as duplicates of each other.
    duplicate_url_mask = (
        df["normalized_url"].ne("")
        & df.duplicated(
            subset=["normalized_url"],
            keep="first"
        )
    )

    removed_duplicate_urls = int(
        duplicate_url_mask.sum()
    )

    df = df[~duplicate_url_mask].copy()

    # Remove exact text duplicates after basic normalization.
    df["_text_hash"] = df["text"].apply(
        create_text_hash
    )

    duplicate_text_mask = df.duplicated(
        subset=["_text_hash"],
        keep="first"
    )

    removed_duplicate_texts = int(
        duplicate_text_mask.sum()
    )

    df = df[~duplicate_text_mask].copy()

    df = df.drop(columns=["_text_hash"])
    df = df.reset_index(drop=True)

    df.to_csv(
        CLEAN_PATH,
        index=False
    )

    report_lines = [
        f"Raw rows: {raw_rows}",
        f"Removed rows without text: {removed_empty_text}",
        f"Removed duplicated normalized URLs: {removed_duplicate_urls}",
        f"Removed duplicated article texts: {removed_duplicate_texts}",
        f"Clean rows: {len(df)}",
        f"Saved to: {CLEAN_PATH}"
    ]

    report = "\n".join(report_lines)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as report_file:
        report_file.write(report)

    print(report)


if __name__ == "__main__":
    main()