import re


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3
}


LISTING_PATTERNS = [
    r"\blist\b",
    r"\breturn only the titles\b",
    r"\barticle titles\b",
    r"\b(?:give|show|provide)\s+(?:me\s+)?(?:\d+|one|two|three)\s+articles?\b",
    r"\bwhich\s+(?:\d+|one|two|three)\s+articles?\b",
    r"\barticles?\s+(?:about|on|related to)\b"
]


RECOMMENDATION_PATTERNS = [
    r"\brecommend\b",
    r"\brecommendation\b",
    r"\bwhich article should i read\b",
    r"\bwhat should i read\b",
    r"\bwhich article is best\b",
    r"\bbest article for\b",
    r"\barticle for a beginner\b",
    r"\bbeginner[- ]friendly\b",
    r"\bwhich article would you choose\b"
]


SUMMARY_PATTERNS = [
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bmain idea\b",
    r"\bkey idea\b",
    r"\bcentral idea\b",
    r"\bcentral argument\b",
    r"\bcentral claim\b",
    r"\bmain argument\b",
    r"\bmain point\b",
    r"\bbriefly explain\b"
]


FACT_PATTERNS = [
    r"\bfind an article\b",
    r"\blocate an article\b",
    r"\bidentify an article\b",
    r"\bprovide the title and author\b",
    r"\btitle and author\b",
    r"\bwho wrote\b"
]


def normalize_text(question):
    text = question.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def count_matches(text, patterns):
    return sum(
        1 for pattern in patterns
        if re.search(pattern, text)
    )


def extract_requested_count(question, default=3):
    text = normalize_text(question)

    digit_match = re.search(r"\b(\d+)\b", text)

    if digit_match:
        requested = int(digit_match.group(1))
        return max(1, min(requested, 3))

    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return number

    return default


def has_explicit_multiple_articles_request(text):
    """
    Detect a real request for several articles.
    This prevents mistakes such as:
    'Find an article that lists practical exercises.'
    """
    patterns = [
        r"\b(?:\d+|two|three)\s+articles?\b",
        r"\barticle titles\b",
        r"\breturn only the titles\b",
        r"\blist\s+(?:exactly\s+)?(?:\d+|two|three)?\s*articles?\b"
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def detect_query_type(question):
    text = normalize_text(question)

    listing_score = count_matches(text, LISTING_PATTERNS)
    recommendation_score = count_matches(
        text,
        RECOMMENDATION_PATTERNS
    )
    summary_score = count_matches(text, SUMMARY_PATTERNS)
    fact_score = count_matches(text, FACT_PATTERNS)

    # Listing requires a clear request for multiple articles.
    if (
        listing_score > 0
        and has_explicit_multiple_articles_request(text)
    ):
        return "listing"

    # Recommendation gets priority over summary when both appear.
    # Example:
    # "Recommend an article and briefly explain its main idea."
    if recommendation_score > 0:
        return "recommendation"

    if summary_score > 0:
        return "summary"

    if fact_score > 0:
        return "fact"

    return "fact"

