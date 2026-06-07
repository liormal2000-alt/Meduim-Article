import re
import requests
from urllib.parse import urlsplit, urlunsplit

from pinecone import Pinecone

from config import (
    LLMOD_BASE_URL,
    LLMOD_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    TOP_K,
    MIN_SCORE,
    MAX_EXTRA_SEARCHES
)

from rag.router import detect_query_type, extract_requested_count

# Reuse the same Pinecone index object instead of recreating it
# for every query.
_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)


def get_query_embedding(question):
    response = requests.post(
        f"{LLMOD_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {LLMOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": [question]
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    return data["data"][0]["embedding"]


def normalize_url(url):
    """
    Remove URL fragments so that:
    example.com/article
    example.com/article#section
    are treated as the same article.
    """
    if not url:
        return ""

    try:
        parts = urlsplit(url.strip())

        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            parts.query,
            ""
        ))

    except ValueError:
        return url.strip().lower()


def normalize_text(value):
    value = str(value or "").lower().strip()
    return re.sub(r"\s+", " ", value)


def match_to_dict(match):
    metadata = match.metadata or {}

    return {
        "article_id": str(metadata.get("article_id", "")),
        "dedup_group_id": str(metadata.get("dedup_group_id", "")),
        "title": str(metadata.get("title", "")),
        "authors": str(metadata.get("authors", "")),
        "url": str(metadata.get("url", "")),
        "tags": str(metadata.get("tags", "")),
        "chunk": str(metadata.get("chunk", "")),
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "score": float(match.score)
    }


def query_index(query_embedding, top_k=TOP_K, metadata_filter=None):
    query_args = {
        "vector": query_embedding,
        "top_k": top_k,
        "namespace": PINECONE_NAMESPACE,
        "include_metadata": True
    }

    if metadata_filter:
        query_args["filter"] = metadata_filter

    result = _index.query(**query_args)

    matches = [
        match_to_dict(match)
        for match in result.matches
    ]

    # Pinecone normally returns sorted matches, but sorting explicitly
    # makes the assumption clear.
    return sorted(
        matches,
        key=lambda item: item["score"],
        reverse=True
    )


def keep_relevant(results):
    return [
        item for item in results
        if item["score"] >= MIN_SCORE
    ]


def get_article_group_key(item):
    """
    Prefer an ingest-time deduplication key when available.
    Otherwise, fall back to normalized URL.
    If URL is missing, use title + authors.
    """
    dedup_group_id = item.get("dedup_group_id", "").strip()

    if dedup_group_id:
        return f"group:{dedup_group_id}"

    normalized_url = normalize_url(item.get("url", ""))

    if normalized_url:
        return f"url:{normalized_url}"

    title = normalize_text(item.get("title", ""))
    authors = normalize_text(item.get("authors", ""))

    if title or authors:
        return f"metadata:{title}|{authors}"

    return f"article_id:{item.get('article_id', '')}"


def distinct_articles(results, limit):
    seen_groups = set()
    selected = []

    for item in results:
        article_id = item["article_id"]
        group_key = get_article_group_key(item)

        if not article_id:
            continue

        if group_key in seen_groups:
            continue

        seen_groups.add(group_key)
        selected.append(item)

        if len(selected) == limit:
            break

    return selected


def add_distinct_articles_if_needed(
    query_embedding,
    current_results,
    wanted_articles
):
    """
    Select distinct articles.
    If the first Pinecone query contains too many chunks from the same
    article, perform up to MAX_EXTRA_SEARCHES additional searches while
    excluding article IDs that were already inspected.
    """
    selected = distinct_articles(
        current_results,
        wanted_articles
    )

    seen_groups = {
        get_article_group_key(item)
        for item in selected
    }

    inspected_article_ids = {
        item["article_id"]
        for item in current_results
        if item["article_id"]
    }

    searches_used = 0

    while (
        len(selected) < wanted_articles
        and searches_used < MAX_EXTRA_SEARCHES
    ):
        metadata_filter = None

        if inspected_article_ids:
            metadata_filter = {
                "article_id": {
                    "$nin": sorted(inspected_article_ids)
                }
            }

        extra_results = query_index(
            query_embedding=query_embedding,
            top_k=TOP_K,
            metadata_filter=metadata_filter
        )

        extra_results = keep_relevant(extra_results)

        if not extra_results:
            break

        new_article_added = False

        for item in extra_results:
            article_id = item["article_id"]

            if article_id:
                inspected_article_ids.add(article_id)

            group_key = get_article_group_key(item)

            if not article_id or group_key in seen_groups:
                continue

            seen_groups.add(group_key)
            selected.append(item)
            new_article_added = True

            if len(selected) == wanted_articles:
                break

        searches_used += 1

        # Even if this round found only duplicates, another search may
        # reveal new articles because inspected IDs are now excluded.
        if not new_article_added and not inspected_article_ids:
            break

    return selected, searches_used


def get_chunks_from_article(
    query_embedding,
    article_id,
    limit
):
    """
    Retrieve the strongest supporting chunks from one selected article.
    """
    article_results = query_index(
        query_embedding=query_embedding,
        top_k=limit,
        metadata_filter={
            "article_id": {
                "$eq": article_id
            }
        }
    )

    return keep_relevant(article_results)[:limit]


def retrieve_fact_context(query_embedding, raw_results):
    """
    Fact retrieval should focus on one concrete article.
    """
    top_article = raw_results[0]

    contexts = get_chunks_from_article(
        query_embedding=query_embedding,
        article_id=top_article["article_id"],
        limit=2
    )

    if not contexts:
        contexts = [top_article]

    return contexts, 1


def retrieve_listing_context(
    query_embedding,
    raw_results,
    question
):
    """
    Listing questions require up to 3 distinct articles.
    One representative chunk per article is enough.
    """
    requested_count = extract_requested_count(
        question,
        default=3
    )

    return add_distinct_articles_if_needed(
        query_embedding=query_embedding,
        current_results=raw_results,
        wanted_articles=requested_count
    )


def retrieve_summary_context(query_embedding, raw_results):
    """
    Summary retrieval should use several chunks from the same
    best-matching article.
    """
    top_article = raw_results[0]

    contexts = get_chunks_from_article(
        query_embedding=query_embedding,
        article_id=top_article["article_id"],
        limit=3
    )

    if not contexts:
        contexts = [top_article]

    return contexts, 1


def retrieve_recommendation_context(
    query_embedding,
    raw_results
):
    """
    Compare up to 3 distinct candidate articles.
    Add up to 2 supporting chunks for each candidate.
    """
    candidates, searches_used = add_distinct_articles_if_needed(
        query_embedding=query_embedding,
        current_results=raw_results,
        wanted_articles=3
    )

    contexts = []

    for candidate in candidates:
        supporting_chunks = get_chunks_from_article(
            query_embedding=query_embedding,
            article_id=candidate["article_id"],
            limit=2
        )

        if supporting_chunks:
            contexts.extend(supporting_chunks)
        else:
            contexts.append(candidate)

    return contexts, searches_used


def retrieve_context(question):
    query_type = detect_query_type(question)
    query_embedding = get_query_embedding(question)

    raw_results = query_index(
        query_embedding=query_embedding,
        top_k=TOP_K
    )

    debug_info = {
        "query_type": query_type,
        "raw_results_count": len(raw_results),
        "relevant_results_count": 0,
        "best_score": None,
        "extra_searches": 0,
        "below_threshold": False,
    }

    if not raw_results:
        debug_info["below_threshold"] = True

        return {
            "query_type": query_type,
            "contexts": [],
            "debug": debug_info
        }

    debug_info["best_score"] = raw_results[0]["score"]

    if raw_results[0]["score"] < MIN_SCORE:
        debug_info["below_threshold"] = True

        return {
            "query_type": query_type,
            "contexts": [],
            "debug": debug_info
        }

    relevant_results = keep_relevant(raw_results)

    debug_info["relevant_results_count"] = len(
        relevant_results
    )

    if not relevant_results:
        debug_info["below_threshold"] = True

        return {
            "query_type": query_type,
            "contexts": [],
            "debug": debug_info
        }

    if query_type == "listing":
        contexts, extra_searches = retrieve_listing_context(
            query_embedding,
            relevant_results,
            question
        )

    elif query_type == "summary":
        contexts, extra_searches = retrieve_summary_context(
            query_embedding,
            relevant_results
        )

    elif query_type == "recommendation":
        contexts, extra_searches = retrieve_recommendation_context(
            query_embedding,
            relevant_results
        )

    else:
        contexts, extra_searches = retrieve_fact_context(
            query_embedding,
            relevant_results
        )

    debug_info["extra_searches"] = extra_searches
    debug_info["selected_contexts_count"] = len(contexts)

    return {
        "query_type": query_type,
        "contexts": contexts,
        "debug": debug_info
    }