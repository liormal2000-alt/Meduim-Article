import re
import tiktoken

from config import TOKEN_ENCODING


# The embedding model text-embedding-3-small uses cl100k_base.
# We select it explicitly because our institutional model name
# includes an additional prefix.
ENCODING = tiktoken.get_encoding(TOKEN_ENCODING)

SEPARATORS = [
    "\n\n",   # paragraphs
    "\n",     # lines
    ". ",     # sentences
    "! ",
    "? ",
    "; ",
    ", ",
    " "       # words
]


def clean_text(text):
    """
    Basic cleanup while preserving paragraph boundaries.
    """
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove unnecessary spaces, but keep newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Do not keep more than one empty line between paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def count_tokens(text):
    """
    Count tokens according to the tokenizer used by the embedding model.
    """
    if not text:
        return 0

    return len(ENCODING.encode(text))


def split_by_token_limit(text, max_tokens):
    """
    Final fallback for unusually long text segments that could not be
    separated naturally. This guarantees that no piece exceeds max_tokens.
    """
    tokens = ENCODING.encode(text)

    pieces = []

    for start in range(0, len(tokens), max_tokens):
        token_slice = tokens[start:start + max_tokens]
        piece = ENCODING.decode(token_slice).strip()

        if piece:
            pieces.append(piece)

    return pieces


def split_and_keep_separator(text, separator):
    """
    Split text while keeping separators such as punctuation or newlines.
    """
    pattern = f"({re.escape(separator)})"
    parts = re.split(pattern, text)

    pieces = []

    index = 0

    while index < len(parts):
        current = parts[index]

        if index + 1 < len(parts):
            current += parts[index + 1]

        current = current.strip()

        if current:
            pieces.append(current)

        index += 2

    return pieces


def recursive_split(text, max_tokens, separators=None):
    """
    Recursively split text using increasingly smaller semantic boundaries.
    """
    text = text.strip()

    if not text:
        return []

    if count_tokens(text) <= max_tokens:
        return [text]

    if separators is None:
        separators = SEPARATORS

    if not separators:
        return split_by_token_limit(text, max_tokens)

    separator = separators[0]
    parts = split_and_keep_separator(text, separator)

    if len(parts) <= 1:
        return recursive_split(
            text=text,
            max_tokens=max_tokens,
            separators=separators[1:]
        )

    final_parts = []

    for part in parts:
        if count_tokens(part) <= max_tokens:
            final_parts.append(part)
        else:
            smaller_parts = recursive_split(
                text=part,
                max_tokens=max_tokens,
                separators=separators[1:]
            )

            final_parts.extend(smaller_parts)

    return final_parts


def build_overlap(previous_chunk, overlap_tokens):
    """
    Keep a suffix from the previous chunk as overlap.
    The suffix is trimmed to start close to a word boundary.
    """
    if not previous_chunk or overlap_tokens <= 0:
        return ""

    tokens = ENCODING.encode(previous_chunk)

    if len(tokens) <= overlap_tokens:
        return previous_chunk.strip()

    suffix = ENCODING.decode(tokens[-overlap_tokens:]).strip()

    # Avoid beginning the overlap in the middle of a word when possible.
    first_space = suffix.find(" ")

    if first_space != -1:
        suffix = suffix[first_space + 1:].strip()

    return suffix


def append_text(base_text, new_text):
    """
    Join text segments cleanly.
    """
    if not base_text:
        return new_text.strip()

    if not new_text:
        return base_text.strip()

    return f"{base_text}\n\n{new_text}".strip()


def chunk_text(text, chunk_size=600, overlap_ratio=0.15):
    """
    Paragraph-first recursive chunking with a strict token limit.

    chunk_size is measured in tokens.
    overlap_ratio must be between 0 and 0.3.
    """
    text = clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_ratio < 0 or overlap_ratio > 0.3:
        raise ValueError("overlap_ratio must be between 0 and 0.3")

    overlap_tokens = int(chunk_size * overlap_ratio)

    semantic_parts = recursive_split(
        text=text,
        max_tokens=chunk_size
    )

    chunks = []
    current_chunk = ""

    for part in semantic_parts:
        candidate = append_text(current_chunk, part)

        if current_chunk and count_tokens(candidate) > chunk_size:
            chunks.append(current_chunk.strip())

            overlap = build_overlap(
                previous_chunk=current_chunk,
                overlap_tokens=overlap_tokens
            )

            candidate = append_text(overlap, part)

            # In rare cases, overlap may prevent the next part from fitting.
            # Remove the overlap rather than exceeding the token limit.
            if count_tokens(candidate) > chunk_size:
                candidate = part.strip()

            current_chunk = candidate

        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Safety validation: never return an oversized chunk.
    for chunk in chunks:
        if count_tokens(chunk) > chunk_size:
            raise ValueError("Generated chunk exceeds token limit")

    return chunks