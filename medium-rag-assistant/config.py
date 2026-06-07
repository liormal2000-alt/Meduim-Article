import os
from dotenv import load_dotenv

load_dotenv()

LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai").rstrip("/")
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medium-rag")
PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "medium-articles-dev-rich"
)

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"

EMBEDDING_DIMENSION = 1536

# Explicit tokenizer for text-embedding-3-small.
TOKEN_ENCODING = "cl100k_base"

# Chunk size is measured in tokens.
# This is only a temporary baseline until we compare several sizes locally.
CHUNK_SIZE = 600
OVERLAP_RATIO = 0.15

# Retrieval parameters will be reviewed after chunking is finalized.
TOP_K = 7
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.21"))

MAX_EXTRA_SEARCHES = 2

