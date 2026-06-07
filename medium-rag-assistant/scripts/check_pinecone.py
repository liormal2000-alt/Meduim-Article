import os
import sys
from pinecone import Pinecone

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)


def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()

    print("Configured index:", PINECONE_INDEX_NAME)
    print("Configured namespace:", PINECONE_NAMESPACE)

    print("\nFull index stats:")
    print(stats)

    print("\nNamespaces found:")

    namespaces = stats.get("namespaces", {})

    if not namespaces:
        print("No namespaces found")
        return

    for namespace_name, namespace_data in namespaces.items():
        print(
            f"- {namespace_name}: "
            f"{namespace_data.get('vector_count', 0)} vectors"
        )


if __name__ == "__main__":
    main()