import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "medium-rag"
DIMENSION = 1536

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully")

else:
    print("Index already exists")