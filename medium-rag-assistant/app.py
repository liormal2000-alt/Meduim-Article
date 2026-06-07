from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from rag.retrieval import retrieve_context
from rag.prompts import build_augmented_prompt
from rag.llm import generate_answer


app = FastAPI()

NO_ANSWER = "I don't know based on the provided Medium articles data."


class PromptRequest(BaseModel):
    question: str


def public_context(contexts):
    return [
        {
            "article_id": item["article_id"],
            "title": item["title"],
            "chunk": item["chunk"],
            "score": item["score"]
        }
        for item in contexts
    ]


@app.get("/")
def home():
    return {
        "message": "Medium RAG Assistant is running"
    }


@app.get("/api/stats")
def get_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


@app.post("/api/prompt")
def answer_prompt(payload: PromptRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    retrieval_result = retrieve_context(question)

    query_type = retrieval_result["query_type"]
    contexts = retrieval_result["contexts"]

    augmented_prompt = build_augmented_prompt(
        question=question,
        query_type=query_type,
        contexts=contexts
    )

    if not contexts:
        return {
            "response": NO_ANSWER,
            "context": [],
            "Augmented_prompt": augmented_prompt
        }

    answer = generate_answer(
        system_prompt=augmented_prompt["System"],
        user_prompt=augmented_prompt["User"]
    )

    return {
        "response": answer,
        "context": public_context(contexts),
        "Augmented_prompt": augmented_prompt
    }