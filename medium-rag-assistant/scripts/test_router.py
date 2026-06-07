import os
import sys

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_DIR)

from rag.router import (
    detect_query_type,
    extract_requested_count
)


QUESTIONS = [
    "Find an article that reframes marketing as a conversation. Provide the title and author.",
    "List exactly 3 articles about education. Return only the titles.",
    "Give me two articles about artificial intelligence.",
    "Which three articles discuss remote work?",
    "Find an article that lists practical writing exercises.",
    "Explain the central argument of an article about pandemics.",
    "What is the main idea of the mindfulness article?",
    "Recommend an article about building habits.",
    "What should I read if I want beginner-friendly productivity advice?",
    "Recommend an article about productivity and briefly explain its main idea.",
    "List three articles about productivity and summarise each briefly."
]


def main():
    for question in QUESTIONS:
        query_type = detect_query_type(question)
        requested_count = extract_requested_count(question)

        print("\nQUESTION:")
        print(question)

        print("TYPE:", query_type)
        print("REQUESTED COUNT:", requested_count)


if __name__ == "__main__":
    main()