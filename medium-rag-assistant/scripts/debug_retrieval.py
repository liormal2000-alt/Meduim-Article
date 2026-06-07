import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from rag.retrieval import retrieve_context


QUESTIONS = [
    "List exactly 3 articles about education. Return only the titles.",
    "Find an article about building habits and provide its title and author.",
    "Find an article about innovation after pandemics and summarise its central argument.",
    "I want beginner-friendly advice on productivity. Which article would you recommend, and why?",
    "What is the weather in Haifa tomorrow?"
]


def print_result(question):
    result = retrieve_context(question)

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    print("\nQUERY TYPE:")
    print(result["query_type"])

    print("\nDEBUG:")
    print(result["debug"])

    print("\nSELECTED CONTEXTS:")

    if not result["contexts"]:
        print("No contexts selected")
        return

    for index, item in enumerate(result["contexts"], start=1):
        print("\n--- Context", index, "---")
        print("Article ID:", item["article_id"])
        print("Score:", item["score"])
        print("Title:", item["title"])
        print("Authors:", item["authors"])
        print("Chunk:", item["chunk"][:350])


def main():
    for question in QUESTIONS:
        print_result(question)


if __name__ == "__main__":
    main()