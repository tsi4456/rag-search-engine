from .hybrid_search import (
    rrf_search,
)

from .utils import DEFAULT_SEARCH_LIMIT

from .enhance_query import get_llm_response


def rrf_search_command(
    query,
    k=60,
    limit=DEFAULT_SEARCH_LIMIT,
):

    results = rrf_search(query, k, limit)[:limit]
    docs = []
    print("Search results:")
    for r in results:
        print(f"  - {r['title']}")
        docs.append(f"{r['title']}: {r['document']}")
    print()

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {docs}

        Provide a comprehensive answer that addresses the query:"""
    print("RAG response:")
    print(f"{get_llm_response(prompt)}")


def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    results = rrf_search(query, limit=limit)
    docs = []
    print("Search results:")
    for r in results:
        print(f"  - {r['title']}")
        docs.append(f"{r['title']}: {r['document']}")
    print()
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {docs}

        Provide a comprehensive answer that addresses the query:"""
    print("LLM Summary:")
    print(f"{get_llm_response(prompt)}")


def citation_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    results = rrf_search(query, limit=limit)
    docs = []
    print("Search results:")
    for r in results:
        print(f"  - {r['title']}")
        docs.append(f"{r['title']}: {r['document']}")
    print()

    prompt = f"""Answer the question or provide information based on the provided documents.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {docs}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources using [1], [2], etc. format when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct and informative

        Answer:"""
    print("LLM Summary:")
    print(f"{get_llm_response(prompt)}")


def question_command(question: str, limit: int = DEFAULT_SEARCH_LIMIT):
    results = rrf_search(question, limit=limit)
    docs = []
    print("Search results:")
    for r in results:
        print(f"  - {r['title']}")
        docs.append(f"{r['title']}: {r['document']}")
    print()

    prompt = (
        prompt
    ) = f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {question}

        Documents:
        {docs}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:"""
    print("Answer:")
    print(f"{get_llm_response(prompt)}")
