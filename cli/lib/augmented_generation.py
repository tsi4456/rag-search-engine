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
        docs.append(f"r['title']: r['document']")

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {docs}

        Provide a comprehensive answer that addresses the query:"""
    print("RAG response:")
    print(f"{get_llm_response(prompt)}")
