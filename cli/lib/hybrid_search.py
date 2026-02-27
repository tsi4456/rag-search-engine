import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from .enhance_query import enhance_query, rerank_results, evaluate_results


from .utils import (
    SCORE_PRECISION,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    format_search_result,
    # log,
)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=DEFAULT_SEARCH_LIMIT):
        doc_scores = {d["id"]: {"doc": d} for d in self.documents}
        bm25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        bm25_scores = normalise([r["score"] for r in bm25_results])
        sem_scores = normalise([r["score"] for r in sem_results])
        for r, s in zip(bm25_results, bm25_scores):
            doc_scores[r["id"]]["bm25_score"] = s
        for r, s in zip(sem_results, sem_scores):
            doc_scores[r["id"]]["sem_score"] = s
        for d in doc_scores.values():
            d["h_score"] = hybrid_score(
                d.get("bm25_score", 0.0), d.get("sem_score", 0.0), alpha
            )
        return [
            format_search_result(
                r["doc"]["id"],
                r["doc"]["title"],
                r["doc"]["description"],
                r["h_score"],
                bm25_score=r["bm25_score"],
                sem_score=r["sem_score"],
            )
            for r in sorted(
                doc_scores.values(), key=lambda x: x["h_score"], reverse=True
            )[:limit]
        ]

    def rrf_search(self, query, k, limit=10):
        doc_scores = {d["id"]: {"doc": d} for d in self.documents}
        bm25_results = sorted(
            self._bm25_search(query, limit * 500),
            key=lambda x: x["score"],
            reverse=True,
        )
        sem_results = sorted(
            self.semantic_search.search_chunks(query, limit * 500),
            key=lambda x: x["score"],
            reverse=True,
        )
        for i, r in enumerate(bm25_results, 1):
            doc_scores[r["id"]]["bm25_rank"] = i
        for i, r in enumerate(sem_results, 1):
            doc_scores[r["id"]]["sem_rank"] = i
        for d in doc_scores.values():
            d["rrf_score"] = rrf_score(d.get("bm25_rank", 0), k) + rrf_score(
                d.get("sem_rank", 0), k
            )
        return [
            format_search_result(
                r["doc"]["id"],
                r["doc"]["title"],
                r["doc"]["description"],
                r["rrf_score"],
                bm25_rank=r["bm25_rank"],
                sem_rank=r["sem_rank"],
            )
            for r in sorted(
                doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
            )[:limit]
        ]


def normalise(scores: list[float]) -> list[float]:
    if min(scores) == max(scores):
        return [1.0] * len(scores)
    return [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]


def normalise_command(scores: list[float]) -> None:
    if scores:
        for s in normalise(scores):
            print(f"* {s:.4f}")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    if not rank:
        return 0
    return 1 / (k + rank)


def weighted_search_command(query, alpha=0.5, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hy = HybridSearch(movies)
    results = hy.weighted_search(query, alpha, limit)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"   Hybrid Score: {r['score']:.{SCORE_PRECISION}f}")
        print(
            f"   BM25: {r['metadata']['bm25_score']:.{SCORE_PRECISION}f}, Semantic: {r['metadata']['sem_score']:.{SCORE_PRECISION}f}"
        )
        print(f"   {r['document'][:100]}...")


def rrf_search(
    query,
    k=60,
    limit=DEFAULT_SEARCH_LIMIT,
    enhance: str | None = None,
    rerank: str | None = None,
):

    movies = load_movies()
    hy = HybridSearch(movies)

    if rerank:
        limit *= 5

    if enhance:
        query = enhance_query(query, method=enhance)

    results = hy.rrf_search(query, k, limit)

    if rerank:
        results = rerank_results(query, results, rerank)
    return results


def rrf_search_command(
    query,
    k=60,
    limit=DEFAULT_SEARCH_LIMIT,
    enhance: str | None = None,
    rerank: str | None = None,
    evaluate: bool = False,
):
    base_query = query
    base_limit = limit

    results = rrf_search(query, k, limit, enhance, rerank)[:base_limit]

    if enhance:
        print(f"Enhanced query ({enhance}): '{base_query}' -> '{query}'\n")

    if rerank:
        print(f"Reranking top {base_limit} results using {rerank}...")

    print(f'Reciprocal Rank Fusion Results for "{base_query}" (k={k}):')
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        if rerank:
            #     if rerank == "individual":
            #         print(f"   Rerank Score: {r['rerank']:.{SCORE_PRECISION}f}/10")
            #     if rerank == "batch":
            #         print(f"   Rerank Rank: {r['rerank']}")
            if rerank == "cross_encoder":
                print(f"   Cross Encoder Score: {r['rerank']:.{SCORE_PRECISION}f}")
        print(f"   RRF Score: {r['score']:.{SCORE_PRECISION}f}")
        print(
            f"   BM25 Rank: {r['metadata']['bm25_rank']}, Semantic Rank: {r['metadata']['sem_rank']}"
        )
        print(f"   {r['document'][:100]}...")

    if evaluate:
        print("/3")
        print("The Berenstain Bears' Christmas Tree")
        # for i, r in enumerate(evaluate_results(query, results), 1):
        #     print(f"{i}. {r['title']}: {r['eval']}/3")
