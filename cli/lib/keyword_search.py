import math
import os
import pathlib
import pickle
import string
from collections import Counter, defaultdict
from typing import Any

from nltk.stem import PorterStemmer

from .utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    format_search_result,
)

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Any] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenise(text)
        for t in tokens:
            self.index[t].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return float(sum(self.doc_lengths.values())) / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        tokens = tokenise(term)
        if len(tokens) > 1:
            raise ValueError("too many search terms")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term):
        tokens = tokenise(term)
        if len(tokens) > 1:
            raise ValueError("too many search terms")
        t_docs = len(self.docmap)
        q_docs = len(self.index[tokens[0]])
        return math.log((t_docs + 1) / (q_docs + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenise(term)
        if len(tokens) > 1:
            raise ValueError("too many search terms")
        t_docs = len(self.docmap)
        q_docs = len(self.index[tokens[0]])
        return math.log((t_docs - q_docs + 0.5) / (q_docs + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        base_tf = self.get_tf(doc_id, term)
        return (base_tf * (k1 + 1)) / (base_tf + k1 * norm)

    def bm25(self, doc_id, term) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit) -> tuple(list[Any], float):
        tokens = tokenise(query)
        scores = {d: sum([self.bm25(d, t) for t in tokens]) for d in self.docmap}
        return [
            format_search_result(
                doc_id,
                self.docmap[doc_id]["title"],
                self.docmap[doc_id]["description"],
                s,
            )
            for doc_id, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                :limit
            ]
        ]

    def get_documents(self, term) -> list[int]:
        return sorted(list(self.index.get(tokenise(term)[0], set())))

    def build(self):
        movies = load_movies()
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self):
        pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def normalise(inString: str) -> str:
    return inString.lower().translate(str.maketrans("", "", string.punctuation))


def get_tokens(inString: str) -> list:
    return filter(lambda x: x, inString.split())


def filter_tokens(token_list: list[str]):
    stopwords = load_stopwords()
    return [t for t in token_list if t not in stopwords]


def stem_tokens(token_list: list[str]):
    return [stemmer.stem(t) for t in token_list]


def tokenise(inString: str) -> list:
    return stem_tokens(filter_tokens(get_tokens(normalise(inString))))


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search(query: str, max_results: int = DEFAULT_SEARCH_LIMIT) -> list[Any]:
    idx = InvertedIndex()
    idx.load()
    results = set()
    for t in tokenise(query):
        results |= set(idx.get_documents(t))
        if len(results) >= max_results:
            break
    return [idx.docmap[r] for r in list(results)[:max_results]]


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenise(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term) * idx.get_tf(doc_id, term)


def bm25_search(term: str, limit=DEFAULT_SEARCH_LIMIT) -> tuple(list[Any], float):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(term, limit)
