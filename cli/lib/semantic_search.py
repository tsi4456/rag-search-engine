import numpy as np
import os
import re
import pathlib
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from .utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    format_search_result,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("cannot embed empty string")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}
        doc_strings = [f"{d['title']}: {d['description']}" for d in documents]
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
        with open(MOVIE_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}
        try:
            with open(MOVIE_EMBEDDINGS_PATH, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        except FileNotFoundError:
            pass
        return self.build_embeddings(documents)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        q_embedding = self.generate_embedding(query)
        scores = [
            (cosine_similarity(self.embeddings[i], q_embedding), self.document_map[d])
            for i, d in enumerate(self.document_map)
        ]
        return [
            format_search_result(
                s[1]["id"],
                s[1]["title"],
                s[1]["description"],
                s[0],
            )
            for s in sorted(scores, key=lambda x: x[0], reverse=True)[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[dict] = []

    def build_chunk_metadata(self, documents):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}
        self.chunk_metadata = []
        chunks: list[str] = []
        for i, d in enumerate(documents):
            desc = d["description"].strip()
            if desc == "":
                continue
            doc_chunks = semantic_chunk(
                desc, DEFAULT_MAX_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
            )
            chunks.extend(doc_chunks)
            self.chunk_metadata.extend(
                [
                    {"movie_idx": i, "chunk_idx": j, "total_chunks": len(doc_chunks)}
                    for j, _ in enumerate(doc_chunks)
                ]
            )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)

        pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
        with open(CHUNK_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}
        try:
            with open(CHUNK_EMBEDDINGS_PATH, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(CHUNK_METADATA_PATH, "r") as f:
                json_data = json.load(f)
                self.chunk_metadata = json_data["chunks"]
            if (
                len(self.chunk_embeddings) == json_data["total_chunks"]
                and len(self.chunk_metadata) == json_data["total_chunks"]
            ):
                return self.chunk_embeddings
        except FileNotFoundError:
            pass
        return self.build_chunk_metadata(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )
        q_embedding = self.generate_embedding(query)
        chunk_scores = []
        score_dict = defaultdict(float)
        for i, c in enumerate(self.chunk_embeddings):
            co_sim = cosine_similarity(c, q_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": (m_idx := self.chunk_metadata[i]["movie_idx"]),
                    "score": co_sim,
                }
            )
            score_dict.update({m_idx: max(co_sim, score_dict[m_idx])})
        results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            format_search_result(
                self.documents[doc_id]["id"],
                self.documents[doc_id]["title"],
                self.documents[doc_id]["description"],
                s,
            )
            for doc_id, s in results
        ]


def verify_model():
    sem = SemanticSearch()
    print(f"Model loaded: {sem.model}")
    print(f"Max sequence length: {sem.model.max_seq_length}")


def verify_embeddings():
    sem = SemanticSearch()
    movies = load_movies()
    embeddings = sem.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_chunks():
    c_sem = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = c_sem.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def embed_query_text(query):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    try:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except ZeroDivisionError:
        return 0.0


def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    sem = SemanticSearch()
    movies = load_movies()
    sem.load_or_create_embeddings(movies)
    return sem.search(query, limit)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    overlap = max(overlap, 0)
    words = text.split()
    chunks = []
    while len(words) >= chunk_size:
        chunk, words = " ".join(words[:chunk_size]), words[chunk_size - overlap :]
        chunks.append(chunk)
    if len(words) > overlap:
        chunks.append(" ".join(words))
    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if not text or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) == 1:
        if not sentences[0].endswith((".", "?", "!")):
            pass
    chunks = []
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= overlap:
        return [
            " ".join(sentences),
        ]
    while len(sentences) >= max_chunk_size:
        chunk, sentences = (
            " ".join(sentences[:max_chunk_size]),
            sentences[max_chunk_size - overlap :],
        )
        chunks.append(chunk)
    if len(sentences) > overlap:
        chunks.append(" ".join(sentences))
    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    c_sem = ChunkedSemanticSearch()
    c_sem.load_or_create_chunk_embeddings(movies)

    for i, r in enumerate(c_sem.search_chunks(query, limit), 1):
        print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['document'][:100]}...")
