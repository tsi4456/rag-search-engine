#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_text,
    embed_chunks,
    embed_query_text,
    search_command,
    search_chunked_command,
    chunk_text,
    semantic_chunk_text,
)
from lib.utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify model information")
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    subparsers.add_parser("embed_chunks", help="Build chunk metadata")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for query text"
    )
    embed_query_parser.add_argument("text", type=str, help="Text to embed")
    search_parser = subparsers.add_parser(
        "search", help="Perform semantic search for a given term"
    )
    search_parser.add_argument("query", type=str, help="Term to search for")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Max results to return",
    )
    chunk_search_parser = subparsers.add_parser(
        "search_chunked", help="Perform semantic search for a given term"
    )
    chunk_search_parser.add_argument("query", type=str, help="Term to search for")
    chunk_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Max results to return",
    )
    chunk_parser = subparsers.add_parser("chunk", help="Split text into chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="Max size of chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap",
    )
    sem_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text by semantic units"
    )
    sem_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    sem_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_MAX_CHUNK_SIZE,
        help="Max size of chunk",
    )
    sem_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap",
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_chunks":
            embed_chunks()
        case "embedquery":
            embed_query_text(args.text)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case "search":
            for i, r in enumerate(search_command(args.query, args.limit), 1):
                print(
                    f"{i}. {r['title']} (score: {r['score']:.4f})\n   {r['description']}[:100]...\n"
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
