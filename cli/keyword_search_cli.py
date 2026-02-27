#!/usr/bin/env python3
import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    bm25_idf_command,
    tf_command,
    bm25_tf_command,
    tf_idf_command,
    search,
    bm25_search,
)

from lib.utils import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Construct searchable movie index")
    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency in target document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("query", type=str, help="Search query")
    idf_parser = subparsers.add_parser(
        "idf", help="Search movies using inverse document frequency"
    )
    idf_parser.add_argument("query", type=str, help="Search query")
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Search movies using both term frequency and inverse document frequency",
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("query", type=str, help="Search query")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "query", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("query", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search(args.query)
            for i, r in enumerate(results):
                print(f"{i + 1}. {r['id']}: {r['title']}")
        case "build":
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            tf = tf_command(args.doc_id, args.query)
            print(
                f'The term "{args.query}" appears in document {args.doc_id} {tf} times'
            )
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.query, args.k1)
            print(
                f"BM25 TF score of '{args.query}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "idf":
            idf = idf_command(args.query)
            print(f"Inverse document frequency of '{args.query}': {idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.query)
            print(f"Inverse document frequency of '{args.query}': {bm25idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.query)
            print(
                f"TF-IDF score of '{args.query}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25search":
            print(f"Searching with BM25 for: {args.query}")
            results = bm25_search(args.query)
            for i, r in enumerate(results):
                print(f"{i + 1}. ({r[0]['id']}) {r[0]['title']} - Score: {r[1]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
