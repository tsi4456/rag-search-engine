#!/usr/bin/env python3

import argparse

from lib.hybrid_search import (
    normalise_command,
    weighted_search_command,
    rrf_search_command,
)

from lib.utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalise_parser = subparsers.add_parser(
        "normalize", help="Normalise search scores"
    )
    normalise_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normalise"
    )
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Query string")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weighting for keyword results"
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results",
    )
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform an RRF search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Query string")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="Tuning parameter for rankings; low values boost high-ranked results",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method (optional)",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method (optional)",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate results",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalise_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(
                args.query,
                k=args.k,
                limit=args.limit,
                enhance=args.enhance,
                rerank=args.rerank_method,
                evaluate=args.evaluate,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
