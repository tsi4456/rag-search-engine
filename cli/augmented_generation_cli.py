import argparse

from lib.augmented_generation import (
    rrf_search_command,
    summarize_command,
    citation_command,
    question_command,
)

from lib.utils import DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser(
        "summarize", help="Search and provide summary of results"
    )
    sum_parser.add_argument("query", type=str, help="Search query for RAG")
    sum_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results",
    )

    cit_parser = subparsers.add_parser(
        "citations", help="Search and provide summary of results including citations"
    )
    cit_parser.add_argument("query", type=str, help="Search query for RAG")
    cit_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results",
    )

    q_parser = subparsers.add_parser(
        "question", help="Answer a question in an appropriate manner"
    )
    q_parser.add_argument("question", type=str, help="Question for RAG")
    q_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rrf_search_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case "citations":
            citation_command(args.query, args.limit)
        case "question":
            question_command(args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
