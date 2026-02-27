import argparse
import json

from lib.hybrid_search import rrf_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        golden = json.load(f)

    for test in golden["test_cases"]:
        test["res"] = [r["title"] for r in rrf_search(test["query"], k=60, limit=limit)]
        test["hits"] = set(test["relevant_docs"]) & set(test["res"])
        test["precision"] = float(len(test["hits"])) / limit
        test["recall"] = float(len(test["hits"])) / len(test["relevant_docs"])
        test["f1_score"] = (
            2
            * (test["precision"] * test["recall"])
            / (test["precision"] + test["recall"])
        )

    print(f"k={limit}\n")
    for test in golden["test_cases"]:
        print(f"- Query: {test['query']}")
        print(f"  - Precision@{limit}: {test['precision']:.4f}")
        print(f"  - Recall@{limit}: {test['recall']:.4f}")
        print(f"  - F1 Score: {test['f1_score']:.4f}")
        print(f"  - Retrieved: {', '.join(test['res'])}")
        print(f"  - Relevant: {', '.join(test['hits'])}")
        print()


if __name__ == "__main__":
    main()
