import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify that the model is correctly embedding images",
    )
    verify_parser.add_argument("image", type=str, help="Test image to embed")

    img_search_parser = subparsers.add_parser(
        "image_search",
        help="Search using an image",
    )
    img_search_parser.add_argument("image", type=str, help="Image to search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            for i, r in enumerate(image_search_command(args.image), 1):
                print(f"{i}. {r['title']} (similarity: {r['score']:.3f})")
                print(f"   {r['description']}[:100]...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
