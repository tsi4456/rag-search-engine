from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .utils import load_movies


class MultimodalSearch:
    def __init__(
        self, documents: list | None = None, /, model_name: str = "clip-ViT-B-32"
    ):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img_path: str):
        return self.model.encode(
            [
                Image.open(img_path),
            ][0]
        )

    def search_with_image(self, img_path: str):
        embedding = self.embed_image(img_path)
        results = [
            {"score": cosine_similarity(embedding, t), **d}
            for d, t in zip(self.documents, self.text_embeddings)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]


def verify_image_embedding(img_path: str):
    mms = MultimodalSearch()
    embedding = mms.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(img_path: str):
    movies = load_movies()
    mms = MultimodalSearch(movies)
    return mms.search_with_image(img_path)
