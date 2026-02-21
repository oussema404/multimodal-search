import pandas as pd
from embedder import CLIPEmbedder
from indexer import VectorIndex
from PIL import Image

class SearchEngine:
    def __init__(self, csv_path="data/captions.csv"):
        self.data = pd.read_csv(csv_path)
        self.embedder = CLIPEmbedder()
        self.index = VectorIndex(dim=512)
        self._build_index()

    def _build_index(self):
        print("Building FAISS index...")
        image_paths = self.data["image_path"].tolist()
        image_embeddings = self.embedder.encode_images(image_paths)
        self.index.add(image_embeddings, self.data.to_dict("records"))
        print(f"Added {len(image_paths)} images to index.")

    def search_text(self, text, k=5):
        query_vector = self.embedder.encode_text([text])
        return self.index.search(query_vector, k)

    def search_image(self, image_path, k=5):
        query_vector = self.embedder.encode_images([image_path])
        return self.index.search(query_vector, k)
