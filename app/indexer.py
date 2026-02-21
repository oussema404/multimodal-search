import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

class VectorIndex:
    def __init__(self, dim=512):
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity
        self.metadata = []

    def add(self, vectors, meta):
        vectors = vectors.cpu().numpy().astype("float32")
        self.index.add(vectors)
        self.metadata.extend(meta)

    def search(self, query_vector, k=5):
        query_vector = query_vector.cpu().numpy().astype("float32")
        scores, idxs = self.index.search(query_vector, k)
        results = [self.metadata[i] for i in idxs[0]]
        return list(zip(results, scores[0]))
