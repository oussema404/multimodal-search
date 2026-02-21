from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import uuid

class QdrantIndex:
    def __init__(self, collection_name="multimodal_search", dim=512):
        self.client = QdrantClient(":memory:")  # use ":memory:" for local
        # self.client = QdrantClient(host="localhost", port=6333) # to use run this before : docker run -p 6333:6333 qdrant/qdrant
        self.collection_name = collection_name

        # Create collection if not exists
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def add(self, vectors, meta):
        vectors = vectors.cpu().numpy().astype("float32")
        payloads = meta
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(ids=ids, vectors=vectors, payloads=payloads)
        )

    def search(self, query_vector, k=5, filters=None):
        query_vector = query_vector.cpu().numpy().astype("float32")[0]
        query_filter = None

        if filters:
            query_filter = models.Filter(
                must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
                      for k, v in filters.items()]
            )

        results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=k
                ).points

        return [(r.payload, r.score) for r in results]
