import asyncio
import uuid
from typing import Any, Dict, List

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import settings

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
UPSERT_BATCH_SIZE = 128


class VectorStoreService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection = settings.QDRANT_COLLECTION_NAME

        self.dense_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        self.dense_size = len(next(iter(self.dense_model.embed(["probe"]))))

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_NAME: VectorParams(size=self.dense_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF)
            },
        )

    @staticmethod
    def _to_sparse(embedding) -> SparseVector:
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    def _index_chunks_sync(self, chunks: List[str], metadata: Dict[str, Any]) -> int:
        dense_embeddings = list(self.dense_model.embed(chunks))
        sparse_embeddings = list(self.sparse_model.embed(chunks))

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_NAME: dense_embeddings[i].tolist(),
                    SPARSE_NAME: self._to_sparse(sparse_embeddings[i]),
                },
                payload={**metadata, "text": chunk, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]

        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + UPSERT_BATCH_SIZE],
            )
        return len(points)

    async def index_chunks(self, chunks: List[str], metadata: Dict[str, Any]) -> int:
        if not chunks:
            return 0
        return await asyncio.to_thread(self._index_chunks_sync, chunks, metadata)

    def _search_hybrid_sync(self, query: str, limit: int) -> List[Dict[str, Any]]:
        query_dense = next(iter(self.dense_model.query_embed(query))).tolist()
        query_sparse = self._to_sparse(next(iter(self.sparse_model.query_embed(query))))

        prefetch_limit = limit * 3

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=query_dense, using=DENSE_NAME, limit=prefetch_limit),
                Prefetch(query=query_sparse, using=SPARSE_NAME, limit=prefetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": point.id,
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            }
            for point in response.points
        ]

    async def search_hybrid(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._search_hybrid_sync, query, limit)


vector_store_service = VectorStoreService()