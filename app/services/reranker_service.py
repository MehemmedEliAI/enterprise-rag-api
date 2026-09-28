from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from app.core.config import settings

class ReRankerService:
    def __init__(self):
        self.ranker = Ranker(model_name=settings.RERANKER_MODEL_NAME)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not documents:
            return []

        pass_docs = [
            {"id": doc["id"], "text": doc["text"], "meta": doc["metadata"]}
            for doc in documents
        ]

        rerank_request = RerankRequest(query=query, passages=pass_docs)
        results = self.ranker.rerank(rerank_request)

        return results[:top_k]

reranker_service = ReRankerService()