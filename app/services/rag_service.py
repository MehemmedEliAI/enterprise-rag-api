import hashlib
from app.services.vector_store import vector_store_service
from app.services.reranker_service import reranker_service
from app.services.llm_service import llm_service
from app.services.cache_service import cache_service

class RAGService:
    def _get_cache_key(self, query: str) -> str:
        return f"rag_query:{hashlib.sha256(query.strip().lower().encode()).hexdigest()}"

    async def answer_query(self, query: str, top_k: int = 3) -> dict:
        cache_key = self._get_cache_key(query)
        cached_response = cache_service.get(cache_key)
        if cached_response:
            cached_response["cached"] = True
            return cached_response

        raw_results = await vector_store_service.search_hybrid(query=query, limit=10)
        if not raw_results:
            response = {
                "query": query,
                "answer": "No relevant context found in stored documents.",
                "contexts": [],
                "cached": False
            }
            return response

        ranked_results = reranker_service.rerank(query=query, documents=raw_results, top_k=top_k)
        contexts = [doc["text"] for doc in ranked_results]

        answer = await llm_service.generate_answer(query=query, contexts=contexts)

        response = {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "cached": False
        }

        cache_service.set(cache_key, response)
        return response

rag_service = RAGService()