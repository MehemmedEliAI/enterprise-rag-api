from fastapi import APIRouter, HTTPException, Query
from app.services.vector_store import vector_store_service
from app.services.reranker_service import reranker_service

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])

@router.get("/search")
async def search_documents(
    query: str = Query(..., description="Query or keywords to search for"),
    top_k: int = Query(3, description="Number of top relevant contexts to return")
):
    try:
        raw_results = vector_store_service.search_hybrid(query=query, limit=10)
        
        if not raw_results:
            return {"query": query, "total_found": 0, "results": []}

        ranked_results = reranker_service.rerank(query=query, documents=raw_results, top_k=top_k)

        return {
            "query": query,
            "total_found": len(ranked_results),
            "results": ranked_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")