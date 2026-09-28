from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1", tags=["RAG"])

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/query")
async def query_rag(request: QueryRequest):
    try:
        return await rag_service.answer_query(query=request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))