from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import document_service
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

@router.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    """PDF veya TXT senedini qebul edir, bolur ve Hybrid Vector Store-a yazir."""
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Yalniz .pdf ve .txt fayllari desteklenir.")

    try:
        content = await file.read()
        extracted_text = await document_service.extract_text_from_file(content, file.filename)
        chunks = document_service.create_chunks(extracted_text)
        
        indexed_count = await vector_store_service.index_chunks(
            chunks=chunks,
            metadata={"filename": file.filename}
        )

        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks": len(chunks),
            "indexed_chunks": indexed_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sened emalinda xeta yarandi: {str(e)}")