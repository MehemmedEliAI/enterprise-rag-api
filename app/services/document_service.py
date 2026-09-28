import io
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    async def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """PDF və ya TXT fayllarından mətni çıxarır."""
        if filename.endswith(".pdf"):
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        elif filename.endswith(".txt"):
            return file_bytes.decode("utf-8")
        else:
            raise ValueError("Desteklenmeyen fayl formati. Yalniz PDF ve TXT qebul olunur.")

    def create_chunks(self, text: str) -> List[str]:
        """Mətni optimal semantik hissələrə bölür."""
        return self.text_splitter.split_text(text)

document_service = DocumentService()