from groq import Groq
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = getattr(settings, "GROQ_MODEL_NAME", "openai/gpt-oss-20b")

    async def generate_answer(self, prompt: str = None, query: str = None, context: str = None, contexts: list = None, **kwargs) -> str:
        if not context and contexts:
            if isinstance(contexts, list):
                context = "\n\n".join([str(c) for c in contexts])
            else:
                context = str(contexts)

        if not prompt:
            if query and context:
                prompt = f"Context:\n{context}\n\nQuestion: {query}"
            elif query:
                prompt = query
            else:
                prompt = "No input provided."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful enterprise assistant. Answer accurately based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

llm_service = LLMService()