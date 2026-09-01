from app.models.schemas import RetrievalContext

class PromptPipeline:
    """Handles prompt assembly and versioning."""
    
    SYSTEM_INSTRUCTION_V1 = """You are a helpful and precise assistant. 
Answer the user's question using ONLY the provided context. 
If you cannot answer the question based on the context, say "I cannot find the answer in the provided context." 
Always cite the source chunk IDs you used to formulate your answer."""

    @staticmethod
    def build_rag_prompt_v1(context: RetrievalContext) -> str:
        """Builds a basic prompt combining context and query."""
        
        context_str = "\n\n".join(
            f"--- CHUNK ID: {chunk.metadata.chunk_id} ---\n{chunk.text}"
            for chunk in context.chunks
        )
        
        prompt = f"""CONTEXT:
{context_str}

QUESTION:
{context.query}

Instructions:
Extract the answer from the context above. Provide a structured response."""
        return prompt
