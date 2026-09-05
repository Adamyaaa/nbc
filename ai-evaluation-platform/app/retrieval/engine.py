from typing import List
from app.retrieval.vector_store import VectorStore
from app.core.llm import LLMClient
from app.models.schemas import DocumentChunk, RetrievalContext

class RetrievalEngine:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.5) -> RetrievalContext:
        """Embeds the query and retrieves the top_k most similar chunks that meet the min_score."""
        # 1. Embed the query
        query_embedding = self.llm_client.generate_embedding(query)
        
        # 2. Search the vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 3. Filter by threshold
        valid_chunks = [chunk for chunk, score in results if score >= min_score]
        
        return RetrievalContext(chunks=valid_chunks, query=query)
