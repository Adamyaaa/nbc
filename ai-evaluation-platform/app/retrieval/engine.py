from typing import List, Tuple
from app.retrieval.vector_store import VectorStore
from app.core.llm import LLMClient
from app.models.schemas import DocumentChunk, RetrievalContext, ExpandedQueries
import logging

logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def expand_query(self, query: str) -> List[str]:
        """Uses the LLM to generate variations of the original query."""
        logger.info(f"Expanding query: '{query}'")
        system_instruction = "You are an expert search assistant. Generate 2 different but highly related versions of the user's query to maximize retrieval in a vector database."
        prompt = f"Original Query: {query}\n\nGenerate exactly 2 alternative queries."
        
        try:
            expanded = self.llm_client.generate_structured(
                prompt=prompt,
                schema=ExpandedQueries,
                system_instruction=system_instruction
            )
            # Combine original query with expanded ones
            all_queries = [query] + expanded.queries
            logger.info(f"Expanded to: {all_queries}")
            return all_queries
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.5, use_expansion: bool = True) -> RetrievalContext:
        """Retrieves chunks, optionally expanding the query to improve recall."""
        queries_to_run = self.expand_query(query) if use_expansion else [query]
        
        all_results = []
        for q in queries_to_run:
            query_embedding = self.llm_client.generate_embedding(q)
            results = self.vector_store.search(query_embedding, top_k=top_k)
            all_results.extend(results)
            
        # Deduplicate by chunk_id and filter by threshold
        unique_chunks = {}
        for chunk, score in all_results:
            if score >= min_score:
                chunk_id = chunk.metadata.chunk_id
                # Keep the highest score if a chunk was found multiple times
                if chunk_id not in unique_chunks or unique_chunks[chunk_id][1] < score:
                    unique_chunks[chunk_id] = (chunk, score)
                    
        # Sort by score descending and take top_k overall
        sorted_results = sorted(unique_chunks.values(), key=lambda x: x[1], reverse=True)[:top_k]
        valid_chunks = [chunk for chunk, score in sorted_results]
        
        return RetrievalContext(chunks=valid_chunks, query=query)
