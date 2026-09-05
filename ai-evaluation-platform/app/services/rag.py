import time
import logging
from typing import Optional
from app.core.llm import LLMClient
from app.retrieval.engine import RetrievalEngine
from app.generation.prompts import PromptPipeline
from app.models.schemas import Answer

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, llm_client: LLMClient, retrieval_engine: RetrievalEngine):
        self.llm_client = llm_client
        self.retrieval_engine = retrieval_engine

    def answer_question(self, query: str, top_k: int = 5, model: Optional[str] = None) -> Answer:
        """End-to-end RAG pipeline execution."""
        logger.info(f"Answering query: '{query}'")
        start_time = time.time()
        
        # 1. Retrieve context
        context = self.retrieval_engine.retrieve(query, top_k=top_k)
        
        if not context.chunks:
            logger.warning(f"No relevant context found for query: '{query}'")
            return Answer(
                answer="I cannot find the answer in the provided context because no relevant documents were found.",
                confidence=0.0,
                sources=[],
                metadata={"latency_seconds": time.time() - start_time, "chunks_retrieved": 0}
            )
        
        logger.info(f"Retrieved {len(context.chunks)} chunks for context.")
        
        # 2. Build prompt
        prompt = PromptPipeline.build_rag_prompt_v1(context)
        system_instruction = PromptPipeline.SYSTEM_INSTRUCTION_V1
        
        # 3. Generate structured answer
        logger.info("Generating structured answer via LLM...")
        answer = self.llm_client.generate_structured(
            prompt=prompt,
            schema=Answer,
            model=model,
            system_instruction=system_instruction
        )
        
        # 4. Attach metadata
        latency = time.time() - start_time
        answer.metadata["latency_seconds"] = latency
        answer.metadata["chunks_retrieved"] = len(context.chunks)
        
        logger.info(f"Answer generated successfully in {latency:.2f}s with confidence {answer.confidence}")
        
        return answer
