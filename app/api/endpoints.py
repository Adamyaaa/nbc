import os
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from app.core.llm import LLMClient
from app.retrieval.vector_store import VectorStore
from app.retrieval.engine import RetrievalEngine
from app.services.rag import RAGService
from app.ingestion.parser import parse_document
from app.ingestion.chunker import Chunker
from app.evaluation.judge import LLMJudge
from app.evaluation.runner import ExperimentRunner
from app.models.schemas import Answer, DocumentChunk

router = APIRouter()

# Dependency Initialization
llm_client = LLMClient()
vector_store = VectorStore()
retrieval_engine = RetrievalEngine(vector_store, llm_client)
rag_service = RAGService(llm_client, retrieval_engine)
judge = LLMJudge(llm_client)
runner = ExperimentRunner(rag_service, judge)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    model: str = None

@router.post("/query", response_model=Answer)
async def query_system(request: QueryRequest):
    try:
        return rag_service.answer_question(request.query, request.top_k, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Uploads, parses, chunks, and embeddings a document into the vector store."""
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Parse
        text = parse_document(temp_path)
        
        # Chunk
        chunker = Chunker()
        chunks = chunker.chunk_text(text, file.filename)
        
        # Embed chunks
        for chunk in chunks:
            chunk.embedding = llm_client.generate_embedding(chunk.text)
            
        # Store
        vector_store.add_chunks(chunks)
        
        os.remove(temp_path)
        return {"status": "success", "chunks_added": len(chunks)}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

class EvalRequest(BaseModel):
    dataset: List[dict] # Expected [{"question": "...", "expected_answer": "..."}]
    
@router.post("/evaluate")
async def evaluate_dataset(request: EvalRequest, background_tasks: BackgroundTasks):
    """Runs a batch evaluation in the background."""
    def run_and_save():
        try:
            report = runner.run_experiment(request.dataset)
            runner.save_report(report)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            
    background_tasks.add_task(run_and_save)
    return {"status": "Evaluation started in background."}
