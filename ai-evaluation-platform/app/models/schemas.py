from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    source_information: Optional[str] = None

class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class RetrievalContext(BaseModel):
    chunks: List[DocumentChunk]
    query: str

class Answer(BaseModel):
    answer: str = Field(description="The final answer to the user's question.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    sources: List[str] = Field(description="List of source chunk IDs used to form the answer.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata such as latency, token usage, etc.")

class EvaluationResult(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0, description="Score indicating if the answer is factually correct compared to the expected answer.")
    relevance: float = Field(ge=0.0, le=1.0, description="Score indicating if the answer addresses the question directly without filler.")
    groundedness: float = Field(ge=0.0, le=1.0, description="Score indicating if the answer is fully supported by the provided context.")
    context_precision: float = Field(ge=0.0, le=1.0, description="Score indicating if the retrieved context was highly relevant to the query (penalizes retrieving useless chunks).")
    context_recall: float = Field(ge=0.0, le=1.0, description="Score indicating if the retrieved context contained all necessary information to answer the query.")
    explanation: str = Field(description="The judge's explanation for the scores.")

class ExpandedQueries(BaseModel):
    queries: List[str] = Field(description="List of expanded or rewritten queries to improve retrieval.")
