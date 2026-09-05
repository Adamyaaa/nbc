import os
from dotenv import load_dotenv

# Load the API key from .env
load_dotenv()

from app.core.llm import LLMClient
from app.retrieval.vector_store import VectorStore
from app.retrieval.engine import RetrievalEngine
from app.services.rag import RAGService
from app.ingestion.chunker import Chunker
from app.models.schemas import DocumentChunk, DocumentMetadata

def run_demo():
    print("1. Initializing the System...")
    llm_client = LLMClient()
    vector_store = VectorStore()
    retrieval_engine = RetrievalEngine(vector_store, llm_client)
    rag_service = RAGService(llm_client, retrieval_engine)
    
    print("\n2. Creating a dummy document about a fictional company...")
    sample_text = """
    Acme Corp was founded in 2015 by Jane Doe. 
    The company specializes in building rocket-powered roller skates.
    In 2023, Acme Corp reached $50 million in annual revenue.
    The headquarters is located in Austin, Texas.
    """
    
    print("\n3. Chunking and Embedding the document...")
    chunker = Chunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text(sample_text, source_filename="acme_history.txt")
    
    # Generate embeddings for each chunk using Gemini
    for chunk in chunks:
        chunk.embedding = llm_client.generate_embedding(chunk.text)
        
    # Save into FAISS Vector Database
    vector_store.add_chunks(chunks)
    print(f"   -> Successfully stored {len(chunks)} chunks in FAISS.")

    print("\n4. Asking the RAG system a question...")
    question = "Where is Acme Corp located and what do they build?"
    print(f"   Question: {question}")
    
    # Run the full retrieval + generation pipeline
    answer = rag_service.answer_question(query=question)
    
    print("\n================ SYSTEM RESPONSE ================")
    print(f"Answer: {answer.answer}")
    print(f"Confidence: {answer.confidence}")
    print(f"Sources Used: {answer.sources}")
    print(f"Latency: {answer.metadata['latency_seconds']:.2f} seconds")
    print("=================================================")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Please add your GEMINI_API_KEY to the .env file first!")
    else:
        run_demo()
