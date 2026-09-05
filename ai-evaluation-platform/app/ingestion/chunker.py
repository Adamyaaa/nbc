import uuid
from typing import List
from app.models.schemas import DocumentChunk, DocumentMetadata

class Chunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source_filename: str, doc_id: str = None) -> List[DocumentChunk]:
        """Splits text into chunks based on natural paragraph boundaries."""
        doc_id = doc_id or str(uuid.uuid4())
        chunks = []
        
        # Split by double newlines (common paragraph separator)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        for p in paragraphs:
            # If a single paragraph is too large, we could split it further, 
            # but for semantic chunking, keeping it intact is often better unless it exceeds a hard limit.
            if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk += p + "\n\n"
            else:
                # Save the current chunk
                if current_chunk:
                    metadata = DocumentMetadata(
                        document_id=doc_id,
                        filename=source_filename,
                        chunk_id=str(uuid.uuid4())
                    )
                    chunks.append(DocumentChunk(text=current_chunk.strip(), metadata=metadata))
                
                # Start new chunk with overlap handling (add last few characters if needed, or just start fresh)
                current_chunk = p + "\n\n"
                
        # Append the final chunk
        if current_chunk:
            metadata = DocumentMetadata(
                document_id=doc_id,
                filename=source_filename,
                chunk_id=str(uuid.uuid4())
            )
            chunks.append(DocumentChunk(text=current_chunk.strip(), metadata=metadata))
                
        return chunks
