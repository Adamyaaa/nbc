import uuid
from typing import List
from app.models.schemas import DocumentChunk, DocumentMetadata

class Chunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source_filename: str, doc_id: str = None) -> List[DocumentChunk]:
        """Splits text into overlapping chunks."""
        doc_id = doc_id or str(uuid.uuid4())
        chunks = []
        
        # Simple character-based sliding window
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # If we are not at the end of the text, try to find a natural break (like a space or newline)
            if end < text_length:
                # Try to step back to the nearest newline or space
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                
                if last_newline != -1 and last_newline > start + (self.chunk_size // 2):
                    end = last_newline + 1
                elif last_space != -1 and last_space > start + (self.chunk_size // 2):
                    end = last_space + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                metadata = DocumentMetadata(
                    document_id=doc_id,
                    filename=source_filename,
                    chunk_id=str(uuid.uuid4())
                )
                chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
            
            start = end - self.overlap
            
            # Prevent infinite loops if overlap is misconfigured
            if start <= 0 or start >= text_length:
                break
                
        return chunks
