import os
import json
import faiss
import numpy as np
from typing import List, Tuple, Dict
from app.models.schemas import DocumentChunk, DocumentMetadata

class VectorStore:
    def __init__(self, dimension: int = 768, index_path: str = "local_index"):
        """Initialize FAISS Vector Store. Default dimension 768 is common for many text embedding models."""
        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dimension) # Inner Product, equivalent to Cosine Sim if vectors are normalized
        self.chunks_map: Dict[int, DocumentChunk] = {} # Map faiss index ID to chunk data
        self._current_id = 0
        
        self.load_index()

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Adds embedded chunks to the FAISS index."""
        if not chunks:
            return

        embeddings = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.metadata.chunk_id} is missing an embedding.")
            embeddings.append(chunk.embedding)
        
        # Convert to numpy array and normalize for cosine similarity
        embedding_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embedding_array)
        
        # Add to index
        self.index.add(embedding_array)
        
        # Map IDs to chunk metadata/text (faiss IndexFlatIP assigns sequential IDs starting from 0 by default)
        for chunk in chunks:
            self.chunks_map[self._current_id] = chunk
            self._current_id += 1
            
        self.save_index()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        """Search the FAISS index for the most similar chunks."""
        if self.index.ntotal == 0:
            return []
            
        q_arr = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q_arr)
        
        distances, indices = self.index.search(q_arr, top_k)
        
        results = []
        # distances and indices are 2D arrays: [num_queries, num_results]
        for idx in indices[0]:
            if idx != -1 and idx in self.chunks_map:
                results.append(self.chunks_map[idx])
                
        return results

    def save_index(self):
        """Save index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        
        # We need to save the chunks_map but it contains Pydantic models
        map_to_save = {k: v.model_dump() for k, v in self.chunks_map.items()}
        with open(os.path.join(self.index_path, "metadata.json"), "w", encoding='utf-8') as f:
            json.dump(map_to_save, f)

    def load_index(self):
        """Load index and metadata from disk if they exist."""
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file = os.path.join(self.index_path, "metadata.json")
        
        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "r", encoding='utf-8') as f:
                loaded_map = json.load(f)
                
            self.chunks_map = {int(k): DocumentChunk.model_validate(v) for k, v in loaded_map.items()}
            self._current_id = len(self.chunks_map)
