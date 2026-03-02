"""
Vector Store using FAISS

Handles storage and similarity search for document embeddings.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import uuid


class VectorStore:
    """
    FAISS-based vector database for semantic search.
    
    Stores embeddings and metadata, enables fast similarity search.
    """
    
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2)
            index_path: Path to load existing index from
        """
        self.dimension = dimension
        self.index_path = index_path
        
        if index_path and Path(index_path).exists():
            self.load(index_path)
        else:
            # Create new FAISS index (L2 distance / cosine similarity)
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = {}  # id -> metadata mapping
            self.id_to_idx = {}  # id -> index position
            self.idx_to_id = {}  # index position -> id
            
        print(f"✓ Vector store initialized. Dimension: {dimension}")
    
    def add(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            metadata: List of metadata dicts (one per embedding)
            ids: Optional list of IDs. If None, UUIDs are generated.
            
        Returns:
            List of IDs for the added embeddings
        """
        n = len(embeddings)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        
        # Validate
        assert len(ids) == n, "Number of IDs must match number of embeddings"
        assert len(metadata) == n, "Number of metadata dicts must match embeddings"
        assert embeddings.shape[1] == self.dimension, f"Embedding dim mismatch: {embeddings.shape[1]} != {self.dimension}"
        
        # Get current index size
        start_idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata and mappings
        for i, (embedding_id, meta) in enumerate(zip(ids, metadata)):
            idx = start_idx + i
            self.id_to_idx[embedding_id] = idx
            self.idx_to_id[idx] = embedding_id
            self.metadata[embedding_id] = meta
        
        print(f"✓ Added {n} vectors. Total: {self.index.ntotal}")
        return ids
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector (shape: [dimension] or [1, dimension])
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: id, score, metadata
        """
        # Ensure 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than k results
                continue
                
            doc_id = self.idx_to_id[idx]
            results.append({
                'id': doc_id,
                'score': float(dist),  # L2 distance (lower is better)
                'metadata': self.metadata[doc_id]
            })
        
        return results
    
    def save(self, path: str):
        """Save index and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }, f)
        
        print(f"✓ Index saved to {path}")
    
    def load(self, path: str):
        """Load index and metadata from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.id_to_idx = data['id_to_idx']
            self.idx_to_id = data['idx_to_id']
        
        print(f"✓ Index loaded from {path}. Vectors: {self.index.ntotal}")
    
    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal


if __name__ == "__main__":
    # Test the vector store
    print("Testing VectorStore...")
    
    # Create store
    store = VectorStore(dimension=384)
    
    # Create dummy embeddings
    embeddings = np.random.rand(10, 384).astype('float32')
    metadata = [{'text': f'Document {i}', 'source': 'test'} for i in range(10)]
    
    # Add to store
    ids = store.add(embeddings, metadata)
    print(f"Added IDs: {ids[:3]}...")
    
    # Search
    query = np.random.rand(384).astype('float32')
    results = store.search(query, top_k=3)
    
    print(f"\nSearch results (top 3):")
    for r in results:
        print(f"  ID: {r['id'][:8]}... | Score: {r['score']:.4f} | Text: {r['metadata']['text']}")
    
    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        
        new_store = VectorStore(dimension=384, index_path=tmpdir)
        assert new_store.size == store.size
        print(f"\n✓ Save/load test passed!")
    
    print("\n✓ VectorStore test completed!")