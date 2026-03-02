"""
OpenAI Embedding Model Wrapper
"""

from openai import OpenAI
from typing import List, Union
import numpy as np
import os


class EmbeddingModel:
    """Wrapper for OpenAI embedding models."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment!")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dimension = 1536  # text-embedding-3-small dimension
        print(f"✓ OpenAI Embeddings initialized. Model: {model_name}")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Convert text to embeddings using OpenAI."""
        # Handle single string
        if isinstance(text, str):
            text = [text]
        
        # Call OpenAI API
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed large batches (OpenAI supports up to 2048 per request)."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed(batch)
            all_embeddings.append(embeddings)
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return np.vstack(all_embeddings)


if __name__ == "__main__":
    # Test
    from dotenv import load_dotenv
    load_dotenv()
    
    embedder = EmbeddingModel()
    
    # Single text
    text = "This is a test document about RAG systems."
    embedding = embedder.embed(text)
    print(f"✓ Single embedding shape: {embedding.shape}")
    
    # Multiple texts
    texts = ["First document", "Second document", "Third document"]
    embeddings = embedder.embed_batch(texts)
    print(f"✓ Batch embeddings shape: {embeddings.shape}")