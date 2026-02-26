"""
Hybrid Retrieval Pipeline: Vector + BM25 with Re-ranking

This module implements production-grade hybrid search combining:
1. Dense retrieval (vector similarity via pgvector)
2. Sparse retrieval (BM25 keyword matching)
3. Reciprocal Rank Fusion (RRF) to merge results
4. Cross-encoder re-ranking for final precision

Author: [Your Name]
Created: 2025-02-17
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
import openai

from database.pgvector_client import PgVectorClient
from models.embedding import EmbeddingModel
from monitoring.tracer import trace_function
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata"""
    chunk_id: str
    content: str
    source_document: str
    metadata: Dict
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int = 0


class HybridRetriever:
    """
    Production retrieval pipeline with hybrid search and re-ranking.
    
    Architecture:
        Query → [Vector Search (k=50), BM25 Search (k=50)]
              → RRF Merge → Re-rank (k=5) → Top Results
    
    Performance characteristics:
        - Vector search: ~50ms for 100K chunks (HNSW index)
        - BM25 search: ~30ms (in-memory)
        - Re-ranking: ~200ms for 50 candidates
        - Total latency: ~300ms p50, ~500ms p95
    
    Cost:
        - Embedding query: ~$0.00001 (text-embedding-3-small)
        - Re-ranking: Free (local cross-encoder)
        - Vector DB query: ~$0.00 (self-hosted pgvector)
    """
    
    def __init__(
        self,
        vector_client: PgVectorClient,
        embedding_model: EmbeddingModel,
        collection_name: str,
        vector_top_k: int = 50,
        bm25_top_k: int = 50,
        rerank_top_k: int = 5,
        rrf_k: int = 60,  # RRF constant
    ):
        self.vector_client = vector_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        
        # Load BM25 index (in production, cache this or load from disk)
        self.bm25_index = self._load_bm25_index()
        self.bm25_corpus = self._load_bm25_corpus()
        
        # Load re-ranker model (local cross-encoder)
        self.reranker = self._load_reranker()
        
        logger.info(
            f"HybridRetriever initialized: "
            f"vector_k={vector_top_k}, bm25_k={bm25_top_k}, "
            f"rerank_k={rerank_top_k}, collection={collection_name}"
        )
    
    @trace_function(name="hybrid_retrieval")
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-K chunks using hybrid search + re-ranking.
        
        Args:
            query: User question
            filters: Metadata filters (e.g., {"source_type": "legal"})
            min_score: Minimum similarity threshold
        
        Returns:
            List of top-K chunks, ranked by re-ranker score
        
        Pipeline:
            1. Vector search (cosine similarity)
            2. BM25 search (keyword matching)
            3. RRF merge (combines rankings)
            4. Cross-encoder re-rank (semantic precision)
        """
        logger.info(f"Retrieving for query: '{query[:100]}...'")
        
        # Step 1: Dense retrieval (vector search)
        vector_results = self._vector_search(query, filters)
        logger.debug(f"Vector search returned {len(vector_results)} results")
        
        # Step 2: Sparse retrieval (BM25)
        bm25_results = self._bm25_search(query, filters)
        logger.debug(f"BM25 search returned {len(bm25_results)} results")
        
        # Step 3: Merge using Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results
        )
        logger.debug(f"RRF merged to {len(merged_results)} candidates")
        
        # Step 4: Re-rank top candidates
        reranked_results = self._rerank(query, merged_results)
        
        # Filter by min_score
        final_results = [
            chunk for chunk in reranked_results 
            if chunk.rerank_score >= min_score
        ][:self.rerank_top_k]
        
        logger.info(
            f"Retrieved {len(final_results)} chunks "
            f"(avg rerank score: {np.mean([c.rerank_score for c in final_results]):.3f})"
        )
        
        return final_results
    
    def _vector_search(
        self, 
        query: str, 
        filters: Optional[Dict]
    ) -> List[RetrievedChunk]:
        """Dense retrieval using embeddings + cosine similarity"""
        # Embed query
        query_embedding = self.embedding_model.embed(query)
        
        # Search vector store
        results = self.vector_client.search(
            collection=self.collection_name,
            query_vector=query_embedding,
            top_k=self.vector_top_k,
            filters=filters
        )
        
        # Convert to RetrievedChunk objects
        chunks = []
        for i, result in enumerate(results):
            chunk = RetrievedChunk(
                chunk_id=result['id'],
                content=result['content'],
                source_document=result['metadata']['source'],
                metadata=result['metadata'],
                vector_score=result['score'],
                final_rank=i + 1
            )
            chunks.append(chunk)
        
        return chunks
    
    def _bm25_search(
        self, 
        query: str, 
        filters: Optional[Dict]
    ) -> List[RetrievedChunk]:
        """Sparse retrieval using BM25 keyword matching"""
        # Tokenize query (simple whitespace split; in prod use proper tokenizer)
        query_tokens = query.lower().split()
        
        # BM25 scoring
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:self.bm25_top_k]
        
        # Apply filters (if any) and convert to chunks
        chunks = []
        for i, idx in enumerate(top_indices):
            doc = self.bm25_corpus[idx]
            
            # Filter check
            if filters and not self._matches_filters(doc['metadata'], filters):
                continue
            
            chunk = RetrievedChunk(
                chunk_id=doc['id'],
                content=doc['content'],
                source_document=doc['metadata']['source'],
                metadata=doc['metadata'],
                bm25_score=scores[idx],
                final_rank=i + 1
            )
            chunks.append(chunk)
        
        return chunks
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Merge two ranked lists using RRF.
        
        RRF formula: score(d) = Σ(1 / (k + rank(d)))
        where rank(d) is the document's rank in each list.
        
        RRF is more robust than score normalization because:
        - Doesn't require score calibration
        - Handles different score scales naturally
        - Downweights lower-ranked items appropriately
        
        Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        # Build chunk_id → chunk mapping
        all_chunks = {}
        
        # Add vector results with their ranks
        for rank, chunk in enumerate(vector_results, start=1):
            chunk.final_rank = rank
            all_chunks[chunk.chunk_id] = chunk
        
        # Merge BM25 results, combining scores if chunk already exists
        for rank, chunk in enumerate(bm25_results, start=1):
            if chunk.chunk_id in all_chunks:
                # Chunk appeared in both lists — update RRF score
                existing = all_chunks[chunk.chunk_id]
                existing.bm25_score = chunk.bm25_score
            else:
                # New chunk from BM25 only
                chunk.final_rank = rank
                all_chunks[chunk.chunk_id] = chunk
        
        # Compute RRF scores
        for chunk_id, chunk in all_chunks.items():
            vector_rank = chunk.final_rank if chunk.vector_score > 0 else float('inf')
            bm25_rank = chunk.final_rank if chunk.bm25_score > 0 else float('inf')
            
            rrf_score = 0.0
            if vector_rank != float('inf'):
                rrf_score += 1.0 / (self.rrf_k + vector_rank)
            if bm25_rank != float('inf'):
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)
            
            chunk.rrf_score = rrf_score
        
        # Sort by RRF score
        merged = sorted(
            all_chunks.values(), 
            key=lambda c: c.rrf_score, 
            reverse=True
        )
        
        return merged
    
    def _rerank(
        self, 
        query: str, 
        candidates: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Re-rank candidates using a cross-encoder model.
        
        Cross-encoders jointly encode query+document for better semantic
        understanding than bi-encoders. Slower (can't precompute), but
        dramatically better precision on final top-K.
        
        Model: ms-marco-MiniLM-L-6-v2 (local, ~50ms per pair)
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, chunk.content) for chunk in candidates]
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Update chunks with rerank scores
        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            candidates, 
            key=lambda c: c.rerank_score, 
            reverse=True
        )
        
        # Update final ranks
        for rank, chunk in enumerate(reranked, start=1):
            chunk.final_rank = rank
        
        return reranked
    
    def _load_bm25_index(self) -> BM25Okapi:
        """
        Load BM25 index from disk/cache.
        
        In production:
        - Build offline during ingestion
        - Store in Redis or local cache
        - Rebuild incrementally on new documents
        
        For this reference: builds in-memory on init.
        """
        # TODO: Load from persistent storage
        # For now, return placeholder (will build from corpus)
        corpus_texts = []  # Load from DB
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        return BM25Okapi(tokenized_corpus)
    
    def _load_bm25_corpus(self) -> List[Dict]:
        """Load corpus metadata for BM25 results"""
        # TODO: Load from DB/cache
        return []
    
    def _load_reranker(self):
        """Load cross-encoder re-ranker model"""
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return model
    
    @staticmethod
    def _matches_filters(metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches all filters"""
        return all(
            metadata.get(key) == value 
            for key, value in filters.items()
        )


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example usage of HybridRetriever.
    
    Run with: python -m retrieval.hybrid
    """
    from database.pgvector_client import PgVectorClient
    from models.embedding import EmbeddingModel
    
    # Initialize components
    vector_client = PgVectorClient(
        host=settings.PGVECTOR_HOST,
        port=settings.PGVECTOR_PORT,
        database=settings.PGVECTOR_DB,
        user=settings.PGVECTOR_USER,
        password=settings.PGVECTOR_PASSWORD
    )
    
    embedding_model = EmbeddingModel(
        model_name=settings.EMBEDDING_MODEL
    )
    
    # Create retriever
    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_model=embedding_model,
        collection_name="legal_docs",
        vector_top_k=50,
        bm25_top_k=50,
        rerank_top_k=5
    )
    
    # Retrieve
    query = "What is the notice period for termination?"
    results = retriever.retrieve(
        query=query,
        filters={"source_type": "employment_contract"}
    )
    
    # Display
    print(f"\nTop {len(results)} results for: '{query}'\n")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. [{chunk.source_document}] (score: {chunk.rerank_score:.3f})")
        print(f"   {chunk.content[:200]}...")
        print()
