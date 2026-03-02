"""Simple test without saving"""

from pathlib import Path
from models.embedder import EmbeddingModel
from database.vector_store import VectorStore
from ingestion.loaders.pdf_loader import PDFLoader
from dotenv import load_dotenv

load_dotenv()

def chunk_text(text: str, chunk_size: int = 1000) -> list:
    words = text.split()
    chunks = []
    words_per_chunk = chunk_size // 5
    
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
    return chunks

print("="*70)
print("RAG PIPELINE TEST")
print("="*70)

# Load PDF
print("\n[1/4] Loading PDF...")
loader = PDFLoader()
docs = loader.load_directory("data/sample_docs")

if not docs:
    print("No PDFs found!")
    exit()

# Chunk
print("\n[2/4] Chunking...")
all_chunks = []
metadata = []

for doc in docs:
    chunks = chunk_text(doc['text'])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata.append({
            'source': doc['metadata']['filename'],
            'chunk': f"{i+1}/{len(chunks)}"
        })

print(f"Created {len(all_chunks)} chunks")

# Embed
print("\n[3/4] Generating embeddings...")
embedder = EmbeddingModel()
embeddings = embedder.embed_batch(all_chunks)
print(f"Generated {embeddings.shape[0]} embeddings")

# Store
print("\n[4/4] Storing in vector database...")
store = VectorStore(dimension=embedder.dimension)
store.add(embeddings, metadata)
print(f"Stored {store.size} vectors")

# Search test
print("\n" + "="*70)
print("TESTING SEARCH")
print("="*70)

query = "What is this document about?"
print(f"\nQuery: {query}")

query_emb = embedder.embed(query)
results = store.search(query_emb, top_k=3)

for i, r in enumerate(results, 1):
    print(f"\n{i}. Source: {r['metadata']['source']}")
    print(f"   Chunk: {r['metadata']['chunk']}")
    print(f"   Score: {r['score']:.4f}")

print("\n" + "="*70)
print("SUCCESS! Your RAG pipeline is working!")
print("="*70)