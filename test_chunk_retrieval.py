"""
Quick test script for chunk retrieval pipeline.
Tests with a single document to validate the implementation.
"""

import sys
sys.path.insert(0, 'c:\\Users\\kilah\\ceng543_project')

from src.utils.chunker import chunk_text
from src.utils.window import build_tokenizer
from src.models.retriever import load_embedding_model, embed_chunks, build_faiss_index, retrieve_top_k, create_query_from_document
from src.models.generator import concatenate_chunks

# Test document (short example)
test_doc = """
Deep learning has revolutionized artificial intelligence in recent years. 
Neural networks with multiple layers can learn complex patterns from data.
Convolutional neural networks excel at image recognition tasks.
Recurrent neural networks are effective for sequential data processing.
Transformers have become the dominant architecture for natural language processing.
Attention mechanisms allow models to focus on relevant parts of the input.
Large language models like GPT demonstrate impressive capabilities.
However, these models require substantial computational resources.
Research continues on making AI more efficient and accessible.
"""

print("🧪 Testing Chunk Retrieval Pipeline\n")

# Step 1: Test chunker
print("=" * 60)
print("STEP 1: Testing Chunker")
print("=" * 60)

tokenizer = build_tokenizer("allenai/led-base-16384")

for method in ["fixed", "sentence", "semantic"]:
    print(f"\n📌 Method: {method}")
    chunks = chunk_text(test_doc, tokenizer, chunk_size=50, overlap=10, method=method)
    print(f"  ✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"  Chunk {i+1} ({len(chunk['ids'])} tokens, pos {chunk['start_token']}-{chunk['end_token']}): {chunk['text'][:80]}...")

# Step 2: Test retriever
print("\n" + "=" * 60)
print("STEP 2: Testing Retriever")
print("=" * 60)

chunks = chunk_text(test_doc, tokenizer, chunk_size=50, overlap=10, method="sentence")
print(f"\n📌 Loading embedding model...")
embedding_model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
print(f"  ✓ Model loaded on {embedding_model.device}")

print(f"\n📌 Embedding {len(chunks)} chunks...")

embeddings = embed_chunks(chunks, embedding_model)
print(f"  ✓ Embeddings shape: {embeddings.shape}")

print(f"\n📌 Building FAISS index...")
index = build_faiss_index(embeddings)
print(f"  ✓ Index created with {index.ntotal} vectors")

print(f"\n📌 Retrieving top-3 chunks...")
query = create_query_from_document(test_doc, tokenizer, strategy="first_tokens", num_tokens=20)
print(f"  Query: {query[:100]}...")

top_k_indices, top_k_scores = retrieve_top_k(
    query, chunks, index, K=3, embedding_model=embedding_model
)
print(f"  ✓ Retrieved indices: {top_k_indices}")
print(f"  ✓ Scores: {[f'{s:.3f}' for s in top_k_scores]}")

for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
    print(f"\n  Rank {i+1} (score={score:.3f}): {chunks[idx]['text'][:100]}...")

# Step 3: Test concatenation
print("\n" + "=" * 60)
print("STEP 3: Testing Concatenation")
print("=" * 60)

context = concatenate_chunks(chunks, top_k_indices, tokenizer, max_tokens=200, preserve_order=True)
context_tokens = len(tokenizer(context, add_special_tokens=False)["input_ids"])
print(f"\n📌 Concatenated context ({context_tokens} tokens):")
print(f"  {context[:200]}...")

# Step 4: Sanity checks
print("\n" + "=" * 60)
print("STEP 4: Sanity Checks")
print("=" * 60)

# Test all methods
for method in ["fixed", "sentence", "semantic"]:
    chunks = chunk_text(test_doc, tokenizer, chunk_size=100, overlap=20, method=method)
    
    # Check 1: All chunks have required keys
    assert all("text" in ch and "ids" in ch and "start_token" in ch and "end_token" in ch for ch in chunks), \
        f"{method}: Missing required keys in chunk dicts"
    
    # Check 2: All chunks are non-empty
    assert all(len(ch["ids"]) > 0 for ch in chunks), f"{method}: Found empty chunks"
    
    # Check 3: Token IDs match text
    for ch in chunks[:3]:  # Check first 3
        reconstructed = tokenizer(ch["text"], add_special_tokens=False)["input_ids"]
        # Allow slight mismatch due to decode/encode round-trip
        assert abs(len(reconstructed) - len(ch["ids"])) <= 5, \
            f"{method}: Token count mismatch (expected ~{len(ch['ids'])}, got {len(reconstructed)})"
    
    print(f"  ✓ {method}: All checks passed ({len(chunks)} chunks)")

# Test overlap ratios
print("\n📌 Testing overlap ratios...")
for overlap_ratio in [0.0, 0.25, 0.5]:
    chunks = chunk_text(test_doc, tokenizer, chunk_size=100, overlap=overlap_ratio, method="fixed")
    print(f"  ✓ Overlap ratio {overlap_ratio}: {len(chunks)} chunks created")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pip install sentence-transformers faiss-gpu nltk")
print("2. Run full test: python main_chunk_retrieval.py --config configs/chunk_retrieval/arxiv_extra_long_c1024_ov0_k3_fixed.yml")
