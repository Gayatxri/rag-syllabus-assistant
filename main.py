import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from src.document_loader import load_all_documents
from src.chunker import chunk_documents
from src.embedder import load_embedding_model, embed_chunks
from src.vector_store import (
    get_collection, store_chunks, search_chunks
)
from src.hybrid_search import HybridSearcher
from src.query_rewriter import rewrite_if_needed

logger = get_logger("main")

def main():
    logger.info("="*50)
    logger.info("Day 12 — Hybrid Search Test")
    logger.info("="*50)

    # Setup
    documents = load_all_documents()
    chunks = chunk_documents(documents)
    model = load_embedding_model()
    embeddings = embed_chunks(model, chunks)
    collection = get_collection()
    store_chunks(collection, chunks, embeddings)

    # Build hybrid searcher
    chunk_texts = [c.page_content for c in chunks]
    chunk_metas = [
        {
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page", 0)
        }
        for c in chunks
    ]
    searcher = HybridSearcher(chunk_texts, chunk_metas)

    # Test queries
    test_queries = [
        "MX3084 LTPC credits",
        "Sendai Framework disaster risk reduction",
        "types of natural disasters",
    ]

    print("\n" + "="*55)
    print("🔍 Hybrid Search Test")
    print("="*55)

    for query in test_queries:
        print(f"\n❓ Query: '{query}'")

        # Rewrite if needed
        rewritten = rewrite_if_needed(query)
        if rewritten != query:
            print(f"🔄 Rewritten: '{rewritten[:60]}...'")

        # Semantic only
        sem_chunks, sem_metas = search_chunks(
            collection, model, rewritten, 3
        )
        print(f"\n📊 Semantic search top result:")
        print(f"   Source: {sem_metas[0]['source']}, "
              f"Page: {sem_metas[0]['page']}")
        print(f"   Preview: {sem_chunks[0][:100]}...")

        # Hybrid search
        hyb_chunks, hyb_metas = searcher.hybrid_search(
            model, collection, rewritten
        )
        print(f"\n🔀 Hybrid search top result:")
        print(f"   Source: {hyb_metas[0]['source']}, "
              f"Page: {hyb_metas[0]['page']}")
        print(f"   Preview: {hyb_chunks[0][:100]}...")
        print("-"*55)

if __name__ == "__main__":
    main()