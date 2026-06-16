import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from src.document_loader import load_all_documents
from src.chunker import chunk_documents
from src.embedder import load_embedding_model, embed_chunks
from src.vector_store import get_collection, store_chunks
from src.hybrid_search import HybridSearcher
from src.query_rewriter import rewrite_if_needed
from src.reranker import rerank_chunks
from src.generator import get_llm_client, build_prompt, generate_answer
from src.validator import validate_question

logger = get_logger("main")

def setup():
    """Setup full pipeline"""
    logger.info("="*50)
    logger.info("Phase 3 Complete Pipeline")
    logger.info("="*50)

    documents = load_all_documents()
    chunks = chunk_documents(documents)
    model = load_embedding_model()
    embeddings = embed_chunks(model, chunks)
    collection = get_collection()
    store_chunks(collection, chunks, embeddings)

    chunk_texts = [c.page_content for c in chunks]
    chunk_metas = [
        {
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page", 0)
        }
        for c in chunks
    ]
    searcher = HybridSearcher(chunk_texts, chunk_metas)
    return model, collection, searcher

def ask(question: str, model, collection,
        searcher) -> str:
    """Full Phase 3 RAG pipeline"""

    print(f"\n{'='*55}")
    print(f"❓ Question: {question}")

    # Step 1 — Validate
    question = validate_question(question)

    # Step 2 — Rewrite if needed
    rewritten = rewrite_if_needed(question)
    if rewritten != question:
        print(f"🔄 Rewritten: {rewritten[:60]}...")

    # Step 3 — Hybrid search (get top 6)
    chunks, metas = searcher.hybrid_search(
        model, collection, rewritten, top_k=6
    )
    print(f"🔍 Retrieved {len(chunks)} chunks via hybrid search")

    # Step 4 — Rerank (pick best 3)
    reranked_chunks, reranked_metas, scores = rerank_chunks(
        question, chunks, metas
    )
    top_chunks = reranked_chunks[:3]
    top_metas = reranked_metas[:3]
    print(f"⚡ Reranked — top scores: "
          f"{[round(s,2) for s in scores[:3]]}")

    # Step 5 — Generate answer
    llm = get_llm_client()
    prompt = build_prompt(question, top_chunks, top_metas)
    answer = generate_answer(llm, prompt)

    print(f"\n✅ Answer: {answer}")
    return answer

def main():
    model, collection, searcher = setup()

    questions = [
        "What are the types of disasters?",
        "What is the Sendai Framework?",
    ]

    for question in questions:
        ask(question, model, collection, searcher)

if __name__ == "__main__":
    main()