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
from src.evaluator import evaluate_single
from src.golden_dataset import GOLDEN_DATASET

logger = get_logger("main")

def setup():
    """Setup full pipeline"""
    logger.info("Setting up RAG pipeline...")
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

def ask_and_evaluate(question: str, model,
                     collection, searcher) -> tuple:
    """Run RAG pipeline and return answer + context"""

    question = validate_question(question)
    rewritten = rewrite_if_needed(question)

    chunks, metas = searcher.hybrid_search(
        model, collection, rewritten, top_k=6
    )
    reranked_chunks, reranked_metas, scores = rerank_chunks(
        question, chunks, metas
    )

    top_chunks = reranked_chunks[:3]
    top_metas = reranked_metas[:3]

    llm = get_llm_client()
    prompt = build_prompt(question, top_chunks, top_metas)
    answer = generate_answer(llm, prompt)
    context = "\n".join(top_chunks)

    return answer, context

def run_evaluation(model, collection, searcher):
    """Run evaluation on golden dataset"""

    logger.info("="*50)
    logger.info("Phase 4 — RAGAS Evaluation")
    logger.info("="*50)

    all_scores = []

    print("\n" + "="*60)
    print("📊 RAGAS EVALUATION REPORT")
    print("="*60)

    for i, item in enumerate(GOLDEN_DATASET):
        question = item["question"]
        print(f"\n[{i+1}/{len(GOLDEN_DATASET)}] {question}")

        # Get RAG answer
        answer, context = ask_and_evaluate(
            question, model, collection, searcher
        )
        print(f"Answer: {answer[:150]}...")

        # Evaluate
        scores = evaluate_single(question, answer, context)
        all_scores.append(scores)

        print(f"Faithfulness:      {scores['faithfulness']:.2f}")
        print(f"Answer Relevance:  {scores['answer_relevance']:.2f}")
        print(f"Context Precision: {scores['context_precision']:.2f}")
        print(f"Overall:           {scores['overall']:.2f}")
        print("-"*60)

    # Final report
    avg_faithfulness = sum(
        s["faithfulness"] for s in all_scores
    ) / len(all_scores)
    avg_relevance = sum(
        s["answer_relevance"] for s in all_scores
    ) / len(all_scores)
    avg_precision = sum(
        s["context_precision"] for s in all_scores
    ) / len(all_scores)
    avg_overall = sum(
        s["overall"] for s in all_scores
    ) / len(all_scores)

    print("\n" + "="*60)
    print("📈 FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Questions evaluated:  {len(GOLDEN_DATASET)}")
    print(f"Avg Faithfulness:     {avg_faithfulness:.2f}")
    print(f"Avg Answer Relevance: {avg_relevance:.2f}")
    print(f"Avg Context Precision:{avg_precision:.2f}")
    print(f"OVERALL RAG SCORE:    {avg_overall:.2f} / 1.0")
    print("="*60)

    logger.info(f"Evaluation complete. Overall score: {avg_overall:.2f}")
    return avg_overall

def main():
    model, collection, searcher = setup()
    run_evaluation(model, collection, searcher)

if __name__ == "__main__":
    main()