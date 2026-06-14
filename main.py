import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from src.document_loader import load_all_documents, get_document_stats
from src.chunker import chunk_documents
from src.embedder import load_embedding_model, embed_chunks
from src.vector_store import get_collection, store_chunks, search_chunks
from src.generator import get_llm_client, build_prompt, generate_answer
from src.validator import validate_question
from src.error_handler import handle_rag_error
from src.exceptions import RAGException, ValidationError
from config.settings import TOP_K_RESULTS

logger = get_logger("main")

def setup_pipeline():
    """Load and index all documents"""
    logger.info("="*50)
    logger.info("Setting up RAG pipeline...")
    logger.info("="*50)

    # Load documents
    documents = load_all_documents()
    stats = get_document_stats(documents)
    logger.info(f"Loaded {stats['total_documents']} documents, "
                f"{stats['total_pages']} pages")

    # Chunk documents
    chunks = chunk_documents(documents)

    # Load embedding model
    model = load_embedding_model()

    # Embed chunks
    embeddings = embed_chunks(model, chunks)

    # Store in ChromaDB
    collection = get_collection()
    store_chunks(collection, chunks, embeddings)

    logger.info("Pipeline setup complete!")
    return model, collection

def ask(question: str, model, collection) -> str:
    """Run complete RAG pipeline for a question"""
    try:
        # Validate question
        question = validate_question(question)

        # Retrieve relevant chunks
        chunks, metadatas = search_chunks(
            collection, model, question, TOP_K_RESULTS
        )

        # Build prompt
        llm_client = get_llm_client()
        prompt = build_prompt(question, chunks, metadatas)

        # Generate answer
        answer = generate_answer(llm_client, prompt)

        return answer

    except ValidationError as e:
        return handle_rag_error(e, "ask")
    except RAGException as e:
        return handle_rag_error(e, "ask")

def main():
    # Setup pipeline
    model, collection = setup_pipeline()

    # Test questions
    questions = [
        "What are the types of disasters mentioned?",
        "What are the LTPC credits for MX3084?",
    ]

    print("\n" + "="*55)
    print("🤖 RAG System Ready! Testing questions...")
    print("="*55)

    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = ask(question, model, collection)
        print(f"\n✅ Answer: {answer}")
        print("-"*55)

if __name__ == "__main__":
    main()