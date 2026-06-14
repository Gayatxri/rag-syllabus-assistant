import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from src.document_loader import load_all_documents, get_document_stats
from src.exceptions import DocumentLoadError

logger = get_logger("main")

def main():
    logger.info("="*50)
    logger.info("Day 9 — Multiple Document Support")
    logger.info("="*50)

    try:
        # Load all documents from folder
        logger.info("Loading all documents...")
        documents = load_all_documents()

        # Get statistics
        stats = get_document_stats(documents)

        print("\n" + "="*50)
        print("📚 Documents loaded successfully!")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total pages: {stats['total_pages']}")
        print("\nDocument breakdown:")
        for doc_name, pages in stats['documents'].items():
            print(f"  📄 {doc_name}: {pages} pages")
        print("="*50)

        # Show sample metadata
        print("\nSample chunk metadata:")
        print(f"  Source: {documents[0].metadata['source']}")
        print(f"  Page: {documents[0].metadata['page']}")
        print(f"  Content preview: {documents[0].page_content[:100]}...")

    except DocumentLoadError as e:
        logger.error(f"Document loading failed: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()