import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    LLM_MODEL, EMBEDDING_MODEL, DOCUMENTS_PATH
)

logger = get_logger("main")

def main():
    logger.info("RAG system starting up...")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"LLM model: {LLM_MODEL}")
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"Top K results: {TOP_K_RESULTS}")
    logger.info(f"Documents path: {DOCUMENTS_PATH}")
    logger.info("Configuration loaded successfully!")
    print("\n✅ Config and logging working perfectly!")

if __name__ == "__main__":
    main()