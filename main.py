import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.logger import get_logger
from src.validator import validate_question, validate_document_path
from src.exceptions import ValidationError
from src.error_handler import handle_rag_error
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    LLM_MODEL, EMBEDDING_MODEL, DOCUMENTS_PATH
)

logger = get_logger("main")

def test_validation():
    logger.info("Testing input validation...")
    test_cases = [
        ("", "empty question"),
        ("Hi", "too short"),
        ("What are the types of disasters?", "valid question"),
        ("A" * 600, "too long"),
    ]
    for question, description in test_cases:
        try:
            validated = validate_question(question)
            logger.info(f"PASSED — {description}: '{validated[:30]}'")
        except ValidationError as e:
            logger.warning(f"CAUGHT — {description}: {e}")

def test_document_validation():
    logger.info("Testing document validation...")
    paths = [
        r"C:\Users\HP\my_rag_project\data\documents\syllabus.pdf",
        "data/fake_document.pdf",
        "data/documents/wrong.docx",
    ]
    for path in paths:
        try:
            validated = validate_document_path(path)
            logger.info(f"PASSED — Document valid: {path}")
        except ValidationError as e:
            logger.warning(f"CAUGHT — {e}")

def main():
    logger.info("="*50)
    logger.info("Day 8 — Error Handling Tests")
    logger.info("="*50)
    test_validation()
    test_document_validation()
    logger.info("="*50)
    logger.info("All error handling tests complete!")
    print("\n✅ Error handling working perfectly!")

if __name__ == "__main__":
    main()