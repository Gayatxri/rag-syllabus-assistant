import os
from src.logger import get_logger
from src.exceptions import ValidationError

logger = get_logger("validator")

def validate_question(question: str) -> str:
    """Validate and clean user question"""
    
    # Check if question exists
    if not question:
        raise ValidationError("Question cannot be empty")
    
    # Clean whitespace
    question = question.strip()
    
    # Check minimum length
    if len(question) < 3:
        raise ValidationError(
            f"Question too short: '{question}'. "
            f"Please ask a proper question."
        )
    
    # Check maximum length
    if len(question) > 500:
        raise ValidationError(
            f"Question too long ({len(question)} chars). "
            f"Please keep it under 500 characters."
        )
    
    logger.info(f"Question validated: '{question[:50]}...' " 
                if len(question) > 50 else 
                f"Question validated: '{question}'")
    return question


def validate_document_path(path: str) -> str:
    """Validate document file path"""
    
    if not path:
        raise ValidationError("Document path cannot be empty")
    
    if not os.path.exists(path):
        raise ValidationError(f"Document not found: {path}")
    
    if not path.lower().endswith('.pdf'):
        raise ValidationError(
            f"Invalid file type. Only PDF files supported. "
            f"Got: {path}"
        )
    
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValidationError(f"Document is empty: {path}")
    
    logger.info(f"Document validated: {path} "
                f"({file_size/1024:.1f} KB)")
    return path


def validate_chunks(chunks: list) -> list:
    """Validate retrieved chunks"""
    
    if not chunks:
        raise ValidationError(
            "No chunks retrieved. "
            "Try rephrasing your question."
        )
    
    logger.info(f"Chunks validated: {len(chunks)} chunks ready")
    return chunks