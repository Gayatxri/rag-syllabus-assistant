import time
from src.logger import get_logger
from src.exceptions import (
    RAGException, LLMError, 
    VectorStoreError, ValidationError
)

logger = get_logger("error_handler")

def handle_rag_error(error: Exception, context: str = "") -> str:
    """Central error handler — returns user-friendly message"""
    
    if isinstance(error, ValidationError):
        logger.warning(f"Validation error in {context}: {error}")
        return f"⚠️ Input issue: {error}"
    
    elif isinstance(error, VectorStoreError):
        logger.error(f"Vector store error in {context}: {error}")
        return "❌ Database error. Please try again in a moment."
    
    elif isinstance(error, LLMError):
        logger.error(f"LLM error in {context}: {error}")
        return "❌ AI service unavailable. Please try again shortly."
    
    elif isinstance(error, RAGException):
        logger.error(f"RAG error in {context}: {error}")
        return f"❌ System error: {error}"
    
    else:
        logger.critical(
            f"Unexpected error in {context}: "
            f"{type(error).__name__}: {error}"
        )
        return "❌ Unexpected error occurred. Check logs for details."


def retry_on_failure(func, max_retries: int = 3, 
                     delay: float = 2.0):
    """Retry a function up to max_retries times"""
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            result = func()
            if attempt > 1:
                logger.info(f"Succeeded on attempt {attempt}")
            return result
            
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries} attempts failed. "
                    f"Last error: {e}"
                )
    
    raise last_error