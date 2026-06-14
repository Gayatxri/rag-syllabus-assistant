class RAGException(Exception):
    """Base exception for RAG system"""
    pass

class DocumentLoadError(RAGException):
    """Raised when document cannot be loaded"""
    pass

class EmbeddingError(RAGException):
    """Raised when embedding generation fails"""
    pass

class VectorStoreError(RAGException):
    """Raised when ChromaDB operations fail"""
    pass

class RetrievalError(RAGException):
    """Raised when retrieval fails"""
    pass

class LLMError(RAGException):
    """Raised when LLM generation fails"""
    pass

class ValidationError(RAGException):
    """Raised when input validation fails"""
    pass