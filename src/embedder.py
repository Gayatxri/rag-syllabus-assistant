import os
from sentence_transformers import SentenceTransformer
from src.logger import get_logger
from src.exceptions import EmbeddingError
from config.settings import EMBEDDING_MODEL

logger = get_logger("embedder")

def load_embedding_model() -> SentenceTransformer:
    """Load embedding model offline"""
    try:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            local_files_only=True
        )
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
        return model
    except Exception as e:
        raise EmbeddingError(f"Failed to load model: {e}")

def embed_chunks(model: SentenceTransformer,
                 chunks: list) -> list:
    """Convert chunks to embeddings"""
    try:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = model.encode(
            texts,
            show_progress_bar=True
        )
        logger.info(
            f"Generated {len(embeddings)} embeddings "
            f"of dimension {embeddings.shape[1]}"
        )
        return embeddings
    except Exception as e:
        raise EmbeddingError(f"Embedding failed: {e}")