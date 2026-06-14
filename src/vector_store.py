import uuid
import chromadb
from chromadb.config import Settings
from src.logger import get_logger
from src.exceptions import VectorStoreError
from config.settings import CHROMA_DB_PATH, COLLECTION_NAME

logger = get_logger("vector_store")

def get_collection():
    """Connect to ChromaDB collection"""
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Connected to ChromaDB: "
            f"{collection.count()} chunks stored"
        )
        return collection
    except Exception as e:
        raise VectorStoreError(f"ChromaDB connection failed: {e}")

def store_chunks(collection, chunks: list,
                 embeddings) -> None:
    """Store chunks and embeddings in ChromaDB"""
    try:
        # Clear existing data
        existing = collection.count()
        if existing > 0:
            collection.delete(
                where={"source": {"$ne": ""}}
            )
            logger.info(f"Cleared {existing} existing chunks")

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0)
            }
            for chunk in chunks
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]

        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")

    except Exception as e:
        raise VectorStoreError(f"Failed to store chunks: {e}")

def search_chunks(collection, model,
                  question: str, top_k: int) -> tuple:
    """Search for relevant chunks"""
    try:
        query_vector = model.encode([question]).tolist()
        results = collection.query(
            query_embeddings=query_vector,
            n_results=top_k
        )
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        logger.info(
            f"Retrieved {len(chunks)} chunks "
            f"for query: '{question[:50]}'"
        )
        return chunks, metadatas
    except Exception as e:
        raise VectorStoreError(f"Search failed: {e}")