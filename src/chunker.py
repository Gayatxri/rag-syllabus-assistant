from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import get_logger
from src.exceptions import RAGException
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger("chunker")

def chunk_documents(documents: list) -> list:
    """Split documents into chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(
            f"Created {len(chunks)} chunks "
            f"from {len(documents)} pages"
        )
        return chunks
    except Exception as e:
        raise RAGException(f"Chunking failed: {e}")