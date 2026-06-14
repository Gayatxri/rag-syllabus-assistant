import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from src.logger import get_logger
from src.exceptions import DocumentLoadError
from src.validator import validate_document_path
from config.settings import DOCUMENTS_PATH

logger = get_logger("document_loader")

def load_single_document(file_path: str) -> list:
    """Load a single PDF document"""
    try:
        validate_document_path(file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add source metadata to each page
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["full_path"] = file_path

        logger.info(
            f"Loaded '{os.path.basename(file_path)}' "
            f"— {len(documents)} pages"
        )
        return documents

    except Exception as e:
        raise DocumentLoadError(
            f"Failed to load {file_path}: {e}"
        )


def load_all_documents(folder_path: str = DOCUMENTS_PATH) -> list:
    """Load all PDF documents from a folder"""

    if not os.path.exists(folder_path):
        raise DocumentLoadError(
            f"Documents folder not found: {folder_path}"
        )

    # Find all PDFs in folder
    pdf_files = glob.glob(
        os.path.join(folder_path, "*.pdf")
    )

    if not pdf_files:
        raise DocumentLoadError(
            f"No PDF files found in: {folder_path}"
        )

    logger.info(
        f"Found {len(pdf_files)} PDF(s) in {folder_path}"
    )

    all_documents = []
    failed_files = []

    for pdf_path in pdf_files:
        try:
            docs = load_single_document(pdf_path)
            all_documents.extend(docs)
        except DocumentLoadError as e:
            logger.warning(f"Skipping file: {e}")
            failed_files.append(pdf_path)

    if failed_files:
        logger.warning(
            f"Failed to load {len(failed_files)} file(s): "
            f"{failed_files}"
        )

    logger.info(
        f"Total loaded: {len(all_documents)} pages "
        f"from {len(pdf_files) - len(failed_files)} documents"
    )

    return all_documents


def get_document_stats(documents: list) -> dict:
    """Get statistics about loaded documents"""

    sources = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources[source] = 0
        sources[source] += 1

    stats = {
        "total_pages": len(documents),
        "total_documents": len(sources),
        "documents": sources
    }

    logger.info(f"Document stats: {stats}")
    return stats