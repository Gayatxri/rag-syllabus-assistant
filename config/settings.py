import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Settings ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024

# ─── Embedding Settings ────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── Chunking Settings ─────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ─── Retrieval Settings ────────────────────────────────────
TOP_K_RESULTS = 3

# ─── Storage Settings ──────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "syllabus"
DOCUMENTS_PATH = "./data/documents"

# ─── Logging Settings ──────────────────────────────────────
LOG_FILE = "./logs/rag.log"
LOG_LEVEL = "INFO"