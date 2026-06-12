import logging
import os
from config.settings import LOG_FILE, LOG_LEVEL

os.makedirs("logs", exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # ─── File handler (saves to logs/rag.log) ─────────────
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    
    # ─── Console handler (shows in terminal) ──────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # ─── Format ───────────────────────────────────────────
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger