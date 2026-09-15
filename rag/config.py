"""
Configuration settings for the GWRHelp RAG support pipeline.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
INDEX_DIR = BASE_DIR / "index"

# Data Files
DATA_RETRIEVAL_PATH = DATA_DIR / "data_retrieval.csv"
GOLDEN_REVIEW_PATH = DATA_DIR / "gwr_golden_review.csv"
GWR_THREADS_PATH = DATA_DIR / "gwr_threads.csv"

# FAISS Artifacts
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "retrieval_metadata.json"

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_CHUNK_WORDS = 350
MAX_CHUNK_TURNS = 6
CHUNK_WINDOW_TURNS = 4
CHUNK_STEP_TURNS = 2

# Allowed Intent Taxonomy
ALLOWED_INTENTS = [
    "BOOKING / TICKETS",
    "TRAIN DELAY / CANCELLATION",
    "TRAIN / SERVICE INFORMATION",
    "REFUND / COMPENSATION",
    "PAYMENT / CHARGES",
    "BAGGAGE / LOST PROPERTY",
    "ACCESSIBILITY / SPECIAL ASSISTANCE",
    "ROUTE / TIMETABLE / STATION",
    "STAFF / SERVICE COMPLAINT",
    "TECHNICAL / WEBSITE / APP",
    "GENERAL CUSTOMER SERVICE",
    "OTHER",
]

# Allowed Decision Options
ALLOWED_DECISIONS = [
    "RESPOND",
    "ASK_CLARIFICATION",
    "REQUEST_INFORMATION",
    "ESCALATE",
]

DEFAULT_TOP_K = 5
