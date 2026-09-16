import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Input directory (Strictly READ-ONLY - stock photos are preserved untouched)
STOCK_DIR = BASE_DIR / "stock"

# Output data directories
DATA_DIR = BASE_DIR / "data"
PROCESSED_PHOTOS_DIR = DATA_DIR / "processed_photos"
DB_PATH = DATA_DIR / "metadata.sqlite"
FAISS_INDEX_PATH = DATA_DIR / "faces.index"

# Model settings
# buffalo_s / buffalo_sc are lightweight, fast, accurate models specified in face_recognition_plan.pdf
INSIGHTFACE_MODEL_NAME = "buffalo_sc"
DETECTION_SIZE = (640, 640)
DETECTION_THRESHOLD = 0.55
SIMILARITY_THRESHOLD = 0.48  # Cosine similarity cutoff for query matching

# Google Drive Configuration
GDRIVE_DIR = DATA_DIR / "gdrive_downloads"
GDRIVE_CREDENTIALS_PATH = BASE_DIR / "credentials.json"
GDRIVE_TOKEN_PATH = BASE_DIR / "token.json"
DEFAULT_GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "")

# Ensure necessary directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
GDRIVE_DIR.mkdir(parents=True, exist_ok=True)

