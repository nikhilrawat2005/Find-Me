import os
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from tqdm import tqdm

from src.config import STOCK_DIR, PROCESSED_PHOTOS_DIR, FAISS_INDEX_PATH
from src.database import (
    init_db,
    get_photo_by_original_name,
    get_next_photo_number,
    insert_photo,
    insert_face,
    get_stats
)
from src.face_engine import FaceEngine

# 512-dimensional vector for InsightFace ArcFace models
EMBEDDING_DIM = 512

class StockIndexer:
    def __init__(self):
        init_db()
        self.face_engine = FaceEngine.get_instance()
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        if FAISS_INDEX_PATH.exists():
            try:
                index = faiss.read_index(str(FAISS_INDEX_PATH))
                return index
            except Exception as e:
                print(f"[Warning] Failed to read existing FAISS index, creating new: {e}")
        # IndexFlatIP calculates inner product, which equals cosine similarity for normalized vectors
        return faiss.IndexFlatIP(EMBEDDING_DIM)

    def save_index(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def index_stock_photos(self) -> dict:
        """
        Processes photos from STOCK_DIR without altering any file in STOCK_DIR.
        1. Numbering: Copies each unindexed image to PROCESSED_PHOTOS_DIR with clean sequential numbering (e.g. photo_0001.jpg).
        2. Embedding: Extracts face embeddings using InsightFace and adds them into FAISS index & SQLite.
        """
        if not STOCK_DIR.exists():
            raise FileNotFoundError(f"Stock directory not found: {STOCK_DIR}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        stock_files = sorted([
            p for p in STOCK_DIR.iterdir() 
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

        print(f"Found {len(stock_files)} photos in stock directory.")

        processed_count = 0
        skipped_count = 0
        new_faces_count = 0

        for file_path in tqdm(stock_files, desc="Indexing Stock Photos"):
            orig_name = file_path.name
            
            # Check if photo was already processed
            existing = get_photo_by_original_name(orig_name)
            if existing:
                skipped_count += 1
                continue

            # Task 1: Numbering
            next_num = get_next_photo_number()
            numbered_filename = f"photo_{next_num:04d}{file_path.suffix.lower()}"
            dest_path = PROCESSED_PHOTOS_DIR / numbered_filename

            # Safe copy (Original remains 100% untouched)
            shutil.copy2(file_path, dest_path)

            # Task 2: Embedding conversion
            faces = self.face_engine.extract_faces_from_file(str(dest_path))
            
            # Record photo in database
            photo_id = insert_photo(
                photo_number=next_num,
                numbered_filename=numbered_filename,
                original_filename=orig_name,
                original_path=str(file_path),
                stored_path=str(dest_path),
                face_count=len(faces)
            )

            # Add each detected face embedding to FAISS and DB
            for face in faces:
                emb = face["embedding"]
                vector_idx = self.index.ntotal  # Current position in FAISS
                
                # Add to FAISS index
                emb_matrix = np.expand_dims(emb, axis=0).astype(np.float32)
                self.index.add(emb_matrix)

                # Record in database
                insert_face(
                    photo_id=photo_id,
                    vector_index=vector_idx,
                    bbox=face["bbox"],
                    det_score=face["det_score"]
                )
                new_faces_count += 1

            processed_count += 1

        # Save FAISS index
        self.save_index()

        stats = get_stats()
        result = {
            "newly_processed_photos": processed_count,
            "skipped_existing_photos": skipped_count,
            "newly_detected_faces": new_faces_count,
            "total_photos": stats["total_photos"],
            "total_faces": stats["total_faces"],
            "faiss_total_vectors": self.index.ntotal
        }
        print(f"Indexing completed: {result}")
        return result

if __name__ == "__main__":
    indexer = StockIndexer()
    indexer.index_stock_photos()
