import os
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from tqdm import tqdm

from src.config import STOCK_DIR, PROCESSED_PHOTOS_DIR, FAISS_INDEX_PATH, GDRIVE_DIR
from src.database import (
    init_db,
    get_photo_by_original_name,
    get_photo_by_gdrive_id,
    get_next_photo_number,
    insert_photo,
    insert_face,
    get_stats
)
from src.face_engine import FaceEngine
from src.gdrive_service import GDriveService

# 512-dimensional vector for InsightFace ArcFace models
EMBEDDING_DIM = 512

class StockIndexer:
    def __init__(self):
        init_db()
        self.face_engine = FaceEngine.get_instance()
        self.index = self._load_or_create_index()
        self.gdrive_service = GDriveService()

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
        Processes photos from local STOCK_DIR without altering any file in STOCK_DIR.
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

        for file_path in tqdm(stock_files, desc="Indexing Local Stock Photos"):
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
                face_count=len(faces),
                source_type="local"
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
            "source": "local",
            "newly_processed_photos": processed_count,
            "skipped_existing_photos": skipped_count,
            "newly_detected_faces": new_faces_count,
            "total_photos": stats["total_photos"],
            "total_faces": stats["total_faces"],
            "local_photos": stats.get("local_photos", 0),
            "gdrive_photos": stats.get("gdrive_photos", 0),
            "faiss_total_vectors": self.index.ntotal
        }
        print(f"Local indexing completed: {result}")
        return result

    def index_gdrive_photos(self, folder_id: str, clean_duplicates_first: bool = True) -> dict:
        """
        Fetches photos directly from a Google Drive folder.
        - Checks for and removes duplicate files on Google Drive (if enabled).
        - Direct in-memory embedding extraction: Images are NOT downloaded to disk;
          they are streamed into RAM, face embeddings are extracted into FAISS & DB.
        """
        duplicates_report = None
        if clean_duplicates_first:
            try:
                duplicates_report = self.gdrive_service.find_and_clean_duplicates(folder_id, auto_delete=True)
                print(f"Duplicate cleanup: Found {duplicates_report['duplicates_found']}, deleted {duplicates_report['duplicates_deleted']}")
            except Exception as e:
                print(f"[Warning] Duplicate check skipped/failed: {e}")

        folder_files = self.gdrive_service.list_folder_photos(folder_id)
        print(f"Found {len(folder_files)} photo(s) in Google Drive folder to index.")

        processed_count = 0
        skipped_count = 0
        new_faces_count = 0

        import cv2

        for file_meta in tqdm(folder_files, desc="Indexing Drive Photos (In-Memory)"):
            gdrive_id = file_meta["id"]
            orig_name = file_meta.get("name", f"{gdrive_id}.jpg")
            
            # Check if this Google Drive file was already indexed
            existing = get_photo_by_gdrive_id(gdrive_id)
            if existing:
                skipped_count += 1
                continue

            # In-memory stream: no permanent disk file created
            try:
                raw_bytes = self.gdrive_service.get_photo_bytes(gdrive_id)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"[Warning] Could not decode image: {orig_name}")
                    continue
            except Exception as dl_err:
                print(f"[Warning] Failed to fetch image {orig_name}: {dl_err}")
                continue

            # Sequential numbering in metadata
            next_num = get_next_photo_number()
            ext = Path(orig_name).suffix.lower()
            if not ext or ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            numbered_filename = f"photo_{next_num:04d}{ext}"

            # Extract face embeddings directly from memory array
            faces = self.face_engine.extract_faces_from_image(img_bgr)

            # Record photo in database with source_type='gdrive', gdrive_file_id, and web link
            gdrive_link = file_meta.get("webViewLink") or f"https://drive.google.com/file/d/{gdrive_id}/view"
            photo_id = insert_photo(
                photo_number=next_num,
                numbered_filename=numbered_filename,
                original_filename=orig_name,
                original_path=f"gdrive://{gdrive_id}/{orig_name}",
                stored_path=gdrive_link,
                face_count=len(faces),
                source_type="gdrive",
                gdrive_file_id=gdrive_id
            )

            # Add each detected face embedding to FAISS and DB
            for face in faces:
                emb = face["embedding"]
                vector_idx = self.index.ntotal

                emb_matrix = np.expand_dims(emb, axis=0).astype(np.float32)
                self.index.add(emb_matrix)

                insert_face(
                    photo_id=photo_id,
                    vector_index=vector_idx,
                    bbox=face["bbox"],
                    det_score=face["det_score"]
                )
                new_faces_count += 1

            processed_count += 1

        self.save_index()


        stats = get_stats()
        result = {
            "source": "gdrive",
            "newly_processed_photos": processed_count,
            "skipped_existing_photos": skipped_count,
            "newly_detected_faces": new_faces_count,
            "total_photos": stats["total_photos"],
            "total_faces": stats["total_faces"],
            "local_photos": stats.get("local_photos", 0),
            "gdrive_photos": stats.get("gdrive_photos", 0),
            "faiss_total_vectors": self.index.ntotal
        }
        print(f"Google Drive indexing completed: {result}")
        return result

if __name__ == "__main__":
    indexer = StockIndexer()
    indexer.index_stock_photos()

