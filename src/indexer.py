import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import faiss
from tqdm import tqdm

from src.config import STOCK_DIR, PROCESSED_PHOTOS_DIR, FAISS_INDEX_PATH, GDRIVE_DIR
from src.database import (
    init_db,
    get_photo_by_original_name,
    get_photo_by_gdrive_id,
    get_next_photo_number,
    get_next_event_photo_number,
    create_or_get_event,
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
        Processes photos from local STOCK_DIR into the default 'Local Stock Showcase' event.
        """
        if not STOCK_DIR.exists():
            raise FileNotFoundError(f"Stock directory not found: {STOCK_DIR}")

        default_event = create_or_get_event(name="Local Stock Showcase", folder_id="local_stock", source_type="local")
        event_id = default_event["id"]

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        stock_files = sorted([
            p for p in STOCK_DIR.iterdir() 
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

        print(f"Found {len(stock_files)} photos in local stock directory.")

        processed_count = 0
        skipped_count = 0
        new_faces_count = 0

        for file_path in tqdm(stock_files, desc="Indexing Local Stock Photos"):
            orig_name = file_path.name
            
            existing = get_photo_by_original_name(orig_name)
            if existing:
                skipped_count += 1
                continue

            next_num = get_next_event_photo_number(event_id)
            numbered_filename = f"photo_{next_num:04d}{file_path.suffix.lower()}"
            dest_path = PROCESSED_PHOTOS_DIR / numbered_filename

            shutil.copy2(file_path, dest_path)
            faces = self.face_engine.extract_faces_from_file(str(dest_path))
            
            photo_id = insert_photo(
                photo_number=next_num,
                numbered_filename=numbered_filename,
                original_filename=orig_name,
                original_path=str(file_path),
                stored_path=str(dest_path),
                face_count=len(faces),
                source_type="local",
                event_id=event_id
            )

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
        return {
            "source": "local",
            "event_id": event_id,
            "event_name": default_event["name"],
            "newly_processed_photos": processed_count,
            "skipped_existing_photos": skipped_count,
            "newly_detected_faces": new_faces_count,
            "total_photos": stats["total_photos"],
            "total_faces": stats["total_faces"],
            "faiss_total_vectors": self.index.ntotal
        }

    def sync_event_from_gdrive(self, folder_id: str, custom_event_name: Optional[str] = None) -> dict:
        """
        Complete 1-Click Autonomous Event Pipeline:
        1. Fetches Folder Name from Google Drive (or uses custom name).
        2. Detects & auto-trashes/unlinks duplicate / same-name files.
        3. Checks naming order — auto-renames all Drive files to sequential format (photo_0001.jpg...).
        4. In-Memory Streaming: Extracts face embeddings without saving files to disk.
        5. Associates all data with this specific Event in SQLite & FAISS.
        """
        import cv2

        cleaned_folder_id = self.gdrive_service.extract_folder_id(folder_id)

        # 1. Determine Event Name & Validate Permissions
        perm_check = self.gdrive_service.check_folder_permission(cleaned_folder_id)
        if not perm_check.get("ok"):
            raise RuntimeError(f"Google Drive Error: {perm_check.get('error')}")
        
        if not perm_check.get("can_edit"):
            account = perm_check.get("account_email") or "Service Account"
            raise RuntimeError(
                f"Permission Denied: Account '{account}' does not have 'Editor' access to this Google Drive folder. "
                f"Please open Google Drive, right click the folder -> Share, and grant '{account}' the 'Editor' role."
            )

        if custom_event_name and custom_event_name.strip():
            event_name = custom_event_name.strip()
        else:
            folder_info = self.gdrive_service.get_folder_details(cleaned_folder_id)
            event_name = folder_info.get("name") or f"Event_{cleaned_folder_id[:6]}"

        event = create_or_get_event(name=event_name, folder_id=cleaned_folder_id, source_type="gdrive")
        event_id = event["id"]
        print(f"=== Starting Autonomous Sync for Event: '{event_name}' (ID: {event_id}) ===")

        # 2. Duplicate Detection & Auto-Trash
        dup_report = self.gdrive_service.find_and_clean_duplicates(cleaned_folder_id, auto_delete=True)
        print(f"Duplicate cleanup: {dup_report['duplicates_found']} found, {dup_report['duplicates_deleted']} cleaned.")

        # 3. Automatic Drive Folder Sequential Renaming
        rename_report = self.gdrive_service.batch_rename_folder(cleaned_folder_id, prefix="photo")
        print(f"Drive Renaming: {rename_report['renamed_count']} files organized to photo_XXXX format.")

        # 4. Fetch Cleaned & Renamed File List
        folder_files = self.gdrive_service.list_folder_photos(cleaned_folder_id)
        print(f"Found {len(folder_files)} photo(s) in Drive event folder ready for in-memory indexing.")

        processed_count = 0
        skipped_count = 0
        new_faces_count = 0

        for file_meta in tqdm(folder_files, desc=f"Indexing Event '{event_name}' (In-Memory)"):
            gdrive_id = file_meta["id"]
            current_drive_name = file_meta.get("name", f"{gdrive_id}.jpg")
            
            existing = get_photo_by_gdrive_id(gdrive_id)
            if existing:
                skipped_count += 1
                continue

            # Stream photo directly into RAM
            try:
                raw_bytes = self.gdrive_service.get_photo_bytes(gdrive_id)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"[Warning] Could not decode image {current_drive_name}")
                    continue
            except Exception as dl_err:
                print(f"[Warning] Could not stream image {current_drive_name}: {dl_err}")
                continue

            # Event photo numbering starts clean for each event
            next_num = get_next_event_photo_number(event_id)
            ext = Path(current_drive_name).suffix.lower() or ".jpg"
            numbered_filename = f"photo_{next_num:04d}{ext}"

            # Extract face embeddings in memory
            faces = self.face_engine.extract_faces_from_image(img_bgr)
            gdrive_link = file_meta.get("webViewLink") or f"https://drive.google.com/file/d/{gdrive_id}/view"

            photo_id = insert_photo(
                photo_number=next_num,
                numbered_filename=numbered_filename,
                original_filename=current_drive_name,
                original_path=f"gdrive://{cleaned_folder_id}/{current_drive_name}",
                stored_path=gdrive_link,
                face_count=len(faces),
                source_type="gdrive",
                gdrive_file_id=gdrive_id,
                event_id=event_id
            )

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

        return {
            "source": "gdrive",
            "event_id": event_id,
            "event_name": event_name,
            "duplicates_removed": dup_report.get("duplicates_deleted", 0),
            "files_renamed": rename_report.get("renamed_count", 0),
            "newly_processed_photos": processed_count,
            "skipped_existing_photos": skipped_count,
            "newly_detected_faces": new_faces_count,
            "total_photos": stats["total_photos"],
            "total_faces": stats["total_faces"],
            "faiss_total_vectors": self.index.ntotal
        }


