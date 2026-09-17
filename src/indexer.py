import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import numpy as np

import faiss
from tqdm import tqdm

from src.config import LOCAL_DIR, STOCK_DIR, PROCESSED_PHOTOS_DIR, FAISS_INDEX_PATH, GDRIVE_DIR
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

# Global live sync progress state
GLOBAL_SYNC_PROGRESS = {
    "is_running": False,
    "event_id": None,
    "event_name": "",
    "phase": "idle",
    "current": 0,
    "total": 0,
    "percentage": 0,
    "message": ""
}

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

    def init_index(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.save_index()

    def rebuild_index_from_db(self):
        """
        Reconstructs the FAISS vector index cleanly from the embeddings stored in SQLite,
        re-assigning sequential vector_index positions so deletions never leave ghost vectors.
        """
        from src.database import get_connection
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM faces WHERE embedding IS NOT NULL ORDER BY id ASC")
        rows = cursor.fetchall()
        
        updated_pairs = []
        vec_idx = 0
        for row in rows:
            face_id = row[0]
            emb_blob = row[1]
            if emb_blob:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if emb.shape[0] == EMBEDDING_DIM:
                    emb_matrix = np.expand_dims(emb, axis=0)
                    self.index.add(emb_matrix)
                    updated_pairs.append((vec_idx, face_id))
                    vec_idx += 1

        cursor.executemany("UPDATE faces SET vector_index = ? WHERE id = ?", updated_pairs)
        conn.commit()
        conn.close()
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def index_stock_photos(self) -> dict:
        """
        Processes photos from LOCAL_DIR (subfolders like local_recruitment) or fallback STOCK_DIR.
        Creates events matching each subfolder name automatically.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        total_processed = 0
        total_skipped = 0
        total_new_faces = 0
        events_processed = []

        # Find folders in LOCAL_DIR or fallback to STOCK_DIR
        target_dirs = []
        if LOCAL_DIR.exists():
            subdirs = [d for d in LOCAL_DIR.iterdir() if d.is_dir()]
            if subdirs:
                target_dirs.extend(subdirs)
            else:
                target_dirs.append(LOCAL_DIR)
        elif STOCK_DIR.exists():
            target_dirs.append(STOCK_DIR)
        else:
            raise FileNotFoundError("Neither local/ nor stock/ directory found.")

        for folder in target_dirs:
            folder_name = folder.name.replace("_", " ").title() if folder != LOCAL_DIR else "Local Stock Showcase"
            folder_slug = f"local_{folder.name}"
            event = create_or_get_event(name=folder_name, folder_id=folder_slug, source_type="local")
            event_id = event["id"]
            events_processed.append(folder_name)

            stock_files = sorted([
                p for p in folder.iterdir() 
                if p.is_file() and p.suffix.lower() in valid_extensions
            ])

            if not stock_files:
                continue

            print(f"Found {len(stock_files)} photo(s) in local folder '{folder_name}'.")

            for file_path in tqdm(stock_files, desc=f"Indexing Local [{folder_name}]"):
                orig_name = file_path.name
                
                existing = get_photo_by_original_name(orig_name)
                if existing:
                    total_skipped += 1
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
                    total_new_faces += 1

                total_processed += 1

        self.save_index()
        stats = get_stats()
        return {
            "source": "local",
            "events_processed": events_processed,
            "newly_processed_photos": total_processed,
            "skipped_existing_photos": total_skipped,
            "newly_detected_faces": total_new_faces,
            "total_photos": stats["total_photos"],
            "total_faces": stats["total_faces"],
            "faiss_total_vectors": self.index.ntotal
        }

    def sync_event_from_gdrive(self, folder_id: str, custom_event_name: Optional[str] = None, fresh_reset: bool = False) -> dict:
        """
        High-Performance 1-Click Autonomous Event Pipeline:
        1. Validates permissions & event details.
        2. If fresh_reset is True, wipes existing database & FAISS entries for this event and forces fresh photo_0001 numbering.
        3. Auto-removes duplicates.
        4. Auto-renames Drive files sequentially (photo_0001.jpg... or continuing from max_existing).
        5. High-speed multi-threaded prefetching queue with real-time progress updates.
        6. In-Memory fast processing with optimized image scaling & retry backoffs.
        """
        import cv2
        from src.database import reset_event_photos

        global GLOBAL_SYNC_PROGRESS

        cleaned_folder_id = self.gdrive_service.extract_folder_id(folder_id)

        GLOBAL_SYNC_PROGRESS.update({
            "is_running": True,
            "event_id": None,
            "event_name": custom_event_name or "Syncing Event...",
            "phase": "validating",
            "current": 0,
            "total": 0,
            "percentage": 0,
            "message": "Validating Google Drive permissions..."
        })

        try:
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
            GLOBAL_SYNC_PROGRESS["event_id"] = event_id
            GLOBAL_SYNC_PROGRESS["event_name"] = event_name

            print(f"=== Starting Autonomous Turbo Sync for Event: '{event_name}' (ID: {event_id}, Fresh Reset: {fresh_reset}) ===")

            # Optional: Clean wipe if user requested Fresh Reset
            if fresh_reset:
                GLOBAL_SYNC_PROGRESS.update({
                    "phase": "resetting",
                    "message": "Resetting existing database records and vector index for this event..."
                })
                print(f"[Reset] Wiping previous photo and face records for Event ID {event_id}...")
                reset_event_photos(event_id)
                self.rebuild_index_from_db()

            # 2. Duplicate Detection & Auto-Trash
            GLOBAL_SYNC_PROGRESS.update({
                "phase": "duplicates",
                "message": "Checking and removing duplicate files in Google Drive..."
            })
            dup_report = self.gdrive_service.find_and_clean_duplicates(cleaned_folder_id, auto_delete=True)
            print(f"Duplicate cleanup: {dup_report['duplicates_found']} found, {dup_report['duplicates_deleted']} cleaned.")

            # 3. Automatic Drive Folder Sequential Renaming
            GLOBAL_SYNC_PROGRESS.update({
                "phase": "renaming",
                "message": "Organizing photo sequence on Google Drive..."
            })
            rename_report = self.gdrive_service.batch_rename_folder(cleaned_folder_id, prefix="photo", fresh_reset=fresh_reset)
            print(f"Drive Renaming: {rename_report['renamed_count']} files organized to photo_XXXX format.")

            # 4. Fetch Cleaned & Renamed File List
            folder_files = self.gdrive_service.list_folder_photos(cleaned_folder_id)
            print(f"Found {len(folder_files)} photo(s) in Drive event folder ready for in-memory indexing.")

            # Filter unindexed files
            to_process = []
            skipped_count = 0
            for file_meta in folder_files:
                gdrive_id = file_meta["id"]
                if not fresh_reset and get_photo_by_gdrive_id(gdrive_id):
                    skipped_count += 1
                else:
                    to_process.append(file_meta)

            processed_count = 0
            new_faces_count = 0

            if not to_process:
                self.save_index()
                stats = get_stats()
                GLOBAL_SYNC_PROGRESS.update({
                    "is_running": False,
                    "phase": "completed",
                    "current": 0,
                    "total": 0,
                    "percentage": 100,
                    "message": f"Event '{event_name}' is fully up to date ({skipped_count} photos already indexed)."
                })
                return {
                    "source": "gdrive",
                    "event_id": event_id,
                    "event_name": event_name,
                    "duplicates_removed": dup_report.get("duplicates_deleted", 0),
                    "files_renamed": rename_report.get("renamed_count", 0),
                    "newly_processed_photos": 0,
                    "skipped_existing_photos": skipped_count,
                    "newly_detected_faces": 0,
                    "total_photos": stats["total_photos"],
                    "total_faces": stats["total_faces"],
                    "faiss_total_vectors": self.index.ntotal
                }

            total_items = len(to_process)
            GLOBAL_SYNC_PROGRESS.update({
                "phase": "indexing",
                "current": 0,
                "total": total_items,
                "percentage": 0,
                "message": f"Starting turbo indexing of {total_items} photo(s)..."
            })

            # Multi-threaded download pipeline (using 4 parallel workers with thread-isolated Drive clients)
            prefetch_queue = queue.Queue(maxsize=12)
            stop_token = object()

            def download_worker():
                try:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        def fetch_task(file_meta):
                            gid = file_meta["id"]
                            try:
                                raw_bytes = self.gdrive_service.get_photo_bytes(gid)
                                return (file_meta, raw_bytes, None)
                            except Exception as err:
                                return (file_meta, None, err)

                        futures = [pool.submit(fetch_task, f_meta) for f_meta in to_process]
                        for fut in as_completed(futures):
                            prefetch_queue.put(fut.result())
                except Exception as e:
                    print(f"[Worker Error] Prefetch pool error: {e}")
                finally:
                    prefetch_queue.put(stop_token)

            downloader_thread = threading.Thread(target=download_worker, daemon=True)
            downloader_thread.start()

            progress_bar = tqdm(total=total_items, desc=f"Turbo Indexing Event '{event_name}'")

            while True:
                item = prefetch_queue.get()
                if item is stop_token:
                    break

                file_meta, raw_bytes, err = item
                gdrive_id = file_meta["id"]
                current_drive_name = file_meta.get("name", f"{gdrive_id}.jpg")

                if err or raw_bytes is None:
                    print(f"\n[Warning] Could not stream {current_drive_name}: {err}")
                    progress_bar.update(1)
                    processed_count += 1
                    pct = int((processed_count / total_items) * 100)
                    GLOBAL_SYNC_PROGRESS.update({
                        "current": processed_count,
                        "percentage": pct,
                        "message": f"Failed {current_drive_name}, continuing... ({processed_count}/{total_items})"
                    })
                    continue

                # In-memory image decode
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"\n[Warning] Could not decode {current_drive_name}")
                    progress_bar.update(1)
                    processed_count += 1
                    pct = int((processed_count / total_items) * 100)
                    GLOBAL_SYNC_PROGRESS.update({
                        "current": processed_count,
                        "percentage": pct,
                        "message": f"Could not decode {current_drive_name} ({processed_count}/{total_items})"
                    })
                    continue

                # Event photo numbering starts clean for each event or continues sequentially
                next_num = get_next_event_photo_number(event_id)
                ext = Path(current_drive_name).suffix.lower() or ".jpg"
                numbered_filename = f"photo_{next_num:04d}{ext}"

                # Fast face extraction with automatic scaling
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
                        det_score=face["det_score"],
                        embedding=emb
                    )
                    new_faces_count += 1

                processed_count += 1
                progress_bar.update(1)
                pct = int((processed_count / total_items) * 100)
                GLOBAL_SYNC_PROGRESS.update({
                    "current": processed_count,
                    "percentage": pct,
                    "message": f"Indexed {numbered_filename} ({len(faces)} faces) — {processed_count}/{total_items} ({pct}%)"
                })

            progress_bar.close()
            downloader_thread.join()

            self.save_index()
            stats = get_stats()

            GLOBAL_SYNC_PROGRESS.update({
                "is_running": False,
                "phase": "completed",
                "current": processed_count,
                "total": total_items,
                "percentage": 100,
                "message": f"Sync completed! {processed_count} photo(s) indexed, {new_faces_count} face(s) registered."
            })

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

        except Exception as e:
            GLOBAL_SYNC_PROGRESS.update({
                "is_running": False,
                "phase": "error",
                "message": f"Error: {str(e)}"
            })
            raise e



