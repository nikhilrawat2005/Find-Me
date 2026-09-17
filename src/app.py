import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

from src.config import BASE_DIR, PROCESSED_PHOTOS_DIR, GDRIVE_DIR, SIMILARITY_THRESHOLD
from src.database import (
    get_stats, 
    get_all_photos, 
    get_photo_by_filename, 
    get_photo_by_id, 
    get_all_events, 
    get_event_by_id,
    delete_event
)
from src.indexer import StockIndexer
from src.searcher import FaceSearcher

app = FastAPI(title="Face Recognition Photo Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

searcher = FaceSearcher()
indexer = StockIndexer()

# Serve static files
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Face Recognition Photo Search System API is running."}

@app.get("/api/events")
def api_events():
    """
    Returns list of all events with photo and face counts.
    """
    return get_all_events()

@app.delete("/api/events/{event_id}")
def api_delete_event(event_id: int):
    """
    Deletes an event along with all its photos and faces, cleanly rebuilding the FAISS index.
    """
    event = get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    delete_event(event_id)
    indexer.rebuild_index_from_db()
    searcher.reload_index()
    return {"status": "success", "message": f"Event '{event['name']}' deleted successfully."}

@app.post("/api/events/{event_id}/resync")
def api_resync_event(
    event_id: int,
    fresh_reset: bool = Query(default=False, description="If True, completely wipes event database/embeddings and re-indexes all photos from photo_0001")
):
    """
    Re-synchronizes an existing event with its Google Drive folder:
    - Validates Editor permissions.
    - If fresh_reset=True: wipes event records & rebuilds FAISS from photo_0001 onwards.
    - If fresh_reset=False: preserves existing photo numbers and appends only new photos sequentially.
    - Removes newly detected duplicates.
    - Indexes unindexed photos in-memory.
    """
    event = get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if not event.get("folder_id"):
        raise HTTPException(status_code=400, detail="This event is not linked to a Google Drive folder")
    
    try:
        result = indexer.sync_event_from_gdrive(
            folder_id=event["folder_id"],
            custom_event_name=event["name"],
            fresh_reset=fresh_reset
        )
        searcher.reload_index()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/events/progress")
def api_events_progress():
    """
    Returns real-time progress of ongoing event sync/indexing tasks.
    """
    from src.indexer import GLOBAL_SYNC_PROGRESS
    return GLOBAL_SYNC_PROGRESS


@app.get("/api/stats")
def api_stats(event_id: Optional[int] = None):
    stats = get_stats(event_id=event_id)
    stats["faiss_vectors"] = searcher.index.ntotal if searcher.index else 0
    return stats

@app.get("/api/photos")
def api_photos(limit: int = 100, offset: int = 0, source: Optional[str] = None, event_id: Optional[int] = None):
    return get_all_photos(limit=limit, offset=offset, source_type=source, event_id=event_id)


@app.get("/api/photos/{photo_identifier}")
def api_get_photo(photo_identifier: str):
    # Try finding by numeric ID first, otherwise filename
    photo_record = None
    if photo_identifier.isdigit():
        photo_record = get_photo_by_id(int(photo_identifier))
    if not photo_record:
        photo_record = get_photo_by_filename(photo_identifier)

    # If local file
    if photo_record and photo_record.get("source_type") == "local":
        local_path = Path(photo_record["stored_path"])
        if local_path.exists():
            return FileResponse(local_path)
    
    # Direct check in processed_photos folder
    file_path = PROCESSED_PHOTOS_DIR / photo_identifier
    if file_path.exists():
        return FileResponse(file_path)

    # If photo originated from Google Drive
    if photo_record and photo_record.get("gdrive_file_id"):
        cached_file = GDRIVE_DIR / f"drive_{photo_record['id']}_{photo_record['numbered_filename']}"
        if cached_file.exists():
            return FileResponse(cached_file)

        try:
            raw_bytes = indexer.gdrive_service.get_photo_bytes(photo_record["gdrive_file_id"])
            try:
                with open(cached_file, "wb") as f:
                    f.write(raw_bytes)
            except Exception:
                pass
            return Response(content=raw_bytes, media_type="image/jpeg")
        except Exception as e:
            err_msg = str(e)
            if "File not found" in err_msg or "404" in err_msg:
                raise HTTPException(status_code=404, detail="Photo deleted or trashed on Drive")
            raise HTTPException(status_code=502, detail=f"Failed to stream from Drive: {err_msg}")

    raise HTTPException(status_code=404, detail="Photo not found")


@app.post("/api/index")
def trigger_indexing():
    """
    Triggers the numbering & embedding conversion pipeline on local stock photos.
    """
    result = indexer.index_stock_photos()
    searcher.reload_index()
    return {"status": "success", "data": result}

@app.get("/api/gdrive/status")
def api_gdrive_status():
    """
    Checks Google Drive API credentials and connectivity.
    """
    return indexer.gdrive_service.check_connection()

@app.post("/api/events/sync")
def api_events_sync(
    folder_id: str = Query(..., description="Google Drive Folder URL or Folder ID"),
    name: Optional[str] = Query(default=None, description="Optional custom name for the Event (e.g. 'Rahul Wedding 2026')")
):
    """
    Autonomous 1-Click Event Pipeline:
    1. Fetches Folder Name from Drive (or uses custom name).
    2. Auto-removes duplicate files.
    3. Auto-renames Drive files sequentially (photo_0001.jpg...).
    4. Extracts face embeddings in-memory directly into FAISS & binds to this Event.
    """
    try:
        result = indexer.sync_event_from_gdrive(folder_id=folder_id, custom_event_name=name)
        searcher.reload_index()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/gdrive/sync")
def api_gdrive_sync(
    folder_id: str = Query(..., description="Google Drive Folder URL or Folder ID"),
    clean_duplicates: bool = Query(default=True, description="Automatically detect and delete duplicate files in Drive"),
    event_name: Optional[str] = Query(default=None, description="Optional custom event name")
):
    """
    Autonomous event sync alias.
    """
    try:
        result = indexer.sync_event_from_gdrive(folder_id=folder_id, custom_event_name=event_name)
        searcher.reload_index()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/gdrive/rename")
def api_gdrive_rename(
    folder_id: str = Query(..., description="Google Drive Folder URL or Folder ID"),
    prefix: str = Query(default="photo", description="Prefix for filenames (e.g. photo -> photo_0001.jpg)")
):
    """
    One-click renames all images in the Google Drive folder sequentially.
    """
    try:
        result = indexer.gdrive_service.batch_rename_folder(folder_id=folder_id, prefix=prefix)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/gdrive/clean-duplicates")
def api_gdrive_clean_duplicates(
    folder_id: str = Query(..., description="Google Drive Folder URL or Folder ID"),
    delete: bool = Query(default=False, description="Whether to actually delete duplicates or just preview")
):
    """
    Finds and optionally deletes duplicate files in a Google Drive folder.
    """
    try:
        result = indexer.gdrive_service.find_and_clean_duplicates(folder_id=folder_id, auto_delete=delete)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/api/search")
async def api_search(
    files: List[UploadFile] = File(...),
    top_k: int = Query(default=100, ge=1, le=300),
    event_id: Optional[int] = Query(default=None, description="Filter search to a specific event ID")
):
    """
    Receives 1 to 4 reference photos and returns matching photos across 3 confidence tiers:
      Strong   (>= 0.52) — near-certain match
      Likely   (>= 0.40) — probable match
      Possible (>= 0.28) — distant / challenging match
    Optionally scoped to a specific event.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least 1 reference photo is required.")

    images_bgr = []
    for f in files:
        contents = await f.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            images_bgr.append(img)

    if not images_bgr:
        raise HTTPException(status_code=400, detail="Could not decode any uploaded image.")

    search_output = searcher.search_by_reference_images(
        images_bgr=images_bgr,
        top_k=top_k,
        event_id=event_id
    )


    return {
        "query_images_count": len(images_bgr),
        "total_matches":      search_output["total_matches"],
        "tier_counts":        search_output["tier_counts"],
        "thresholds":         search_output.get("thresholds", {}),
        "results":            search_output["results"],
    }

@app.post("/api/download-zip")
async def api_download_zip(data: dict):
    """
    Bundles the matched photos into a single ZIP file for download.
    Supports both local processed photos and Google Drive photos (streamed directly into ZIP).
    """
    photo_ids = data.get("photo_ids", [])
    if not photo_ids:
        raise HTTPException(status_code=400, detail="No photos selected for download")

    from src.database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in photo_ids)
    cursor.execute(f"SELECT * FROM photos WHERE id IN ({placeholders})", photo_ids)
    photos = [dict(r) for r in cursor.fetchall()]
    conn.close()

    import zipfile
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for p in photos:
            filename = p.get("numbered_filename") or f"photo_{p['id']}.jpg"
            if p.get("source_type") == "gdrive" and p.get("gdrive_file_id"):
                try:
                    file_bytes = indexer.gdrive_service.get_photo_bytes(p["gdrive_file_id"])
                    zip_file.writestr(filename, file_bytes)
                except Exception as e:
                    print(f"Failed to add Drive photo {filename} to zip: {e}")
            else:
                local_path = PROCESSED_PHOTOS_DIR / filename
                if local_path.exists():
                    zip_file.write(local_path, arcname=filename)

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=matched_photos.zip"}
    )

