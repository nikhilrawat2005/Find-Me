import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

from src.config import BASE_DIR, PROCESSED_PHOTOS_DIR, SIMILARITY_THRESHOLD
from src.database import get_stats, get_all_photos, get_photo_by_filename
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

@app.get("/api/stats")
def api_stats():
    stats = get_stats()
    stats["faiss_vectors"] = searcher.index.ntotal if searcher.index else 0
    return stats

@app.get("/api/photos")
def api_photos(limit: int = 100, offset: int = 0, source: Optional[str] = None):
    return get_all_photos(limit=limit, offset=offset, source_type=source)

@app.get("/api/photos/{filename}")
def api_get_photo(filename: str):
    file_path = PROCESSED_PHOTOS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)

    # If photo originated from Google Drive (not saved on disk)
    photo_record = get_photo_by_filename(filename)
    if photo_record and photo_record.get("gdrive_file_id"):
        try:
            raw_bytes = indexer.gdrive_service.get_photo_bytes(photo_record["gdrive_file_id"])
            return Response(content=raw_bytes, media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to stream from Drive: {e}")

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

@app.post("/api/gdrive/sync")
def api_gdrive_sync(
    folder_id: str = Query(..., description="Google Drive Folder URL or Folder ID"),
    clean_duplicates: bool = Query(default=True, description="Automatically detect and delete duplicate files in Drive")
):
    """
    Scans Google Drive folder, removes duplicates, and indexes face embeddings directly in-memory.
    """
    try:
        result = indexer.index_gdrive_photos(folder_id=folder_id, clean_duplicates_first=clean_duplicates)
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
    top_k: int = Query(default=100, ge=1, le=300)
):
    """
    Receives 1 to 4 reference photos and returns matching photos across 3 confidence tiers:
      Strong   (>= 0.52) — near-certain match
      Likely   (>= 0.40) — probable match
      Possible (>= 0.28) — distant / challenging match
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

