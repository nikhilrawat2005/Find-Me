import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.config import BASE_DIR, PROCESSED_PHOTOS_DIR, SIMILARITY_THRESHOLD
from src.database import get_stats, get_all_photos
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
def api_photos(limit: int = 100, offset: int = 0):
    return get_all_photos(limit=limit, offset=offset)

@app.get("/api/photos/{filename}")
def api_get_photo(filename: str):
    file_path = PROCESSED_PHOTOS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(file_path)

@app.post("/api/index")
def trigger_indexing():
    """
    Triggers the numbering & embedding conversion pipeline on stock photos.
    """
    result = indexer.index_stock_photos()
    searcher.reload_index()
    return {"status": "success", "data": result}

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
