import sqlite3
from typing import Optional, List, Dict, Any
from src.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: Photos table tracking original filename, new numbering, source type, and face counts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_number INTEGER UNIQUE NOT NULL,
            numbered_filename TEXT UNIQUE NOT NULL,
            original_filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            face_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'processed',
            source_type TEXT DEFAULT 'local',
            gdrive_file_id TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if migration needed for existing databases
    cursor.execute("PRAGMA table_info(photos)")
    existing_cols = [col[1] for col in cursor.fetchall()]
    if "source_type" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN source_type TEXT DEFAULT 'local'")
    if "gdrive_file_id" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN gdrive_file_id TEXT")
    
    # Table 2: Face Embeddings table storing bounding boxes, scores, and FAISS vector index position
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            vector_index INTEGER NOT NULL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            det_score REAL,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_photo_number ON photos(photo_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_index ON faces(vector_index)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gdrive_file_id ON photos(gdrive_file_id)")
    
    conn.commit()
    conn.close()

def get_photo_by_original_name(original_filename: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE original_filename = ?", (original_filename,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_photo_by_gdrive_id(gdrive_file_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE gdrive_file_id = ?", (gdrive_file_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_photo_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE numbered_filename = ? OR original_filename = ?", (filename, filename))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_next_photo_number() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(photo_number), 0) + 1 FROM photos")
    res = cursor.fetchone()[0]
    conn.close()
    return int(res)

def insert_photo(photo_number: int, numbered_filename: str, original_filename: str, original_path: str, stored_path: str, face_count: int, source_type: str = "local", gdrive_file_id: Optional[str] = None) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO photos (photo_number, numbered_filename, original_filename, original_path, stored_path, face_count, source_type, gdrive_file_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (photo_number, numbered_filename, original_filename, original_path, stored_path, face_count, source_type, gdrive_file_id))
    photo_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return photo_id

def insert_face(photo_id: int, vector_index: int, bbox: List[float], det_score: float) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO faces (photo_id, vector_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (photo_id, vector_index, bbox[0], bbox[1], bbox[2], bbox[3], det_score))
    face_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return face_id

def get_faces_by_vector_indices(vector_indices: List[int]) -> List[Dict[str, Any]]:
    if not vector_indices:
        return []
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in vector_indices)
    query = f"""
        SELECT f.*, p.photo_number, p.numbered_filename, p.original_filename, p.stored_path, p.face_count, p.source_type, p.gdrive_file_id
        FROM faces f
        JOIN photos p ON f.photo_id = p.id
        WHERE f.vector_index IN ({placeholders})
    """
    cursor.execute(query, vector_indices)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM photos")
    total_photos = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM faces")
    total_faces = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM photos WHERE source_type = 'local'")
    local_photos = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM photos WHERE source_type = 'gdrive'")
    gdrive_photos = cursor.fetchone()[0]
    conn.close()
    return {
        "total_photos": total_photos,
        "total_faces": total_faces,
        "local_photos": local_photos,
        "gdrive_photos": gdrive_photos
    }

def get_all_photos(limit: int = 200, offset: int = 0, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if source_type:
        cursor.execute("SELECT * FROM photos WHERE source_type = ? ORDER BY photo_number ASC LIMIT ? OFFSET ?", (source_type, limit, offset))
    else:
        cursor.execute("SELECT * FROM photos ORDER BY photo_number ASC LIMIT ? OFFSET ?", (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

