import sqlite3
import numpy as np
from typing import Optional, List, Dict, Any
from src.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 0: Events table tracking folders/events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            folder_id TEXT UNIQUE,
            source_type TEXT DEFAULT 'gdrive',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 1: Photos table tracking original filename, new numbering, source type, event_id, and face counts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            photo_number INTEGER NOT NULL,
            numbered_filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            face_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'processed',
            source_type TEXT DEFAULT 'local',
            gdrive_file_id TEXT,
            section_name TEXT,
            subfolder_path TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL
        )
    """)
    
    # Check if migration needed for existing databases
    cursor.execute("PRAGMA table_info(photos)")
    existing_cols = [col[1] for col in cursor.fetchall()]
    if "source_type" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN source_type TEXT DEFAULT 'local'")
    if "gdrive_file_id" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN gdrive_file_id TEXT")
    if "event_id" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN event_id INTEGER REFERENCES events(id)")
    if "section_name" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN section_name TEXT")
    if "subfolder_path" not in existing_cols:
        cursor.execute("ALTER TABLE photos ADD COLUMN subfolder_path TEXT")

    # Ensure a default event exists for existing local photos
    cursor.execute("SELECT id FROM events WHERE name = 'Local Stock Showcase' OR folder_id = 'local_stock'")
    default_event = cursor.fetchone()
    if not default_event:
        cursor.execute("INSERT INTO events (name, folder_id, source_type) VALUES ('Local Stock Showcase', 'local_stock', 'local')")
        default_event_id = cursor.lastrowid
    else:
        default_event_id = default_event[0]

    # Assign null event_ids to default event
    cursor.execute("UPDATE photos SET event_id = ? WHERE event_id IS NULL AND source_type = 'local'", (default_event_id,))
    
    # Table 2: Face Embeddings table storing bounding boxes, scores, embedding blob, and FAISS vector index position
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
            embedding BLOB,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("PRAGMA table_info(faces)")
    existing_face_cols = [col[1] for col in cursor.fetchall()]
    if "embedding" not in existing_face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN embedding BLOB")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_photo_number ON photos(photo_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_photo_event ON photos(event_id)")
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

def get_photo_by_id(photo_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
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

def get_next_event_photo_number(event_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(photo_number), 0) + 1 FROM photos WHERE event_id = ?", (event_id,))
    res = cursor.fetchone()[0]
    conn.close()
    return int(res)

def create_or_get_event(name: str, folder_id: Optional[str] = None, source_type: str = "gdrive") -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    if folder_id:
        cursor.execute("SELECT * FROM events WHERE folder_id = ?", (folder_id,))
        row = cursor.fetchone()
        if row:
            conn.close()
            return dict(row)
    
    cursor.execute("INSERT INTO events (name, folder_id, source_type) VALUES (?, ?, ?)", (name, folder_id, source_type))
    event_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def get_all_events() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        SELECT e.*, 
               COUNT(p.id) as photo_count,
               COALESCE(SUM(p.face_count), 0) as total_faces
        FROM events e
        LEFT JOIN photos p ON e.id = p.event_id
        GROUP BY e.id
        ORDER BY e.created_at DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_event_by_id(event_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def insert_photo(
    photo_number: int,
    numbered_filename: str,
    original_filename: str,
    original_path: str,
    stored_path: str,
    face_count: int,
    source_type: str = "local",
    gdrive_file_id: Optional[str] = None,
    event_id: Optional[int] = None,
    section_name: Optional[str] = None,
    subfolder_path: Optional[str] = None
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO photos (photo_number, numbered_filename, original_filename, original_path, stored_path, face_count, source_type, gdrive_file_id, event_id, section_name, subfolder_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (photo_number, numbered_filename, original_filename, original_path, stored_path, face_count, source_type, gdrive_file_id, event_id, section_name, subfolder_path))
    photo_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return photo_id

def insert_face(photo_id: int, vector_index: int, bbox: List[float], det_score: float, embedding: Optional[np.ndarray] = None) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    cursor.execute("""
        INSERT INTO faces (photo_id, vector_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (photo_id, vector_index, bbox[0], bbox[1], bbox[2], bbox[3], det_score, emb_blob))
    face_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return face_id

def reset_event_photos(event_id: int) -> bool:
    """
    Clears all photos and faces belonging to an event so it can be cleanly re-synced from scratch.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE event_id = ?)", (event_id,))
    cursor.execute("DELETE FROM photos WHERE event_id = ?", (event_id,))
    conn.commit()
    conn.close()
    return True

def get_faces_by_vector_indices(vector_indices: List[int], event_id: Optional[int] = None) -> List[Dict[str, Any]]:
    if not vector_indices:
        return []
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in vector_indices)
    params = list(vector_indices)
    event_clause = ""
    if event_id:
        event_clause = "AND p.event_id = ?"
        params.append(event_id)

    query = f"""
        SELECT f.*, p.photo_number, p.numbered_filename, p.original_filename, p.stored_path, p.face_count, p.source_type, p.gdrive_file_id, p.event_id,
               p.section_name, p.subfolder_path,
               COALESCE(e.name, 'Local Stock Showcase') as event_name
        FROM faces f
        JOIN photos p ON f.photo_id = p.id
        LEFT JOIN events e ON p.event_id = e.id
        WHERE f.vector_index IN ({placeholders}) {event_clause}
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_stats(event_id: Optional[int] = None) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    
    if event_id is not None:
        cursor.execute("SELECT COUNT(*) FROM photos WHERE event_id = ?", (event_id,))
        total_photos = cursor.fetchone()[0]
        cursor.execute("""
            SELECT COUNT(f.id) FROM faces f
            JOIN photos p ON f.photo_id = p.id
            WHERE p.event_id = ?
        """, (event_id,))
        total_faces = cursor.fetchone()[0]
        cursor.execute("SELECT source_type FROM events WHERE id = ?", (event_id,))
        ev_row = cursor.fetchone()
        source_type = ev_row[0] if ev_row else "unknown"
        conn.close()
        return {
            "event_id": event_id,
            "total_photos": total_photos,
            "total_faces": total_faces,
            "source_type": source_type
        }

    cursor.execute("SELECT COUNT(*) FROM photos")
    total_photos = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM faces")
    total_faces = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM photos WHERE source_type = 'local'")
    local_photos = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM photos WHERE source_type = 'gdrive'")
    gdrive_photos = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events")
    total_events = cursor.fetchone()[0]
    conn.close()
    return {
        "total_photos": total_photos,
        "total_faces": total_faces,
        "local_photos": local_photos,
        "gdrive_photos": gdrive_photos,
        "total_events": total_events
    }

def get_all_photos(limit: int = 200, offset: int = 0, source_type: Optional[str] = None, event_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    conditions = []
    params = []
    if source_type:
        conditions.append("source_type = ?")
        params.append(source_type)
    if event_id:
        conditions.append("event_id = ?")
        params.append(event_id)
    
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])
    query = f"SELECT * FROM photos {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_event(event_id: int) -> bool:
    """
    Deletes an event and its associated photos and faces.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE event_id = ?)", (event_id,))
    cursor.execute("DELETE FROM photos WHERE event_id = ?", (event_id,))
    cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    return True



