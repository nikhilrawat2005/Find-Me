import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from typing import List, Dict, Any, Optional
from src.config import INSIGHTFACE_MODEL_NAME, DETECTION_SIZE, DETECTION_THRESHOLD

class FaceEngine:
    _instance: Optional['FaceEngine'] = None

    def __init__(self, name: str = INSIGHTFACE_MODEL_NAME):
        # We use providers=['CPUExecutionProvider'] for maximum portability and compatibility
        self.app = FaceAnalysis(
            name=name,
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    @classmethod
    def get_instance(cls) -> 'FaceEngine':
        if cls._instance is None:
            cls._instance = FaceEngine()
        return cls._instance

    def extract_faces_from_image(self, img_bgr: np.ndarray, min_det_score: float = DETECTION_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Detects faces and computes 512-D L2-normalized feature embeddings.
        Returns a list of dicts: {'bbox': [x1, y1, x2, y2], 'det_score': float, 'embedding': np.ndarray}
        """
        if img_bgr is None:
            return []
            
        faces = self.app.get(img_bgr)
        results = []
        for face in faces:
            if face.det_score < min_det_score:
                continue
            
            emb = face.embedding
            # L2 normalize embedding for Cosine similarity / FAISS Inner Product search
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
            results.append({
                "bbox": [float(x) for x in face.bbox],
                "det_score": float(face.det_score),
                "embedding": emb.astype(np.float32)
            })
        return results

    def extract_faces_from_file(self, file_path: str, min_det_score: float = DETECTION_THRESHOLD) -> List[Dict[str, Any]]:
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return self.extract_faces_from_image(img, min_det_score)
