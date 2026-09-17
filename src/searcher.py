import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import FAISS_INDEX_PATH
from src.database import get_faces_by_vector_indices
from src.face_engine import FaceEngine

# ── Multi-Tier Matching Thresholds ──────────────────────────────────────────
TIER_STRONG   = 0.52   # score >= 0.52  → Strong Match   🟢
TIER_LIKELY   = 0.40   # score >= 0.40  → Likely Match   🟡
TIER_POSSIBLE = 0.30   # score >= 0.30  → Possible Match 🟠 (Only for small/distant faces)

# Face Dimension Boundary (pixels)
# Faces >= 120px in max dimension are considered regular/large faces (high detail).
# Faces < 120px are distant/small faces (low resolution/compressed features).
SMALL_FACE_MAX_DIM = 120.0
NORMAL_FACE_MIN_SIMILARITY = 0.395  # Normal faces must have at least ~40% similarity


def _assign_tier(score: float, is_small_face: bool = False) -> Dict[str, str]:
    if score >= TIER_STRONG:
        return {"tier": "strong",   "label": "Strong Match",   "color": "green"}
    elif score >= TIER_LIKELY:
        return {"tier": "likely",   "label": "Likely Match",   "color": "amber"}
    else:
        label = "Distant Match" if is_small_face else "Possible Match"
        return {"tier": "possible", "label": label, "color": "orange"}


class FaceSearcher:
    def __init__(self):
        self.face_engine = FaceEngine.get_instance()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.reload_index()

    def reload_index(self):
        if FAISS_INDEX_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        else:
            self.index = None

    def search_by_reference_images(
        self,
        images_bgr: List[np.ndarray],
        top_k: int = 100,
        event_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Smart Scale-Aware & Crowd-Aware Face Search:
        - Regular / Large Faces (>=120px): Strict cutoff at 40% (0.40) to prevent false positives.
        - Distant / Small Faces (<120px): Relaxed cutoff at 30% (0.30) so compressed/distant shots are never missed.
        - Crowd filter: Group photos with > 10 people discard loose possible matches.
        """
        if self.index is None or self.index.ntotal == 0:
            self.reload_index()
            if self.index is None or self.index.ntotal == 0:
                return {"results": [], "tier_counts": {}, "total_matches": 0}

        # ── Build query embedding ──────────────────────────────────────────
        query_embeddings = []
        for img in images_bgr:
            faces = self.face_engine.extract_faces_from_image(img)
            if faces:
                best_face = max(faces, key=lambda f: f["det_score"])
                query_embeddings.append(best_face["embedding"])

        if not query_embeddings:
            return {"results": [], "tier_counts": {}, "total_matches": 0}

        # Average + normalize (handles multi-photo queries)
        mean_embedding = np.mean(query_embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        query_vector = np.expand_dims(mean_embedding, axis=0).astype(np.float32)

        # ── FAISS search — use wide net (top_k large, floor threshold) ─────
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        scores  = scores[0]
        indices = indices[0]

        # Filter at absolute floor threshold (0.30)
        valid_matches = [
            (int(idx), float(score))
            for score, idx in zip(scores, indices)
            if idx >= 0 and float(score) >= TIER_POSSIBLE
        ]

        if not valid_matches:
            return {"results": [], "tier_counts": {}, "total_matches": 0}

        valid_indices = [m[0] for m in valid_matches]
        score_map     = {m[0]: m[1] for m in valid_matches}

        # ── Fetch metadata (filtered by event_id if provided) ──────────────
        db_faces = get_faces_by_vector_indices(valid_indices, event_id=event_id)

        # Group by photo — keep highest score per photo while applying scale-aware constraints
        photos_dict: Dict[int, Dict[str, Any]] = {}
        for face_info in db_faces:
            photo_id   = face_info["photo_id"]
            vec_idx    = face_info["vector_index"]
            similarity = score_map.get(vec_idx, 0.0)

            # Compute face bounding box dimensions
            bx1, by1 = face_info["bbox_x1"], face_info["bbox_y1"]
            bx2, by2 = face_info["bbox_x2"], face_info["bbox_y2"]
            face_w = max(0.0, float(bx2 - bx1)) if (bx1 is not None and bx2 is not None) else 0.0
            face_h = max(0.0, float(by2 - by1)) if (by1 is not None and by2 is not None) else 0.0
            max_face_dim = max(face_w, face_h)
            is_small = max_face_dim < SMALL_FACE_MAX_DIM

            # Dynamic Scale Rule:
            # If face is normal/large (clear facial detail), similarity MUST be >= 40% (0.395)
            # Only genuinely small/distant faces are allowed in the 30% - 39% range.
            if not is_small and similarity < NORMAL_FACE_MIN_SIMILARITY:
                continue

            if photo_id not in photos_dict:
                tier_info = _assign_tier(similarity, is_small_face=is_small)
                photos_dict[photo_id] = {
                    "photo_id":          photo_id,
                    "photo_number":      face_info["photo_number"],
                    "numbered_filename": face_info["numbered_filename"],
                    "original_filename": face_info["original_filename"],
                    "stored_path":       face_info["stored_path"],
                    "face_count":        face_info["face_count"],
                    "source_type":       face_info.get("source_type", "local"),
                    "gdrive_file_id":    face_info.get("gdrive_file_id"),
                    "event_id":          face_info.get("event_id"),
                    "event_name":        face_info.get("event_name", "Local Stock"),
                    "section_name":      face_info.get("section_name"),
                    "subfolder_path":    face_info.get("subfolder_path"),
                    "similarity_score":  round(similarity, 4),
                    "percentage":        round(similarity * 100, 1),
                    "tier":              tier_info["tier"],
                    "tier_label":        tier_info["label"],
                    "tier_color":        tier_info["color"],
                    "is_small_face":     is_small,
                    "matched_faces":     []
                }
            else:
                # Update if this face has a higher score
                if similarity > photos_dict[photo_id]["similarity_score"]:
                    tier_info = _assign_tier(similarity, is_small_face=is_small)
                    photos_dict[photo_id]["similarity_score"] = round(similarity, 4)
                    photos_dict[photo_id]["percentage"]       = round(similarity * 100, 1)
                    photos_dict[photo_id]["tier"]             = tier_info["tier"]
                    photos_dict[photo_id]["tier_label"]       = tier_info["label"]
                    photos_dict[photo_id]["tier_color"]       = tier_info["color"]
                    photos_dict[photo_id]["is_small_face"]     = is_small

            photos_dict[photo_id]["matched_faces"].append({
                "face_id":       face_info["id"],
                "score":         round(similarity, 4),
                "max_face_dim":  round(max_face_dim, 1),
                "is_small_face": is_small,
                "bbox":          [bx1, by1, bx2, by2]
            })

        # ── Sort descending by score ───────────────────────────────────────
        sorted_results = sorted(
            photos_dict.values(),
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        # ── Group-photo filter ─────────────────────────────────────────────
        # If a photo has > 10 faces (crowd/group shot), a loose "Possible" match is
        # rejected to eliminate false alarms in large audiences.
        GROUP_FACE_THRESHOLD = 10
        filtered_results = [
            r for r in sorted_results
            if not (r["face_count"] > GROUP_FACE_THRESHOLD and r["tier"] == "possible")
        ]

        # ── Tier counts for UI summary ─────────────────────────────────────
        tier_counts = {"strong": 0, "likely": 0, "possible": 0}
        for r in filtered_results:
            tier_counts[r["tier"]] += 1

        return {
            "results":       filtered_results,
            "tier_counts":   tier_counts,
            "total_matches": len(filtered_results),
            "thresholds":    {
                "strong":   TIER_STRONG,
                "likely":   TIER_LIKELY,
                "possible": TIER_POSSIBLE,
            }
        }
