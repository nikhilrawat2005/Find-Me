import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import FAISS_INDEX_PATH
from src.database import get_faces_by_vector_indices
from src.face_engine import FaceEngine

# ── 3-Tier matching thresholds ──────────────────────────────────────────────
TIER_STRONG   = 0.52   # score >= 0.52  → Strong Match   🟢
TIER_LIKELY   = 0.40   # score >= 0.40  → Likely Match   🟡
TIER_POSSIBLE = 0.30   # score >= 0.30  → Possible Match 🟠
# All results below TIER_POSSIBLE are discarded.


def _assign_tier(score: float) -> Dict[str, str]:
    if score >= TIER_STRONG:
        return {"tier": "strong",   "label": "Strong Match",   "color": "green"}
    elif score >= TIER_LIKELY:
        return {"tier": "likely",   "label": "Likely Match",   "color": "amber"}
    else:
        return {"tier": "possible", "label": "Possible Match", "color": "orange"}


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
        3-Tier face search:
        - Searches FAISS at the lowest threshold (0.28) to cast a wide net.
        - Optionally filters results by event_id.
        - Every returned result is classified into one of three tiers based on
          its cosine similarity score:
            Strong   (>= 0.52) — near-certain match
            Likely   (>= 0.40) — probable match
            Possible (>= 0.28) — distant / challenging match
        - Results are sorted descending by score.
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

        # ── FAISS search — use wide net (top_k large, low threshold) ──────
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        scores  = scores[0]
        indices = indices[0]

        # Filter at floor threshold
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


        # Group by photo — keep highest score per photo
        photos_dict: Dict[int, Dict[str, Any]] = {}
        for face_info in db_faces:
            photo_id  = face_info["photo_id"]
            vec_idx   = face_info["vector_index"]
            similarity = score_map.get(vec_idx, 0.0)

            if photo_id not in photos_dict:
                tier_info = _assign_tier(similarity)
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
                    "event_name":        face_info.get("event_name", "Local Stock Showcase"),
                    "similarity_score":  round(similarity, 4),
                    "percentage":        round(similarity * 100, 1),
                    "tier":              tier_info["tier"],
                    "tier_label":        tier_info["label"],
                    "tier_color":        tier_info["color"],
                    "matched_faces":     []
                }
            else:
                # Update if this face has a higher score
                if similarity > photos_dict[photo_id]["similarity_score"]:
                    tier_info = _assign_tier(similarity)
                    photos_dict[photo_id]["similarity_score"] = round(similarity, 4)
                    photos_dict[photo_id]["percentage"]       = round(similarity * 100, 1)
                    photos_dict[photo_id]["tier"]             = tier_info["tier"]
                    photos_dict[photo_id]["tier_label"]       = tier_info["label"]
                    photos_dict[photo_id]["tier_color"]       = tier_info["color"]

            photos_dict[photo_id]["matched_faces"].append({
                "face_id": face_info["id"],
                "score":   round(similarity, 4),
                "bbox":    [face_info["bbox_x1"], face_info["bbox_y1"],
                            face_info["bbox_x2"], face_info["bbox_y2"]]
            })

        # ── Sort descending by score ───────────────────────────────────────
        sorted_results = sorted(
            photos_dict.values(),
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        # ── Group-photo filter ─────────────────────────────────────────────
        # If a photo has > 10 faces (crowd/group shot), a "Possible" match is
        # too risky — too many people to be confident. Only keep Strong/Likely.
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
