import cv2
import numpy as np
from typing import Dict, Any, Tuple


class ImageQualityAnalyzer:
    """
    Analyzes image quality and environmental conditions to compute an optimal
    face-matching similarity threshold.

    Strategy:
    - Start from a sensible base threshold (0.45).
    - Apply continuous, proportional adjustments based on measured image quality,
      rather than hard-coded condition buckets.
    - This makes the threshold flexible for every kind of real-world photo:
      event photos, outdoor daylight, dark auditoriums, overexposed flash, blurry
      crowd shots, profile views, far-away faces, etc.
    - Final threshold is clamped to [0.28, 0.58] to prevent extreme false positives
      or false negatives.
    """

    # --- Calibration constants ---
    BASE_THRESHOLD        = 0.45   # Neutral starting point
    MIN_THRESHOLD         = 0.28   # Absolute floor — allows very challenging matches
    MAX_THRESHOLD         = 0.58   # Absolute ceiling — prevents false positives in ideal shots

    # Brightness boundaries
    DARK_EXTREME_MAX      = 45     # Very dark — heavy shadows, night scenes
    DARK_MAX              = 75     # Under-exposed indoor/evening shots
    DIM_MAX               = 100    # Slightly dim — cloudy, indoor no flash
    BRIGHT_MIN            = 160    # Slightly bright
    OVERBRIGHT_MIN        = 200    # Overexposed / harsh flash

    # Contrast boundaries
    FLAT_CONTRAST_MAX     = 25     # Very flat / washed out
    LOW_CONTRAST_MAX      = 42     # Low contrast
    HIGH_CONTRAST_MIN     = 65     # Good dynamic range
    VIVID_CONTRAST_MIN    = 85     # Very high contrast (harsh shadows possible)

    # Sharpness (Laplacian variance) boundaries
    VERY_BLURRY_MAX       = 20     # Heavily blurred / motion blur
    BLURRY_MAX            = 60     # Soft / slightly out-of-focus
    SHARP_MIN             = 150    # Clear and sharp
    VERY_SHARP_MIN        = 350    # Crisp, professional-quality

    @staticmethod
    def _brightness_penalty(brightness: float) -> Tuple[float, str]:
        """
        Computes a threshold delta based on brightness.
        Dark and over-bright photos get a negative delta (lower threshold = more permissive).
        Ideal brightness photos get a positive delta (stricter threshold).
        """
        b = brightness
        if b < ImageQualityAnalyzer.DARK_EXTREME_MAX:
            # Very dark: face features heavily degraded
            delta = -0.10
            note = f"Very dark image (brightness={b:.0f}). Large threshold relaxation."
        elif b < ImageQualityAnalyzer.DARK_MAX:
            # Scale -0.07 to -0.03 as brightness goes 45→75
            t = (b - ImageQualityAnalyzer.DARK_EXTREME_MAX) / (
                ImageQualityAnalyzer.DARK_MAX - ImageQualityAnalyzer.DARK_EXTREME_MAX)
            delta = -0.07 + 0.04 * t
            note = f"Dark image (brightness={b:.0f}). Threshold relaxed."
        elif b < ImageQualityAnalyzer.DIM_MAX:
            t = (b - ImageQualityAnalyzer.DARK_MAX) / (
                ImageQualityAnalyzer.DIM_MAX - ImageQualityAnalyzer.DARK_MAX)
            delta = -0.03 + 0.03 * t  # -0.03 → 0.00
            note = f"Dim lighting (brightness={b:.0f})."
        elif b < ImageQualityAnalyzer.BRIGHT_MIN:
            # Ideal range: 100 – 160 → slight positive nudge
            delta = 0.01
            note = ""
        elif b < ImageQualityAnalyzer.OVERBRIGHT_MIN:
            t = (b - ImageQualityAnalyzer.BRIGHT_MIN) / (
                ImageQualityAnalyzer.OVERBRIGHT_MIN - ImageQualityAnalyzer.BRIGHT_MIN)
            delta = -0.02 * t  # gently lower as we approach over-bright
            note = f"Bright image (brightness={b:.0f}). Slight threshold adjustment."
        else:
            # Over-exposed / flash-washed: highlights destroy surface detail
            t = min((b - ImageQualityAnalyzer.OVERBRIGHT_MIN) / 50.0, 1.0)
            delta = -0.04 - 0.04 * t
            note = f"Over-bright / flash-washed (brightness={b:.0f}). Threshold lowered."
        return round(delta, 3), note

    @staticmethod
    def _contrast_penalty(contrast: float) -> Tuple[float, str]:
        """
        Low contrast → more permissive. High contrast → slightly stricter.
        """
        c = contrast
        if c < ImageQualityAnalyzer.FLAT_CONTRAST_MAX:
            delta = -0.06
            note = f"Very flat contrast (std={c:.1f}). Significant threshold relaxation."
        elif c < ImageQualityAnalyzer.LOW_CONTRAST_MAX:
            t = (c - ImageQualityAnalyzer.FLAT_CONTRAST_MAX) / (
                ImageQualityAnalyzer.LOW_CONTRAST_MAX - ImageQualityAnalyzer.FLAT_CONTRAST_MAX)
            delta = -0.06 + 0.04 * t   # -0.06 → -0.02
            note = f"Low contrast (std={c:.1f})."
        elif c < ImageQualityAnalyzer.HIGH_CONTRAST_MIN:
            delta = 0.0
            note = ""
        elif c < ImageQualityAnalyzer.VIVID_CONTRAST_MIN:
            delta = 0.02
            note = ""
        else:
            # Harsh shadows (very high contrast) can confuse facial geometry
            t = min((c - ImageQualityAnalyzer.VIVID_CONTRAST_MIN) / 30.0, 1.0)
            delta = 0.02 - 0.03 * t   # +0.02 → -0.01
            note = f"Very high contrast/harsh shadows (std={c:.1f}). Minor adjustment."
        return round(delta, 3), note

    @staticmethod
    def _sharpness_penalty(sharpness: float) -> Tuple[float, str]:
        """
        Blurry images → more permissive. Very sharp → stricter.
        """
        s = sharpness
        if s < ImageQualityAnalyzer.VERY_BLURRY_MAX:
            delta = -0.09
            note = f"Heavily blurred / motion blur (sharpness={s:.1f}). Large threshold relaxation."
        elif s < ImageQualityAnalyzer.BLURRY_MAX:
            t = (s - ImageQualityAnalyzer.VERY_BLURRY_MAX) / (
                ImageQualityAnalyzer.BLURRY_MAX - ImageQualityAnalyzer.VERY_BLURRY_MAX)
            delta = -0.09 + 0.07 * t   # -0.09 → -0.02
            note = f"Blurry / soft focus (sharpness={s:.1f}). Threshold relaxed."
        elif s < ImageQualityAnalyzer.SHARP_MIN:
            delta = 0.0
            note = ""
        elif s < ImageQualityAnalyzer.VERY_SHARP_MIN:
            delta = 0.02
            note = ""
        else:
            delta = 0.04
            note = f"Very sharp / high-res image (sharpness={s:.1f}). Stricter matching."
        return round(delta, 3), note

    @staticmethod
    def _face_size_penalty(img_bgr: np.ndarray) -> Tuple[float, str]:
        """
        Estimate whether the face is likely small (far away or group photo)
        by checking the relative size of the largest skin-tone region.
        Small relative face area → lower threshold (more permissive).
        """
        h, w = img_bgr.shape[:2]
        total_pixels = h * w

        # Convert to HSV and detect approximate skin tones
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        skin_pixels = int(np.sum(skin_mask > 0))
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0

        if skin_ratio < 0.02:
            # Very small face / no clear face region (far away, side angle, crowd)
            delta = -0.06
            note = f"Small or obscured face region detected (skin_ratio={skin_ratio:.3f}). Lowering threshold."
        elif skin_ratio < 0.06:
            t = (skin_ratio - 0.02) / 0.04
            delta = -0.06 + 0.04 * t   # -0.06 → -0.02
            note = f"Relatively small face in frame (skin_ratio={skin_ratio:.3f})."
        elif skin_ratio < 0.15:
            delta = 0.0
            note = ""
        else:
            # Very close portrait — high face coverage → tighten slightly
            delta = 0.02
            note = ""
        return round(delta, 3), note

    @staticmethod
    def _infer_condition_label(brightness: float, contrast: float, sharpness: float) -> str:
        if brightness < ImageQualityAnalyzer.DARK_EXTREME_MAX:
            return "very_dark"
        if brightness < ImageQualityAnalyzer.DARK_MAX:
            return "dark"
        if brightness < ImageQualityAnalyzer.DIM_MAX:
            return "dim"
        if brightness > ImageQualityAnalyzer.OVERBRIGHT_MIN:
            return "over_bright"
        if brightness > ImageQualityAnalyzer.BRIGHT_MIN:
            return "bright"
        if sharpness < ImageQualityAnalyzer.VERY_BLURRY_MAX:
            return "very_blurry"
        if sharpness < ImageQualityAnalyzer.BLURRY_MAX:
            return "blurry"
        if sharpness > ImageQualityAnalyzer.VERY_SHARP_MIN and contrast > ImageQualityAnalyzer.HIGH_CONTRAST_MIN:
            return "optimal"
        return "normal"

    @staticmethod
    def analyze_image_properties(img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Full quality analysis returning an adaptively computed threshold and diagnostics.
        """
        if img_bgr is None:
            return {
                "brightness": 128.0,
                "contrast": 50.0,
                "sharpness": 100.0,
                "condition": "normal",
                "recommended_threshold": ImageQualityAnalyzer.BASE_THRESHOLD,
                "confidence": "default",
                "notes": "No image data. Using default threshold."
            }

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness  = float(np.mean(gray))
        contrast    = float(np.std(gray))
        sharpness   = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ---- Collect all adjustment deltas ----
        b_delta, b_note = ImageQualityAnalyzer._brightness_penalty(brightness)
        c_delta, c_note = ImageQualityAnalyzer._contrast_penalty(contrast)
        s_delta, s_note = ImageQualityAnalyzer._sharpness_penalty(sharpness)
        f_delta, f_note = ImageQualityAnalyzer._face_size_penalty(img_bgr)

        total_delta = b_delta + c_delta + s_delta + f_delta
        raw_threshold = ImageQualityAnalyzer.BASE_THRESHOLD + total_delta

        # Clamp to safe range
        recommended = float(np.clip(raw_threshold, ImageQualityAnalyzer.MIN_THRESHOLD, ImageQualityAnalyzer.MAX_THRESHOLD))

        condition = ImageQualityAnalyzer._infer_condition_label(brightness, contrast, sharpness)

        notes = " | ".join(n for n in [b_note, c_note, s_note, f_note] if n)
        if not notes:
            notes = "Optimal conditions. No adjustments needed."

        # Confidence label based on magnitude of total adjustment
        abs_delta = abs(total_delta)
        if abs_delta < 0.03:
            confidence = "high"
        elif abs_delta < 0.08:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "brightness":            round(brightness, 1),
            "contrast":              round(contrast, 1),
            "sharpness":             round(sharpness, 1),
            "condition":             condition,
            "recommended_threshold": round(recommended, 3),
            "confidence":            confidence,
            "delta_breakdown": {
                "brightness_delta": b_delta,
                "contrast_delta":   c_delta,
                "sharpness_delta":  s_delta,
                "face_size_delta":  f_delta,
                "total_delta":      round(total_delta, 3),
            },
            "notes": notes
        }
