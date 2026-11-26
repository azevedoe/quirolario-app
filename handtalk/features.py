"""Features do hand-talk (68 dims): landmarks normalizados + ângulos dos dedos.

Cópia de inferência alinhada a recognition/features.py do hand-talk — não altera
esse repositório. MediaPipe usa hand_landmarker.task na raiz do libras-app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RAIZ = Path(__file__).resolve().parent.parent
MODEL_PATH = RAIZ / "hand_landmarker.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

NUM_LANDMARKS = 21
NUM_LANDMARK_FEATURES = NUM_LANDMARKS * 3
_WRIST_IDX = 0
_MIDDLE_MCP_IDX = 9

FINGER_JOINTS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
FINGER_NAMES = list(FINGER_JOINTS.keys())
NUM_FINGER_FEATURES = len(FINGER_NAMES)
FEATURE_VECTOR_SIZE = NUM_LANDMARK_FEATURES + NUM_FINGER_FEATURES  # 68

_landmarker: Optional[vision.HandLandmarker] = None


def _garantir_modelo() -> Path:
    if not MODEL_PATH.exists():
        import urllib.request
        urllib.request.urlretrieve(URL_MODELO, MODEL_PATH)
    return MODEL_PATH


def _get_landmarker() -> vision.HandLandmarker:
    global _landmarker
    if _landmarker is None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_garantir_modelo())),
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        _landmarker = vision.HandLandmarker.create_from_options(options)
    return _landmarker


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    wrist = raw[_WRIST_IDX]
    translated = raw - wrist
    scale = float(np.linalg.norm(translated[_MIDDLE_MCP_IDX]))
    if scale < 1e-6:
        scale = 1e-6
    return (translated / scale).flatten().astype(np.float32)


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def compute_finger_angles(raw: np.ndarray) -> np.ndarray:
    angles = [
        _angle_at_joint(raw[base], raw[mid], raw[tip])
        for base, mid, tip in FINGER_JOINTS.values()
    ]
    return np.array(angles, dtype=np.float32)


def _detect_hand(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    landmarker = _get_landmarker()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def extract_landmarks(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    raw = _detect_hand(frame_bgr)
    if raw is None:
        return None
    return np.concatenate([normalize_landmarks(raw), compute_finger_angles(raw)])
