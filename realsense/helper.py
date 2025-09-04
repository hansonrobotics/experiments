import numpy as np
import cv2

from typing import List, Optional

def _l2_normalize(embedding):
    if embedding is None:
        return None
    emb = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm

def _cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    a = emb1.flatten().astype(np.float32)
    b = emb2.flatten().astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def _median_embedding(emb_list: List[np.ndarray]) -> Optional[np.ndarray]:
    if not emb_list:
        return None
    arr = np.stack([np.asarray(e, dtype=np.float32).flatten() for e in emb_list], axis=0)
    med = np.median(arr, axis=0)
    return _l2_normalize(med)

def _crop_is_valid(img, min_side=40, min_std=8.0):
    if img is None:
        return False
    h, w = img.shape[:2]
    if h < min_side or w < min_side:
        return False
    if img.size == 0:
        return False
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < min_std:
            return False
    except Exception:
        return False
    return True