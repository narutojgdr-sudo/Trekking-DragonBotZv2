#!/usr/bin/env python3
"""Utility functions for cone detection."""
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


# =========================
# STATE / UTILS
# =========================
class ConeState(Enum):
    """Enum for cone tracking states."""
    SUSPECT = auto()
    CONFIRMED = auto()
    LOST = auto()


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value v between lo and hi."""
    return max(lo, min(hi, v))


def safe_roi(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract a safe ROI from image given a bounding box."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x = int(clamp(x, 0, w - 1))
    y = int(clamp(y, 0, h - 1))
    x2 = int(clamp(x + bw, 0, w))
    y2 = int(clamp(y + bh, 0, h))
    if x2 <= x or y2 <= y:
        return img[0:0, 0:0], (0, 0, 0, 0)
    return img[y:y2, x:x2], (x, y, x2 - x, y2 - y)


def x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate horizontal overlap ratio between two bounding boxes."""
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    if aw <= 0 or bw <= 0:
        return 0.0
    a1, a2 = ax, ax + aw
    b1, b2 = bx, bx + bw
    inter = max(0, min(a2, b2) - max(a1, b1))
    denom = float(min(aw, bw))
    if denom <= 0:
        return 0.0
    return inter / denom


def bbox_union(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Calculate the union bounding box of a list of boxes."""
    if not boxes:
        return None
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2 = [b[0] + b[2] for b in boxes]
    y2 = [b[1] + b[3] for b in boxes]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x2)
    y_max = max(y2)
    w = max(0, x_max - x_min)
    h = max(0, y_max - y_min)
    return (x_min, y_min, w, h)


def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box."""
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Euclidean distance between centers of two bounding boxes."""
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))
