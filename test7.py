#!/usr/bin/env python3
import cv2
import numpy as np
import time
import logging
import yaml
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from collections import deque

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "index": 0,
        "capture_width": 1920,
        "capture_height": 1080,
        "process_width": 960,
        "process_height": 540,
        "fps": 30,
        "backend": "v4l2",
        "max_consecutive_read_failures": 120,
    },
    "debug": {
        "show_windows": True,
        "show_mask": True,
        "show_rejection_reason": False,
        "draw_suspects": False,     # se True, desenha SUSPECT em amarelo também
        "show_groups": False,
    },
    "hsv_orange": {
        "low_1": [0, 90, 90],
        "high_1": [28, 255, 255],
        "low_2": [160, 90, 80],
        "high_2": [179, 255, 255],
    },
    "morphology": {
        "kernel_open": 3,
        "kernel_close": 7,
        "open_iterations": 1,
        "close_iterations": 1,
    },
    "grouping": {
        "min_part_area": 80,
        "max_y_gap": 80,
        "min_x_overlap_ratio": 0.20,
        "max_x_center_diff": 80,
        "pad_x": 8,
        "pad_y": 12,
    },
    "geometry": {
        "min_group_area": 1400,
        "max_group_area": 450000,
        "aspect_min": 1.0,
        "aspect_max": 6.0,
        "profile_slices": 10,
        "min_profile_score": 0.35,
        "min_fill_ratio": 0.08,
        "max_fill_ratio": 0.65,
        "min_frame_score": 0.35,
        "confirm_avg_score": 0.55,
    },
    "weights": {
        "profile": 0.50,
        "fill": 0.35,
        "aspect": 0.15,
    },
    "tracking": {
        "max_tracks": 8,              # MULTI: máximo de cones simultâneos
        "association_max_distance": 140,  # MULTI: distância máxima (px) para casar detecção->track

        "ema_alpha": 0.25,
        "lost_timeout": 0.6,
        "score_window": 10,
        "min_frames_for_confirm": 6,
        "grace_frames": 12,
        "grace_seconds": 0.0,
        "min_confirmed_age_frames": 0,  # opcional: exigir idade mínima para mostrar
    },
    "clahe": {"clip_limit": 1.8, "tile_grid_size": [8, 8]},
    # Novas opções de cor / robustez
    "color": {
        "enable_gray_world": True,
        "enable_lab_fallback": True,
        "lab_a_range": [140, 200],   # ajuste se necessário
        "lab_b_range": [130, 200],   # opcional
        "enable_rg": False,
        "rg_thresholds": {"r_min": 0.30, "r_max": 0.75, "g_min": 0.12, "g_max": 0.50},
        "enable_backproj": False,
        "backproj_hist_path": "cone_hist.npy",
        "backproj_thresh": 50
    }
}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str = "cone_config.yaml") -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    except Exception as e:
        logger.exception(f"Falha ao carregar config '{path}': {e}. Usando DEFAULT_CONFIG.")
        cfg = dict(DEFAULT_CONFIG)
    cfg = deep_merge(DEFAULT_CONFIG, cfg)
    return cfg

def save_config(config: Dict[str, Any], path: str = "cone_config.yaml") -> None:
    try:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config salva em {path}")
    except Exception as e:
        logger.exception(f"Falha ao salvar config '{path}': {e}")

# =========================
# STATE / UTILS
# =========================
class ConeState(Enum):
    SUSPECT = auto()
    CONFIRMED = auto()
    LOST = auto()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def safe_roi(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
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
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))

# =========================
# COLOR HELPERS
# =========================
def gray_world(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world color normalization."""
    img = bgr.astype(np.float32)
    avg = img.mean(axis=(0,1))
    avg_gray = float(np.mean(avg))
    # avoid division by zero
    scale = avg_gray / (avg + 1e-6)
    img = img * scale.reshape((1,1,3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def rg_chromaticity_mask(bgr: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Compute simple r,g chromaticity mask. thresholds values in [0..1]."""
    img = bgr.astype(np.float32)
    s = img.sum(axis=2) + 1e-6
    r = img[:,:,2] / s
    g = img[:,:,1] / s
    cond = (r >= thresholds["r_min"]) & (r <= thresholds["r_max"]) & (g >= thresholds["g_min"]) & (g <= thresholds["g_max"])
    return (cond.astype(np.uint8) * 255)

def load_backproj_hist(path: str):
    """Try to load a histogram saved as numpy .npy (HxW)."""
    if not path or not os.path.exists(path):
        return None
    try:
        hist = np.load(path)
        # ensure float32
        hist = hist.astype(np.float32)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist
    except Exception as e:
        logger.warning(f"Falha ao carregar histograma de backproj '{path}': {e}")
        return None

def mask_from_backproj(hsv: np.ndarray, hist: np.ndarray, thresh: int = 50) -> np.ndarray:
    """Given an HxS hist normalized 0..255, compute backprojection mask."""
    if hist is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    back = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)
    _, mask = cv2.threshold(back, thresh, 255, cv2.THRESH_BINARY)
    return mask

# =========================
# DETECTOR (mesmo pipeline, retorna múltiplos candidatos)
# =========================
class ConeDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        hsv = config["hsv_orange"]
        self.low_1 = np.array(hsv["low_1"])
        self.high_1 = np.array(hsv["high_1"])
        self.low_2 = np.array(hsv["low_2"])
        self.high_2 = np.array(hsv["high_2"])

        m = config["morphology"]
        self.k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (m["kernel_open"], m["kernel_open"]))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (m["kernel_close"], m["kernel_close"]))

        c = config["clahe"]
        self.clahe = cv2.createCLAHE(clipLimit=float(c["clip_limit"]), tileGridSize=tuple(c["tile_grid_size"]))

        # color options
        copt = config.get("color", {})
        self.enable_gray_world = bool(copt.get("enable_gray_world", False))
        self.enable_lab_fallback = bool(copt.get("enable_lab_fallback", False))
        self.lab_a_range = tuple(copt.get("lab_a_range", [140, 200]))
        self.lab_b_range = tuple(copt.get("lab_b_range", [130, 200]))
        self.enable_rg = bool(copt.get("enable_rg", False))
        self.rg_thresholds = copt.get("rg_thresholds", {"r_min": 0.30, "r_max": 0.75, "g_min": 0.12, "g_max": 0.50})
        self.enable_backproj = bool(copt.get("enable_backproj", False))
        self.backproj_hist_path = copt.get("backproj_hist_path", "")
        self.backproj_thresh = int(copt.get("backproj_thresh", 50))
        self.hist = load_backproj_hist(self.backproj_hist_path) if self.enable_backproj else None
        if self.hist is not None:
            logger.info(f"Backprojection hist loaded from {self.backproj_hist_path}")

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        # optional color normalization
        if self.enable_gray_world:
            bgr = gray_world(bgr)
        bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self.clahe.apply(v)
        return cv2.merge([h, s, v])

    def get_mask(self, bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """
        Combine multiple masks:
        - hsv threshold (base)
        - optional lab a/b threshold
        - optional rg chromaticity
        - optional backprojection (if hist available)
        Final mask = OR(all enabled)
        """
        # HSV mask
        m1 = cv2.inRange(hsv, self.low_1, self.high_1)
        m2 = cv2.inRange(hsv, self.low_2, self.high_2)
        mask_hsv = cv2.bitwise_or(m1, m2)

        masks = [mask_hsv]

        # Lab fallback
        if self.enable_lab_fallback:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
            a = lab[:,:,1]
            b = lab[:,:,2]
            a_low, a_high = self.lab_a_range
            b_low, b_high = self.lab_b_range
            mask_a = cv2.inRange(a, a_low, a_high)
            mask_b = cv2.inRange(b, b_low, b_high)
            mask_lab = cv2.bitwise_and(mask_a, mask_b)
            masks.append(mask_lab)

        # rg chromaticity
        if self.enable_rg:
            mask_rg = rg_chromaticity_mask(bgr, self.rg_thresholds)
            masks.append(mask_rg)

        # backprojection
        if self.enable_backproj and self.hist is not None:
            mask_bp = mask_from_backproj(hsv, self.hist, thresh=self.backproj_thresh)
            masks.append(mask_bp)

        # combine masks
        combined = masks[0].copy()
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        # morphological cleanup
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.k_open, iterations=self.config["morphology"]["open_iterations"])
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.k_close, iterations=self.config["morphology"]["close_iterations"])
        return combined

    def _part_boxes(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_part = self.config["grouping"]["min_part_area"]
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) < min_part:
                continue
            boxes.append(cv2.boundingRect(c))
        boxes.sort(key=lambda b: b[1])
        return boxes

    def _build_groups(self, boxes: List[Tuple[int, int, int, int]]) -> List[List[int]]:
        g = self.config["grouping"]
        n = len(boxes)
        if n == 0:
            return []
        adj = [[] for _ in range(n)]
        for i in range(n):
            ax, ay, aw, ah = boxes[i]
            if aw <= 0 or ah <= 0:
                continue
            acx = ax + aw / 2.0
            ay2 = ay + ah
            for j in range(i + 1, n):
                bx, by, bw, bh = boxes[j]
                if bw <= 0 or bh <= 0:
                    continue
                bcx = bx + bw / 2.0
                gap = by - ay2
                if gap > g["max_y_gap"]:
                    continue
                ov = x_overlap_ratio(boxes[i], boxes[j])
                centers_close = abs(acx - bcx) <= g["max_x_center_diff"]
                if ov >= g["min_x_overlap_ratio"] or centers_close:
                    adj[i].append(j)
                    adj[j].append(i)
        visited = [False] * n
        groups: List[List[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            groups.append(comp)
        return groups

    def _pad_bbox(self, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        g = self.config["grouping"]
        x, y, w, h = bbox
        pad_x = g["pad_x"]
        pad_y = g["pad_y"]
        x1 = int(clamp(x - pad_x, 0, frame_w - 1))
        y1 = int(clamp(y - pad_y, 0, frame_h - 1))
        x2 = int(clamp(x + w + pad_x, 0, frame_w))
        y2 = int(clamp(y + h + pad_y, 0, frame_h))
        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def _profile_score(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Compute profile score based on width (columns occupied) per horizontal slice.
        Returns float in [0,1].
        """
        roi, b = safe_roi(mask, bbox)
        _, _, w, h = b
        if w <= 0 or h <= 0:
            return 0.0

        n = max(4, int(self.config["geometry"]["profile_slices"]))
        sh = max(1, h // n)

        widths: List[float] = []
        for i in range(n):
            y1 = i * sh
            y2 = min(h, (i + 1) * sh)
            if y2 <= y1:
                continue
            sl = roi[y1:y2, :]  # shape (rows, cols)
            if sl.size == 0:
                widths.append(0.0)
                continue
            # count columns that have at least one non-zero pixel (occupied width)
            col_any = np.any(sl > 0, axis=0)
            width_px = int(np.count_nonzero(col_any))
            widths.append(width_px / float(w))   # normalized width 0..1 for the slice

        if len(widths) < 4:
            return 0.0

        # ignore very small slices
        nw = np.array(widths, dtype=np.float32)
        valid_mask = nw > 0.03
        valid = nw[valid_mask]
        if valid.size < 4:
            return 0.0

        # monotonicity heuristic: count how many consecutive slices do not shrink too much
        mono = 0.0
        denom = float(valid.size - 1)
        for i in range(valid.size - 1):
            if valid[i + 1] >= valid[i] * 0.85:
                mono += 1.0
            elif valid[i + 1] >= valid[i] * 0.70:
                mono += 0.5
        mono_score = mono / (denom + 1e-6)

        # top vs bottom average (last third vs first third)
        third = max(1, valid.size // 3)
        top_avg = float(np.mean(valid[:third]))
        bot_avg = float(np.mean(valid[-third:]))
        ratio = bot_avg / (top_avg + 1e-6)
        base_score = clamp((ratio - 1.0) / 1.2, 0.0, 1.0)

        return 0.55 * mono_score + 0.45 * base_score

    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[Tuple[int, int, int, int], float, dict]], np.ndarray, List[Tuple[Tuple[int, int, int, int], str]]]:
        hsv = self.preprocess(frame)
        mask = self.get_mask(frame, hsv)

        frame_h, frame_w = mask.shape[:2]
        boxes = self._part_boxes(mask)
        groups_idx = self._build_groups(boxes)

        geo = self.config["geometry"]
        wts = self.config["weights"]

        results: List[Tuple[Tuple[int, int, int, int], float, dict]] = []
        rejects: List[Tuple[Tuple[int, int, int, int], str]] = []

        for comp in groups_idx:
            comp_boxes = [boxes[i] for i in comp]
            ub = bbox_union(comp_boxes)
            if ub is None:
                continue
            group_bbox = self._pad_bbox(ub, frame_w, frame_h)
            x, y, w, h = group_bbox
            area = w * h

            if area < geo["min_group_area"] or area > geo["max_group_area"]:
                if self.config["debug"]["show_rejection_reason"]:
                    rejects.append((group_bbox, f"area={area}"))
                continue

            aspect = h / (w + 1e-6)
            if aspect < geo["aspect_min"] or aspect > geo["aspect_max"]:
                if self.config["debug"]["show_rejection_reason"]:
                    rejects.append((group_bbox, f"aspect={aspect:.2f}"))
                continue

            roi, _ = safe_roi(mask, group_bbox)
            fill = cv2.countNonZero(roi) / (area + 1e-6)
            if fill < geo["min_fill_ratio"] or fill > geo["max_fill_ratio"]:
                if self.config["debug"]["show_rejection_reason"]:
                    rejects.append((group_bbox, f"fill={fill:.2f}"))
                continue

            p = self._profile_score(mask, group_bbox)
            aspect_score = 1.0 - abs(aspect - 2.0) / 2.5
            aspect_score = clamp(aspect_score, 0.0, 1.0)

            score = (p * wts["profile"]) + (fill * wts["fill"]) + (aspect_score * wts["aspect"])
            if score < geo["min_frame_score"]:
                if self.config["debug"]["show_rejection_reason"]:
                    rejects.append((group_bbox, f"score={score:.2f}"))
                continue

            data = {"profile": p, "fill": fill, "aspect": aspect, "aspect_score": aspect_score, "score": score, "parts": len(comp_boxes)}
            results.append((group_bbox, score, data))

        results.sort(key=lambda x: x[1], reverse=True)
        return results, mask, rejects

# =========================
# MULTI TRACKING
# =========================
@dataclass
class Track:
    track_id: int
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0

    state: ConeState = ConeState.SUSPECT
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_good_time: float = field(default_factory=time.time)

    score_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    miss_counter: int = 0

    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(self.cx - self.w / 2), int(self.cy - self.h / 2), int(self.w), int(self.h))

    def avg_score(self) -> float:
        if not self.score_hist:
            return 0.0
        return float(np.mean(self.score_hist))

    def update(self, bbox: Tuple[int, int, int, int], score: float, alpha: float, score_window: int):
        if self.score_hist.maxlen != score_window:
            self.score_hist = deque(self.score_hist, maxlen=score_window)

        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        if self.w <= 0:
            self.cx, self.cy, self.w, self.h = cx, cy, w, h
        else:
            self.cx = alpha * cx + (1 - alpha) * self.cx
            self.cy = alpha * cy + (1 - alpha) * self.cy
            self.w = alpha * w + (1 - alpha) * self.w
            self.h = alpha * h + (1 - alpha) * self.h

        self.score_hist.append(float(score))
        now = time.time()
        self.last_seen = now
        self.last_good_time = now
        self.miss_counter = 0

class MultiConeTracker:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["tracking"]
        self.geo = config["geometry"]
        self.tracks: List[Track] = []
        self.next_id = 0

    def _make_track(self, det_bbox: Tuple[int, int, int, int], det_score: float) -> Track:
        t = Track(track_id=self.next_id, score_hist=deque(maxlen=int(self.cfg["score_window"])))
        self.next_id += 1
        t.update(det_bbox, det_score, alpha=float(self.cfg["ema_alpha"]), score_window=int(self.cfg["score_window"]))
        return t

    def _associate_greedy(self, detections: List[Tuple[Tuple[int, int, int, int], float, dict]]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Associa tracks->detections por distância (greedy).
        Retorna:
        - matches: {track_index: detection_index}
        - unmatched_tracks: [track_index]
        - unmatched_detections: [detection_index]
        """
        if not self.tracks:
            return {}, [], list(range(len(detections)))
        if not detections:
            return {}, list(range(len(self.tracks))), []

        max_dist = float(self.cfg["association_max_distance"])
        pairs = []
        for ti, trk in enumerate(self.tracks):
            tb = trk.bbox()
            for di, (db, _s, _d) in enumerate(detections):
                dist = bbox_distance(tb, db)
                if dist <= max_dist:
                    pairs.append((dist, ti, di))

        pairs.sort(key=lambda x: x[0])  # menor distância primeiro

        matched_tracks = set()
        matched_dets = set()
        matches: Dict[int, int] = {}

        for dist, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matches[ti] = di
            matched_tracks.add(ti)
            matched_dets.add(di)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float, dict]]):
        now = time.time()

        # 1) Expira tracks antigos
        alive = []
        for t in self.tracks:
            if now - t.last_seen <= float(self.cfg["lost_timeout"]):
                alive.append(t)
        self.tracks = alive

        # 2) Associa
        matches, unmatched_tracks, unmatched_dets = self._associate_greedy(detections)

        # 3) Atualiza os casados
        for ti, di in matches.items():
            bbox, score, _data = detections[di]
            self.tracks[ti].update(bbox, score, alpha=float(self.cfg["ema_alpha"]), score_window=int(self.cfg["score_window"]))

        # 4) Atualiza "miss" dos não casados (para grace)
        for ti in unmatched_tracks:
            t = self.tracks[ti]
            if t.state == ConeState.CONFIRMED:
                t.miss_counter += 1
                grace_seconds = float(self.cfg.get("grace_seconds", 0.0))
                if grace_seconds > 0.0:
                    if now - t.last_good_time > grace_seconds:
                        t.state = ConeState.SUSPECT
                        t.score_hist.clear()
                        t.miss_counter = 0
                else:
                    grace_frames = int(self.cfg.get("grace_frames", 12))
                    if t.miss_counter > grace_frames:
                        t.state = ConeState.SUSPECT
                        t.score_hist.clear()
                        t.miss_counter = 0

        # 5) Cria tracks para detecções não casadas
        for di in unmatched_dets:
            if len(self.tracks) >= int(self.cfg["max_tracks"]):
                break
            bbox, score, _data = detections[di]
            self.tracks.append(self._make_track(bbox, score))

        # 6) Decide estado (CONFIRMED) por média
        for t in self.tracks:
            if len(t.score_hist) >= int(self.cfg["min_frames_for_confirm"]) and t.avg_score() >= float(self.geo["confirm_avg_score"]):
                t.state = ConeState.CONFIRMED
            else:
                if t.state != ConeState.CONFIRMED:
                    t.state = ConeState.SUSPECT

    def confirmed_tracks(self) -> List[Track]:
        min_age = int(self.cfg.get("min_confirmed_age_frames", 0))
        # min_age em frames não é armazenado; mantendo simples: retorna CONFIRMED diretamente
        return [t for t in self.tracks if t.state == ConeState.CONFIRMED]

# =========================
# VISUALIZER
# =========================
class Visualizer:
    COLORS = [
        (0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (180, 180, 0), (0, 180, 180),
    ]

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["debug"]

    def _color(self, track_id: int) -> Tuple[int, int, int]:
        return self.COLORS[track_id % len(self.COLORS)]

    def draw(self, frame: np.ndarray, tracks: List[Track], rejects: List[Tuple[Tuple[int, int, int, int], str]], fps: float):
        if not self.cfg["show_windows"]:
            return frame

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # desenha tracks
        for t in tracks:
            if t.state == ConeState.CONFIRMED:
                color = self._color(t.track_id)
            else:
                if not self.cfg.get("draw_suspects", False):
                    continue
                color = (0, 255, 255)

            x, y, w, h = t.bbox()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID {t.track_id} {t.state.name} avg={t.avg_score():.2f}",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )

        # rejeições (opcional)
        if self.cfg.get("show_rejection_reason", False):
            for bbox, reason in rejects:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, reason, (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        return frame

# =========================
# APP
# =========================
class App:
    def __init__(self):
        self.config = load_config()
        self.detector = ConeDetector(self.config)
        self.tracker = MultiConeTracker(self.config)
        self.vis = Visualizer(self.config)

    def run(self):
        cam = self.config["camera"]
        cap = cv2.VideoCapture(cam["index"], cv2.CAP_V4L2)

        if not cap.isOpened():
            raise RuntimeError("Camera não abriu (cap.isOpened() == False)")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["capture_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["capture_height"])
        cap.set(cv2.CAP_PROP_FPS, cam["fps"])

        t_last = time.time()
        fail_count = 0
        max_fail = int(cam.get("max_consecutive_read_failures", 120))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count >= max_fail:
                        logger.error("Muitas falhas consecutivas de leitura da câmera. Saindo.")
                        break
                    time.sleep(0.01)
                    continue
                fail_count = 0

                proc = cv2.resize(frame, (cam["process_width"], cam["process_height"]))

                detections, mask, rejects = self.detector.detect(proc)
                self.tracker.update(detections)

                now = time.time()
                fps = 1.0 / (now - t_last + 1e-6)
                t_last = now

                # SOMENTE CONFIRMED por padrão (cfg.draw_suspects controla)
                tracks_to_draw = self.tracker.tracks if self.config["debug"].get("draw_suspects", False) else self.tracker.confirmed_tracks()
                out = self.vis.draw(proc.copy(), tracks_to_draw, rejects, fps)

                if self.config["debug"]["show_windows"]:
                    cv2.imshow("Tracker", out)
                    if self.config["debug"]["show_mask"]:
                        cv2.imshow("Mask", mask)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("s"):
                        save_config(self.config)
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    App().run()
