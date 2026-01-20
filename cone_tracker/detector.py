#!/usr/bin/env python3
"""Cone detector implementation."""
import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .color_utils import gray_world, load_backproj_hist, mask_from_backproj, rg_chromaticity_mask
from .utils import bbox_union, clamp, safe_roi, x_overlap_ratio

logger = logging.getLogger(__name__)


# =========================
# DETECTOR (returns multiple candidates)
# =========================
class ConeDetector:
    """Cone detection using color-based segmentation and geometric validation."""
    
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
        """Preprocess BGR image to HSV with optional color normalization."""
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
            a = lab[:, :, 1]
            b = lab[:, :, 2]
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
        return np.where(combined > 0, 255, 0).astype(np.uint8)

    def _part_boxes(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find part boxes from mask."""
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
        """Build connected groups of parts."""
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
        """Add padding to bounding box."""
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
        """
        Detect cones in frame.
        
        Returns:
            Tuple of (results, mask, rejects) where:
            - results: List of (bbox, score, data) for detected cones
            - mask: Binary mask of detected regions
            - rejects: List of (bbox, reason) for rejected candidates
        """
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
