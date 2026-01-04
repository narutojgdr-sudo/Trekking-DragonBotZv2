#!/usr/bin/env python3
"""Multi-object tracking for cone detection."""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import ConeState, bbox_distance


# =========================
# MULTI TRACKING
# =========================
@dataclass
class Track:
    """Represents a tracked cone object."""
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
        """Get bounding box as (x, y, w, h)."""
        return (int(self.cx - self.w / 2), int(self.cy - self.h / 2), int(self.w), int(self.h))

    def avg_score(self) -> float:
        """Get average score from history."""
        if not self.score_hist:
            return 0.0
        return float(np.mean(self.score_hist))

    def update(self, bbox: Tuple[int, int, int, int], score: float, alpha: float, score_window: int):
        """Update track with new detection."""
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
    """Multi-object tracker for cone detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["tracking"]
        self.geo = config["geometry"]
        self.tracks: List[Track] = []
        self.next_id = 0

    def _make_track(self, det_bbox: Tuple[int, int, int, int], det_score: float) -> Track:
        """Create a new track from detection."""
        t = Track(track_id=self.next_id, score_hist=deque(maxlen=int(self.cfg["score_window"])))
        self.next_id += 1
        t.update(det_bbox, det_score, alpha=float(self.cfg["ema_alpha"]), score_window=int(self.cfg["score_window"]))
        return t

    def _associate_greedy(self, detections: List[Tuple[Tuple[int, int, int, int], float, dict]]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Associate tracks->detections by distance (greedy).
        Returns:
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

        pairs.sort(key=lambda x: x[0])  # smallest distance first

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
        """Update tracker with new detections."""
        now = time.time()

        # 1) Expire old tracks
        alive = []
        for t in self.tracks:
            if now - t.last_seen <= float(self.cfg["lost_timeout"]):
                alive.append(t)
        self.tracks = alive

        # 2) Associate
        matches, unmatched_tracks, unmatched_dets = self._associate_greedy(detections)

        # 3) Update matched tracks
        for ti, di in matches.items():
            bbox, score, _data = detections[di]
            self.tracks[ti].update(bbox, score, alpha=float(self.cfg["ema_alpha"]), score_window=int(self.cfg["score_window"]))

        # 4) Update miss counter for unmatched tracks (for grace period)
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

        # 5) Create tracks for unmatched detections
        for di in unmatched_dets:
            if len(self.tracks) >= int(self.cfg["max_tracks"]):
                break
            bbox, score, _data = detections[di]
            self.tracks.append(self._make_track(bbox, score))

        # 6) Decide state (CONFIRMED) by average
        for t in self.tracks:
            if len(t.score_hist) >= int(self.cfg["min_frames_for_confirm"]) and t.avg_score() >= float(self.geo["confirm_avg_score"]):
                t.state = ConeState.CONFIRMED
            else:
                if t.state != ConeState.CONFIRMED:
                    t.state = ConeState.SUSPECT

    def confirmed_tracks(self) -> List[Track]:
        """Get list of confirmed tracks."""
        min_age = int(self.cfg.get("min_confirmed_age_frames", 0))
        # min_age in frames is not stored; keeping it simple: return CONFIRMED directly
        return [t for t in self.tracks if t.state == ConeState.CONFIRMED]
