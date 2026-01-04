#!/usr/bin/env python3
"""Visualization for cone detection and tracking."""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .tracker import Track
from .utils import ConeState


# =========================
# VISUALIZER
# =========================
class Visualizer:
    """Visualize cone detection and tracking results."""
    
    COLORS = [
        (0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (180, 180, 0), (0, 180, 180),
    ]

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["debug"]

    def _color(self, track_id: int) -> Tuple[int, int, int]:
        """Get color for track ID."""
        return self.COLORS[track_id % len(self.COLORS)]

    def draw(self, frame: np.ndarray, tracks: List[Track], rejects: List[Tuple[Tuple[int, int, int, int], str]], fps: float, config_reload_msg: str = None):
        """Draw tracking visualization on frame."""
        if not self.cfg["show_windows"]:
            return frame

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Tracks info
        confirmed_count = sum(1 for t in tracks if t.state == ConeState.CONFIRMED)
        cv2.putText(frame, f"Tracks: {len(tracks)} ({confirmed_count} confirmed)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Config reload message (if houver)
        if config_reload_msg:
            cv2.putText(frame, config_reload_msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # draw tracks
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

        # rejections (optional)
        if self.cfg.get("show_rejection_reason", False):
            for bbox, reason in rejects:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, reason, (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        return frame
