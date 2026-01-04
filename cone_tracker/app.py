#!/usr/bin/env python3
"""Main application for cone tracking."""
import logging
import time

import cv2

from .config import load_config, save_config
from .detector import ConeDetector
from .tracker import MultiConeTracker
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


# =========================
# APP
# =========================
class App:
    """Main application for cone detection and tracking."""
    
    def __init__(self):
        self.config = load_config()
        self.detector = ConeDetector(self.config)
        self.tracker = MultiConeTracker(self.config)
        self.vis = Visualizer(self.config)

    def run(self):
        """Run the main application loop."""
        cam = self.config["camera"]
        cap = cv2.VideoCapture(cam["index"], cv2.CAP_V4L2)

        if not cap.isOpened():
            raise RuntimeError("Camera failed to open (cap.isOpened() == False)")

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
                        logger.error("Too many consecutive camera read failures. Exiting.")
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
