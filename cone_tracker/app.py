#!/usr/bin/env python3
"""Main application for cone tracking."""
import logging
import os
import time

import cv2

from .config import load_config, save_config, watch_config
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
        self.config_reload_msg = None
        self.config_reload_time = 0.0

    def reload_config(self):
        """Reload configuration and reinitialize components."""
        logger.info("⚙️ Recarregando configuração...")
        self.config = load_config()
        self.detector = ConeDetector(self.config)
        self.tracker = MultiConeTracker(self.config)
        self.vis = Visualizer(self.config)
        self.config_reload_msg = "⚙️ Config recarregada!"
        self.config_reload_time = time.time()
        logger.info("✅ Configuração recarregada com sucesso!")

    def run(self):
        """Run the main application loop."""
        cam = self.config["camera"]
        
        # Check if video_path is configured and file exists
        video_path = cam.get("video_path", "")
        using_video = False
        
        if video_path and os.path.exists(video_path):
            # Use video file
            cap = cv2.VideoCapture(video_path)
            using_video = True
            logger.info(f"📹 Usando vídeo: {video_path}")
        elif video_path and not os.path.exists(video_path):
            # Video path specified but file doesn't exist - warn and fallback to camera
            logger.warning(f"⚠️  Vídeo não encontrado: {video_path}. Usando câmera como fallback.")
            cap = cv2.VideoCapture(cam["index"], cv2.CAP_V4L2)
            logger.info(f"📷 Usando câmera: index {cam['index']}")
        else:
            # No video path or empty - use camera
            cap = cv2.VideoCapture(cam["index"], cv2.CAP_V4L2)
            logger.info(f"📷 Usando câmera: index {cam['index']}")

        if not cap.isOpened():
            raise RuntimeError("Camera failed to open (cap.isOpened() == False)")

        # Only apply camera settings if not using video file
        if not using_video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["capture_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["capture_height"])
            cap.set(cv2.CAP_PROP_FPS, cam["fps"])

        t_last = time.time()
        fail_count = 0
        max_fail = int(cam.get("max_consecutive_read_failures", 120))
        
        # Config watch setup
        config_path = "cone_config.yaml"
        watch_config(config_path)  # Initialize watcher

        try:
            while True:
                # Check for config file changes
                if watch_config(config_path):
                    self.reload_config()
                
                # Clear reload message after 3 seconds
                if self.config_reload_msg and (time.time() - self.config_reload_time) > 3.0:
                    self.config_reload_msg = None
                
                ret, frame = cap.read()
                if not ret:
                    # If using video and reached end, restart it
                    if using_video:
                        logger.info("🔄 Reiniciando vídeo...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            logger.error("Failed to restart video. Exiting.")
                            break
                    else:
                        # Camera read failure
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

                # Only CONFIRMED tracks by default (cfg.draw_suspects controls)
                tracks_to_draw = self.tracker.tracks if self.config["debug"].get("draw_suspects", False) else self.tracker.confirmed_tracks()
                out = self.vis.draw(proc.copy(), tracks_to_draw, rejects, fps, self.config_reload_msg)

                if self.config["debug"]["show_windows"]:
                    cv2.imshow("Tracker", out)
                    if self.config["debug"]["show_mask"]:
                        cv2.imshow("Mask", mask)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("s"):
                        save_config(self.config)
                    if k == ord("r"):
                        self.reload_config()
        finally:
            cap.release()
            cv2.destroyAllWindows()
