#!/usr/bin/env python3
"""Main application for cone tracking."""
import logging
import math
import os
import time

import cv2

from .config import load_config, save_config, watch_config
from .detector import ConeDetector
from .tracker import MultiConeTracker
from .utils import ConeState
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

    def _debug_print_heading(self, tracks: list, frame_w: int):
        """
        Print human-friendly heading/steering debug info to terminal.
        
        Args:
            tracks: List of Track objects to debug print
            frame_w: Frame width in pixels (for center calculation)
        """
        # Check if debug printing is enabled
        if not self.config["debug"].get("print_heading", False):
            return
        
        # Get HFOV from config with fallback to 70 degrees
        hfov_deg = self.config["camera"].get("hfov_deg", 70.0)
        
        # Validate HFOV is in a reasonable range (10-170 degrees)
        if hfov_deg < 10.0 or hfov_deg > 170.0:
            logger.warning(f"Invalid hfov_deg={hfov_deg}, using fallback of 70.0")
            hfov_deg = 70.0
        
        hfov_rad = math.radians(hfov_deg)
        
        # Compute focal length using pinhole model: focal = (width/2) / tan(hfov/2)
        focal_px = (frame_w / 2.0) / math.tan(hfov_rad / 2.0)
        
        # Get optional cone height for distance estimation
        cone_height_m = self.config["debug"].get("cone_height_m", None)
        
        # Frame center
        center_x = frame_w / 2.0
        
        if not tracks:
            # No tracks detected
            logger.info("HEADING_DBG: detected=False")
            return
        
        # Process each track
        for track in tracks:
            # Calculate horizontal error
            err_px = track.cx - center_x
            
            # Convert pixel error to angle using the horizontal field of view
            # Using atan2 for proper handling of signs (equivalent to atan(err_px/focal_px) since focal_px > 0)
            angle_rad = math.atan2(err_px, focal_px)
            angle_deg = math.degrees(angle_rad)
            
            # Build log message
            msg = f"HEADING_DBG: detected=True id={track.track_id} cx={track.cx:.1f} err_px={err_px:+.1f} err_deg={angle_deg:+.2f} bbox_h={int(track.h)}"
            
            # Optionally estimate distance if cone height is provided
            # Use threshold of 1.0 pixel to avoid unrealistic distance calculations
            if cone_height_m is not None and track.h > 1.0:
                # Pinhole model: distance = (real_height * focal_length) / pixel_height
                est_dist_m = (cone_height_m * focal_px) / track.h
                msg += f" est_dist={est_dist_m:.2f}m"
            
            # Add average score
            msg += f" avg_score={track.avg_score():.2f}"
            
            logger.info(msg)

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
        
        # Setup video writer if output path is configured
        video_writer = None
        output_path = cam.get("output_video_path", "")
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
            fps_out = cam.get("fps", 30)  # Use camera FPS setting
            size = (cam["process_width"], cam["process_height"])  # Tamanho do frame processado
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_out, size)
            if video_writer.isOpened():
                logger.info(f"💾 Salvando vídeo processado em: {output_path}")
            else:
                logger.warning(f"⚠️ Não foi possível criar arquivo de vídeo: {output_path}")
                video_writer = None

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
                
                # Log rejections if configured
                if rejects and self.config["debug"].get("log_rejections", False):
                    logger.info(f"🔴 Frame com {len(rejects)} rejeições:")
                    for bbox, reason in rejects[:5]:  # Show up to 5 rejections
                        logger.info(f"   ✗ {reason}")
                
                self.tracker.update(detections)
                
                # Debug print heading info for confirmed tracks
                tracks_for_control = self.tracker.confirmed_tracks()
                self._debug_print_heading(tracks_for_control, cam["process_width"])
                
                # Log suspects after tracking
                if self.config["debug"].get("log_suspects", False):
                    suspects = [t for t in self.tracker.tracks if t.state == ConeState.SUSPECT]
                    if suspects:
                        logger.info(f"🟡 Frame com {len(suspects)} suspects:")
                        for t in suspects[:5]:  # Show up to 5 suspects
                            logger.info(f"   ? ID {t.track_id}: frames={t.frames_seen}, avg={t.avg_score():.2f}")

                now = time.time()
                fps = 1.0 / (now - t_last + 1e-6)
                t_last = now

                # Only CONFIRMED tracks by default (cfg.draw_suspects controls)
                tracks_to_draw = self.tracker.tracks if self.config["debug"].get("draw_suspects", False) else self.tracker.confirmed_tracks()
                out = self.vis.draw(proc.copy(), tracks_to_draw, rejects, fps, self.config_reload_msg)
                
                # Salvar frame processado se video_writer estiver configurado
                if video_writer is not None:
                    video_writer.write(out)

                if self.config["debug"]["show_windows"]:
                    try:
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
                    except cv2.error as e:
                        logger.warning(f"⚠️ Não foi possível mostrar janelas (ambiente sem GUI): {e}")
                        logger.info("💡 Dica: Desabilite 'show_windows' no config ou use 'output_video_path'")
                        # Continue processing but stop trying to show windows
                        self.config["debug"]["show_windows"] = False
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                logger.info(f"✅ Vídeo processado salvo em: {output_path}")
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass  # Ignorar se GUI não estiver disponível
