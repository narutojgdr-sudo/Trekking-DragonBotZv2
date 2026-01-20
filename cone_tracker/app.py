#!/usr/bin/env python3
"""Main application for cone tracking."""
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from .utils import ConeState
from .run_csv_logger import RunCSVLogger
from .config import load_config, save_config, watch_config
from .detector import ConeDetector
from .tracker import MultiConeTracker
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

# Constants for debug heading calculations
MIN_BBOX_HEIGHT_FOR_DISTANCE = 10.0  # Minimum bbox height (px) to compute distance estimate
RUN_LOG_FLUSH_EVERY = 10
DEFAULT_RUN_LOG_FILENAME_PATTERN = "run_{source}_{start_ts}.jsonl"
RUN_LOG_MAX_FILENAME_ATTEMPTS = 1000
HFOV_MIN_DEG = 10.0
HFOV_MAX_DEG = 170.0
HFOV_FALLBACK_DEG = 70.0
MIN_FRAME_BGR = (1, 1, 3)
MIN_FRAME_GRAY = (1, 1)


@dataclass(frozen=True)
class SourceInfo:
    source_label: str
    using_video: bool
    video_path: str
    camera_index: Optional[int]
    conflict: bool
    video_missing: bool


# =========================
# APP
# =========================
class App:
    """Main application for cone detection and tracking."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config if config is not None else load_config()
        self.source_label = None
        self.using_video = False
        self.video_path = ""
        self.camera_index = None
        self.video_missing = False
        self.run_log_handle = None
        self.run_log_path = None
        self.run_start_ts = None
        self.frame_idx = 0
        self.csv_logger = RunCSVLogger()
        self._video_ts_fallback_logged = False
        self._playback_start_ts = None
        self._playback_frame_duration = None
        source_info = self._resolve_source(self.config)
        if source_info.conflict:
            msg = "CONFIG ERROR: Both camera.index and camera.video_path are set. Choose only one input source (camera or video)."
            logger.error(f"[source=unknown] {msg}")
            raise SystemExit(msg)
        self.source_label = source_info.source_label
        self.using_video = source_info.using_video
        self.video_path = source_info.video_path
        self.camera_index = source_info.camera_index
        self.video_missing = source_info.video_missing
        self.detector = ConeDetector(self.config)
        self.tracker = MultiConeTracker(self.config)
        self.vis = Visualizer(self.config)
        self.vis.source_label = self.source_label
        self.config_reload_msg = None
        self.config_reload_time = 0.0
        self._display_ready = False
        self._display_show_mask = False

    def _ensure_display_config(self) -> None:
        debug_cfg = self.config["debug"]
        if not debug_cfg.get("show_windows", False):
            debug_cfg["show_mask"] = False
            self._display_ready = False
            self._display_show_mask = False
            return
        if not os.environ.get("DISPLAY"):
            self._log_with_source(logging.WARNING, "DISPLAY not set; disabling GUI windows")
            debug_cfg["show_windows"] = False
            debug_cfg["show_mask"] = False
            self._display_ready = False
            self._display_show_mask = False
            return
        self._display_ready = True

    def _init_display(self) -> None:
        self._ensure_display_config()
        if not self._display_ready:
            return
        try:
            cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
            if self.config["debug"].get("show_mask", False):
                cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
            cv2.imshow("Tracker", np.zeros(MIN_FRAME_BGR, dtype=np.uint8))
            if self.config["debug"].get("show_mask", False):
                cv2.imshow("Mask", np.zeros(MIN_FRAME_GRAY, dtype=np.uint8))
            cv2.waitKey(1)
            self._display_show_mask = self.config["debug"].get("show_mask", False)
        except cv2.error as exc:
            self._log_with_source(logging.WARNING, f"⚠️ Unable to initialize GUI windows: {exc}")
            self.config["debug"]["show_windows"] = False
            self.config["debug"]["show_mask"] = False
            self._display_ready = False
            self._display_show_mask = False

    def _shutdown_display(self) -> None:
        if not self._display_ready:
            return
        for name in ("Mask", "Tracker"):
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
        self._display_ready = False
        self._display_show_mask = False

    def _refresh_display(self) -> None:
        was_ready = self._display_ready
        self._ensure_display_config()
        if was_ready and not self._display_ready:
            self._shutdown_display()
            return
        if self._display_ready and not was_ready:
            self._init_display()
            return
        if not self._display_ready:
            return
        show_mask = self.config["debug"].get("show_mask", False)
        if show_mask and not self._display_show_mask:
            try:
                cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Mask", np.zeros(MIN_FRAME_GRAY, dtype=np.uint8))
                cv2.waitKey(1)
                self._display_show_mask = True
            except cv2.error as exc:
                self._log_with_source(logging.WARNING, f"⚠️ Unable to show mask window: {exc}")
                self.config["debug"]["show_windows"] = False
                self.config["debug"]["show_mask"] = False
                self._shutdown_display()
        elif not show_mask and self._display_show_mask:
            try:
                cv2.destroyWindow("Mask")
            except cv2.error:
                pass
            self._display_show_mask = False

    def _resolve_source(self, config) -> SourceInfo:
        cam = config["camera"]
        video_path = cam.get("video_path", "") or ""
        camera_index = cam.get("index", None)
        index_is_set = self._is_camera_index_configured(camera_index)
        video_configured = bool(video_path)
        video_exists = video_configured and os.path.exists(video_path)
        conflict = video_exists and index_is_set
        video_missing = video_configured and not video_exists
        source_label = "video" if video_exists else "camera"
        return SourceInfo(
            source_label=source_label,
            using_video=video_exists,
            video_path=video_path,
            camera_index=camera_index,
            conflict=conflict,
            video_missing=video_missing,
        )

    @staticmethod
    def _is_camera_index_configured(camera_index: Optional[int]) -> bool:
        return camera_index is not None and isinstance(camera_index, int) and camera_index >= 0

    def _log_with_source(self, level: int, message: str) -> None:
        source = self.source_label or "unknown"
        logger.log(level, f"[source={source}] {message}")

    def _iso_timestamp(self) -> str:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return timestamp.replace("+00:00", "Z")

    def _sanitize_timestamp(self, timestamp: str) -> str:
        return timestamp.replace(":", "-").replace(".", "-")

    def _init_run_log(self) -> None:
        debug_cfg = self.config["debug"]
        run_log_dir = debug_cfg.get("run_log_dir", "logs")
        filename_pattern = debug_cfg.get("run_log_filename_pattern", DEFAULT_RUN_LOG_FILENAME_PATTERN)
        os.makedirs(run_log_dir, exist_ok=True)
        safe_ts = self._sanitize_timestamp(self.run_start_ts)
        filename = filename_pattern.format(source=self.source_label, start_ts=safe_ts)
        filename = os.path.basename(filename)
        if not filename:
            filename = f"run_{self.source_label}_{safe_ts}.jsonl"
        base, ext = os.path.splitext(filename)
        candidate = filename
        counter = 1
        while os.path.exists(os.path.join(run_log_dir, candidate)) and counter < RUN_LOG_MAX_FILENAME_ATTEMPTS:
            candidate = f"{base}_{counter}{ext}"
            counter += 1
        if counter >= RUN_LOG_MAX_FILENAME_ATTEMPTS and os.path.exists(os.path.join(run_log_dir, candidate)):
            candidate = f"{base}_{int(time.time() * 1000)}{ext}"
        self.run_log_path = os.path.join(run_log_dir, candidate)
        try:
            self.run_log_handle = open(self.run_log_path, "a", encoding="utf-8")
        except OSError as exc:
            self.run_log_handle = None
            self.run_log_path = None
            self._log_with_source(logging.ERROR, f"⚠️ Failed to open run log: {exc}")
            return
        self._log_with_source(logging.INFO, f"📝 Run log export enabled: {self.run_log_path}")

    def _write_run_log(self, record: dict) -> None:
        if not self.run_log_handle:
            return
        self.run_log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.frame_idx % RUN_LOG_FLUSH_EVERY == 0:
            self.run_log_handle.flush()

    def _focal_px(self, frame_w: int) -> float:
        hfov_deg = self.config["camera"].get("hfov_deg", HFOV_FALLBACK_DEG)
        if hfov_deg < HFOV_MIN_DEG or hfov_deg > HFOV_MAX_DEG:
            self._log_with_source(logging.WARNING, f"Invalid hfov_deg={hfov_deg}, using fallback of {HFOV_FALLBACK_DEG}")
            hfov_deg = HFOV_FALLBACK_DEG
        hfov_rad = math.radians(hfov_deg)
        return (frame_w / 2.0) / math.tan(hfov_rad / 2.0)

    def reload_config(self):
        """Reload configuration and reinitialize components."""
        self._log_with_source(logging.INFO, "⚙️ Recarregando configuração...")
        new_config = load_config()
        source_info = self._resolve_source(new_config)
        if source_info.conflict:
            msg = "CONFIG ERROR: Both camera.index and camera.video_path are set. Choose only one input source (camera or video)."
            self._log_with_source(logging.ERROR, f"{msg} Reload ignorado.")
            return
        if source_info.source_label != self.source_label:
            self._log_with_source(logging.WARNING, "Config mudou a fonte de entrada; reinicie o app para aplicar (mantendo a fonte atual).")
        if source_info.video_missing:
            self._log_with_source(logging.WARNING, "VIDEO PATH configured but file not found -> falling back to camera")
        self.config = new_config
        self.detector = ConeDetector(self.config)
        self.tracker = MultiConeTracker(self.config)
        self.vis = Visualizer(self.config)
        self.vis.source_label = self.source_label
        self._refresh_display()
        self.config_reload_msg = "⚙️ Config recarregada!"
        self.config_reload_time = time.time()
        self._log_with_source(logging.INFO, "✅ Configuração recarregada com sucesso!")

    def _debug_print_heading(self, tracks: list, frame_w: int, frame_idx: int):
        """
        Print human-friendly heading/steering debug info to terminal.
        
        Args:
            tracks: List of Track objects to debug print
            frame_w: Frame width in pixels (for center calculation)
            frame_idx: Current frame index
        """
        # Check if debug printing is enabled
        if not self.config["debug"].get("print_heading", False):
            return
        
        # Compute focal length using pinhole model: focal = (width/2) / tan(hfov/2)
        focal_px = self._focal_px(frame_w)
        
        # Get optional cone height for distance estimation
        cone_height_m = self.config["debug"].get("cone_height_m", None)
        
        # Frame center
        center_x = frame_w / 2.0
        timestamp = self._iso_timestamp()
        
        if not tracks:
            # No tracks detected
            logger.info(f"HEADING_DBG [source={self.source_label}] {timestamp} frame={frame_idx}: detected=False")
            return
        
        # Process each track
        for track in tracks:
            # Calculate horizontal error
            err_px = track.cx - center_x
            
            # Convert pixel error to angle using the horizontal field of view
            # angle = arctan(opposite/adjacent) = arctan(err_px / focal_px)
            angle_rad = math.atan(err_px / focal_px)
            angle_deg = math.degrees(angle_rad)
            
            # Build log message
            msg = f"HEADING_DBG [source={self.source_label}] {timestamp} frame={frame_idx}: detected=True id={track.track_id} cx={track.cx:.1f} err_px={err_px:+.1f} err_deg={angle_deg:+.2f} bbox_h={int(track.h)}"
            
            # Optionally estimate distance if cone height is provided
            # Use threshold to avoid unrealistic distance calculations for very small bboxes
            if cone_height_m is not None and track.h > MIN_BBOX_HEIGHT_FOR_DISTANCE:
                # Pinhole model: distance = (real_height * focal_length) / pixel_height
                est_dist_m = (cone_height_m * focal_px) / track.h
                msg += f" est_dist={est_dist_m:.2f}m"
            
            # Add average score
            msg += f" avg_score={track.avg_score():.2f}"
            
            logger.info(msg)

    def run(self, max_frames: Optional[int] = None):
        """Run the main application loop."""
        cam = self.config["camera"]
        using_video = self.using_video
        video_path = self.video_path
        camera_index = self.camera_index if self.camera_index is not None else 0
        
        if self.video_missing:
            self._log_with_source(logging.WARNING, "VIDEO PATH configured but file not found -> falling back to camera")
        
        if using_video:
            cap = cv2.VideoCapture(video_path)
            self._log_with_source(logging.INFO, f"📹 Usando vídeo: {video_path}")
        else:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            self._log_with_source(logging.INFO, f"📷 Usando câmera: index {camera_index}")

        if not cap.isOpened():
            self._log_with_source(logging.ERROR, "Camera failed to open (cap.isOpened() == False)")
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
                self._log_with_source(logging.INFO, f"💾 Salvando vídeo processado em: {output_path}")
            else:
                self._log_with_source(logging.WARNING, f"⚠️ Não foi possível criar arquivo de vídeo: {output_path}")
                video_writer = None

        self.frame_idx = 0
        self.run_start_ts = self._iso_timestamp()
        if self.config["debug"].get("export_run_log", False):
            self._init_run_log()
        try:
            self.csv_logger.open_if_enabled(self.config, self.source_label, self.run_start_ts)
        except Exception as exc:
            self._log_with_source(logging.WARNING, f"⚠️ Failed to init CSV export: {exc}")
        self._init_display()

        t_last = time.time()
        fail_count = 0
        max_fail = int(cam.get("max_consecutive_read_failures", 120))
        if using_video and cam.get("playback_mode", "fast") == "realtime":
            fps_setting = cam.get("fps", 30)
            if fps_setting >= 1:
                self._playback_start_ts = time.time()
                self._playback_frame_duration = 1.0 / fps_setting
            else:
                self._log_with_source(logging.WARNING, f"Invalid fps for realtime playback: {fps_setting}")
        
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
                        self._log_with_source(logging.INFO, "🔄 Reiniciando vídeo...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            self._log_with_source(logging.ERROR, "Failed to restart video. Exiting.")
                            break
                    else:
                        # Camera read failure
                        fail_count += 1
                        if fail_count >= max_fail:
                            self._log_with_source(logging.ERROR, "Too many consecutive camera read failures. Exiting.")
                            break
                        time.sleep(0.01)
                        continue
                fail_count = 0

                proc = cv2.resize(frame, (cam["process_width"], cam["process_height"]))

                detections, mask, rejects = self.detector.detect(proc)
                
                # Log rejections if configured
                if rejects and self.config["debug"].get("log_rejections", False):
                    self._log_with_source(logging.INFO, f"🔴 Frame com {len(rejects)} rejeições:")
                    for bbox, reason in rejects[:5]:  # Show up to 5 rejections
                        self._log_with_source(logging.INFO, f"   ✗ {reason}")
                
                self.tracker.update(detections)
                
                # Debug print heading info for confirmed tracks
                tracks_for_control = self.tracker.confirmed_tracks()
                self._debug_print_heading(tracks_for_control, cam["process_width"], self.frame_idx)
                
                # Log suspects after tracking
                if self.config["debug"].get("log_suspects", False):
                    suspects = [t for t in self.tracker.tracks if t.state == ConeState.SUSPECT]
                    if suspects:
                        self._log_with_source(logging.INFO, f"🟡 Frame com {len(suspects)} suspects:")
                        for t in suspects[:5]:  # Show up to 5 suspects
                            self._log_with_source(logging.INFO, f"   ? ID {t.track_id}: frames={t.frames_seen}, avg={t.avg_score():.2f}")

                now = time.time()
                fps = 1.0 / (now - t_last + 1e-6)
                t_last = now
                ts_wallclock_ms = int(now * 1000)
                if self.run_log_handle or self.csv_logger.enabled:
                    ts_source_ms = ts_wallclock_ms
                    if using_video:
                        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        if pos_msec is not None and (pos_msec > 0 or self.frame_idx == 0):
                            ts_source_ms = int(pos_msec)
                        else:
                            fps_setting = cam.get("fps", 30)
                            if fps_setting > 0:
                                ts_source_ms = int(self.frame_idx * (1000.0 / fps_setting))
                                if not self._video_ts_fallback_logged:
                                    self._log_with_source(logging.WARNING, "CAP_PROP_POS_MSEC unavailable; using frame_idx-based timestamp")
                                    self._video_ts_fallback_logged = True
                            else:
                                ts_source_ms = ts_wallclock_ms
                                if not self._video_ts_fallback_logged:
                                    self._log_with_source(logging.WARNING, "CAP_PROP_POS_MSEC unavailable and fps invalid; using wallclock timestamp")
                                    self._video_ts_fallback_logged = True

                    if self.run_log_handle:
                        focal_px = self._focal_px(cam["process_width"])
                        center_x = cam["process_width"] / 2.0
                        cone_height_m = self.config["debug"].get("cone_height_m", None)
                        track_payload = []
                        for track in tracks_for_control:
                            err_px = track.cx - center_x
                            angle_deg = math.degrees(math.atan(err_px / focal_px))
                            est_dist_m = None
                            if cone_height_m is not None and track.h > MIN_BBOX_HEIGHT_FOR_DISTANCE:
                                est_dist_m = (cone_height_m * focal_px) / track.h
                            track_payload.append({
                                "id": track.track_id,
                                "bbox": list(track.bbox()),
                                "cx": float(track.cx),
                                "cy": float(track.cy),
                                "err_px": float(err_px),
                                "err_deg": float(angle_deg),
                                "bbox_h": int(track.h),
                                "est_dist_m": None if est_dist_m is None else float(est_dist_m),
                                "avg_score": float(track.avg_score()),
                            })

                        record = {
                            "run_start_ts": self.run_start_ts,
                            "frame_idx": self.frame_idx,
                            "ts_wallclock_ms": ts_wallclock_ms,
                            "ts_source_ms": ts_source_ms,
                            "source": self.source_label,
                            "detected": bool(tracks_for_control),
                            "selected_target_id": None,
                            "tracks": track_payload,
                            "rejects_count": len(rejects),
                            "fps": float(fps),
                        }
                        self._write_run_log(record)

                    if self.csv_logger.enabled:
                        try:
                            self.csv_logger.log_frame(
                                frame_idx=self.frame_idx,
                                frame_w=cam["process_width"],
                                tracks=tracks_for_control,
                                ts_wallclock_ms=ts_wallclock_ms,
                                ts_source_ms=ts_source_ms,
                                fps=fps,
                                hfov_deg=cam.get("hfov_deg", HFOV_FALLBACK_DEG),
                                cone_height_m=self.config["debug"].get("cone_height_m", None),
                            )
                        except Exception as exc:
                            self._log_with_source(logging.WARNING, f"⚠️ CSV log error: {exc}")

                # Only CONFIRMED tracks by default (cfg.draw_suspects controls)
                tracks_to_draw = self.tracker.tracks if self.config["debug"].get("draw_suspects", False) else self.tracker.confirmed_tracks()
                out = self.vis.draw(proc.copy(), tracks_to_draw, rejects, fps, self.config_reload_msg)
                
                # Salvar frame processado se video_writer estiver configurado
                if video_writer is not None:
                    video_writer.write(out)

                if self.config["debug"]["show_windows"]:
                    try:
                        self.vis.show(out, mask, self.config["debug"].get("show_mask", False))
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("q"):
                            break
                        if k == ord("s"):
                            save_config(self.config)
                        if k == ord("r"):
                            self.reload_config()
                    except cv2.error as e:
                        self._log_with_source(logging.WARNING, f"⚠️ Não foi possível mostrar janelas (ambiente sem GUI): {e}")
                        self._log_with_source(logging.INFO, "💡 Dica: Desabilite 'show_windows' no config ou use 'output_video_path'")
                        # Continue processing but stop trying to show windows
                        self.config["debug"]["show_windows"] = False
                        self.config["debug"]["show_mask"] = False
                        self._shutdown_display()
                elif self._display_ready:
                    self._shutdown_display()

                if using_video and self._playback_frame_duration:
                    target_time = self._playback_start_ts + (self.frame_idx + 1) * self._playback_frame_duration
                    sleep_time = target_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self.frame_idx += 1
                if max_frames is not None and self.frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                self._log_with_source(logging.INFO, f"✅ Vídeo processado salvo em: {output_path}")
            if self.run_log_handle:
                try:
                    self.run_log_handle.flush()
                    self.run_log_handle.close()
                    self._log_with_source(logging.INFO, f"✅ Run log exported to: {self.run_log_path}")
                except Exception:
                    self._log_with_source(logging.WARNING, "⚠️ Failed to close run log")
            if self.csv_logger.enabled:
                try:
                    self.csv_logger.close()
                    if self.csv_logger.csv_path:
                        self._log_with_source(logging.INFO, f"✅ CSV exported to: {self.csv_logger.csv_path}")
                except Exception:
                    self._log_with_source(logging.WARNING, "⚠️ Failed to close CSV export")
            try:
                self._shutdown_display()
            except (cv2.error, Exception):
                pass  # Ignorar se GUI não estiver disponível
