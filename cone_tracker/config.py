#!/usr/bin/env python3
"""Configuration management for cone detection system."""
import logging
import os
from typing import Any, Dict

import yaml

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
    """Deep merge two dictionaries."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "cone_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file, merging with defaults."""
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
    """Save configuration to YAML file."""
    try:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config salva em {path}")
    except Exception as e:
        logger.exception(f"Falha ao salvar config '{path}': {e}")
