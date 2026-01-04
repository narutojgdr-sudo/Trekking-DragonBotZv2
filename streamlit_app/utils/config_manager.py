"""Config manager for validation and preset management."""
import copy
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration validation and presets."""
    
    # Preset configurations
    PRESETS = {
        "Competition Optimized": {
            "tracking": {
                "lost_timeout": 3.0,
                "min_frames_for_confirm": 4,
                "association_max_distance": 250,
                "ema_alpha": 0.35,
                "grace_frames": 20
            },
            "geometry": {
                "confirm_avg_score": 0.50,
                "min_frame_score": 0.30
            },
            "debug": {
                "draw_suspects": True,
                "show_rejection_reason": True
            }
        },
        "Bright Outdoor": {
            "hsv_orange": {
                "low_1": [0, 70, 120],
                "high_1": [25, 255, 255],
                "low_2": [160, 70, 100],
                "high_2": [179, 255, 255]
            },
            "tracking": {
                "lost_timeout": 3.0,
                "min_frames_for_confirm": 4,
                "association_max_distance": 250
            }
        },
        "Indoor/Shadows": {
            "hsv_orange": {
                "low_1": [0, 100, 60],
                "high_1": [30, 255, 200],
                "low_2": [160, 100, 50],
                "high_2": [179, 255, 200]
            },
            "tracking": {
                "lost_timeout": 2.5,
                "min_frames_for_confirm": 5
            }
        },
        "Permissive": {
            "geometry": {
                "min_frame_score": 0.25,
                "confirm_avg_score": 0.45,
                "min_fill_ratio": 0.06,
                "max_fill_ratio": 0.70
            },
            "tracking": {
                "min_frames_for_confirm": 3,
                "association_max_distance": 300
            }
        },
        "Strict": {
            "geometry": {
                "min_frame_score": 0.40,
                "confirm_avg_score": 0.60,
                "min_fill_ratio": 0.10,
                "max_fill_ratio": 0.60
            },
            "tracking": {
                "min_frames_for_confirm": 6,
                "association_max_distance": 150
            }
        }
    }
    
    @staticmethod
    def validate_hsv_value(value: int, component: str) -> Tuple[bool, str]:
        """
        Validate HSV value.
        
        Args:
            value: HSV component value
            component: Component name ('H', 'S', or 'V')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if component == 'H':
            if 0 <= value <= 179:
                return True, ""
            return False, f"H value must be between 0 and 179 (got {value})"
        else:  # S or V
            if 0 <= value <= 255:
                return True, ""
            return False, f"{component} value must be between 0 and 255 (got {value})"
    
    @staticmethod
    def validate_hsv_range(low: List[int], high: List[int]) -> Tuple[bool, str]:
        """
        Validate HSV range.
        
        Args:
            low: Lower HSV bound [H, S, V]
            high: Upper HSV bound [H, S, V]
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate individual values
        components = ['H', 'S', 'V']
        for i, comp in enumerate(components):
            valid, msg = ConfigManager.validate_hsv_value(low[i], comp)
            if not valid:
                return False, f"Low {msg}"
            valid, msg = ConfigManager.validate_hsv_value(high[i], comp)
            if not valid:
                return False, f"High {msg}"
        
        # Validate that low < high (except for H which can wrap)
        for i in range(1, 3):  # S and V
            if low[i] >= high[i]:
                return False, f"{components[i]} low must be less than high"
        
        return True, ""
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str) -> Tuple[bool, str]:
        """
        Validate value is within range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Parameter name for error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if min_val <= value <= max_val:
            return True, ""
        return False, f"{name} must be between {min_val} and {max_val} (got {value})"
    
    @staticmethod
    def validate_min_max(min_val: float, max_val: float, name: str) -> Tuple[bool, str]:
        """
        Validate min < max.
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            name: Parameter name for error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if min_val < max_val:
            return True, ""
        return False, f"{name} min must be less than max"
    
    @staticmethod
    def validate_tracking_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate tracking configuration.
        
        Args:
            config: Tracking config dict
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate individual parameters
        validations = [
            ('lost_timeout', 0.1, 10.0),
            ('min_frames_for_confirm', 1, 20),
            ('association_max_distance', 50, 500),
            ('ema_alpha', 0.0, 1.0),
            ('grace_frames', 1, 100),
            ('max_tracks', 1, 20)
        ]
        
        for param, min_val, max_val in validations:
            if param in config:
                valid, msg = ConfigManager.validate_range(
                    config[param], min_val, max_val, param
                )
                if not valid:
                    errors.append(msg)
        
        return errors
    
    @staticmethod
    def validate_geometry_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate geometry configuration.
        
        Args:
            config: Geometry config dict
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate area range
        if 'min_group_area' in config and 'max_group_area' in config:
            valid, msg = ConfigManager.validate_min_max(
                config['min_group_area'],
                config['max_group_area'],
                'group_area'
            )
            if not valid:
                errors.append(msg)
        
        # Validate aspect ratio
        if 'aspect_min' in config and 'aspect_max' in config:
            valid, msg = ConfigManager.validate_min_max(
                config['aspect_min'],
                config['aspect_max'],
                'aspect_ratio'
            )
            if not valid:
                errors.append(msg)
        
        # Validate fill ratio
        if 'min_fill_ratio' in config and 'max_fill_ratio' in config:
            valid, msg = ConfigManager.validate_min_max(
                config['min_fill_ratio'],
                config['max_fill_ratio'],
                'fill_ratio'
            )
            if not valid:
                errors.append(msg)
        
        # Validate scores
        score_params = ['min_frame_score', 'confirm_avg_score', 'min_profile_score']
        for param in score_params:
            if param in config:
                valid, msg = ConfigManager.validate_range(
                    config[param], 0.0, 1.0, param
                )
                if not valid:
                    errors.append(msg)
        
        return errors
    
    @staticmethod
    def get_preset(preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get preset configuration by name.
        
        Args:
            preset_name: Name of preset
            
        Returns:
            Preset config dict or None if not found
        """
        return copy.deepcopy(ConfigManager.PRESETS.get(preset_name))
    
    @staticmethod
    def list_presets() -> List[str]:
        """Get list of available preset names."""
        return list(ConfigManager.PRESETS.keys())
    
    @staticmethod
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        out = copy.deepcopy(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ConfigManager.deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    
    @staticmethod
    def get_warnings(config: Dict[str, Any]) -> List[str]:
        """
        Get warnings for potentially problematic config values.
        
        Args:
            config: Full config dict
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check tracking parameters
        if 'tracking' in config:
            tracking = config['tracking']
            
            if tracking.get('lost_timeout', 0) < 1.0:
                warnings.append(
                    "⚠️ lost_timeout < 1.0s: May cause premature track deletion. "
                    "Recommended: 2.0-3.0s for stable tracking."
                )
            
            if tracking.get('min_frames_for_confirm', 0) < 3:
                warnings.append(
                    "⚠️ min_frames_for_confirm < 3: May confirm false positives quickly. "
                    "Recommended: 4-6 frames."
                )
            
            if tracking.get('association_max_distance', 0) < 100:
                warnings.append(
                    "⚠️ association_max_distance < 100px: May lose fast-moving cones. "
                    "Recommended: 200-300px for competition."
                )
        
        # Check geometry parameters
        if 'geometry' in config:
            geometry = config['geometry']
            
            if geometry.get('confirm_avg_score', 1.0) > 0.60:
                warnings.append(
                    "⚠️ confirm_avg_score > 0.60: May be too strict. "
                    "Recommended: 0.45-0.55 for balance."
                )
            
            if geometry.get('min_frame_score', 1.0) > 0.40:
                warnings.append(
                    "⚠️ min_frame_score > 0.40: May reject good detections. "
                    "Recommended: 0.30-0.35."
                )
        
        return warnings
