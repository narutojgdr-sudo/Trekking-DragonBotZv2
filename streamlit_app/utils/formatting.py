"""
Formatting utilities for Streamlit dashboard
"""
from typing import Any


def format_metric(value: Any, unit: str = "", default: str = "N/A") -> str:
    """
    Format numeric value with thousands separator
    
    Args:
        value: Value to format (int, float, or None)
        unit: Unit to append (e.g., " px²", " ms", "%")
        default: Default value if not a number
        
    Returns:
        Formatted string
        
    Examples:
        >>> format_metric(1000, " px²")
        '1,000 px²'
        >>> format_metric(None, " px²")
        'N/A'
        >>> format_metric("invalid", "%")
        'N/A'
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Thousands separator + unit
        return f"{value:,}{unit}"
    return default


def safe_get(config: dict, *keys, default: Any = None) -> Any:
    """
    Safely navigate nested dictionary
    
    Args:
        config: Configuration dictionary
        *keys: Keys to navigate (e.g., 'detection', 'hsv', 'lower')
        default: Default value if key doesn't exist
        
    Returns:
        Found value or default
        
    Examples:
        >>> safe_get({'a': {'b': 1}}, 'a', 'b')
        1
        >>> safe_get({'a': {}}, 'a', 'b', default=0)
        0
    """
    result = config
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                return default
        else:
            return default
    return result
