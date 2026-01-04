# Cone Tracker - Modular Cone Detection and Tracking System

## Overview

This project implements a real-time cone detection and tracking system using computer vision. The code has been modularized for better organization, maintainability, and reusability.

## Project Structure

```
.
├── test7.py                    # Main entry point
└── cone_tracker/               # Main package
    ├── __init__.py            # Package initialization and exports
    ├── app.py                 # Main application class
    ├── config.py              # Configuration management
    ├── detector.py            # Cone detection logic
    ├── tracker.py             # Multi-object tracking
    ├── visualizer.py          # Visualization and drawing
    ├── utils.py               # Utility functions (bbox operations, etc.)
    └── color_utils.py         # Color processing utilities
```

## Module Descriptions

### `config.py`
- **Purpose**: Configuration file management
- **Key Components**:
  - `DEFAULT_CONFIG`: Default configuration dictionary
  - `load_config()`: Load configuration from YAML file
  - `save_config()`: Save configuration to YAML file
  - `deep_merge()`: Deep merge configuration dictionaries

### `utils.py`
- **Purpose**: General utility functions
- **Key Components**:
  - `ConeState`: Enum for tracking states (SUSPECT, CONFIRMED, LOST)
  - `clamp()`: Clamp values between min/max
  - `safe_roi()`: Extract safe region of interest from image
  - `bbox_*()`: Bounding box operations (union, center, distance, overlap)

### `color_utils.py`
- **Purpose**: Color processing and enhancement
- **Key Components**:
  - `gray_world()`: Color normalization
  - `rg_chromaticity_mask()`: R/G chromaticity-based mask
  - `load_backproj_hist()`: Load histogram for backprojection
  - `mask_from_backproj()`: Create mask from histogram backprojection

### `detector.py`
- **Purpose**: Cone detection using color segmentation and geometric validation
- **Key Components**:
  - `ConeDetector`: Main detector class
    - Preprocessing (Gaussian blur, CLAHE)
    - Multi-mask generation (HSV, Lab, chromaticity, backprojection)
    - Part grouping and filtering
    - Geometric validation (area, aspect ratio, fill ratio, profile score)

### `tracker.py`
- **Purpose**: Multi-object tracking
- **Key Components**:
  - `Track`: Dataclass representing a tracked cone
  - `MultiConeTracker`: Multi-object tracker
    - Greedy association algorithm
    - Exponential moving average smoothing
    - State management (SUSPECT → CONFIRMED)
    - Grace period handling

### `visualizer.py`
- **Purpose**: Visualization of detection and tracking results
- **Key Components**:
  - `Visualizer`: Drawing class
    - Multi-color track rendering
    - FPS display
    - Rejection reason display (optional)

### `app.py`
- **Purpose**: Main application orchestration
- **Key Components**:
  - `App`: Main application class
    - Camera initialization
    - Main processing loop
    - Detector/tracker/visualizer integration
    - User input handling (quit, save config)

## Usage

### Basic Usage

Run the main script:

```bash
python3 test7.py
```

### As a Library

You can also import and use the modules programmatically:

```python
from cone_tracker import App, ConeDetector, MultiConeTracker, load_config

# Use the full application
app = App()
app.run()

# Or use individual components
config = load_config("my_config.yaml")
detector = ConeDetector(config)
tracker = MultiConeTracker(config)

# Process a frame
detections, mask, rejects = detector.detect(frame)
tracker.update(detections)
confirmed = tracker.confirmed_tracks()
```

## Configuration

The system uses YAML configuration files (`cone_config.yaml`). Key configuration sections:

- **camera**: Camera settings (resolution, FPS, device index)
- **debug**: Visualization options
- **hsv_orange**: HSV color thresholds for orange detection
- **morphology**: Morphological operation parameters
- **grouping**: Part grouping parameters
- **geometry**: Geometric validation thresholds
- **weights**: Score weighting for detection
- **tracking**: Tracker parameters (association distance, EMA alpha, etc.)
- **clahe**: Contrast enhancement settings
- **color**: Advanced color processing options

## Dependencies

- Python 3.6+
- OpenCV (cv2)
- NumPy
- PyYAML

## Key Features

1. **Modular Design**: Clean separation of concerns with dedicated modules
2. **Multi-Object Tracking**: Track multiple cones simultaneously
3. **Robust Detection**: Multiple color spaces and validation criteria
4. **Configurable**: Extensive YAML-based configuration
5. **Real-time Processing**: Optimized for camera input
6. **State Management**: Track lifecycle (SUSPECT → CONFIRMED → LOST)

## Benefits of Modularization

1. **Maintainability**: Easier to understand and modify individual components
2. **Testability**: Each module can be tested independently
3. **Reusability**: Components can be used in other projects
4. **Scalability**: Easy to extend with new features
5. **Collaboration**: Multiple developers can work on different modules

## License

[Add your license information here]
