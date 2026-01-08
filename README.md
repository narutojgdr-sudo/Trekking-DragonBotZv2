# Cone Tracker - Modular Cone Detection and Tracking System

## Overview

This project implements a real-time cone detection and tracking system using computer vision. The code has been modularized for better organization, maintainability, and reusability.

## Key Features

1. **Modular Design**: Clean separation of concerns with dedicated modules
2. **Multi-Object Tracking**: Track multiple cones simultaneously
3. **Robust Detection**: Multiple color spaces and validation criteria
4. **Hot-Reload Configuration**: Automatic detection and reload of configuration changes
5. **Real-time Processing**: Optimized for camera input
6. **State Management**: Track lifecycle (SUSPECT → CONFIRMED → LOST)
7. **Debug Logging**: Comprehensive logging for tracking diagnostics

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
  - `watch_config()`: Monitor configuration file for changes (hot-reload)

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

### Keyboard Controls

- **q**: Quit the application
- **s**: Save current configuration to `cone_config.yaml`
- **r**: Manually reload configuration from `cone_config.yaml`

### Hot-Reload Configuration

The system automatically detects changes to `cone_config.yaml` and reloads the configuration in real-time. This allows you to tune parameters without restarting the application.

**Setup:**
```bash
# Copy the example configuration
cp cone_config.yaml.example cone_config.yaml

# Run the application
python3 test7.py
```

**Using Hot-Reload:**

1. Run the application: `python3 test7.py`
2. Edit `cone_config.yaml` in another terminal/editor
3. Save the file
4. The system automatically detects the change and reloads
5. A message "⚙️ Config recarregada!" appears on screen for 3 seconds

You can also manually force a reload by pressing the **r** key.

### Using Video Files Instead of Camera

The system supports using pre-recorded video files as input instead of live camera feed. This is useful for testing, debugging, or processing recorded footage.

**Configuration:**

Edit your `cone_config.yaml` file and set the `video_path` parameter:

```yaml
camera:
  video_path: "videos/teste_cones.mp4"  # Path to your video file
  index: 0                               # Camera index (used as fallback)
  # ... other settings
```

**Behavior:**

1. **Video file exists**: The system will use the video file and display `"📹 Usando vídeo: {path}"`
2. **Video file doesn't exist**: The system shows a warning and automatically falls back to the camera
3. **Empty video_path**: The system uses the camera (default behavior)
4. **Video reaches end**: The video automatically loops from the beginning

**Important Notes:**

- Camera settings (`capture_width`, `capture_height`, `fps`) are **not applied** to video files (they use the video's native properties)
- Video files automatically loop when they reach the end
- The fallback to camera is automatic if the video file is not found
- Leave `video_path` empty (`""`) to use the camera directly

**Example:**

```yaml
camera:
  video_path: "videos/teste_cones.mp4"  # Use video file
  # video_path: ""                       # Use camera (empty or omitted)
  index: 0
  capture_width: 1280
  capture_height: 720
  fps: 30
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

- **camera**: Camera settings (resolution, FPS, device index, video file path)
- **debug**: Visualization options
- **hsv_orange**: HSV color thresholds for orange detection
- **morphology**: Morphological operation parameters
- **grouping**: Part grouping parameters
- **geometry**: Geometric validation thresholds
- **weights**: Score weighting for detection
- **tracking**: Tracker parameters (association distance, EMA alpha, etc.)
- **clahe**: Contrast enhancement settings
- **color**: Advanced color processing options

### Optimized Tracking Parameters

The default `cone_config.yaml` includes optimized parameters to prevent tracks from being deleted prematurely:

```yaml
tracking:
  lost_timeout: 3.0              # Track survives 3 seconds without detection (90 frames @ 30fps)
  association_max_distance: 250   # Larger distance for moving cones
  min_frames_for_confirm: 4       # Faster confirmation (4 frames instead of 6)
  ema_alpha: 0.35                # More responsive smoothing
  grace_frames: 20               # Longer grace period for confirmed tracks

geometry:
  confirm_avg_score: 0.50        # Lower threshold for confirmation
  min_frame_score: 0.30          # Accept detections with slightly lower scores

debug:
  draw_suspects: true            # Show SUSPECT tracks in yellow
  show_rejection_reason: true    # Display rejection reasons
```

### Key Parameter Explanations

- **lost_timeout**: How long (in seconds) a track can go without being matched to a detection before being deleted. **Important**: The previous value of 0.6s was too short, causing tracks to be deleted during momentary detection failures, leading to high track IDs and tracks never reaching CONFIRMED state.

- **association_max_distance**: Maximum pixel distance to associate a detection with an existing track. Larger values help with fast-moving cones.

- **min_frames_for_confirm**: Minimum number of frames with good detections before a track transitions from SUSPECT to CONFIRMED state.

- **grace_frames**: Number of frames a CONFIRMED track can miss detections before reverting to SUSPECT state.

## Logging and Debugging

The application includes comprehensive logging to help debug tracking issues:

### Enabling Logging

Logging is enabled by default in `test7.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Log Messages

The tracker logs important events:

- **Track Deletion**: When a track is deleted due to timeout
  ```
  🗑️  Track 5 DELETADO: frames=3, avg=0.58, idade=0.45s
  ```

- **Track Confirmation**: When a track transitions to CONFIRMED state
  ```
  ✅ Track 2 CONFIRMADO! frames=4, avg=0.62
  ```

- **Config Reload**: When configuration is reloaded
  ```
  ⚙️ Recarregando configuração...
  ✅ Configuração recarregada com sucesso!
  ```

### On-Screen Information

The visualization displays:
- **FPS**: Current frames per second
- **Track Count**: Total tracks and confirmed tracks (e.g., "Tracks: 3 (2 confirmed)")
- **Track Details**: ID, state (CONFIRMED/SUSPECT), and average score for each track
- **Config Reload Message**: Temporary message when config is reloaded

## Troubleshooting

### Problem: Tracks Never Reach CONFIRMED State

**Symptoms**: 
- Track IDs keep increasing (e.g., ID=541)
- Tracks shown as SUSPECT but never CONFIRMED
- Same cone gets different IDs constantly

**Cause**: `lost_timeout` is too short. Tracks are being deleted during momentary detection failures before accumulating enough frames for confirmation.

**Solution**: Increase `lost_timeout` in `cone_config.yaml`:
```yaml
tracking:
  lost_timeout: 3.0  # Instead of 0.6
```

### Problem: Tracks Not Associating to Moving Cones

**Symptoms**:
- New tracks created instead of updating existing ones
- Tracks "jump" between different cones

**Cause**: `association_max_distance` is too small for fast-moving cones.

**Solution**: Increase `association_max_distance`:
```yaml
tracking:
  association_max_distance: 250  # Instead of 140
```

### Problem: Tracks Take Too Long to Confirm

**Symptoms**:
- Cones detected but stay in SUSPECT state for many frames

**Solutions**:
1. Reduce `min_frames_for_confirm`:
   ```yaml
   tracking:
     min_frames_for_confirm: 4  # Instead of 6
   ```

2. Lower the confirmation threshold:
   ```yaml
   geometry:
     confirm_avg_score: 0.50  # Instead of 0.55
   ```

### Problem: Too Many False Positives Confirmed

**Symptoms**:
- Non-cone objects being tracked as CONFIRMED

**Solutions**:
1. Increase `min_frames_for_confirm` to require more evidence
2. Increase `confirm_avg_score` to require higher quality detections
3. Adjust color and geometry thresholds to be more restrictive

## Dependencies

- Python 3.6+
- OpenCV (cv2)
- NumPy
- PyYAML

## Benefits of Modularization

1. **Maintainability**: Easier to understand and modify individual components
2. **Testability**: Each module can be tested independently
3. **Reusability**: Components can be used in other projects
4. **Scalability**: Easy to extend with new features
5. **Collaboration**: Multiple developers can work on different modules

## License

[Add your license information here]
