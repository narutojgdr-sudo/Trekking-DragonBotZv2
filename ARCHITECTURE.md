# Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        test7.py                              │
│                    (Main Entry Point)                        │
│                                                              │
│  from cone_tracker import App                                │
│  App().run()                                                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   cone_tracker Package                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │              app.py (App)                      │         │
│  │  - Main application loop                       │         │
│  │  - Camera initialization                       │         │
│  │  - Component orchestration                     │         │
│  └──┬────────────┬────────────┬───────────────────┘         │
│     │            │            │                              │
│     ▼            ▼            ▼                              │
│  ┌─────────┐ ┌──────────┐ ┌────────────┐                   │
│  │detector │ │ tracker  │ │visualizer  │                   │
│  │  .py    │ │   .py    │ │   .py      │                   │
│  └────┬────┘ └────┬─────┘ └────────────┘                   │
│       │           │                                          │
│       │           │                                          │
│       ▼           ▼                                          │
│  ┌──────────────────────┐                                   │
│  │     utils.py         │                                   │
│  │  - ConeState enum    │                                   │
│  │  - Bbox operations   │                                   │
│  │  - Helper functions  │                                   │
│  └──────────────────────┘                                   │
│       ▲                                                      │
│       │                                                      │
│  ┌────┴────────────┐                                        │
│  │ color_utils.py  │                                        │
│  │  - Gray world   │                                        │
│  │  - Chromaticity │                                        │
│  │  - Backprojection│                                       │
│  └─────────────────┘                                        │
│       ▲                                                      │
│       │                                                      │
│  ┌────┴────────────┐                                        │
│  │   config.py     │                                        │
│  │  - YAML loader  │                                        │
│  │  - Defaults     │                                        │
│  │  - Deep merge   │                                        │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Camera Frame
    │
    ▼
┌─────────────┐
│  Detector   │ ◄── Config
│             │ ◄── Color Utils
│  - Preprocess
│  - Get mask
│  - Group parts
│  - Validate
└──────┬──────┘
       │
       │ Detections [(bbox, score, data), ...]
       │
       ▼
┌─────────────┐
│   Tracker   │ ◄── Utils (bbox ops)
│             │
│  - Associate
│  - Update
│  - Confirm
└──────┬──────┘
       │
       │ Confirmed Tracks
       │
       ▼
┌─────────────┐
│ Visualizer  │
│             │
│  - Draw bbox
│  - Draw info
│  - Show FPS
└──────┬──────┘
       │
       ▼
  Display Frame
```

## Module Responsibilities

### config.py
- Manages configuration loading and saving
- Provides default configuration
- Deep merges user settings with defaults

### utils.py
- General-purpose utility functions
- Bounding box operations (union, distance, overlap)
- ROI extraction
- ConeState enumeration

### color_utils.py
- Color space processing
- Gray-world normalization
- R/G chromaticity masks
- Histogram backprojection

### detector.py
- Main cone detection logic
- Image preprocessing (blur, CLAHE)
- Multi-mask generation (HSV, Lab, chromaticity)
- Part grouping and geometric validation
- Profile scoring

### tracker.py
- Multi-object tracking implementation
- Greedy association algorithm
- Exponential moving average smoothing
- State management (SUSPECT → CONFIRMED)
- Track lifecycle management

### visualizer.py
- Visual output generation
- Color-coded track rendering
- FPS display
- Debug visualizations

### app.py
- Application orchestration
- Camera management
- Main processing loop
- User input handling
