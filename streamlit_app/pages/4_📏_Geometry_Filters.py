"""Geometry filters configuration page."""
import streamlit as st
from utils.config_manager import ConfigManager

st.set_page_config(
    page_title="Geometry Filters",
    page_icon="📏",
    layout="wide"
)

# Initialize session state
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}

st.title("📏 Geometry Filters")
st.markdown("Configure geometric validation parameters for cone detection")
st.markdown("---")

# Load current config or use defaults
config = st.session_state.current_config
if 'geometry' not in config:
    config['geometry'] = {
        "min_group_area": 1400,
        "max_group_area": 450000,
        "aspect_min": 1.0,
        "aspect_max": 6.0,
        "min_fill_ratio": 0.08,
        "max_fill_ratio": 0.65,
        "min_frame_score": 0.35,
        "confirm_avg_score": 0.55,
        "profile_slices": 10,
        "min_profile_score": 0.35
    }

geometry = config['geometry']

# Quick presets
st.header("Quick Presets")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔓 Permissive Filters", use_container_width=True):
        preset = ConfigManager.get_preset("Permissive")
        if preset and 'geometry' in preset:
            config['geometry'].update(preset['geometry'])
            st.success("Loaded Permissive preset")
            st.rerun()

with col2:
    if st.button("🎯 Strict Filters", use_container_width=True):
        preset = ConfigManager.get_preset("Strict")
        if preset and 'geometry' in preset:
            config['geometry'].update(preset['geometry'])
            st.success("Loaded Strict preset")
            st.rerun()

st.markdown("---")

# Area filters
st.header("📐 Area Filters")
st.markdown("*Filter detections by bounding box area*")

col1, col2 = st.columns(2)

with col1:
    min_group_area = st.number_input(
        "Minimum Area (pixels²)",
        min_value=100,
        max_value=100000,
        value=int(geometry.get('min_group_area', 1400)),
        step=100,
        help="Minimum bounding box area. Filters out very small detections."
    )
    geometry['min_group_area'] = min_group_area

with col2:
    max_group_area = st.number_input(
        "Maximum Area (pixels²)",
        min_value=1000,
        max_value=1000000,
        value=int(geometry.get('max_group_area', 450000)),
        step=1000,
        help="Maximum bounding box area. Filters out very large detections."
    )
    geometry['max_group_area'] = max_group_area

# Validate min < max
if min_group_area >= max_group_area:
    st.error("❌ Minimum area must be less than maximum area!")

st.info(f"💡 Accepting detections between {min_group_area:,} and {max_group_area:,} pixels²")

st.markdown("---")

# Aspect ratio filters
st.header("📊 Aspect Ratio Filters")
st.markdown("*Filter by height/width ratio (cones are typically tall)*")

col1, col2 = st.columns(2)

with col1:
    aspect_min = st.slider(
        "Minimum Aspect Ratio",
        min_value=0.1,
        max_value=10.0,
        value=float(geometry.get('aspect_min', 1.0)),
        step=0.1,
        help="Minimum height/width ratio. Cones should be taller than wide."
    )
    geometry['aspect_min'] = aspect_min

with col2:
    aspect_max = st.slider(
        "Maximum Aspect Ratio",
        min_value=0.1,
        max_value=10.0,
        value=float(geometry.get('aspect_max', 6.0)),
        step=0.1,
        help="Maximum height/width ratio. Very thin detections are rejected."
    )
    geometry['aspect_max'] = aspect_max

# Validate min < max
if aspect_min >= aspect_max:
    st.error("❌ Minimum aspect ratio must be less than maximum!")
else:
    st.info(f"💡 Accepting aspect ratios between {aspect_min:.1f} and {aspect_max:.1f}")

# Visual guide
st.caption("Example: A cone 200px tall and 50px wide has aspect ratio = 4.0")

st.markdown("---")

# Fill ratio filters
st.header("🎨 Fill Ratio Filters")
st.markdown("*Filter by how much of the bounding box is filled with color*")

col1, col2 = st.columns(2)

with col1:
    min_fill_ratio = st.slider(
        "Minimum Fill Ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(geometry.get('min_fill_ratio', 0.08)),
        step=0.01,
        help="Minimum ratio of colored pixels to total bbox area. Too low = noisy detections."
    )
    geometry['min_fill_ratio'] = min_fill_ratio

with col2:
    max_fill_ratio = st.slider(
        "Maximum Fill Ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(geometry.get('max_fill_ratio', 0.65)),
        step=0.01,
        help="Maximum fill ratio. Too high may indicate blob instead of cone."
    )
    geometry['max_fill_ratio'] = max_fill_ratio

# Validate min < max
if min_fill_ratio >= max_fill_ratio:
    st.error("❌ Minimum fill ratio must be less than maximum!")
else:
    st.info(f"💡 Accepting fill ratios between {min_fill_ratio:.2f} and {max_fill_ratio:.2f}")

st.caption("Fill ratio = (colored pixels in bbox) / (total bbox area)")

st.markdown("---")

# Score thresholds
st.header("⭐ Score Thresholds")
st.markdown("*Quality scores for detection and confirmation*")

col1, col2 = st.columns(2)

with col1:
    min_frame_score = st.slider(
        "Minimum Frame Score",
        min_value=0.0,
        max_value=1.0,
        value=float(geometry.get('min_frame_score', 0.35)),
        step=0.05,
        help="Minimum score for a detection to be considered in a single frame."
    )
    
    if min_frame_score > 0.40:
        st.warning("⚠️ High threshold may reject good detections!")
    
    geometry['min_frame_score'] = min_frame_score

with col2:
    confirm_avg_score = st.slider(
        "Confirmation Average Score",
        min_value=0.0,
        max_value=1.0,
        value=float(geometry.get('confirm_avg_score', 0.55)),
        step=0.05,
        help="Average score required for track confirmation (SUSPECT → CONFIRMED)."
    )
    
    if confirm_avg_score > 0.60:
        st.warning("⚠️ High threshold may delay confirmation!")
    
    geometry['confirm_avg_score'] = confirm_avg_score

st.info("💡 Frame score must be met per-frame. Confirmation score is averaged over multiple frames.")

st.markdown("---")

# Advanced parameters
with st.expander("🔧 Advanced Parameters"):
    st.subheader("Profile Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        profile_slices = st.slider(
            "Profile Slices",
            min_value=5,
            max_value=20,
            value=int(geometry.get('profile_slices', 10)),
            help="Number of horizontal slices for cone profile analysis."
        )
        geometry['profile_slices'] = profile_slices
    
    with col2:
        min_profile_score = st.slider(
            "Min Profile Score",
            min_value=0.0,
            max_value=1.0,
            value=float(geometry.get('min_profile_score', 0.35)),
            step=0.05,
            help="Minimum score for cone profile shape validation."
        )
        geometry['min_profile_score'] = min_profile_score
    
    st.caption("Profile analysis checks if the detection has a cone-like tapered shape")

# Validation
st.markdown("---")
st.header("Validation")

errors = ConfigManager.validate_geometry_config(geometry)

if errors:
    for error in errors:
        st.error(f"❌ {error}")
else:
    st.success("✅ All geometry parameters are valid")

# Warnings
warnings = ConfigManager.get_warnings(config)
if warnings:
    st.subheader("⚠️ Recommendations")
    for warning in warnings:
        if 'geometry' in warning.lower() or 'score' in warning.lower():
            st.warning(warning)

# Save section
st.markdown("---")
st.header("Save Changes")

col1, col2 = st.columns([2, 1])

with col1:
    st.info("💡 Changes are saved to session. Use the Connection page to upload to remote server.")

with col2:
    if st.button("💾 Save to Session", type="primary", use_container_width=True):
        st.session_state.current_config = config
        st.success("✅ Saved to session!")

# Tuning guide
with st.expander("💡 Tuning Guide"):
    st.markdown("""
    ### Understanding Geometry Filters
    
    Geometry filters validate that detections have cone-like properties:
    
    1. **Area**: Cones should be a reasonable size
    2. **Aspect Ratio**: Cones are taller than they are wide
    3. **Fill Ratio**: Cones fill part of the bbox, but not all (due to tapered shape)
    4. **Profile**: Cones should taper from wide base to narrow top
    
    ---
    
    ### Common Tuning Scenarios
    
    #### Too Many False Positives (Non-Cones Detected)
    
    **Solutions**:
    1. ✅ **Increase `min_frame_score`** to 0.40-0.45
    2. ✅ **Increase `confirm_avg_score`** to 0.60-0.65
    3. ✅ **Narrow aspect ratio range** (e.g., 1.5-4.5)
    4. ✅ **Adjust fill ratio** to exclude blobs
    
    #### Missing Valid Cones
    
    **Solutions**:
    1. ✅ **Decrease `min_frame_score`** to 0.25-0.30
    2. ✅ **Decrease `confirm_avg_score`** to 0.45-0.50
    3. ✅ **Widen area range** (especially if cones appear small/large)
    4. ✅ **Widen aspect ratio range**
    5. Check color tuning (may need to widen HSV ranges)
    
    #### Detections at Wrong Distance
    
    **Too Close (Large in Image)**:
    - Increase `max_group_area`
    - May need to adjust `max_fill_ratio`
    
    **Too Far (Small in Image)**:
    - Decrease `min_group_area`
    - May need to adjust `min_fill_ratio`
    
    ---
    
    ### Parameter Recommendations by Application
    
    **🏆 Competition (Balanced):**
    ```
    min_group_area: 1400
    max_group_area: 450000
    aspect_min: 1.0
    aspect_max: 6.0
    min_fill_ratio: 0.08
    max_fill_ratio: 0.65
    min_frame_score: 0.30
    confirm_avg_score: 0.50
    ```
    
    **🔓 Permissive (Catch Everything):**
    ```
    min_group_area: 800
    max_group_area: 600000
    aspect_min: 0.8
    aspect_max: 8.0
    min_fill_ratio: 0.06
    max_fill_ratio: 0.70
    min_frame_score: 0.25
    confirm_avg_score: 0.45
    ```
    
    **🎯 Strict (High Precision):**
    ```
    min_group_area: 2000
    max_group_area: 400000
    aspect_min: 1.5
    aspect_max: 5.0
    min_fill_ratio: 0.10
    max_fill_ratio: 0.60
    min_frame_score: 0.40
    confirm_avg_score: 0.60
    ```
    
    ---
    
    ### How Scores Are Calculated
    
    Detection score is a weighted combination of:
    - **Profile Score** (50%): How cone-like is the tapered shape?
    - **Fill Score** (35%): Is fill ratio in acceptable range?
    - **Aspect Score** (15%): Is aspect ratio reasonable?
    
    Formula:
    ```
    score = 0.50*profile + 0.35*fill + 0.15*aspect
    ```
    
    A detection needs:
    - `score >= min_frame_score` to be accepted in that frame
    - `average_score >= confirm_avg_score` to be CONFIRMED (averaged over multiple frames)
    
    ---
    
    ### Visual Examples
    
    **Good Cone Detection** (score ~0.60):
    - Area: 5000 px² (in range)
    - Aspect: 3.5 (tall)
    - Fill: 0.35 (partial fill, tapered)
    - Profile: 0.75 (good taper)
    
    **False Positive - Blob** (score ~0.25):
    - Area: 3000 px² (in range)
    - Aspect: 1.1 (nearly square)
    - Fill: 0.80 (too full, no taper)
    - Profile: 0.20 (no cone shape)
    
    **False Positive - Line** (score ~0.30):
    - Area: 2000 px² (in range)
    - Aspect: 8.5 (too thin)
    - Fill: 0.15 (sparse)
    - Profile: 0.40 (somewhat linear)
    """)

# Parameter definitions
with st.expander("📚 Parameter Definitions"):
    st.markdown("""
    ### Detailed Parameter Explanations
    
    **min_group_area / max_group_area** (pixels²):
    - Total area of the bounding box around detection
    - Formula: `area = width * height`
    - Use to filter unrealistic sizes (too small = noise, too large = not a cone)
    - Depends on camera resolution and cone distance
    
    **aspect_min / aspect_max** (ratio):
    - Height divided by width of bounding box
    - Formula: `aspect = height / width`
    - Cones are vertical, so aspect > 1.0 is expected
    - Typical cone: 1.5 - 5.0 depending on viewing angle
    
    **min_fill_ratio / max_fill_ratio** (0.0 - 1.0):
    - Fraction of bbox filled with detected color
    - Formula: `fill = colored_pixels / (width * height)`
    - Cones taper, so fill < 1.0 expected
    - Too high = solid blob, too low = sparse noise
    
    **min_frame_score** (0.0 - 1.0):
    - Minimum composite score for detection in single frame
    - Detection rejected immediately if below threshold
    - Prevents bad detections from entering tracking system
    
    **confirm_avg_score** (0.0 - 1.0):
    - Average score required for track confirmation
    - Averaged over `score_window` recent frames (see Tracking Config)
    - Track stays in SUSPECT state until this threshold is met
    
    **profile_slices** (count):
    - Number of horizontal slices for profile analysis
    - Each slice measures width at that height
    - More slices = finer analysis, but slower
    
    **min_profile_score** (0.0 - 1.0):
    - How well the width profile matches a cone (wide at base, narrow at top)
    - Calculated from width measurements across slices
    - High score = good taper, low score = cylindrical or irregular
    """)
