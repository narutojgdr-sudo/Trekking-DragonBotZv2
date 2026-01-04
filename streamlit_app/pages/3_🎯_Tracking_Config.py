"""Tracking configuration page."""
import streamlit as st
from utils.config_manager import ConfigManager

st.set_page_config(
    page_title="Tracking Config",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}

st.title("🎯 Tracking Configuration")
st.markdown("Configure multi-object tracking parameters")
st.markdown("---")

# Load current config or use defaults
config = st.session_state.current_config
if 'tracking' not in config:
    config['tracking'] = {
        "lost_timeout": 3.0,
        "min_frames_for_confirm": 4,
        "association_max_distance": 250,
        "ema_alpha": 0.35,
        "grace_frames": 20,
        "score_window": 10,
        "max_tracks": 8
    }

tracking = config['tracking']

# Quick presets
st.header("Quick Presets")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏆 Competition Optimized", use_container_width=True):
        preset = ConfigManager.get_preset("Competition Optimized")
        if preset and 'tracking' in preset:
            config['tracking'].update(preset['tracking'])
            st.success("Loaded Competition preset")
            st.rerun()

with col2:
    if st.button("🔓 Permissive", use_container_width=True):
        preset = ConfigManager.get_preset("Permissive")
        if preset and 'tracking' in preset:
            config['tracking'].update(preset['tracking'])
            st.success("Loaded Permissive preset")
            st.rerun()

with col3:
    if st.button("🎯 Strict", use_container_width=True):
        preset = ConfigManager.get_preset("Strict")
        if preset and 'tracking' in preset:
            config['tracking'].update(preset['tracking'])
            st.success("Loaded Strict preset")
            st.rerun()

st.markdown("---")

# Critical parameters section
st.header("⚠️ Critical Parameters")
st.markdown("*These parameters have the biggest impact on tracking performance*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lost Timeout")
    lost_timeout = st.slider(
        "Track Lifetime (seconds)",
        min_value=0.1,
        max_value=10.0,
        value=float(tracking.get('lost_timeout', 3.0)),
        step=0.1,
        help="How long a track survives without detections before being deleted"
    )
    
    if lost_timeout < 1.0:
        st.warning("⚠️ **Warning**: Value < 1.0s may cause tracks to be deleted too quickly!")
        st.info("💡 **Tip**: If IDs keep increasing, try 2.0-3.0s")
    elif lost_timeout > 5.0:
        st.info("ℹ️ High values may keep old tracks too long")
    
    tracking['lost_timeout'] = lost_timeout
    
    # Show frame equivalent at 30fps
    frames = int(lost_timeout * 30)
    st.caption(f"≈ {frames} frames at 30 FPS")

with col2:
    st.subheader("Association Distance")
    association_max_distance = st.slider(
        "Max Distance (pixels)",
        min_value=50,
        max_value=500,
        value=int(tracking.get('association_max_distance', 250)),
        step=10,
        help="Maximum pixel distance to associate a detection with an existing track"
    )
    
    if association_max_distance < 150:
        st.warning("⚠️ **Warning**: Low values may lose fast-moving cones!")
        st.info("💡 **Tip**: For competition, try 200-300px")
    elif association_max_distance > 350:
        st.info("ℹ️ Very high values may cause incorrect associations")
    
    tracking['association_max_distance'] = association_max_distance

st.markdown("---")

# Confirmation parameters
st.header("Confirmation Settings")
st.markdown("*Control when tracks transition from SUSPECT to CONFIRMED*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Frames to Confirm")
    min_frames_for_confirm = st.slider(
        "Min Frames Required",
        min_value=1,
        max_value=20,
        value=int(tracking.get('min_frames_for_confirm', 4)),
        help="Number of good detections needed before confirming a track"
    )
    
    if min_frames_for_confirm < 3:
        st.warning("⚠️ **Warning**: Low values may confirm false positives quickly!")
    elif min_frames_for_confirm > 8:
        st.info("ℹ️ High values may delay valid cone confirmation")
    
    tracking['min_frames_for_confirm'] = min_frames_for_confirm

with col2:
    st.subheader("Grace Period")
    grace_frames = st.slider(
        "Grace Frames",
        min_value=1,
        max_value=100,
        value=int(tracking.get('grace_frames', 20)),
        help="Frames a CONFIRMED track can miss before reverting to SUSPECT"
    )
    
    if grace_frames < 10:
        st.info("ℹ️ Low grace period makes tracks more fragile")
    
    tracking['grace_frames'] = grace_frames

st.markdown("---")

# Advanced parameters
with st.expander("🔧 Advanced Parameters"):
    st.subheader("Smoothing & Limits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ema_alpha = st.slider(
            "EMA Alpha (Smoothing)",
            min_value=0.0,
            max_value=1.0,
            value=float(tracking.get('ema_alpha', 0.35)),
            step=0.05,
            help="Exponential moving average weight. Higher = more responsive to changes."
        )
        tracking['ema_alpha'] = ema_alpha
        
        if ema_alpha < 0.2:
            st.info("ℹ️ Low alpha = very smooth but slow to react")
        elif ema_alpha > 0.5:
            st.info("ℹ️ High alpha = fast reactions but less smooth")
    
    with col2:
        max_tracks = st.number_input(
            "Max Simultaneous Tracks",
            min_value=1,
            max_value=20,
            value=int(tracking.get('max_tracks', 8)),
            help="Maximum number of cones to track simultaneously"
        )
        tracking['max_tracks'] = max_tracks
    
    score_window = st.slider(
        "Score Window Size",
        min_value=1,
        max_value=50,
        value=int(tracking.get('score_window', 10)),
        help="Number of recent frames to average for score calculation"
    )
    tracking['score_window'] = score_window

# Validation
st.markdown("---")
st.header("Validation")

errors = ConfigManager.validate_tracking_config(tracking)

if errors:
    for error in errors:
        st.error(f"❌ {error}")
else:
    st.success("✅ All tracking parameters are valid")

# Warnings
warnings = ConfigManager.get_warnings(config)
if warnings:
    st.subheader("⚠️ Recommendations")
    for warning in warnings:
        if 'tracking' in warning.lower():
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

# Diagnostic help
with st.expander("🩺 Diagnostic Guide"):
    st.markdown("""
    ### Common Problems & Solutions
    
    #### Problem: Track IDs Keep Increasing
    **Symptoms**: 
    - IDs like 541, 542, 543...
    - Tracks never reach CONFIRMED
    - Same cone gets new ID constantly
    
    **Cause**: Tracks are being deleted before confirmation
    
    **Solutions**:
    1. ✅ **Increase `lost_timeout`** to 2.0-3.0s
    2. ✅ **Decrease `min_frames_for_confirm`** to 3-4 frames
    3. ✅ **Increase `association_max_distance`** to 200-300px
    
    ---
    
    #### Problem: Tracks Lost During Fast Movement
    **Symptoms**:
    - Tracks disappear when cones move quickly
    - New tracks created instead of updating existing
    
    **Solutions**:
    1. ✅ **Increase `association_max_distance`** to 250-350px
    2. ✅ **Increase `grace_frames`** to 15-25
    3. Consider increasing `ema_alpha` for faster response
    
    ---
    
    #### Problem: False Positives Getting Confirmed
    **Symptoms**:
    - Non-cone objects tracked as CONFIRMED
    - Spurious detections persist
    
    **Solutions**:
    1. ✅ **Increase `min_frames_for_confirm`** to 6-8 frames
    2. Adjust color/geometry filters in other pages
    3. Consider decreasing `grace_frames`
    
    ---
    
    #### Problem: Tracks Too Jittery/Unstable
    **Symptoms**:
    - Track positions jump around
    - Bounding boxes flicker
    
    **Solutions**:
    1. ✅ **Decrease `ema_alpha`** to 0.20-0.30 for smoother tracking
    2. Ensure detections are stable (check color tuning)
    
    ---
    
    ### Parameter Recommendations by Scenario
    
    **🏆 Competition (Fast Detection, Stable):**
    ```
    lost_timeout: 3.0s
    min_frames_for_confirm: 4
    association_max_distance: 250px
    ema_alpha: 0.35
    grace_frames: 20
    ```
    
    **🐌 Slow Moving / Static:**
    ```
    lost_timeout: 5.0s
    min_frames_for_confirm: 6
    association_max_distance: 150px
    ema_alpha: 0.25
    grace_frames: 30
    ```
    
    **⚡ Fast Moving / Racing:**
    ```
    lost_timeout: 2.0s
    min_frames_for_confirm: 3
    association_max_distance: 350px
    ema_alpha: 0.45
    grace_frames: 15
    ```
    
    **🔬 Testing / Debug:**
    ```
    lost_timeout: 10.0s
    min_frames_for_confirm: 2
    association_max_distance: 400px
    ema_alpha: 0.40
    grace_frames: 50
    ```
    """)

# Parameter explanation
with st.expander("📚 Parameter Definitions"):
    st.markdown("""
    ### Detailed Parameter Explanations
    
    **lost_timeout** (seconds):
    - How long a track can go without being matched to a detection before deletion
    - Too low: Tracks deleted during temporary occlusion/missed detection
    - Too high: Old tracks linger unnecessarily
    - Recommended: 2.0-3.0s for most applications
    
    **association_max_distance** (pixels):
    - Maximum distance between a detection and track centroid to associate them
    - Too low: Misses fast-moving cones, creates new tracks
    - Too high: May incorrectly associate different cones
    - Recommended: 200-300px for competition
    
    **min_frames_for_confirm** (frames):
    - Number of good detections required before SUSPECT → CONFIRMED transition
    - Too low: False positives confirmed quickly
    - Too high: Valid cones take long to confirm
    - Recommended: 4-6 frames
    
    **ema_alpha** (0.0 - 1.0):
    - Weight for exponential moving average smoothing
    - Formula: `new_pos = alpha * detection + (1-alpha) * old_pos`
    - Low (0.1-0.3): Very smooth, slow to react
    - High (0.4-0.6): Fast reactions, less smooth
    - Recommended: 0.30-0.40
    
    **grace_frames** (frames):
    - How many frames a CONFIRMED track can miss detections before reverting to SUSPECT
    - Provides resilience against temporary detection failures
    - Recommended: 15-25 frames
    
    **score_window** (frames):
    - Number of recent frames used for rolling average score
    - Larger window = smoother score, slower to react to changes
    - Recommended: 8-12 frames
    
    **max_tracks** (count):
    - Hard limit on simultaneous tracks
    - Prevents runaway track creation
    - Recommended: 8-12 for cone tracking
    """)
