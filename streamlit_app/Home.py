"""Main dashboard home page."""
import streamlit as st
from datetime import datetime
import sys
import os

# Add parent directory to path to import cone_tracker modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Cone Tracker Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'last_sync' not in st.session_state:
    st.session_state.last_sync = None

# Header
st.title("🚦 Cone Tracker Dashboard")
st.markdown("**Remote Configuration Manager for DragonBot Cone Detection System**")

# Connection status banner
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.session_state.connected:
        st.success("🟢 Connected to remote server")
    else:
        st.error("🔴 Not connected to remote server")

with col2:
    if st.session_state.last_sync:
        sync_time = st.session_state.last_sync.strftime('%H:%M:%S')
        st.info(f"🕐 Last sync: {sync_time}")

with col3:
    if st.session_state.current_config:
        st.info(f"📊 Config loaded")
    else:
        st.warning("⚠️ No config loaded")

st.markdown("---")

# Quick actions
st.header("⚡ Quick Actions")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🔌 Connection", use_container_width=True, type="primary"):
        st.switch_page("pages/1_🔌_Connection.py")

with col2:
    if st.button("🎨 Color Tuning", use_container_width=True):
        st.switch_page("pages/2_🎨_Color_Tuning.py")

with col3:
    if st.button("🎯 Tracking", use_container_width=True):
        st.switch_page("pages/3_🎯_Tracking_Config.py")

with col4:
    if st.button("📏 Geometry", use_container_width=True):
        st.switch_page("pages/4_📏_Geometry_Filters.py")

with col5:
    if st.button("⚡ Presets", use_container_width=True):
        st.switch_page("pages/5_⚡_Presets.py")

st.markdown("---")

# Current configuration overview
st.header("📊 Current Configuration Overview")

if st.session_state.current_config:
    config = st.session_state.current_config
    
    # Create metrics layout
    st.subheader("Key Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Tracking**")
        if 'tracking' in config:
            tracking = config['tracking']
            st.metric("Lost Timeout", f"{tracking.get('lost_timeout', 'N/A')}s", 
                     help="Track lifetime without detection")
            st.metric("Confirm Frames", tracking.get('min_frames_for_confirm', 'N/A'),
                     help="Frames needed for confirmation")
            st.metric("Max Distance", f"{tracking.get('association_max_distance', 'N/A')}px",
                     help="Association distance threshold")
        else:
            st.info("No tracking config")
    
    with col2:
        st.markdown("**📏 Geometry**")
        if 'geometry' in config:
            geometry = config['geometry']
            st.metric("Confirm Score", geometry.get('confirm_avg_score', 'N/A'),
                     help="Average score for confirmation")
            st.metric("Frame Score", geometry.get('min_frame_score', 'N/A'),
                     help="Per-frame minimum score")
            st.metric("Aspect Range", 
                     f"{geometry.get('aspect_min', 'N/A')}-{geometry.get('aspect_max', 'N/A')}",
                     help="Acceptable aspect ratios")
        else:
            st.info("No geometry config")
    
    with col3:
        st.markdown("**🎨 HSV Color**")
        if 'hsv_orange' in config:
            hsv = config['hsv_orange']
            if 'low_1' in hsv and 'high_1' in hsv:
                st.metric("Hue Range 1", 
                         f"{hsv['low_1'][0]}-{hsv['high_1'][0]}°",
                         help="Primary orange hue range")
                st.metric("Sat Range 1", 
                         f"{hsv['low_1'][1]}-{hsv['high_1'][1]}",
                         help="Saturation range")
                st.metric("Val Range 1", 
                         f"{hsv['low_1'][2]}-{hsv['high_1'][2]}",
                         help="Brightness range")
        else:
            st.info("No HSV config")
    
    # Full config viewer
    with st.expander("📄 View Full Configuration"):
        st.json(config)
    
    # Configuration warnings
    from utils.config_manager import ConfigManager
    warnings = ConfigManager.get_warnings(config)
    
    if warnings:
        st.markdown("---")
        st.subheader("⚠️ Configuration Warnings")
        for warning in warnings:
            st.warning(warning)

else:
    st.info("👆 No configuration loaded. Go to **Connection** page to download config from remote server.")

st.markdown("---")

# Competition tips
st.header("💡 Competition Tips & Common Issues")

tab1, tab2, tab3 = st.tabs(["🔧 Troubleshooting", "🏆 Competition Guide", "📚 Quick Reference"])

with tab1:
    st.subheader("Common Problems & Solutions")
    
    st.markdown("""
    ### 🆘 Track IDs Keep Increasing
    
    **Symptom**: IDs like 541, 542, 543... tracks never CONFIRMED
    
    **Cause**: Tracks deleted before confirmation
    
    **Fix**:
    1. Go to **🎯 Tracking Config**
    2. Increase `lost_timeout` to 2.0-3.0s
    3. Decrease `min_frames_for_confirm` to 3-4
    
    ---
    
    ### 🆘 Missing Cone Detections
    
    **Symptom**: Cones visible but not detected
    
    **Fix**:
    1. Check **🎨 Color Tuning** - widen HSV ranges
    2. Check **📏 Geometry** - lower `min_frame_score` to 0.25-0.30
    3. Try **⚡ Presets** - load "Permissive" preset
    
    ---
    
    ### 🆘 Too Many False Positives
    
    **Symptom**: Non-cone objects being tracked
    
    **Fix**:
    1. Go to **📏 Geometry Filters**
    2. Increase `min_frame_score` to 0.40-0.45
    3. Increase `confirm_avg_score` to 0.60
    4. Narrow aspect ratio range
    5. Try **⚡ Presets** - load "Strict" preset
    
    ---
    
    ### 🆘 Tracking Lost During Fast Movement
    
    **Symptom**: Tracks disappear when robot/cones move
    
    **Fix**:
    1. Go to **🎯 Tracking Config**
    2. Increase `association_max_distance` to 250-350px
    3. Increase `grace_frames` to 20-30
    
    ---
    
    ### 🆘 Colors Wrong (Too Bright/Dark)
    
    **Symptom**: Detections fail in bright sun or shadows
    
    **Fix**:
    1. Go to **🎨 Color Tuning**
    2. For bright conditions: load "Bright Outdoor" preset
    3. For shadows/indoor: load "Indoor/Shadows" preset
    4. Adjust V (brightness) ranges manually
    """)

with tab2:
    st.subheader("Competition Day Strategy")
    
    st.markdown("""
    ### Before Competition
    
    **Practice Rounds** (Day Before):
    1. ✅ Test detection in actual competition environment
    2. ✅ Try "Competition Optimized" preset as baseline
    3. ✅ Adjust HSV for lighting conditions
    4. ✅ Export working config and save backup
    5. ✅ Test that hot-reload works on your system
    
    **Morning Setup** (Competition Day):
    1. ✅ Connect to robot via SFTP
    2. ✅ Download current config to verify
    3. ✅ Have backup configs ready to upload
    4. ✅ Test one complete detection cycle
    
    ---
    
    ### During Competition
    
    **Between Runs**:
    - 📊 Review tracking performance logs
    - 🔧 Make **small**, targeted adjustments
    - ✅ Test changes before next run
    - 💾 Keep backup of working config
    
    **Quick Fixes**:
    - Lighting changed? → Adjust HSV brightness (V ranges)
    - Tracks lost? → Increase lost_timeout or association distance
    - False positives? → Increase score thresholds
    
    **Don't**:
    - ❌ Change multiple parameters at once
    - ❌ Use untested presets during runs
    - ❌ Make dramatic changes without testing
    
    ---
    
    ### After Each Run
    
    1. 📝 Note what worked / didn't work
    2. 💾 Save successful configs with descriptive names
    3. 🔍 Analyze logs for track deletion patterns
    4. 🎯 Plan specific adjustments for next run
    
    ---
    
    ### Emergency Recovery
    
    **If Detection Fails Completely**:
    1. 🔄 Reload last known good config
    2. 🔌 Verify SFTP connection
    3. 🎯 Load "Competition Optimized" preset
    4. 🌞 Load "Bright Outdoor" or "Indoor" for HSV
    5. ✅ Test before next run
    """)

with tab3:
    st.subheader("Quick Reference")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Parameters
        
        **Tracking**:
        - `lost_timeout`: 2.0-3.0s (competition)
        - `min_frames_for_confirm`: 4 (balanced)
        - `association_max_distance`: 250px (competition)
        
        **Geometry**:
        - `confirm_avg_score`: 0.50 (balanced)
        - `min_frame_score`: 0.30-0.35 (entry threshold)
        
        **HSV (Orange)**:
        - H: 0-28° (primary), 160-179° (secondary)
        - S: 70-100+ (lower for outdoor)
        - V: 80-120+ (adjust for lighting)
        """)
    
    with col2:
        st.markdown("""
        ### 📖 Page Guide
        
        **🔌 Connection**:
        - SFTP setup
        - Download/upload config
        - File browser
        
        **🎨 Color Tuning**:
        - HSV range adjustment
        - Lighting presets
        - Color preview
        
        **🎯 Tracking Config**:
        - Track lifetime settings
        - Confirmation thresholds
        - Association distance
        
        **📏 Geometry Filters**:
        - Area/aspect/fill filters
        - Score thresholds
        - Shape validation
        
        **⚡ Presets**:
        - Pre-configured setups
        - Export/import configs
        - Preset comparison
        """)

st.markdown("---")

# System information
with st.expander("ℹ️ System Information"):
    st.markdown("""
    ### About This Dashboard
    
    **Cone Tracker Remote Configuration Dashboard**
    
    This Streamlit application provides a web-based interface for remotely configuring
    the DragonBot cone detection and tracking system.
    
    **Features**:
    - 🔌 SFTP connection to remote robot
    - 🎨 HSV color tuning for different lighting
    - 🎯 Multi-object tracking parameter tuning
    - 📏 Geometric filter configuration
    - ⚡ Quick preset configurations
    - 💾 Config export/import
    - 📊 Real-time validation and warnings
    
    **Technology Stack**:
    - Streamlit for UI
    - Paramiko for SFTP
    - PyYAML for config parsing
    
    **Hot-Reload Compatible**:
    Changes uploaded via this dashboard are automatically detected and reloaded
    by the cone tracker application (if running with hot-reload enabled).
    
    ---
    
    ### Quick Start
    
    1. **Connect**: Go to Connection page and enter SFTP credentials
    2. **Download**: Download current config from remote server
    3. **Edit**: Use Color/Tracking/Geometry pages to adjust parameters
    4. **Upload**: Save changes back to remote server
    5. **Test**: Verify detection performance
    
    ---
    
    ### Documentation
    
    For detailed documentation, see:
    - Project README: `/README.md`
    - Streamlit README: `/streamlit_app/README.md`
    - Architecture: `/ARCHITECTURE.md`
    
    ---
    
    ### Support
    
    For issues or questions:
    - Check troubleshooting guide above
    - Review parameter definitions in each page
    - Consult competition tips
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Cone Tracker Dashboard v1.0 | DragonBot Trekking Team</p>
    </div>
    """,
    unsafe_allow_html=True
)
