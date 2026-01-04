"""Color tuning page for HSV parameter adjustment."""
import streamlit as st
import numpy as np
from PIL import Image
from utils.config_manager import ConfigManager

st.set_page_config(
    page_title="Color Tuning",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}

st.title("🎨 Color Tuning")
st.markdown("Adjust HSV color ranges for orange cone detection")
st.markdown("---")

# Load current config or use defaults
config = st.session_state.current_config
if 'hsv_orange' not in config:
    config['hsv_orange'] = {
        "low_1": [0, 90, 90],
        "high_1": [28, 255, 255],
        "low_2": [160, 90, 80],
        "high_2": [179, 255, 255]
    }

hsv_config = config['hsv_orange']

# Quick presets
st.header("Quick Presets")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌞 Bright Outdoor", use_container_width=True):
        preset = ConfigManager.get_preset("Bright Outdoor")
        if preset and 'hsv_orange' in preset:
            config['hsv_orange'] = preset['hsv_orange']
            st.success("Loaded Bright Outdoor preset")
            st.rerun()

with col2:
    if st.button("🏢 Indoor/Shadows", use_container_width=True):
        preset = ConfigManager.get_preset("Indoor/Shadows")
        if preset and 'hsv_orange' in preset:
            config['hsv_orange'] = preset['hsv_orange']
            st.success("Loaded Indoor/Shadows preset")
            st.rerun()

with col3:
    if st.button("🔄 Reset to Default", use_container_width=True):
        config['hsv_orange'] = {
            "low_1": [0, 90, 90],
            "high_1": [28, 255, 255],
            "low_2": [160, 90, 80],
            "high_2": [179, 255, 255]
        }
        st.success("Reset to defaults")
        st.rerun()

st.markdown("---")

# HSV Range 1 (Primary orange range)
st.header("HSV Range 1 (Primary Orange)")
st.markdown("*Main orange detection range (typically 0-28° hue)*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lower Bound")
    low_h1 = st.slider(
        "Hue Low",
        min_value=0,
        max_value=179,
        value=hsv_config['low_1'][0],
        help="Lower hue threshold (0-179). Orange is typically 0-30."
    )
    low_s1 = st.slider(
        "Saturation Low",
        min_value=0,
        max_value=255,
        value=hsv_config['low_1'][1],
        help="Lower saturation threshold. Higher values = more vivid colors only."
    )
    low_v1 = st.slider(
        "Value Low",
        min_value=0,
        max_value=255,
        value=hsv_config['low_1'][2],
        help="Lower brightness threshold. Higher values = brighter pixels only."
    )

with col2:
    st.subheader("Upper Bound")
    high_h1 = st.slider(
        "Hue High",
        min_value=0,
        max_value=179,
        value=hsv_config['high_1'][0],
        help="Upper hue threshold (0-179). Orange is typically 0-30."
    )
    high_s1 = st.slider(
        "Saturation High",
        min_value=0,
        max_value=255,
        value=hsv_config['high_1'][1],
        help="Upper saturation threshold. Usually set to 255."
    )
    high_v1 = st.slider(
        "Value High",
        min_value=0,
        max_value=255,
        value=hsv_config['high_1'][1],
        help="Upper brightness threshold. Usually set to 255."
    )

# Update config
config['hsv_orange']['low_1'] = [low_h1, low_s1, low_v1]
config['hsv_orange']['high_1'] = [high_h1, high_s1, high_v1]

st.markdown("---")

# HSV Range 2 (Secondary orange range - wraps around)
st.header("HSV Range 2 (Wrap-around)")
st.markdown("*Secondary range for orange hues near red (typically 160-179° hue)*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lower Bound")
    low_h2 = st.slider(
        "Hue Low ",
        min_value=0,
        max_value=179,
        value=hsv_config['low_2'][0],
        help="Lower hue threshold for wrap-around range. Typically 160-170."
    )
    low_s2 = st.slider(
        "Saturation Low ",
        min_value=0,
        max_value=255,
        value=hsv_config['low_2'][1],
        help="Lower saturation threshold."
    )
    low_v2 = st.slider(
        "Value Low ",
        min_value=0,
        max_value=255,
        value=hsv_config['low_2'][2],
        help="Lower brightness threshold."
    )

with col2:
    st.subheader("Upper Bound")
    high_h2 = st.slider(
        "Hue High ",
        min_value=0,
        max_value=179,
        value=hsv_config['high_2'][0],
        help="Upper hue threshold. Usually 179 for maximum range."
    )
    high_s2 = st.slider(
        "Saturation High ",
        min_value=0,
        max_value=255,
        value=hsv_config['high_2'][1],
        help="Upper saturation threshold."
    )
    high_v2 = st.slider(
        "Value High ",
        min_value=0,
        max_value=255,
        value=hsv_config['high_2'][2],
        help="Upper brightness threshold."
    )

# Update config
config['hsv_orange']['low_2'] = [low_h2, low_s2, low_v2]
config['hsv_orange']['high_2'] = [high_h2, high_s2, high_v2]

# Validation
st.markdown("---")
st.header("Validation")

errors = []
valid1, msg1 = ConfigManager.validate_hsv_range(
    config['hsv_orange']['low_1'],
    config['hsv_orange']['high_1']
)
if not valid1:
    errors.append(f"Range 1: {msg1}")

valid2, msg2 = ConfigManager.validate_hsv_range(
    config['hsv_orange']['low_2'],
    config['hsv_orange']['high_2']
)
if not valid2:
    errors.append(f"Range 2: {msg2}")

if errors:
    for error in errors:
        st.error(f"❌ {error}")
else:
    st.success("✅ All HSV values are valid")

# Color preview
st.markdown("---")
st.header("Color Preview")

col1, col2 = st.columns(2)

def create_hsv_preview(h_range, s, v, label):
    """Create a color preview image for given HSV range."""
    width = 300
    height = 100
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient
    for x in range(width):
        hue = int(h_range[0] + (h_range[1] - h_range[0]) * x / width)
        hsv_color = np.array([[[hue, s, v]]], dtype=np.uint8)
        # Convert to RGB for display
        import cv2
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
        img[:, x] = rgb_color
    
    return Image.fromarray(img)

with col1:
    st.subheader("Range 1 Preview")
    try:
        import cv2
        # Show color at middle of ranges
        mid_h1 = (low_h1 + high_h1) // 2
        mid_s1 = (low_s1 + high_s1) // 2
        mid_v1 = (low_v1 + high_v1) // 2
        
        preview1 = create_hsv_preview([low_h1, high_h1], mid_s1, mid_v1, "Range 1")
        st.image(preview1, use_container_width=True, caption=f"H: {low_h1}-{high_h1}, S: {low_s1}-{high_s1}, V: {low_v1}-{high_v1}")
    except ImportError:
        st.info("Install OpenCV (cv2) to see color preview")

with col2:
    st.subheader("Range 2 Preview")
    try:
        import cv2
        mid_h2 = (low_h2 + high_h2) // 2
        mid_s2 = (low_s2 + high_s2) // 2
        mid_v2 = (low_v2 + high_v2) // 2
        
        preview2 = create_hsv_preview([low_h2, high_h2], mid_s2, mid_v2, "Range 2")
        st.image(preview2, use_container_width=True, caption=f"H: {low_h2}-{high_h2}, S: {low_s2}-{high_s2}, V: {low_v2}-{high_v2}")
    except ImportError:
        st.info("Install OpenCV (cv2) to see color preview")

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

# Tips
with st.expander("💡 Tuning Tips"):
    st.markdown("""
    ### HSV Color Space Basics
    - **H (Hue)**: Color type (0-179 in OpenCV)
        - Orange: 0-30°
        - Red wraps around: 0-10° and 160-179°
    - **S (Saturation)**: Color intensity (0-255)
        - Low: Washed out, pale colors
        - High: Vivid, pure colors
    - **V (Value)**: Brightness (0-255)
        - Low: Dark pixels
        - High: Bright pixels
    
    ### Tuning for Different Conditions
    
    **Bright Outdoor (Direct Sunlight):**
    - Increase V Low (e.g., 120) to filter dark shadows
    - May need to adjust S Low (e.g., 70-80) for sun-bleached colors
    
    **Indoor/Low Light:**
    - Decrease V Low (e.g., 60-80) to catch darker cones
    - May need higher S Low to avoid non-orange shadows
    
    **Shadows/Mixed Lighting:**
    - Use wider V range (e.g., 60-200)
    - Increase S Low to filter gray shadows
    
    ### Common Problems
    
    **Too many false positives:**
    - Increase S Low (more saturation required)
    - Narrow H range
    - Increase V Low (filter dark areas)
    
    **Missing cone detections:**
    - Widen H range
    - Lower S Low
    - Lower V Low
    - Check if cone color falls in your ranges
    
    **Unstable detection (flickers):**
    - Slightly widen ranges for more tolerance
    - Check lighting consistency
    """)
