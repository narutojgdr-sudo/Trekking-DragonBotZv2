"""Presets management page."""
import streamlit as st
import copy
from utils.config_manager import ConfigManager
from utils.formatting import format_metric, safe_get

st.set_page_config(
    page_title="Presets",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}

st.title("⚡ Configuration Presets")
st.markdown("Quick preset configurations for common scenarios")
st.markdown("---")

# Available presets
st.header("📦 Available Presets")

presets = ConfigManager.list_presets()

# Create columns for preset cards
cols = st.columns(2)

for idx, preset_name in enumerate(presets):
    with cols[idx % 2]:
        with st.container():
            st.subheader(preset_name)
            
            preset_config = ConfigManager.get_preset(preset_name)
            
            # Display preset details
            if preset_name == "Competition Optimized":
                st.markdown("""
                **🏆 Best for competition environments**
                - Fast confirmation (4 frames)
                - Balanced thresholds (0.50 confirm score)
                - 3s track lifetime
                - 250px association distance
                
                **Optimized for:**
                - Stable tracking
                - Quick response
                - Reliable performance
                """)
            
            elif preset_name == "Bright Outdoor":
                st.markdown("""
                **🌞 Best for bright sunlight**
                - Adjusted HSV for high brightness
                - Filters dark shadows
                - Wider saturation tolerance
                
                **Optimized for:**
                - Direct sunlight
                - High contrast scenes
                - Sun-bleached colors
                """)
            
            elif preset_name == "Indoor/Shadows":
                st.markdown("""
                **🏢 Best for low light conditions**
                - Lower brightness thresholds
                - Catches darker cones
                - Adjusted for mixed lighting
                
                **Optimized for:**
                - Indoor environments
                - Shadowed areas
                - Lower light levels
                """)
            
            elif preset_name == "Permissive":
                st.markdown("""
                **🔓 Catch all possible cones**
                - Low thresholds (0.45 confirm)
                - Fast confirmation (3 frames)
                - Wide association (300px)
                
                **Optimized for:**
                - Testing/debugging
                - Uncertain conditions
                - Maximum recall
                """)
            
            elif preset_name == "Strict":
                st.markdown("""
                **🎯 High precision, low false positives**
                - High thresholds (0.60 confirm)
                - Slow confirmation (6 frames)
                - Narrow association (150px)
                
                **Optimized for:**
                - Clean environments
                - Minimal distractions
                - Maximum precision
                """)
            
            # Show what parameters this preset modifies
            with st.expander("📋 Preset Details"):
                st.json(preset_config)
            
            # Load button
            if st.button(f"Load {preset_name}", key=f"load_{preset_name}", width="stretch", type="primary"):
                # Merge preset with current config
                if st.session_state.current_config:
                    merged = ConfigManager.deep_merge(
                        st.session_state.current_config,
                        preset_config
                    )
                    st.session_state.current_config = merged
                else:
                    st.session_state.current_config = copy.deepcopy(preset_config)
                
                st.success(f"✅ Loaded '{preset_name}' preset!")
                st.rerun()

st.markdown("---")

# Current configuration
st.header("📊 Current Configuration")

if st.session_state.current_config:
    # Show key parameters
    config = st.session_state.current_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tracking Parameters")
        if 'tracking' in config:
            tracking = config['tracking']
            st.metric("Lost Timeout", format_metric(tracking.get('lost_timeout'), "s"))
            st.metric("Min Frames to Confirm", format_metric(tracking.get('min_frames_for_confirm')))
            st.metric("Association Distance", format_metric(tracking.get('association_max_distance'), "px"))
            st.metric("EMA Alpha", format_metric(tracking.get('ema_alpha')))
            st.metric("Grace Frames", format_metric(tracking.get('grace_frames')))
        else:
            st.info("No tracking configuration loaded")
    
    with col2:
        st.subheader("Geometry Parameters")
        if 'geometry' in config:
            geometry = config['geometry']
            st.metric("Confirm Avg Score", format_metric(geometry.get('confirm_avg_score')))
            st.metric("Min Frame Score", format_metric(geometry.get('min_frame_score')))
            st.metric("Min Area", format_metric(geometry.get('min_group_area'), " px²"))
            st.metric("Max Area", format_metric(geometry.get('max_group_area'), " px²"))
            min_aspect = format_metric(geometry.get('aspect_min'))
            max_aspect = format_metric(geometry.get('aspect_max'))
            st.metric("Aspect Range", f"{min_aspect} - {max_aspect}")
        else:
            st.info("No geometry configuration loaded")
    
    # HSV parameters
    if 'hsv_orange' in config:
        with st.expander("🎨 HSV Color Configuration"):
            hsv = config['hsv_orange']
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Range 1:**")
                st.write(f"Low: {hsv.get('low_1', 'N/A')}")
                st.write(f"High: {hsv.get('high_1', 'N/A')}")
            
            with col2:
                st.write("**Range 2:**")
                st.write(f"Low: {hsv.get('low_2', 'N/A')}")
                st.write(f"High: {hsv.get('high_2', 'N/A')}")
    
    # Full configuration viewer
    with st.expander("📄 View Full Configuration"):
        st.json(config)
    
    # Export/Import
    st.markdown("---")
    st.subheader("💾 Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        import json
        config_json = json.dumps(config, indent=2)
        st.download_button(
            label="📥 Download Config as JSON",
            data=config_json,
            file_name="cone_config_export.json",
            mime="application/json",
            width="stretch"
        )
    
    with col2:
        # Export as YAML
        import yaml
        config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
        st.download_button(
            label="📥 Download Config as YAML",
            data=config_yaml,
            file_name="cone_config_export.yaml",
            mime="text/yaml",
            width="stretch"
        )

else:
    st.info("No configuration loaded. Load a preset or download config from Connection page.")

st.markdown("---")

# Import configuration
st.header("📤 Import Configuration")

uploaded_file = st.file_uploader(
    "Upload a configuration file (YAML or JSON)",
    type=['yaml', 'yml', 'json'],
    help="Upload a previously exported configuration file"
)

if uploaded_file is not None:
    try:
        import yaml
        import json
        
        content = uploaded_file.read().decode('utf-8')
        
        # Try parsing as YAML first (works for both YAML and JSON)
        try:
            imported_config = yaml.safe_load(content)
        except:
            # Fallback to JSON
            imported_config = json.loads(content)
        
        st.success("✅ File loaded successfully!")
        
        # Show preview
        with st.expander("📄 Preview Imported Configuration"):
            st.json(imported_config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Apply Imported Config", type="primary", width="stretch"):
                st.session_state.current_config = imported_config
                st.success("Configuration applied!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Merge with Current", width="stretch"):
                if st.session_state.current_config:
                    merged = ConfigManager.deep_merge(
                        st.session_state.current_config,
                        imported_config
                    )
                    st.session_state.current_config = merged
                    st.success("Configuration merged!")
                    st.rerun()
                else:
                    st.session_state.current_config = imported_config
                    st.success("Configuration applied!")
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

st.markdown("---")

# Custom preset creation
with st.expander("🔧 Create Custom Preset"):
    st.markdown("""
    ### Creating Custom Presets
    
    Currently, custom presets can be created by:
    
    1. **Configure parameters** in other pages (Color, Tracking, Geometry)
    2. **Export current config** using the download buttons above
    3. **Save the file** for future use
    4. **Import the file** when needed using the file uploader
    
    ### Sharing Presets
    
    Exported configurations can be:
    - Shared with team members
    - Versioned in source control
    - Backed up for different competition environments
    - Used as templates for similar scenarios
    
    ### Preset Best Practices
    
    **Name your exports descriptively:**
    - `competition_outdoor_sunny.yaml`
    - `practice_indoor_2024.yaml`
    - `debug_permissive.yaml`
    
    **Document your presets:**
    - Add comments in YAML files
    - Note lighting conditions
    - Record date and location
    - Include performance notes
    
    **Test before competition:**
    - Validate preset works in target environment
    - Adjust as needed
    - Keep backup of working configs
    """)

# Preset comparison
with st.expander("📊 Preset Comparison"):
    st.markdown("""
    ### Preset Comparison Matrix
    
    | Preset | Confirm Score | Min Frames | Lost Timeout | Association Dist | Use Case |
    |--------|--------------|------------|--------------|------------------|----------|
    | 🏆 Competition | 0.50 | 4 | 3.0s | 250px | **Balanced, reliable** |
    | 🔓 Permissive | 0.45 | 3 | N/A | 300px | **Catch everything** |
    | 🎯 Strict | 0.60 | 6 | N/A | 150px | **High precision** |
    | 🌞 Bright Outdoor | N/A | N/A | N/A | N/A | **HSV for sunlight** |
    | 🏢 Indoor/Shadows | N/A | N/A | N/A | N/A | **HSV for low light** |
    
    **Quick Selection Guide:**
    
    - **New environment?** → Start with Competition
    - **Too many false positives?** → Try Strict
    - **Missing cones?** → Try Permissive
    - **Bright day?** → Add Bright Outdoor (HSV only)
    - **Indoor/Shadows?** → Add Indoor/Shadows (HSV only)
    
    **Combining Presets:**
    
    You can merge multiple presets:
    1. Load Competition preset (tracking + geometry)
    2. Then load Bright Outdoor (adds HSV tuning)
    3. Result: Competition tracking with outdoor HSV
    
    This is useful for combining tracking presets with color presets.
    """)

# Tips
with st.expander("💡 Tips & Recommendations"):
    st.markdown("""
    ### When to Use Each Preset
    
    **🏆 Competition Optimized:**
    - Default choice for competition
    - Proven balanced parameters
    - Use as baseline, then fine-tune
    
    **🌞 Bright Outdoor:**
    - Direct sunlight conditions
    - Outdoor competitions during day
    - High contrast environments
    - Combine with Competition tracking
    
    **🏢 Indoor/Shadows:**
    - Indoor testing/competition
    - Low light conditions
    - Mixed lighting with shadows
    - Combine with Competition tracking
    
    **🔓 Permissive:**
    - Initial testing in new environment
    - When you're missing detections
    - Debugging detection pipeline
    - Finding optimal threshold range
    
    **🎯 Strict:**
    - Clean, controlled environment
    - When false positives are a problem
    - After initial tuning with Permissive
    - When precision > recall is priority
    
    ---
    
    ### Recommended Workflow
    
    1. **Start**: Load Competition Optimized
    2. **Test**: Run detection in target environment
    3. **Assess**:
       - Missing cones? → Try Permissive or adjust HSV
       - False positives? → Try Strict or tighten geometry
       - Wrong colors? → Try Bright Outdoor or Indoor presets
    4. **Fine-tune**: Adjust individual parameters in other pages
    5. **Export**: Save working config for competition day
    6. **Backup**: Keep multiple configs for different scenarios
    
    ---
    
    ### Competition Day Strategy
    
    **Before Competition:**
    - Test presets in practice rounds
    - Export working configurations
    - Have backup configs ready
    
    **During Competition:**
    - Start with tested preset
    - Monitor detection quality
    - Make small adjustments if needed
    - Don't change too many parameters at once
    
    **Common Adjustments:**
    - Lighting changed? → Adjust HSV V (brightness) ranges
    - More/less cones detected? → Adjust score thresholds
    - Tracking unstable? → Adjust lost_timeout or association distance
    """)
