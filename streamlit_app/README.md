# Cone Tracker Remote Configuration Dashboard

A Streamlit-based multi-page web dashboard for remotely configuring the DragonBot cone detection and tracking system via SFTP.

## 🚀 Features

- **🔌 SFTP Connection Management**: Connect to remote robot and manage config files
- **🎨 HSV Color Tuning**: Adjust color detection ranges for different lighting conditions
- **🎯 Tracking Configuration**: Fine-tune multi-object tracking parameters
- **📏 Geometry Filters**: Configure shape and size validation filters
- **⚡ Quick Presets**: Pre-configured settings for common scenarios
- **💾 Config Export/Import**: Save and load configurations
- **🔄 Hot-Reload Compatible**: Changes automatically detected by running tracker
- **✅ Real-time Validation**: Instant feedback on parameter validity
- **📊 Visual Previews**: Color and parameter visualization

## 📋 Requirements

- Python 3.7+
- Network access to remote robot (SSH/SFTP)
- Dependencies listed in `requirements.txt`

## 🔧 Installation

### 1. Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
streamlit --version
```

## 🎮 Usage

### Starting the Dashboard

```bash
cd streamlit_app
streamlit run Home.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

#### Custom Port

```bash
streamlit run Home.py --server.port 8502
```

#### Remote Access

To access from other devices on the network:

```bash
streamlit run Home.py --server.address 0.0.0.0 --server.port 8501
```

Then access via `http://<your-ip>:8501`

### Basic Workflow

1. **🔌 Connection Page**
   - Enter SFTP credentials (host, username, password)
   - Test connection
   - Browse remote filesystem
   - Download current config

2. **🎨 Color Tuning**
   - Adjust HSV ranges for orange detection
   - Use presets for bright/indoor conditions
   - Preview color ranges

3. **🎯 Tracking Config**
   - Set track lifetime (`lost_timeout`)
   - Configure confirmation thresholds
   - Adjust association distance

4. **📏 Geometry Filters**
   - Set area/aspect/fill ratio ranges
   - Configure score thresholds
   - Validate cone shape

5. **⚡ Presets**
   - Load pre-configured settings
   - Export current config
   - Import saved configs

6. **💾 Save & Upload**
   - Return to Connection page
   - Upload modified config
   - Automatic backup created

## 📖 Page Guide

### 🏠 Home

- Connection status
- Quick navigation
- Current config overview
- Troubleshooting guide
- Competition tips

### 🔌 Connection

**Purpose**: Establish SFTP connection and manage remote files

**Features**:
- SFTP credential input
- Connection testing
- Remote file browser
- File information display
- Download/upload operations
- Automatic backups

**Typical Workflow**:
```
1. Enter host IP (e.g., 192.168.1.100)
2. Enter username/password
3. Click "Test Connection"
4. Verify file path
5. Click "Download Config"
6. Edit in other pages
7. Return and click "Upload Config"
```

### 🎨 Color Tuning

**Purpose**: Adjust HSV color ranges for cone detection

**Key Parameters**:
- `low_1` / `high_1`: Primary orange range (0-28° hue)
- `low_2` / `high_2`: Secondary range (160-179° hue, wraps around)

**Quick Presets**:
- **Bright Outdoor**: Adjusted for direct sunlight
- **Indoor/Shadows**: Adjusted for low light

**Tips**:
- Increase V (brightness) low for outdoor
- Decrease V low for indoor/shadows
- Adjust S (saturation) to filter gray shadows

### 🎯 Tracking Config

**Purpose**: Configure multi-object tracking behavior

**Critical Parameters**:
- `lost_timeout` (2.0-3.0s): Track survival time without detection
- `min_frames_for_confirm` (4): Frames needed for CONFIRMED state
- `association_max_distance` (250px): Max distance for detection-track matching

**Common Issues**:
- IDs increasing → Increase `lost_timeout`
- Lost during movement → Increase `association_max_distance`
- Slow confirmation → Decrease `min_frames_for_confirm`

### 📏 Geometry Filters

**Purpose**: Validate detections have cone-like geometry

**Key Filters**:
- **Area**: Min/max bounding box size
- **Aspect Ratio**: Height/width (cones are tall)
- **Fill Ratio**: Portion of bbox filled (cones taper)
- **Scores**: Quality thresholds

**Tuning**:
- False positives → Increase scores, narrow ranges
- Missing cones → Decrease scores, widen ranges

### ⚡ Presets

**Purpose**: Quick-load proven configurations

**Available Presets**:
- 🏆 **Competition Optimized**: Balanced, proven settings
- 🌞 **Bright Outdoor**: HSV for sunlight
- 🏢 **Indoor/Shadows**: HSV for low light
- 🔓 **Permissive**: Catch everything (testing)
- 🎯 **Strict**: High precision (clean environments)

**Features**:
- Load preset configurations
- Export current config (JSON/YAML)
- Import saved configs
- Merge multiple presets

## 🔐 SFTP Configuration

### Connection Settings

**Required**:
- Host/IP address
- Port (default: 22)
- Username
- Password or SSH key path

**Example**:
```
Host: 192.168.1.100
Port: 22
Username: orangepi
Password: ********
Remote Path: /home/orangepi/Trekking-DragonBotZv2/cone_config.yaml
```

### SSH Key Authentication

Instead of password:

1. Generate SSH key (if not exists):
   ```bash
   ssh-keygen -t rsa -b 4096
   ```

2. Copy to remote server:
   ```bash
   ssh-copy-id username@host
   ```

3. In dashboard, select "SSH Key" and enter path (e.g., `~/.ssh/id_rsa`)

### Troubleshooting Connection

**Connection Refused**:
- Verify SSH is running: `sudo systemctl status ssh`
- Check firewall: `sudo ufw status`
- Verify IP address is correct

**Authentication Failed**:
- Verify username/password
- Check SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Try password authentication first

**File Not Found**:
- Use file browser to navigate
- Verify absolute path (starts with `/`)
- Check file permissions

## 📊 Configuration Reference

### HSV Orange Detection

```yaml
hsv_orange:
  low_1: [0, 90, 90]      # H, S, V lower bound (primary)
  high_1: [28, 255, 255]  # H, S, V upper bound (primary)
  low_2: [160, 90, 80]    # H, S, V lower bound (wrap-around)
  high_2: [179, 255, 255] # H, S, V upper bound (wrap-around)
```

### Tracking Parameters

```yaml
tracking:
  lost_timeout: 3.0                    # Track lifetime (seconds)
  min_frames_for_confirm: 4            # Frames to confirm
  association_max_distance: 250        # Match distance (pixels)
  ema_alpha: 0.35                      # Smoothing factor
  grace_frames: 20                     # CONFIRMED grace period
  score_window: 10                     # Score averaging window
  max_tracks: 8                        # Max simultaneous tracks
```

### Geometry Filters

```yaml
geometry:
  min_group_area: 1400                 # Min bbox area (px²)
  max_group_area: 450000               # Max bbox area (px²)
  aspect_min: 1.0                      # Min height/width
  aspect_max: 6.0                      # Max height/width
  min_fill_ratio: 0.08                 # Min fill fraction
  max_fill_ratio: 0.65                 # Max fill fraction
  min_frame_score: 0.30                # Per-frame threshold
  confirm_avg_score: 0.50              # Confirmation threshold
  profile_slices: 10                   # Profile analysis slices
  min_profile_score: 0.35              # Profile score threshold
```

## 🏆 Competition Guide

### Pre-Competition Checklist

- [ ] Test SFTP connection to robot
- [ ] Download current config as backup
- [ ] Test hot-reload functionality
- [ ] Verify dashboard accessible from pit area
- [ ] Have backup configs ready
- [ ] Test in practice rounds

### Competition Day Workflow

**Morning**:
1. Connect to robot
2. Download and verify config
3. Test detection once

**Between Runs**:
1. Review performance
2. Make small adjustments
3. Upload changes
4. Verify hot-reload worked

**Emergency**:
1. Load last known good config
2. Try "Competition Optimized" preset
3. Test before next run

### Quick Fixes

| Problem | Solution |
|---------|----------|
| IDs increasing | Increase `lost_timeout` to 3.0s |
| Missing cones | Load "Permissive" preset |
| False positives | Load "Strict" preset |
| Too bright | Load "Bright Outdoor" preset |
| Too dark | Load "Indoor/Shadows" preset |
| Lost during movement | Increase `association_max_distance` |

## 🛠️ Development

### Project Structure

```
streamlit_app/
├── Home.py                          # Main dashboard
├── pages/
│   ├── 1_🔌_Connection.py          # SFTP connection
│   ├── 2_🎨_Color_Tuning.py        # HSV tuning
│   ├── 3_🎯_Tracking_Config.py     # Tracking params
│   ├── 4_📏_Geometry_Filters.py    # Geometry filters
│   └── 5_⚡_Presets.py             # Preset management
├── utils/
│   ├── __init__.py
│   ├── sftp_manager.py             # SFTP operations
│   └── config_manager.py           # Config validation
├── .streamlit/
│   └── config.toml                 # Streamlit settings
├── requirements.txt
└── README.md
```

### Adding Custom Presets

Edit `utils/config_manager.py`:

```python
PRESETS = {
    "My Custom Preset": {
        "tracking": {
            "lost_timeout": 2.5,
            # ... more parameters
        },
        "geometry": {
            # ... geometry parameters
        }
    }
}
```

### Customizing Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B35"      # Orange accent
backgroundColor = "#FFFFFF"    # White background
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🔒 Security Notes

- **Never commit credentials** to version control
- Use SSH keys instead of passwords when possible
- Dashboard creates automatic backups before overwriting
- Backups saved with timestamp: `cone_config.yaml.bak.YYYYMMDD_HHMMSS`
- Consider using `.streamlit/secrets.toml` for credentials (not implemented by default)

## 🐛 Troubleshooting

### Dashboard Won't Start

```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Import Errors

```bash
# Verify dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.7+
```

### SFTP Connection Issues

See [SFTP Configuration](#sftp-configuration) section above.

### Config Changes Not Applied

- Verify upload succeeded (check for success message)
- Check tracker application is running with hot-reload
- Wait 1-2 seconds for hot-reload to detect change
- Check tracker logs for "Config reloaded" message

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Paramiko Documentation](https://www.paramiko.org/)
- [Cone Tracker Main README](../README.md)
- [Architecture Documentation](../ARCHITECTURE.md)

## 📝 License

Same as parent project.

## 🤝 Contributing

Contributions welcome! Please:
1. Test changes thoroughly
2. Update documentation
3. Follow existing code style
4. Add comments for complex logic

## 💬 Support

For issues or questions:
1. Check this README
2. Review troubleshooting guide
3. Check main project documentation
4. Review code comments in source files
