"""SFTP Connection configuration page."""
import streamlit as st
import os
import re
from datetime import datetime
from utils.sftp_manager import SFTPManager


def is_valid_host(host: str) -> bool:
    """Validate IP address or hostname."""
    # IPv4 pattern
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    # Hostname pattern
    hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    
    if re.match(ipv4_pattern, host):
        # Validate IP octets
        octets = host.split('.')
        return all(0 <= int(octet) <= 255 for octet in octets)
    return bool(re.match(hostname_pattern, host))


def is_valid_port(port: int) -> bool:
    """Validate port number."""
    return 1 <= port <= 65535


def is_valid_path(path: str) -> bool:
    """Validate Unix file path."""
    # Must start with / and contain valid characters
    return bool(re.match(r'^/[a-zA-Z0-9/_\-\.]+$', path))


st.set_page_config(
    page_title="SFTP Connection",
    page_icon="🔌",
    layout="wide"
)

# Initialize session state
if 'sftp_manager' not in st.session_state:
    st.session_state.sftp_manager = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'remote_path' not in st.session_state:
    st.session_state.remote_path = '/home/orangepi/Trekking-DragonBotZv2/cone_config.yaml'
if 'last_sync' not in st.session_state:
    st.session_state.last_sync = None

st.title("🔌 SFTP Connection")
st.markdown("---")

# Connection settings
st.header("Connection Settings")

col1, col2 = st.columns(2)

with col1:
    host = st.text_input(
        "Host/IP Address",
        value=st.session_state.get('sftp_host', '192.168.1.100'),
        help="IP address or hostname of the remote server"
    )
    username = st.text_input(
        "Username",
        value=st.session_state.get('sftp_username', 'root'),
        help="SSH username"
    )

with col2:
    port = st.number_input(
        "Port",
        min_value=1,
        max_value=65535,
        value=st.session_state.get('sftp_port', 22),
        help="SSH port (default: 22)"
    )
    
    auth_method = st.radio(
        "Authentication Method",
        options=["Password", "SSH Key"],
        index=0,
        horizontal=True
    )

if auth_method == "Password":
    password = st.text_input(
        "Password",
        type="password",
        help="SSH password"
    )
    key_path = None
else:
    password = None
    key_path = st.text_input(
        "SSH Key Path",
        value="~/.ssh/id_rsa",
        help="Path to SSH private key file"
    )
    if key_path:
        key_path = os.path.expanduser(key_path)

# Test connection button
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔍 Test Connection", type="primary"):
        # Validate inputs first
        validation_errors = []
        
        if not is_valid_host(host):
            validation_errors.append("❌ Invalid hostname or IP address")
        
        if not is_valid_port(port):
            validation_errors.append("❌ Port must be between 1 and 65535")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            with st.spinner("Testing connection..."):
                try:
                    # Store credentials
                    st.session_state.sftp_host = host
                    st.session_state.sftp_port = port
                    st.session_state.sftp_username = username
                    
                    # Create SFTP manager
                    manager = SFTPManager(
                        host=host,
                        port=port,
                        username=username,
                        password=password,
                        key_path=key_path
                    )
                    
                    success, message = manager.connect()
                    if success:
                        st.session_state.sftp_manager = manager
                        st.session_state.connected = True
                        st.success(f"✅ {message}")
                    else:
                        st.session_state.connected = False
                        st.error(f"❌ {message}")
                        
                        # Troubleshooting hints
                        if "port" in message.lower() or "network" in message.lower():
                            st.warning("""
                            **Troubleshooting Steps:**
                            1. Check if SSH service is running: `systemctl status ssh`
                            2. Start SSH service: `sudo systemctl start ssh`
                            3. Check firewall: `sudo ufw allow 22`
                            4. Verify IP address: `hostname -I`
                            """)
                        elif "auth" in message.lower():
                            st.warning("""
                            **Authentication Failed:**
                            - Verify username and password
                            - Try using SSH key instead
                            - Check if user has SSH access: `cat /etc/ssh/sshd_config`
                            """)
                except Exception as e:
                    st.session_state.connected = False
                    st.error(f"❌ Unexpected error: {str(e)}")

with col2:
    if st.button("Disconnect") and st.session_state.connected:
        if st.session_state.sftp_manager:
            st.session_state.sftp_manager.disconnect()
        st.session_state.connected = False
        st.session_state.sftp_manager = None
        st.info("Disconnected from server")

with col3:
    # Connection status indicator
    if st.session_state.connected:
        st.success("🟢 Connected")
    else:
        st.error("🔴 Disconnected")

st.markdown("---")

# Remote file configuration
st.header("📁 Remote Config File")

remote_path = st.text_input(
    "Remote Config Path",
    value=st.session_state.remote_path,
    help="Full path to cone_config.yaml on remote server"
)

# Update session state
st.session_state.remote_path = remote_path

# File browser
if st.session_state.connected and st.session_state.sftp_manager:
    st.subheader("File Browser")
    
    # Extract directory from remote path
    current_dir = os.path.dirname(remote_path) or "/"
    
    browse_path = st.text_input(
        "Browse Directory",
        value=current_dir,
        help="Directory to browse"
    )
    
    if st.button("📂 List Files"):
        with st.spinner("Loading directory..."):
            try:
                items = st.session_state.sftp_manager.list_directory(browse_path)
                
                if items:
                    st.write(f"**Contents of {browse_path}:**")
                    
                    # Display as table
                    for item in items:
                        icon = "📁" if item['is_dir'] else "📄"
                        size_str = f"{item['size']:,} bytes" if not item['is_dir'] else "DIR"
                        mtime_str = datetime.fromtimestamp(item['mtime']).strftime('%Y-%m-%d %H:%M:%S')
                        
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        with col1:
                            st.write(f"{icon} `{item['name']}`")
                        with col2:
                            st.write(size_str)
                        with col3:
                            st.write(mtime_str)
                        with col4:
                            st.write(item['permissions'])
                else:
                    st.warning("Directory is empty or inaccessible")
            except Exception as e:
                st.error(f"Error listing directory: {str(e)}")

# File information
if st.session_state.connected and st.session_state.sftp_manager:
    st.markdown("---")
    st.subheader("File Information")
    
    if st.session_state.sftp_manager.file_exists(remote_path):
        file_info = st.session_state.sftp_manager.get_file_info(remote_path)
        
        if file_info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", "✅ File exists")
            with col2:
                st.metric("Size", f"{file_info['size']:,} bytes")
            with col3:
                # Calculate time ago
                now = datetime.now()
                mtime = file_info['mtime']
                delta = now - mtime
                
                if delta.total_seconds() < 60:
                    time_ago = f"{int(delta.total_seconds())} seconds ago"
                elif delta.total_seconds() < 3600:
                    time_ago = f"{int(delta.total_seconds() / 60)} minutes ago"
                elif delta.total_seconds() < 86400:
                    time_ago = f"{int(delta.total_seconds() / 3600)} hours ago"
                else:
                    time_ago = f"{int(delta.total_seconds() / 86400)} days ago"
                
                st.metric("Last Modified", time_ago)
            
            st.info(f"📊 Permissions: {file_info['permissions']}")
    else:
        st.error("❌ File does not exist at specified path")

# File operations
if st.session_state.connected and st.session_state.sftp_manager:
    st.markdown("---")
    st.header("File Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Config", type="primary"):
            # Validate path
            if not is_valid_path(remote_path):
                st.error("❌ Invalid file path format")
            else:
                with st.spinner("Downloading config..."):
                    try:
                        config, message = st.session_state.sftp_manager.download_config(remote_path)
                        if config:
                            st.session_state.current_config = config
                            st.session_state.last_sync = datetime.now()
                            st.success(f"✅ {message}")
                            
                            # Show preview
                            with st.expander("📄 Config Preview"):
                                st.json(config)
                        else:
                            st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
    
    with col2:
        # Upload current config (if exists in session)
        if 'current_config' in st.session_state:
            if st.button("💾 Upload Config", type="secondary"):
                # Validate path
                if not is_valid_path(remote_path):
                    st.error("❌ Invalid file path format")
                else:
                    with st.spinner("Uploading config..."):
                        try:
                            success, message = st.session_state.sftp_manager.upload_config(
                                st.session_state.current_config,
                                remote_path,
                                backup=True
                            )
                            if success:
                                st.session_state.last_sync = datetime.now()
                                st.success(f"✅ {message} (Backup created)")
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.info("Download config first before uploading")

# Last sync info
if st.session_state.last_sync:
    st.markdown("---")
    st.info(f"🕐 Last synchronized: {st.session_state.last_sync.strftime('%Y-%m-%d %H:%M:%S')}")

# Help section
with st.expander("ℹ️ Help & Troubleshooting"):
    st.markdown("""
    ### Common Issues
    
    **Connection Failed:**
    - Verify the host IP address is correct
    - Check that SSH is enabled on the remote server
    - Ensure firewall allows SSH connections (port 22)
    - Verify username and password are correct
    
    **File Not Found:**
    - Use the file browser to navigate and find the correct path
    - Make sure the path is absolute (starts with `/`)
    - Check file permissions on remote server
    
    **Permission Denied:**
    - Ensure your user has read/write access to the file
    - Try using a user with appropriate permissions (e.g., root)
    
    ### Tips
    - The system automatically creates backups before overwriting files
    - Backups are saved with timestamp: `cone_config.yaml.bak.YYYYMMDD_HHMMSS`
    - Use the file browser to explore the remote filesystem
    - Download config before making edits in other pages
    """)
