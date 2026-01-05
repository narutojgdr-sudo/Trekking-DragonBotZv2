"""Troubleshooting and diagnostics page."""
import streamlit as st
import socket
import platform
import sys

st.set_page_config(
    page_title="Troubleshooting",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Troubleshooting & Diagnostics")

# System info
st.header("💻 System Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

with col2:
    st.metric("Platform", platform.system())

with col3:
    st.metric("Architecture", platform.machine())

# Dependencies check
st.header("📦 Dependencies Check")

dependencies = {
    "streamlit": "streamlit",
    "paramiko": "paramiko",
    "pyyaml": "yaml"
}

for name, import_name in dependencies.items():
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        st.success(f"✅ {name}: {version}")
    except ImportError:
        st.error(f"❌ {name}: Not installed")

# Network diagnostics
st.header("🌐 Network Diagnostics")

host = st.text_input("Host to test", "192.168.1.10")
port = st.number_input("Port to test", 1, 65535, 22)

if st.button("🔍 Test Connection"):
    with st.spinner("Testing..."):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                st.success(f"✅ Port {port} is OPEN on {host}")
            else:
                st.error(f"❌ Port {port} is CLOSED on {host}")
                
                st.warning("""
                **Common Solutions:**
                
                **On the remote server:**
                ```bash
                # Check SSH status
                systemctl status ssh
                
                # Start SSH service
                sudo systemctl start ssh
                sudo systemctl enable ssh
                
                # Allow firewall
                sudo ufw allow 22
                sudo ufw status
                
                # Check IP address
                hostname -I
                ip addr show
                ```
                """)
        except socket.gaierror:
            st.error(f"❌ Cannot resolve hostname: {host}")
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")

# Common issues
st.header("❓ Common Issues")

with st.expander("🔴 'S_ISDIR' attribute error"):
    st.markdown("""
    **Error:** `module 'paramiko.sftp_attr' has no attribute 'S_ISDIR'`
    
    **Cause:** Missing `import stat` statement
    
    **Solution:** This should be fixed in the latest version. Update code or reinstall.
    """)

with st.expander("🔴 Connection refused / Port closed"):
    st.markdown("""
    **Error:** `Unable to connect to port 22`
    
    **Solutions:**
    1. Verify SSH service is running: `systemctl status ssh`
    2. Start SSH: `sudo systemctl start ssh`
    3. Check firewall: `sudo ufw allow 22`
    4. Verify correct IP address
    5. Test manually: `ssh user@host`
    """)

with st.expander("🔴 Authentication failed"):
    st.markdown("""
    **Error:** `Authentication failed`
    
    **Solutions:**
    1. Verify username and password are correct
    2. Check if user has SSH access
    3. Try SSH key authentication instead
    4. Check `/etc/ssh/sshd_config` for AllowUsers/DenyUsers
    """)

with st.expander("🔴 Permission denied"):
    st.markdown("""
    **Error:** `Permission denied` when uploading
    
    **Solutions:**
    1. Check file permissions: `ls -l /path/to/file`
    2. Check directory ownership: `ls -ld /path/to/directory`
    3. Ensure user has write permissions
    4. Try with sudo user (root)
    """)

with st.expander("⚠️ use_container_width deprecation"):
    st.markdown("""
    **Warning:** `use_container_width` will be removed after 2025-12-31
    
    **Solution:** This has been fixed in the latest version. All instances have been replaced with `width="stretch"`.
    """)

with st.expander("⚠️ CORS/XSRF configuration warning"):
    st.markdown("""
    **Warning:** `server.enableCORS=false` is not compatible with `server.enableXsrfProtection=true`
    
    **Solution:** This has been fixed. The config now correctly sets `enableCORS=true`.
    """)

# Logs viewer
st.header("📜 Recent Logs")

if st.button("Show Console Logs"):
    st.info("Check terminal/console for detailed logs")
    st.code("""
    # Enable debug logging in your script:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    """)

# Quick links
st.markdown("---")
st.header("🔗 Quick Links")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📖 Documentation
    - [Project README](../../README.md)
    - [Streamlit README](../README.md)
    - [Architecture](../../ARCHITECTURE.md)
    """)

with col2:
    st.markdown("""
    ### 🛠️ Tools
    - Use **Connection** page to test SFTP
    - Check **Home** page for tips
    - Review config in other pages
    """)
