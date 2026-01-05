"""SFTP Manager for remote config file operations."""
import io
import logging
import os
import socket
import stat
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import paramiko
import yaml

logger = logging.getLogger(__name__)


def test_port(host: str, port: int, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test if a port is open on remote host.
    
    Args:
        host: Remote host IP or hostname
        port: Port number to test
        timeout: Connection timeout in seconds
    
    Returns:
        (success, message)
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            return True, f"Port {port} is open"
        else:
            return False, f"Port {port} is closed or filtered"
    except socket.gaierror:
        return False, f"Hostname {host} could not be resolved"
    except socket.timeout:
        return False, f"Connection to {host}:{port} timed out"
    except Exception as e:
        return False, f"Network error: {str(e)}"


class SFTPManager:
    """Manages SFTP connections and file operations."""
    
    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "root",
        password: Optional[str] = None,
        key_path: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize SFTP manager.
        
        Args:
            host: SSH host/IP address
            port: SSH port (default 22)
            username: SSH username
            password: SSH password (if using password auth)
            key_path: Path to SSH private key (if using key auth)
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path
        self.timeout = timeout
        
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.sftp_client: Optional[paramiko.SFTPClient] = None
        self.connected = False
    
    def connect(self) -> Tuple[bool, str]:
        """
        Establish SFTP connection.
        
        Returns:
            (success, message)
        """
        # Test port first
        port_ok, port_msg = test_port(self.host, self.port, timeout=5)
        if not port_ok:
            logger.error(f"❌ {port_msg}")
            return False, f"Network error: {port_msg}"
        
        try:
            # Create SSH client
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect using password or key
            connect_kwargs = {
                'hostname': self.host,
                'port': self.port,
                'username': self.username,
                'timeout': self.timeout
            }
            
            if self.password:
                connect_kwargs['password'] = self.password
            elif self.key_path and os.path.exists(self.key_path):
                connect_kwargs['key_filename'] = self.key_path
            else:
                msg = "No valid authentication method provided"
                logger.error(f"❌ {msg}")
                return False, msg
            
            self.ssh_client.connect(**connect_kwargs)
            
            # Open SFTP session
            self.sftp_client = self.ssh_client.open_sftp()
            self.connected = True
            
            logger.info(f"✅ Connected to {self.host}:{self.port}")
            return True, "Connected successfully"
            
        except paramiko.AuthenticationException:
            msg = "Authentication failed - check username/password"
            logger.error(f"❌ {msg}")
            self.connected = False
            return False, msg
        except paramiko.SSHException as e:
            msg = f"SSH error: {str(e)}"
            logger.error(f"❌ {msg}")
            self.connected = False
            return False, msg
        except Exception as e:
            msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ {msg}")
            self.connected = False
            return False, msg
    
    def disconnect(self):
        """Close SFTP connection."""
        try:
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
            self.connected = False
            logger.info("Disconnected from SFTP server")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def list_directory(self, path: str) -> List[Dict[str, Any]]:
        """
        List files and directories in remote path.
        
        Args:
            path: Remote directory path
            
        Returns:
            List of dicts with file/directory information
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return []
        
        try:
            items = []
            for attr in self.sftp_client.listdir_attr(path):
                is_dir = stat.S_ISDIR(attr.st_mode)
                items.append({
                    'name': attr.filename,
                    'is_dir': is_dir,
                    'size': attr.st_size,
                    'mtime': attr.st_mtime,
                    'permissions': oct(attr.st_mode)[-3:]
                })
            return sorted(items, key=lambda x: (not x['is_dir'], x['name']))
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        """
        Check if remote file exists.
        
        Args:
            remote_path: Remote file path
            
        Returns:
            True if file exists, False otherwise
        """
        if not self.connected or not self.sftp_client:
            return False
        
        try:
            self.sftp_client.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def get_file_info(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about remote file.
        
        Args:
            remote_path: Remote file path
            
        Returns:
            Dict with file information or None if error
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return None
        
        try:
            stat = self.sftp_client.stat(remote_path)
            return {
                'size': stat.st_size,
                'mtime': datetime.fromtimestamp(stat.st_mtime),
                'permissions': oct(stat.st_mode)[-3:],
                'mode': oct(stat.st_mode)
            }
        except FileNotFoundError:
            logger.error(f"File not found: {remote_path}")
            return None
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return None
    
    def download_config(self, remote_path: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Download and parse YAML config from remote server.
        
        Args:
            remote_path: Remote YAML file path
            
        Returns:
            (config_dict, message)
        """
        if not self.connected or not self.sftp_client:
            return None, "Not connected to SFTP server"
        
        try:
            # Download file to memory
            with io.BytesIO() as file_buffer:
                self.sftp_client.getfo(remote_path, file_buffer)
                file_buffer.seek(0)
                
                # Parse YAML
                config = yaml.safe_load(file_buffer)
                logger.info(f"✅ Downloaded config from {remote_path}")
                return config, f"Downloaded successfully from {remote_path}"
                
        except FileNotFoundError:
            msg = f"File not found: {remote_path}"
            logger.error(msg)
            return None, msg
        except yaml.YAMLError as e:
            msg = f"Invalid YAML format: {str(e)}"
            logger.error(msg)
            return None, msg
        except PermissionError:
            msg = f"Permission denied reading {remote_path}"
            logger.error(msg)
            return None, msg
        except Exception as e:
            msg = f"Download failed: {str(e)}"
            logger.error(msg)
            return None, msg
    
    def upload_config(
        self,
        config: Dict[str, Any],
        remote_path: str,
        backup: bool = True
    ) -> Tuple[bool, str]:
        """
        Upload config to remote server.
        
        Args:
            config: Config dictionary to upload
            remote_path: Remote file path
            backup: Whether to create backup before overwriting
            
        Returns:
            (success, message)
        """
        if not self.connected or not self.sftp_client:
            return False, "Not connected to SFTP server"
        
        try:
            # Create backup if requested and file exists
            if backup and self.file_exists(remote_path):
                backup_success, backup_path = self.create_backup(remote_path)
                if not backup_success:
                    logger.warning(f"Backup failed: {backup_path}, proceeding anyway")
            
            # Convert config to YAML
            yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
            
            # Upload to remote
            with io.BytesIO(yaml_content.encode('utf-8')) as file_buffer:
                self.sftp_client.putfo(file_buffer, remote_path)
            
            logger.info(f"✅ Uploaded config to {remote_path}")
            return True, f"Uploaded successfully to {remote_path}"
            
        except PermissionError:
            msg = f"Permission denied writing to {remote_path}"
            logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"Upload failed: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def create_backup(self, remote_path: str) -> Tuple[bool, str]:
        """
        Create backup of remote file with timestamp.
        
        Args:
            remote_path: Remote file path to backup
            
        Returns:
            (success, backup_path_or_error_message)
        """
        if not self.connected or not self.sftp_client:
            return False, "Not connected to SFTP server"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{remote_path}.bak.{timestamp}"
            
            # Copy file
            with io.BytesIO() as buffer:
                self.sftp_client.getfo(remote_path, buffer)
                buffer.seek(0)
                self.sftp_client.putfo(buffer, backup_path)
            
            logger.info(f"✅ Created backup: {backup_path}")
            return True, backup_path
            
        except Exception as e:
            msg = f"Error creating backup: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Download file from remote to local path.
        
        Args:
            remote_path: Remote file path
            local_path: Local file path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return False
        
        try:
            self.sftp_client.get(remote_path, local_path)
            logger.info(f"✅ Downloaded {remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """
        Upload file from local to remote path.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return False
        
        try:
            self.sftp_client.put(local_path, remote_path)
            logger.info(f"✅ Uploaded {local_path} to {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
