"""SFTP Manager for remote config file operations."""
import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import paramiko
import yaml

logger = logging.getLogger(__name__)


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
    
    def connect(self) -> bool:
        """
        Establish SFTP connection.
        
        Returns:
            True if connection successful, False otherwise
        """
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
                logger.error("No valid authentication method provided")
                return False
            
            self.ssh_client.connect(**connect_kwargs)
            
            # Open SFTP session
            self.sftp_client = self.ssh_client.open_sftp()
            self.connected = True
            
            logger.info(f"✅ Connected to {self.host}:{self.port}")
            return True
            
        except paramiko.AuthenticationException:
            logger.error("❌ Authentication failed")
            self.connected = False
            return False
        except paramiko.SSHException as e:
            logger.error(f"❌ SSH error: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
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
                is_dir = paramiko.sftp_attr.S_ISDIR(attr.st_mode)
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
    
    def download_config(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """
        Download and parse YAML config from remote server.
        
        Args:
            remote_path: Remote YAML file path
            
        Returns:
            Parsed config dict or None if error
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return None
        
        try:
            # Download file to memory
            with io.BytesIO() as file_buffer:
                self.sftp_client.getfo(remote_path, file_buffer)
                file_buffer.seek(0)
                
                # Parse YAML
                config = yaml.safe_load(file_buffer)
                logger.info(f"✅ Downloaded config from {remote_path}")
                return config
                
        except FileNotFoundError:
            logger.error(f"File not found: {remote_path}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading config: {e}")
            return None
    
    def upload_config(
        self,
        config: Dict[str, Any],
        remote_path: str,
        backup: bool = True
    ) -> bool:
        """
        Upload config to remote server.
        
        Args:
            config: Config dictionary to upload
            remote_path: Remote file path
            backup: Whether to create backup before overwriting
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return False
        
        try:
            # Create backup if requested and file exists
            if backup and self.file_exists(remote_path):
                backup_path = self.create_backup(remote_path)
                if not backup_path:
                    logger.warning("Failed to create backup, proceeding anyway")
            
            # Convert config to YAML
            yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
            
            # Upload to remote
            with io.BytesIO(yaml_content.encode('utf-8')) as file_buffer:
                self.sftp_client.putfo(file_buffer, remote_path)
            
            logger.info(f"✅ Uploaded config to {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading config: {e}")
            return False
    
    def create_backup(self, remote_path: str) -> Optional[str]:
        """
        Create backup of remote file with timestamp.
        
        Args:
            remote_path: Remote file path to backup
            
        Returns:
            Backup file path or None if error
        """
        if not self.connected or not self.sftp_client:
            logger.error("Not connected to SFTP server")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{remote_path}.bak.{timestamp}"
            
            # Copy file
            with io.BytesIO() as buffer:
                self.sftp_client.getfo(remote_path, buffer)
                buffer.seek(0)
                self.sftp_client.putfo(buffer, backup_path)
            
            logger.info(f"✅ Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
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
