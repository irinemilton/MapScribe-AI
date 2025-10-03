"""
File management for the REST API.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import logging
from datetime import datetime, timedelta


class APIFileManager:
    """Manages file operations for the API."""
    
    def __init__(self, upload_folder: str = 'uploads', results_folder: str = 'api_results'):
        """
        Initialize the file manager.
        
        Args:
            upload_folder: Directory for uploaded files
            results_folder: Directory for result files
        """
        self.upload_folder = Path(upload_folder)
        self.results_folder = Path(results_folder)
        self.metadata_folder = self.results_folder / 'metadata'
        
        # Create directories
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.metadata_folder.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def save_uploaded_file(self, file: FileStorage, result_id: str) -> Path:
        """
        Save uploaded file to disk.
        
        Args:
            file: Uploaded file
            result_id: Unique result identifier
            
        Returns:
            Path to saved file
        """
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Create unique filename with result ID
        file_extension = Path(filename).suffix
        safe_filename = f"{result_id}{file_extension}"
        
        # Save file
        file_path = self.upload_folder / safe_filename
        file.save(str(file_path))
        
        self.logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def save_result(self, result_id: str, content: str, task: str) -> Path:
        """
        Save processing result to file.
        
        Args:
            result_id: Unique result identifier
            content: Result content
            task: Competition task
            
        Returns:
            Path to saved result file
        """
        # Create result filename
        result_filename = f"{result_id}_{task}_results.txt"
        result_path = self.results_folder / result_filename
        
        # Save result content
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save metadata
        self._save_result_metadata(result_id, {
            'task': task,
            'result_file': str(result_path),
            'created_at': datetime.now().isoformat(),
            'file_size': result_path.stat().st_size
        })
        
        self.logger.info(f"Saved result file: {result_path}")
        return result_path
    
    def get_result_file_path(self, result_id: str) -> Optional[Path]:
        """
        Get path to result file.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Path to result file or None if not found
        """
        metadata = self._load_result_metadata(result_id)
        if not metadata:
            return None
        
        result_file = metadata.get('result_file')
        if not result_file:
            return None
        
        result_path = Path(result_file)
        return result_path if result_path.exists() else None
    
    def get_result_info(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get result information.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Result information dictionary or None if not found
        """
        metadata = self._load_result_metadata(result_id)
        if not metadata:
            return None
        
        result_path = Path(metadata['result_file'])
        if not result_path.exists():
            return None
        
        # Load result content for statistics
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count lines (detections/recognitions)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            info = {
                'result_id': result_id,
                'task': metadata['task'],
                'created_at': metadata['created_at'],
                'file_size': metadata['file_size'],
                'line_count': len(lines),
                'download_url': f'/results/{result_id}/download',
                'status': 'available'
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to load result info for {result_id}: {e}")
            return None
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old uploaded and result files.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Clean up uploaded files
        for file_path in self.upload_folder.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up old upload: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to clean up {file_path}: {e}")
        
        # Clean up result files and metadata
        for metadata_file in self.metadata_folder.glob('*.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                created_at = datetime.fromisoformat(metadata['created_at'])
                if created_at < cutoff_time:
                    # Remove result file
                    result_file = Path(metadata['result_file'])
                    if result_file.exists():
                        result_file.unlink()
                        cleaned_count += 1
                    
                    # Remove metadata file
                    metadata_file.unlink()
                    cleaned_count += 1
                    
                    self.logger.info(f"Cleaned up old result: {metadata['result_file']}")
                    
            except Exception as e:
                self.logger.error(f"Failed to clean up metadata {metadata_file}: {e}")
        
        self.logger.info(f"Cleanup completed: {cleaned_count} files removed")
        return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics dictionary
        """
        stats = {
            'upload_folder': str(self.upload_folder),
            'results_folder': str(self.results_folder),
            'upload_count': 0,
            'upload_size': 0,
            'result_count': 0,
            'result_size': 0,
            'total_size': 0
        }
        
        # Count uploads
        if self.upload_folder.exists():
            for file_path in self.upload_folder.iterdir():
                if file_path.is_file():
                    stats['upload_count'] += 1
                    stats['upload_size'] += file_path.stat().st_size
        
        # Count results
        if self.results_folder.exists():
            for file_path in self.results_folder.iterdir():
                if file_path.is_file() and file_path.suffix == '.txt':
                    stats['result_count'] += 1
                    stats['result_size'] += file_path.stat().st_size
        
        stats['total_size'] = stats['upload_size'] + stats['result_size']
        
        return stats
    
    def delete_result(self, result_id: str) -> bool:
        """
        Delete a specific result and its files.
        
        Args:
            result_id: Result identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Load metadata
            metadata = self._load_result_metadata(result_id)
            if not metadata:
                return False
            
            # Delete result file
            result_file = Path(metadata['result_file'])
            if result_file.exists():
                result_file.unlink()
            
            # Delete uploaded file if it exists
            upload_files = list(self.upload_folder.glob(f"{result_id}.*"))
            for upload_file in upload_files:
                upload_file.unlink()
            
            # Delete metadata
            metadata_file = self.metadata_folder / f"{result_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            self.logger.info(f"Deleted result: {result_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete result {result_id}: {e}")
            return False
    
    def _save_result_metadata(self, result_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save result metadata to file.
        
        Args:
            result_id: Result identifier
            metadata: Metadata dictionary
        """
        metadata_file = self.metadata_folder / f"{result_id}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_result_metadata(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Load result metadata from file.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.metadata_folder / f"{result_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {result_id}: {e}")
            return None
    
    def create_backup(self, backup_path: str) -> bool:
        """
        Create backup of all results.
        
        Args:
            backup_path: Path for backup archive
            
        Returns:
            True if backup created successfully, False otherwise
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Create archive of results folder
            shutil.make_archive(
                str(backup_dir.with_suffix('')),
                'zip',
                str(self.results_folder.parent),
                str(self.results_folder.name)
            )
            
            self.logger.info(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore from backup archive.
        
        Args:
            backup_path: Path to backup archive
            
        Returns:
            True if restored successfully, False otherwise
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return False
            
            # Extract archive
            shutil.unpack_archive(
                str(backup_file),
                str(self.results_folder.parent)
            )
            
            self.logger.info(f"Restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False