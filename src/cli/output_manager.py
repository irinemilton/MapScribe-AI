"""
Output management for CLI operations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import logging


class OutputManager:
    """Manages output formatting and saving for CLI operations."""
    
    def __init__(self):
        """Initialize the output manager."""
        self.logger = logging.getLogger(__name__)
    
    def save_output(self, content: str, output_path: Path, format_type: str = 'txt') -> None:
        """
        Save output content to file.
        
        Args:
            content: Content to save
            output_path: Path to save the file
            format_type: Output format ('txt', 'json')
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate encoding
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Output saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save output to {output_path}: {e}")
            raise
    
    def format_json_output(self, data: Dict[str, Any], pretty: bool = True) -> str:
        """
        Format data as JSON string.
        
        Args:
            data: Data to format
            pretty: Whether to use pretty formatting
            
        Returns:
            JSON formatted string
        """
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False)
    
    def create_output_directory(self, base_path: Path, task: str) -> Path:
        """
        Create organized output directory structure.
        
        Args:
            base_path: Base output path
            task: Competition task identifier
            
        Returns:
            Created output directory path
        """
        output_dir = base_path / task
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_output_filename(self, input_path: Path, task: str, format_type: str = 'txt') -> str:
        """
        Generate standardized output filename.
        
        Args:
            input_path: Input file path
            task: Competition task
            format_type: Output format
            
        Returns:
            Generated filename
        """
        stem = input_path.stem
        extension = 'json' if format_type == 'json' else 'txt'
        return f"{stem}_{task}_results.{extension}"