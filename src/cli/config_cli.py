"""
Configuration management CLI for historical map text processing system.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

from ..config.config_manager import ConfigManager


class ConfigCLI:
    """Command-line interface for configuration management."""
    
    def __init__(self):
        """Initialize the configuration CLI."""
        self.config_manager = ConfigManager()
        self.logger = logging.getLogger(__name__)
    
    def handle_config_command(self, args: argparse.Namespace) -> int:
        """
        Handle configuration-related commands.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if args.config_action == 'show':
                return self._show_config()
            elif args.config_action == 'set':
                return self._set_config_value(args.key, args.value)
            elif args.config_action == 'reset':
                return self._reset_config()
            elif args.config_action == 'validate':
                return self._validate_config(args.config_file)
            else:
                self.logger.error("Unknown configuration action")
                return 1
                
        except Exception as e:
            self.logger.error(f"Configuration command failed: {e}")
            return 1
    
    def _show_config(self) -> int:
        """Show current configuration."""
        try:
            config = self.config_manager.get_current_config()
            
            print("Current Configuration:")
            print("=" * 50)
            self._print_config_dict(config)
            
            # Show config file locations
            print("\nConfiguration Files:")
            print(f"  Default: {self.config_manager.get_default_config_path()}")
            
            user_config_path = self.config_manager.get_user_config_path()
            if user_config_path.exists():
                print(f"  User: {user_config_path}")
            else:
                print(f"  User: {user_config_path} (not found)")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to show configuration: {e}")
            return 1
    
    def _set_config_value(self, key: str, value: str) -> int:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'detection.confidence_threshold')
            value: Configuration value as string
            
        Returns:
            Exit code
        """
        try:
            # Parse value to appropriate type
            parsed_value = self._parse_config_value(value)
            
            # Set the configuration value
            self.config_manager.set_config_value(key, parsed_value)
            
            print(f"Configuration updated: {key} = {parsed_value}")
            
            # Save to user config file
            user_config_path = self.config_manager.get_user_config_path()
            self.config_manager.save_user_config(user_config_path)
            
            print(f"Configuration saved to: {user_config_path}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration value: {e}")
            return 1
    
    def _reset_config(self) -> int:
        """Reset configuration to defaults."""
        try:
            # Confirm with user
            response = input("Are you sure you want to reset configuration to defaults? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Configuration reset cancelled")
                return 0
            
            # Reset to defaults
            self.config_manager.reset_to_defaults()
            
            # Remove user config file if it exists
            user_config_path = self.config_manager.get_user_config_path()
            if user_config_path.exists():
                user_config_path.unlink()
                print(f"Removed user configuration file: {user_config_path}")
            
            print("Configuration reset to defaults")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return 1
    
    def _validate_config(self, config_file: str) -> int:
        """
        Validate a configuration file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Exit code
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.error(f"Configuration file not found: {config_path}")
                return 1
            
            # Load and validate configuration
            is_valid, errors = self.config_manager.validate_config_file(config_path)
            
            if is_valid:
                print(f"Configuration file is valid: {config_path}")
                
                # Show loaded configuration
                config = self.config_manager.load_config(config_path)
                print("\nLoaded Configuration:")
                print("-" * 30)
                self._print_config_dict(config)
                
                return 0
            else:
                print(f"Configuration file is invalid: {config_path}")
                print("\nValidation Errors:")
                for error in errors:
                    print(f"  - {error}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Failed to validate configuration: {e}")
            return 1
    
    def _parse_config_value(self, value: str) -> Any:
        """
        Parse configuration value from string to appropriate type.
        
        Args:
            value: String value to parse
            
        Returns:
            Parsed value with appropriate type
        """
        # Try to parse as JSON first (handles numbers, booleans, lists, dicts)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try common boolean values
        if value.lower() in ['true', 'yes', 'on', '1']:
            return True
        elif value.lower() in ['false', 'no', 'off', '0']:
            return False
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _print_config_dict(self, config: Dict[str, Any], indent: int = 0) -> None:
        """
        Print configuration dictionary with proper formatting.
        
        Args:
            config: Configuration dictionary
            indent: Indentation level
        """
        for key, value in config.items():
            prefix = "  " * indent
            
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_config_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: {json.dumps(value)}")
            else:
                print(f"{prefix}{key}: {value}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get a template configuration with all available options.
        
        Returns:
            Template configuration dictionary
        """
        return {
            "detection": {
                "model_type": "yolo",
                "model_path": "models/detection/yolo_word_detector.pt",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "input_size": [640, 640],
                "batch_size": 1
            },
            "recognition": {
                "model_type": "crnn",
                "model_path": "models/recognition/crnn_recognizer.pt",
                "max_sequence_length": 32,
                "beam_width": 5,
                "languages": ["en", "zh"],
                "confidence_threshold": 0.3
            },
            "preprocessing": {
                "resize_mode": "maintain_aspect",
                "target_size": [1024, 1024],
                "normalize": True,
                "enhance_contrast": True,
                "noise_reduction": True
            },
            "linking": {
                "spatial_threshold": 50,
                "confidence_threshold": 0.4,
                "grouping_enabled": True
            },
            "output": {
                "format": "icdar2025",
                "save_intermediate": False,
                "coordinate_precision": 2
            },
            "system": {
                "device": "auto",
                "num_workers": 4,
                "memory_limit": "8GB",
                "log_level": "INFO"
            }
        }
    
    def create_config_template(self, output_path: str) -> int:
        """
        Create a configuration template file.
        
        Args:
            output_path: Path to save the template
            
        Returns:
            Exit code
        """
        try:
            template = self.get_config_template()
            output_file = Path(output_path)
            
            # Determine format based on file extension
            if output_file.suffix.lower() == '.json':
                with open(output_file, 'w') as f:
                    json.dump(template, f, indent=2)
            else:
                # Default to YAML
                with open(output_file, 'w') as f:
                    yaml.dump(template, f, default_flow_style=False, indent=2)
            
            print(f"Configuration template created: {output_file}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration template: {e}")
            return 1