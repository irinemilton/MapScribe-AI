"""
System configuration management for the Historical Map Text Processing System.
Handles parameter management, validation, and runtime configuration updates.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import logging
from datetime import datetime

from ..config.base_config import SystemConfig, DetectionConfig, RecognitionConfig, PreprocessingConfig, EvaluationConfig
from ..config.config_manager import ConfigManager


@dataclass
class RuntimeConfiguration:
    """Runtime configuration that can be updated during system operation."""
    
    # Processing parameters
    batch_size: int = 1
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Quality settings
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    
    # Output settings
    save_intermediate: bool = False
    output_format: str = 'txt'
    
    # Performance settings
    enable_gpu: bool = True
    enable_optimization: bool = True
    
    # Monitoring settings
    enable_monitoring: bool = True
    log_level: str = 'INFO'
    
    # Error handling settings
    enable_graceful_degradation: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0.0 and 1.0")
        if self.output_format not in ['txt', 'json', 'xml']:
            raise ValueError("output_format must be 'txt', 'json', or 'xml'")
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")


class SystemConfigurationManager:
    """
    Manages system configuration including loading, saving, validation,
    and runtime updates of configuration parameters.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        
        # Configuration state
        self._system_config: Optional[SystemConfig] = None
        self._runtime_config: Optional[RuntimeConfiguration] = None
        self._config_history: List[Dict[str, Any]] = []
        
        # Configuration file paths
        self._default_config_path = Path("config/default_config.yaml")
        self._user_config_path = Path.home() / ".historical_map_processor" / "config.yaml"
        self._runtime_config_path = Path.home() / ".historical_map_processor" / "runtime_config.json"
        
        # Ensure config directories exist
        self._user_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_system_configuration(self, config_path: Optional[str] = None) -> SystemConfig:
        """
        Load system configuration from file or use defaults.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            SystemConfig: Loaded system configuration
        """
        try:
            if config_path:
                # Load from specified path
                config = self.config_manager.load_config(config_path)
                self.logger.info(f"Loaded configuration from: {config_path}")
            elif self._user_config_path.exists():
                # Load from user config
                config = self.config_manager.load_config(str(self._user_config_path))
                self.logger.info(f"Loaded user configuration from: {self._user_config_path}")
            elif self._default_config_path.exists():
                # Load from default config
                config = self.config_manager.load_config(str(self._default_config_path))
                self.logger.info(f"Loaded default configuration from: {self._default_config_path}")
            else:
                # Use built-in defaults
                config = self.config_manager.get_default_config()
                self.logger.info("Using built-in default configuration")
            
            # Validate configuration
            if not self.config_manager.validate_config(config):
                raise ValueError("Configuration validation failed")
            
            self._system_config = config
            self._add_to_history("load_system_config", {"source": config_path or "default"})
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load system configuration: {e}")
            raise
    
    def load_runtime_configuration(self) -> RuntimeConfiguration:
        """
        Load runtime configuration from file or use defaults.
        
        Returns:
            RuntimeConfiguration: Loaded runtime configuration
        """
        try:
            if self._runtime_config_path.exists():
                with open(self._runtime_config_path, 'r') as f:
                    config_data = json.load(f)
                
                config = RuntimeConfiguration(**config_data)
                self.logger.info(f"Loaded runtime configuration from: {self._runtime_config_path}")
            else:
                config = RuntimeConfiguration()
                self.logger.info("Using default runtime configuration")
            
            self._runtime_config = config
            self._add_to_history("load_runtime_config", {})
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load runtime configuration: {e}")
            # Return default configuration on error
            config = RuntimeConfiguration()
            self._runtime_config = config
            return config
    
    def save_system_configuration(self, config: SystemConfig, 
                                config_path: Optional[str] = None) -> None:
        """
        Save system configuration to file.
        
        Args:
            config: SystemConfig to save
            config_path: Optional path to save configuration
        """
        try:
            save_path = config_path or str(self._user_config_path)
            self.config_manager.save_config(config, save_path)
            
            self._system_config = config
            self._add_to_history("save_system_config", {"path": save_path})
            
            self.logger.info(f"Saved system configuration to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save system configuration: {e}")
            raise
    
    def save_runtime_configuration(self, config: RuntimeConfiguration) -> None:
        """
        Save runtime configuration to file.
        
        Args:
            config: RuntimeConfiguration to save
        """
        try:
            config_data = asdict(config)
            
            with open(self._runtime_config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self._runtime_config = config
            self._add_to_history("save_runtime_config", {})
            
            self.logger.info(f"Saved runtime configuration to: {self._runtime_config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save runtime configuration: {e}")
            raise
    
    def update_system_parameter(self, parameter_path: str, value: Any) -> None:
        """
        Update a system configuration parameter using dot notation.
        
        Args:
            parameter_path: Parameter path in dot notation (e.g., 'detection.confidence_threshold')
            value: New parameter value
        """
        if not self._system_config:
            raise RuntimeError("System configuration not loaded")
        
        try:
            # Parse parameter path
            path_parts = parameter_path.split('.')
            
            if len(path_parts) < 2:
                raise ValueError("Parameter path must have at least 2 parts (section.parameter)")
            
            section = path_parts[0]
            param = '.'.join(path_parts[1:])
            
            # Update the appropriate section
            if section == 'detection':
                self._update_detection_config(param, value)
            elif section == 'recognition':
                self._update_recognition_config(param, value)
            elif section == 'preprocessing':
                self._update_preprocessing_config(param, value)
            elif section == 'evaluation':
                self._update_evaluation_config(param, value)
            else:
                # Update system-level parameter
                if hasattr(self._system_config, section):
                    setattr(self._system_config, section, value)
                else:
                    raise ValueError(f"Unknown system parameter: {section}")
            
            self._add_to_history("update_parameter", {
                "parameter": parameter_path,
                "value": value
            })
            
            self.logger.info(f"Updated parameter {parameter_path} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to update parameter {parameter_path}: {e}")
            raise
    
    def update_runtime_parameter(self, parameter: str, value: Any) -> None:
        """
        Update a runtime configuration parameter.
        
        Args:
            parameter: Parameter name
            value: New parameter value
        """
        if not self._runtime_config:
            raise RuntimeError("Runtime configuration not loaded")
        
        try:
            if hasattr(self._runtime_config, parameter):
                # Validate value before setting
                old_value = getattr(self._runtime_config, parameter)
                setattr(self._runtime_config, parameter, value)
                
                # Validate the updated configuration
                try:
                    self._runtime_config.__post_init__()
                except ValueError as e:
                    # Revert on validation error
                    setattr(self._runtime_config, parameter, old_value)
                    raise ValueError(f"Invalid value for {parameter}: {e}")
                
                self._add_to_history("update_runtime_parameter", {
                    "parameter": parameter,
                    "old_value": old_value,
                    "new_value": value
                })
                
                self.logger.info(f"Updated runtime parameter {parameter} = {value}")
            else:
                raise ValueError(f"Unknown runtime parameter: {parameter}")
                
        except Exception as e:
            self.logger.error(f"Failed to update runtime parameter {parameter}: {e}")
            raise
    
    def get_system_configuration(self) -> Optional[SystemConfig]:
        """Get current system configuration."""
        return self._system_config
    
    def get_runtime_configuration(self) -> Optional[RuntimeConfiguration]:
        """Get current runtime configuration."""
        return self._runtime_config
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        summary = {
            'system_config_loaded': self._system_config is not None,
            'runtime_config_loaded': self._runtime_config is not None,
            'config_files': {
                'default_config': str(self._default_config_path),
                'user_config': str(self._user_config_path),
                'runtime_config': str(self._runtime_config_path)
            },
            'file_exists': {
                'default_config': self._default_config_path.exists(),
                'user_config': self._user_config_path.exists(),
                'runtime_config': self._runtime_config_path.exists()
            }
        }
        
        if self._system_config:
            summary['system_config'] = {
                'detection_model': self._system_config.detection.model_architecture,
                'recognition_model': self._system_config.recognition.model_architecture,
                'device': self._system_config.detection.device,
                'log_level': self._system_config.log_level
            }
        
        if self._runtime_config:
            summary['runtime_config'] = asdict(self._runtime_config)
        
        return summary
    
    def validate_configuration(self, config: Union[SystemConfig, RuntimeConfiguration]) -> bool:
        """
        Validate configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(config, SystemConfig):
                return self.config_manager.validate_config(config)
            elif isinstance(config, RuntimeConfiguration):
                # Validation is done in __post_init__
                config.__post_init__()
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def reset_to_defaults(self, config_type: str = 'both') -> None:
        """
        Reset configuration to defaults.
        
        Args:
            config_type: Type of config to reset ('system', 'runtime', or 'both')
        """
        try:
            if config_type in ['system', 'both']:
                self._system_config = self.config_manager.get_default_config()
                self.logger.info("Reset system configuration to defaults")
            
            if config_type in ['runtime', 'both']:
                self._runtime_config = RuntimeConfiguration()
                self.logger.info("Reset runtime configuration to defaults")
            
            self._add_to_history("reset_to_defaults", {"config_type": config_type})
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            raise
    
    def export_configuration(self, export_path: str, format_type: str = 'yaml') -> None:
        """
        Export current configuration to file.
        
        Args:
            export_path: Path to export file
            format_type: Export format ('yaml' or 'json')
        """
        try:
            export_data = {
                'system_config': asdict(self._system_config) if self._system_config else None,
                'runtime_config': asdict(self._runtime_config) if self._runtime_config else None,
                'export_timestamp': datetime.now().isoformat(),
                'config_history': self._config_history[-10:]  # Last 10 changes
            }
            
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'yaml':
                with open(export_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
            elif format_type == 'json':
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Exported configuration to: {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            raise
    
    def import_configuration(self, import_path: str) -> None:
        """
        Import configuration from file.
        
        Args:
            import_path: Path to import file
        """
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            # Load import data
            if import_path.suffix.lower() in ['.yaml', '.yml']:
                with open(import_path, 'r') as f:
                    import_data = yaml.safe_load(f)
            elif import_path.suffix.lower() == '.json':
                with open(import_path, 'r') as f:
                    import_data = json.load(f)
            else:
                raise ValueError(f"Unsupported import format: {import_path.suffix}")
            
            # Import system configuration
            if 'system_config' in import_data and import_data['system_config']:
                system_config_data = import_data['system_config']
                self._system_config = self._dict_to_system_config(system_config_data)
            
            # Import runtime configuration
            if 'runtime_config' in import_data and import_data['runtime_config']:
                runtime_config_data = import_data['runtime_config']
                self._runtime_config = RuntimeConfiguration(**runtime_config_data)
            
            self._add_to_history("import_configuration", {"source": str(import_path)})
            
            self.logger.info(f"Imported configuration from: {import_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            raise
    
    def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self._config_history.copy()
    
    def _update_detection_config(self, param: str, value: Any) -> None:
        """Update detection configuration parameter."""
        if hasattr(self._system_config.detection, param):
            setattr(self._system_config.detection, param, value)
        else:
            raise ValueError(f"Unknown detection parameter: {param}")
    
    def _update_recognition_config(self, param: str, value: Any) -> None:
        """Update recognition configuration parameter."""
        if hasattr(self._system_config.recognition, param):
            setattr(self._system_config.recognition, param, value)
        else:
            raise ValueError(f"Unknown recognition parameter: {param}")
    
    def _update_preprocessing_config(self, param: str, value: Any) -> None:
        """Update preprocessing configuration parameter."""
        if hasattr(self._system_config.preprocessing, param):
            setattr(self._system_config.preprocessing, param, value)
        else:
            raise ValueError(f"Unknown preprocessing parameter: {param}")
    
    def _update_evaluation_config(self, param: str, value: Any) -> None:
        """Update evaluation configuration parameter."""
        if hasattr(self._system_config.evaluation, param):
            setattr(self._system_config.evaluation, param, value)
        else:
            raise ValueError(f"Unknown evaluation parameter: {param}")
    
    def _dict_to_system_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object."""
        detection_config = DetectionConfig(**config_dict.get('detection', {}))
        recognition_config = RecognitionConfig(**config_dict.get('recognition', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # System-wide settings
        system_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['detection', 'recognition', 'preprocessing', 'evaluation']}
        
        return SystemConfig(
            detection=detection_config,
            recognition=recognition_config,
            preprocessing=preprocessing_config,
            evaluation=evaluation_config,
            **system_settings
        )
    
    def _add_to_history(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to configuration history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        self._config_history.append(entry)
        
        # Keep only last 100 entries
        if len(self._config_history) > 100:
            self._config_history = self._config_history[-100:]