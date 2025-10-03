"""
Configuration manager implementation for loading and saving configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any
from .base_config import SystemConfig, ConfigManagerInterface, DetectionConfig, RecognitionConfig, PreprocessingConfig, EvaluationConfig


class ConfigManager(ConfigManagerInterface):
    """Implementation of configuration management."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config_cache: Dict[str, SystemConfig] = {}
    
    def load_config(self, config_path: str) -> SystemConfig:
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            SystemConfig: Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Check cache first
        if config_path in self._config_cache:
            return self._config_cache[config_path]
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_file.suffix}")
            
            # Convert dict to SystemConfig
            config = self._dict_to_config(config_data)
            
            # Validate and cache
            if self.validate_config(config):
                self._config_cache[config_path] = config
                return config
            else:
                raise ValueError("Configuration validation failed")
                
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config: SystemConfig, config_path: str) -> None:
        """
        Save configuration to JSON or YAML file.
        
        Args:
            config: SystemConfig to save
            config_path: Path where to save the configuration
            
        Raises:
            ValueError: If config is invalid or save fails
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration cannot be saved")
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict
        config_dict = self._config_to_dict(config)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif config_file.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported config format: {config_file.suffix}")
            
            # Update cache
            self._config_cache[config_path] = config
            
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {str(e)}")
    
    def validate_config(self, config: SystemConfig) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: SystemConfig to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Validation is handled by dataclass __post_init__ methods
            # Additional custom validation can be added here
            
            # Check that model paths exist if specified
            if config.detection.model_path and not Path(config.detection.model_path).exists():
                print(f"Warning: Detection model path does not exist: {config.detection.model_path}")
            
            if config.recognition.model_path and not Path(config.recognition.model_path).exists():
                print(f"Warning: Recognition model path does not exist: {config.recognition.model_path}")
            
            # Check device compatibility
            valid_devices = ['cpu', 'cuda', 'mps']
            if config.detection.device not in valid_devices:
                return False
            if config.recognition.device not in valid_devices:
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {str(e)}")
            return False
    
    def get_default_config(self) -> SystemConfig:
        """
        Get default system configuration.
        
        Returns:
            SystemConfig: Default configuration
        """
        return SystemConfig()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig."""
        detection_dict = config_dict.get('detection', {})
        recognition_dict = config_dict.get('recognition', {})
        preprocessing_dict = config_dict.get('preprocessing', {})
        evaluation_dict = config_dict.get('evaluation', {})
        
        detection_config = DetectionConfig(**detection_dict)
        recognition_config = RecognitionConfig(**recognition_dict)
        preprocessing_config = PreprocessingConfig(**preprocessing_dict)
        evaluation_config = EvaluationConfig(**evaluation_dict)
        
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
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig to dictionary."""
        return {
            'detection': {
                'model_architecture': config.detection.model_architecture,
                'model_path': config.detection.model_path,
                'input_size': config.detection.input_size,
                'confidence_threshold': config.detection.confidence_threshold,
                'nms_threshold': config.detection.nms_threshold,
                'batch_size': config.detection.batch_size,
                'device': config.detection.device
            },
            'recognition': {
                'model_architecture': config.recognition.model_architecture,
                'model_path': config.recognition.model_path,
                'max_sequence_length': config.recognition.max_sequence_length,
                'vocabulary_size': config.recognition.vocabulary_size,
                'languages': config.recognition.languages,
                'beam_width': config.recognition.beam_width,
                'device': config.recognition.device
            },
            'preprocessing': {
                'target_size': config.preprocessing.target_size,
                'enhance_contrast': config.preprocessing.enhance_contrast,
                'denoise': config.preprocessing.denoise,
                'normalize': config.preprocessing.normalize,
                'auto_rotate': config.preprocessing.auto_rotate,
                'color_space': config.preprocessing.color_space
            },
            'evaluation': {
                'iou_threshold': config.evaluation.iou_threshold,
                'confidence_threshold': config.evaluation.confidence_threshold,
                'calculate_character_accuracy': config.evaluation.calculate_character_accuracy,
                'calculate_word_accuracy': config.evaluation.calculate_word_accuracy,
                'save_detailed_results': config.evaluation.save_detailed_results,
                'output_format': config.evaluation.output_format
            },
            'log_level': config.log_level,
            'output_dir': config.output_dir,
            'temp_dir': config.temp_dir,
            'max_workers': config.max_workers,
            'memory_limit_gb': config.memory_limit_gb
        }