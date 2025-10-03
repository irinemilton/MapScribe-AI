# System Integration Documentation

## Overview

The Historical Map Text Processing System is a comprehensive, modular pipeline designed for the ICDAR 2025 competition. This document describes the system architecture, component integration, configuration management, and usage guidelines.

## System Architecture

### Core Components

The system consists of several integrated modules:

1. **Processing Pipeline**
   - Image preprocessing and enhancement
   - Text detection (word and phrase level)
   - Text recognition (single and multi-language)
   - Text linking and spatial association

2. **Model Management**
   - Model loading and version control
   - Training pipeline integration
   - Performance optimization

3. **Configuration Management**
   - System configuration handling
   - Runtime parameter management
   - Configuration validation and persistence

4. **Error Handling & Monitoring**
   - Comprehensive error handling
   - Graceful degradation
   - Performance monitoring
   - System health checks

5. **Output & Evaluation**
   - Competition format compliance
   - Evaluation metrics calculation
   - Result formatting and export

### Integration Flow

```
Input Image → Preprocessing → Detection → Recognition → Linking → Output Formatting
     ↓              ↓            ↓           ↓          ↓            ↓
Configuration → Model Mgmt → Error Handling → Monitoring → Evaluation
```

## System Configuration

### Configuration Hierarchy

The system uses a hierarchical configuration approach:

1. **Built-in Defaults**: Hardcoded fallback values
2. **Default Config File**: `config/default_config.yaml`
3. **User Config File**: `~/.historical_map_processor/config.yaml`
4. **Runtime Config**: `~/.historical_map_processor/runtime_config.json`
5. **Command Line Arguments**: Override specific parameters

### Configuration Sections

#### System Configuration (`SystemConfig`)

```yaml
# Detection settings
detection:
  model_architecture: "yolo"
  model_path: "models/detection/yolo_v8.pt"
  input_size: [640, 640]
  confidence_threshold: 0.5
  nms_threshold: 0.4
  batch_size: 8
  device: "cuda"

# Recognition settings
recognition:
  model_architecture: "crnn"
  model_path: "models/recognition/crnn_model.pt"
  max_sequence_length: 32
  vocabulary_size: 1000
  languages: ["en", "zh"]
  beam_width: 5
  device: "cuda"

# Preprocessing settings
preprocessing:
  target_size: [1024, 1024]
  enhance_contrast: true
  denoise: true
  normalize: true
  auto_rotate: true
  color_space: "RGB"

# Evaluation settings
evaluation:
  iou_threshold: 0.5
  confidence_threshold: 0.5
  calculate_character_accuracy: true
  calculate_word_accuracy: true
  save_detailed_results: true
  output_format: "txt"

# System-wide settings
log_level: "INFO"
output_dir: "./results"
temp_dir: "./temp"
max_workers: 4
memory_limit_gb: 8.0
```

#### Runtime Configuration (`RuntimeConfiguration`)

```json
{
  "batch_size": 1,
  "max_workers": 4,
  "memory_limit_gb": 8.0,
  "confidence_threshold": 0.5,
  "nms_threshold": 0.4,
  "save_intermediate": false,
  "output_format": "txt",
  "enable_gpu": true,
  "enable_optimization": true,
  "enable_monitoring": true,
  "log_level": "INFO",
  "enable_graceful_degradation": true,
  "max_retries": 3
}
```

## Usage Examples

### Basic Usage

```python
from src.core.system_integration import create_system

# Create and initialize system
system = create_system("config/my_config.yaml")

# Process single image
result = system.process_image("image.jpg", "task1")

# Process batch
results = system.process_batch(["img1.jpg", "img2.jpg"], "task3")

# Evaluate results
metrics = system.evaluate_results(results, "ground_truth.json")

# Shutdown system
system.shutdown()
```

### Advanced Configuration

```python
from src.core.system_integration import HistoricalMapSystem
from src.core.system_configuration import SystemConfigurationManager

# Create system with custom configuration
config_manager = SystemConfigurationManager()
system_config = config_manager.load_system_configuration("custom_config.yaml")
runtime_config = config_manager.load_runtime_configuration()

# Update parameters
config_manager.update_system_parameter("detection.confidence_threshold", 0.7)
config_manager.update_runtime_parameter("batch_size", 4)

# Initialize system
system = HistoricalMapSystem()
system.initialize(system_config)

# Process with updated configuration
result = system.process_image("image.jpg", "task2")
```

### Command Line Interface

```bash
# Process single image
python -m src.cli process image.jpg task1 -o results.txt --confidence 0.7

# Batch processing
python -m src.cli batch /path/to/images task3 -o /path/to/results --parallel 4

# Configuration management
python -m src.cli config show
python -m src.cli config set detection.confidence_threshold 0.8

# Evaluation
python -m src.cli evaluate /path/to/results /path/to/ground_truth task1
```

### REST API

```bash
# Start API server
python -m src.api.main_api

# Process image via API
curl -X POST -F "file=@image.jpg" http://localhost:5000/process/task1

# Batch processing
curl -X POST -F "files=@img1.jpg" -F "files=@img2.jpg" http://localhost:5000/batch/task3
```

## Error Handling

### Error Types

1. **Configuration Errors**: Invalid parameters, missing files
2. **Processing Errors**: Image loading, model inference failures
3. **System Errors**: Memory issues, device unavailability
4. **Validation Errors**: Invalid input formats, corrupted data

### Graceful Degradation

The system implements graceful degradation strategies:

- **Model Fallback**: Switch to backup models on failure
- **Quality Reduction**: Lower processing quality to continue operation
- **Partial Results**: Return available results even if some processing fails
- **Error Recovery**: Automatic retry with different parameters

### Error Logging

```python
# Error handling example
try:
    result = system.process_image("image.jpg", "task1")
except ProcessingError as e:
    logger.error(f"Processing failed: {e}")
    # System automatically attempts graceful degradation
    degraded_result = system.get_degraded_result()
except SystemError as e:
    logger.critical(f"System error: {e}")
    # System shutdown may be required
```

## Performance Monitoring

### Metrics Tracked

- **Processing Time**: Per image and batch processing times
- **Memory Usage**: Peak and average memory consumption
- **GPU Utilization**: GPU memory and compute usage
- **Throughput**: Images processed per second
- **Error Rates**: Success/failure ratios
- **Model Performance**: Inference times and accuracy metrics

### Monitoring Interface

```python
# Get system status
status = system.get_system_status()
print(f"Total processed: {status['processing_stats']['total_processed']}")
print(f"Success rate: {status['processing_stats']['successful'] / status['processing_stats']['total_processed']}")

# Get real-time metrics
metrics = system.monitor.get_current_metrics()
print(f"Current memory usage: {metrics['memory_usage_gb']:.2f} GB")
print(f"GPU utilization: {metrics['gpu_utilization']:.1f}%")
```

## Model Management

### Model Loading

```python
# Load specific model versions
system.model_manager.load_detection_model("models/yolo_v8_custom.pt", config)
system.model_manager.load_recognition_model("models/crnn_multilingual.pt", config)

# Switch model versions
system.model_manager.switch_model_version("detection", "v2.0")

# Get model information
metadata = system.model_manager.get_model_metadata("detection")
```

### Training Integration

```python
# Train detection model
from src.models.training_pipeline import TrainingPipeline

trainer = TrainingPipeline()
new_model = trainer.train_detection_model(training_dataset)

# Evaluate and deploy
metrics = trainer.evaluate_model_performance(new_model, test_dataset)
if metrics['f1_score'] > 0.85:
    system.model_manager.deploy_model(new_model, "detection")
```

## Competition Task Support

### Task Definitions

- **Task 1**: Word Detection - Detect individual word boundaries
- **Task 2**: Phrase Detection - Detect phrase-level text regions
- **Task 3**: Word Detection + Recognition - Detect and recognize words
- **Task 4**: Phrase Detection + Recognition - Detect and recognize phrases

### Output Formats

Each task produces competition-compliant output:

```
# Task 1 & 2 (Detection only)
x,y,width,height,confidence

# Task 3 & 4 (Detection + Recognition)
x,y,width,height,confidence,text
```

### Evaluation Metrics

- **Detection**: Precision, Recall, F1-score, IoU
- **Recognition**: Character accuracy, Word accuracy, BLEU score
- **End-to-end**: Combined detection and recognition metrics

## Deployment Considerations

### System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **CUDA**: 11.0+ (for GPU acceleration)
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: 10GB+ for models and temporary files

### Performance Optimization

1. **GPU Acceleration**: Enable CUDA for model inference
2. **Batch Processing**: Process multiple images simultaneously
3. **Model Optimization**: Use TensorRT or ONNX for faster inference
4. **Memory Management**: Configure appropriate batch sizes and memory limits
5. **Parallel Processing**: Utilize multiple CPU cores for preprocessing

### Scalability

- **Horizontal Scaling**: Deploy multiple instances behind load balancer
- **Vertical Scaling**: Increase memory and GPU resources
- **Model Serving**: Use dedicated model serving frameworks
- **Caching**: Implement result caching for repeated requests

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Lower input image resolution
   - Enable memory optimization

2. **Model Loading Failures**
   - Check model file paths
   - Verify model compatibility
   - Update model versions

3. **Poor Recognition Accuracy**
   - Adjust confidence thresholds
   - Enable preprocessing enhancements
   - Use appropriate language models

4. **Slow Processing**
   - Enable GPU acceleration
   - Optimize batch sizes
   - Use performance profiling

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Enable detailed monitoring
system.monitor.enable_detailed_monitoring()

# Save intermediate results
system.config.save_intermediate = True
```

## API Reference

### Core Classes

- `HistoricalMapSystem`: Main system integration class
- `SystemConfigurationManager`: Configuration management
- `ModelManager`: Model loading and management
- `CompetitionEvaluator`: Evaluation and metrics
- `ICDAR2025Formatter`: Output formatting

### Key Methods

- `system.initialize(config)`: Initialize system with configuration
- `system.process_image(path, task)`: Process single image
- `system.process_batch(paths, task)`: Process multiple images
- `system.evaluate_results(results, ground_truth)`: Evaluate results
- `system.get_system_status()`: Get system status and metrics
- `system.shutdown()`: Gracefully shutdown system

## Contributing

### Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Check code quality: `flake8 src/`

### Adding New Components

1. Implement required interfaces
2. Add configuration parameters
3. Include error handling
4. Write comprehensive tests
5. Update documentation

### Testing

- Unit tests for individual components
- Integration tests for pipeline
- Performance benchmarks
- Competition format validation