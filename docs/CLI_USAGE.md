# Historical Map Text Processing CLI Usage Guide

## Overview

The Historical Map Text Processing CLI provides command-line access to all four ICDAR 2025 competition tasks:

- **Task 1**: Word Detection
- **Task 2**: Phrase Detection  
- **Task 3**: Word Detection + Recognition
- **Task 4**: Phrase Detection + Recognition

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Process Single Image

Process a single historical map image:

```bash
# Word detection
python cli.py process image.jpg task1 -o results.txt

# Phrase detection with recognition
python cli.py process map.png task4 -o output.txt --verbose

# Custom configuration
python cli.py process image.tiff task3 -c config/custom.yaml --confidence 0.7
```

### Batch Processing

Process multiple images in a directory:

```bash
# Basic batch processing
python cli.py batch /path/to/images task1 -o /path/to/results

# Parallel processing with 4 workers
python cli.py batch ./maps task4 --parallel 4 --report

# Resume interrupted processing
python cli.py batch ./images task2 --resume --verbose
```

### Configuration Management

Manage system configuration:

```bash
# Show current configuration
python cli.py config show

# Set configuration values
python cli.py config set detection.confidence_threshold 0.8
python cli.py config set recognition.languages '["en", "zh"]'

# Reset to defaults
python cli.py config reset

# Validate configuration file
python cli.py config validate config/my_config.yaml
```

### Evaluation

Evaluate results against ground truth:

```bash
# Evaluate task results
python cli.py evaluate /path/to/results /path/to/ground_truth task1

# Generate detailed report
python cli.py evaluate ./results ./ground_truth task3 -o evaluation_report.json --format json

# Custom metrics
python cli.py evaluate ./results ./gt task2 --metrics precision recall f1 iou
```

### System Information

Display system and model information:

```bash
# Show all information
python cli.py info

# Show specific information
python cli.py info --system
python cli.py info --models
python cli.py info --tasks
python cli.py info --config
```

## Command Reference

### Process Command

```
python cli.py process <input> <task> [options]
```

**Arguments:**
- `input`: Input image file path
- `task`: Competition task (task1, task2, task3, task4)

**Options:**
- `-o, --output`: Output file path (default: auto-generated)
- `-c, --config`: Configuration file path
- `--model-path`: Custom model path
- `--confidence`: Confidence threshold (default: 0.5)
- `--format`: Output format (txt, json)
- `--save-intermediate`: Save intermediate results
- `-v, --verbose`: Enable verbose output

### Batch Command

```
python cli.py batch <input_dir> <task> [options]
```

**Arguments:**
- `input_dir`: Directory containing input images
- `task`: Competition task (task1, task2, task3, task4)

**Options:**
- `-o, --output-dir`: Output directory (default: ./results)
- `-c, --config`: Configuration file path
- `--pattern`: File pattern for images (default: *.{jpg,jpeg,png,tiff,tif})
- `--parallel`: Number of parallel processes (default: 1)
- `--resume`: Resume interrupted processing
- `--report`: Generate processing report
- `-v, --verbose`: Enable verbose output

### Config Command

```
python cli.py config <action> [options]
```

**Actions:**
- `show`: Display current configuration
- `set <key> <value>`: Set configuration value
- `reset`: Reset to default configuration
- `validate <file>`: Validate configuration file

### Evaluate Command

```
python cli.py evaluate <results_dir> <ground_truth> <task> [options]
```

**Arguments:**
- `results_dir`: Directory containing result files
- `ground_truth`: Ground truth annotations file/directory
- `task`: Competition task to evaluate

**Options:**
- `-o, --output`: Evaluation report output file
- `--metrics`: Metrics to calculate (precision, recall, f1, iou, accuracy)
- `--format`: Report format (txt, json, csv)

### Info Command

```
python cli.py info [options]
```

**Options:**
- `--models`: Show model information
- `--config`: Show configuration information
- `--tasks`: Show task information
- `--system`: Show system information

## Configuration

### Configuration Files

The system uses YAML configuration files:

- **Default**: `config/default_config.yaml`
- **CLI-specific**: `config/cli_config.yaml`
- **User**: `~/.historical_map_processor/config.yaml`

### Configuration Keys

Use dot notation to set nested configuration values:

```bash
# Detection settings
python cli.py config set detection.confidence_threshold 0.8
python cli.py config set detection.model_path "models/custom_detector.pt"

# Recognition settings
python cli.py config set recognition.languages '["en", "zh", "fr"]'
python cli.py config set recognition.beam_width 10

# System settings
python cli.py config set system.device "cuda:0"
python cli.py config set system.num_workers 8
```

### Sample Configuration

```yaml
detection:
  model_type: "yolo"
  confidence_threshold: 0.5
  nms_threshold: 0.4

recognition:
  model_type: "crnn"
  languages: ["en", "zh"]
  confidence_threshold: 0.3

preprocessing:
  resize_mode: "maintain_aspect"
  enhance_contrast: true
  noise_reduction: true

output:
  format: "icdar2025"
  save_intermediate: false
```

## Output Formats

### ICDAR 2025 Format (Default)

**Task 1 & 2 (Detection only):**
```
x1,y1,x2,y2,confidence
100,50,300,100,0.85
150,120,250,160,0.92
```

**Task 3 & 4 (Detection + Recognition):**
```
x1,y1,x2,y2,confidence,"text"
100,50,300,100,0.85,"Sample Text"
150,120,250,160,0.92,"Another Text"
```

### JSON Format

```json
{
  "task": "task3",
  "task_name": "Word Detection + Recognition",
  "processing_time": 2.34,
  "detections": [
    {
      "bbox": {"x": 100, "y": 50, "width": 200, "height": 50},
      "confidence": 0.85,
      "type": "word"
    }
  ],
  "recognized_texts": [
    {
      "text": "Sample Text",
      "language": "en",
      "confidence": 0.80,
      "bbox": {"x": 100, "y": 50, "width": 200, "height": 50}
    }
  ]
}
```

## Examples

### Complete Workflow Example

```bash
# 1. Process single image for testing
python cli.py process test_image.jpg task1 --verbose

# 2. Configure system for batch processing
python cli.py config set detection.confidence_threshold 0.7
python cli.py config set system.num_workers 4

# 3. Run batch processing
python cli.py batch ./historical_maps task3 --parallel 4 --report

# 4. Evaluate results
python cli.py evaluate ./results ./ground_truth task3 --format json

# 5. Check system status
python cli.py info --system
```

### Advanced Usage

```bash
# Process with custom model
python cli.py process image.jpg task4 \
  --model-path ./models/custom_model.pt \
  --confidence 0.8 \
  --save-intermediate \
  --format json

# Batch process with specific pattern and resume
python cli.py batch ./maps task2 \
  --pattern "*.tiff" \
  --parallel 8 \
  --resume \
  --output-dir ./results_v2

# Comprehensive evaluation
python cli.py evaluate ./results ./ground_truth task1 \
  --metrics precision recall f1 iou \
  --format csv \
  --output detailed_evaluation.csv
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and Python path is correct
2. **Model Loading**: Check model paths and ensure models are compatible
3. **Memory Issues**: Reduce batch size or parallel workers for large images
4. **Permission Errors**: Ensure write permissions for output directories

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python cli.py process image.jpg task1 --verbose
```

### Log Files

Check log files for detailed error information:
- Default location: `./processing.log`
- Configure in: `config/cli_config.yaml`

## Performance Tips

1. **Parallel Processing**: Use `--parallel` for batch operations
2. **GPU Acceleration**: Configure CUDA device in system settings
3. **Memory Management**: Adjust batch sizes for available memory
4. **Resume Capability**: Use `--resume` for large batch jobs
5. **Intermediate Results**: Save with `--save-intermediate` for debugging