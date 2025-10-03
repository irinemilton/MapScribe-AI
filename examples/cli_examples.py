#!/usr/bin/env python3
"""
Example scripts demonstrating CLI usage for historical map text processing.
"""

import subprocess
import sys
from pathlib import Path


def run_cli_command(command: str) -> int:
    """
    Run a CLI command and return the exit code.
    
    Args:
        command: CLI command to run
        
    Returns:
        Exit code
    """
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    print(f"Exit code: {result.returncode}\n")
    return result.returncode


def example_single_image_processing():
    """Example: Process a single image for each task."""
    print("=== Single Image Processing Examples ===\n")
    
    # Create sample image path (would be actual image in real usage)
    sample_image = "examples/sample_map.jpg"
    
    # Task 1: Word Detection
    run_cli_command(f"python cli.py process {sample_image} task1 -o results_task1.txt --verbose")
    
    # Task 2: Phrase Detection
    run_cli_command(f"python cli.py process {sample_image} task2 -o results_task2.txt")
    
    # Task 3: Word Detection + Recognition
    run_cli_command(f"python cli.py process {sample_image} task3 --format json -o results_task3.json")
    
    # Task 4: Phrase Detection + Recognition with custom confidence
    run_cli_command(f"python cli.py process {sample_image} task4 --confidence 0.7 --save-intermediate")


def example_batch_processing():
    """Example: Batch process multiple images."""
    print("=== Batch Processing Examples ===\n")
    
    # Basic batch processing
    run_cli_command("python cli.py batch examples/ task1 -o batch_results/")
    
    # Parallel batch processing with report
    run_cli_command("python cli.py batch examples/ task3 --parallel 2 --report --verbose")
    
    # Resume interrupted batch processing
    run_cli_command("python cli.py batch examples/ task4 --resume --output-dir resumed_results/")


def example_configuration_management():
    """Example: Configuration management."""
    print("=== Configuration Management Examples ===\n")
    
    # Show current configuration
    run_cli_command("python cli.py config show")
    
    # Set configuration values
    run_cli_command("python cli.py config set detection.confidence_threshold 0.8")
    run_cli_command("python cli.py config set recognition.languages '[\"en\", \"zh\"]'")
    run_cli_command("python cli.py config set system.device cuda:0")
    
    # Validate configuration file
    run_cli_command("python cli.py config validate config/default_config.yaml")
    
    # Show updated configuration
    run_cli_command("python cli.py config show")


def example_evaluation():
    """Example: Evaluate results against ground truth."""
    print("=== Evaluation Examples ===\n")
    
    # Basic evaluation
    run_cli_command("python cli.py evaluate batch_results/ examples/ground_truth/ task1")
    
    # Detailed evaluation with custom metrics
    run_cli_command("python cli.py evaluate batch_results/ examples/ground_truth/ task3 --metrics precision recall f1 --format json -o evaluation.json")
    
    # CSV format evaluation
    run_cli_command("python cli.py evaluate batch_results/ examples/ground_truth/ task4 --format csv -o evaluation.csv")


def example_system_info():
    """Example: Display system information."""
    print("=== System Information Examples ===\n")
    
    # Show all information
    run_cli_command("python cli.py info")
    
    # Show specific information
    run_cli_command("python cli.py info --system")
    run_cli_command("python cli.py info --models")
    run_cli_command("python cli.py info --tasks")
    run_cli_command("python cli.py info --config")


def example_advanced_usage():
    """Example: Advanced CLI usage scenarios."""
    print("=== Advanced Usage Examples ===\n")
    
    # Process with custom configuration file
    run_cli_command("python cli.py process examples/sample_map.jpg task3 -c config/custom_config.yaml --verbose")
    
    # Batch process with specific file pattern
    run_cli_command("python cli.py batch examples/ task2 --pattern '*.tiff' --parallel 4")
    
    # Process and save intermediate results
    run_cli_command("python cli.py process examples/sample_map.jpg task4 --save-intermediate --format json")


def create_sample_files():
    """Create sample files for demonstration."""
    print("=== Creating Sample Files ===\n")
    
    # Create sample configuration
    sample_config = """
detection:
  confidence_threshold: 0.6
  model_path: "models/custom_detector.pt"

recognition:
  languages: ["en", "zh"]
  confidence_threshold: 0.4

preprocessing:
  enhance_contrast: true
  noise_reduction: true
"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "custom_config.yaml", "w") as f:
        f.write(sample_config)
    
    print("Created sample configuration file: config/custom_config.yaml")
    
    # Create sample ground truth directory structure
    gt_dir = Path("examples/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample ground truth file
    sample_gt = """100,50,300,100,1.0,"Sample Text"
150,120,250,160,1.0,"Another Text"
"""
    
    with open(gt_dir / "sample_map_task3_gt.txt", "w") as f:
        f.write(sample_gt)
    
    print("Created sample ground truth file: examples/ground_truth/sample_map_task3_gt.txt")


def main():
    """Run all CLI examples."""
    print("Historical Map Text Processing CLI Examples")
    print("=" * 50)
    
    # Create sample files first
    create_sample_files()
    
    # Run examples
    try:
        example_system_info()
        example_configuration_management()
        example_single_image_processing()
        example_batch_processing()
        example_evaluation()
        example_advanced_usage()
        
        print("All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())