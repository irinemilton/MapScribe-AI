"""
Main command-line interface for ICDAR 2025 Historical Map Text Processing System.
Supports all four competition tasks with batch processing and configuration management.
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..config.config_manager import ConfigManager
from ..models.data_structures import ProcessingResult
from ..evaluation.competition_formatter import ICDAR2025Formatter
from ..core.system_integration import HistoricalMapSystem
from .config_cli import ConfigCLI
from .batch_processor import BatchProcessor
from .output_manager import OutputManager


class HistoricalMapCLI:
    """
    Command-line interface for historical map text processing system.
    Provides access to all four ICDAR 2025 competition tasks.
    """
    
    def __init__(self):
        """Initialize the CLI with default configuration."""
        self.config_manager = ConfigManager()
        self.formatter = ICDAR2025Formatter()
        self.config_cli = ConfigCLI()
        self.batch_processor = BatchProcessor()
        self.output_manager = OutputManager()
        self.logger = self._setup_logging()
        
        # Initialize integrated system
        self.processing_system = None
        
        # Task definitions
        self.tasks = {
            'task1': 'Word Detection',
            'task2': 'Phrase Detection', 
            'task3': 'Word Detection + Recognition',
            'task4': 'Phrase Detection + Recognition'
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure the argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description='ICDAR 2025 Historical Map Text Processing System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        
        # Main command subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Process command
        process_parser = subparsers.add_parser('process', help='Process historical map images')
        self._add_process_arguments(process_parser)
        
        # Batch command
        batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
        self._add_batch_arguments(batch_parser)
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Manage system configuration')
        self._add_config_arguments(config_parser)
        
        # Evaluate command
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate results against ground truth')
        self._add_evaluate_arguments(eval_parser)
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Display system information')
        self._add_info_arguments(info_parser)
        
        return parser
    
    def _add_process_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the process command."""
        parser.add_argument('input', help='Input image file path')
        parser.add_argument('task', choices=list(self.tasks.keys()), 
                          help='Competition task to execute')
        parser.add_argument('-o', '--output', help='Output file path (default: auto-generated)')
        parser.add_argument('-c', '--config', help='Configuration file path')
        parser.add_argument('--model-path', help='Custom model path')
        parser.add_argument('--confidence', type=float, default=0.5,
                          help='Confidence threshold (default: 0.5)')
        parser.add_argument('--format', choices=['txt', 'json'], default='txt',
                          help='Output format (default: txt)')
        parser.add_argument('--save-intermediate', action='store_true',
                          help='Save intermediate processing results')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose output')
    
    def _add_batch_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the batch command."""
        parser.add_argument('input_dir', help='Input directory containing images')
        parser.add_argument('task', choices=list(self.tasks.keys()),
                          help='Competition task to execute')
        parser.add_argument('-o', '--output-dir', help='Output directory (default: ./results)')
        parser.add_argument('-c', '--config', help='Configuration file path')
        parser.add_argument('--pattern', default='*.{jpg,jpeg,png,tiff,tif}',
                          help='File pattern for image matching (default: *.{jpg,jpeg,png,tiff,tif})')
        parser.add_argument('--parallel', type=int, default=1,
                          help='Number of parallel processes (default: 1)')
        parser.add_argument('--resume', action='store_true',
                          help='Resume interrupted batch processing')
        parser.add_argument('--report', action='store_true',
                          help='Generate processing report')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose output')
    
    def _add_config_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the config command."""
        config_subparsers = parser.add_subparsers(dest='config_action', help='Configuration actions')
        
        # Show config
        config_subparsers.add_parser('show', help='Show current configuration')
        
        # Set config
        set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key (dot notation supported)')
        set_parser.add_argument('value', help='Configuration value')
        
        # Reset config
        config_subparsers.add_parser('reset', help='Reset to default configuration')
        
        # Validate config
        validate_parser = config_subparsers.add_parser('validate', help='Validate configuration file')
        validate_parser.add_argument('config_file', help='Configuration file to validate')
    
    def _add_evaluate_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the evaluate command."""
        parser.add_argument('results_dir', help='Directory containing result files')
        parser.add_argument('ground_truth', help='Ground truth annotations file or directory')
        parser.add_argument('task', choices=list(self.tasks.keys()),
                          help='Competition task to evaluate')
        parser.add_argument('-o', '--output', help='Evaluation report output file')
        parser.add_argument('--metrics', nargs='+', 
                          choices=['precision', 'recall', 'f1', 'iou', 'accuracy'],
                          default=['precision', 'recall', 'f1'],
                          help='Metrics to calculate')
        parser.add_argument('--format', choices=['txt', 'json', 'csv'], default='txt',
                          help='Report format (default: txt)')
    
    def _add_info_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for the info command."""
        parser.add_argument('--models', action='store_true',
                          help='Show available models information')
        parser.add_argument('--config', action='store_true',
                          help='Show configuration information')
        parser.add_argument('--tasks', action='store_true',
                          help='Show supported tasks information')
        parser.add_argument('--system', action='store_true',
                          help='Show system information')
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with provided arguments.
        
        Args:
            args: Command line arguments (default: sys.argv[1:])
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        parser = self.create_parser()
        
        if args is None:
            args = sys.argv[1:]
        
        if not args:
            parser.print_help()
            return 0
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Set up logging level
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            
            # Route to appropriate command handler
            if parsed_args.command == 'process':
                return self._handle_process_command(parsed_args)
            elif parsed_args.command == 'batch':
                return self._handle_batch_command(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config_command(parsed_args)
            elif parsed_args.command == 'evaluate':
                return self._handle_evaluate_command(parsed_args)
            elif parsed_args.command == 'info':
                return self._handle_info_command(parsed_args)
            else:
                parser.print_help()
                return 0
                
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return 1
    
    def _handle_process_command(self, args: argparse.Namespace) -> int:
        """Handle the process command."""
        try:
            # Validate input file
            input_path = Path(args.input)
            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return 1
            
            # Load configuration
            config = self._load_config(args.config)
            
            # Set up output path
            output_path = self._get_output_path(args.output, input_path, args.task, args.format)
            
            self.logger.info(f"Processing {input_path} for {self.tasks[args.task]}")
            self.logger.info(f"Output will be saved to: {output_path}")
            
            # Process the image
            result = self._process_single_image(input_path, args.task, config, args)
            
            # Format and save output
            formatted_output = self.formatter.format_processing_result(result, args.task, input_path.name)
            
            if args.format == 'json':
                formatted_output = self._convert_to_json(result, args.task)
            
            self.output_manager.save_output(formatted_output, output_path, args.format)
            
            # Save intermediate results if requested
            if args.save_intermediate:
                self._save_intermediate_results(result, output_path.parent, input_path.name)
            
            self.logger.info(f"Processing completed successfully")
            self._print_processing_summary(result)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return 1
    
    def _handle_batch_command(self, args: argparse.Namespace) -> int:
        """Handle the batch command."""
        try:
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                self.logger.error(f"Input directory not found: {input_dir}")
                return 1
            
            output_dir = Path(args.output_dir) if args.output_dir else Path('./results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load configuration
            config = self._load_config(args.config)
            
            self.logger.info(f"Starting batch processing: {self.tasks[args.task]}")
            self.logger.info(f"Input directory: {input_dir}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Run batch processing
            results = self.batch_processor.process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                task=args.task,
                config=config,
                pattern=args.pattern,
                parallel=args.parallel,
                resume=args.resume
            )
            
            # Generate report if requested
            if args.report:
                report_path = output_dir / f"batch_report_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.batch_processor.generate_report(results, report_path)
                self.logger.info(f"Batch report saved to: {report_path}")
            
            self.logger.info(f"Batch processing completed: {len(results)} images processed")
            return 0
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return 1
    
    def _handle_config_command(self, args: argparse.Namespace) -> int:
        """Handle the config command."""
        try:
            return self.config_cli.handle_config_command(args)
        except Exception as e:
            self.logger.error(f"Configuration command failed: {e}")
            return 1
    
    def _handle_evaluate_command(self, args: argparse.Namespace) -> int:
        """Handle the evaluate command."""
        try:
            results_dir = Path(args.results_dir)
            if not results_dir.exists():
                self.logger.error(f"Results directory not found: {results_dir}")
                return 1
            
            ground_truth_path = Path(args.ground_truth)
            if not ground_truth_path.exists():
                self.logger.error(f"Ground truth file/directory not found: {ground_truth_path}")
                return 1
            
            self.logger.info(f"Evaluating {self.tasks[args.task]} results")
            
            # Run evaluation
            from ..evaluation.competition_evaluator import CompetitionEvaluator
            evaluator = CompetitionEvaluator()
            
            metrics = evaluator.evaluate_task_results(
                results_dir=results_dir,
                ground_truth_path=ground_truth_path,
                task=args.task,
                metrics=args.metrics
            )
            
            # Generate and save report
            output_path = args.output or f"evaluation_report_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
            self._save_evaluation_report(metrics, output_path, args.format)
            
            self.logger.info(f"Evaluation completed. Report saved to: {output_path}")
            self._print_evaluation_summary(metrics)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return 1
    
    def _handle_info_command(self, args: argparse.Namespace) -> int:
        """Handle the info command."""
        try:
            if args.models:
                self._print_models_info()
            elif args.config:
                self._print_config_info()
            elif args.tasks:
                self._print_tasks_info()
            elif args.system:
                self._print_system_info()
            else:
                # Print all info if no specific flag is provided
                self._print_all_info()
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Info command failed: {e}")
            return 1
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use default."""
        if config_path:
            return self.config_manager.load_config(Path(config_path))
        else:
            return self.config_manager.get_default_config()
    
    def _get_output_path(self, output_arg: Optional[str], input_path: Path, 
                        task: str, format_type: str) -> Path:
        """Generate output path based on arguments."""
        if output_arg:
            return Path(output_arg)
        
        # Auto-generate output path
        stem = input_path.stem
        extension = 'json' if format_type == 'json' else 'txt'
        return Path(f"{stem}_{task}_results.{extension}")
    
    def _process_single_image(self, input_path: Path, task: str, 
                            config: Dict[str, Any], args: argparse.Namespace) -> ProcessingResult:
        """Process a single image and return results."""
        try:
            # Initialize processing system if not already done
            if self.processing_system is None:
                self.processing_system = HistoricalMapSystem()
                self.processing_system.initialize()
            
            # Process the image using integrated system
            result = self.processing_system.process_image(str(input_path), task)
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            # Return error result
            from ..models.data_structures import ProcessingResult
            return ProcessingResult(
                task_type=task,
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=0.0,
                metrics={'error': str(e)}
            )
    
    def _convert_to_json(self, result: ProcessingResult, task: str) -> str:
        """Convert processing result to JSON format."""
        result_dict = {
            'task': task,
            'task_name': self.tasks[task],
            'processing_time': result.processing_time,
            'detections': [],
            'recognized_texts': [],
            'metrics': result.metrics
        }
        
        # Convert detections
        for detection in result.detections:
            det_dict = {
                'bbox': {
                    'x': detection.bbox.x,
                    'y': detection.bbox.y,
                    'width': detection.bbox.width,
                    'height': detection.bbox.height
                },
                'confidence': detection.confidence,
                'type': detection.detection_type
            }
            result_dict['detections'].append(det_dict)
        
        # Convert recognized texts
        for recognized in result.recognized_texts:
            rec_dict = {
                'text': recognized.text,
                'language': recognized.language,
                'confidence': recognized.recognition_confidence,
                'bbox': {
                    'x': recognized.detection.bbox.x,
                    'y': recognized.detection.bbox.y,
                    'width': recognized.detection.bbox.width,
                    'height': recognized.detection.bbox.height
                }
            }
            result_dict['recognized_texts'].append(rec_dict)
        
        return json.dumps(result_dict, indent=2)
    
    def _save_intermediate_results(self, result: ProcessingResult, output_dir: Path, image_name: str) -> None:
        """Save intermediate processing results."""
        intermediate_dir = output_dir / 'intermediate' / Path(image_name).stem
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detections
        if result.detections:
            det_path = intermediate_dir / 'detections.json'
            with open(det_path, 'w') as f:
                json.dump([{
                    'bbox': [d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height],
                    'confidence': d.confidence,
                    'type': d.detection_type
                } for d in result.detections], f, indent=2)
        
        # Save recognition results
        if result.recognized_texts:
            rec_path = intermediate_dir / 'recognition.json'
            with open(rec_path, 'w') as f:
                json.dump([{
                    'text': r.text,
                    'language': r.language,
                    'confidence': r.recognition_confidence
                } for r in result.recognized_texts], f, indent=2)
    
    def _save_evaluation_report(self, metrics: Dict[str, float], output_path: str, format_type: str) -> None:
        """Save evaluation report in specified format."""
        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif format_type == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in metrics.items():
                    writer.writerow([key, value])
        else:  # txt format
            with open(output_path, 'w') as f:
                f.write("Evaluation Report\n")
                f.write("================\n\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
    
    def _print_processing_summary(self, result: ProcessingResult) -> None:
        """Print processing summary to console."""
        print(f"\nProcessing Summary:")
        print(f"  Task: {self.tasks[result.task_type]}")
        print(f"  Processing Time: {result.processing_time:.2f}s")
        print(f"  Detections Found: {len(result.detections)}")
        if result.recognized_texts:
            print(f"  Texts Recognized: {len(result.recognized_texts)}")
        if result.metrics:
            print(f"  Average Confidence: {result.metrics.get('confidence_avg', 0):.3f}")
    
    def _print_evaluation_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary to console."""
        print(f"\nEvaluation Summary:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    def _print_models_info(self) -> None:
        """Print available models information."""
        print("Available Models:")
        print("  Detection Models:")
        print("    - YOLO-based word detector")
        print("    - CRAFT-based phrase detector")
        print("  Recognition Models:")
        print("    - CRNN for Latin characters")
        print("    - Multilingual model for Chinese characters")
    
    def _print_config_info(self) -> None:
        """Print configuration information."""
        config = self.config_manager.get_default_config()
        print("Configuration Information:")
        print(f"  Config file locations:")
        print(f"    - Default: config/default_config.yaml")
        print(f"    - User: ~/.historical_map_processor/config.yaml")
        print(f"  Current settings:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    
    def _print_tasks_info(self) -> None:
        """Print supported tasks information."""
        print("Supported Competition Tasks:")
        for task_id, task_name in self.tasks.items():
            print(f"  {task_id}: {task_name}")
            task_info = self.formatter.get_task_info(task_id)
            print(f"    Output format: {task_info['output_format']}")
            print(f"    File extension: {task_info['file_extension']}")
    
    def _print_system_info(self) -> None:
        """Print system information."""
        import platform
        import torch
        
        print("System Information:")
        print(f"  Platform: {platform.platform()}")
        print(f"  Python: {platform.python_version()}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Devices: {torch.cuda.device_count()}")
    
    def _print_all_info(self) -> None:
        """Print all system information."""
        self._print_system_info()
        print()
        self._print_tasks_info()
        print()
        self._print_models_info()
        print()
        self._print_config_info()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _get_usage_examples(self) -> str:
        """Get usage examples for help text."""
        return """
Examples:
  # Process single image for word detection
  python -m src.cli process image.jpg task1 -o results.txt
  
  # Batch process directory for phrase recognition
  python -m src.cli batch /path/to/images task4 -o /path/to/results --parallel 4
  
  # Evaluate results against ground truth
  python -m src.cli evaluate /path/to/results /path/to/ground_truth task1
  
  # Show system information
  python -m src.cli info --system
  
  # Configure system settings
  python -m src.cli config set detection.confidence_threshold 0.7
"""


def main() -> int:
    """Main entry point for the CLI."""
    cli = HistoricalMapCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())