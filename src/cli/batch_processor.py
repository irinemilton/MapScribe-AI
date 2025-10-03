"""
Batch processing functionality for historical map text processing CLI.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import glob

from ..models.data_structures import ProcessingResult
from ..evaluation.competition_formatter import ICDAR2025Formatter


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    image_path: str
    task: str
    success: bool
    processing_time: float
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    detection_count: int = 0
    recognition_count: int = 0


class BatchProcessor:
    """Handles batch processing of multiple historical map images."""
    
    def __init__(self):
        """Initialize the batch processor."""
        self.formatter = ICDAR2025Formatter()
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, input_dir: Path, output_dir: Path, task: str,
                     config: Dict[str, Any], pattern: str = "*.{jpg,jpeg,png,tiff,tif}",
                     parallel: int = 1, resume: bool = False) -> List[BatchResult]:
        """
        Process multiple images in batch mode.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output files
            task: Competition task to execute
            config: Processing configuration
            pattern: File pattern for image matching
            parallel: Number of parallel processes
            resume: Whether to resume interrupted processing
            
        Returns:
            List of batch processing results
        """
        # Find input images
        image_paths = self._find_images(input_dir, pattern)
        
        if not image_paths:
            self.logger.warning(f"No images found in {input_dir} matching pattern {pattern}")
            return []
        
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing results if resuming
        existing_results = {}
        if resume:
            existing_results = self._load_existing_results(output_dir, task)
            self.logger.info(f"Resuming: found {len(existing_results)} existing results")
        
        # Filter images that need processing
        images_to_process = []
        for img_path in image_paths:
            output_path = self._get_batch_output_path(output_dir, img_path, task)
            if not resume or not output_path.exists():
                images_to_process.append(img_path)
        
        self.logger.info(f"Processing {len(images_to_process)} images ({len(image_paths) - len(images_to_process)} already completed)")
        
        # Process images
        results = []
        
        if parallel > 1:
            results = self._process_parallel(images_to_process, output_dir, task, config, parallel)
        else:
            results = self._process_sequential(images_to_process, output_dir, task, config)
        
        # Combine with existing results
        all_results = list(existing_results.values()) + results
        
        # Save batch summary
        self._save_batch_summary(all_results, output_dir, task)
        
        return all_results
    
    def _find_images(self, input_dir: Path, pattern: str) -> List[Path]:
        """
        Find image files matching the specified pattern.
        
        Args:
            input_dir: Directory to search
            pattern: File pattern (supports glob patterns)
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        # Handle multiple extensions in pattern
        if '{' in pattern and '}' in pattern:
            # Extract extensions from pattern like "*.{jpg,jpeg,png}"
            base_pattern = pattern.split('{')[0]
            extensions_part = pattern.split('{')[1].split('}')[0]
            extensions = [ext.strip() for ext in extensions_part.split(',')]
            
            for ext in extensions:
                search_pattern = base_pattern + ext
                image_paths.extend(input_dir.glob(search_pattern))
        else:
            # Simple glob pattern
            image_paths.extend(input_dir.glob(pattern))
        
        # Sort for consistent processing order
        return sorted(image_paths)
    
    def _load_existing_results(self, output_dir: Path, task: str) -> Dict[str, BatchResult]:
        """
        Load existing batch results for resuming.
        
        Args:
            output_dir: Output directory
            task: Competition task
            
        Returns:
            Dictionary of existing results keyed by image path
        """
        results = {}
        
        # Look for existing output files
        for output_file in output_dir.glob(f"*_{task}_results.txt"):
            # Extract image name from output filename
            image_name = output_file.name.replace(f"_{task}_results.txt", "")
            
            # Create a result entry for existing file
            result = BatchResult(
                image_path=image_name,
                task=task,
                success=True,
                processing_time=0.0,
                output_path=str(output_file),
                detection_count=self._count_detections_in_file(output_file),
                recognition_count=self._count_recognitions_in_file(output_file, task)
            )
            
            results[image_name] = result
        
        return results
    
    def _process_sequential(self, image_paths: List[Path], output_dir: Path,
                          task: str, config: Dict[str, Any]) -> List[BatchResult]:
        """
        Process images sequentially.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Output directory
            task: Competition task
            config: Processing configuration
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing {i}/{len(image_paths)}: {image_path.name}")
            
            result = self._process_single_image_batch(image_path, output_dir, task, config)
            results.append(result)
            
            # Log progress
            if result.success:
                self.logger.info(f"  ✓ Completed in {result.processing_time:.2f}s")
            else:
                self.logger.error(f"  ✗ Failed: {result.error_message}")
        
        return results
    
    def _process_parallel(self, image_paths: List[Path], output_dir: Path,
                         task: str, config: Dict[str, Any], parallel: int) -> List[BatchResult]:
        """
        Process images in parallel.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Output directory
            task: Competition task
            config: Processing configuration
            parallel: Number of parallel processes
            
        Returns:
            List of processing results
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_image_batch, img_path, output_dir, task, config): img_path
                for img_path in image_paths
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"  ✓ ({completed}/{len(image_paths)}) {image_path.name} - {result.processing_time:.2f}s")
                    else:
                        self.logger.error(f"  ✗ ({completed}/{len(image_paths)}) {image_path.name} - {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"  ✗ ({completed}/{len(image_paths)}) {image_path.name} - Unexpected error: {e}")
                    
                    result = BatchResult(
                        image_path=str(image_path),
                        task=task,
                        success=False,
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    results.append(result)
        
        return results
    
    def _process_single_image_batch(self, image_path: Path, output_dir: Path,
                                  task: str, config: Dict[str, Any]) -> BatchResult:
        """
        Process a single image in batch mode.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            task: Competition task
            config: Processing configuration
            
        Returns:
            Batch processing result
        """
        start_time = time.time()
        
        try:
            # Generate output path
            output_path = self._get_batch_output_path(output_dir, image_path, task)
            
            # Process the image (mock implementation)
            processing_result = self._mock_process_image(image_path, task, config)
            
            # Format and save output
            formatted_output = self.formatter.format_processing_result(processing_result, task, image_path.name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            
            processing_time = time.time() - start_time
            
            return BatchResult(
                image_path=str(image_path),
                task=task,
                success=True,
                processing_time=processing_time,
                output_path=str(output_path),
                detection_count=len(processing_result.detections),
                recognition_count=len(processing_result.recognized_texts)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return BatchResult(
                image_path=str(image_path),
                task=task,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _get_batch_output_path(self, output_dir: Path, image_path: Path, task: str) -> Path:
        """
        Generate output path for batch processing.
        
        Args:
            output_dir: Output directory
            image_path: Input image path
            task: Competition task
            
        Returns:
            Output file path
        """
        stem = image_path.stem
        return output_dir / f"{stem}_{task}_results.txt"
    
    def _mock_process_image(self, image_path: Path, task: str, config: Dict[str, Any]) -> ProcessingResult:
        """
        Mock image processing for demonstration.
        Replace with actual processing pipeline integration.
        
        Args:
            image_path: Path to input image
            task: Competition task
            config: Processing configuration
            
        Returns:
            Mock processing result
        """
        from ..models.data_structures import ProcessingResult, Detection, RecognizedText, BoundingBox
        import random
        
        # Simulate processing delay
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate mock detections
        detections = []
        num_detections = random.randint(1, 5)
        
        for i in range(num_detections):
            detection = Detection(
                bbox=BoundingBox(
                    x=random.randint(50, 500),
                    y=random.randint(50, 300),
                    width=random.randint(50, 200),
                    height=random.randint(20, 60),
                    confidence=random.uniform(0.5, 0.95)
                ),
                detection_type='word' if task in ['task1', 'task3'] else 'phrase',
                confidence=random.uniform(0.5, 0.95),
                image_coordinates=(0, 0, 0, 0)
            )
            detections.append(detection)
        
        # Generate mock recognition results
        recognized_texts = []
        if task in ['task3', 'task4']:
            for detection in detections:
                recognized_text = RecognizedText(
                    detection=detection,
                    text=f"Sample Text {random.randint(1, 100)}",
                    language="en",
                    recognition_confidence=random.uniform(0.6, 0.9),
                    character_confidences=[random.uniform(0.5, 0.9) for _ in range(5)]
                )
                recognized_texts.append(recognized_text)
        
        return ProcessingResult(
            task_type=task,
            detections=detections,
            recognized_texts=recognized_texts,
            linked_texts=[],
            processing_time=random.uniform(0.5, 2.0),
            metrics={'confidence_avg': random.uniform(0.7, 0.9)}
        )
    
    def _count_detections_in_file(self, file_path: Path) -> int:
        """Count detections in output file."""
        try:
            with open(file_path, 'r') as f:
                lines = f.read().strip().split('\n')
                return len([line for line in lines if line.strip()])
        except Exception:
            return 0
    
    def _count_recognitions_in_file(self, file_path: Path, task: str) -> int:
        """Count recognitions in output file."""
        if task in ['task3', 'task4']:
            return self._count_detections_in_file(file_path)
        return 0
    
    def _save_batch_summary(self, results: List[BatchResult], output_dir: Path, task: str) -> None:
        """
        Save batch processing summary.
        
        Args:
            results: List of batch results
            output_dir: Output directory
            task: Competition task
        """
        summary_path = output_dir / f"batch_summary_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'task': task,
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'successful': len([r for r in results if r.success]),
            'failed': len([r for r in results if not r.success]),
            'total_processing_time': sum(r.processing_time for r in results),
            'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0,
            'total_detections': sum(r.detection_count for r in results if r.success),
            'total_recognitions': sum(r.recognition_count for r in results if r.success),
            'results': [asdict(result) for result in results]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Batch summary saved to: {summary_path}")
    
    def generate_report(self, results: List[BatchResult], output_path: Path) -> None:
        """
        Generate detailed batch processing report.
        
        Args:
            results: List of batch results
            output_path: Path to save the report
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        report = {
            'summary': {
                'total_images': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'total_processing_time': sum(r.processing_time for r in results),
                'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0
            },
            'statistics': {
                'total_detections': sum(r.detection_count for r in successful_results),
                'total_recognitions': sum(r.recognition_count for r in successful_results),
                'average_detections_per_image': sum(r.detection_count for r in successful_results) / len(successful_results) if successful_results else 0,
                'average_recognitions_per_image': sum(r.recognition_count for r in successful_results) / len(successful_results) if successful_results else 0
            },
            'failed_images': [
                {
                    'image_path': r.image_path,
                    'error_message': r.error_message
                }
                for r in failed_results
            ],
            'detailed_results': [asdict(result) for result in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Detailed report saved to: {output_path}")