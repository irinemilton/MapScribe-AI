"""
Integration tests for the complete historical map text processing pipeline.
Implements comprehensive end-to-end testing with performance benchmarking,
regression testing, and competition format compliance validation.

Requirements addressed:
- 6.4: Processing time and throughput metrics
- 6.5: Detailed comparison reports  
- 7.1: Enhancement techniques for text visibility
- 7.2: Normalization for consistent processing
"""

import unittest
import tempfile
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

# Import system components
from src.models.data_structures import (
    ProcessingResult, Detection, RecognizedText, LinkedText,
    BoundingBox, DetectionConfig, RecognitionConfig
)
from src.evaluation.competition_evaluator import ICDAR2025Evaluator
from src.evaluation.competition_formatter import ICDAR2025Formatter
from src.evaluation.metrics_calculator import CompetitionMetricsCalculator

# Import test utilities
from tests.test_utilities import (
    MockImageGenerator, MockModelObjects, TestDataFactory,
    PerformanceTestUtilities
)


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests for complete processing pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_images_dir = Path(self.temp_dir) / "test_images"
        self.test_images_dir.mkdir(exist_ok=True)
        
        # Create sample historical map images
        self.sample_images = self._create_sample_historical_maps()
        
        # Initialize evaluator and formatter
        self.evaluator = ICDAR2025Evaluator(iou_threshold=0.5)
        self.formatter = ICDAR2025Formatter()
        
        # Performance tracking
        self.performance_tracker = PerformanceTestUtilities()
        
        # Mock system components
        self.mock_detector = MockModelObjects.create_mock_detector()
        self.mock_recognizer = MockModelObjects.create_mock_recognizer()
        self.mock_linker = MockModelObjects.create_mock_linker()
        self.mock_processor = MockModelObjects.create_mock_image_processor()
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_historical_maps(self) -> List[Tuple[str, np.ndarray]]:
        """Create sample historical map images for testing."""
        sample_images = []
        
        # Create different types of historical maps
        map_configs = [
            {"width": 1024, "height": 768, "name": "standard_map.jpg"},
            {"width": 2048, "height": 1536, "name": "high_res_map.jpg"},
            {"width": 800, "height": 600, "name": "small_map.jpg"},
            {"width": 1200, "height": 900, "name": "degraded_map.jpg"}
        ]
        
        for config in map_configs:
            # Create base historical map image
            image = MockImageGenerator.create_historical_map_image(
                config["width"], config["height"]
            )
            
            # Apply degradation for some images
            if "degraded" in config["name"]:
                image = MockImageGenerator.create_degraded_image(image, degradation_level=0.4)
            
            # Save image to temp directory
            image_path = self.test_images_dir / config["name"]
            # Simulate saving (in real implementation would use cv2.imwrite)
            sample_images.append((str(image_path), image))
        
        return sample_images
    
    def test_complete_task1_pipeline(self):
        """Test complete Task 1 (Word Detection) pipeline end-to-end."""
        task_type = "word_detection"
        
        for image_path, image in self.sample_images:
            with self.subTest(image_path=image_path):
                # Start performance measurement
                start_time = time.time()
                
                # Step 1: Image preprocessing
                processed_image = self.mock_processor.preprocess(image)
                self.assertIsNotNone(processed_image)
                
                # Step 2: Text detection
                detections = self.mock_detector.detect_words(processed_image)
                self.assertIsInstance(detections, list)
                self.assertGreater(len(detections), 0)
                
                # Step 3: Create processing result
                processing_result = ProcessingResult(
                    task_type=task_type,
                    detections=detections,
                    recognized_texts=[],
                    linked_texts=[],
                    processing_time=time.time() - start_time,
                    metrics={}
                )
                
                # Step 4: Format output
                formatted_output = self.formatter.format_task1_output(detections)
                self.assertIsInstance(formatted_output, str)
                self.assertGreater(len(formatted_output), 0)
                
                # Step 5: Validate format compliance
                format_valid = self.formatter.validate_competition_format("task1", formatted_output)
                self.assertTrue(format_valid, f"Invalid format for task1")
                
                # Step 6: Performance validation
                processing_time = processing_result.processing_time
                self.assertLess(processing_time, 30.0, "Processing time too slow")
                
                # Verify detection quality
                for detection in detections:
                    self.assertIsInstance(detection, Detection)
                    self.assertGreaterEqual(detection.confidence, 0.0)
                    self.assertLessEqual(detection.confidence, 1.0)
                    self.assertGreater(detection.bbox.width, 0)
                    self.assertGreater(detection.bbox.height, 0)
    
    def test_complete_task2_pipeline(self):
        """Test complete Task 2 (Phrase Detection) pipeline end-to-end."""
        task_type = "phrase_detection"
        
        for image_path, image in self.sample_images:
            with self.subTest(image_path=image_path):
                # Start performance measurement
                start_time = time.time()
                
                # Step 1: Image preprocessing
                processed_image = self.mock_processor.preprocess(image)
                
                # Step 2: Phrase detection
                detections = self.mock_detector.detect_phrases(processed_image)
                self.assertIsInstance(detections, list)
                self.assertGreater(len(detections), 0)
                
                # Step 3: Create processing result
                processing_result = ProcessingResult(
                    task_type=task_type,
                    detections=detections,
                    recognized_texts=[],
                    linked_texts=[],
                    processing_time=time.time() - start_time,
                    metrics={}
                )
                
                # Step 4: Format and validate output
                formatted_output = self.formatter.format_task2_output(detections)
                format_valid = self.formatter.validate_competition_format("task2", formatted_output)
                self.assertTrue(format_valid)
                
                # Verify phrase detection characteristics
                for detection in detections:
                    self.assertEqual(detection.detection_type, "phrase")
                    # Phrases should generally be larger than individual words
                    self.assertGreater(detection.bbox.width, 20)
    
    def test_complete_task3_pipeline(self):
        """Test complete Task 3 (Word Detection + Recognition) pipeline end-to-end."""
        task_type = "word_detection_recognition"
        
        for image_path, image in self.sample_images:
            with self.subTest(image_path=image_path):
                # Start performance measurement
                start_time = time.time()
                
                # Step 1: Image preprocessing
                processed_image = self.mock_processor.preprocess(image)
                
                # Step 2: Word detection
                detections = self.mock_detector.detect_words(processed_image)
                
                # Step 3: Text recognition
                recognized_texts = self.mock_recognizer.recognize_words(detections)
                self.assertIsInstance(recognized_texts, list)
                self.assertEqual(len(recognized_texts), len(detections))
                
                # Step 4: Create processing result
                processing_result = ProcessingResult(
                    task_type=task_type,
                    detections=detections,
                    recognized_texts=recognized_texts,
                    linked_texts=[],
                    processing_time=time.time() - start_time,
                    metrics={}
                )
                
                # Step 5: Format and validate output
                formatted_output = self.formatter.format_task3_output(recognized_texts)
                format_valid = self.formatter.validate_competition_format("task3", formatted_output)
                self.assertTrue(format_valid)
                
                # Verify recognition quality
                for recognized_text in recognized_texts:
                    self.assertIsInstance(recognized_text, RecognizedText)
                    self.assertIsInstance(recognized_text.text, str)
                    self.assertGreater(len(recognized_text.text), 0)
                    self.assertGreaterEqual(recognized_text.recognition_confidence, 0.0)
                    self.assertLessEqual(recognized_text.recognition_confidence, 1.0)
    
    def test_complete_task4_pipeline(self):
        """Test complete Task 4 (Phrase Detection + Recognition) pipeline end-to-end."""
        task_type = "phrase_detection_recognition"
        
        for image_path, image in self.sample_images:
            with self.subTest(image_path=image_path):
                # Start performance measurement
                start_time = time.time()
                
                # Step 1: Image preprocessing
                processed_image = self.mock_processor.preprocess(image)
                
                # Step 2: Phrase detection
                detections = self.mock_detector.detect_phrases(processed_image)
                
                # Step 3: Phrase recognition
                recognized_texts = self.mock_recognizer.recognize_phrases(detections)
                
                # Step 4: Text linking (optional for task 4)
                linked_texts = self.mock_linker.link_to_coordinates(recognized_texts, image)
                
                # Step 5: Create processing result
                processing_result = ProcessingResult(
                    task_type=task_type,
                    detections=detections,
                    recognized_texts=recognized_texts,
                    linked_texts=linked_texts,
                    processing_time=time.time() - start_time,
                    metrics={}
                )
                
                # Step 6: Format and validate output
                formatted_output = self.formatter.format_task4_output(recognized_texts)
                format_valid = self.formatter.validate_competition_format("task4", formatted_output)
                self.assertTrue(format_valid)
                
                # Verify complete pipeline results
                self.assertEqual(len(recognized_texts), len(detections))
                self.assertGreaterEqual(len(linked_texts), 0)
                
                # Verify phrase characteristics
                for recognized_text in recognized_texts:
                    self.assertEqual(recognized_text.detection.detection_type, "phrase")
                    # Phrases should contain multiple words or longer text
                    self.assertGreaterEqual(len(recognized_text.text.split()), 1)
    
    def test_multilingual_pipeline(self):
        """Test pipeline with multilingual text (English and Chinese)."""
        # Create multilingual test data
        multilingual_data = TestDataFactory.create_multilingual_test_data()
        
        for image_path, image in self.sample_images[:2]:  # Test with first 2 images
            with self.subTest(image_path=image_path):
                # Process with multilingual recognition
                processed_image = self.mock_processor.preprocess(image)
                detections = self.mock_detector.detect_words(processed_image)
                
                # Simulate multilingual recognition
                recognized_texts = []
                for i, detection in enumerate(detections):
                    # Alternate between English and Chinese
                    if i % 2 == 0:
                        text_data = multilingual_data['english'][i % len(multilingual_data['english'])]
                    else:
                        text_data = multilingual_data['chinese'][i % len(multilingual_data['chinese'])]
                    
                    recognized_text = RecognizedText(
                        detection=detection,
                        text=text_data['text'],
                        language=text_data['language'],
                        recognition_confidence=text_data['confidence'],
                        character_confidences=text_data['character_confidences']
                    )
                    recognized_texts.append(recognized_text)
                
                # Verify multilingual handling
                languages_found = set(rt.language for rt in recognized_texts)
                self.assertIn('en', languages_found)
                self.assertIn('zh', languages_found)
                
                # Test formatting with multilingual content
                formatted_output = self.formatter.format_task3_output(recognized_texts)
                self.assertIsInstance(formatted_output, str)
                self.assertGreater(len(formatted_output), 0)


class TestPerformanceBenchmarking(unittest.TestCase):
    """Performance benchmarking and timing measurement tests."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.performance_tracker = PerformanceTestUtilities()
        self.mock_detector = MockModelObjects.create_mock_detector()
        self.mock_recognizer = MockModelObjects.create_mock_recognizer()
        self.mock_processor = MockModelObjects.create_mock_image_processor()
        
        # Create test images of different sizes
        self.test_images = [
            MockImageGenerator.create_historical_map_image(640, 480),   # Small
            MockImageGenerator.create_historical_map_image(1024, 768),  # Medium
            MockImageGenerator.create_historical_map_image(2048, 1536), # Large
            MockImageGenerator.create_historical_map_image(4096, 3072)  # Very Large
        ]
        
        self.image_sizes = ["Small", "Medium", "Large", "Very Large"]
    
    def test_processing_time_benchmarks(self):
        """Test processing time benchmarks for different image sizes."""
        benchmark_results = {}
        
        for i, (image, size_name) in enumerate(zip(self.test_images, self.image_sizes)):
            with self.subTest(size=size_name):
                # Measure preprocessing time
                processed_image, preprocess_time = self.performance_tracker.measure_execution_time(
                    self.mock_processor.preprocess, image
                )
                
                # Measure detection time
                detections, detection_time = self.performance_tracker.measure_execution_time(
                    self.mock_detector.detect_words, processed_image
                )
                
                # Measure recognition time
                recognized_texts, recognition_time = self.performance_tracker.measure_execution_time(
                    self.mock_recognizer.recognize_words, detections
                )
                
                total_time = preprocess_time + detection_time + recognition_time
                
                # Store benchmark results
                benchmark_results[size_name] = {
                    'image_size': image.shape,
                    'preprocess_time': preprocess_time,
                    'detection_time': detection_time,
                    'recognition_time': recognition_time,
                    'total_time': total_time,
                    'pixels': image.shape[0] * image.shape[1]
                }
                
                # Performance assertions
                self.assertLess(preprocess_time, 5.0, f"Preprocessing too slow for {size_name}")
                self.assertLess(detection_time, 10.0, f"Detection too slow for {size_name}")
                self.assertLess(recognition_time, 15.0, f"Recognition too slow for {size_name}")
                self.assertLess(total_time, 30.0, f"Total processing too slow for {size_name}")
        
        # Verify performance scaling
        self._verify_performance_scaling(benchmark_results)
    
    def test_throughput_measurements(self):
        """Test throughput measurements for batch processing."""
        batch_sizes = [1, 5, 10, 20]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Create batch of images
                batch_images = [self.test_images[0]] * batch_size  # Use small images for speed
                
                # Measure batch processing time
                start_time = time.time()
                
                batch_results = []
                for image in batch_images:
                    processed_image = self.mock_processor.preprocess(image)
                    detections = self.mock_detector.detect_words(processed_image)
                    recognized_texts = self.mock_recognizer.recognize_words(detections)
                    batch_results.append(recognized_texts)
                
                total_time = time.time() - start_time
                throughput = batch_size / total_time  # images per second
                
                throughput_results[batch_size] = {
                    'batch_size': batch_size,
                    'total_time': total_time,
                    'throughput': throughput,
                    'avg_time_per_image': total_time / batch_size
                }
                
                # Throughput assertions
                self.assertGreater(throughput, 0.1, f"Throughput too low for batch size {batch_size}")
                self.assertEqual(len(batch_results), batch_size)
        
        # Verify throughput scaling
        self._verify_throughput_scaling(throughput_results)
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large image and monitor memory
        large_image = self.test_images[-1]  # Very large image
        
        # Measure memory during processing
        processed_image = self.mock_processor.preprocess(large_image)
        memory_after_preprocess = process.memory_info().rss / 1024 / 1024
        
        detections = self.mock_detector.detect_words(processed_image)
        memory_after_detection = process.memory_info().rss / 1024 / 1024
        
        recognized_texts = self.mock_recognizer.recognize_words(detections)
        memory_after_recognition = process.memory_info().rss / 1024 / 1024
        
        # Memory usage assertions
        memory_increase = memory_after_recognition - initial_memory
        self.assertLess(memory_increase, 500, "Memory usage too high (>500MB)")
        
        # Verify memory doesn't grow excessively between steps
        preprocess_increase = memory_after_preprocess - initial_memory
        detection_increase = memory_after_detection - memory_after_preprocess
        recognition_increase = memory_after_recognition - memory_after_detection
        
        self.assertLess(preprocess_increase, 200, "Preprocessing memory usage too high")
        self.assertLess(detection_increase, 200, "Detection memory usage too high")
        self.assertLess(recognition_increase, 200, "Recognition memory usage too high")
    
    def _verify_performance_scaling(self, benchmark_results: Dict[str, Dict[str, Any]]):
        """Verify that performance scales reasonably with image size."""
        sizes = list(benchmark_results.keys())
        
        # Check that larger images don't take disproportionately longer
        for i in range(len(sizes) - 1):
            current_size = sizes[i]
            next_size = sizes[i + 1]
            
            current_pixels = benchmark_results[current_size]['pixels']
            next_pixels = benchmark_results[next_size]['pixels']
            
            current_time = benchmark_results[current_size]['total_time']
            next_time = benchmark_results[next_size]['total_time']
            
            # Time should scale roughly linearly with pixels (allowing for some overhead)
            pixel_ratio = next_pixels / current_pixels
            time_ratio = next_time / current_time
            
            # Allow up to 2x overhead for larger images
            self.assertLess(time_ratio, pixel_ratio * 2, 
                          f"Performance scaling poor between {current_size} and {next_size}")
    
    def _verify_throughput_scaling(self, throughput_results: Dict[int, Dict[str, Any]]):
        """Verify that throughput scales reasonably with batch size."""
        batch_sizes = sorted(throughput_results.keys())
        
        # Throughput should generally increase or stay stable with larger batches
        for i in range(len(batch_sizes) - 1):
            current_batch = batch_sizes[i]
            next_batch = batch_sizes[i + 1]
            
            current_throughput = throughput_results[current_batch]['throughput']
            next_throughput = throughput_results[next_batch]['throughput']
            
            # Allow for some variation, but throughput shouldn't drop significantly
            throughput_ratio = next_throughput / current_throughput
            self.assertGreater(throughput_ratio, 0.5, 
                             f"Throughput drops too much from batch {current_batch} to {next_batch}")


class TestRegressionTesting(unittest.TestCase):
    """Regression testing for model accuracy maintenance."""
    
    def setUp(self):
        """Set up regression test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_results_path = Path(self.temp_dir) / "baseline_results.json"
        
        # Create baseline results
        self.baseline_results = self._create_baseline_results()
        self._save_baseline_results()
        
        # Initialize components
        self.evaluator = ICDAR2025Evaluator()
        self.mock_detector = MockModelObjects.create_mock_detector()
        self.mock_recognizer = MockModelObjects.create_mock_recognizer()
    
    def tearDown(self):
        """Clean up regression test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_baseline_results(self) -> Dict[str, Any]:
        """Create baseline results for regression testing."""
        return {
            'task1': {
                'detection_metrics': {
                    'precision': 0.85,
                    'recall': 0.82,
                    'f1_score': 0.835,
                    'average_iou': 0.75
                }
            },
            'task2': {
                'detection_metrics': {
                    'precision': 0.78,
                    'recall': 0.76,
                    'f1_score': 0.77,
                    'average_iou': 0.70
                }
            },
            'task3': {
                'detection_metrics': {
                    'precision': 0.85,
                    'recall': 0.82,
                    'f1_score': 0.835,
                    'average_iou': 0.75
                },
                'recognition_metrics': {
                    'character_accuracy': 0.92,
                    'word_accuracy': 0.88,
                    'exact_match_accuracy': 0.85
                }
            },
            'task4': {
                'detection_metrics': {
                    'precision': 0.78,
                    'recall': 0.76,
                    'f1_score': 0.77,
                    'average_iou': 0.70
                },
                'recognition_metrics': {
                    'character_accuracy': 0.89,
                    'word_accuracy': 0.84,
                    'exact_match_accuracy': 0.80
                }
            }
        }
    
    def _save_baseline_results(self):
        """Save baseline results to file."""
        with open(self.baseline_results_path, 'w') as f:
            json.dump(self.baseline_results, f, indent=2)
    
    def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results from file."""
        with open(self.baseline_results_path, 'r') as f:
            return json.load(f)
    
    def test_task1_regression(self):
        """Test Task 1 regression against baseline results."""
        task_id = 'task1'
        baseline = self.baseline_results[task_id]['detection_metrics']
        
        # Generate current results
        test_image = MockImageGenerator.create_historical_map_image(1024, 768)
        detections = self.mock_detector.detect_words(test_image)
        
        # Create ground truth for evaluation
        ground_truth_detections = TestDataFactory.create_sample_detections(count=len(detections))
        
        # Calculate current metrics
        current_metrics = self.evaluator.evaluate_detection(detections, ground_truth_detections)
        
        # Regression assertions (allow 5% degradation)
        tolerance = 0.05
        self.assertGreaterEqual(current_metrics['precision'], 
                               baseline['precision'] - tolerance,
                               f"Precision regression in {task_id}")
        self.assertGreaterEqual(current_metrics['recall'], 
                               baseline['recall'] - tolerance,
                               f"Recall regression in {task_id}")
        self.assertGreaterEqual(current_metrics['f1_score'], 
                               baseline['f1_score'] - tolerance,
                               f"F1-score regression in {task_id}")
        self.assertGreaterEqual(current_metrics['average_iou'], 
                               baseline['average_iou'] - tolerance,
                               f"IoU regression in {task_id}")
    
    def test_task3_regression(self):
        """Test Task 3 regression against baseline results."""
        task_id = 'task3'
        baseline = self.baseline_results[task_id]
        
        # Generate current results
        test_image = MockImageGenerator.create_historical_map_image(1024, 768)
        detections = self.mock_detector.detect_words(test_image)
        recognized_texts = self.mock_recognizer.recognize_words(detections)
        
        # Create ground truth
        ground_truth_detections = TestDataFactory.create_sample_detections(count=len(detections))
        ground_truth_texts = [f"word_{i}" for i in range(len(recognized_texts))]
        
        # Calculate current metrics
        detection_metrics = self.evaluator.evaluate_detection(detections, ground_truth_detections)
        recognition_metrics = self.evaluator.evaluate_recognition(recognized_texts, ground_truth_texts)
        
        # Regression assertions
        tolerance = 0.05
        
        # Detection metrics
        baseline_det = baseline['detection_metrics']
        self.assertGreaterEqual(detection_metrics['precision'], 
                               baseline_det['precision'] - tolerance)
        self.assertGreaterEqual(detection_metrics['recall'], 
                               baseline_det['recall'] - tolerance)
        
        # Recognition metrics
        baseline_rec = baseline['recognition_metrics']
        self.assertGreaterEqual(recognition_metrics['character_accuracy'], 
                               baseline_rec['character_accuracy'] - tolerance)
        self.assertGreaterEqual(recognition_metrics['word_accuracy'], 
                               baseline_rec['word_accuracy'] - tolerance)
    
    def test_accuracy_trend_monitoring(self):
        """Test accuracy trend monitoring over multiple runs."""
        num_runs = 5
        task_id = 'task1'
        
        accuracy_history = []
        
        for run in range(num_runs):
            # Generate test data
            test_image = MockImageGenerator.create_historical_map_image(1024, 768)
            detections = self.mock_detector.detect_words(test_image)
            ground_truth = TestDataFactory.create_sample_detections(count=len(detections))
            
            # Calculate metrics
            metrics = self.evaluator.evaluate_detection(detections, ground_truth)
            accuracy_history.append(metrics['f1_score'])
        
        # Verify accuracy stability
        mean_accuracy = np.mean(accuracy_history)
        std_accuracy = np.std(accuracy_history)
        
        # Accuracy should be stable (low standard deviation)
        self.assertLess(std_accuracy, 0.1, "Accuracy too unstable across runs")
        
        # Mean accuracy should be reasonable
        self.assertGreater(mean_accuracy, 0.7, "Mean accuracy too low")
        
        # No single run should be drastically different
        for accuracy in accuracy_history:
            self.assertGreater(accuracy, mean_accuracy - 2 * std_accuracy, 
                             "Accuracy outlier detected")
            self.assertLess(accuracy, mean_accuracy + 2 * std_accuracy, 
                           "Accuracy outlier detected")


class TestCompetitionFormatCompliance(unittest.TestCase):
    """Automated testing for competition format compliance."""
    
    def setUp(self):
        """Set up format compliance test fixtures."""
        self.formatter = ICDAR2025Formatter()
        self.evaluator = ICDAR2025Evaluator()
        
        # Create sample data for all tasks
        self.sample_detections = TestDataFactory.create_sample_detections(count=5)
        self.sample_recognized_texts = TestDataFactory.create_sample_recognized_texts(self.sample_detections)
        self.sample_linked_texts = TestDataFactory.create_sample_linked_texts(self.sample_recognized_texts)
    
    def test_task1_format_compliance(self):
        """Test Task 1 output format compliance."""
        task_id = 'task1'
        
        # Generate formatted output
        formatted_output = self.formatter.format_task1_output(self.sample_detections)
        
        # Validate format
        is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
        self.assertTrue(is_valid, f"Task 1 format validation failed")
        
        # Check specific format requirements
        lines = formatted_output.strip().split('\n')
        self.assertGreater(len(lines), 0, "Empty output")
        
        for line in lines:
            if line.strip():  # Skip empty lines
                # Each line should contain bounding box coordinates
                parts = line.split(',')
                self.assertGreaterEqual(len(parts), 4, f"Invalid line format: {line}")
                
                # Coordinates should be numeric
                try:
                    coords = [float(part.strip()) for part in parts[:4]]
                    self.assertEqual(len(coords), 4, "Should have 4 coordinates")
                    
                    # Coordinates should be non-negative
                    for coord in coords:
                        self.assertGreaterEqual(coord, 0, "Negative coordinate")
                        
                except ValueError:
                    self.fail(f"Non-numeric coordinates in line: {line}")
    
    def test_task2_format_compliance(self):
        """Test Task 2 output format compliance."""
        task_id = 'task2'
        
        # Create phrase detections
        phrase_detections = []
        for detection in self.sample_detections:
            phrase_detection = Detection(
                bbox=detection.bbox,
                detection_type='phrase',
                confidence=detection.confidence,
                image_coordinates=detection.image_coordinates
            )
            phrase_detections.append(phrase_detection)
        
        # Generate and validate formatted output
        formatted_output = self.formatter.format_task2_output(phrase_detections)
        is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
        self.assertTrue(is_valid, f"Task 2 format validation failed")
        
        # Verify phrase-specific requirements
        lines = formatted_output.strip().split('\n')
        for line in lines:
            if line.strip():
                # Should contain phrase bounding box information
                self.assertIn(',', line, "Missing comma separators")
    
    def test_task3_format_compliance(self):
        """Test Task 3 output format compliance."""
        task_id = 'task3'
        
        # Generate and validate formatted output
        formatted_output = self.formatter.format_task3_output(self.sample_recognized_texts)
        is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
        self.assertTrue(is_valid, f"Task 3 format validation failed")
        
        # Check format requirements for detection + recognition
        lines = formatted_output.strip().split('\n')
        for line in lines:
            if line.strip():
                # Should contain both coordinates and text
                parts = line.split(',')
                self.assertGreaterEqual(len(parts), 5, f"Missing text in line: {line}")
                
                # First 4 parts should be coordinates
                try:
                    coords = [float(part.strip()) for part in parts[:4]]
                    self.assertEqual(len(coords), 4)
                except ValueError:
                    self.fail(f"Invalid coordinates in line: {line}")
                
                # Last part should be text
                text_part = ','.join(parts[4:]).strip()
                self.assertGreater(len(text_part), 0, "Empty text field")
    
    def test_task4_format_compliance(self):
        """Test Task 4 output format compliance."""
        task_id = 'task4'
        
        # Create phrase recognized texts
        phrase_recognized_texts = []
        for recognized_text in self.sample_recognized_texts:
            phrase_detection = Detection(
                bbox=recognized_text.detection.bbox,
                detection_type='phrase',
                confidence=recognized_text.detection.confidence,
                image_coordinates=recognized_text.detection.image_coordinates
            )
            
            phrase_recognized_text = RecognizedText(
                detection=phrase_detection,
                text=recognized_text.text,
                language=recognized_text.language,
                recognition_confidence=recognized_text.recognition_confidence,
                character_confidences=recognized_text.character_confidences
            )
            phrase_recognized_texts.append(phrase_recognized_text)
        
        # Generate and validate formatted output
        formatted_output = self.formatter.format_task4_output(phrase_recognized_texts)
        is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
        self.assertTrue(is_valid, f"Task 4 format validation failed")
    
    def test_multilingual_format_compliance(self):
        """Test format compliance with multilingual text."""
        # Create multilingual recognized texts
        multilingual_data = TestDataFactory.create_multilingual_test_data()
        
        multilingual_recognized_texts = []
        for i, detection in enumerate(self.sample_detections):
            if i % 2 == 0:
                text_data = multilingual_data['english'][0]
            else:
                text_data = multilingual_data['chinese'][0]
            
            recognized_text = RecognizedText(
                detection=detection,
                text=text_data['text'],
                language=text_data['language'],
                recognition_confidence=text_data['confidence'],
                character_confidences=text_data['character_confidences']
            )
            multilingual_recognized_texts.append(recognized_text)
        
        # Test format compliance for multilingual content
        for task_id in ['task3', 'task4']:
            with self.subTest(task_id=task_id):
                if task_id == 'task3':
                    formatted_output = self.formatter.format_task3_output(multilingual_recognized_texts)
                else:
                    formatted_output = self.formatter.format_task4_output(multilingual_recognized_texts)
                
                is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
                self.assertTrue(is_valid, f"Multilingual format validation failed for {task_id}")
                
                # Verify multilingual text is properly encoded
                self.assertIsInstance(formatted_output, str)
                # Should contain both English and Chinese characters
                has_english = any(ord(c) < 128 for c in formatted_output if c.isalpha())
                has_chinese = any(ord(c) > 127 for c in formatted_output)
                
                # At least one should be present (depending on test data)
                self.assertTrue(has_english or has_chinese, "No recognizable text characters found")
    
    def test_edge_case_format_compliance(self):
        """Test format compliance with edge cases."""
        # Test with empty results
        empty_detections = []
        empty_recognized_texts = []
        
        for task_id in ['task1', 'task2', 'task3', 'task4']:
            with self.subTest(task_id=task_id, case="empty"):
                if task_id in ['task1', 'task2']:
                    formatted_output = getattr(self.formatter, f'format_{task_id}_output')(empty_detections)
                else:
                    formatted_output = getattr(self.formatter, f'format_{task_id}_output')(empty_recognized_texts)
                
                # Empty results should still produce valid format
                is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
                self.assertTrue(is_valid, f"Empty results format validation failed for {task_id}")
        
        # Test with single detection/recognition
        single_detection = [self.sample_detections[0]]
        single_recognized_text = [self.sample_recognized_texts[0]]
        
        for task_id in ['task1', 'task2', 'task3', 'task4']:
            with self.subTest(task_id=task_id, case="single"):
                if task_id in ['task1', 'task2']:
                    formatted_output = getattr(self.formatter, f'format_{task_id}_output')(single_detection)
                else:
                    formatted_output = getattr(self.formatter, f'format_{task_id}_output')(single_recognized_text)
                
                is_valid = self.formatter.validate_competition_format(task_id, formatted_output)
                self.assertTrue(is_valid, f"Single result format validation failed for {task_id}")


if __name__ == '__main__':
    unittest.main()