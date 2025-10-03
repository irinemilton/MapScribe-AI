"""
Tests for comprehensive competition evaluator.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.evaluation.competition_evaluator import ICDAR2025Evaluator
from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText, ProcessingResult
)


class TestICDAR2025Evaluator:
    """Test cases for ICDAR 2025 comprehensive evaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ICDAR2025Evaluator(iou_threshold=0.5)
        
        # Create sample data
        self.bbox1 = BoundingBox(x=10, y=20, width=100, height=30, confidence=0.95)
        self.bbox2 = BoundingBox(x=150, y=50, width=80, height=25, confidence=0.87)
        
        self.detection1 = Detection(
            bbox=self.bbox1,
            detection_type='word',
            confidence=0.95,
            image_coordinates=(10, 20, 110, 50)
        )
        
        self.detection2 = Detection(
            bbox=self.bbox2,
            detection_type='word',
            confidence=0.87,
            image_coordinates=(150, 50, 230, 75)
        )
        
        self.recognized_text1 = RecognizedText(
            detection=self.detection1,
            text="London",
            language="en",
            recognition_confidence=0.93,
            character_confidences=[0.95, 0.92, 0.94, 0.91, 0.96, 0.89]
        )
        
        self.recognized_text2 = RecognizedText(
            detection=self.detection2,
            text="Bridge",
            language="en",
            recognition_confidence=0.88,
            character_confidences=[0.90, 0.87, 0.89, 0.85, 0.91, 0.86]
        )
        
        self.linked_text1 = LinkedText(
            recognized_text=self.recognized_text1,
            map_coordinates=(100.5, 200.3),
            spatial_confidence=0.91,
            related_features=["river", "landmark"]
        )
        
        self.processing_result = ProcessingResult(
            task_type='word_detection_recognition',
            detections=[self.detection1, self.detection2],
            recognized_texts=[self.recognized_text1, self.recognized_text2],
            linked_texts=[self.linked_text1],
            processing_time=1.5,
            metrics={'precision': 0.95, 'recall': 0.87}
        )
    
    def test_evaluate_detection(self):
        """Test detection evaluation."""
        predictions = [self.detection1, self.detection2]
        ground_truth = [self.detection1, self.detection2]  # Perfect match
        
        metrics = self.evaluator.evaluate_detection(predictions, ground_truth)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'average_iou' in metrics
        
        # Perfect match should give high scores
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_evaluate_recognition(self):
        """Test recognition evaluation."""
        predictions = [self.recognized_text1, self.recognized_text2]
        ground_truth = ["London", "Bridge"]
        
        metrics = self.evaluator.evaluate_recognition(predictions, ground_truth)
        
        assert 'character_accuracy' in metrics
        assert 'word_accuracy' in metrics
        assert 'exact_match_accuracy' in metrics
        assert 'average_confidence' in metrics
        
        # Perfect match should give high accuracy
        assert metrics['character_accuracy'] == 1.0
        assert metrics['word_accuracy'] == 1.0
        assert metrics['exact_match_accuracy'] == 1.0
    
    def test_evaluate_linking(self):
        """Test linking evaluation."""
        predictions = [self.linked_text1]
        ground_truth = [self.linked_text1]  # Perfect match
        
        metrics = self.evaluator.evaluate_linking(predictions, ground_truth)
        
        assert 'spatial_accuracy' in metrics
        assert 'coordinate_precision' in metrics
        assert 'linking_precision' in metrics
        assert 'linking_recall' in metrics
        assert 'linking_f1' in metrics
        
        # Perfect match should give high scores
        assert metrics['linking_precision'] == 1.0
        assert metrics['linking_recall'] == 1.0
        assert metrics['linking_f1'] == 1.0
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        box1 = (10, 20, 110, 50)
        box2 = (10, 20, 110, 50)
        iou = self.evaluator.calculate_iou(box1, box2)
        assert iou == 1.0
        
        box1 = (10, 20, 50, 60)
        box2 = (100, 100, 150, 150)
        iou = self.evaluator.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_evaluate_competition_task_task1(self):
        """Test evaluation of Task 1 (Word Detection)."""
        ground_truth = {
            'detections': [
                {
                    'bbox': {'x': 10, 'y': 20, 'width': 100, 'height': 30},
                    'detection_type': 'word',
                    'confidence': 0.95,
                    'image_coordinates': [10, 20, 110, 50]
                },
                {
                    'bbox': {'x': 150, 'y': 50, 'width': 80, 'height': 25},
                    'detection_type': 'word',
                    'confidence': 0.87,
                    'image_coordinates': [150, 50, 230, 75]
                }
            ]
        }
        
        results = self.evaluator.evaluate_competition_task(
            'task1', self.processing_result, ground_truth, 'test_image.jpg'
        )
        
        assert results['task_id'] == 'task1'
        assert results['image_name'] == 'test_image.jpg'
        assert 'formatted_output' in results
        assert 'metrics' in results
        assert 'validation' in results
        assert results['validation']['format_valid'] is True
        
        # Should have detection metrics
        assert 'precision' in results['metrics']
        assert 'recall' in results['metrics']
        assert 'f1_score' in results['metrics']
    
    def test_evaluate_competition_task_task3(self):
        """Test evaluation of Task 3 (Word Detection + Recognition)."""
        ground_truth = {
            'detections': [
                {
                    'bbox': {'x': 10, 'y': 20, 'width': 100, 'height': 30},
                    'detection_type': 'word',
                    'confidence': 0.95,
                    'image_coordinates': [10, 20, 110, 50]
                }
            ],
            'texts': ["London", "Bridge"]
        }
        
        results = self.evaluator.evaluate_competition_task(
            'task3', self.processing_result, ground_truth, 'test_image.jpg'
        )
        
        assert results['task_id'] == 'task3'
        assert 'formatted_output' in results
        assert 'metrics' in results
        assert results['validation']['format_valid'] is True
        
        # Should have both detection and recognition metrics
        assert 'detection' in results['metrics']
        assert 'recognition' in results['metrics']
        
        # Should have evaluation report
        assert 'evaluation_report' in results
        assert 'summary' in results['evaluation_report']
        assert 'detailed_metrics' in results['evaluation_report']
    
    def test_evaluate_batch_results(self):
        """Test batch evaluation."""
        batch_predictions = [self.processing_result, self.processing_result]
        batch_ground_truth = [
            {
                'detections': [
                    {
                        'bbox': {'x': 10, 'y': 20, 'width': 100, 'height': 30},
                        'detection_type': 'word',
                        'confidence': 0.95,
                        'image_coordinates': [10, 20, 110, 50]
                    }
                ]
            },
            {
                'detections': [
                    {
                        'bbox': {'x': 150, 'y': 50, 'width': 80, 'height': 25},
                        'detection_type': 'word',
                        'confidence': 0.87,
                        'image_coordinates': [150, 50, 230, 75]
                    }
                ]
            }
        ]
        image_names = ['image1.jpg', 'image2.jpg']
        
        results = self.evaluator.evaluate_batch_results(
            'task1', batch_predictions, batch_ground_truth, image_names
        )
        
        assert results['task_id'] == 'task1'
        assert results['total_samples'] == 2
        assert 'individual_results' in results
        assert 'aggregated_metrics' in results
        
        # Check individual results
        assert len(results['individual_results']) == 2
        for result in results['individual_results']:
            assert 'task_id' in result
            assert 'image_name' in result
            assert 'formatted_output' in result
            assert 'metrics' in result
        
        # Check aggregated metrics
        if 'detection' in results['aggregated_metrics']:
            det_metrics = results['aggregated_metrics']['detection']
            assert 'mean_precision' in det_metrics
            assert 'mean_recall' in det_metrics
            assert 'mean_f1_score' in det_metrics
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        results = {
            'task_id': 'task1',
            'metrics': {'precision': 0.95, 'recall': 0.87, 'f1_score': 0.91}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "evaluation_results.json"
            
            self.evaluator.save_evaluation_results(results, output_path)
            
            # Verify file was created and contains correct content
            assert output_path.exists()
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            
            assert loaded_results == results
    
    def test_save_competition_submission(self):
        """Test saving competition submission."""
        formatted_output = "10,20,110,50,0.9500"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "task1_submission.txt"
            
            self.evaluator.save_competition_submission(
                'task1', formatted_output, output_path, 'test_image.jpg'
            )
            
            # Verify file was created and contains correct content
            assert output_path.exists()
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert content == formatted_output
    
    def test_parse_ground_truth_detections(self):
        """Test parsing ground truth detection data."""
        gt_data = [
            {
                'bbox': {'x': 10, 'y': 20, 'width': 100, 'height': 30},
                'detection_type': 'word',
                'confidence': 0.95,
                'image_coordinates': [10, 20, 110, 50]
            }
        ]
        
        detections = self.evaluator._parse_ground_truth_detections(gt_data)
        
        assert len(detections) == 1
        detection = detections[0]
        assert detection.bbox.x == 10
        assert detection.bbox.y == 20
        assert detection.bbox.width == 100
        assert detection.bbox.height == 30
        assert detection.detection_type == 'word'
        assert detection.confidence == 0.95
    
    def test_parse_ground_truth_linked_texts(self):
        """Test parsing ground truth linked text data."""
        gt_data = [
            {
                'detection': {
                    'bbox': {'x': 10, 'y': 20, 'width': 100, 'height': 30},
                    'detection_type': 'word',
                    'confidence': 0.95,
                    'image_coordinates': [10, 20, 110, 50]
                },
                'text': 'London',
                'language': 'en',
                'recognition_confidence': 0.93,
                'character_confidences': [0.95, 0.92, 0.94, 0.91, 0.96, 0.89],
                'map_coordinates': [100.5, 200.3],
                'spatial_confidence': 0.91,
                'related_features': ['river', 'landmark']
            }
        ]
        
        linked_texts = self.evaluator._parse_ground_truth_linked_texts(gt_data)
        
        assert len(linked_texts) == 1
        linked_text = linked_texts[0]
        assert linked_text.recognized_text.text == 'London'
        assert linked_text.map_coordinates == (100.5, 200.3)
        assert linked_text.spatial_confidence == 0.91
        assert linked_text.related_features == ['river', 'landmark']
    
    def test_evaluate_batch_results_mismatched_lengths(self):
        """Test batch evaluation with mismatched input lengths."""
        batch_predictions = [self.processing_result]
        batch_ground_truth = [{'detections': []}, {'detections': []}]  # Different length
        
        with pytest.raises(ValueError, match="Number of predictions must match"):
            self.evaluator.evaluate_batch_results('task1', batch_predictions, batch_ground_truth)
    
    def test_evaluate_competition_task_with_errors(self):
        """Test competition task evaluation with invalid data."""
        # Invalid ground truth format
        invalid_ground_truth = {'invalid_key': 'invalid_value'}
        
        results = self.evaluator.evaluate_competition_task(
            'task1', self.processing_result, invalid_ground_truth
        )
        
        # Should still return results structure but may have errors
        assert 'task_id' in results
        assert 'formatted_output' in results
        assert 'validation' in results


if __name__ == "__main__":
    pytest.main([__file__])