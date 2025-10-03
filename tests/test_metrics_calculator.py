"""
Tests for competition metrics calculator.
"""

import pytest
import numpy as np
from src.evaluation.metrics_calculator import CompetitionMetricsCalculator
from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText
)


class TestCompetitionMetricsCalculator:
    """Test cases for competition metrics calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = CompetitionMetricsCalculator(iou_threshold=0.5)
        
        # Create sample bounding boxes
        self.bbox1 = BoundingBox(x=10, y=20, width=100, height=30, confidence=0.95)
        self.bbox2 = BoundingBox(x=150, y=50, width=80, height=25, confidence=0.87)
        self.bbox3 = BoundingBox(x=15, y=25, width=90, height=25, confidence=0.82)  # Overlaps with bbox1
        
        # Create sample detections
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
        
        self.detection3 = Detection(
            bbox=self.bbox3,
            detection_type='word',
            confidence=0.82,
            image_coordinates=(15, 25, 105, 50)
        )
        
        # Create sample recognized texts
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
        
        # Create sample linked texts
        self.linked_text1 = LinkedText(
            recognized_text=self.recognized_text1,
            map_coordinates=(100.5, 200.3),
            spatial_confidence=0.91,
            related_features=["river", "landmark"]
        )
        
        self.linked_text2 = LinkedText(
            recognized_text=self.recognized_text2,
            map_coordinates=(150.2, 180.7),
            spatial_confidence=0.85,
            related_features=["bridge", "crossing"]
        )
    
    def test_calculate_precision_recall_f1(self):
        """Test precision, recall, and F1 score calculation."""
        # Perfect case
        metrics = self.calculator.calculate_precision_recall_f1(tp=10, fp=0, fn=0)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        
        # Typical case
        metrics = self.calculator.calculate_precision_recall_f1(tp=8, fp=2, fn=3)
        expected_precision = 8 / (8 + 2)  # 0.8
        expected_recall = 8 / (8 + 3)     # 0.727...
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        
        assert abs(metrics['precision'] - expected_precision) < 1e-6
        assert abs(metrics['recall'] - expected_recall) < 1e-6
        assert abs(metrics['f1_score'] - expected_f1) < 1e-6
        
        # Edge case: no predictions
        metrics = self.calculator.calculate_precision_recall_f1(tp=0, fp=0, fn=5)
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
    
    def test_calculate_character_accuracy(self):
        """Test character-level accuracy calculation."""
        # Perfect match
        accuracy = self.calculator.calculate_character_accuracy("London", "London")
        assert accuracy == 1.0
        
        # Complete mismatch
        accuracy = self.calculator.calculate_character_accuracy("London", "Bridge")
        assert accuracy < 1.0
        
        # Partial match
        accuracy = self.calculator.calculate_character_accuracy("Londn", "London")
        assert 0.0 < accuracy < 1.0
        
        # Empty strings
        accuracy = self.calculator.calculate_character_accuracy("", "")
        assert accuracy == 1.0
        
        # One empty string
        accuracy = self.calculator.calculate_character_accuracy("London", "")
        assert accuracy == 0.0
        
        accuracy = self.calculator.calculate_character_accuracy("", "London")
        assert accuracy == 0.0
    
    def test_calculate_word_accuracy(self):
        """Test word-level accuracy calculation."""
        # Perfect match
        predicted = ["London", "Bridge", "Tower"]
        ground_truth = ["London", "Bridge", "Tower"]
        accuracy = self.calculator.calculate_word_accuracy(predicted, ground_truth)
        assert accuracy == 1.0
        
        # Partial match
        predicted = ["London", "Brdge", "Tower"]
        ground_truth = ["London", "Bridge", "Tower"]
        accuracy = self.calculator.calculate_word_accuracy(predicted, ground_truth)
        assert 0.0 < accuracy < 1.0
        
        # No match
        predicted = ["Paris", "Eiffel", "Tower"]
        ground_truth = ["London", "Bridge", "Tower"]
        accuracy = self.calculator.calculate_word_accuracy(predicted, ground_truth)
        assert 0.0 <= accuracy < 1.0
        
        # Empty lists
        accuracy = self.calculator.calculate_word_accuracy([], [])
        assert accuracy == 1.0
        
        # One empty list
        accuracy = self.calculator.calculate_word_accuracy(["London"], [])
        assert accuracy == 0.0
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        # Perfect overlap
        box1 = (10, 20, 110, 50)
        box2 = (10, 20, 110, 50)
        iou = self.calculator.calculate_iou(box1, box2)
        assert iou == 1.0
        
        # No overlap
        box1 = (10, 20, 50, 60)
        box2 = (100, 100, 150, 150)
        iou = self.calculator.calculate_iou(box1, box2)
        assert iou == 0.0
        
        # Partial overlap
        box1 = (10, 20, 110, 50)
        box2 = (50, 30, 150, 70)
        iou = self.calculator.calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        
        # Adjacent boxes (no overlap)
        box1 = (10, 20, 50, 60)
        box2 = (50, 20, 90, 60)
        iou = self.calculator.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_calculate_detection_metrics(self):
        """Test detection metrics calculation."""
        predictions = [self.detection1, self.detection2]
        ground_truth = [self.detection1, self.detection3]  # detection3 overlaps with detection1
        
        metrics = self.calculator.calculate_detection_metrics(predictions, ground_truth)
        
        # Should have some matches due to overlap
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'average_iou' in metrics
        assert 'true_positives' in metrics
        assert 'false_positives' in metrics
        assert 'false_negatives' in metrics
        
        # Check value ranges
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        assert 0.0 <= metrics['average_iou'] <= 1.0
    
    def test_calculate_detection_metrics_empty(self):
        """Test detection metrics with empty inputs."""
        # Empty predictions
        metrics = self.calculator.calculate_detection_metrics([], [self.detection1])
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
        
        # Empty ground truth
        metrics = self.calculator.calculate_detection_metrics([self.detection1], [])
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 0.0
        
        # Both empty
        metrics = self.calculator.calculate_detection_metrics([], [])
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_calculate_recognition_metrics(self):
        """Test recognition metrics calculation."""
        predictions = [self.recognized_text1, self.recognized_text2]
        ground_truth = ["London", "Bridge"]
        
        metrics = self.calculator.calculate_recognition_metrics(predictions, ground_truth)
        
        assert 'character_accuracy' in metrics
        assert 'word_accuracy' in metrics
        assert 'exact_match_accuracy' in metrics
        assert 'average_confidence' in metrics
        
        # Perfect match should give high accuracy
        assert metrics['character_accuracy'] == 1.0
        assert metrics['word_accuracy'] == 1.0
        assert metrics['exact_match_accuracy'] == 1.0
        assert 0.0 < metrics['average_confidence'] <= 1.0
    
    def test_calculate_recognition_metrics_partial_match(self):
        """Test recognition metrics with partial matches."""
        predictions = [self.recognized_text1, self.recognized_text2]
        ground_truth = ["Londn", "Brdge"]  # Typos
        
        metrics = self.calculator.calculate_recognition_metrics(predictions, ground_truth)
        
        # Should have good but not perfect accuracy
        assert 0.0 < metrics['character_accuracy'] < 1.0
        assert 0.0 <= metrics['word_accuracy'] < 1.0
        assert metrics['exact_match_accuracy'] == 0.0  # No exact matches
    
    def test_calculate_linking_metrics(self):
        """Test linking metrics calculation."""
        predictions = [self.linked_text1, self.linked_text2]
        ground_truth = [self.linked_text1, self.linked_text2]  # Perfect match
        
        metrics = self.calculator.calculate_linking_metrics(predictions, ground_truth)
        
        assert 'spatial_accuracy' in metrics
        assert 'coordinate_precision' in metrics
        assert 'average_spatial_confidence' in metrics
        assert 'linking_precision' in metrics
        assert 'linking_recall' in metrics
        assert 'linking_f1' in metrics
        
        # Perfect match should give high scores
        assert metrics['linking_precision'] == 1.0
        assert metrics['linking_recall'] == 1.0
        assert metrics['linking_f1'] == 1.0
    
    def test_calculate_linking_metrics_spatial_error(self):
        """Test linking metrics with spatial errors."""
        # Create ground truth with different coordinates
        gt_linked_text1 = LinkedText(
            recognized_text=self.recognized_text1,
            map_coordinates=(105.0, 205.0),  # Slightly different from prediction
            spatial_confidence=0.90,
            related_features=["river", "landmark"]
        )
        
        predictions = [self.linked_text1]
        ground_truth = [gt_linked_text1]
        
        metrics = self.calculator.calculate_linking_metrics(predictions, ground_truth)
        
        # Should still match on text but have spatial error
        assert metrics['linking_precision'] == 1.0
        assert metrics['coordinate_precision'] < 1.0
        assert 'average_spatial_error' in metrics
        assert metrics['average_spatial_error'] > 0.0
    
    def test_generate_evaluation_report(self):
        """Test comprehensive evaluation report generation."""
        detection_metrics = {'precision': 0.9, 'recall': 0.85, 'f1_score': 0.875}
        recognition_metrics = {'character_accuracy': 0.92, 'word_accuracy': 0.88, 'exact_match_accuracy': 0.75}
        linking_metrics = {'spatial_accuracy': 0.80, 'linking_f1': 0.82}
        
        report = self.calculator.generate_evaluation_report(
            detection_metrics=detection_metrics,
            recognition_metrics=recognition_metrics,
            linking_metrics=linking_metrics
        )
        
        assert 'summary' in report
        assert 'detailed_metrics' in report
        
        # Check summary contains key metrics
        assert 'detection_f1' in report['summary']
        assert 'character_accuracy' in report['summary']
        assert 'spatial_accuracy' in report['summary']
        assert 'overall_score' in report['summary']
        
        # Check detailed metrics
        assert 'detection' in report['detailed_metrics']
        assert 'recognition' in report['detailed_metrics']
        assert 'linking' in report['detailed_metrics']
        
        # Overall score should be reasonable
        assert 0.0 <= report['summary']['overall_score'] <= 1.0
    
    def test_edit_distance_calculation(self):
        """Test edit distance calculation."""
        # Identical strings
        distance = self.calculator._calculate_edit_distance("hello", "hello")
        assert distance == 0
        
        # One character difference
        distance = self.calculator._calculate_edit_distance("hello", "hallo")
        assert distance == 1
        
        # Insertion
        distance = self.calculator._calculate_edit_distance("hello", "helloo")
        assert distance == 1
        
        # Deletion
        distance = self.calculator._calculate_edit_distance("hello", "hell")
        assert distance == 1
        
        # Complete difference
        distance = self.calculator._calculate_edit_distance("abc", "xyz")
        assert distance == 3
        
        # Empty strings
        distance = self.calculator._calculate_edit_distance("", "")
        assert distance == 0
        
        distance = self.calculator._calculate_edit_distance("hello", "")
        assert distance == 5
    
    def test_match_detections_by_iou(self):
        """Test detection matching by IoU."""
        predictions = [self.detection1, self.detection2]
        ground_truth = [self.detection3, self.detection2]  # detection3 overlaps with detection1
        
        matches, ious = self.calculator._match_detections_by_iou(predictions, ground_truth)
        
        # Should find at least one match (detection2 with itself)
        assert len(matches) >= 1
        assert len(ious) == len(matches)
        
        # All IoUs should be above threshold
        for iou in ious:
            assert iou >= self.calculator.iou_threshold
        
        # Matches should be valid indices
        for pred_idx, gt_idx in matches:
            assert 0 <= pred_idx < len(predictions)
            assert 0 <= gt_idx < len(ground_truth)
    
    def test_match_linked_texts(self):
        """Test linked text matching."""
        predictions = [self.linked_text1, self.linked_text2]
        ground_truth = [self.linked_text1, self.linked_text2]
        
        matches = self.calculator._match_linked_texts(predictions, ground_truth)
        
        # Should match all texts perfectly
        assert len(matches) == 2
        
        # Matches should be valid indices
        for pred_idx, gt_idx in matches:
            assert 0 <= pred_idx < len(predictions)
            assert 0 <= gt_idx < len(ground_truth)
    
    def test_calculate_text_similarity(self):
        """Test text similarity calculation."""
        # Identical text
        similarity = self.calculator._calculate_text_similarity("London", "London")
        assert similarity == 1.0
        
        # Similar text
        similarity = self.calculator._calculate_text_similarity("London", "Londn")
        assert 0.0 < similarity < 1.0
        
        # Different text
        similarity = self.calculator._calculate_text_similarity("London", "Paris")
        assert 0.0 <= similarity < 1.0


if __name__ == "__main__":
    pytest.main([__file__])