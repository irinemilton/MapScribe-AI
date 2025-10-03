"""
Evaluation metrics calculator for ICDAR 2025 competition tasks.
Implements precision, recall, F1-score, IoU-based detection metrics, 
and character/word-level accuracy measurements.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import difflib
from ..models.data_structures import Detection, RecognizedText, LinkedText, BoundingBox
from .interfaces import MetricsCalculatorInterface


class CompetitionMetricsCalculator(MetricsCalculatorInterface):
    """
    Comprehensive metrics calculator for ICDAR 2025 competition evaluation.
    Supports detection, recognition, and linking evaluation metrics.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for detection matching (default: 0.5)
        """
        self.iou_threshold = iou_threshold
    
    def calculate_precision_recall_f1(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score from confusion matrix values.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def calculate_character_accuracy(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate character-level accuracy using edit distance.
        
        Args:
            predicted: Predicted text string
            ground_truth: Ground truth text string
            
        Returns:
            Character accuracy score (0.0 to 1.0)
        """
        if not ground_truth:
            return 1.0 if not predicted else 0.0
        
        if not predicted:
            return 0.0
        
        # Calculate edit distance (Levenshtein distance)
        edit_distance = self._calculate_edit_distance(predicted, ground_truth)
        
        # Calculate accuracy as 1 - (edit_distance / max_length)
        max_length = max(len(predicted), len(ground_truth))
        accuracy = 1.0 - (edit_distance / max_length) if max_length > 0 else 1.0
        
        return max(0.0, accuracy)  # Ensure non-negative
    
    def calculate_word_accuracy(self, predicted: List[str], ground_truth: List[str]) -> float:
        """
        Calculate word-level accuracy.
        
        Args:
            predicted: List of predicted words
            ground_truth: List of ground truth words
            
        Returns:
            Word accuracy score (0.0 to 1.0)
        """
        if not ground_truth:
            return 1.0 if not predicted else 0.0
        
        if not predicted:
            return 0.0
        
        # Count exact word matches
        correct_words = 0
        total_words = len(ground_truth)
        
        # Use sequence matching to align predicted and ground truth words
        matcher = difflib.SequenceMatcher(None, predicted, ground_truth)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                correct_words += (i2 - i1)
        
        return correct_words / total_words if total_words > 0 else 0.0
    
    def calculate_detection_metrics(self, predictions: List[Detection], ground_truth: List[Detection]) -> Dict[str, float]:
        """
        Calculate detection evaluation metrics using IoU-based matching.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            Dictionary of detection metrics
        """
        if not ground_truth:
            return {
                'precision': 1.0 if not predictions else 0.0,
                'recall': 1.0,
                'f1_score': 1.0 if not predictions else 0.0,
                'average_iou': 0.0,
                'true_positives': 0,
                'false_positives': len(predictions),
                'false_negatives': 0
            }
        
        if not predictions:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'average_iou': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(ground_truth)
            }
        
        # Match predictions to ground truth using IoU
        matches, ious = self._match_detections_by_iou(predictions, ground_truth)
        
        # Count true positives, false positives, false negatives
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        # Calculate average IoU for matched detections
        avg_iou = np.mean(ious) if ious else 0.0
        
        # Calculate precision, recall, F1
        metrics = self.calculate_precision_recall_f1(tp, fp, fn)
        metrics['average_iou'] = avg_iou
        
        return metrics
    
    def calculate_recognition_metrics(self, predictions: List[RecognizedText], ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate recognition evaluation metrics.
        
        Args:
            predictions: List of predicted recognized text
            ground_truth: List of ground truth text strings
            
        Returns:
            Dictionary of recognition metrics
        """
        if not ground_truth:
            return {
                'character_accuracy': 1.0 if not predictions else 0.0,
                'word_accuracy': 1.0 if not predictions else 0.0,
                'exact_match_accuracy': 1.0 if not predictions else 0.0,
                'average_confidence': 0.0
            }
        
        if not predictions:
            return {
                'character_accuracy': 0.0,
                'word_accuracy': 0.0,
                'exact_match_accuracy': 0.0,
                'average_confidence': 0.0
            }
        
        # Extract predicted text strings
        predicted_texts = [rt.text for rt in predictions]
        
        # Align predictions with ground truth (take minimum length)
        min_length = min(len(predicted_texts), len(ground_truth))
        aligned_predictions = predicted_texts[:min_length]
        aligned_ground_truth = ground_truth[:min_length]
        
        # Calculate character-level accuracy
        char_accuracies = []
        for pred, gt in zip(aligned_predictions, aligned_ground_truth):
            char_acc = self.calculate_character_accuracy(pred, gt)
            char_accuracies.append(char_acc)
        
        avg_char_accuracy = np.mean(char_accuracies) if char_accuracies else 0.0
        
        # Calculate word-level accuracy
        pred_words = []
        gt_words = []
        for pred, gt in zip(aligned_predictions, aligned_ground_truth):
            pred_words.extend(pred.split())
            gt_words.extend(gt.split())
        
        word_accuracy = self.calculate_word_accuracy(pred_words, gt_words)
        
        # Calculate exact match accuracy
        exact_matches = sum(1 for pred, gt in zip(aligned_predictions, aligned_ground_truth) if pred == gt)
        exact_match_accuracy = exact_matches / len(aligned_ground_truth) if aligned_ground_truth else 0.0
        
        # Calculate average confidence
        confidences = [rt.recognition_confidence for rt in predictions[:min_length]]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'character_accuracy': avg_char_accuracy,
            'word_accuracy': word_accuracy,
            'exact_match_accuracy': exact_match_accuracy,
            'average_confidence': avg_confidence,
            'total_predictions': len(predictions),
            'total_ground_truth': len(ground_truth)
        }
    
    def calculate_linking_metrics(self, predictions: List[LinkedText], ground_truth: List[LinkedText]) -> Dict[str, float]:
        """
        Calculate text linking evaluation metrics.
        
        Args:
            predictions: List of predicted linked text
            ground_truth: List of ground truth linked text
            
        Returns:
            Dictionary of linking metrics
        """
        if not ground_truth:
            return {
                'spatial_accuracy': 1.0 if not predictions else 0.0,
                'coordinate_precision': 0.0,
                'average_spatial_confidence': 0.0,
                'linking_precision': 1.0 if not predictions else 0.0,
                'linking_recall': 1.0,
                'linking_f1': 1.0 if not predictions else 0.0
            }
        
        if not predictions:
            return {
                'spatial_accuracy': 0.0,
                'coordinate_precision': 0.0,
                'average_spatial_confidence': 0.0,
                'linking_precision': 0.0,
                'linking_recall': 0.0,
                'linking_f1': 0.0
            }
        
        # Match linked texts based on text content and spatial proximity
        matches = self._match_linked_texts(predictions, ground_truth)
        
        # Calculate linking precision, recall, F1
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        linking_metrics = self.calculate_precision_recall_f1(tp, fp, fn)
        
        # Calculate spatial accuracy for matched pairs
        spatial_errors = []
        for pred_idx, gt_idx in matches:
            pred_coords = predictions[pred_idx].map_coordinates
            gt_coords = ground_truth[gt_idx].map_coordinates
            
            # Calculate Euclidean distance
            distance = np.sqrt((pred_coords[0] - gt_coords[0])**2 + (pred_coords[1] - gt_coords[1])**2)
            spatial_errors.append(distance)
        
        # Calculate coordinate precision (inverse of average error)
        avg_spatial_error = np.mean(spatial_errors) if spatial_errors else float('inf')
        coordinate_precision = 1.0 / (1.0 + avg_spatial_error)
        
        # Calculate spatial accuracy (percentage within threshold)
        spatial_threshold = 10.0  # pixels
        accurate_links = sum(1 for error in spatial_errors if error <= spatial_threshold)
        spatial_accuracy = accurate_links / len(matches) if matches else 0.0
        
        # Calculate average spatial confidence
        confidences = [lt.spatial_confidence for lt in predictions]
        avg_spatial_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'spatial_accuracy': spatial_accuracy,
            'coordinate_precision': coordinate_precision,
            'average_spatial_confidence': avg_spatial_confidence,
            'linking_precision': linking_metrics['precision'],
            'linking_recall': linking_metrics['recall'],
            'linking_f1': linking_metrics['f1_score'],
            'average_spatial_error': avg_spatial_error if spatial_errors else 0.0
        }
    
    def generate_evaluation_report(self, 
                                 detection_metrics: Optional[Dict[str, float]] = None,
                                 recognition_metrics: Optional[Dict[str, float]] = None,
                                 linking_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            detection_metrics: Detection evaluation metrics
            recognition_metrics: Recognition evaluation metrics
            linking_metrics: Linking evaluation metrics
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'summary': {},
            'detailed_metrics': {}
        }
        
        if detection_metrics:
            report['detailed_metrics']['detection'] = detection_metrics
            report['summary']['detection_f1'] = detection_metrics.get('f1_score', 0.0)
            report['summary']['detection_precision'] = detection_metrics.get('precision', 0.0)
            report['summary']['detection_recall'] = detection_metrics.get('recall', 0.0)
        
        if recognition_metrics:
            report['detailed_metrics']['recognition'] = recognition_metrics
            report['summary']['character_accuracy'] = recognition_metrics.get('character_accuracy', 0.0)
            report['summary']['word_accuracy'] = recognition_metrics.get('word_accuracy', 0.0)
            report['summary']['exact_match_accuracy'] = recognition_metrics.get('exact_match_accuracy', 0.0)
        
        if linking_metrics:
            report['detailed_metrics']['linking'] = linking_metrics
            report['summary']['spatial_accuracy'] = linking_metrics.get('spatial_accuracy', 0.0)
            report['summary']['linking_f1'] = linking_metrics.get('linking_f1', 0.0)
        
        # Calculate overall score (weighted average)
        scores = []
        if detection_metrics:
            scores.append(detection_metrics.get('f1_score', 0.0))
        if recognition_metrics:
            scores.append(recognition_metrics.get('character_accuracy', 0.0))
        if linking_metrics:
            scores.append(linking_metrics.get('spatial_accuracy', 0.0))
        
        report['summary']['overall_score'] = np.mean(scores) if scores else 0.0
        
        return report
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_i <= x1_i or y2_i <= y1_i:
            intersection_area = 0.0
        else:
            intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate edit distance (Levenshtein distance) between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _match_detections_by_iou(self, predictions: List[Detection], ground_truth: List[Detection]) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Match predictions to ground truth using IoU-based assignment.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            Tuple of (matches, ious) where matches is list of (pred_idx, gt_idx) pairs
        """
        matches = []
        ious = []
        used_gt = set()
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        for i, pred in enumerate(predictions):
            pred_box = pred.bbox.coordinates
            for j, gt in enumerate(ground_truth):
                gt_box = gt.bbox.coordinates
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Greedy matching: assign each prediction to best available ground truth
        for i in range(len(predictions)):
            best_iou = 0.0
            best_j = -1
            
            for j in range(len(ground_truth)):
                if j not in used_gt and iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= self.iou_threshold:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            
            if best_j >= 0:
                matches.append((i, best_j))
                ious.append(best_iou)
                used_gt.add(best_j)
        
        return matches, ious
    
    def _match_linked_texts(self, predictions: List[LinkedText], ground_truth: List[LinkedText]) -> List[Tuple[int, int]]:
        """
        Match predicted linked texts to ground truth based on text content and spatial proximity.
        
        Args:
            predictions: List of predicted linked texts
            ground_truth: List of ground truth linked texts
            
        Returns:
            List of (pred_idx, gt_idx) matches
        """
        matches = []
        used_gt = set()
        
        for i, pred in enumerate(predictions):
            best_score = 0.0
            best_j = -1
            
            for j, gt in enumerate(ground_truth):
                if j in used_gt:
                    continue
                
                # Calculate text similarity
                text_similarity = self._calculate_text_similarity(pred.recognized_text.text, gt.recognized_text.text)
                
                # Calculate spatial proximity (inverse of distance)
                spatial_distance = np.sqrt(
                    (pred.map_coordinates[0] - gt.map_coordinates[0])**2 + 
                    (pred.map_coordinates[1] - gt.map_coordinates[1])**2
                )
                spatial_similarity = 1.0 / (1.0 + spatial_distance)
                
                # Combined score (weighted)
                combined_score = 0.7 * text_similarity + 0.3 * spatial_similarity
                
                if combined_score > best_score and combined_score > 0.5:  # Threshold for matching
                    best_score = combined_score
                    best_j = j
            
            if best_j >= 0:
                matches.append((i, best_j))
                used_gt.add(best_j)
        
        return matches
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using character-level accuracy.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Text similarity score (0.0 to 1.0)
        """
        return self.calculate_character_accuracy(text1, text2)