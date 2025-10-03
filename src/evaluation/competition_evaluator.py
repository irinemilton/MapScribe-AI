"""
Comprehensive evaluator for ICDAR 2025 competition tasks.
Integrates formatting and metrics calculation for complete evaluation pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from .interfaces import EvaluatorInterface
from .competition_formatter import ICDAR2025Formatter
from .metrics_calculator import CompetitionMetricsCalculator
from ..models.data_structures import (
    Detection, RecognizedText, LinkedText, ProcessingResult, 
    BoundingBox
)


class ICDAR2025Evaluator(EvaluatorInterface):
    """
    Comprehensive evaluator for ICDAR 2025 Historical Map Text Processing competition.
    Provides complete evaluation pipeline including formatting and metrics calculation.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the evaluator.
        
        Args:
            iou_threshold: IoU threshold for detection matching (default: 0.5)
        """
        self.formatter = ICDAR2025Formatter()
        self.metrics_calculator = CompetitionMetricsCalculator(iou_threshold=iou_threshold)
        self.iou_threshold = iou_threshold
    
    def evaluate_detection(self, predictions: List[Detection], ground_truth: List[Detection]) -> Dict[str, float]:
        """
        Evaluate detection performance against ground truth.
        
        Args:
            predictions: Predicted detections
            ground_truth: Ground truth detections
            
        Returns:
            Dictionary of evaluation metrics
        """
        return self.metrics_calculator.calculate_detection_metrics(predictions, ground_truth)
    
    def evaluate_recognition(self, predictions: List[RecognizedText], ground_truth: List[str]) -> Dict[str, float]:
        """
        Evaluate recognition performance against ground truth.
        
        Args:
            predictions: Predicted recognized text
            ground_truth: Ground truth text strings
            
        Returns:
            Dictionary of evaluation metrics
        """
        return self.metrics_calculator.calculate_recognition_metrics(predictions, ground_truth)
    
    def evaluate_linking(self, predictions: List[LinkedText], ground_truth: List[LinkedText]) -> Dict[str, float]:
        """
        Evaluate text linking performance against ground truth.
        
        Args:
            predictions: Predicted linked text
            ground_truth: Ground truth linked text
            
        Returns:
            Dictionary of evaluation metrics
        """
        return self.metrics_calculator.calculate_linking_metrics(predictions, ground_truth)
    
    def calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score
        """
        return self.metrics_calculator.calculate_iou(box1, box2)
    
    def evaluate_competition_task(self, 
                                task_id: str,
                                predictions: ProcessingResult,
                                ground_truth: Dict[str, Any],
                                image_name: str = "") -> Dict[str, Any]:
        """
        Evaluate a complete competition task with formatting and metrics.
        
        Args:
            task_id: Competition task identifier ('task1', 'task2', 'task3', 'task4')
            predictions: Processing results from the system
            ground_truth: Ground truth data for evaluation
            image_name: Name of the processed image
            
        Returns:
            Complete evaluation results including formatted output and metrics
        """
        results = {
            'task_id': task_id,
            'image_name': image_name,
            'formatted_output': '',
            'metrics': {},
            'validation': {'format_valid': False, 'errors': []}
        }
        
        try:
            # Generate formatted output
            formatted_output = self.formatter.format_processing_result(predictions, task_id, image_name)
            results['formatted_output'] = formatted_output
            
            # Validate format
            format_valid = self.formatter.validate_competition_format(task_id, formatted_output)
            results['validation']['format_valid'] = format_valid
            
            if not format_valid:
                results['validation']['errors'].append(f"Invalid format for {task_id}")
            
            # Calculate metrics based on task type
            if task_id in ['task1', 'task2']:
                # Detection-only tasks
                if 'detections' in ground_truth:
                    gt_detections = self._parse_ground_truth_detections(ground_truth['detections'])
                    detection_metrics = self.evaluate_detection(predictions.detections, gt_detections)
                    results['metrics'] = detection_metrics
                else:
                    # No ground truth available, set empty metrics
                    results['metrics'] = {}
                
            elif task_id in ['task3', 'task4']:
                # Detection + Recognition tasks
                metrics = {}
                
                # Detection metrics
                if 'detections' in ground_truth:
                    gt_detections = self._parse_ground_truth_detections(ground_truth['detections'])
                    detection_metrics = self.evaluate_detection(predictions.detections, gt_detections)
                    metrics['detection'] = detection_metrics
                
                # Recognition metrics
                if 'texts' in ground_truth:
                    gt_texts = ground_truth['texts']
                    recognition_metrics = self.evaluate_recognition(predictions.recognized_texts, gt_texts)
                    metrics['recognition'] = recognition_metrics
                
                # Linking metrics (if available)
                if 'linked_texts' in ground_truth and predictions.linked_texts:
                    gt_linked_texts = self._parse_ground_truth_linked_texts(ground_truth['linked_texts'])
                    linking_metrics = self.evaluate_linking(predictions.linked_texts, gt_linked_texts)
                    metrics['linking'] = linking_metrics
                
                results['metrics'] = metrics
                
                # If no metrics were calculated, set empty dict
                if not results['metrics']:
                    results['metrics'] = {}
            
            # Generate comprehensive report
            if results['metrics']:
                detection_metrics = results['metrics'].get('detection')
                recognition_metrics = results['metrics'].get('recognition')
                linking_metrics = results['metrics'].get('linking')
                
                # For single-metric tasks, wrap in appropriate structure
                if task_id in ['task1', 'task2'] and isinstance(results['metrics'], dict) and 'precision' in results['metrics']:
                    detection_metrics = results['metrics']
                    recognition_metrics = None
                    linking_metrics = None
                
                report = self.metrics_calculator.generate_evaluation_report(
                    detection_metrics=detection_metrics,
                    recognition_metrics=recognition_metrics,
                    linking_metrics=linking_metrics
                )
                results['evaluation_report'] = report
        
        except Exception as e:
            results['validation']['errors'].append(f"Evaluation error: {str(e)}")
        
        return results
    
    def evaluate_batch_results(self, 
                             task_id: str,
                             batch_predictions: List[ProcessingResult],
                             batch_ground_truth: List[Dict[str, Any]],
                             image_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate batch processing results for a competition task.
        
        Args:
            task_id: Competition task identifier
            batch_predictions: List of processing results
            batch_ground_truth: List of ground truth data
            image_names: Optional list of image names
            
        Returns:
            Aggregated evaluation results for the batch
        """
        if len(batch_predictions) != len(batch_ground_truth):
            raise ValueError("Number of predictions must match number of ground truth samples")
        
        if image_names is None:
            image_names = [f"image_{i}" for i in range(len(batch_predictions))]
        
        individual_results = []
        aggregated_metrics = {
            'detection': {'precision': [], 'recall': [], 'f1_score': [], 'average_iou': []},
            'recognition': {'character_accuracy': [], 'word_accuracy': [], 'exact_match_accuracy': []},
            'linking': {'spatial_accuracy': [], 'linking_f1': [], 'coordinate_precision': []}
        }
        
        # Evaluate each sample
        for i, (pred, gt, img_name) in enumerate(zip(batch_predictions, batch_ground_truth, image_names)):
            result = self.evaluate_competition_task(task_id, pred, gt, img_name)
            individual_results.append(result)
            
            # Collect metrics for aggregation
            if 'metrics' in result:
                metrics = result['metrics']
                
                # Handle single-metric tasks (task1, task2)
                if isinstance(metrics, dict) and 'precision' in metrics:
                    detection_metrics = metrics
                    aggregated_metrics['detection']['precision'].append(detection_metrics.get('precision', 0.0))
                    aggregated_metrics['detection']['recall'].append(detection_metrics.get('recall', 0.0))
                    aggregated_metrics['detection']['f1_score'].append(detection_metrics.get('f1_score', 0.0))
                    aggregated_metrics['detection']['average_iou'].append(detection_metrics.get('average_iou', 0.0))
                
                # Handle multi-metric tasks (task3, task4)
                else:
                    if 'detection' in metrics:
                        det_metrics = metrics['detection']
                        aggregated_metrics['detection']['precision'].append(det_metrics.get('precision', 0.0))
                        aggregated_metrics['detection']['recall'].append(det_metrics.get('recall', 0.0))
                        aggregated_metrics['detection']['f1_score'].append(det_metrics.get('f1_score', 0.0))
                        aggregated_metrics['detection']['average_iou'].append(det_metrics.get('average_iou', 0.0))
                    
                    if 'recognition' in metrics:
                        rec_metrics = metrics['recognition']
                        aggregated_metrics['recognition']['character_accuracy'].append(rec_metrics.get('character_accuracy', 0.0))
                        aggregated_metrics['recognition']['word_accuracy'].append(rec_metrics.get('word_accuracy', 0.0))
                        aggregated_metrics['recognition']['exact_match_accuracy'].append(rec_metrics.get('exact_match_accuracy', 0.0))
                    
                    if 'linking' in metrics:
                        link_metrics = metrics['linking']
                        aggregated_metrics['linking']['spatial_accuracy'].append(link_metrics.get('spatial_accuracy', 0.0))
                        aggregated_metrics['linking']['linking_f1'].append(link_metrics.get('linking_f1', 0.0))
                        aggregated_metrics['linking']['coordinate_precision'].append(link_metrics.get('coordinate_precision', 0.0))
        
        # Calculate aggregated statistics
        batch_summary = {
            'task_id': task_id,
            'total_samples': len(batch_predictions),
            'individual_results': individual_results,
            'aggregated_metrics': {}
        }
        
        # Aggregate detection metrics
        if aggregated_metrics['detection']['precision']:
            batch_summary['aggregated_metrics']['detection'] = {
                'mean_precision': float(np.mean(aggregated_metrics['detection']['precision'])),
                'mean_recall': float(np.mean(aggregated_metrics['detection']['recall'])),
                'mean_f1_score': float(np.mean(aggregated_metrics['detection']['f1_score'])),
                'mean_iou': float(np.mean(aggregated_metrics['detection']['average_iou'])),
                'std_precision': float(np.std(aggregated_metrics['detection']['precision'])),
                'std_recall': float(np.std(aggregated_metrics['detection']['recall'])),
                'std_f1_score': float(np.std(aggregated_metrics['detection']['f1_score']))
            }
        
        # Aggregate recognition metrics
        if aggregated_metrics['recognition']['character_accuracy']:
            batch_summary['aggregated_metrics']['recognition'] = {
                'mean_character_accuracy': float(np.mean(aggregated_metrics['recognition']['character_accuracy'])),
                'mean_word_accuracy': float(np.mean(aggregated_metrics['recognition']['word_accuracy'])),
                'mean_exact_match_accuracy': float(np.mean(aggregated_metrics['recognition']['exact_match_accuracy'])),
                'std_character_accuracy': float(np.std(aggregated_metrics['recognition']['character_accuracy'])),
                'std_word_accuracy': float(np.std(aggregated_metrics['recognition']['word_accuracy']))
            }
        
        # Aggregate linking metrics
        if aggregated_metrics['linking']['spatial_accuracy']:
            batch_summary['aggregated_metrics']['linking'] = {
                'mean_spatial_accuracy': float(np.mean(aggregated_metrics['linking']['spatial_accuracy'])),
                'mean_linking_f1': float(np.mean(aggregated_metrics['linking']['linking_f1'])),
                'mean_coordinate_precision': float(np.mean(aggregated_metrics['linking']['coordinate_precision'])),
                'std_spatial_accuracy': float(np.std(aggregated_metrics['linking']['spatial_accuracy'])),
                'std_linking_f1': float(np.std(aggregated_metrics['linking']['linking_f1']))
            }
        
        return batch_summary
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save the results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def save_competition_submission(self, 
                                  task_id: str,
                                  formatted_output: str,
                                  output_path: Path,
                                  image_name: str = "") -> None:
        """
        Save formatted competition submission to file.
        
        Args:
            task_id: Competition task identifier
            formatted_output: Formatted output string
            output_path: Path to save the submission file
            image_name: Name of the processed image
        """
        self.formatter.save_competition_output(task_id, formatted_output, output_path, image_name)
    
    def _parse_ground_truth_detections(self, gt_data: List[Dict[str, Any]]) -> List[Detection]:
        """
        Parse ground truth detection data into Detection objects.
        
        Args:
            gt_data: Ground truth detection data
            
        Returns:
            List of Detection objects
        """
        detections = []
        for item in gt_data:
            bbox = BoundingBox(
                x=item['bbox']['x'],
                y=item['bbox']['y'],
                width=item['bbox']['width'],
                height=item['bbox']['height'],
                confidence=item.get('confidence', 1.0)
            )
            
            detection = Detection(
                bbox=bbox,
                detection_type=item.get('detection_type', 'word'),
                confidence=item.get('confidence', 1.0),
                image_coordinates=tuple(item.get('image_coordinates', bbox.coordinates))
            )
            detections.append(detection)
        
        return detections
    
    def _parse_ground_truth_linked_texts(self, gt_data: List[Dict[str, Any]]) -> List[LinkedText]:
        """
        Parse ground truth linked text data into LinkedText objects.
        
        Args:
            gt_data: Ground truth linked text data
            
        Returns:
            List of LinkedText objects
        """
        linked_texts = []
        for item in gt_data:
            # Parse detection
            bbox = BoundingBox(
                x=item['detection']['bbox']['x'],
                y=item['detection']['bbox']['y'],
                width=item['detection']['bbox']['width'],
                height=item['detection']['bbox']['height'],
                confidence=item['detection'].get('confidence', 1.0)
            )
            
            detection = Detection(
                bbox=bbox,
                detection_type=item['detection'].get('detection_type', 'word'),
                confidence=item['detection'].get('confidence', 1.0),
                image_coordinates=tuple(item['detection'].get('image_coordinates', bbox.coordinates))
            )
            
            # Parse recognized text
            recognized_text = RecognizedText(
                detection=detection,
                text=item['text'],
                language=item.get('language', 'en'),
                recognition_confidence=item.get('recognition_confidence', 1.0),
                character_confidences=item.get('character_confidences', [1.0] * len(item['text']))
            )
            
            # Parse linked text
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=tuple(item['map_coordinates']),
                spatial_confidence=item.get('spatial_confidence', 1.0),
                related_features=item.get('related_features', [])
            )
            
            linked_texts.append(linked_text)
        
        return linked_texts