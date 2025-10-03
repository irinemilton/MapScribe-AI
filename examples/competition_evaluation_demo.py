"""
Demonstration of the ICDAR 2025 competition output formatting and evaluation system.
Shows how to format outputs and calculate metrics for all four competition tasks.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import ICDAR2025Formatter, CompetitionMetricsCalculator, ICDAR2025Evaluator
from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText, ProcessingResult
)


def create_sample_data():
    """Create sample processing results for demonstration."""
    
    # Create sample bounding boxes
    bbox1 = BoundingBox(x=50, y=100, width=120, height=35, confidence=0.92)
    bbox2 = BoundingBox(x=200, y=150, width=90, height=28, confidence=0.87)
    bbox3 = BoundingBox(x=45, y=95, width=270, height=40, confidence=0.89)
    
    # Create detections
    word_detection1 = Detection(
        bbox=bbox1,
        detection_type='word',
        confidence=0.92,
        image_coordinates=(50, 100, 170, 135)
    )
    
    word_detection2 = Detection(
        bbox=bbox2,
        detection_type='word',
        confidence=0.87,
        image_coordinates=(200, 150, 290, 178)
    )
    
    phrase_detection = Detection(
        bbox=bbox3,
        detection_type='phrase',
        confidence=0.89,
        image_coordinates=(45, 95, 315, 135)
    )
    
    # Create recognized texts
    recognized_word1 = RecognizedText(
        detection=word_detection1,
        text="Thames",
        language="en",
        recognition_confidence=0.94,
        character_confidences=[0.96, 0.93, 0.95, 0.92, 0.94, 0.91]
    )
    
    recognized_word2 = RecognizedText(
        detection=word_detection2,
        text="River",
        language="en",
        recognition_confidence=0.89,
        character_confidences=[0.91, 0.88, 0.90, 0.87, 0.92]
    )
    
    recognized_phrase = RecognizedText(
        detection=phrase_detection,
        text="Thames River",
        language="en",
        recognition_confidence=0.91,
        character_confidences=[0.96, 0.93, 0.95, 0.92, 0.94, 0.91, 0.88, 0.91, 0.88, 0.90, 0.87, 0.92]
    )
    
    # Create linked text
    linked_text = LinkedText(
        recognized_text=recognized_phrase,
        map_coordinates=(162.5, 115.0),
        spatial_confidence=0.88,
        related_features=["river", "waterway", "landmark"]
    )
    
    # Create processing result
    processing_result = ProcessingResult(
        task_type='phrase_detection_recognition',
        detections=[word_detection1, word_detection2, phrase_detection],
        recognized_texts=[recognized_word1, recognized_word2, recognized_phrase],
        linked_texts=[linked_text],
        processing_time=2.3,
        metrics={'detection_confidence': 0.89, 'recognition_confidence': 0.91}
    )
    
    return processing_result


def demonstrate_formatting():
    """Demonstrate competition output formatting for all tasks."""
    print("=== ICDAR 2025 Competition Output Formatting Demo ===\n")
    
    # Create sample data
    result = create_sample_data()
    formatter = ICDAR2025Formatter()
    
    print("Sample Processing Result:")
    print(f"- Task Type: {result.task_type}")
    print(f"- Detections: {len(result.detections)}")
    print(f"- Recognized Texts: {len(result.recognized_texts)}")
    print(f"- Linked Texts: {len(result.linked_texts)}")
    print(f"- Processing Time: {result.processing_time}s\n")
    
    # Demonstrate Task 1: Word Detection
    print("Task 1 - Word Detection Output:")
    task1_output = formatter.format_processing_result(result, 'task1', 'historical_map_001.jpg')
    print(task1_output)
    print(f"Format Valid: {formatter.validate_competition_format('task1', task1_output)}\n")
    
    # Demonstrate Task 2: Phrase Detection
    print("Task 2 - Phrase Detection Output:")
    task2_output = formatter.format_processing_result(result, 'task2', 'historical_map_001.jpg')
    print(task2_output)
    print(f"Format Valid: {formatter.validate_competition_format('task2', task2_output)}\n")
    
    # Demonstrate Task 3: Word Detection + Recognition
    print("Task 3 - Word Detection + Recognition Output:")
    task3_output = formatter.format_processing_result(result, 'task3', 'historical_map_001.jpg')
    print(task3_output)
    print(f"Format Valid: {formatter.validate_competition_format('task3', task3_output)}\n")
    
    # Demonstrate Task 4: Phrase Detection + Recognition
    print("Task 4 - Phrase Detection + Recognition Output:")
    task4_output = formatter.format_processing_result(result, 'task4', 'historical_map_001.jpg')
    print(task4_output)
    print(f"Format Valid: {formatter.validate_competition_format('task4', task4_output)}\n")


def demonstrate_metrics():
    """Demonstrate evaluation metrics calculation."""
    print("=== Evaluation Metrics Calculation Demo ===\n")
    
    # Create sample data
    result = create_sample_data()
    calculator = CompetitionMetricsCalculator()
    
    # Create ground truth data
    gt_detections = result.detections  # Perfect match for demo
    gt_texts = ["Thames", "River", "Thames River"]
    gt_linked_texts = result.linked_texts  # Perfect match for demo
    
    # Calculate detection metrics
    print("Detection Metrics:")
    detection_metrics = calculator.calculate_detection_metrics(result.detections, gt_detections)
    for metric, value in detection_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Calculate recognition metrics
    print("Recognition Metrics:")
    recognition_metrics = calculator.calculate_recognition_metrics(result.recognized_texts, gt_texts)
    for metric, value in recognition_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Calculate linking metrics
    print("Linking Metrics:")
    linking_metrics = calculator.calculate_linking_metrics(result.linked_texts, gt_linked_texts)
    for metric, value in linking_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Generate comprehensive report
    print("Comprehensive Evaluation Report:")
    report = calculator.generate_evaluation_report(
        detection_metrics=detection_metrics,
        recognition_metrics=recognition_metrics,
        linking_metrics=linking_metrics
    )
    
    print("Summary Metrics:")
    for metric, value in report['summary'].items():
        print(f"  {metric}: {value:.4f}")
    print()


def demonstrate_comprehensive_evaluation():
    """Demonstrate comprehensive competition evaluation."""
    print("=== Comprehensive Competition Evaluation Demo ===\n")
    
    # Create sample data
    result = create_sample_data()
    evaluator = ICDAR2025Evaluator()
    
    # Create ground truth in expected format
    ground_truth = {
        'detections': [
            {
                'bbox': {'x': 50, 'y': 100, 'width': 120, 'height': 35},
                'detection_type': 'word',
                'confidence': 0.92,
                'image_coordinates': [50, 100, 170, 135]
            },
            {
                'bbox': {'x': 200, 'y': 150, 'width': 90, 'height': 28},
                'detection_type': 'word',
                'confidence': 0.87,
                'image_coordinates': [200, 150, 290, 178]
            }
        ],
        'texts': ["Thames", "River", "Thames River"]
    }
    
    # Evaluate Task 3 (Word Detection + Recognition)
    print("Task 3 Comprehensive Evaluation:")
    evaluation_result = evaluator.evaluate_competition_task(
        'task3', result, ground_truth, 'historical_map_001.jpg'
    )
    
    print(f"Task ID: {evaluation_result['task_id']}")
    print(f"Image: {evaluation_result['image_name']}")
    print(f"Format Valid: {evaluation_result['validation']['format_valid']}")
    print()
    
    print("Formatted Output:")
    print(evaluation_result['formatted_output'])
    print()
    
    if 'evaluation_report' in evaluation_result:
        print("Evaluation Summary:")
        summary = evaluation_result['evaluation_report']['summary']
        for metric, value in summary.items():
            print(f"  {metric}: {value:.4f}")
    print()


def main():
    """Run all demonstrations."""
    try:
        demonstrate_formatting()
        demonstrate_metrics()
        demonstrate_comprehensive_evaluation()
        
        print("=== Demo Complete ===")
        print("The competition output formatting system successfully:")
        print("✓ Formats outputs for all four ICDAR 2025 tasks")
        print("✓ Validates format compliance")
        print("✓ Calculates comprehensive evaluation metrics")
        print("✓ Provides integrated evaluation pipeline")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()