"""
Sample data generator for integration testing.
Creates realistic test data including historical map images, ground truth annotations,
and various test scenarios for comprehensive integration testing.

Requirements addressed:
- 6.4: Processing time and throughput metrics (realistic data for benchmarking)
- 6.5: Detailed comparison reports (ground truth data for evaluation)
- 7.1: Enhancement techniques validation (degraded images)
- 7.2: Normalization consistency (various image formats and sizes)
"""

import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import asdict
import tempfile
import random

from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText,
    DetectionConfig, RecognitionConfig
)
from tests.test_utilities import MockImageGenerator, TestDataFactory


class IntegrationSampleDataGenerator:
    """Generates comprehensive sample data for integration testing."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the sample data generator.
        
        Args:
            config_path: Path to integration test configuration file
        """
        self.config_path = config_path or Path("tests/integration_test_config.yaml")
        self.config = self._load_config()
        
        # Initialize random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load integration test configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file doesn't exist."""
        return {
            'test_data_generation': {
                'historical_maps': {
                    'background_color_range': [200, 240],
                    'text_density': 0.1,
                    'degradation_levels': [0.0, 0.2, 0.4, 0.6]
                },
                'text_generation': {
                    'languages': ['en', 'zh'],
                    'word_lengths': [3, 12],
                    'phrase_lengths': [2, 6],
                    'confidence_ranges': {
                        'high': [0.8, 1.0],
                        'medium': [0.5, 0.8],
                        'low': [0.2, 0.5]
                    }
                },
                'bbox_generation': {
                    'min_width': 20,
                    'max_width': 200,
                    'min_height': 15,
                    'max_height': 50,
                    'overlap_probability': 0.1
                }
            },
            'end_to_end_testing': {
                'sample_images': {
                    'count': 4,
                    'types': ['standard_map', 'high_resolution_map', 'degraded_map', 'multilingual_map']
                }
            }
        }
    
    def generate_complete_test_dataset(self, output_dir: Path) -> Dict[str, Any]:
        """
        Generate a complete test dataset for integration testing.
        
        Args:
            output_dir: Directory to save generated test data
            
        Returns:
            Dictionary containing paths and metadata for generated data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'generation_timestamp': np.datetime64('now').isoformat(),
            'config': self.config,
            'images': {},
            'ground_truth': {},
            'test_scenarios': {}
        }
        
        # Generate sample images
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        sample_config = self.config['end_to_end_testing']['sample_images']
        
        for i, image_type in enumerate(sample_config['types']):
            image_data = self._generate_sample_image(image_type, i)
            
            # Save image data (in real implementation would save actual image files)
            image_filename = f"{image_type}_{i:03d}.jpg"
            image_path = images_dir / image_filename
            
            # Store image metadata
            dataset_info['images'][image_filename] = {
                'path': str(image_path),
                'type': image_type,
                'dimensions': image_data['image'].shape,
                'degradation_level': image_data.get('degradation_level', 0.0),
                'text_regions': len(image_data.get('text_regions', []))
            }
            
            # Generate ground truth annotations
            ground_truth = self._generate_ground_truth_annotations(
                image_data, image_filename
            )
            
            # Save ground truth
            gt_filename = f"{image_type}_{i:03d}_gt.json"
            gt_path = output_dir / "ground_truth" / gt_filename
            gt_path.parent.mkdir(exist_ok=True)
            
            with open(gt_path, 'w') as f:
                json.dump(ground_truth, f, indent=2, default=str)
            
            dataset_info['ground_truth'][image_filename] = {
                'path': str(gt_path),
                'detections_count': len(ground_truth.get('detections', [])),
                'texts_count': len(ground_truth.get('texts', [])),
                'languages': list(set(gt['language'] for gt in ground_truth.get('texts', []) if 'language' in gt))
            }
        
        # Generate test scenarios
        scenarios_dir = output_dir / "scenarios"
        scenarios_dir.mkdir(exist_ok=True)
        
        test_scenarios = self._generate_test_scenarios()
        for scenario_name, scenario_data in test_scenarios.items():
            scenario_path = scenarios_dir / f"{scenario_name}.json"
            with open(scenario_path, 'w') as f:
                json.dump(scenario_data, f, indent=2, default=str)
            
            dataset_info['test_scenarios'][scenario_name] = {
                'path': str(scenario_path),
                'description': scenario_data.get('description', ''),
                'test_count': len(scenario_data.get('test_cases', []))
            }
        
        # Save dataset metadata
        metadata_path = output_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        return dataset_info
    
    def _generate_sample_image(self, image_type: str, index: int) -> Dict[str, Any]:
        """Generate a sample image based on type."""
        # Get image dimensions based on type
        if image_type == 'standard_map':
            width, height = 1024, 768
            degradation = 0.0
        elif image_type == 'high_resolution_map':
            width, height = 2048, 1536
            degradation = 0.0
        elif image_type == 'degraded_map':
            width, height = 800, 600
            degradation = 0.4
        elif image_type == 'multilingual_map':
            width, height = 1200, 900
            degradation = 0.2
        else:
            width, height = 1024, 768
            degradation = 0.0
        
        # Generate base historical map image
        image = MockImageGenerator.create_historical_map_image(width, height)
        
        # Apply degradation if specified
        if degradation > 0:
            image = MockImageGenerator.create_degraded_image(image, degradation)
        
        # Generate text regions for this image
        text_regions = self._generate_text_regions_for_image(
            width, height, image_type
        )
        
        return {
            'image': image,
            'degradation_level': degradation,
            'text_regions': text_regions,
            'image_type': image_type
        }
    
    def _generate_text_regions_for_image(self, width: int, height: int, 
                                       image_type: str) -> List[Dict[str, Any]]:
        """Generate text regions for an image."""
        text_config = self.config['test_data_generation']['text_generation']
        bbox_config = self.config['test_data_generation']['bbox_generation']
        
        # Determine number of text regions based on image type
        if image_type == 'multilingual_map':
            num_regions = random.randint(8, 15)
        elif image_type == 'high_resolution_map':
            num_regions = random.randint(12, 20)
        elif image_type == 'degraded_map':
            num_regions = random.randint(3, 8)
        else:
            num_regions = random.randint(5, 12)
        
        text_regions = []
        
        for i in range(num_regions):
            # Generate bounding box
            bbox_width = random.randint(bbox_config['min_width'], bbox_config['max_width'])
            bbox_height = random.randint(bbox_config['min_height'], bbox_config['max_height'])
            
            # Ensure bbox fits within image
            max_x = width - bbox_width
            max_y = height - bbox_height
            
            if max_x <= 0 or max_y <= 0:
                continue
            
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Generate text content
            if image_type == 'multilingual_map' and i % 3 == 0:
                # Include Chinese text for multilingual maps
                language = 'zh'
                text = self._generate_chinese_text()
            else:
                language = 'en'
                text = self._generate_english_text()
            
            # Determine if this is a word or phrase
            is_phrase = len(text.split()) > 1
            detection_type = 'phrase' if is_phrase else 'word'
            
            # Generate confidence scores
            confidence_level = random.choice(['high', 'medium', 'low'])
            conf_range = text_config['confidence_ranges'][confidence_level]
            detection_confidence = random.uniform(conf_range[0], conf_range[1])
            recognition_confidence = random.uniform(conf_range[0], conf_range[1])
            
            # Generate character-level confidences
            char_confidences = [
                random.uniform(conf_range[0], conf_range[1]) 
                for _ in text
            ]
            
            text_region = {
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': bbox_width,
                    'height': bbox_height
                },
                'text': text,
                'language': language,
                'detection_type': detection_type,
                'detection_confidence': detection_confidence,
                'recognition_confidence': recognition_confidence,
                'character_confidences': char_confidences,
                'map_coordinates': [
                    x + bbox_width / 2,  # Center X
                    y + bbox_height / 2  # Center Y
                ]
            }
            
            text_regions.append(text_region)
        
        return text_regions
    
    def _generate_english_text(self) -> str:
        """Generate realistic English text for historical maps."""
        # Common words found on historical maps
        place_names = [
            'London', 'Paris', 'Berlin', 'Rome', 'Madrid', 'Vienna',
            'Prague', 'Warsaw', 'Budapest', 'Amsterdam', 'Brussels',
            'Stockholm', 'Copenhagen', 'Oslo', 'Helsinki'
        ]
        
        geographic_terms = [
            'River', 'Lake', 'Mountain', 'Valley', 'Forest', 'Plain',
            'Hill', 'Bay', 'Cape', 'Island', 'Peninsula', 'Strait'
        ]
        
        directions = ['North', 'South', 'East', 'West', 'Northeast', 'Northwest']
        
        # Generate text based on random selection
        text_type = random.choice(['place_name', 'geographic', 'direction', 'compound'])
        
        if text_type == 'place_name':
            return random.choice(place_names)
        elif text_type == 'geographic':
            return random.choice(geographic_terms)
        elif text_type == 'direction':
            return random.choice(directions)
        else:
            # Compound phrase
            parts = [
                random.choice(directions),
                random.choice(place_names)
            ]
            return ' '.join(parts)
    
    def _generate_chinese_text(self) -> str:
        """Generate realistic Chinese text for historical maps."""
        # Common Chinese characters and words for geographic features
        chinese_places = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都']
        chinese_geographic = ['河', '湖', '山', '岛', '港', '湾', '海', '江']
        chinese_directions = ['东', '南', '西', '北', '东南', '西北']
        
        text_type = random.choice(['place', 'geographic', 'direction', 'compound'])
        
        if text_type == 'place':
            return random.choice(chinese_places)
        elif text_type == 'geographic':
            return random.choice(chinese_geographic)
        elif text_type == 'direction':
            return random.choice(chinese_directions)
        else:
            # Compound
            return random.choice(chinese_directions) + random.choice(chinese_places)
    
    def _generate_ground_truth_annotations(self, image_data: Dict[str, Any], 
                                         image_filename: str) -> Dict[str, Any]:
        """Generate ground truth annotations for an image."""
        text_regions = image_data.get('text_regions', [])
        
        # Convert text regions to ground truth format
        detections = []
        texts = []
        linked_texts = []
        
        for region in text_regions:
            # Detection ground truth
            detection_gt = {
                'bbox': region['bbox'],
                'detection_type': region['detection_type'],
                'confidence': region['detection_confidence'],
                'image_coordinates': [
                    region['bbox']['x'],
                    region['bbox']['y'],
                    region['bbox']['x'] + region['bbox']['width'],
                    region['bbox']['y'] + region['bbox']['height']
                ]
            }
            detections.append(detection_gt)
            
            # Recognition ground truth
            text_gt = {
                'text': region['text'],
                'language': region['language'],
                'recognition_confidence': region['recognition_confidence'],
                'character_confidences': region['character_confidences']
            }
            texts.append(text_gt)
            
            # Linking ground truth
            linked_text_gt = {
                'detection': detection_gt,
                'text': region['text'],
                'language': region['language'],
                'recognition_confidence': region['recognition_confidence'],
                'character_confidences': region['character_confidences'],
                'map_coordinates': region['map_coordinates'],
                'spatial_confidence': random.uniform(0.7, 1.0),
                'related_features': []
            }
            linked_texts.append(linked_text_gt)
        
        return {
            'image_filename': image_filename,
            'image_dimensions': image_data['image'].shape,
            'detections': detections,
            'texts': [text['text'] for text in texts],
            'text_details': texts,
            'linked_texts': linked_texts,
            'metadata': {
                'image_type': image_data['image_type'],
                'degradation_level': image_data.get('degradation_level', 0.0),
                'total_text_regions': len(text_regions),
                'languages': list(set(region['language'] for region in text_regions))
            }
        }
    
    def _generate_test_scenarios(self) -> Dict[str, Any]:
        """Generate various test scenarios for comprehensive testing."""
        scenarios = {}
        
        # Performance testing scenarios
        scenarios['performance_benchmarks'] = {
            'description': 'Performance benchmarking test cases',
            'test_cases': [
                {
                    'name': 'small_image_processing',
                    'image_size': [640, 480],
                    'expected_max_time': 5.0,
                    'text_regions': 3
                },
                {
                    'name': 'large_image_processing',
                    'image_size': [2048, 1536],
                    'expected_max_time': 15.0,
                    'text_regions': 15
                },
                {
                    'name': 'batch_processing',
                    'batch_size': 10,
                    'image_size': [1024, 768],
                    'expected_throughput': 0.5
                }
            ]
        }
        
        # Regression testing scenarios
        scenarios['regression_tests'] = {
            'description': 'Regression testing against baseline metrics',
            'test_cases': [
                {
                    'name': 'task1_baseline_comparison',
                    'task_type': 'task1',
                    'baseline_metrics': {
                        'precision': 0.85,
                        'recall': 0.82,
                        'f1_score': 0.835
                    },
                    'tolerance': 0.05
                },
                {
                    'name': 'multilingual_accuracy_regression',
                    'task_type': 'task3',
                    'languages': ['en', 'zh'],
                    'baseline_accuracy': 0.88,
                    'tolerance': 0.03
                }
            ]
        }
        
        # Format compliance scenarios
        scenarios['format_compliance'] = {
            'description': 'Competition format compliance test cases',
            'test_cases': [
                {
                    'name': 'task1_format_validation',
                    'task_type': 'task1',
                    'required_fields': ['x1', 'y1', 'x2', 'y2'],
                    'format_rules': ['coordinates_numeric', 'coordinates_non_negative']
                },
                {
                    'name': 'multilingual_text_encoding',
                    'task_type': 'task3',
                    'text_languages': ['en', 'zh'],
                    'encoding_requirements': ['utf-8', 'proper_escaping']
                },
                {
                    'name': 'edge_case_handling',
                    'cases': ['empty_results', 'single_detection', 'overlapping_boxes']
                }
            ]
        }
        
        # End-to-end pipeline scenarios
        scenarios['end_to_end_pipeline'] = {
            'description': 'Complete pipeline integration test cases',
            'test_cases': [
                {
                    'name': 'complete_task1_pipeline',
                    'task_type': 'task1',
                    'steps': ['preprocessing', 'detection', 'formatting', 'validation'],
                    'quality_checks': ['bbox_validity', 'confidence_ranges', 'format_compliance']
                },
                {
                    'name': 'complete_task4_pipeline',
                    'task_type': 'task4',
                    'steps': ['preprocessing', 'detection', 'recognition', 'linking', 'formatting'],
                    'quality_checks': ['detection_quality', 'recognition_accuracy', 'spatial_linking', 'format_compliance']
                }
            ]
        }
        
        return scenarios
    
    def generate_baseline_results(self, output_path: Path) -> Dict[str, Any]:
        """Generate baseline results for regression testing."""
        baseline_config = self.config.get('regression_testing', {}).get('baseline_metrics', {})
        
        baseline_results = {
            'generation_timestamp': np.datetime64('now').isoformat(),
            'baseline_metrics': baseline_config,
            'test_conditions': {
                'image_sizes': ['1024x768', '2048x1536'],
                'degradation_levels': [0.0, 0.2, 0.4],
                'languages': ['en', 'zh'],
                'test_iterations': 5
            }
        }
        
        # Save baseline results
        with open(output_path, 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        return baseline_results


def main():
    """Main entry point for sample data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample data for integration testing')
    parser.add_argument('--output-dir', type=Path, default=Path('tests/sample_data'),
                       help='Output directory for generated data')
    parser.add_argument('--config', type=Path, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = IntegrationSampleDataGenerator(config_path=args.config)
    
    # Generate complete test dataset
    print(f"Generating test dataset in {args.output_dir}...")
    dataset_info = generator.generate_complete_test_dataset(args.output_dir)
    
    print(f"Generated {len(dataset_info['images'])} test images")
    print(f"Generated {len(dataset_info['ground_truth'])} ground truth files")
    print(f"Generated {len(dataset_info['test_scenarios'])} test scenarios")
    
    # Generate baseline results
    baseline_path = args.output_dir / "baseline_results.json"
    baseline_results = generator.generate_baseline_results(baseline_path)
    print(f"Generated baseline results: {baseline_path}")
    
    print("Sample data generation complete!")


if __name__ == '__main__':
    main()