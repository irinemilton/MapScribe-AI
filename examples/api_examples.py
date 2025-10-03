#!/usr/bin/env python3
"""
Example scripts demonstrating REST API usage for historical map text processing.
"""

import requests
import json
import time
from pathlib import Path
import io
from PIL import Image
import numpy as np


class HistoricalMapAPIClient:
    """Client for interacting with the Historical Map Text Processing API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_system_info(self) -> dict:
        """Get system information."""
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
    
    def process_image(self, image_path: str, task: str, **kwargs) -> dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
            task: Competition task (task1, task2, task3, task4)
            **kwargs: Additional parameters (confidence, format, save_intermediate)
            
        Returns:
            Processing result dictionary
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = kwargs
            
            response = self.session.post(
                f"{self.base_url}/process/{task}",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def process_batch(self, image_paths: list, task: str, **kwargs) -> dict:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            task: Competition task
            **kwargs: Additional parameters
            
        Returns:
            Batch processing result dictionary
        """
        files = []
        for path in image_paths:
            files.append(('files', open(path, 'rb')))
        
        try:
            data = kwargs
            response = self.session.post(
                f"{self.base_url}/batch/{task}",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close all opened files
            for _, file_obj in files:
                file_obj.close()
    
    def get_result(self, result_id: str) -> dict:
        """Get processing result information."""
        response = self.session.get(f"{self.base_url}/results/{result_id}")
        response.raise_for_status()
        return response.json()
    
    def download_result(self, result_id: str, output_path: str = None) -> str:
        """
        Download result file.
        
        Args:
            result_id: Result identifier
            output_path: Optional output file path
            
        Returns:
            Result content as string
        """
        response = self.session.get(f"{self.base_url}/results/{result_id}/download")
        response.raise_for_status()
        
        content = response.text
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content
    
    def get_configuration(self) -> dict:
        """Get current system configuration."""
        response = self.session.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()
    
    def update_configuration(self, config_updates: dict) -> dict:
        """
        Update system configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            Update confirmation
        """
        response = self.session.post(
            f"{self.base_url}/config",
            json=config_updates
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> dict:
        """List available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def switch_model(self, model_type: str, version: str) -> dict:
        """
        Switch model version.
        
        Args:
            model_type: Type of model (detection, recognition)
            version: Model version
            
        Returns:
            Switch confirmation
        """
        response = self.session.post(
            f"{self.base_url}/models/{model_type}/switch",
            json={'version': version}
        )
        response.raise_for_status()
        return response.json()
    
    def evaluate_results(self, task: str, results: list, ground_truth: list) -> dict:
        """
        Evaluate results against ground truth.
        
        Args:
            task: Competition task
            results: List of result entries
            ground_truth: List of ground truth entries
            
        Returns:
            Evaluation metrics
        """
        response = self.session.post(
            f"{self.base_url}/evaluate/{task}",
            json={
                'results': results,
                'ground_truth': ground_truth
            }
        )
        response.raise_for_status()
        return response.json()


def create_sample_image(filename: str, width: int = 800, height: int = 600) -> str:
    """
    Create a sample image for testing.
    
    Args:
        filename: Output filename
        width: Image width
        height: Image height
        
    Returns:
        Path to created image
    """
    # Create a simple image with some text-like patterns
    image = Image.new('RGB', (width, height), color='white')
    
    # Add some noise to simulate a historical map
    pixels = np.array(image)
    noise = np.random.randint(0, 50, (height, width, 3))
    pixels = np.clip(pixels.astype(int) - noise, 0, 255).astype(np.uint8)
    
    # Add some rectangular regions to simulate text
    for i in range(5):
        x = np.random.randint(50, width - 150)
        y = np.random.randint(50, height - 50)
        w = np.random.randint(80, 150)
        h = np.random.randint(20, 40)
        
        # Dark rectangle for text
        pixels[y:y+h, x:x+w] = [50, 50, 50]
    
    image = Image.fromarray(pixels)
    image.save(filename)
    return filename


def example_basic_usage():
    """Example: Basic API usage."""
    print("=== Basic API Usage Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    try:
        # 1. Check API health
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        
        # 2. Get system information
        print("\n2. Getting system information...")
        info = client.get_system_info()
        print(f"   API Version: {info['api']['version']}")
        print(f"   Max file size: {info['api']['max_file_size']} bytes")
        print(f"   Supported tasks: {list(info['tasks'].keys())}")
        
        # 3. Create sample image
        print("\n3. Creating sample image...")
        sample_image = create_sample_image('sample_map.jpg')
        print(f"   Created: {sample_image}")
        
        # 4. Process single image
        print("\n4. Processing single image...")
        result = client.process_image(
            image_path=sample_image,
            task='task1',
            confidence=0.7,
            format='json'
        )
        print(f"   Result ID: {result['result_id']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Detections found: {result['results']['statistics']['detection_count']}")
        
        # 5. Download result
        print("\n5. Downloading result...")
        result_content = client.download_result(result['result_id'], 'result.txt')
        print(f"   Result saved to: result.txt")
        print(f"   Content preview: {result_content[:100]}...")
        
        print("\n✓ Basic usage example completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API server. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_batch_processing():
    """Example: Batch processing multiple images."""
    print("=== Batch Processing Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    try:
        # Create multiple sample images
        print("1. Creating sample images...")
        image_paths = []
        for i in range(3):
            filename = f'sample_map_{i+1}.jpg'
            create_sample_image(filename, width=600 + i*100, height=400 + i*50)
            image_paths.append(filename)
            print(f"   Created: {filename}")
        
        # Process batch
        print("\n2. Processing batch...")
        batch_result = client.process_batch(
            image_paths=image_paths,
            task='task3',
            confidence=0.6
        )
        
        print(f"   Batch ID: {batch_result['batch_id']}")
        print(f"   Total files: {batch_result['summary']['total_files']}")
        print(f"   Successful: {batch_result['summary']['successful']}")
        print(f"   Failed: {batch_result['summary']['failed']}")
        print(f"   Total processing time: {batch_result['summary']['total_processing_time']:.2f}s")
        
        # Download individual results
        print("\n3. Downloading individual results...")
        for i, result in enumerate(batch_result['results']):
            if result['success']:
                file_id = result['file_id']
                output_file = f'batch_result_{i+1}.txt'
                client.download_result(file_id, output_file)
                print(f"   Downloaded: {output_file}")
        
        print("\n✓ Batch processing example completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_configuration_management():
    """Example: Configuration management."""
    print("=== Configuration Management Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    try:
        # Get current configuration
        print("1. Getting current configuration...")
        config = client.get_configuration()
        print("   Current detection confidence threshold:", 
              config['configuration'].get('detection', {}).get('confidence_threshold', 'Not set'))
        
        # Update configuration
        print("\n2. Updating configuration...")
        updates = {
            'detection.confidence_threshold': 0.8,
            'recognition.languages': ['en', 'zh']
        }
        
        update_result = client.update_configuration(updates)
        print(f"   Updated keys: {update_result['updated_keys']}")
        
        # Verify update
        print("\n3. Verifying configuration update...")
        updated_config = client.get_configuration()
        new_threshold = updated_config['configuration'].get('detection', {}).get('confidence_threshold')
        print(f"   New detection confidence threshold: {new_threshold}")
        
        print("\n✓ Configuration management example completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_model_management():
    """Example: Model management."""
    print("=== Model Management Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    try:
        # List available models
        print("1. Listing available models...")
        models = client.list_models()
        
        print("   Detection models:")
        for model in models['models']['detection_models']:
            print(f"     - {model['name']} (v{model['version']}): {model['description']}")
        
        print("   Recognition models:")
        for model in models['models']['recognition_models']:
            print(f"     - {model['name']} (v{model['version']}): {model['description']}")
        
        # Switch model version (mock operation)
        print("\n2. Switching model version...")
        switch_result = client.switch_model('detection', '2.0')
        print(f"   {switch_result['message']}")
        
        print("\n✓ Model management example completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_evaluation():
    """Example: Result evaluation."""
    print("=== Evaluation Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    try:
        # Create sample results and ground truth data
        print("1. Preparing evaluation data...")
        
        results = [
            {'bbox': [100, 50, 300, 80], 'confidence': 0.85, 'text': 'Sample Text'},
            {'bbox': [150, 120, 250, 150], 'confidence': 0.92, 'text': 'Another Text'},
            {'bbox': [200, 200, 350, 230], 'confidence': 0.78, 'text': 'Third Text'}
        ]
        
        ground_truth = [
            {'bbox': [105, 52, 295, 78], 'text': 'Sample Text'},
            {'bbox': [148, 118, 252, 152], 'text': 'Another Text'},
            {'bbox': [195, 205, 355, 235], 'text': 'Third Text'}
        ]
        
        print(f"   Results: {len(results)} entries")
        print(f"   Ground truth: {len(ground_truth)} entries")
        
        # Evaluate results
        print("\n2. Running evaluation...")
        evaluation = client.evaluate_results('task3', results, ground_truth)
        
        print("   Evaluation metrics:")
        for metric, value in evaluation['evaluation']['metrics'].items():
            print(f"     {metric}: {value:.4f}")
        
        print(f"   Total predictions: {evaluation['evaluation']['summary']['total_predictions']}")
        print(f"   Total ground truth: {evaluation['evaluation']['summary']['total_ground_truth']}")
        
        print("\n✓ Evaluation example completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_error_handling():
    """Example: Error handling."""
    print("=== Error Handling Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    # Test various error conditions
    print("1. Testing invalid task...")
    try:
        create_sample_image('test_image.jpg')
        client.process_image('test_image.jpg', 'invalid_task')
    except requests.exceptions.HTTPError as e:
        print(f"   Expected error: {e.response.status_code} - {e.response.json()['error']}")
    
    print("\n2. Testing non-existent result...")
    try:
        client.get_result('non-existent-id')
    except requests.exceptions.HTTPError as e:
        print(f"   Expected error: {e.response.status_code} - {e.response.json()['error']}")
    
    print("\n3. Testing invalid configuration...")
    try:
        client.update_configuration({'invalid.key': 'value'})
    except requests.exceptions.HTTPError as e:
        print(f"   Expected error: {e.response.status_code} - {e.response.json()['error']}")
    
    print("\n✓ Error handling example completed!")


def example_performance_testing():
    """Example: Performance testing."""
    print("=== Performance Testing Example ===\n")
    
    client = HistoricalMapAPIClient()
    
    try:
        # Create test image
        print("1. Creating test image...")
        test_image = create_sample_image('performance_test.jpg', 1200, 800)
        
        # Test processing times for different tasks
        tasks = ['task1', 'task2', 'task3', 'task4']
        
        print("\n2. Testing processing times...")
        for task in tasks:
            start_time = time.time()
            
            result = client.process_image(
                image_path=test_image,
                task=task,
                confidence=0.5
            )
            
            end_time = time.time()
            api_time = end_time - start_time
            processing_time = result['processing_time']
            
            print(f"   {task}: API call {api_time:.2f}s, Processing {processing_time:.2f}s")
        
        print("\n✓ Performance testing example completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def cleanup_sample_files():
    """Clean up sample files created during examples."""
    print("\n=== Cleaning up sample files ===")
    
    patterns = [
        'sample_map*.jpg',
        'performance_test.jpg',
        'test_image.jpg',
        'result*.txt',
        'batch_result*.txt'
    ]
    
    import glob
    
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                Path(file_path).unlink()
                print(f"   Removed: {file_path}")
            except Exception as e:
                print(f"   Failed to remove {file_path}: {e}")


def main():
    """Run all API examples."""
    print("Historical Map Text Processing API Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        print("\n" + "="*50 + "\n")
        
        example_batch_processing()
        print("\n" + "="*50 + "\n")
        
        example_configuration_management()
        print("\n" + "="*50 + "\n")
        
        example_model_management()
        print("\n" + "="*50 + "\n")
        
        example_evaluation()
        print("\n" + "="*50 + "\n")
        
        example_error_handling()
        print("\n" + "="*50 + "\n")
        
        example_performance_testing()
        
        print("\n" + "="*50)
        print("All examples completed!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        cleanup_sample_files()


if __name__ == "__main__":
    main()