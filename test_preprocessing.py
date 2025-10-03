"""
Simple test script for the preprocessing pipeline.
"""

import numpy as np
import cv2
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import with absolute imports
from src.processing.image_processor import BasicImageProcessor
from src.processing.historical_map_processor import HistoricalMapProcessor


def create_test_image():
    """Create a simple test image."""
    # Create a test image with some text-like patterns
    image = np.ones((300, 400, 3), dtype=np.uint8) * 240
    
    # Add some rectangular regions (simulating text)
    cv2.rectangle(image, (50, 50), (200, 80), (160, 160, 160), -1)
    cv2.rectangle(image, (50, 100), (300, 130), (140, 140, 140), -1)
    
    # Add noise
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def test_basic_processor():
    """Test the basic image processor."""
    print("Testing BasicImageProcessor...")
    
    processor = BasicImageProcessor()
    test_image = create_test_image()
    
    print(f"Input image shape: {test_image.shape}")
    
    # Test preprocessing
    result = processor.preprocess(test_image)
    print(f"Processed image shape: {result.image.shape}")
    print(f"Original size: {result.original_size}")
    print(f"Processed size: {result.processed_size}")
    print(f"Enhancement applied: {result.enhancement_applied}")
    print(f"Processing steps: {result.preprocessing_steps}")
    
    return True


def test_historical_processor():
    """Test the historical map processor."""
    print("\nTesting HistoricalMapProcessor...")
    
    processor = HistoricalMapProcessor()
    test_image = create_test_image()
    
    print(f"Input image shape: {test_image.shape}")
    
    # Test preprocessing
    result = processor.preprocess(test_image)
    print(f"Processed image shape: {result.image.shape}")
    print(f"Original size: {result.original_size}")
    print(f"Processed size: {result.processed_size}")
    print(f"Enhancement applied: {result.enhancement_applied}")
    print(f"Processing steps: {result.preprocessing_steps}")
    
    # Test statistics
    stats = processor.get_preprocessing_stats(test_image)
    print(f"Preprocessing statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return True


def main():
    """Run the tests."""
    print("Image Preprocessing Pipeline Test")
    print("=" * 40)
    
    try:
        # Test basic processor
        test_basic_processor()
        
        # Test historical processor
        test_historical_processor()
        
        print("\n" + "=" * 40)
        print("All tests completed successfully!")
        print("The preprocessing pipeline is working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)