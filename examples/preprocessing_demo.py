"""
Demonstration of the image preprocessing pipeline for historical maps.
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from processing.image_processor import BasicImageProcessor
from processing.historical_map_processor import HistoricalMapProcessor


def create_sample_historical_map():
    """Create a sample historical map-like image for demonstration."""
    # Create base image with aged paper color
    image = np.ones((600, 800, 3), dtype=np.uint8) * 245
    
    # Add some "aged" color variation
    aged_overlay = np.random.randint(230, 255, image.shape, dtype=np.uint8)
    image = cv2.addWeighted(image, 0.7, aged_overlay, 0.3, 0)
    
    # Add faded text regions (simulating place names)
    text_regions = [
        ((50, 50), (250, 80), "City Name"),
        ((100, 150), (300, 180), "River Name"),
        ((400, 200), (600, 230), "Mountain Range"),
        ((200, 350), (450, 380), "Historical Location"),
        ((500, 450), (750, 480), "Ancient Settlement")
    ]
    
    for (x1, y1), (x2, y2), label in text_regions:
        # Create faded text effect with varying intensities
        fade_intensity = np.random.randint(140, 180)
        cv2.rectangle(image, (x1, y1), (x2, y2), 
                     (fade_intensity, fade_intensity, fade_intensity), -1)
        
        # Add some text-like noise within the region
        text_noise = np.random.randint(-10, 10, (y2-y1, x2-x1, 3), dtype=np.int16)
        region = image[y1:y2, x1:x2].astype(np.int16) + text_noise
        image[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
    
    # Add map features (rivers, roads, etc.)
    # River (curved line)
    points = np.array([[100, 300], [200, 280], [300, 290], [400, 270], [500, 280]], np.int32)
    cv2.polylines(image, [points], False, (120, 150, 180), 3)
    
    # Roads (straight lines)
    cv2.line(image, (0, 400), (800, 420), (100, 100, 100), 2)
    cv2.line(image, (300, 0), (320, 600), (100, 100, 100), 2)
    
    # Add noise and artifacts
    noise = np.random.randint(-15, 15, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some small artifacts (ink spots, dust)
    for _ in range(30):
        x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        size = np.random.randint(1, 4)
        cv2.circle(image, (x, y), size, (80, 80, 80), -1)
    
    # Simulate slight rotation
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_angle = np.random.uniform(-3, 3)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                          borderValue=(245, 245, 245))
    
    return image


def demonstrate_basic_preprocessing():
    """Demonstrate basic image preprocessing."""
    print("=== Basic Image Preprocessing Demo ===")
    
    # Create sample image
    sample_image = create_sample_historical_map()
    print(f"Created sample image with shape: {sample_image.shape}")
    
    # Initialize basic processor
    basic_processor = BasicImageProcessor()
    
    # Demonstrate individual operations
    print("\n1. Testing individual operations:")
    
    # Noise reduction
    denoised = basic_processor.reduce_noise(sample_image, method='bilateral')
    print(f"   - Noise reduction: {denoised.shape}")
    
    # Contrast enhancement
    enhanced = basic_processor.enhance_contrast(sample_image, method='clahe')
    print(f"   - Contrast enhancement: {enhanced.shape}")
    
    # Resolution normalization
    normalized = basic_processor.normalize_resolution(sample_image, (512, 512))
    print(f"   - Resolution normalization: {normalized.shape}")
    
    # Complete preprocessing
    print("\n2. Complete preprocessing pipeline:")
    result = basic_processor.preprocess(sample_image)
    print(f"   - Original size: {result.original_size}")
    print(f"   - Processed size: {result.processed_size}")
    print(f"   - Enhancement applied: {result.enhancement_applied}")
    print(f"   - Processing steps: {len(result.preprocessing_steps)}")
    for i, step in enumerate(result.preprocessing_steps, 1):
        print(f"     {i}. {step}")
    
    return sample_image, result


def demonstrate_historical_map_preprocessing():
    """Demonstrate advanced historical map preprocessing."""
    print("\n=== Historical Map Preprocessing Demo ===")
    
    # Create sample historical map
    sample_map = create_sample_historical_map()
    print(f"Created sample historical map with shape: {sample_map.shape}")
    
    # Initialize historical map processor
    hist_processor = HistoricalMapProcessor(
        text_enhancement_strength=2.0,
        artifact_removal_threshold=0.4,
        rotation_detection_sensitivity=0.7
    )
    
    # Demonstrate individual advanced operations
    print("\n1. Testing advanced operations:")
    
    # Faded text enhancement
    enhanced_text = hist_processor.enhance_faded_text(sample_map, method='combined')
    print(f"   - Faded text enhancement: {enhanced_text.shape}")
    
    # Rotation detection and correction
    corrected, angle = hist_processor.detect_and_correct_rotation(sample_map)
    print(f"   - Rotation correction: {corrected.shape}, detected angle: {angle:.2f}°")
    
    # Color space transformation
    color_transformed = hist_processor.transform_color_space_for_text(sample_map, 'lab_enhancement')
    print(f"   - Color space transformation: {color_transformed.shape}")
    
    # Artifact removal
    cleaned = hist_processor.remove_artifacts_and_noise(sample_map, method='combined')
    print(f"   - Artifact removal: {cleaned.shape}")
    
    # Complete preprocessing
    print("\n2. Complete historical map preprocessing pipeline:")
    result = hist_processor.preprocess(sample_map)
    print(f"   - Original size: {result.original_size}")
    print(f"   - Processed size: {result.processed_size}")
    print(f"   - Enhancement applied: {result.enhancement_applied}")
    print(f"   - Processing steps: {len(result.preprocessing_steps)}")
    for i, step in enumerate(result.preprocessing_steps, 1):
        print(f"     {i}. {step}")
    
    # Get preprocessing statistics
    print("\n3. Preprocessing statistics:")
    stats = hist_processor.get_preprocessing_stats(sample_map)
    for stat_name, value in stats.items():
        print(f"   - {stat_name}: {value:.4f}")
    
    return sample_map, result


def save_demo_results(original_basic, processed_basic, original_hist, processed_hist):
    """Save demonstration results to files."""
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    # Save basic preprocessing results
    cv2.imwrite(str(output_dir / "original_basic.png"), original_basic)
    cv2.imwrite(str(output_dir / "processed_basic.png"), processed_basic.image)
    
    # Save historical map preprocessing results
    cv2.imwrite(str(output_dir / "original_historical.png"), original_hist)
    cv2.imwrite(str(output_dir / "processed_historical.png"), processed_hist.image)
    
    print(f"\nDemo results saved to: {output_dir}")
    print("Files created:")
    print("  - original_basic.png")
    print("  - processed_basic.png")
    print("  - original_historical.png")
    print("  - processed_historical.png")


def main():
    """Run the preprocessing demonstration."""
    print("Historical Map Text Processing - Preprocessing Pipeline Demo")
    print("=" * 60)
    
    try:
        # Demonstrate basic preprocessing
        original_basic, processed_basic = demonstrate_basic_preprocessing()
        
        # Demonstrate historical map preprocessing
        original_hist, processed_hist = demonstrate_historical_map_preprocessing()
        
        # Save results
        save_demo_results(original_basic, processed_basic, original_hist, processed_hist)
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("The preprocessing pipeline is ready for historical map text processing.")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()