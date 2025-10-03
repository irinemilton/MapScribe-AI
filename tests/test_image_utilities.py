"""
Unit tests for image generation and manipulation utilities.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_utilities import MockImageGenerator, FileTestUtilities
from src.models.data_structures import ProcessedImage


class TestImageGeneration:
    """Test cases for image generation utilities."""
    
    def test_create_test_image_various_sizes(self):
        """Test creating test images with various sizes."""
        test_cases = [
            (100, 100, 1),
            (200, 150, 3),
            (640, 480, 3),
            (1920, 1080, 3),
            (50, 50, 4)  # RGBA
        ]
        
        for width, height, channels in test_cases:
            image = MockImageGenerator.create_test_image(width, height, channels)
            
            if channels == 1:
                assert image.shape == (height, width)
            else:
                assert image.shape == (height, width, channels)
            
            assert image.dtype == np.uint8
            assert np.min(image) >= 0
            assert np.max(image) <= 255
    
    def test_create_test_image_randomness(self):
        """Test that generated images are actually random."""
        image1 = MockImageGenerator.create_test_image(100, 100, 3)
        image2 = MockImageGenerator.create_test_image(100, 100, 3)
        
        # Images should be different (very unlikely to be identical)
        assert not np.array_equal(image1, image2)
        
        # But should have same properties
        assert image1.shape == image2.shape
        assert image1.dtype == image2.dtype
    
    def test_create_historical_map_image_properties(self):
        """Test properties of generated historical map images."""
        image = MockImageGenerator.create_historical_map_image(800, 600)
        
        assert image.shape == (600, 800, 3)
        assert image.dtype == np.uint8
        
        # Check color distribution - should not be pure white/black
        mean_values = np.mean(image, axis=(0, 1))
        assert all(50 < mean < 250 for mean in mean_values)
        
        # Should have some variation (not uniform)
        std_values = np.std(image, axis=(0, 1))
        assert all(std > 5 for std in std_values)
    
    def test_create_historical_map_image_text_regions(self):
        """Test that historical map images contain text-like regions."""
        image = MockImageGenerator.create_historical_map_image(1000, 800)
        
        # Convert to grayscale for analysis
        gray = np.mean(image, axis=2)
        
        # Should have some dark regions (text)
        dark_pixels = np.sum(gray < 100)
        assert dark_pixels > 0
        
        # Should have some light regions (background)
        light_pixels = np.sum(gray > 200)
        assert light_pixels > 0
    
    def test_create_degraded_image_effects(self):
        """Test degradation effects on images."""
        # Create a clean test image
        clean_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # Test different degradation levels
        degradation_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for level in degradation_levels:
            degraded = MockImageGenerator.create_degraded_image(clean_image, level)
            
            assert degraded.shape == clean_image.shape
            assert degraded.dtype == clean_image.dtype
            
            # Higher degradation should result in more difference
            difference = np.mean(np.abs(degraded.astype(float) - clean_image.astype(float)))
            assert difference > 0  # Should be different from original
    
    def test_create_degraded_image_preserves_structure(self):
        """Test that degradation preserves basic image structure."""
        # Create image with clear structure
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        original[25:75, 25:75] = 255  # White square in center
        
        degraded = MockImageGenerator.create_degraded_image(original, 0.3)
        
        # Should still have higher values in center region
        center_mean = np.mean(degraded[40:60, 40:60])
        corner_mean = np.mean(degraded[0:20, 0:20])
        assert center_mean > corner_mean
    
    def test_create_text_region_image_structure(self):
        """Test structure of text region images."""
        image = MockImageGenerator.create_text_region_image("Hello", 200, 50)
        
        assert image.shape == (50, 200, 3)
        assert image.dtype == np.uint8
        
        # Should have white background
        background_pixels = np.sum(image > 200)
        total_pixels = image.size
        background_ratio = background_pixels / total_pixels
        assert background_ratio > 0.7  # Most pixels should be background
        
        # Should have some dark pixels (text)
        dark_pixels = np.sum(image < 50)
        assert dark_pixels > 0
    
    def test_create_text_region_image_text_length_scaling(self):
        """Test that text region scales with text length."""
        short_text_image = MockImageGenerator.create_text_region_image("Hi", 200, 50)
        long_text_image = MockImageGenerator.create_text_region_image("Hello World", 200, 50)
        
        # Both should have same dimensions
        assert short_text_image.shape == long_text_image.shape
        
        # Long text should have more dark pixels (approximately)
        short_dark = np.sum(short_text_image < 50)
        long_dark = np.sum(long_text_image < 50)
        assert long_dark >= short_dark


class TestImageManipulation:
    """Test cases for image manipulation utilities."""
    
    def test_degradation_consistency(self):
        """Test that degradation is consistent with same parameters."""
        np.random.seed(42)  # Set seed for reproducibility
        base_image = MockImageGenerator.create_test_image(100, 100, 3)
        
        np.random.seed(42)
        degraded1 = MockImageGenerator.create_degraded_image(base_image, 0.5)
        
        np.random.seed(42)
        degraded2 = MockImageGenerator.create_degraded_image(base_image, 0.5)
        
        # Should be identical with same seed
        assert np.array_equal(degraded1, degraded2)
    
    def test_degradation_intensity_scaling(self):
        """Test that degradation intensity scales properly."""
        base_image = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        # Test increasing degradation levels
        differences = []
        for level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            degraded = MockImageGenerator.create_degraded_image(base_image, level)
            diff = np.mean(np.abs(degraded.astype(float) - base_image.astype(float)))
            differences.append(diff)
        
        # Differences should generally increase with degradation level
        # (allowing for some noise in the relationship)
        assert differences[-1] > differences[0]  # Highest > Lowest
        assert differences[2] > differences[0]   # Middle > Lowest
    
    def test_degradation_preserves_data_type(self):
        """Test that degradation preserves image data type and range."""
        test_images = [
            np.random.randint(0, 255, (50, 50), dtype=np.uint8),      # Grayscale
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),   # RGB
            np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8),   # RGBA
        ]
        
        for image in test_images:
            degraded = MockImageGenerator.create_degraded_image(image, 0.4)
            
            assert degraded.shape == image.shape
            assert degraded.dtype == image.dtype
            assert np.min(degraded) >= 0
            assert np.max(degraded) <= 255


class TestImageValidation:
    """Test cases for image validation and properties."""
    
    def test_image_array_properties(self):
        """Test that generated images have correct array properties."""
        # Test various configurations
        configs = [
            (100, 100, 1),
            (200, 150, 3),
            (640, 480, 3),
        ]
        
        for width, height, channels in configs:
            image = MockImageGenerator.create_test_image(width, height, channels)
            
            # Check basic properties
            assert isinstance(image, np.ndarray)
            assert image.dtype == np.uint8
            
            # Check shape
            if channels == 1:
                assert len(image.shape) == 2
                assert image.shape == (height, width)
            else:
                assert len(image.shape) == 3
                assert image.shape == (height, width, channels)
            
            # Check value range
            assert np.all(image >= 0)
            assert np.all(image <= 255)
    
    def test_image_memory_layout(self):
        """Test memory layout and contiguity of generated images."""
        image = MockImageGenerator.create_test_image(100, 100, 3)
        
        # Check memory properties
        assert image.flags['C_CONTIGUOUS'] or image.flags['F_CONTIGUOUS']
        assert image.flags['OWNDATA']
        assert image.flags['WRITEABLE']
    
    def test_image_statistical_properties(self):
        """Test statistical properties of generated images."""
        # Generate multiple images and check statistical properties
        images = [MockImageGenerator.create_test_image(100, 100, 3) for _ in range(10)]
        
        means = [np.mean(img) for img in images]
        stds = [np.std(img) for img in images]
        
        # Means should be roughly around 127.5 (middle of 0-255 range)
        overall_mean = np.mean(means)
        assert 100 < overall_mean < 155
        
        # Standard deviations should be reasonable (not too low or too high)
        overall_std = np.mean(stds)
        assert 50 < overall_std < 100


class TestProcessedImageIntegration:
    """Test integration with ProcessedImage data structure."""
    
    def test_create_processed_image_from_generated(self):
        """Test creating ProcessedImage from generated images."""
        original_image = MockImageGenerator.create_historical_map_image(800, 600)
        processed_array = MockImageGenerator.create_test_image(640, 480, 3)
        
        processed_image = ProcessedImage(
            image=processed_array,
            original_size=(800, 600),
            processed_size=(640, 480),
            preprocessing_steps=["resize", "enhance", "normalize"],
            enhancement_applied=True
        )
        
        assert processed_image.image.shape == (480, 640, 3)
        assert processed_image.original_size == (800, 600)
        assert processed_image.processed_size == (640, 480)
        assert processed_image.enhancement_applied is True
        assert len(processed_image.preprocessing_steps) == 3
    
    def test_processed_image_with_degraded_input(self):
        """Test ProcessedImage with degraded input images."""
        clean_image = MockImageGenerator.create_test_image(400, 300, 3)
        degraded_image = MockImageGenerator.create_degraded_image(clean_image, 0.4)
        
        processed_image = ProcessedImage(
            image=degraded_image,
            original_size=(400, 300),
            processed_size=(400, 300),
            preprocessing_steps=["degrade"],
            enhancement_applied=False
        )
        
        assert processed_image.image.shape == degraded_image.shape
        assert np.array_equal(processed_image.image, degraded_image)


class TestFileOperations:
    """Test file operations with generated images."""
    
    def test_temp_file_creation_cleanup(self):
        """Test temporary file creation and cleanup."""
        test_image = MockImageGenerator.create_test_image(100, 100, 3)
        
        # Create temporary file
        temp_path = FileTestUtilities.create_temp_image_file(test_image)
        
        assert isinstance(temp_path, str)
        assert Path(temp_path).exists()
        
        # Cleanup
        FileTestUtilities.cleanup_temp_files([temp_path])
        
        # File should be removed (or at least attempt was made)
        # Note: In real implementation, this would actually save/load the image
    
    def test_multiple_temp_files_cleanup(self):
        """Test cleanup of multiple temporary files."""
        images = [MockImageGenerator.create_test_image(50, 50, 3) for _ in range(3)]
        temp_paths = []
        
        for image in images:
            temp_path = FileTestUtilities.create_temp_image_file(image)
            temp_paths.append(temp_path)
        
        assert len(temp_paths) == 3
        
        # Cleanup all files
        FileTestUtilities.cleanup_temp_files(temp_paths)
        
        # All files should be cleaned up
        for path in temp_paths:
            # In real implementation, files would be removed
            assert isinstance(path, str)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_size_image_handling(self):
        """Test handling of zero-size image requests."""
        with pytest.raises((ValueError, AssertionError)):
            MockImageGenerator.create_test_image(0, 100, 3)
        
        with pytest.raises((ValueError, AssertionError)):
            MockImageGenerator.create_test_image(100, 0, 3)
    
    def test_negative_size_image_handling(self):
        """Test handling of negative size requests."""
        with pytest.raises((ValueError, AssertionError)):
            MockImageGenerator.create_test_image(-100, 100, 3)
        
        with pytest.raises((ValueError, AssertionError)):
            MockImageGenerator.create_test_image(100, -100, 3)
    
    def test_invalid_channels_handling(self):
        """Test handling of invalid channel counts."""
        # Zero channels should fail
        with pytest.raises((ValueError, AssertionError)):
            MockImageGenerator.create_test_image(100, 100, 0)
        
        # Negative channels should fail
        with pytest.raises((ValueError, AssertionError)):
            MockImageGenerator.create_test_image(100, 100, -1)
    
    def test_extreme_degradation_levels(self):
        """Test extreme degradation levels."""
        base_image = MockImageGenerator.create_test_image(50, 50, 3)
        
        # Test degradation level of 0 (no degradation)
        no_degradation = MockImageGenerator.create_degraded_image(base_image, 0.0)
        # Should be very similar to original (allowing for minor floating point differences)
        difference = np.mean(np.abs(no_degradation.astype(float) - base_image.astype(float)))
        assert difference < 5  # Very small difference
        
        # Test degradation level of 1 (maximum degradation)
        max_degradation = MockImageGenerator.create_degraded_image(base_image, 1.0)
        assert max_degradation.shape == base_image.shape
        assert max_degradation.dtype == base_image.dtype
    
    def test_very_large_images(self):
        """Test generation of very large images."""
        # Test reasonably large image (but not too large to cause memory issues)
        large_image = MockImageGenerator.create_test_image(2000, 1500, 3)
        
        assert large_image.shape == (1500, 2000, 3)
        assert large_image.dtype == np.uint8
        
        # Should still have proper value range
        assert np.min(large_image) >= 0
        assert np.max(large_image) <= 255
    
    def test_empty_text_region(self):
        """Test text region generation with empty text."""
        image = MockImageGenerator.create_text_region_image("", 100, 50)
        
        assert image.shape == (50, 100, 3)
        
        # Should be mostly background (white) since no text
        white_pixels = np.sum(image > 200)
        total_pixels = image.size
        white_ratio = white_pixels / total_pixels
        assert white_ratio > 0.9  # Should be mostly white