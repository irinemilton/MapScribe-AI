"""
Unit tests for basic image processing utilities.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from src.processing.image_processor import BasicImageProcessor
from src.models.data_structures import ProcessedImage


class TestBasicImageProcessor:
    """Test suite for BasicImageProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a BasicImageProcessor instance for testing."""
        return BasicImageProcessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image with text-like patterns
        image = np.ones((200, 300, 3), dtype=np.uint8) * 255  # White background
        
        # Add some rectangular "text" regions
        cv2.rectangle(image, (50, 50), (150, 80), (0, 0, 0), -1)  # Black rectangle
        cv2.rectangle(image, (50, 100), (200, 130), (0, 0, 0), -1)  # Black rectangle
        cv2.rectangle(image, (50, 150), (120, 180), (0, 0, 0), -1)  # Black rectangle
        
        # Add some noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    @pytest.fixture
    def sample_grayscale_image(self):
        """Create a sample grayscale test image."""
        image = np.ones((200, 300), dtype=np.uint8) * 255  # White background
        
        # Add some rectangular regions
        cv2.rectangle(image, (50, 50), (150, 80), 0, -1)  # Black rectangle
        cv2.rectangle(image, (50, 100), (200, 130), 128, -1)  # Gray rectangle
        
        return image
    
    def test_initialization(self):
        """Test BasicImageProcessor initialization."""
        processor = BasicImageProcessor()
        assert processor.default_target_size == (1024, 1024)
        assert processor.preserve_aspect_ratio is True
        assert processor.interpolation_method == cv2.INTER_LANCZOS4
        
        # Test custom initialization
        custom_processor = BasicImageProcessor(
            default_target_size=(512, 512),
            preserve_aspect_ratio=False,
            interpolation_method=cv2.INTER_LINEAR
        )
        assert custom_processor.default_target_size == (512, 512)
        assert custom_processor.preserve_aspect_ratio is False
        assert custom_processor.interpolation_method == cv2.INTER_LINEAR
    
    def test_load_save_image(self, processor, sample_image):
        """Test image loading and saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test image
            test_path = Path(temp_dir) / "test_image.png"
            processor.save_image(sample_image, test_path)
            
            # Load image back
            loaded_image = processor.load_image(test_path)
            
            # Check that loaded image has correct properties
            assert loaded_image.shape == sample_image.shape
            assert loaded_image.dtype == np.uint8
            
            # Test different formats
            jpg_path = Path(temp_dir) / "test_image.jpg"
            processor.save_image(sample_image, jpg_path, quality=90)
            loaded_jpg = processor.load_image(jpg_path)
            assert loaded_jpg.shape == sample_image.shape
    
    def test_load_image_errors(self, processor):
        """Test error handling in image loading."""
        # Test non-existent file
        with pytest.raises(ValueError, match="Image file not found"):
            processor.load_image("non_existent_file.jpg")
        
        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            with pytest.raises(ValueError, match="Unsupported image format"):
                processor.load_image(temp_file.name)
    
    def test_convert_format(self, processor, sample_image, sample_grayscale_image):
        """Test image format conversion."""
        # Test BGR to RGB conversion
        rgb_image = processor.convert_format(sample_image, 'RGB')
        assert rgb_image.shape == sample_image.shape
        # Check that channels are swapped (R and B channels)
        np.testing.assert_array_equal(rgb_image[:, :, 0], sample_image[:, :, 2])
        np.testing.assert_array_equal(rgb_image[:, :, 2], sample_image[:, :, 0])
        
        # Test BGR to grayscale conversion
        gray_image = processor.convert_format(sample_image, 'GRAY')
        assert len(gray_image.shape) == 2
        assert gray_image.shape[:2] == sample_image.shape[:2]
        
        # Test grayscale to BGR conversion
        bgr_from_gray = processor.convert_format(sample_grayscale_image, 'BGR')
        assert len(bgr_from_gray.shape) == 3
        assert bgr_from_gray.shape[2] == 3
        
        # Test HSV conversion
        hsv_image = processor.convert_format(sample_image, 'HSV')
        assert hsv_image.shape == sample_image.shape
        
        # Test LAB conversion
        lab_image = processor.convert_format(sample_image, 'LAB')
        assert lab_image.shape == sample_image.shape
        
        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported target format"):
            processor.convert_format(sample_image, 'INVALID')
    
    def test_normalize_resolution(self, processor, sample_image):
        """Test image resolution normalization."""
        # Test with default target size
        normalized = processor.normalize_resolution(sample_image)
        assert normalized.shape[:2] == processor.default_target_size[::-1]  # (height, width)
        
        # Test with custom target size
        target_size = (400, 300)
        normalized_custom = processor.normalize_resolution(sample_image, target_size)
        assert normalized_custom.shape[1] == target_size[0]  # width
        assert normalized_custom.shape[0] == target_size[1]  # height
        
        # Test without preserving aspect ratio
        processor.preserve_aspect_ratio = False
        normalized_no_aspect = processor.normalize_resolution(sample_image, target_size)
        assert normalized_no_aspect.shape[:2] == target_size[::-1]
    
    def test_reduce_noise(self, processor, sample_image):
        """Test noise reduction methods."""
        # Test bilateral filter
        denoised_bilateral = processor.reduce_noise(sample_image, 'bilateral')
        assert denoised_bilateral.shape == sample_image.shape
        assert denoised_bilateral.dtype == sample_image.dtype
        
        # Test gaussian blur
        denoised_gaussian = processor.reduce_noise(sample_image, 'gaussian')
        assert denoised_gaussian.shape == sample_image.shape
        
        # Test median filter
        denoised_median = processor.reduce_noise(sample_image, 'median')
        assert denoised_median.shape == sample_image.shape
        
        # Test non-local means
        denoised_nlm = processor.reduce_noise(sample_image, 'nlm')
        assert denoised_nlm.shape == sample_image.shape
        
        # Test morphological filter
        denoised_morph = processor.reduce_noise(sample_image, 'morphological')
        assert denoised_morph.shape == sample_image.shape
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported noise reduction method"):
            processor.reduce_noise(sample_image, 'invalid')
    
    def test_enhance_contrast(self, processor, sample_image):
        """Test contrast enhancement methods."""
        # Test CLAHE
        enhanced_clahe = processor.enhance_contrast(sample_image, 'clahe')
        assert enhanced_clahe.shape == sample_image.shape
        assert enhanced_clahe.dtype == sample_image.dtype
        
        # Test histogram equalization
        enhanced_hist = processor.enhance_contrast(sample_image, 'histogram_eq')
        assert enhanced_hist.shape == sample_image.shape
        
        # Test adaptive histogram equalization
        enhanced_adaptive = processor.enhance_contrast(sample_image, 'adaptive')
        assert enhanced_adaptive.shape == sample_image.shape
        
        # Test gamma correction
        enhanced_gamma = processor.enhance_contrast(sample_image, 'gamma')
        assert enhanced_gamma.shape == sample_image.shape
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported contrast enhancement method"):
            processor.enhance_contrast(sample_image, 'invalid')
    
    def test_normalize_image(self, processor, sample_image):
        """Test image normalization methods."""
        # Test min-max normalization
        normalized_minmax = processor.normalize_image(sample_image, 'minmax')
        assert normalized_minmax.shape == sample_image.shape
        assert normalized_minmax.dtype == np.uint8
        assert np.min(normalized_minmax) >= 0
        assert np.max(normalized_minmax) <= 255
        
        # Test z-score normalization
        normalized_zscore = processor.normalize_image(sample_image, 'zscore')
        assert normalized_zscore.shape == sample_image.shape
        assert normalized_zscore.dtype == np.uint8
        
        # Test robust normalization
        normalized_robust = processor.normalize_image(sample_image, 'robust')
        assert normalized_robust.shape == sample_image.shape
        assert normalized_robust.dtype == np.uint8
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            processor.normalize_image(sample_image, 'invalid')
    
    def test_standardize_image(self, processor, sample_image, sample_grayscale_image):
        """Test image standardization."""
        # Test with color image
        standardized = processor.standardize_image(sample_image)
        assert standardized.shape == sample_image.shape
        assert standardized.dtype == np.uint8
        assert len(standardized.shape) == 3
        assert standardized.shape[2] == 3
        
        # Test with grayscale image
        standardized_gray = processor.standardize_image(sample_grayscale_image)
        assert standardized_gray.dtype == np.uint8
        assert len(standardized_gray.shape) == 3
        assert standardized_gray.shape[2] == 3
        
        # Test with float image
        float_image = sample_image.astype(np.float32) / 255.0
        standardized_float = processor.standardize_image(float_image)
        assert standardized_float.dtype == np.uint8
        assert len(standardized_float.shape) == 3
        
        # Test with 4-channel image (RGBA)
        rgba_image = np.concatenate([sample_image, np.ones((sample_image.shape[0], 
                                                           sample_image.shape[1], 1), 
                                                          dtype=np.uint8) * 255], axis=2)
        standardized_rgba = processor.standardize_image(rgba_image)
        assert standardized_rgba.shape[2] == 3  # Alpha channel removed
    
    def test_preprocess(self, processor, sample_image):
        """Test complete preprocessing pipeline."""
        result = processor.preprocess(sample_image)
        
        # Check return type
        assert isinstance(result, ProcessedImage)
        
        # Check processed image properties
        assert result.image.dtype == np.uint8
        assert len(result.image.shape) == 3
        assert result.image.shape[2] == 3
        
        # Check metadata
        assert result.original_size == (sample_image.shape[1], sample_image.shape[0])
        assert result.processed_size == processor.default_target_size
        assert result.enhancement_applied is True
        assert len(result.preprocessing_steps) > 0
        
        # Check that all expected steps are present
        expected_steps = [
            "standardize_format",
            "noise_reduction_bilateral", 
            "contrast_enhancement_clahe",
            "resolution_normalization",
            "pixel_normalization"
        ]
        for step in expected_steps:
            assert step in result.preprocessing_steps
    
    def test_enhance_text_visibility(self, processor, sample_image, sample_grayscale_image):
        """Test text visibility enhancement."""
        # Test with color image
        enhanced_color = processor.enhance_text_visibility(sample_image)
        assert enhanced_color.shape == sample_image.shape
        assert enhanced_color.dtype == sample_image.dtype
        
        # Test with grayscale image
        enhanced_gray = processor.enhance_text_visibility(sample_grayscale_image)
        assert enhanced_gray.shape == sample_grayscale_image.shape
        assert enhanced_gray.dtype == sample_grayscale_image.dtype
    
    def test_correct_orientation(self, processor, sample_image):
        """Test orientation correction."""
        # Test with normal image (should return unchanged or minimally changed)
        corrected = processor.correct_orientation(sample_image)
        assert corrected.shape == sample_image.shape
        assert corrected.dtype == sample_image.dtype
        
        # Test with artificially rotated image
        center = (sample_image.shape[1] // 2, sample_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 10, 1.0)  # 10 degree rotation
        rotated = cv2.warpAffine(sample_image, rotation_matrix, 
                               (sample_image.shape[1], sample_image.shape[0]))
        
        corrected_rotated = processor.correct_orientation(rotated)
        assert corrected_rotated.shape == rotated.shape
        assert corrected_rotated.dtype == rotated.dtype


if __name__ == "__main__":
    pytest.main([__file__])