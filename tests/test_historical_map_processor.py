"""
Unit tests for historical map processing utilities.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.processing.historical_map_processor import HistoricalMapProcessor
from src.processing.image_processor import BasicImageProcessor
from src.models.data_structures import ProcessedImage


class TestHistoricalMapProcessor:
    """Test suite for HistoricalMapProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a HistoricalMapProcessor instance for testing."""
        return HistoricalMapProcessor()
    
    @pytest.fixture
    def basic_processor(self):
        """Create a BasicImageProcessor instance for testing."""
        return BasicImageProcessor()
    
    @pytest.fixture
    def sample_historical_map(self):
        """Create a sample historical map-like image."""
        # Create a more complex image simulating a historical map
        image = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light background
        
        # Add some "faded" text regions with varying intensities
        cv2.rectangle(image, (50, 50), (200, 80), (180, 180, 180), -1)  # Faded text
        cv2.rectangle(image, (50, 100), (300, 130), (160, 160, 160), -1)  # More faded
        cv2.rectangle(image, (50, 150), (150, 180), (140, 140, 140), -1)  # Very faded
        
        # Add some "map features" (darker regions)
        cv2.rectangle(image, (400, 200), (550, 350), (100, 120, 100), -1)  # Green area
        cv2.circle(image, (300, 300), 30, (120, 100, 80), -1)  # Brown circle
        
        # Add some noise and artifacts
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some small artifacts (dots)
        for _ in range(20):
            x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
        
        return image
    
    @pytest.fixture
    def rotated_image(self, sample_historical_map):
        """Create a rotated version of the sample image."""
        center = (sample_historical_map.shape[1] // 2, sample_historical_map.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)  # 5 degree rotation
        rotated = cv2.warpAffine(sample_historical_map, rotation_matrix,
                               (sample_historical_map.shape[1], sample_historical_map.shape[0]),
                               borderValue=(255, 255, 255))
        return rotated
    
    def test_initialization(self):
        """Test HistoricalMapProcessor initialization."""
        processor = HistoricalMapProcessor()
        assert processor.text_enhancement_strength == 1.5
        assert processor.artifact_removal_threshold == 0.3
        assert processor.rotation_detection_sensitivity == 0.5
        assert processor.basic_processor is not None
        
        # Test custom initialization
        basic_proc = BasicImageProcessor()
        custom_processor = HistoricalMapProcessor(
            basic_processor=basic_proc,
            text_enhancement_strength=2.0,
            artifact_removal_threshold=0.5,
            rotation_detection_sensitivity=0.8
        )
        assert custom_processor.basic_processor is basic_proc
        assert custom_processor.text_enhancement_strength == 2.0
        assert custom_processor.artifact_removal_threshold == 0.5
        assert custom_processor.rotation_detection_sensitivity == 0.8
        
        # Test parameter bounds
        bounded_processor = HistoricalMapProcessor(
            text_enhancement_strength=5.0,  # Should be clamped to 3.0
            artifact_removal_threshold=1.5,  # Should be clamped to 0.9
            rotation_detection_sensitivity=-0.5  # Should be clamped to 0.1
        )
        assert bounded_processor.text_enhancement_strength == 3.0
        assert bounded_processor.artifact_removal_threshold == 0.9
        assert bounded_processor.rotation_detection_sensitivity == 0.1
    
    def test_enhance_faded_text(self, processor, sample_historical_map):
        """Test faded text enhancement methods."""
        # Test adaptive enhancement
        enhanced_adaptive = processor.enhance_faded_text(sample_historical_map, 'adaptive')
        assert enhanced_adaptive.shape == sample_historical_map.shape
        assert enhanced_adaptive.dtype == sample_historical_map.dtype
        
        # Test unsharp masking
        enhanced_unsharp = processor.enhance_faded_text(sample_historical_map, 'unsharp')
        assert enhanced_unsharp.shape == sample_historical_map.shape
        
        # Test top-hat transform
        enhanced_tophat = processor.enhance_faded_text(sample_historical_map, 'tophat')
        assert enhanced_tophat.shape == sample_historical_map.shape
        
        # Test combined method
        enhanced_combined = processor.enhance_faded_text(sample_historical_map, 'combined')
        assert enhanced_combined.shape == sample_historical_map.shape
        
        # Test with grayscale image
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        enhanced_gray = processor.enhance_faded_text(gray_image, 'adaptive')
        assert enhanced_gray.shape == gray_image.shape
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported enhancement method"):
            processor.enhance_faded_text(sample_historical_map, 'invalid')
    
    def test_detect_and_correct_rotation(self, processor, sample_historical_map, rotated_image):
        """Test rotation detection and correction."""
        # Test with normal image (should detect minimal rotation)
        corrected, angle = processor.detect_and_correct_rotation(sample_historical_map)
        assert corrected.shape[:2] == sample_historical_map.shape[:2] or corrected.shape[0] >= sample_historical_map.shape[0]
        assert isinstance(angle, float)
        
        # Test with rotated image (should detect and correct rotation)
        corrected_rotated, detected_angle = processor.detect_and_correct_rotation(rotated_image)
        assert corrected_rotated.shape[0] >= rotated_image.shape[0]  # May be larger due to rotation
        assert isinstance(detected_angle, float)
        
        # Test with custom angle range
        corrected_custom, angle_custom = processor.detect_and_correct_rotation(
            sample_historical_map, angle_range=(-10, 10), angle_step=1.0)
        assert isinstance(angle_custom, float)
        
        # Test with grayscale image
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        corrected_gray, angle_gray = processor.detect_and_correct_rotation(gray_image)
        assert len(corrected_gray.shape) == 2 or len(corrected_gray.shape) == 3
    
    def test_transform_color_space_for_text(self, processor, sample_historical_map):
        """Test color space transformations for text enhancement."""
        # Test LAB enhancement
        lab_enhanced = processor.transform_color_space_for_text(sample_historical_map, 'lab_enhancement')
        assert lab_enhanced.shape == sample_historical_map.shape
        assert lab_enhanced.dtype == sample_historical_map.dtype
        
        # Test HSV saturation
        hsv_enhanced = processor.transform_color_space_for_text(sample_historical_map, 'hsv_saturation')
        assert hsv_enhanced.shape == sample_historical_map.shape
        
        # Test YUV luminance
        yuv_enhanced = processor.transform_color_space_for_text(sample_historical_map, 'yuv_luminance')
        assert yuv_enhanced.shape == sample_historical_map.shape
        
        # Test custom channel
        custom_enhanced = processor.transform_color_space_for_text(sample_historical_map, 'custom_channel')
        assert custom_enhanced.shape == sample_historical_map.shape
        
        # Test with grayscale image (should return unchanged)
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        gray_result = processor.transform_color_space_for_text(gray_image, 'lab_enhancement')
        np.testing.assert_array_equal(gray_result, gray_image)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported color transformation method"):
            processor.transform_color_space_for_text(sample_historical_map, 'invalid')
    
    def test_remove_artifacts_and_noise(self, processor, sample_historical_map):
        """Test artifact and noise removal methods."""
        # Test morphological method
        cleaned_morph = processor.remove_artifacts_and_noise(sample_historical_map, 'morphological')
        assert cleaned_morph.shape == sample_historical_map.shape
        assert cleaned_morph.dtype == sample_historical_map.dtype
        
        # Test connected components method
        cleaned_cc = processor.remove_artifacts_and_noise(sample_historical_map, 'connected_components')
        assert cleaned_cc.shape == sample_historical_map.shape
        
        # Test watershed method
        cleaned_watershed = processor.remove_artifacts_and_noise(sample_historical_map, 'watershed')
        assert cleaned_watershed.shape == sample_historical_map.shape
        
        # Test combined method
        cleaned_combined = processor.remove_artifacts_and_noise(sample_historical_map, 'combined')
        assert cleaned_combined.shape == sample_historical_map.shape
        
        # Test with grayscale image
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        cleaned_gray = processor.remove_artifacts_and_noise(gray_image, 'morphological')
        assert len(cleaned_gray.shape) == 2 or len(cleaned_gray.shape) == 3
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported artifact removal method"):
            processor.remove_artifacts_and_noise(sample_historical_map, 'invalid')
    
    def test_preprocess(self, processor, sample_historical_map):
        """Test complete preprocessing pipeline."""
        result = processor.preprocess(sample_historical_map)
        
        # Check return type
        assert isinstance(result, ProcessedImage)
        
        # Check processed image properties
        assert result.image.dtype == np.uint8
        assert len(result.image.shape) == 3
        assert result.image.shape[2] == 3
        
        # Check metadata
        assert result.original_size == (sample_historical_map.shape[1], sample_historical_map.shape[0])
        assert result.enhancement_applied is True
        assert len(result.preprocessing_steps) > 0
        
        # Check that expected steps are present
        expected_steps = [
            "standardize_format",
            "color_space_transformation",
            "faded_text_enhancement",
            "artifact_removal",
            "noise_reduction",
            "final_contrast_enhancement",
            "resolution_normalization"
        ]
        for step in expected_steps:
            assert any(step in s for s in result.preprocessing_steps), f"Missing step: {step}"
    
    def test_enhance_text_visibility(self, processor, sample_historical_map):
        """Test enhanced text visibility method."""
        enhanced = processor.enhance_text_visibility(sample_historical_map)
        assert enhanced.shape == sample_historical_map.shape
        assert enhanced.dtype == sample_historical_map.dtype
        
        # Test with grayscale
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        enhanced_gray = processor.enhance_text_visibility(gray_image)
        assert enhanced_gray.shape == gray_image.shape
    
    def test_correct_orientation(self, processor, sample_historical_map, rotated_image):
        """Test orientation correction method."""
        # Test with normal image
        corrected = processor.correct_orientation(sample_historical_map)
        assert corrected.shape[:2] == sample_historical_map.shape[:2] or corrected.shape[0] >= sample_historical_map.shape[0]
        
        # Test with rotated image
        corrected_rotated = processor.correct_orientation(rotated_image)
        assert corrected_rotated.shape[0] >= rotated_image.shape[0]
    
    def test_normalize_resolution(self, processor, sample_historical_map):
        """Test resolution normalization."""
        # Test with default size
        normalized = processor.normalize_resolution(sample_historical_map)
        assert normalized.shape[:2] == processor.basic_processor.default_target_size[::-1]
        
        # Test with custom size
        target_size = (400, 300)
        normalized_custom = processor.normalize_resolution(sample_historical_map, target_size)
        assert normalized_custom.shape[1] == target_size[0]  # width
        assert normalized_custom.shape[0] == target_size[1]  # height
    
    def test_get_preprocessing_stats(self, processor, sample_historical_map):
        """Test preprocessing statistics calculation."""
        stats = processor.get_preprocessing_stats(sample_historical_map)
        
        # Check that all expected statistics are present
        expected_stats = [
            'mean_intensity', 'std_intensity', 'contrast_ratio',
            'dynamic_range', 'entropy', 'edge_density', 'text_likelihood'
        ]
        for stat in expected_stats:
            assert stat in stats
            assert isinstance(stats[stat], float)
            assert not np.isnan(stats[stat])
        
        # Check reasonable value ranges
        assert 0 <= stats['mean_intensity'] <= 255
        assert 0 <= stats['std_intensity'] <= 255
        assert 0 <= stats['contrast_ratio']
        assert 0 <= stats['dynamic_range'] <= 255
        assert 0 <= stats['entropy'] <= 8  # Max entropy for 8-bit image
        assert 0 <= stats['edge_density'] <= 1
        assert 0 <= stats['text_likelihood'] <= 1
        
        # Test with grayscale image
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        stats_gray = processor.get_preprocessing_stats(gray_image)
        assert len(stats_gray) == len(expected_stats)
    
    def test_private_methods(self, processor, sample_historical_map):
        """Test private helper methods."""
        gray_image = cv2.cvtColor(sample_historical_map, cv2.COLOR_BGR2GRAY)
        
        # Test entropy calculation
        entropy = processor._calculate_entropy(gray_image)
        assert isinstance(entropy, float)
        assert 0 <= entropy <= 8
        
        # Test edge density calculation
        edge_density = processor._calculate_edge_density(gray_image)
        assert isinstance(edge_density, float)
        assert 0 <= edge_density <= 1
        
        # Test text likelihood estimation
        text_likelihood = processor._estimate_text_likelihood(gray_image)
        assert isinstance(text_likelihood, float)
        assert 0 <= text_likelihood <= 1
        
        # Test image rotation
        rotated = processor._rotate_image(sample_historical_map, 10)
        assert rotated.shape[0] >= sample_historical_map.shape[0]  # May be larger
        assert rotated.dtype == sample_historical_map.dtype
    
    def test_integration_with_basic_processor(self, basic_processor):
        """Test integration with BasicImageProcessor."""
        # Create processor with custom basic processor
        processor = HistoricalMapProcessor(basic_processor=basic_processor)
        assert processor.basic_processor is basic_processor
        
        # Test that basic processor methods are accessible
        sample_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        normalized = processor.normalize_resolution(sample_image, (200, 200))
        assert normalized.shape == (200, 200, 3)
        
        # Test preprocessing uses the basic processor
        result = processor.preprocess(sample_image)
        assert isinstance(result, ProcessedImage)


if __name__ == "__main__":
    pytest.main([__file__])