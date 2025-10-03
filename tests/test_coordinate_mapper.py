"""
Tests for coordinate mapping utilities.
"""

import pytest
import math
from unittest.mock import Mock

from src.linking.coordinate_mapper import CoordinateMapper, CoordinateTransform
from src.models.data_structures import Detection, BoundingBox


class TestCoordinateTransform:
    """Test CoordinateTransform dataclass."""
    
    def test_valid_transform(self):
        """Test creating valid coordinate transform."""
        transform = CoordinateTransform(
            scale_x=2.0, scale_y=1.5, 
            offset_x=10.0, offset_y=5.0,
            rotation_angle=0.1
        )
        
        assert transform.scale_x == 2.0
        assert transform.scale_y == 1.5
        assert transform.offset_x == 10.0
        assert transform.offset_y == 5.0
        assert transform.rotation_angle == 0.1
    
    def test_invalid_scale_factors(self):
        """Test validation of scale factors."""
        with pytest.raises(ValueError, match="Scale factors must be positive"):
            CoordinateTransform(scale_x=0, scale_y=1.0, offset_x=0, offset_y=0)
        
        with pytest.raises(ValueError, match="Scale factors must be positive"):
            CoordinateTransform(scale_x=1.0, scale_y=-1.0, offset_x=0, offset_y=0)


class TestCoordinateMapper:
    """Test CoordinateMapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = CoordinateMapper()
        self.transform = CoordinateTransform(
            scale_x=100.0, scale_y=100.0,
            offset_x=-50.0, offset_y=-50.0
        )
    
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = CoordinateMapper()
        assert mapper.transform is not None
        assert mapper.transform.scale_x == 1.0
        assert mapper.transform.scale_y == 1.0
    
    def test_initialization_with_transform(self):
        """Test mapper initialization with custom transform."""
        mapper = CoordinateMapper(self.transform)
        assert mapper.transform == self.transform
    
    def test_set_transform(self):
        """Test setting coordinate transform."""
        self.mapper.set_transform(self.transform)
        assert self.mapper.transform == self.transform
    
    def test_pixel_to_coordinate_basic(self):
        """Test basic pixel to coordinate transformation."""
        image_size = (1000, 1000)
        pixel_coords = (500, 300)
        
        map_coords = self.mapper.pixel_to_coordinate(pixel_coords, image_size)
        
        # With default transform (scale=1, offset=0)
        expected_x = 500 / 1000  # 0.5
        expected_y = 300 / 1000  # 0.3
        
        assert abs(map_coords[0] - expected_x) < 1e-6
        assert abs(map_coords[1] - expected_y) < 1e-6
    
    def test_pixel_to_coordinate_with_transform(self):
        """Test pixel to coordinate transformation with custom transform."""
        self.mapper.set_transform(self.transform)
        image_size = (1000, 1000)
        pixel_coords = (500, 300)
        
        map_coords = self.mapper.pixel_to_coordinate(pixel_coords, image_size)
        
        # Normalized coordinates: (0.5, 0.3)
        # With transform: scale=(100, 100), offset=(-50, -50)
        expected_x = 0.5 * 100.0 - 50.0  # 0
        expected_y = 0.3 * 100.0 - 50.0  # -20
        
        assert abs(map_coords[0] - expected_x) < 1e-6
        assert abs(map_coords[1] - expected_y) < 1e-6
    
    def test_coordinate_to_pixel_basic(self):
        """Test basic coordinate to pixel transformation."""
        image_size = (1000, 1000)
        map_coords = (0.5, 0.3)
        
        pixel_coords = self.mapper.coordinate_to_pixel(map_coords, image_size)
        
        # With default transform
        expected_x = int(0.5 * 1000)  # 500
        expected_y = int(0.3 * 1000)  # 300
        
        assert pixel_coords[0] == expected_x
        assert pixel_coords[1] == expected_y
    
    def test_coordinate_to_pixel_with_transform(self):
        """Test coordinate to pixel transformation with custom transform."""
        self.mapper.set_transform(self.transform)
        image_size = (1000, 1000)
        map_coords = (0.0, -20.0)  # Should map back to (500, 300)
        
        pixel_coords = self.mapper.coordinate_to_pixel(map_coords, image_size)
        
        # Reverse transform: (0 + 50) / 100 = 0.5, (-20 + 50) / 100 = 0.3
        # Pixel coords: 0.5 * 1000 = 500, 0.3 * 1000 = 300
        assert pixel_coords[0] == 500
        assert pixel_coords[1] == 300
    
    def test_pixel_coordinate_roundtrip(self):
        """Test roundtrip conversion between pixel and map coordinates."""
        self.mapper.set_transform(self.transform)
        image_size = (1000, 1000)
        original_pixel = (750, 250)
        
        # Convert to map coordinates and back
        map_coords = self.mapper.pixel_to_coordinate(original_pixel, image_size)
        final_pixel = self.mapper.coordinate_to_pixel(map_coords, image_size)
        
        assert abs(final_pixel[0] - original_pixel[0]) <= 1
        assert abs(final_pixel[1] - original_pixel[1]) <= 1
    
    def test_pixel_to_coordinate_with_rotation(self):
        """Test pixel to coordinate transformation with rotation."""
        transform = CoordinateTransform(
            scale_x=1.0, scale_y=1.0,
            offset_x=0.0, offset_y=0.0,
            rotation_angle=math.pi/4  # 45 degrees
        )
        self.mapper.set_transform(transform)
        
        image_size = (1000, 1000)
        pixel_coords = (750, 250)  # Top-right quadrant
        
        map_coords = self.mapper.pixel_to_coordinate(pixel_coords, image_size)
        
        # Should be rotated from original position
        assert map_coords[0] != 0.75  # x-coordinate changed due to rotation
        assert map_coords[1] != 0.25  # y-coordinate changed due to rotation
    
    def test_coordinate_bounds_clamping(self):
        """Test that pixel coordinates are clamped to image bounds."""
        image_size = (100, 100)
        
        # Test coordinates outside image bounds
        map_coords = (2.0, -1.0)  # Outside normalized bounds
        pixel_coords = self.mapper.coordinate_to_pixel(map_coords, image_size)
        
        # Should be clamped to image bounds
        assert 0 <= pixel_coords[0] < image_size[0]
        assert 0 <= pixel_coords[1] < image_size[1]
    
    def test_analyze_spatial_relationships_empty(self):
        """Test spatial relationship analysis with empty input."""
        relationships = self.mapper.analyze_spatial_relationships([])
        
        assert relationships == {}
    
    def test_analyze_spatial_relationships_single(self):
        """Test spatial relationship analysis with single detection."""
        bbox = BoundingBox(x=100, y=100, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 120)
        )
        
        relationships = self.mapper.analyze_spatial_relationships([detection])
        
        assert 'isolated' in relationships
        assert detection in relationships['isolated']
    
    def test_analyze_spatial_relationships_horizontal(self):
        """Test detection of horizontally aligned elements."""
        # Create two horizontally aligned detections
        bbox1 = BoundingBox(x=100, y=100, width=50, height=20, confidence=0.9)
        bbox2 = BoundingBox(x=200, y=105, width=50, height=20, confidence=0.8)
        
        detection1 = Detection(
            bbox=bbox1, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 120)
        )
        detection2 = Detection(
            bbox=bbox2, detection_type='word', confidence=0.8,
            image_coordinates=(200, 105, 250, 125)
        )
        
        relationships = self.mapper.analyze_spatial_relationships([detection1, detection2])
        
        assert 'horizontal_aligned' in relationships
        assert len(relationships['horizontal_aligned']) >= 2
    
    def test_calculate_spatial_confidence(self):
        """Test spatial confidence calculation."""
        bbox = BoundingBox(x=100, y=100, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 120)
        )
        
        # Test with coordinates close to detection center
        map_coordinates = (0.125, 0.11)  # Close to detection center in normalized coords
        
        confidence = self.mapper.calculate_spatial_confidence(detection, map_coordinates)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high for close coordinates
    
    def test_get_bounding_box_center(self):
        """Test bounding box center calculation."""
        bbox = BoundingBox(x=100, y=200, width=50, height=30, confidence=0.9)
        detection = Detection(
            bbox=bbox, detection_type='word', confidence=0.9,
            image_coordinates=(100, 200, 150, 230)
        )
        
        center = self.mapper.get_bounding_box_center(detection)
        
        expected_x = 100 + 50 // 2  # 125
        expected_y = 200 + 30 // 2  # 215
        
        assert center == (expected_x, expected_y)
    
    def test_calculate_overlap_ratio(self):
        """Test overlap ratio calculation between detections."""
        # Create overlapping bounding boxes
        bbox1 = BoundingBox(x=100, y=100, width=50, height=30, confidence=0.9)
        bbox2 = BoundingBox(x=120, y=110, width=50, height=30, confidence=0.8)
        
        detection1 = Detection(
            bbox=bbox1, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 130)
        )
        detection2 = Detection(
            bbox=bbox2, detection_type='word', confidence=0.8,
            image_coordinates=(120, 110, 170, 140)
        )
        
        overlap_ratio = self.mapper.calculate_overlap_ratio(detection1, detection2)
        
        assert 0.0 <= overlap_ratio <= 1.0
        assert overlap_ratio > 0  # Should have some overlap
    
    def test_calculate_overlap_ratio_no_overlap(self):
        """Test overlap ratio calculation with no overlap."""
        bbox1 = BoundingBox(x=100, y=100, width=50, height=30, confidence=0.9)
        bbox2 = BoundingBox(x=200, y=200, width=50, height=30, confidence=0.8)
        
        detection1 = Detection(
            bbox=bbox1, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 130)
        )
        detection2 = Detection(
            bbox=bbox2, detection_type='word', confidence=0.8,
            image_coordinates=(200, 200, 250, 230)
        )
        
        overlap_ratio = self.mapper.calculate_overlap_ratio(detection1, detection2)
        
        assert overlap_ratio == 0.0
    
    def test_find_text_alignment_horizontal(self):
        """Test finding horizontally aligned text groups."""
        # Create horizontally aligned detections
        detections = []
        for i in range(3):
            bbox = BoundingBox(x=100 + i*60, y=100, width=50, height=20, confidence=0.9)
            detection = Detection(
                bbox=bbox, detection_type='word', confidence=0.9,
                image_coordinates=(100 + i*60, 100, 150 + i*60, 120)
            )
            detections.append(detection)
        
        aligned_groups = self.mapper.find_text_alignment(detections, 'horizontal')
        
        assert len(aligned_groups) == 1
        assert len(aligned_groups[0]) == 3
    
    def test_estimate_reading_order(self):
        """Test reading order estimation."""
        # Create detections in various positions
        detections = []
        positions = [(100, 100), (200, 100), (100, 150), (200, 150)]
        
        for i, (x, y) in enumerate(positions):
            bbox = BoundingBox(x=x, y=y, width=50, height=20, confidence=0.9)
            detection = Detection(
                bbox=bbox, detection_type='word', confidence=0.9,
                image_coordinates=(x, y, x+50, y+20)
            )
            detections.append(detection)
        
        ordered_detections = self.mapper.estimate_reading_order(detections)
        
        assert len(ordered_detections) == 4
        # First detection should be top-left (100, 100)
        first_center = self.mapper.get_bounding_box_center(ordered_detections[0])
        assert first_center == (125, 110)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Populate cache by calling methods
        self.mapper.analyze_spatial_relationships([])
        
        # Clear cache
        self.mapper.clear_cache()
        
        # Cache should be empty
        assert len(self.mapper._spatial_cache) == 0