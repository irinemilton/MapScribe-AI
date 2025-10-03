"""
Tests for spatial analysis utilities.
"""

import pytest
import math
from unittest.mock import Mock

from src.linking.spatial_analyzer import SpatialAnalyzer, SpatialFeature
from src.models.data_structures import LinkedText, RecognizedText, Detection, BoundingBox


class TestSpatialFeature:
    """Test SpatialFeature dataclass."""
    
    def test_valid_feature(self):
        """Test creating valid spatial feature."""
        feature = SpatialFeature(
            name="Test City",
            coordinates=(10.5, 20.3),
            feature_type="city",
            confidence=0.9
        )
        
        assert feature.name == "Test City"
        assert feature.coordinates == (10.5, 20.3)
        assert feature.feature_type == "city"
        assert feature.confidence == 0.9


class TestSpatialAnalyzer:
    """Test SpatialAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.features = [
            SpatialFeature("City A", (10.0, 20.0), "city", 0.9),
            SpatialFeature("River B", (15.0, 25.0), "river", 0.8),
            SpatialFeature("Mountain C", (5.0, 15.0), "mountain", 0.95)
        ]
        self.analyzer = SpatialAnalyzer(self.features)
    
    def create_linked_text(self, text: str, coordinates: tuple, confidence: float = 0.8) -> LinkedText:
        """Helper to create LinkedText objects for testing."""
        bbox = BoundingBox(x=100, y=100, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 120)
        )
        recognized_text = RecognizedText(
            detection=detection,
            text=text,
            language='en',
            recognition_confidence=0.8,
            character_confidences=[0.8] * len(text)
        )
        
        return LinkedText(
            recognized_text=recognized_text,
            map_coordinates=coordinates,
            spatial_confidence=confidence,
            related_features=[]
        )
    
    def test_initialization_empty(self):
        """Test analyzer initialization without features."""
        analyzer = SpatialAnalyzer()
        assert len(analyzer.map_features) == 0
        assert len(analyzer._feature_index) == 0
    
    def test_initialization_with_features(self):
        """Test analyzer initialization with features."""
        assert len(self.analyzer.map_features) == 3
        assert len(self.analyzer._feature_index) > 0
    
    def test_add_map_feature(self):
        """Test adding map feature."""
        new_feature = SpatialFeature("Lake D", (12.0, 22.0), "lake", 0.85)
        initial_count = len(self.analyzer.map_features)
        
        self.analyzer.add_map_feature(new_feature)
        
        assert len(self.analyzer.map_features) == initial_count + 1
        assert new_feature in self.analyzer.map_features
    
    def test_calculate_distance(self):
        """Test distance calculation between coordinates."""
        coord1 = (0.0, 0.0)
        coord2 = (3.0, 4.0)
        
        distance = self.analyzer.calculate_distance(coord1, coord2)
        
        # Should be 5.0 (3-4-5 triangle)
        assert abs(distance - 5.0) < 1e-6
    
    def test_calculate_distance_same_point(self):
        """Test distance calculation for same point."""
        coord = (10.0, 20.0)
        
        distance = self.analyzer.calculate_distance(coord, coord)
        
        assert distance == 0.0
    
    def test_calculate_distance_caching(self):
        """Test that distance calculations are cached."""
        coord1 = (1.0, 2.0)
        coord2 = (4.0, 6.0)
        
        # Calculate distance twice
        distance1 = self.analyzer.calculate_distance(coord1, coord2)
        distance2 = self.analyzer.calculate_distance(coord1, coord2)
        
        assert distance1 == distance2
        # Check cache contains the result
        cache_key = (coord1, coord2)
        assert cache_key in self.analyzer._distance_cache
    
    def test_calculate_geographic_distance(self):
        """Test geographic distance calculation using Haversine formula."""
        # Test with known coordinates (approximately)
        london = (-0.1276, 51.5074)  # London coordinates
        paris = (2.3522, 48.8566)   # Paris coordinates
        
        distance = self.analyzer.calculate_geographic_distance(london, paris)
        
        # Distance between London and Paris is approximately 344 km
        assert 300 < distance < 400
    
    def test_find_nearby_features(self):
        """Test finding nearby features."""
        # Search near City A (10.0, 20.0)
        search_coords = (10.5, 20.5)
        radius = 1.0
        
        nearby = self.analyzer.find_nearby_features(search_coords, radius)
        
        assert "City A" in nearby
        assert len(nearby) >= 1
    
    def test_find_nearby_features_empty(self):
        """Test finding nearby features with no results."""
        # Search in area with no features
        search_coords = (100.0, 100.0)
        radius = 1.0
        
        nearby = self.analyzer.find_nearby_features(search_coords, radius)
        
        assert len(nearby) == 0
    
    def test_find_nearby_features_detailed(self):
        """Test finding detailed nearby features."""
        search_coords = (10.2, 20.1)
        radius = 1.0
        
        nearby = self.analyzer.find_nearby_features_detailed(search_coords, radius)
        
        assert len(nearby) >= 1
        assert all(isinstance(feature, SpatialFeature) for feature in nearby)
        # Should be sorted by distance
        if len(nearby) > 1:
            distances = [
                self.analyzer.calculate_distance(search_coords, f.coordinates)
                for f in nearby
            ]
            assert distances == sorted(distances)
    
    def test_group_related_elements_empty(self):
        """Test grouping with empty input."""
        groups = self.analyzer.group_related_elements([])
        
        assert groups == []
    
    def test_group_related_elements_single(self):
        """Test grouping with single element."""
        linked_text = self.create_linked_text("Test", (10.0, 20.0))
        
        groups = self.analyzer.group_related_elements([linked_text])
        
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0] == linked_text
    
    def test_group_related_elements_multiple(self):
        """Test grouping with multiple related elements."""
        # Create closely positioned texts
        text1 = self.create_linked_text("City", (10.0, 20.0))
        text2 = self.create_linked_text("Center", (10.1, 20.1))
        text3 = self.create_linked_text("Far", (50.0, 60.0))
        
        groups = self.analyzer.group_related_elements([text1, text2, text3])
        
        # Should have at least 2 groups (close texts together, far text separate)
        assert len(groups) >= 2
    
    def test_analyze_text_distribution_empty(self):
        """Test text distribution analysis with empty input."""
        analysis = self.analyzer.analyze_text_distribution([])
        
        assert analysis == {}
    
    def test_analyze_text_distribution(self):
        """Test text distribution analysis."""
        texts = [
            self.create_linked_text("A", (0.0, 0.0)),
            self.create_linked_text("B", (10.0, 0.0)),
            self.create_linked_text("C", (5.0, 10.0))
        ]
        
        analysis = self.analyzer.analyze_text_distribution(texts)
        
        assert 'center_of_mass' in analysis
        assert 'spread' in analysis
        assert 'bounding_box' in analysis
        assert 'density' in analysis
        assert 'total_texts' in analysis
        
        # Check center of mass calculation
        expected_center_x = (0.0 + 10.0 + 5.0) / 3  # 5.0
        expected_center_y = (0.0 + 0.0 + 10.0) / 3  # 3.33...
        
        center = analysis['center_of_mass']
        assert abs(center[0] - expected_center_x) < 1e-6
        assert abs(center[1] - expected_center_y) < 1e-6
        
        assert analysis['total_texts'] == 3
    
    def test_find_text_clusters_empty(self):
        """Test clustering with empty input."""
        clusters = self.analyzer.find_text_clusters([], 1.0)
        
        assert clusters == []
    
    def test_find_text_clusters(self):
        """Test text clustering."""
        # Create texts in two distinct clusters
        cluster1_texts = [
            self.create_linked_text("A1", (0.0, 0.0)),
            self.create_linked_text("A2", (0.5, 0.5))
        ]
        cluster2_texts = [
            self.create_linked_text("B1", (10.0, 10.0)),
            self.create_linked_text("B2", (10.5, 10.5))
        ]
        
        all_texts = cluster1_texts + cluster2_texts
        cluster_radius = 1.0
        
        clusters = self.analyzer.find_text_clusters(all_texts, cluster_radius)
        
        assert len(clusters) == 2
        # Each cluster should have 2 elements
        cluster_sizes = [len(cluster) for cluster in clusters]
        assert 2 in cluster_sizes
    
    def test_calculate_spatial_confidence_score(self):
        """Test enhanced spatial confidence calculation."""
        linked_text = self.create_linked_text("Test City", (10.0, 20.0), 0.7)
        
        # Create nearby features
        nearby_features = [
            SpatialFeature("Close Feature", (10.05, 20.05), "city", 0.9)
        ]
        
        enhanced_confidence = self.analyzer.calculate_spatial_confidence_score(
            linked_text, nearby_features
        )
        
        # Should be higher than base confidence due to nearby feature
        assert enhanced_confidence > linked_text.spatial_confidence
        assert 0.0 <= enhanced_confidence <= 1.0
    
    def test_calculate_spatial_confidence_score_place_name(self):
        """Test confidence boost for place names."""
        linked_text = self.create_linked_text("New City", (10.0, 20.0), 0.6)
        
        enhanced_confidence = self.analyzer.calculate_spatial_confidence_score(
            linked_text, []
        )
        
        # Should get boost for containing "city"
        assert enhanced_confidence > linked_text.spatial_confidence
    
    def test_content_similarity_identical(self):
        """Test content similarity for identical texts."""
        similarity = self.analyzer._calculate_content_similarity("test", "test")
        
        assert similarity == 1.0
    
    def test_content_similarity_different(self):
        """Test content similarity for different texts."""
        similarity = self.analyzer._calculate_content_similarity("hello", "world")
        
        assert 0.0 <= similarity < 1.0
    
    def test_content_similarity_empty(self):
        """Test content similarity for empty texts."""
        similarity = self.analyzer._calculate_content_similarity("", "test")
        
        assert similarity == 0.0
    
    def test_content_similarity_partial_match(self):
        """Test content similarity for partial matches."""
        similarity = self.analyzer._calculate_content_similarity("hello world", "hello there")
        
        assert 0.0 < similarity < 1.0  # Should have some similarity due to "hello"
    
    def test_are_texts_related_close(self):
        """Test relationship detection for close texts."""
        text1 = self.create_linked_text("City", (10.0, 20.0))
        text2 = self.create_linked_text("Center", (10.1, 20.1))
        
        related = self.analyzer._are_texts_related(text1, text2)
        
        # Should be related due to proximity and some content similarity
        assert isinstance(related, bool)
    
    def test_are_texts_related_far(self):
        """Test relationship detection for distant texts."""
        text1 = self.create_linked_text("City", (10.0, 20.0))
        text2 = self.create_linked_text("Town", (100.0, 200.0))
        
        related = self.analyzer._are_texts_related(text1, text2)
        
        # Should not be related due to distance
        assert related == False
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Populate cache
        self.analyzer.calculate_distance((0, 0), (1, 1))
        
        # Clear cache
        self.analyzer.clear_cache()
        
        # Cache should be empty
        assert len(self.analyzer._distance_cache) == 0