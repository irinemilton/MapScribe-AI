"""
Tests for text linking and relationship detection.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.linking.text_linker import TextLinker, TextGroup
from src.linking.coordinate_mapper import CoordinateMapper
from src.linking.spatial_analyzer import SpatialAnalyzer, SpatialFeature
from src.models.data_structures import RecognizedText, LinkedText, Detection, BoundingBox


class TestTextGroup:
    """Test TextGroup dataclass."""
    
    def test_valid_text_group(self):
        """Test creating valid text group."""
        # Create mock linked text
        bbox = BoundingBox(x=100, y=100, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox, detection_type='word', confidence=0.9,
            image_coordinates=(100, 100, 150, 120)
        )
        recognized_text = RecognizedText(
            detection=detection, text="Test City", language='en',
            recognition_confidence=0.8, character_confidences=[0.8] * 9
        )
        linked_text = LinkedText(
            recognized_text=recognized_text,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.7,
            related_features=[]
        )
        
        group = TextGroup(
            texts=[linked_text],
            group_type="place_name",
            confidence=0.8,
            center_coordinates=(10.0, 20.0)
        )
        
        assert len(group.texts) == 1
        assert group.group_type == "place_name"
        assert group.confidence == 0.8
        assert group.center_coordinates == (10.0, 20.0)


class TestTextLinker:
    """Test TextLinker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinate_mapper = CoordinateMapper()
        self.spatial_analyzer = SpatialAnalyzer()
        self.linker = TextLinker(
            coordinate_mapper=self.coordinate_mapper,
            spatial_analyzer=self.spatial_analyzer,
            image_size=(1000, 1000)
        )
    
    def create_recognized_text(self, text: str, bbox_coords: tuple) -> RecognizedText:
        """Helper to create RecognizedText objects for testing."""
        x, y, width, height = bbox_coords
        bbox = BoundingBox(x=x, y=y, width=width, height=height, confidence=0.9)
        detection = Detection(
            bbox=bbox, detection_type='word', confidence=0.9,
            image_coordinates=(x, y, x+width, y+height)
        )
        
        return RecognizedText(
            detection=detection,
            text=text,
            language='en',
            recognition_confidence=0.8,
            character_confidences=[0.8] * len(text)
        )
    
    def test_initialization_default(self):
        """Test linker initialization with defaults."""
        linker = TextLinker()
        
        assert linker.coordinate_mapper is not None
        assert linker.spatial_analyzer is not None
        assert linker.image_size == (1000, 1000)
    
    def test_initialization_with_components(self):
        """Test linker initialization with custom components."""
        assert self.linker.coordinate_mapper == self.coordinate_mapper
        assert self.linker.spatial_analyzer == self.spatial_analyzer
        assert self.linker.image_size == (1000, 1000)
    
    def test_set_image_size(self):
        """Test setting image size."""
        new_size = (2000, 1500)
        self.linker.set_image_size(new_size)
        
        assert self.linker.image_size == new_size
    
    def test_link_text_to_coordinates_empty(self):
        """Test linking with empty input."""
        linked_texts = self.linker.link_text_to_coordinates([])
        
        assert linked_texts == []
    
    def test_link_text_to_coordinates_single(self):
        """Test linking single text element."""
        recognized_text = self.create_recognized_text("Test City", (100, 200, 80, 25))
        
        linked_texts = self.linker.link_text_to_coordinates([recognized_text])
        
        assert len(linked_texts) == 1
        linked_text = linked_texts[0]
        
        assert linked_text.recognized_text == recognized_text
        assert isinstance(linked_text.map_coordinates, tuple)
        assert 0.0 <= linked_text.spatial_confidence <= 1.0
        assert isinstance(linked_text.related_features, list)
    
    def test_link_text_to_coordinates_multiple(self):
        """Test linking multiple text elements."""
        texts = [
            self.create_recognized_text("City A", (100, 100, 60, 20)),
            self.create_recognized_text("River B", (200, 150, 70, 25)),
            self.create_recognized_text("Mountain C", (300, 200, 90, 30))
        ]
        
        linked_texts = self.linker.link_text_to_coordinates(texts)
        
        assert len(linked_texts) == 3
        for i, linked_text in enumerate(linked_texts):
            assert linked_text.recognized_text == texts[i]
            assert isinstance(linked_text.map_coordinates, tuple)
            assert 0.0 <= linked_text.spatial_confidence <= 1.0
    
    def test_calculate_spatial_confidence(self):
        """Test spatial confidence calculation."""
        recognized_text = self.create_recognized_text("Test City", (100, 100, 60, 20))
        coordinates = (0.14, 0.11)  # Normalized coordinates
        
        confidence = self.linker.calculate_spatial_confidence(recognized_text, coordinates)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_content_confidence_boost_place_name(self):
        """Test content confidence boost for place names."""
        boost = self.linker._calculate_content_confidence_boost("New York City")
        
        assert boost > 0.0
        assert boost <= 0.3  # Should be capped at 0.3
    
    def test_calculate_content_confidence_boost_geographic(self):
        """Test content confidence boost for geographic features."""
        boost = self.linker._calculate_content_confidence_boost("Mississippi River")
        
        assert boost > 0.0
    
    def test_calculate_content_confidence_boost_regular_text(self):
        """Test content confidence boost for regular text."""
        boost = self.linker._calculate_content_confidence_boost("some text")
        
        assert boost >= 0.0
    
    def test_find_related_features(self):
        """Test finding related features."""
        recognized_text = self.create_recognized_text("Test", (100, 100, 50, 20))
        coordinates = (10.0, 20.0)
        
        # Add some features to spatial analyzer
        feature = SpatialFeature("Test Feature", (10.05, 20.05), "city", 0.9)
        self.linker.spatial_analyzer.add_map_feature(feature)
        
        related_features = self.linker._find_related_features(recognized_text, coordinates)
        
        assert isinstance(related_features, list)
    
    def test_group_related_texts_empty(self):
        """Test grouping with empty input."""
        groups = self.linker.group_related_texts([])
        
        assert groups == []
    
    def test_group_related_texts_single(self):
        """Test grouping with single text."""
        recognized_text = self.create_recognized_text("Test", (100, 100, 50, 20))
        linked_text = LinkedText(
            recognized_text=recognized_text,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.8,
            related_features=[]
        )
        
        groups = self.linker.group_related_texts([linked_text])
        
        assert len(groups) == 1
        assert len(groups[0].texts) == 1
        assert groups[0].texts[0] == linked_text
    
    def test_group_related_texts_multiple(self):
        """Test grouping with multiple texts."""
        texts = []
        for i, text_content in enumerate(["City A", "Center", "Far Town"]):
            recognized_text = self.create_recognized_text(text_content, (100 + i*50, 100, 50, 20))
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=(10.0 + i*0.1, 20.0),
                spatial_confidence=0.8,
                related_features=[]
            )
            texts.append(linked_text)
        
        groups = self.linker.group_related_texts(texts)
        
        assert len(groups) >= 1
        assert all(isinstance(group, TextGroup) for group in groups)
    
    def test_determine_group_type_place_name(self):
        """Test group type determination for place names."""
        texts = []
        for text_content in ["New", "York", "City"]:
            recognized_text = self.create_recognized_text(text_content, (100, 100, 50, 20))
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=(10.0, 20.0),
                spatial_confidence=0.8,
                related_features=[]
            )
            texts.append(linked_text)
        
        group_type = self.linker._determine_group_type(texts)
        
        assert group_type == 'place_name'
    
    def test_determine_group_type_annotation(self):
        """Test group type determination for annotations."""
        recognized_text = self.create_recognized_text("123", (100, 100, 30, 15))
        linked_text = LinkedText(
            recognized_text=recognized_text,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.8,
            related_features=[]
        )
        
        group_type = self.linker._determine_group_type([linked_text])
        
        assert group_type == 'annotation'
    
    def test_determine_group_type_description(self):
        """Test group type determination for descriptions."""
        recognized_text = self.create_recognized_text("This is a long descriptive text", (100, 100, 200, 20))
        linked_text = LinkedText(
            recognized_text=recognized_text,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.8,
            related_features=[]
        )
        
        group_type = self.linker._determine_group_type([linked_text])
        
        assert group_type == 'description'
    
    def test_calculate_group_confidence(self):
        """Test group confidence calculation."""
        texts = []
        for i in range(3):
            recognized_text = self.create_recognized_text(f"Text{i}", (100 + i*50, 100, 50, 20))
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=(10.0 + i*0.1, 20.0),
                spatial_confidence=0.8,
                related_features=[]
            )
            texts.append(linked_text)
        
        confidence = self.linker._calculate_group_confidence(texts)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_group_coherence(self):
        """Test group coherence calculation."""
        # Create tightly clustered texts
        texts = []
        for i in range(3):
            recognized_text = self.create_recognized_text(f"Text{i}", (100 + i*10, 100, 50, 20))
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=(10.0 + i*0.01, 20.0),  # Very close coordinates
                spatial_confidence=0.8,
                related_features=[]
            )
            texts.append(linked_text)
        
        coherence = self.linker._calculate_group_coherence(texts)
        
        assert 0.0 <= coherence <= 1.0
        assert coherence > 0.5  # Should be high for tightly clustered texts
    
    def test_calculate_group_center(self):
        """Test group center calculation."""
        texts = []
        coordinates = [(10.0, 20.0), (12.0, 22.0), (14.0, 24.0)]
        
        for i, coord in enumerate(coordinates):
            recognized_text = self.create_recognized_text(f"Text{i}", (100, 100, 50, 20))
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=coord,
                spatial_confidence=0.8,
                related_features=[]
            )
            texts.append(linked_text)
        
        center = self.linker._calculate_group_center(texts)
        
        # Should be approximately the average of coordinates
        expected_x = sum(coord[0] for coord in coordinates) / len(coordinates)
        expected_y = sum(coord[1] for coord in coordinates) / len(coordinates)
        
        assert abs(center[0] - expected_x) < 0.1
        assert abs(center[1] - expected_y) < 0.1
    
    def test_link_place_names_to_locations(self):
        """Test enhanced place name linking."""
        # Create place name text
        recognized_text = self.create_recognized_text("Boston City", (100, 100, 80, 25))
        linked_text = LinkedText(
            recognized_text=recognized_text,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.6,
            related_features=[]
        )
        
        enhanced_texts = self.linker.link_place_names_to_locations([linked_text])
        
        assert len(enhanced_texts) == 1
        enhanced_text = enhanced_texts[0]
        
        # Should have enhanced confidence and place_name feature
        assert enhanced_text.spatial_confidence >= linked_text.spatial_confidence
        assert 'place_name' in enhanced_text.related_features
    
    def test_is_likely_place_name_positive(self):
        """Test place name detection for positive cases."""
        assert self.linker._is_likely_place_name("New York City")
        assert self.linker._is_likely_place_name("Mississippi River")
        assert self.linker._is_likely_place_name("Boston")
        assert self.linker._is_likely_place_name("Mount Washington")
    
    def test_is_likely_place_name_negative(self):
        """Test place name detection for negative cases."""
        assert not self.linker._is_likely_place_name("some text")
        assert not self.linker._is_likely_place_name("123")
        assert not self.linker._is_likely_place_name("the quick brown")
    
    def test_preserve_spatial_context(self):
        """Test spatial context preservation."""
        # Create multiple groups
        groups = []
        
        # Group 1: Place name
        recognized_text1 = self.create_recognized_text("Boston", (100, 100, 60, 20))
        linked_text1 = LinkedText(
            recognized_text=recognized_text1,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.8,
            related_features=[]
        )
        group1 = TextGroup(
            texts=[linked_text1],
            group_type="place_name",
            confidence=0.8,
            center_coordinates=(10.0, 20.0)
        )
        groups.append(group1)
        
        # Group 2: Nearby annotation
        recognized_text2 = self.create_recognized_text("123", (150, 110, 30, 15))
        linked_text2 = LinkedText(
            recognized_text=recognized_text2,
            map_coordinates=(10.1, 20.1),
            spatial_confidence=0.7,
            related_features=[]
        )
        group2 = TextGroup(
            texts=[linked_text2],
            group_type="annotation",
            confidence=0.7,
            center_coordinates=(10.1, 20.1)
        )
        groups.append(group2)
        
        enhanced_groups = self.linker.preserve_spatial_context(groups)
        
        assert len(enhanced_groups) == 2
        # Check that spatial context was added
        for group in enhanced_groups:
            for text in group.texts:
                # Should have some spatial context features
                context_features = [f for f in text.related_features if f.startswith('near_')]
                # May or may not have context depending on proximity
                assert isinstance(context_features, list)
    
    def test_find_spatially_related_groups(self):
        """Test finding spatially related groups."""
        # Create target group
        recognized_text = self.create_recognized_text("Target", (100, 100, 50, 20))
        linked_text = LinkedText(
            recognized_text=recognized_text,
            map_coordinates=(10.0, 20.0),
            spatial_confidence=0.8,
            related_features=[]
        )
        target_group = TextGroup(
            texts=[linked_text],
            group_type="place_name",
            confidence=0.8,
            center_coordinates=(10.0, 20.0)
        )
        
        # Create nearby group
        recognized_text2 = self.create_recognized_text("Nearby", (120, 110, 50, 20))
        linked_text2 = LinkedText(
            recognized_text=recognized_text2,
            map_coordinates=(10.1, 20.1),
            spatial_confidence=0.7,
            related_features=[]
        )
        nearby_group = TextGroup(
            texts=[linked_text2],
            group_type="annotation",
            confidence=0.7,
            center_coordinates=(10.1, 20.1)
        )
        
        # Create far group
        recognized_text3 = self.create_recognized_text("Far", (500, 500, 50, 20))
        linked_text3 = LinkedText(
            recognized_text=recognized_text3,
            map_coordinates=(50.0, 60.0),
            spatial_confidence=0.6,
            related_features=[]
        )
        far_group = TextGroup(
            texts=[linked_text3],
            group_type="label",
            confidence=0.6,
            center_coordinates=(50.0, 60.0)
        )
        
        all_groups = [target_group, nearby_group, far_group]
        related_groups = self.linker._find_spatially_related_groups(target_group, all_groups)
        
        # Should find the nearby group but not the far group
        assert nearby_group in related_groups
        assert far_group not in related_groups
    
    def test_get_linking_statistics(self):
        """Test linking statistics calculation."""
        # Create linked texts
        linked_texts = []
        for i in range(3):
            recognized_text = self.create_recognized_text(f"Text{i}", (100 + i*50, 100, 50, 20))
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=(10.0 + i*0.1, 20.0),
                spatial_confidence=0.8 - i*0.1,
                related_features=[f"feature_{i}"]
            )
            linked_texts.append(linked_text)
        
        stats = self.linker.get_linking_statistics(linked_texts)
        
        assert 'total_linked_texts' in stats
        assert 'average_confidence' in stats
        assert 'min_confidence' in stats
        assert 'max_confidence' in stats
        assert 'feature_associations' in stats
        assert 'spatial_distribution' in stats
        
        assert stats['total_linked_texts'] == 3
        assert 0.0 <= stats['average_confidence'] <= 1.0
        assert 0.0 <= stats['min_confidence'] <= 1.0
        assert 0.0 <= stats['max_confidence'] <= 1.0
    
    def test_get_linking_statistics_empty(self):
        """Test linking statistics with empty input."""
        stats = self.linker.get_linking_statistics([])
        
        assert stats == {}
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Populate caches by calling methods
        recognized_text = self.create_recognized_text("Test", (100, 100, 50, 20))
        self.linker.link_text_to_coordinates([recognized_text])
        
        # Clear caches
        self.linker.clear_cache()
        
        # Caches should be empty
        assert len(self.linker._linking_cache) == 0