"""
Text linking and relationship detection for historical map text processing.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from .interfaces import LinkingInterface
from .coordinate_mapper import CoordinateMapper
from .spatial_analyzer import SpatialAnalyzer, SpatialFeature
from ..models.data_structures import RecognizedText, LinkedText, Detection


@dataclass
class TextGroup:
    """Represents a group of related text elements."""
    texts: List[LinkedText]
    group_type: str  # 'place_name', 'annotation', 'label', 'description'
    confidence: float
    center_coordinates: Tuple[float, float]


class TextLinker(LinkingInterface):
    """
    Main text linking system that combines coordinate mapping and spatial analysis.
    
    This class handles:
    - Linking recognized text to map coordinates
    - Grouping related text elements
    - Place name to location linking
    - Spatial context preservation
    """
    
    def __init__(self, 
                 coordinate_mapper: Optional[CoordinateMapper] = None,
                 spatial_analyzer: Optional[SpatialAnalyzer] = None,
                 image_size: Tuple[int, int] = (1000, 1000)):
        """
        Initialize text linker.
        
        Args:
            coordinate_mapper: Coordinate transformation utilities
            spatial_analyzer: Spatial analysis utilities
            image_size: Size of the image being processed
        """
        self.coordinate_mapper = coordinate_mapper or CoordinateMapper()
        self.spatial_analyzer = spatial_analyzer or SpatialAnalyzer()
        self.image_size = image_size
        self._place_name_patterns = self._initialize_place_name_patterns()
        self._linking_cache = {}
    
    def _initialize_place_name_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for identifying place names."""
        return {
            'city_indicators': ['city', 'town', 'village', 'settlement', 'capital'],
            'geographic_features': ['river', 'lake', 'mountain', 'hill', 'valley', 'bay', 'sea', 'ocean'],
            'administrative': ['county', 'province', 'state', 'region', 'district'],
            'directional': ['north', 'south', 'east', 'west', 'upper', 'lower', 'new', 'old'],
            'common_suffixes': ['burg', 'ville', 'ton', 'ford', 'field', 'wood', 'land', 'shire']
        }
    
    def set_image_size(self, image_size: Tuple[int, int]):
        """Set the image size for coordinate transformations."""
        self.image_size = image_size
        self._linking_cache.clear()
    
    def link_text_to_coordinates(self, recognized_texts: List[RecognizedText]) -> List[LinkedText]:
        """
        Link recognized text to map coordinates.
        
        Args:
            recognized_texts: List of recognized text elements
            
        Returns:
            List of linked text elements with spatial associations
        """
        linked_texts = []
        
        for recognized_text in recognized_texts:
            # Get pixel coordinates from detection bounding box
            bbox = recognized_text.detection.bbox
            pixel_center = (
                bbox.x + bbox.width // 2,
                bbox.y + bbox.height // 2
            )
            
            # Transform to map coordinates
            map_coordinates = self.coordinate_mapper.pixel_to_coordinate(
                pixel_center, self.image_size
            )
            
            # Calculate spatial confidence
            spatial_confidence = self.calculate_spatial_confidence(
                recognized_text, map_coordinates
            )
            
            # Find related features
            related_features = self._find_related_features(
                recognized_text, map_coordinates
            )
            
            # Create linked text
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=map_coordinates,
                spatial_confidence=spatial_confidence,
                related_features=related_features
            )
            
            linked_texts.append(linked_text)
        
        return linked_texts
    
    def calculate_spatial_confidence(self, text: RecognizedText, coordinates: Tuple[float, float]) -> float:
        """
        Calculate confidence score for spatial association.
        
        Args:
            text: Recognized text element
            coordinates: Map coordinates
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from coordinate mapper
        base_confidence = self.coordinate_mapper.calculate_spatial_confidence(
            text.detection, coordinates
        )
        
        # Enhance confidence based on text content
        content_boost = self._calculate_content_confidence_boost(text.text)
        
        # Enhance confidence based on recognition quality
        recognition_boost = text.recognition_confidence * 0.1
        
        # Combine confidence factors
        total_confidence = base_confidence + content_boost + recognition_boost
        
        return min(1.0, max(0.0, total_confidence))
    
    def _calculate_content_confidence_boost(self, text: str) -> float:
        """Calculate confidence boost based on text content."""
        text_lower = text.lower()
        boost = 0.0
        
        # Check for place name indicators
        for category, patterns in self._place_name_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    if category == 'city_indicators':
                        boost += 0.15
                    elif category == 'geographic_features':
                        boost += 0.12
                    elif category == 'administrative':
                        boost += 0.10
                    elif category == 'directional':
                        boost += 0.05
                    elif category == 'common_suffixes':
                        boost += 0.08
        
        # Check for proper nouns (capitalized words)
        words = text.split()
        capitalized_ratio = sum(1 for word in words if word and word[0].isupper()) / len(words) if words else 0
        boost += capitalized_ratio * 0.1
        
        # Check for numeric content (coordinates, dates, etc.)
        if any(char.isdigit() for char in text):
            boost += 0.05
        
        return min(0.3, boost)  # Cap the boost at 0.3
    
    def _find_related_features(self, text: RecognizedText, coordinates: Tuple[float, float]) -> List[str]:
        """Find map features related to the text."""
        # Define search radius based on text size and type
        bbox = text.detection.bbox
        text_size = max(bbox.width, bbox.height)
        
        # Larger search radius for larger text (likely more important)
        search_radius = (text_size / 100.0) * 2.0  # Adjust scaling as needed
        
        # Find nearby features
        nearby_features = self.spatial_analyzer.find_nearby_features(
            coordinates, search_radius
        )
        
        return nearby_features
    
    def group_related_texts(self, linked_texts: List[LinkedText]) -> List[TextGroup]:
        """
        Group related text elements based on spatial and content relationships.
        
        Args:
            linked_texts: List of linked text elements
            
        Returns:
            List of text groups
        """
        if not linked_texts:
            return []
        
        # Use spatial analyzer for initial grouping
        spatial_groups = self.spatial_analyzer.group_related_elements(linked_texts)
        
        # Enhance grouping with content analysis
        enhanced_groups = []
        
        for spatial_group in spatial_groups:
            # Analyze group content to determine type
            group_type = self._determine_group_type(spatial_group)
            
            # Calculate group confidence
            group_confidence = self._calculate_group_confidence(spatial_group)
            
            # Calculate group center
            center_coordinates = self._calculate_group_center(spatial_group)
            
            # Create text group
            text_group = TextGroup(
                texts=spatial_group,
                group_type=group_type,
                confidence=group_confidence,
                center_coordinates=center_coordinates
            )
            
            enhanced_groups.append(text_group)
        
        return enhanced_groups
    
    def _determine_group_type(self, texts: List[LinkedText]) -> str:
        """Determine the type of a text group based on content."""
        if not texts:
            return 'unknown'
        
        # Analyze text content
        all_text = ' '.join(text.recognized_text.text.lower() for text in texts)
        
        # Check for place name indicators
        place_indicators = 0
        for patterns in self._place_name_patterns.values():
            for pattern in patterns:
                if pattern in all_text:
                    place_indicators += 1
        
        if place_indicators >= 2:
            return 'place_name'
        
        # Check for annotation patterns
        if any(char.isdigit() for char in all_text) and len(texts) == 1:
            return 'annotation'
        
        # Check for descriptive text (longer texts)
        avg_length = sum(len(text.recognized_text.text) for text in texts) / len(texts)
        if avg_length > 10:
            return 'description'
        
        return 'label'
    
    def _calculate_group_confidence(self, texts: List[LinkedText]) -> float:
        """Calculate confidence score for a text group."""
        if not texts:
            return 0.0
        
        # Average spatial confidence
        avg_spatial_confidence = sum(text.spatial_confidence for text in texts) / len(texts)
        
        # Group coherence (how well texts fit together spatially)
        coherence_score = self._calculate_group_coherence(texts)
        
        # Size factor (larger groups may be more reliable)
        size_factor = min(len(texts) / 5.0, 1.0)  # Normalize by expected max group size
        
        # Combine factors
        group_confidence = (avg_spatial_confidence * 0.5 + 
                          coherence_score * 0.3 + 
                          size_factor * 0.2)
        
        return min(1.0, max(0.0, group_confidence))
    
    def _calculate_group_coherence(self, texts: List[LinkedText]) -> float:
        """Calculate how coherent a group of texts is spatially."""
        if len(texts) <= 1:
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                distance = self.spatial_analyzer.calculate_distance(
                    texts[i].map_coordinates,
                    texts[j].map_coordinates
                )
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        # Calculate coherence based on distance variance
        avg_distance = sum(distances) / len(distances)
        variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
        
        # Lower variance = higher coherence
        coherence = 1.0 / (1.0 + variance)
        
        return coherence
    
    def _calculate_group_center(self, texts: List[LinkedText]) -> Tuple[float, float]:
        """Calculate the center coordinates of a text group."""
        if not texts:
            return (0.0, 0.0)
        
        # Weighted average based on spatial confidence
        total_weight = sum(text.spatial_confidence for text in texts)
        
        if total_weight == 0:
            # Unweighted average if no confidence scores
            center_x = sum(text.map_coordinates[0] for text in texts) / len(texts)
            center_y = sum(text.map_coordinates[1] for text in texts) / len(texts)
        else:
            # Weighted average
            center_x = sum(text.map_coordinates[0] * text.spatial_confidence for text in texts) / total_weight
            center_y = sum(text.map_coordinates[1] * text.spatial_confidence for text in texts) / total_weight
        
        return (center_x, center_y)
    
    def link_place_names_to_locations(self, linked_texts: List[LinkedText]) -> List[LinkedText]:
        """
        Enhanced linking for place names to specific locations.
        
        Args:
            linked_texts: List of linked text elements
            
        Returns:
            List of linked texts with enhanced place name associations
        """
        enhanced_texts = []
        
        for linked_text in linked_texts:
            text_content = linked_text.recognized_text.text.lower()
            
            # Check if this looks like a place name
            is_place_name = self._is_likely_place_name(text_content)
            
            if is_place_name:
                # Enhance spatial confidence for place names
                enhanced_confidence = self.spatial_analyzer.calculate_spatial_confidence_score(
                    linked_text, 
                    self.spatial_analyzer.find_nearby_features_detailed(
                        linked_text.map_coordinates, 1.0
                    )
                )
                
                # Update spatial confidence
                enhanced_text = LinkedText(
                    recognized_text=linked_text.recognized_text,
                    map_coordinates=linked_text.map_coordinates,
                    spatial_confidence=enhanced_confidence,
                    related_features=linked_text.related_features + ['place_name']
                )
                
                enhanced_texts.append(enhanced_text)
            else:
                enhanced_texts.append(linked_text)
        
        return enhanced_texts
    
    def _is_likely_place_name(self, text: str) -> bool:
        """Determine if text is likely a place name."""
        text_lower = text.lower()
        
        # Check for place name indicators
        indicator_count = 0
        for patterns in self._place_name_patterns.values():
            for pattern in patterns:
                if pattern in text_lower:
                    indicator_count += 1
        
        # Check for proper noun characteristics
        words = text.split()
        if words:
            capitalized_ratio = sum(1 for word in words if word and word[0].isupper()) / len(words)
        else:
            capitalized_ratio = 0
        
        # Heuristic: likely place name if has indicators or mostly capitalized
        return indicator_count >= 1 or capitalized_ratio >= 0.5
    
    def preserve_spatial_context(self, text_groups: List[TextGroup]) -> List[TextGroup]:
        """
        Preserve spatial context relationships between text groups.
        
        Args:
            text_groups: List of text groups
            
        Returns:
            Text groups with enhanced spatial context preservation
        """
        if not text_groups:
            return []
        
        enhanced_groups = []
        
        for group in text_groups:
            # Find spatially related groups
            related_groups = self._find_spatially_related_groups(group, text_groups)
            
            # Add spatial context information to related features
            spatial_context = [f"near_{related_group.group_type}" for related_group in related_groups]
            
            # Update each text in the group
            enhanced_texts = []
            for text in group.texts:
                enhanced_text = LinkedText(
                    recognized_text=text.recognized_text,
                    map_coordinates=text.map_coordinates,
                    spatial_confidence=text.spatial_confidence,
                    related_features=text.related_features + spatial_context
                )
                enhanced_texts.append(enhanced_text)
            
            # Create enhanced group
            enhanced_group = TextGroup(
                texts=enhanced_texts,
                group_type=group.group_type,
                confidence=group.confidence,
                center_coordinates=group.center_coordinates
            )
            
            enhanced_groups.append(enhanced_group)
        
        return enhanced_groups
    
    def _find_spatially_related_groups(self, target_group: TextGroup, all_groups: List[TextGroup]) -> List[TextGroup]:
        """Find groups that are spatially related to the target group."""
        related_groups = []
        
        # Define proximity threshold based on group type
        if target_group.group_type == 'place_name':
            proximity_threshold = 2.0  # Larger threshold for place names
        else:
            proximity_threshold = 1.0
        
        for group in all_groups:
            if group == target_group:
                continue
            
            # Calculate distance between group centers
            distance = self.spatial_analyzer.calculate_distance(
                target_group.center_coordinates,
                group.center_coordinates
            )
            
            if distance <= proximity_threshold:
                related_groups.append(group)
        
        return related_groups
    
    def get_linking_statistics(self, linked_texts: List[LinkedText]) -> Dict[str, any]:
        """
        Get statistics about the linking process.
        
        Args:
            linked_texts: List of linked text elements
            
        Returns:
            Dictionary containing linking statistics
        """
        if not linked_texts:
            return {}
        
        # Calculate confidence statistics
        confidences = [text.spatial_confidence for text in linked_texts]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Count feature associations
        feature_counts = defaultdict(int)
        for text in linked_texts:
            for feature in text.related_features:
                feature_counts[feature] += 1
        
        # Analyze text distribution
        distribution_stats = self.spatial_analyzer.analyze_text_distribution(linked_texts)
        
        return {
            'total_linked_texts': len(linked_texts),
            'average_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'feature_associations': dict(feature_counts),
            'spatial_distribution': distribution_stats
        }
    
    def clear_cache(self):
        """Clear internal caches."""
        self._linking_cache.clear()
        self.coordinate_mapper.clear_cache()
        self.spatial_analyzer.clear_cache()