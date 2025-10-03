"""
Spatial analysis utilities for text linking and relationship detection.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from .interfaces import SpatialAnalyzerInterface
from ..models.data_structures import LinkedText, Detection, RecognizedText


@dataclass
class SpatialFeature:
    """Represents a spatial feature on the map."""
    name: str
    coordinates: Tuple[float, float]
    feature_type: str  # 'city', 'river', 'mountain', 'border', etc.
    confidence: float


class SpatialAnalyzer(SpatialAnalyzerInterface):
    """
    Analyzes spatial relationships and provides utilities for text linking.
    
    This class handles:
    - Distance calculations between coordinates
    - Finding nearby map features
    - Grouping spatially related text elements
    - Analyzing spatial patterns in text distribution
    """
    
    def __init__(self, map_features: Optional[List[SpatialFeature]] = None):
        """
        Initialize spatial analyzer.
        
        Args:
            map_features: List of known map features for spatial context
        """
        self.map_features = map_features or []
        self._feature_index = self._build_feature_index()
        self._distance_cache = {}
    
    def add_map_feature(self, feature: SpatialFeature):
        """Add a map feature to the spatial context."""
        self.map_features.append(feature)
        self._feature_index = self._build_feature_index()
    
    def _build_feature_index(self) -> Dict[str, List[SpatialFeature]]:
        """Build spatial index for efficient feature lookup."""
        index = defaultdict(list)
        for feature in self.map_features:
            # Create grid-based spatial index
            grid_x = int(feature.coordinates[0] * 10)  # 0.1 degree grid
            grid_y = int(feature.coordinates[1] * 10)
            grid_key = f"{grid_x},{grid_y}"
            index[grid_key].append(feature)
        return dict(index)
    
    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate distance between two coordinates.
        
        Args:
            coord1: First coordinate (longitude, latitude)
            coord2: Second coordinate (longitude, latitude)
            
        Returns:
            Distance in appropriate units (pixels for image coordinates, 
            degrees for geographic coordinates)
        """
        cache_key = (coord1, coord2)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        x1, y1 = coord1
        x2, y2 = coord2
        
        # Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Cache the result
        self._distance_cache[cache_key] = distance
        
        return distance
    
    def calculate_geographic_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate geographic distance using Haversine formula for lat/lon coordinates.
        
        Args:
            coord1: First coordinate (longitude, latitude) in degrees
            coord2: Second coordinate (longitude, latitude) in degrees
            
        Returns:
            Distance in kilometers
        """
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius = 6371.0
        
        return earth_radius * c
    
    def find_nearby_features(self, coordinates: Tuple[float, float], radius: float) -> List[str]:
        """
        Find map features near given coordinates.
        
        Args:
            coordinates: Target coordinates (longitude, latitude)
            radius: Search radius
            
        Returns:
            List of feature names within the radius
        """
        nearby_features = []
        
        for feature in self.map_features:
            distance = self.calculate_distance(coordinates, feature.coordinates)
            if distance <= radius:
                nearby_features.append(feature.name)
        
        return nearby_features
    
    def find_nearby_features_detailed(self, coordinates: Tuple[float, float], radius: float) -> List[SpatialFeature]:
        """
        Find detailed map features near given coordinates.
        
        Args:
            coordinates: Target coordinates (longitude, latitude)
            radius: Search radius
            
        Returns:
            List of SpatialFeature objects within the radius
        """
        nearby_features = []
        
        # For small datasets, just check all features (more reliable than spatial indexing)
        for feature in self.map_features:
            distance = self.calculate_distance(coordinates, feature.coordinates)
            if distance <= radius:
                nearby_features.append(feature)
        
        # Sort by distance
        nearby_features.sort(key=lambda f: self.calculate_distance(coordinates, f.coordinates))
        
        return nearby_features
    
    def group_related_elements(self, linked_texts: List[LinkedText]) -> List[List[LinkedText]]:
        """
        Group spatially related text elements.
        
        Args:
            linked_texts: List of linked text elements
            
        Returns:
            List of groups, where each group contains related text elements
        """
        if not linked_texts:
            return []
        
        groups = []
        processed = set()
        
        for i, text in enumerate(linked_texts):
            if i in processed:
                continue
            
            group = [text]
            processed.add(i)
            
            # Find related texts based on spatial proximity and content similarity
            for j, other_text in enumerate(linked_texts):
                if j in processed:
                    continue
                
                if self._are_texts_related(text, other_text):
                    group.append(other_text)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_texts_related(self, text1: LinkedText, text2: LinkedText) -> bool:
        """
        Determine if two text elements are spatially related.
        
        Args:
            text1: First text element
            text2: Second text element
            
        Returns:
            True if texts are considered related
        """
        # Calculate spatial distance
        distance = self.calculate_distance(text1.map_coordinates, text2.map_coordinates)
        
        # Define proximity threshold based on text size and type
        bbox1 = text1.recognized_text.detection.bbox
        bbox2 = text2.recognized_text.detection.bbox
        
        avg_size = (max(bbox1.width, bbox1.height) + max(bbox2.width, bbox2.height)) / 2
        proximity_threshold = avg_size * 3  # Adjust based on requirements
        
        if distance > proximity_threshold:
            return False
        
        # Check content similarity for additional context
        content_similarity = self._calculate_content_similarity(
            text1.recognized_text.text, 
            text2.recognized_text.text
        )
        
        # Consider texts related if they are close and have some content similarity
        return distance <= proximity_threshold and content_similarity > 0.3
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate content similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple similarity based on common words and characters
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            # Character-level similarity for short texts
            chars1 = set(text1.lower())
            chars2 = set(text2.lower())
            
            if not chars1 or not chars2:
                return 0.0
            
            intersection = len(chars1.intersection(chars2))
            union = len(chars1.union(chars2))
            return intersection / union
        
        # Word-level similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def analyze_text_distribution(self, linked_texts: List[LinkedText]) -> Dict[str, any]:
        """
        Analyze the spatial distribution of text elements.
        
        Args:
            linked_texts: List of linked text elements
            
        Returns:
            Dictionary containing distribution analysis results
        """
        if not linked_texts:
            return {}
        
        coordinates = [text.map_coordinates for text in linked_texts]
        
        # Calculate center of mass
        center_x = sum(coord[0] for coord in coordinates) / len(coordinates)
        center_y = sum(coord[1] for coord in coordinates) / len(coordinates)
        center_of_mass = (center_x, center_y)
        
        # Calculate spread (standard deviation)
        distances_from_center = [
            self.calculate_distance(coord, center_of_mass) 
            for coord in coordinates
        ]
        
        mean_distance = sum(distances_from_center) / len(distances_from_center)
        variance = sum((d - mean_distance)**2 for d in distances_from_center) / len(distances_from_center)
        spread = math.sqrt(variance)
        
        # Find bounding box
        min_x = min(coord[0] for coord in coordinates)
        max_x = max(coord[0] for coord in coordinates)
        min_y = min(coord[1] for coord in coordinates)
        max_y = max(coord[1] for coord in coordinates)
        
        # Calculate density (texts per unit area)
        area = (max_x - min_x) * (max_y - min_y)
        density = len(linked_texts) / area if area > 0 else 0
        
        return {
            'center_of_mass': center_of_mass,
            'spread': spread,
            'bounding_box': {
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y
            },
            'density': density,
            'total_texts': len(linked_texts),
            'mean_distance_from_center': mean_distance
        }
    
    def find_text_clusters(self, linked_texts: List[LinkedText], cluster_radius: float) -> List[List[LinkedText]]:
        """
        Find clusters of text elements using spatial proximity.
        
        Args:
            linked_texts: List of linked text elements
            cluster_radius: Maximum distance for elements to be in same cluster
            
        Returns:
            List of clusters, each containing related text elements
        """
        if not linked_texts:
            return []
        
        clusters = []
        processed = set()
        
        for i, text in enumerate(linked_texts):
            if i in processed:
                continue
            
            # Start new cluster
            cluster = [text]
            processed.add(i)
            
            # Find all texts within cluster radius
            cluster_coords = [text.map_coordinates]
            
            # Iteratively expand cluster
            changed = True
            while changed:
                changed = False
                for j, other_text in enumerate(linked_texts):
                    if j in processed:
                        continue
                    
                    # Check if text is close to any text in current cluster
                    for cluster_coord in cluster_coords:
                        distance = self.calculate_distance(other_text.map_coordinates, cluster_coord)
                        if distance <= cluster_radius:
                            cluster.append(other_text)
                            cluster_coords.append(other_text.map_coordinates)
                            processed.add(j)
                            changed = True
                            break
            
            clusters.append(cluster)
        
        return clusters
    
    def calculate_spatial_confidence_score(self, linked_text: LinkedText, context_features: List[SpatialFeature]) -> float:
        """
        Calculate enhanced spatial confidence score considering map context.
        
        Args:
            linked_text: Text element with spatial linking
            context_features: Nearby map features for context
            
        Returns:
            Enhanced confidence score between 0.0 and 1.0
        """
        base_confidence = linked_text.spatial_confidence
        
        # Boost confidence if near known features
        feature_boost = 0.0
        if context_features:
            min_distance = min(
                self.calculate_distance(linked_text.map_coordinates, feature.coordinates)
                for feature in context_features
            )
            
            # Closer to features = higher confidence
            if min_distance < 0.1:  # Very close
                feature_boost = 0.2
            elif min_distance < 0.5:  # Moderately close
                feature_boost = 0.1
        
        # Consider text content for place names
        text_content = linked_text.recognized_text.text.lower()
        content_boost = 0.0
        
        # Common place name indicators
        place_indicators = ['city', 'town', 'river', 'mountain', 'lake', 'sea', 'ocean']
        if any(indicator in text_content for indicator in place_indicators):
            content_boost = 0.1
        
        # Combine confidence factors
        enhanced_confidence = base_confidence + feature_boost + content_boost
        
        return min(1.0, enhanced_confidence)
    
    def clear_cache(self):
        """Clear internal caches."""
        self._distance_cache.clear()