"""
Coordinate mapping utilities for transforming between pixel and map coordinates.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass

from .interfaces import CoordinateMapperInterface
from ..models.data_structures import Detection, RecognizedText


@dataclass
class CoordinateTransform:
    """Configuration for coordinate transformation."""
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    rotation_angle: float = 0.0  # in radians
    
    def __post_init__(self):
        """Validate transformation parameters."""
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("Scale factors must be positive")


class CoordinateMapper(CoordinateMapperInterface):
    """
    Handles pixel-to-coordinate transformations and spatial relationship analysis.
    
    This class provides utilities for:
    - Converting between pixel coordinates and map coordinates
    - Analyzing spatial relationships between text elements
    - Calculating confidence scores for spatial associations
    """
    
    def __init__(self, transform: Optional[CoordinateTransform] = None):
        """
        Initialize coordinate mapper.
        
        Args:
            transform: Coordinate transformation configuration
        """
        self.transform = transform or CoordinateTransform(
            scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0
        )
        self._spatial_cache = {}
    
    def set_transform(self, transform: CoordinateTransform):
        """Set coordinate transformation parameters."""
        self.transform = transform
        self._spatial_cache.clear()  # Clear cache when transform changes
    
    def pixel_to_coordinate(self, pixel_coords: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Transform pixel coordinates to map coordinates.
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            image_size: (width, height) of the image
            
        Returns:
            Tuple of (longitude, latitude) map coordinates
        """
        px, py = pixel_coords
        img_width, img_height = image_size
        
        # Normalize pixel coordinates to [0, 1]
        norm_x = px / img_width
        norm_y = py / img_height
        
        # Apply rotation if specified
        if self.transform.rotation_angle != 0:
            cos_angle = math.cos(self.transform.rotation_angle)
            sin_angle = math.sin(self.transform.rotation_angle)
            
            # Rotate around center
            center_x, center_y = 0.5, 0.5
            norm_x -= center_x
            norm_y -= center_y
            
            rotated_x = norm_x * cos_angle - norm_y * sin_angle
            rotated_y = norm_x * sin_angle + norm_y * cos_angle
            
            norm_x = rotated_x + center_x
            norm_y = rotated_y + center_y
        
        # Apply scale and offset transformation
        map_x = norm_x * self.transform.scale_x + self.transform.offset_x
        map_y = norm_y * self.transform.scale_y + self.transform.offset_y
        
        return (map_x, map_y)
    
    def coordinate_to_pixel(self, map_coords: Tuple[float, float], image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Transform map coordinates to pixel coordinates.
        
        Args:
            map_coords: (longitude, latitude) map coordinates
            image_size: (width, height) of the image
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        map_x, map_y = map_coords
        img_width, img_height = image_size
        
        # Reverse scale and offset transformation
        norm_x = (map_x - self.transform.offset_x) / self.transform.scale_x
        norm_y = (map_y - self.transform.offset_y) / self.transform.scale_y
        
        # Apply reverse rotation if specified
        if self.transform.rotation_angle != 0:
            cos_angle = math.cos(-self.transform.rotation_angle)  # Negative for reverse
            sin_angle = math.sin(-self.transform.rotation_angle)
            
            # Rotate around center
            center_x, center_y = 0.5, 0.5
            norm_x -= center_x
            norm_y -= center_y
            
            rotated_x = norm_x * cos_angle - norm_y * sin_angle
            rotated_y = norm_x * sin_angle + norm_y * cos_angle
            
            norm_x = rotated_x + center_x
            norm_y = rotated_y + center_y
        
        # Convert normalized coordinates to pixels
        px = int(norm_x * img_width)
        py = int(norm_y * img_height)
        
        # Clamp to image bounds
        px = max(0, min(px, img_width - 1))
        py = max(0, min(py, img_height - 1))
        
        return (px, py)
    
    def analyze_spatial_relationships(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """
        Analyze spatial relationships between text elements.
        
        Args:
            detections: List of text detections
            
        Returns:
            Dictionary mapping relationship types to detection groups
        """
        if not detections:
            return {}
        
        relationships = {
            'horizontal_aligned': [],
            'vertical_aligned': [],
            'clustered': [],
            'isolated': []
        }
        
        # Calculate centers for all detections
        centers = []
        for detection in detections:
            bbox = detection.bbox
            center_x = bbox.x + bbox.width // 2
            center_y = bbox.y + bbox.height // 2
            centers.append((center_x, center_y))
        
        # Analyze relationships
        for i, detection in enumerate(detections):
            center = centers[i]
            related_detections = []
            
            # Find horizontally aligned elements
            horizontal_threshold = detection.bbox.height * 0.5
            for j, other_detection in enumerate(detections):
                if i != j:
                    other_center = centers[j]
                    if abs(center[1] - other_center[1]) <= horizontal_threshold:
                        related_detections.append(other_detection)
            
            if related_detections:
                relationships['horizontal_aligned'].extend([detection] + related_detections)
            
            # Find vertically aligned elements
            related_detections = []
            vertical_threshold = detection.bbox.width * 0.5
            for j, other_detection in enumerate(detections):
                if i != j:
                    other_center = centers[j]
                    if abs(center[0] - other_center[0]) <= vertical_threshold:
                        related_detections.append(other_detection)
            
            if related_detections:
                relationships['vertical_aligned'].extend([detection] + related_detections)
            
            # Find clustered elements (within proximity)
            related_detections = []
            cluster_threshold = max(detection.bbox.width, detection.bbox.height) * 2
            for j, other_detection in enumerate(detections):
                if i != j:
                    other_center = centers[j]
                    distance = math.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                    if distance <= cluster_threshold:
                        related_detections.append(other_detection)
            
            if related_detections:
                relationships['clustered'].extend([detection] + related_detections)
            else:
                relationships['isolated'].append(detection)
        
        # Remove duplicates (Detection objects are not hashable, so use manual deduplication)
        for key in relationships:
            unique_detections = []
            seen_ids = set()
            for detection in relationships[key]:
                detection_id = id(detection)  # Use object id for uniqueness
                if detection_id not in seen_ids:
                    unique_detections.append(detection)
                    seen_ids.add(detection_id)
            relationships[key] = unique_detections
        
        return relationships
    
    def calculate_spatial_confidence(self, detection: Detection, map_coordinates: Tuple[float, float]) -> float:
        """
        Calculate confidence score for spatial association between detection and coordinates.
        
        Args:
            detection: Text detection
            map_coordinates: Target map coordinates
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Get detection center in pixel coordinates
        bbox = detection.bbox
        detection_center = (bbox.x + bbox.width // 2, bbox.y + bbox.height // 2)
        
        # Convert map coordinates to pixel coordinates (assuming standard image size)
        # This would need actual image size in real implementation
        estimated_image_size = (1000, 1000)  # Default assumption
        pixel_coords = self.coordinate_to_pixel(map_coordinates, estimated_image_size)
        
        # Calculate distance between detection center and target coordinates
        distance = math.sqrt(
            (detection_center[0] - pixel_coords[0])**2 + 
            (detection_center[1] - pixel_coords[1])**2
        )
        
        # Normalize distance based on detection size
        max_distance = max(bbox.width, bbox.height) * 2
        normalized_distance = min(distance / max_distance, 1.0)
        
        # Calculate confidence (inverse of normalized distance)
        base_confidence = 1.0 - normalized_distance
        
        # Adjust confidence based on detection confidence
        spatial_confidence = base_confidence * detection.confidence
        
        # Apply additional factors
        # Higher confidence for larger text (more likely to be important)
        size_factor = min(bbox.area / 10000, 1.0)  # Normalize by typical text area
        spatial_confidence *= (0.8 + 0.2 * size_factor)
        
        return max(0.0, min(1.0, spatial_confidence))
    
    def get_bounding_box_center(self, detection: Detection) -> Tuple[int, int]:
        """Get the center coordinates of a detection's bounding box."""
        bbox = detection.bbox
        center_x = bbox.x + bbox.width // 2
        center_y = bbox.y + bbox.height // 2
        return (center_x, center_y)
    
    def calculate_overlap_ratio(self, detection1: Detection, detection2: Detection) -> float:
        """
        Calculate the overlap ratio between two detections.
        
        Args:
            detection1: First detection
            detection2: Second detection
            
        Returns:
            Overlap ratio between 0.0 and 1.0
        """
        bbox1 = detection1.bbox
        bbox2 = detection2.bbox
        
        # Calculate intersection
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        union_area = bbox1.area + bbox2.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def find_text_alignment(self, detections: List[Detection], alignment_type: str = 'horizontal') -> List[List[Detection]]:
        """
        Find groups of text elements that are aligned horizontally or vertically.
        
        Args:
            detections: List of text detections
            alignment_type: 'horizontal' or 'vertical'
            
        Returns:
            List of detection groups that are aligned
        """
        if not detections:
            return []
        
        aligned_groups = []
        processed = set()
        
        for i, detection in enumerate(detections):
            if i in processed:
                continue
            
            group = [detection]
            processed.add(i)
            
            bbox = detection.bbox
            center = self.get_bounding_box_center(detection)
            
            # Define alignment threshold based on detection size
            if alignment_type == 'horizontal':
                threshold = bbox.height * 0.5
                coord_index = 1  # y-coordinate
            else:  # vertical
                threshold = bbox.width * 0.5
                coord_index = 0  # x-coordinate
            
            # Find aligned detections
            for j, other_detection in enumerate(detections):
                if j in processed:
                    continue
                
                other_center = self.get_bounding_box_center(other_detection)
                
                if abs(center[coord_index] - other_center[coord_index]) <= threshold:
                    group.append(other_detection)
                    processed.add(j)
            
            if len(group) > 1:  # Only include groups with multiple elements
                aligned_groups.append(group)
        
        return aligned_groups
    
    def estimate_reading_order(self, detections: List[Detection]) -> List[Detection]:
        """
        Estimate the reading order of text detections based on spatial layout.
        
        Args:
            detections: List of text detections
            
        Returns:
            Detections sorted in estimated reading order
        """
        if not detections:
            return []
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        def sort_key(detection):
            bbox = detection.bbox
            center = self.get_bounding_box_center(detection)
            # Use row-based sorting with some tolerance for y-coordinate
            row = center[1] // (bbox.height + 10)  # Group into rows
            return (row, center[0])
        
        return sorted(detections, key=sort_key)
    
    def clear_cache(self):
        """Clear internal spatial analysis cache."""
        self._spatial_cache.clear()