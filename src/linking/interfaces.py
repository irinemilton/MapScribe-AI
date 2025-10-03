"""
Interfaces for text linking and spatial association components.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..models.data_structures import RecognizedText, LinkedText, Detection


class LinkingInterface(ABC):
    """Base interface for text linking operations."""
    
    @abstractmethod
    def link_text_to_coordinates(self, recognized_texts: List[RecognizedText]) -> List[LinkedText]:
        """Link recognized text to map coordinates."""
        pass
    
    @abstractmethod
    def calculate_spatial_confidence(self, text: RecognizedText, coordinates: Tuple[float, float]) -> float:
        """Calculate confidence score for spatial association."""
        pass


class CoordinateMapperInterface(ABC):
    """Interface for coordinate mapping operations."""
    
    @abstractmethod
    def pixel_to_coordinate(self, pixel_coords: Tuple[int, int], image_size: Tuple[int, int]) -> Tuple[float, float]:
        """Transform pixel coordinates to map coordinates."""
        pass
    
    @abstractmethod
    def coordinate_to_pixel(self, map_coords: Tuple[float, float], image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Transform map coordinates to pixel coordinates."""
        pass
    
    @abstractmethod
    def analyze_spatial_relationships(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Analyze spatial relationships between text elements."""
        pass


class SpatialAnalyzerInterface(ABC):
    """Interface for spatial analysis operations."""
    
    @abstractmethod
    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates."""
        pass
    
    @abstractmethod
    def find_nearby_features(self, coordinates: Tuple[float, float], radius: float) -> List[str]:
        """Find map features near given coordinates."""
        pass
    
    @abstractmethod
    def group_related_elements(self, linked_texts: List[LinkedText]) -> List[List[LinkedText]]:
        """Group spatially related text elements."""
        pass