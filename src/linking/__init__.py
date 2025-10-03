"""
Text linking and spatial association module for historical map text processing.
"""

from .coordinate_mapper import CoordinateMapper
from .spatial_analyzer import SpatialAnalyzer
from .text_linker import TextLinker, TextGroup
from .interfaces import LinkingInterface

__all__ = [
    'CoordinateMapper',
    'SpatialAnalyzer',
    'TextLinker',
    'TextGroup',
    'LinkingInterface'
]