"""
REST API package for historical map text processing system.
"""

from .main_api import create_app, HistoricalMapAPI

__all__ = ['create_app', 'HistoricalMapAPI']