"""
Evaluation module for historical map text processing system.
Provides competition formatting, metrics calculation, and comprehensive evaluation.
"""

from .interfaces import EvaluatorInterface, MetricsCalculatorInterface
from .competition_formatter import ICDAR2025Formatter
from .metrics_calculator import CompetitionMetricsCalculator
from .competition_evaluator import ICDAR2025Evaluator

__all__ = [
    'EvaluatorInterface',
    'MetricsCalculatorInterface', 
    'ICDAR2025Formatter',
    'CompetitionMetricsCalculator',
    'ICDAR2025Evaluator'
]