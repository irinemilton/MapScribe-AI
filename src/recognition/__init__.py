"""
Text recognition components for OCR and multi-language support.
"""

from .interfaces import TextRecognizerInterface, RecognitionModelInterface
from .base_recognizer import BaseTextRecognizer, TextPreprocessor, CharacterConfidenceTracker
from .crnn_recognizer import (
    CRNNModel, BeamSearchDecoder, CRNNRecognitionModel, SingleLanguageCRNNRecognizer
)
from .multilingual_recognizer import (
    LanguageDetector, ChineseRecognitionModel, MultilingualTextRecognizer
)
from .accuracy_metrics import RecognitionAccuracyMetrics

__all__ = [
    'TextRecognizerInterface',
    'RecognitionModelInterface', 
    'BaseTextRecognizer',
    'TextPreprocessor',
    'CharacterConfidenceTracker',
    'CRNNModel',
    'BeamSearchDecoder', 
    'CRNNRecognitionModel',
    'SingleLanguageCRNNRecognizer',
    'LanguageDetector',
    'ChineseRecognitionModel',
    'MultilingualTextRecognizer',
    'RecognitionAccuracyMetrics'
]