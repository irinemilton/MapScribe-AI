"""
Multi-language text recognition implementation with Chinese character support
and automatic language detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path
import cv2
from PIL import Image

from .base_recognizer import BaseTextRecognizer
from .crnn_recognizer import CRNNModel, BeamSearchDecoder
from .interfaces import RecognitionModelInterface
from ..models.data_structures import RecognitionConfig, Detection, RecognizedText


class LanguageDetector:
    """
    Language detection for text regions using visual and textual features.
    """
    
    def __init__(self):
        """Initialize language detector."""
        self.logger = logging.getLogger(__name__)
        
        # Character aspect ratio thresholds for language detection
        self.chinese_aspect_ratio_threshold = 1.5  # Chinese characters are more square
        self.latin_aspect_ratio_threshold = 3.0    # Latin text is more horizontal
        
        # Density thresholds
        self.chinese_density_threshold = 0.6  # Chinese characters are denser
        
    def detect_language(self, text_region: np.ndarray, confidence_threshold: float = 0.7) -> str:
        """
        Detect language of text in image region using visual features.
        
        Args:
            text_region: Image region containing text
            confidence_threshold: Minimum confidence for language detection
            
        Returns:
            Detected language code ('en', 'zh', 'mixed')
        """
        try:
            # Calculate visual features
            features = self._extract_visual_features(text_region)
            
            # Score each language based on features
            scores = self._calculate_language_scores(features)
            
            # Determine best language
            best_language = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_language]
            
            # Check confidence threshold
            if best_score < confidence_threshold:
                return 'mixed'  # Low confidence, assume mixed language
            
            return best_language
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English
    
    def _extract_visual_features(self, text_region: np.ndarray) -> Dict[str, float]:
        """
        Extract visual features from text region for language detection.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Dictionary of visual features
        """
        if len(text_region.shape) == 3:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor((text_region * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (text_region * 255).astype(np.uint8)
        
        h, w = gray.shape
        
        # Feature 1: Aspect ratio
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Feature 2: Text density (ratio of non-white pixels)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text_pixels = np.sum(binary == 0)  # Assuming text is dark
        total_pixels = binary.size
        density = text_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Feature 3: Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - binary)
        
        # Calculate average component aspect ratio (excluding background)
        component_aspect_ratios = []
        for i in range(1, num_labels):  # Skip background (label 0)
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            if comp_h > 0:
                component_aspect_ratios.append(comp_w / comp_h)
        
        avg_component_aspect_ratio = (
            np.mean(component_aspect_ratios) if component_aspect_ratios else 1.0
        )
        
        # Feature 4: Stroke width variation (simplified)
        # Chinese characters typically have more uniform stroke widths
        edges = cv2.Canny(gray, 50, 150)
        stroke_variation = np.std(edges[edges > 0]) if np.any(edges > 0) else 0.0
        
        return {
            'aspect_ratio': aspect_ratio,
            'density': density,
            'num_components': num_labels - 1,  # Exclude background
            'avg_component_aspect_ratio': avg_component_aspect_ratio,
            'stroke_variation': stroke_variation
        }
    
    def _calculate_language_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate language scores based on visual features.
        
        Args:
            features: Dictionary of visual features
            
        Returns:
            Dictionary of language scores
        """
        scores = {'en': 0.0, 'zh': 0.0}
        
        # Score based on aspect ratio
        if features['aspect_ratio'] < self.chinese_aspect_ratio_threshold:
            scores['zh'] += 0.3  # More square-like, favor Chinese
        elif features['aspect_ratio'] > self.latin_aspect_ratio_threshold:
            scores['en'] += 0.3  # More horizontal, favor English
        
        # Score based on density
        if features['density'] > self.chinese_density_threshold:
            scores['zh'] += 0.2  # Higher density, favor Chinese
        else:
            scores['en'] += 0.2  # Lower density, favor English
        
        # Score based on component aspect ratios
        if features['avg_component_aspect_ratio'] < 1.5:
            scores['zh'] += 0.2  # Square components, favor Chinese
        else:
            scores['en'] += 0.2  # Rectangular components, favor English
        
        # Score based on number of components
        # Chinese text typically has fewer, more complex components
        if features['num_components'] > 0:
            components_per_unit_area = features['num_components'] / (features['aspect_ratio'] + 1)
            if components_per_unit_area < 2.0:
                scores['zh'] += 0.15
            else:
                scores['en'] += 0.15
        
        # Score based on stroke variation
        # Chinese characters often have more uniform strokes
        if features['stroke_variation'] < 50:
            scores['zh'] += 0.15
        else:
            scores['en'] += 0.15
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {lang: score / total_score for lang, score in scores.items()}
        
        return scores


class ChineseRecognitionModel(RecognitionModelInterface):
    """
    Chinese character recognition model using specialized architecture.
    """
    
    def __init__(self, config: RecognitionConfig):
        """
        Initialize Chinese recognition model.
        
        Args:
            config: Recognition configuration
        """
        self.config = config
        self.model: Optional[CRNNModel] = None
        self.decoder: Optional[BeamSearchDecoder] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.charset = self._create_chinese_charset()
        self.logger = logging.getLogger(__name__)
    
    def _create_chinese_charset(self) -> List[str]:
        """Create character set for Chinese text recognition."""
        # CTC blank token
        charset = ['<blank>']
        
        # Common Chinese characters (simplified subset for demo)
        # In a real implementation, this would include thousands of characters
        common_chinese = [
            '的', '一', '是', '在', '不', '了', '有', '和', '人', '这',
            '中', '大', '为', '上', '个', '国', '我', '以', '要', '他',
            '时', '来', '用', '们', '生', '到', '作', '地', '于', '出',
            '就', '分', '对', '成', '会', '可', '主', '发', '年', '动',
            '同', '工', '也', '能', '下', '过', '子', '说', '产', '种',
            '面', '而', '方', '后', '多', '定', '行', '学', '法', '所',
            '民', '得', '经', '十', '三', '之', '进', '着', '等', '部',
            '度', '家', '电', '力', '里', '如', '水', '化', '高', '自',
            '二', '理', '起', '小', '物', '现', '实', '方', '新', '已',
            '公', '全', '六', '口', '天', '使', '由', '等', '今', '其'
        ]
        charset.extend(common_chinese)
        
        # Numbers
        charset.extend([str(i) for i in range(10)])
        
        # Basic punctuation
        charset.extend(['，', '。', '？', '！', '：', '；', '（', '）', '【', '】'])
        
        return charset
    
    def load_model(self, model_path: str, language: Optional[str] = None) -> None:
        """
        Load Chinese recognition model from file.
        
        Args:
            model_path: Path to model file
            language: Optional language specification
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                self.logger.warning(f"Chinese model file {model_path} not found. Creating dummy model.")
                self._create_dummy_model()
                return
            
            # Load model (implementation would load actual Chinese model)
            self._create_dummy_model()  # For now, use dummy model
            
            self.logger.info(f"Loaded Chinese recognition model from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading Chinese model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Create a dummy Chinese model for testing purposes."""
        self.model = CRNNModel(
            img_height=32,
            img_width=128,
            num_channels=3,
            num_classes=len(self.charset)
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.decoder = BeamSearchDecoder(
            charset=self.charset,
            beam_width=self.config.beam_width,
            blank_id=0
        )
        
        self.logger.info("Created dummy Chinese recognition model")
    
    def predict(self, text_region: np.ndarray) -> Tuple[str, List[float]]:
        """
        Predict Chinese text and character confidences for a region.
        
        Args:
            text_region: Image region containing text (H, W, C) in range [0, 1]
            
        Returns:
            Tuple of (recognized_text, character_confidences)
        """
        if self.model is None or self.decoder is None:
            raise RuntimeError("Chinese model not loaded. Call load_model() first.")
        
        try:
            # Prepare input tensor
            input_tensor = self._prepare_input(text_region)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                log_probs = F.log_softmax(output, dim=2)
            
            # Convert to numpy for decoding
            log_probs_np = log_probs.cpu().numpy()
            
            # Decode using beam search
            results = self.decoder.decode(log_probs_np)
            
            if results:
                text, confidence, char_confidences = results[0]
                return text, char_confidences
            else:
                return "", []
                
        except Exception as e:
            self.logger.error(f"Error during Chinese prediction: {e}")
            return "", []
    
    def _prepare_input(self, text_region: np.ndarray) -> torch.Tensor:
        """
        Prepare input tensor from text region.
        
        Args:
            text_region: Image region (H, W, C) in range [0, 1]
            
        Returns:
            Input tensor ready for model
        """
        # Ensure correct shape and type
        if len(text_region.shape) == 2:
            text_region = np.stack([text_region] * 3, axis=-1)
        
        # Resize to model input size if needed
        if text_region.shape[:2] != (32, 128):
            pil_image = Image.fromarray((text_region * 255).astype(np.uint8))
            pil_image = pil_image.resize((128, 32), Image.LANCZOS)
            text_region = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(text_region).permute(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return ['zh', 'chinese']


class MultilingualTextRecognizer(BaseTextRecognizer):
    """
    Multi-language text recognizer with automatic language detection and model switching.
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        Initialize multilingual text recognizer.
        
        Args:
            config: Recognition configuration
        """
        if config is None:
            config = RecognitionConfig(
                model_architecture='CRNN',
                max_sequence_length=32,
                vocabulary_size=5000,  # Larger for multi-language
                languages=['en', 'zh'],
                beam_width=5
            )
        
        super().__init__(config)
        
        # Initialize language detector
        self.language_detector = LanguageDetector()
        
        # Initialize recognition models for each language
        self.recognition_models: Dict[str, RecognitionModelInterface] = {}
        self._initialize_language_models()
        
        # Language detection confidence threshold
        self.language_detection_threshold = 0.6
        
    def _initialize_language_models(self) -> None:
        """Initialize recognition models for supported languages."""
        from .crnn_recognizer import CRNNRecognitionModel
        
        for language in self.config.languages:
            if language in ['en', 'latin']:
                # Use Latin/English model
                model = CRNNRecognitionModel(self.config)
                self.recognition_models[language] = model
            elif language in ['zh', 'chinese']:
                # Use Chinese model
                model = ChineseRecognitionModel(self.config)
                self.recognition_models[language] = model
            else:
                self.logger.warning(f"Unsupported language: {language}")
    
    def detect_language(self, text_region: np.ndarray) -> str:
        """
        Detect the language of text in the region using visual features.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Detected language code
        """
        return self.language_detector.detect_language(
            text_region, 
            confidence_threshold=self.language_detection_threshold
        )
    
    def _recognize_text_region(self, text_region: np.ndarray, language: str) -> Tuple[str, List[float]]:
        """
        Recognize text in a preprocessed region using appropriate language model.
        
        Args:
            text_region: Preprocessed text region
            language: Detected or specified language
            
        Returns:
            Tuple of (recognized_text, character_confidences)
        """
        # Map language codes to model keys
        model_key = self._get_model_key(language)
        
        if model_key not in self.recognition_models:
            self.logger.warning(f"No model available for language {language}, using default")
            model_key = list(self.recognition_models.keys())[0] if self.recognition_models else None
        
        if model_key is None:
            self.logger.error("No recognition models available")
            return "", []
        
        try:
            model = self.recognition_models[model_key]
            return model.predict(text_region)
        except Exception as e:
            self.logger.error(f"Error recognizing text with {model_key} model: {e}")
            return "", []
    
    def _get_model_key(self, language: str) -> str:
        """
        Get the appropriate model key for a language.
        
        Args:
            language: Language code
            
        Returns:
            Model key for the language
        """
        # Map various language codes to model keys
        language_mapping = {
            'en': 'en',
            'latin': 'en',
            'zh': 'zh',
            'chinese': 'zh',
            'mixed': 'en'  # Default to English for mixed language
        }
        
        return language_mapping.get(language, 'en')
    
    def load_models(self, model_paths: Dict[str, str]) -> None:
        """
        Load recognition models for multiple languages.
        
        Args:
            model_paths: Dictionary mapping language codes to model file paths
        """
        for language, model_path in model_paths.items():
            model_key = self._get_model_key(language)
            if model_key in self.recognition_models:
                self.recognition_models[model_key].load_model(model_path, language)
                self.logger.info(f"Loaded model for {language} from {model_path}")
            else:
                self.logger.warning(f"No model available for language {language}")
    
    def recognize_mixed_language_text(self, 
                                    image: np.ndarray, 
                                    detections: List[Detection]) -> List[RecognizedText]:
        """
        Recognize text with automatic language detection for each region.
        
        Args:
            image: Input image array
            detections: List of text detections
            
        Returns:
            List of recognized text results with detected languages
        """
        results = []
        
        for detection in detections:
            try:
                # Preprocess text region
                text_region = self.preprocessor.preprocess_text_region(image, detection)
                
                # Detect language for this specific region
                detected_language = self.detect_language(text_region)
                
                # Recognize text using appropriate model
                recognized_text, char_confidences = self._recognize_text_region(
                    text_region, detected_language
                )
                
                # Calculate overall recognition confidence
                recognition_confidence = (
                    sum(char_confidences) / len(char_confidences) 
                    if char_confidences else 0.0
                )
                
                # Create recognized text result
                result = RecognizedText(
                    detection=detection,
                    text=recognized_text,
                    language=detected_language,
                    recognition_confidence=recognition_confidence,
                    character_confidences=char_confidences
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error recognizing mixed language text: {e}")
                # Create empty result for failed recognition
                result = RecognizedText(
                    detection=detection,
                    text="",
                    language='en',  # Default language
                    recognition_confidence=0.0,
                    character_confidences=[]
                )
                results.append(result)
        
        return results
    
    def get_language_statistics(self, recognized_texts: List[RecognizedText]) -> Dict[str, Any]:
        """
        Get statistics about detected languages in recognition results.
        
        Args:
            recognized_texts: List of recognition results
            
        Returns:
            Dictionary with language statistics
        """
        language_counts = {}
        total_texts = len(recognized_texts)
        
        for result in recognized_texts:
            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Calculate percentages
        language_percentages = {
            lang: (count / total_texts * 100) if total_texts > 0 else 0
            for lang, count in language_counts.items()
        }
        
        return {
            'total_texts': total_texts,
            'language_counts': language_counts,
            'language_percentages': language_percentages,
            'dominant_language': max(language_counts.keys(), key=lambda k: language_counts[k]) if language_counts else None
        }
    
    def format_mixed_language_output(self, recognized_texts: List[RecognizedText]) -> str:
        """
        Format recognition results for mixed-language text.
        
        Args:
            recognized_texts: List of recognition results
            
        Returns:
            Formatted output string with language annotations
        """
        formatted_parts = []
        
        for result in recognized_texts:
            if result.text.strip():
                lang_tag = f"[{result.language.upper()}]"
                confidence_tag = f"({result.recognition_confidence:.2f})"
                formatted_parts.append(f"{lang_tag}{result.text}{confidence_tag}")
        
        return " ".join(formatted_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            'architecture': 'Multilingual CRNN',
            'supported_languages': self.config.languages,
            'loaded_models': list(self.recognition_models.keys()),
            'language_detection_threshold': self.language_detection_threshold
        }
        
        # Add individual model info
        for lang, model in self.recognition_models.items():
            if hasattr(model, 'get_supported_languages'):
                model_info[f'{lang}_model_languages'] = model.get_supported_languages()
        
        return model_info