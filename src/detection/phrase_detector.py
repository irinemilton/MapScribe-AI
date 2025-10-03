"""
Phrase-level text detection implementation.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from ..models.data_structures import (
    ProcessedImage, BoundingBox, Detection, DetectionConfig
)
from .base_detector import TextDetector
from .yolo_detector import YOLOWordDetector
from .bbox_utils import BoundingBoxUtils


@dataclass
class PhraseGroupingConfig:
    """Configuration for phrase grouping algorithms."""
    line_overlap_threshold: float = 0.5  # Minimum vertical overlap to be on same line
    horizontal_gap_ratio: float = 0.8    # Max gap as ratio of average word width
    min_words_per_phrase: int = 1        # Minimum words to form a phrase
    max_words_per_phrase: int = 10       # Maximum words in a single phrase
    confidence_weight: float = 0.3       # Weight for confidence in grouping decisions


class PhraseDetector(TextDetector):
    """
    Phrase-level text detection using word grouping algorithms.
    
    This class extends the base TextDetector to provide phrase-level detection
    by grouping word detections based on spatial proximity and text line analysis.
    """
    
    def __init__(self, config: DetectionConfig, 
                 phrase_config: Optional[PhraseGroupingConfig] = None):
        """
        Initialize phrase detector.
        
        Args:
            config: Detection configuration
            phrase_config: Optional phrase grouping configuration
        """
        super().__init__(config)
        self.phrase_config = phrase_config or PhraseGroupingConfig()
        self.word_detector: Optional[YOLOWordDetector] = None
        
    def load_word_detector(self, model_path: str) -> None:
        """
        Load word detection model for phrase detection.
        
        Args:
            model_path: Path to word detection model
        """
        self.word_detector = YOLOWordDetector(model_path, self.config)
        self.logger.info(f"Loaded word detector for phrase detection: {model_path}")
    
    def detect_phrases(self, image: ProcessedImage) -> List[Detection]:
        """
        Detect phrase regions by grouping word detections.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of phrase detections
            
        Raises:
            RuntimeError: If word detector is not loaded
        """
        if self.word_detector is None:
            raise RuntimeError("Word detector not loaded. Call load_word_detector() first.")
        
        # First detect individual words
        word_boxes = self.word_detector.detect_words(image)
        
        if not word_boxes:
            return []
        
        # Convert to Detection objects for processing
        word_detections = [
            Detection(
                bbox=bbox,
                detection_type='word',
                confidence=bbox.confidence,
                image_coordinates=bbox.coordinates
            )
            for bbox in word_boxes
        ]
        
        # Group words into phrases using multiple algorithms
        phrase_groups = self._group_words_into_phrases(word_detections)
        
        # Create phrase detections from groups
        phrase_detections = []
        for group in phrase_groups:
            phrase_detection = self._create_phrase_from_words(group)
            phrase_detections.append(phrase_detection)
        
        self.logger.info(f"Grouped {len(word_detections)} words into {len(phrase_detections)} phrases")
        return phrase_detections
    
    def _group_words_into_phrases(self, word_detections: List[Detection]) -> List[List[Detection]]:
        """
        Group word detections into phrases using spatial analysis.
        
        Args:
            word_detections: List of word detections
            
        Returns:
            List of word groups, each representing a phrase
        """
        if not word_detections:
            return []
        
        # Step 1: Group words by text lines
        line_groups = self._group_words_by_lines(word_detections)
        
        # Step 2: Within each line, group words into phrases
        phrase_groups = []
        for line_words in line_groups:
            line_phrases = self._group_line_words_into_phrases(line_words)
            phrase_groups.extend(line_phrases)
        
        # Step 3: Post-process groups (merge close phrases, split long ones)
        final_groups = self._post_process_phrase_groups(phrase_groups)
        
        return final_groups
    
    def _group_words_by_lines(self, word_detections: List[Detection]) -> List[List[Detection]]:
        """
        Group word detections by text lines based on vertical overlap.
        
        Args:
            word_detections: List of word detections
            
        Returns:
            List of word groups, each representing a text line
        """
        if not word_detections:
            return []
        
        # Sort words by y-coordinate (top to bottom)
        sorted_words = sorted(word_detections, key=lambda d: d.bbox.y)
        
        lines = []
        current_line = [sorted_words[0]]
        
        for i in range(1, len(sorted_words)):
            current_word = sorted_words[i]
            
            # Check if current word overlaps vertically with any word in current line
            overlaps_with_line = False
            for line_word in current_line:
                if self._words_on_same_line(line_word, current_word):
                    overlaps_with_line = True
                    break
            
            if overlaps_with_line:
                current_line.append(current_word)
            else:
                # Sort current line by x-coordinate and add to lines
                current_line.sort(key=lambda d: d.bbox.x)
                lines.append(current_line)
                current_line = [current_word]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda d: d.bbox.x)
            lines.append(current_line)
        
        return lines
    
    def _words_on_same_line(self, word1: Detection, word2: Detection) -> bool:
        """
        Determine if two words are on the same text line.
        
        Args:
            word1: First word detection
            word2: Second word detection
            
        Returns:
            True if words are on the same line
        """
        # Calculate vertical overlap
        y1_1, y2_1 = word1.bbox.y, word1.bbox.y + word1.bbox.height
        y1_2, y2_2 = word2.bbox.y, word2.bbox.y + word2.bbox.height
        
        overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        min_height = min(word1.bbox.height, word2.bbox.height)
        
        overlap_ratio = overlap / min_height if min_height > 0 else 0
        
        return overlap_ratio >= self.phrase_config.line_overlap_threshold
    
    def _group_line_words_into_phrases(self, line_words: List[Detection]) -> List[List[Detection]]:
        """
        Group words within a text line into phrases.
        
        Args:
            line_words: List of word detections on the same line
            
        Returns:
            List of phrase groups from this line
        """
        if not line_words:
            return []
        
        if len(line_words) == 1:
            return [line_words]
        
        # Sort words by x-coordinate (left to right)
        sorted_words = sorted(line_words, key=lambda d: d.bbox.x)
        
        phrases = []
        current_phrase = [sorted_words[0]]
        
        for i in range(1, len(sorted_words)):
            current_word = sorted_words[i]
            last_word = current_phrase[-1]
            
            # Check if words should be in the same phrase
            if self._should_group_words(last_word, current_word, current_phrase):
                current_phrase.append(current_word)
            else:
                # Start new phrase
                if len(current_phrase) >= self.phrase_config.min_words_per_phrase:
                    phrases.append(current_phrase)
                current_phrase = [current_word]
        
        # Add the last phrase
        if len(current_phrase) >= self.phrase_config.min_words_per_phrase:
            phrases.append(current_phrase)
        
        return phrases
    
    def _should_group_words(self, word1: Detection, word2: Detection, 
                          current_phrase: List[Detection]) -> bool:
        """
        Determine if two adjacent words should be grouped into the same phrase.
        
        Args:
            word1: Previous word in sequence
            word2: Current word to consider
            current_phrase: Current phrase being built
            
        Returns:
            True if words should be grouped together
        """
        # Check phrase length limit
        if len(current_phrase) >= self.phrase_config.max_words_per_phrase:
            return False
        
        # Calculate horizontal gap
        gap = word2.bbox.x - (word1.bbox.x + word1.bbox.width)
        
        # Calculate average word width in current phrase
        avg_width = sum(w.bbox.width for w in current_phrase) / len(current_phrase)
        
        # Maximum allowed gap
        max_gap = avg_width * self.phrase_config.horizontal_gap_ratio
        
        # Basic spatial criterion
        if gap > max_gap:
            return False
        
        # Consider confidence scores (higher confidence words are more likely to group)
        confidence_factor = (word1.confidence + word2.confidence) / 2
        confidence_bonus = confidence_factor * self.phrase_config.confidence_weight
        
        # Adjust gap threshold based on confidence
        adjusted_max_gap = max_gap * (1 + confidence_bonus)
        
        return gap <= adjusted_max_gap
    
    def _post_process_phrase_groups(self, phrase_groups: List[List[Detection]]) -> List[List[Detection]]:
        """
        Post-process phrase groups to merge close phrases and split long ones.
        
        Args:
            phrase_groups: Initial phrase groups
            
        Returns:
            Refined phrase groups
        """
        if not phrase_groups:
            return []
        
        # Step 1: Merge phrases that are very close
        merged_groups = self._merge_close_phrases(phrase_groups)
        
        # Step 2: Split overly long phrases
        final_groups = self._split_long_phrases(merged_groups)
        
        return final_groups
    
    def _merge_close_phrases(self, phrase_groups: List[List[Detection]]) -> List[List[Detection]]:
        """
        Merge phrases that are very close to each other.
        
        Args:
            phrase_groups: List of phrase groups
            
        Returns:
            Merged phrase groups
        """
        if len(phrase_groups) <= 1:
            return phrase_groups
        
        # Sort phrases by position (top-left corner)
        sorted_phrases = sorted(phrase_groups, 
                              key=lambda group: (min(w.bbox.y for w in group),
                                               min(w.bbox.x for w in group)))
        
        merged = []
        current_group = sorted_phrases[0]
        
        for i in range(1, len(sorted_phrases)):
            next_group = sorted_phrases[i]
            
            if self._should_merge_phrases(current_group, next_group):
                # Merge the groups
                current_group = current_group + next_group
                # Sort merged group by x-coordinate
                current_group.sort(key=lambda d: d.bbox.x)
            else:
                merged.append(current_group)
                current_group = next_group
        
        merged.append(current_group)
        return merged
    
    def _should_merge_phrases(self, phrase1: List[Detection], phrase2: List[Detection]) -> bool:
        """
        Determine if two phrases should be merged.
        
        Args:
            phrase1: First phrase group
            phrase2: Second phrase group
            
        Returns:
            True if phrases should be merged
        """
        # Get bounding boxes for each phrase
        bbox1 = self._get_phrase_bbox(phrase1)
        bbox2 = self._get_phrase_bbox(phrase2)
        
        # Check if phrases are on the same line
        if not self._words_on_same_line(
            Detection(bbox=bbox1, detection_type='phrase', confidence=0.5, image_coordinates=bbox1.coordinates),
            Detection(bbox=bbox2, detection_type='phrase', confidence=0.5, image_coordinates=bbox2.coordinates)
        ):
            return False
        
        # Calculate gap between phrases
        if bbox1.x < bbox2.x:
            gap = bbox2.x - (bbox1.x + bbox1.width)
        else:
            gap = bbox1.x - (bbox2.x + bbox2.width)
        
        # Merge if gap is very small
        avg_width = (bbox1.width + bbox2.width) / 2
        merge_threshold = avg_width * 0.3  # Very close threshold
        
        return gap <= merge_threshold
    
    def _split_long_phrases(self, phrase_groups: List[List[Detection]]) -> List[List[Detection]]:
        """
        Split overly long phrases into smaller ones.
        
        Args:
            phrase_groups: List of phrase groups
            
        Returns:
            Split phrase groups
        """
        split_groups = []
        
        for group in phrase_groups:
            if len(group) <= self.phrase_config.max_words_per_phrase:
                split_groups.append(group)
            else:
                # Split long phrase into smaller chunks
                chunks = self._split_phrase_group(group)
                split_groups.extend(chunks)
        
        return split_groups
    
    def _split_phrase_group(self, phrase_group: List[Detection]) -> List[List[Detection]]:
        """
        Split a long phrase group into smaller chunks.
        
        Args:
            phrase_group: Long phrase group to split
            
        Returns:
            List of smaller phrase chunks
        """
        max_size = self.phrase_config.max_words_per_phrase
        chunks = []
        
        for i in range(0, len(phrase_group), max_size):
            chunk = phrase_group[i:i + max_size]
            if len(chunk) >= self.phrase_config.min_words_per_phrase:
                chunks.append(chunk)
        
        return chunks
    
    def _get_phrase_bbox(self, phrase_words: List[Detection]) -> BoundingBox:
        """
        Calculate bounding box that encompasses all words in a phrase.
        
        Args:
            phrase_words: List of word detections in phrase
            
        Returns:
            Bounding box encompassing the phrase
        """
        if not phrase_words:
            raise ValueError("Cannot create bounding box from empty phrase")
        
        word_bboxes = [word.bbox for word in phrase_words]
        return BoundingBoxUtils.merge_bboxes(word_bboxes)
    
    def _create_phrase_from_words(self, words: List[Detection]) -> Detection:
        """
        Create a phrase detection from a group of word detections.
        
        Args:
            words: List of word detections to group
            
        Returns:
            Phrase detection encompassing all words
        """
        if not words:
            raise ValueError("Cannot create phrase from empty word list")
        
        # Calculate phrase bounding box
        phrase_bbox = self._get_phrase_bbox(words)
        
        # Calculate phrase confidence (weighted average)
        total_area = sum(w.bbox.area for w in words)
        if total_area > 0:
            weighted_confidence = sum(w.confidence * w.bbox.area for w in words) / total_area
        else:
            weighted_confidence = sum(w.confidence for w in words) / len(words)
        
        # Create phrase detection
        phrase_detection = Detection(
            bbox=BoundingBox(
                x=phrase_bbox.x,
                y=phrase_bbox.y,
                width=phrase_bbox.width,
                height=phrase_bbox.height,
                confidence=weighted_confidence
            ),
            detection_type='phrase',
            confidence=weighted_confidence,
            image_coordinates=phrase_bbox.coordinates
        )
        
        return phrase_detection
    
    def get_phrase_statistics(self, phrase_detections: List[Detection]) -> Dict[str, float]:
        """
        Calculate statistics for phrase detections.
        
        Args:
            phrase_detections: List of phrase detections
            
        Returns:
            Dictionary with phrase statistics
        """
        if not phrase_detections:
            return {
                'num_phrases': 0,
                'avg_confidence': 0.0,
                'avg_width': 0.0,
                'avg_height': 0.0,
                'avg_area': 0.0
            }
        
        confidences = [p.confidence for p in phrase_detections]
        widths = [p.bbox.width for p in phrase_detections]
        heights = [p.bbox.height for p in phrase_detections]
        areas = [p.bbox.area for p in phrase_detections]
        
        return {
            'num_phrases': len(phrase_detections),
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'avg_area': sum(areas) / len(areas),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }