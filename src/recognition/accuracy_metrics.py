"""
Accuracy metrics for text recognition evaluation.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from difflib import SequenceMatcher
import logging


class RecognitionAccuracyMetrics:
    """
    Calculate character-level and word-level accuracy metrics for text recognition.
    """
    
    def __init__(self):
        """Initialize accuracy metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_character_accuracy(self, 
                                   predicted_texts: List[str], 
                                   ground_truth_texts: List[str]) -> Dict[str, float]:
        """
        Calculate character-level accuracy metrics.
        
        Args:
            predicted_texts: List of predicted text strings
            ground_truth_texts: List of ground truth text strings
            
        Returns:
            Dictionary containing character-level metrics
        """
        if len(predicted_texts) != len(ground_truth_texts):
            raise ValueError("Predicted and ground truth lists must have same length")
        
        total_chars = 0
        correct_chars = 0
        total_edit_distance = 0
        
        for pred, gt in zip(predicted_texts, ground_truth_texts):
            # Character-level comparison
            char_correct, char_total = self._calculate_char_level_accuracy(pred, gt)
            correct_chars += char_correct
            total_chars += char_total
            
            # Edit distance (Levenshtein distance)
            edit_dist = self._calculate_edit_distance(pred, gt)
            total_edit_distance += edit_dist
        
        # Calculate metrics
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        avg_edit_distance = total_edit_distance / len(predicted_texts) if predicted_texts else 0.0
        
        # Normalized edit distance (0 = perfect, 1 = completely wrong)
        max_possible_distance = sum(max(len(pred), len(gt)) for pred, gt in zip(predicted_texts, ground_truth_texts))
        normalized_edit_distance = total_edit_distance / max_possible_distance if max_possible_distance > 0 else 0.0
        
        return {
            'character_accuracy': char_accuracy,
            'total_characters': total_chars,
            'correct_characters': correct_chars,
            'average_edit_distance': avg_edit_distance,
            'normalized_edit_distance': normalized_edit_distance,
            'character_error_rate': 1.0 - char_accuracy
        }
    
    def calculate_word_accuracy(self, 
                              predicted_texts: List[str], 
                              ground_truth_texts: List[str]) -> Dict[str, float]:
        """
        Calculate word-level accuracy metrics.
        
        Args:
            predicted_texts: List of predicted text strings
            ground_truth_texts: List of ground truth text strings
            
        Returns:
            Dictionary containing word-level metrics
        """
        if len(predicted_texts) != len(ground_truth_texts):
            raise ValueError("Predicted and ground truth lists must have same length")
        
        total_words = 0
        correct_words = 0
        exact_matches = 0
        
        for pred, gt in zip(predicted_texts, ground_truth_texts):
            # Exact string match
            if pred.strip().lower() == gt.strip().lower():
                exact_matches += 1
            
            # Word-level comparison
            pred_words = pred.strip().split()
            gt_words = gt.strip().split()
            
            word_correct, word_total = self._calculate_word_level_accuracy(pred_words, gt_words)
            correct_words += word_correct
            total_words += word_total
        
        # Calculate metrics
        word_accuracy = correct_words / total_words if total_words > 0 else 0.0
        exact_match_accuracy = exact_matches / len(predicted_texts) if predicted_texts else 0.0
        
        return {
            'word_accuracy': word_accuracy,
            'total_words': total_words,
            'correct_words': correct_words,
            'exact_match_accuracy': exact_match_accuracy,
            'word_error_rate': 1.0 - word_accuracy
        }
    
    def calculate_sequence_accuracy(self, 
                                  predicted_texts: List[str], 
                                  ground_truth_texts: List[str]) -> Dict[str, float]:
        """
        Calculate sequence-level accuracy using similarity measures.
        
        Args:
            predicted_texts: List of predicted text strings
            ground_truth_texts: List of ground truth text strings
            
        Returns:
            Dictionary containing sequence-level metrics
        """
        if len(predicted_texts) != len(ground_truth_texts):
            raise ValueError("Predicted and ground truth lists must have same length")
        
        total_similarity = 0.0
        jaro_winkler_similarity = 0.0
        
        for pred, gt in zip(predicted_texts, ground_truth_texts):
            # Sequence similarity using difflib
            similarity = SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
            total_similarity += similarity
            
            # Jaro-Winkler similarity (approximated using SequenceMatcher)
            jw_sim = self._calculate_jaro_winkler_similarity(pred, gt)
            jaro_winkler_similarity += jw_sim
        
        avg_similarity = total_similarity / len(predicted_texts) if predicted_texts else 0.0
        avg_jaro_winkler = jaro_winkler_similarity / len(predicted_texts) if predicted_texts else 0.0
        
        return {
            'sequence_similarity': avg_similarity,
            'jaro_winkler_similarity': avg_jaro_winkler,
            'sequence_error_rate': 1.0 - avg_similarity
        }
    
    def calculate_comprehensive_metrics(self, 
                                      predicted_texts: List[str], 
                                      ground_truth_texts: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive accuracy metrics.
        
        Args:
            predicted_texts: List of predicted text strings
            ground_truth_texts: List of ground truth text strings
            
        Returns:
            Dictionary containing all accuracy metrics
        """
        char_metrics = self.calculate_character_accuracy(predicted_texts, ground_truth_texts)
        word_metrics = self.calculate_word_accuracy(predicted_texts, ground_truth_texts)
        seq_metrics = self.calculate_sequence_accuracy(predicted_texts, ground_truth_texts)
        
        # Combine all metrics
        comprehensive_metrics = {
            'character_metrics': char_metrics,
            'word_metrics': word_metrics,
            'sequence_metrics': seq_metrics,
            'sample_count': len(predicted_texts)
        }
        
        # Add overall summary
        comprehensive_metrics['summary'] = {
            'character_accuracy': char_metrics['character_accuracy'],
            'word_accuracy': word_metrics['word_accuracy'],
            'exact_match_accuracy': word_metrics['exact_match_accuracy'],
            'sequence_similarity': seq_metrics['sequence_similarity']
        }
        
        return comprehensive_metrics
    
    def _calculate_char_level_accuracy(self, predicted: str, ground_truth: str) -> Tuple[int, int]:
        """
        Calculate character-level accuracy for a single pair.
        
        Args:
            predicted: Predicted text
            ground_truth: Ground truth text
            
        Returns:
            Tuple of (correct_chars, total_chars)
        """
        # Use the ground truth length as reference for total character count
        total_chars = len(ground_truth)
        correct = 0
        
        # Compare characters up to the minimum length
        min_len = min(len(predicted), len(ground_truth))
        for i in range(min_len):
            if predicted[i].lower() == ground_truth[i].lower():
                correct += 1
        
        # If predicted is longer than ground truth, those extra characters are wrong
        # If predicted is shorter than ground truth, those missing characters are wrong
        # In both cases, we only count correct matches within the overlap
        
        return correct, total_chars
    
    def _calculate_word_level_accuracy(self, predicted_words: List[str], gt_words: List[str]) -> Tuple[int, int]:
        """
        Calculate word-level accuracy for a single pair.
        
        Args:
            predicted_words: List of predicted words
            gt_words: List of ground truth words
            
        Returns:
            Tuple of (correct_words, total_words)
        """
        # Use the longer word list as reference
        max_len = max(len(predicted_words), len(gt_words))
        correct = 0
        
        for i in range(max_len):
            pred_word = predicted_words[i] if i < len(predicted_words) else ''
            gt_word = gt_words[i] if i < len(gt_words) else ''
            
            if pred_word.lower().strip() == gt_word.lower().strip():
                correct += 1
        
        return correct, max_len
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (edit) distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate Jaro-Winkler similarity (approximated).
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Jaro-Winkler similarity score
        """
        # Simplified implementation using SequenceMatcher
        # This is an approximation of Jaro-Winkler similarity
        matcher = SequenceMatcher(None, s1.lower(), s2.lower())
        similarity = matcher.ratio()
        
        # Apply prefix bonus (Winkler modification)
        prefix_length = 0
        for i in range(min(len(s1), len(s2), 4)):  # Max prefix length of 4
            if s1[i].lower() == s2[i].lower():
                prefix_length += 1
            else:
                break
        
        # Jaro-Winkler formula approximation
        jaro_winkler = similarity + (0.1 * prefix_length * (1 - similarity))
        
        return min(jaro_winkler, 1.0)
    
    def print_metrics_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted metrics report.
        
        Args:
            metrics: Comprehensive metrics dictionary
        """
        print("=" * 60)
        print("TEXT RECOGNITION ACCURACY REPORT")
        print("=" * 60)
        
        if 'summary' in metrics:
            print("\nSUMMARY:")
            summary = metrics['summary']
            print(f"  Character Accuracy: {summary['character_accuracy']:.3f}")
            print(f"  Word Accuracy: {summary['word_accuracy']:.3f}")
            print(f"  Exact Match Accuracy: {summary['exact_match_accuracy']:.3f}")
            print(f"  Sequence Similarity: {summary['sequence_similarity']:.3f}")
        
        if 'character_metrics' in metrics:
            print("\nCHARACTER-LEVEL METRICS:")
            char_metrics = metrics['character_metrics']
            print(f"  Accuracy: {char_metrics['character_accuracy']:.3f}")
            print(f"  Error Rate: {char_metrics['character_error_rate']:.3f}")
            print(f"  Correct/Total: {char_metrics['correct_characters']}/{char_metrics['total_characters']}")
            print(f"  Avg Edit Distance: {char_metrics['average_edit_distance']:.2f}")
            print(f"  Normalized Edit Distance: {char_metrics['normalized_edit_distance']:.3f}")
        
        if 'word_metrics' in metrics:
            print("\nWORD-LEVEL METRICS:")
            word_metrics = metrics['word_metrics']
            print(f"  Accuracy: {word_metrics['word_accuracy']:.3f}")
            print(f"  Error Rate: {word_metrics['word_error_rate']:.3f}")
            print(f"  Correct/Total: {word_metrics['correct_words']}/{word_metrics['total_words']}")
            print(f"  Exact Match Accuracy: {word_metrics['exact_match_accuracy']:.3f}")
        
        if 'sequence_metrics' in metrics:
            print("\nSEQUENCE-LEVEL METRICS:")
            seq_metrics = metrics['sequence_metrics']
            print(f"  Sequence Similarity: {seq_metrics['sequence_similarity']:.3f}")
            print(f"  Jaro-Winkler Similarity: {seq_metrics['jaro_winkler_similarity']:.3f}")
            print(f"  Sequence Error Rate: {seq_metrics['sequence_error_rate']:.3f}")
        
        print(f"\nSample Count: {metrics.get('sample_count', 0)}")
        print("=" * 60)