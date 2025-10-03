"""
Advanced preprocessing utilities specifically designed for historical maps.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
from pathlib import Path
import logging
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import filters, morphology, restoration, exposure, segmentation, measure
import warnings

from .interfaces import ImageProcessorInterface
from .image_processor import BasicImageProcessor
from ..models.data_structures import ProcessedImage


class HistoricalMapProcessor(ImageProcessorInterface):
    """
    Advanced image processing utilities specifically for historical maps.
    
    Handles challenging aspects of historical maps including faded ink,
    overlapping symbols, varying fonts, handwritten text, and artifacts.
    """
    
    def __init__(self, 
                 basic_processor: Optional[BasicImageProcessor] = None,
                 text_enhancement_strength: float = 1.5,
                 artifact_removal_threshold: float = 0.3,
                 rotation_detection_sensitivity: float = 0.5):
        """
        Initialize the historical map processor.
        
        Args:
            basic_processor: Basic image processor for fundamental operations
            text_enhancement_strength: Strength of text visibility enhancement (0.5-3.0)
            artifact_removal_threshold: Threshold for artifact detection (0.1-0.9)
            rotation_detection_sensitivity: Sensitivity for rotation detection (0.1-1.0)
        """
        self.basic_processor = basic_processor or BasicImageProcessor()
        self.text_enhancement_strength = max(0.5, min(3.0, text_enhancement_strength))
        self.artifact_removal_threshold = max(0.1, min(0.9, artifact_removal_threshold))
        self.rotation_detection_sensitivity = max(0.1, min(1.0, rotation_detection_sensitivity))
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def enhance_faded_text(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Enhance visibility of faded text in historical maps.
        
        Args:
            image: Input image array
            method: Enhancement method ('adaptive', 'unsharp', 'tophat', 'combined')
            
        Returns:
            Enhanced image with improved text visibility
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better text enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image.copy()
        
        if method == 'adaptive':
            # Adaptive histogram equalization with local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0 * self.text_enhancement_strength, 
                                  tileGridSize=(8, 8))
            enhanced = clahe.apply(l_channel)
            
            # Additional local contrast enhancement
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        elif method == 'unsharp':
            # Unsharp masking for text sharpening
            gaussian = cv2.GaussianBlur(l_channel, (0, 0), 2.0)
            enhanced = cv2.addWeighted(l_channel, 1.0 + self.text_enhancement_strength, 
                                     gaussian, -self.text_enhancement_strength, 0)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        elif method == 'tophat':
            # Top-hat transform to enhance dark text on light background
            kernel_size = int(15 * self.text_enhancement_strength)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (kernel_size, kernel_size))
            tophat = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, kernel)
            enhanced = cv2.add(l_channel, tophat)
            
        elif method == 'combined':
            # Combine multiple enhancement techniques
            # Step 1: CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0 * self.text_enhancement_strength, 
                                  tileGridSize=(8, 8))
            enhanced = clahe.apply(l_channel)
            
            # Step 2: Unsharp masking
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
            enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
            
            # Step 3: Top-hat transform
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            tophat_scaled = (tophat * 0.5).astype(np.uint8)
            enhanced = cv2.add(enhanced, tophat_scaled)
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        else:
            raise ValueError(f"Unsupported enhancement method: {method}")
        
        # Convert back to original format
        if len(image.shape) == 3:
            lab[:, :, 0] = enhanced
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return enhanced
    
    def detect_and_correct_rotation(self, image: np.ndarray, 
                                  angle_range: Tuple[float, float] = (-45, 45),
                                  angle_step: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image rotation using text line analysis.
        
        Args:
            image: Input image array
            angle_range: Range of angles to test (min_angle, max_angle)
            angle_step: Step size for angle testing
            
        Returns:
            Tuple of (corrected_image, detected_angle)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Enhance text for better line detection
        enhanced = self.enhance_faded_text(gray, method='adaptive')
        
        # Apply edge detection
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            # Fallback to projection profile method
            return self._correct_rotation_by_projection(image, angle_range, angle_step)
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Avoid division by zero
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Normalize angle to [-45, 45] range
                while angle > 45:
                    angle -= 90
                while angle < -45:
                    angle += 90
                angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # Use median angle for robustness
        detected_angle = np.median(angles)
        
        # Apply rotation correction if angle is significant
        if abs(detected_angle) > 1.0 / self.rotation_detection_sensitivity:
            corrected = self._rotate_image(image, -detected_angle)
            return corrected, detected_angle
        
        return image, 0.0
    
    def _correct_rotation_by_projection(self, image: np.ndarray,
                                      angle_range: Tuple[float, float],
                                      angle_step: float) -> Tuple[np.ndarray, float]:
        """
        Correct rotation using projection profile analysis.
        
        Args:
            image: Input image array
            angle_range: Range of angles to test
            angle_step: Step size for angle testing
            
        Returns:
            Tuple of (corrected_image, detected_angle)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Enhance text for better projection
        enhanced = self.enhance_faded_text(gray, method='adaptive')
        
        # Test different rotation angles
        best_angle = 0.0
        best_score = 0.0
        
        min_angle, max_angle = angle_range
        angles_to_test = np.arange(min_angle, max_angle + angle_step, angle_step)
        
        for angle in angles_to_test:
            # Rotate image
            rotated = self._rotate_image(enhanced, angle)
            
            # Calculate horizontal projection profile
            projection = np.sum(rotated, axis=1)
            
            # Calculate variance of projection (higher variance = better alignment)
            score = np.var(projection)
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        # Apply best rotation
        if abs(best_angle) > 0.5:
            corrected = self._rotate_image(image, best_angle)
            return corrected, best_angle
        
        return image, 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image array
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to avoid cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                               flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
        
        return rotated
    
    def transform_color_space_for_text(self, image: np.ndarray, 
                                     method: str = 'lab_enhancement') -> np.ndarray:
        """
        Transform color space to improve text contrast.
        
        Args:
            image: Input image array
            method: Transformation method ('lab_enhancement', 'hsv_saturation', 
                   'yuv_luminance', 'custom_channel')
            
        Returns:
            Color-transformed image array
        """
        if len(image.shape) != 3:
            return image  # Already grayscale
        
        if method == 'lab_enhancement':
            # LAB color space enhancement for better text contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Enhance L channel (lightness)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            
            # Reduce color saturation to emphasize text
            lab[:, :, 1] = lab[:, :, 1] * 0.7  # Reduce A channel
            lab[:, :, 2] = lab[:, :, 2] * 0.7  # Reduce B channel
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        elif method == 'hsv_saturation':
            # HSV enhancement with reduced saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Enhance value (brightness) channel
            v_channel = hsv[:, :, 2]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(v_channel)
            
            # Reduce saturation to make text more prominent
            hsv[:, :, 1] = hsv[:, :, 1] * 0.5
            
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif method == 'yuv_luminance':
            # YUV enhancement focusing on luminance
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Enhance Y channel (luminance)
            y_channel = yuv[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(y_channel)
            
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        elif method == 'custom_channel':
            # Create custom channel combination for text enhancement
            b, g, r = cv2.split(image)
            
            # Weighted combination emphasizing contrast
            custom = cv2.addWeighted(g, 0.5, r, 0.3, 0)
            custom = cv2.addWeighted(custom, 0.8, b, 0.2, 0)
            
            # Enhance the custom channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(custom)
            
            # Convert back to BGR
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        else:
            raise ValueError(f"Unsupported color transformation method: {method}")
    
    def remove_artifacts_and_noise(self, image: np.ndarray, 
                                 method: str = 'morphological') -> np.ndarray:
        """
        Remove artifacts and background noise while preserving text.
        
        Args:
            image: Input image array
            method: Removal method ('morphological', 'connected_components', 
                   'watershed', 'combined')
            
        Returns:
            Cleaned image array
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'morphological':
            # Morphological operations to remove small artifacts
            # Opening to remove small noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_small)
            
            # Closing to fill small gaps in text
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
            
        elif method == 'connected_components':
            # Remove small connected components (likely artifacts)
            # Threshold image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = 255 - binary  # Invert for text as foreground
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # Calculate size threshold based on image size and parameter
            image_area = gray.shape[0] * gray.shape[1]
            min_component_size = int(image_area * self.artifact_removal_threshold * 0.001)
            
            # Create mask for components above threshold
            mask = np.zeros_like(binary)
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                    mask[labels == i] = 255
            
            # Apply mask to original image
            cleaned = cv2.bitwise_and(gray, mask)
            cleaned = 255 - cleaned  # Invert back
            
        elif method == 'watershed':
            # Watershed segmentation to separate overlapping elements
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            if len(image.shape) == 3:
                markers = cv2.watershed(image, markers)
                cleaned = image.copy()
                cleaned[markers == -1] = [255, 255, 255]  # Mark boundaries
                cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
            else:
                # For grayscale, create a 3-channel version for watershed
                image_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                markers = cv2.watershed(image_3ch, markers)
                cleaned = gray.copy()
                cleaned[markers == -1] = 255
            
        elif method == 'combined':
            # Combine multiple cleaning techniques
            # Step 1: Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Step 2: Connected components filtering
            _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = 255 - binary
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            image_area = gray.shape[0] * gray.shape[1]
            min_size = int(image_area * self.artifact_removal_threshold * 0.0005)
            
            mask = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    mask[labels == i] = 255
            
            cleaned = cv2.bitwise_and(cleaned, 255 - mask)
            
            # Step 3: Final morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
        else:
            raise ValueError(f"Unsupported artifact removal method: {method}")
        
        # Convert back to original format
        if len(image.shape) == 3:
            return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        else:
            return cleaned
    
    def preprocess(self, image: np.ndarray) -> ProcessedImage:
        """
        Complete preprocessing pipeline for historical maps.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ProcessedImage with metadata
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        preprocessing_steps = []
        
        # Step 1: Basic standardization
        processed = self.basic_processor.standardize_image(image)
        preprocessing_steps.append("standardize_format")
        
        # Step 2: Color space transformation for better text contrast
        processed = self.transform_color_space_for_text(processed, method='lab_enhancement')
        preprocessing_steps.append("color_space_transformation")
        
        # Step 3: Enhance faded text visibility
        processed = self.enhance_faded_text(processed, method='combined')
        preprocessing_steps.append("faded_text_enhancement")
        
        # Step 4: Detect and correct rotation
        processed, rotation_angle = self.detect_and_correct_rotation(processed)
        if abs(rotation_angle) > 0.5:
            preprocessing_steps.append(f"rotation_correction_{rotation_angle:.1f}deg")
        
        # Step 5: Remove artifacts and noise
        processed = self.remove_artifacts_and_noise(processed, method='combined')
        preprocessing_steps.append("artifact_removal")
        
        # Step 6: Basic noise reduction
        processed = self.basic_processor.reduce_noise(processed, method='bilateral')
        preprocessing_steps.append("noise_reduction")
        
        # Step 7: Final contrast enhancement
        processed = self.basic_processor.enhance_contrast(processed, method='clahe')
        preprocessing_steps.append("final_contrast_enhancement")
        
        # Step 8: Normalize resolution
        processed = self.basic_processor.normalize_resolution(processed)
        preprocessing_steps.append("resolution_normalization")
        
        processed_size = (processed.shape[1], processed.shape[0])  # (width, height)
        
        return ProcessedImage(
            image=processed,
            original_size=original_size,
            processed_size=processed_size,
            preprocessing_steps=preprocessing_steps,
            enhancement_applied=True
        )
    
    def enhance_text_visibility(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced text visibility for historical maps.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        return self.enhance_faded_text(image, method='combined')
    
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced orientation correction for historical maps.
        
        Args:
            image: Input image array
            
        Returns:
            Orientation-corrected image array
        """
        corrected, _ = self.detect_and_correct_rotation(image)
        return corrected
    
    def normalize_resolution(self, image: np.ndarray, 
                           target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Normalize image resolution using the basic processor.
        
        Args:
            image: Input image array
            target_size: Target (width, height), uses default if None
            
        Returns:
            Resized image array
        """
        return self.basic_processor.normalize_resolution(image, target_size)
    
    def get_preprocessing_stats(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate preprocessing statistics for the image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of preprocessing statistics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate various image quality metrics
        stats = {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'contrast_ratio': float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0,
            'dynamic_range': float(np.max(gray) - np.min(gray)),
            'entropy': self._calculate_entropy(gray),
            'edge_density': self._calculate_edge_density(gray),
            'text_likelihood': self._estimate_text_likelihood(gray)
        }
        
        return stats
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy as a measure of information content."""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density as a measure of detail."""
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        return float(edge_density)
    
    def _estimate_text_likelihood(self, image: np.ndarray) -> float:
        """Estimate likelihood of text presence in the image."""
        # Apply text enhancement
        enhanced = self.enhance_faded_text(image, method='adaptive')
        
        # Calculate horizontal and vertical projections
        h_proj = np.sum(enhanced, axis=1)
        v_proj = np.sum(enhanced, axis=0)
        
        # Calculate variance of projections (higher variance suggests text lines)
        h_var = np.var(h_proj)
        v_var = np.var(v_proj)
        
        # Normalize by image size
        h_var_norm = h_var / (image.shape[1] ** 2)
        v_var_norm = v_var / (image.shape[0] ** 2)
        
        # Text likelihood is higher when horizontal variance is higher than vertical
        text_likelihood = min(1.0, h_var_norm / (v_var_norm + 1e-6))
        return float(text_likelihood)