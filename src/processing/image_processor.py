"""
Basic image processing utilities for historical map preprocessing.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path
import logging
from PIL import Image, ImageEnhance
from scipy import ndimage
from skimage import filters, morphology, restoration, exposure
import warnings

from .interfaces import ImageProcessorInterface
from ..models.data_structures import ProcessedImage


class BasicImageProcessor(ImageProcessorInterface):
    """
    Basic image processing utilities for historical map preprocessing.
    
    Implements fundamental image operations including loading, resizing,
    format conversion, noise reduction, and contrast enhancement.
    """
    
    def __init__(self, 
                 default_target_size: Tuple[int, int] = (1024, 1024),
                 preserve_aspect_ratio: bool = True,
                 interpolation_method: int = cv2.INTER_LANCZOS4):
        """
        Initialize the basic image processor.
        
        Args:
            default_target_size: Default target size for normalization
            preserve_aspect_ratio: Whether to preserve aspect ratio during resizing
            interpolation_method: OpenCV interpolation method for resizing
        """
        self.default_target_size = default_target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.interpolation_method = interpolation_method
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file with format validation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array (BGR format)
            
        Raises:
            ValueError: If image cannot be loaded or is invalid
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        if image_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            # Load with OpenCV (BGR format)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            
            if image is None:
                # Fallback to PIL for better format support
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            if image is None or image.size == 0:
                raise ValueError(f"Failed to load image: {image_path}")
            
            self.logger.info(f"Loaded image: {image_path} with shape {image.shape}")
            return image
            
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path], 
                   quality: int = 95) -> None:
        """
        Save image to file with quality control.
        
        Args:
            image: Image array to save
            output_path: Output file path
            quality: JPEG quality (0-100) for lossy formats
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set compression parameters based on format
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(output_path), image, 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_path.suffix.lower() == '.png':
            cv2.imwrite(str(output_path), image, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(output_path), image)
        
        self.logger.info(f"Saved image to: {output_path}")
    
    def convert_format(self, image: np.ndarray, target_format: str) -> np.ndarray:
        """
        Convert image between color formats.
        
        Args:
            image: Input image array
            target_format: Target format ('RGB', 'BGR', 'GRAY', 'HSV', 'LAB')
            
        Returns:
            Converted image array
        """
        if len(image.shape) == 2:
            # Already grayscale
            if target_format == 'GRAY':
                return image
            elif target_format in ['RGB', 'BGR']:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Assume input is BGR (OpenCV default)
        if target_format == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target_format == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target_format == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target_format == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif target_format == 'BGR':
            return image.copy()
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def normalize_resolution(self, image: np.ndarray, 
                           target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Normalize image resolution to target size.
        
        Args:
            image: Input image array
            target_size: Target (width, height), uses default if None
            
        Returns:
            Resized image array
        """
        if target_size is None:
            target_size = self.default_target_size
        
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        if self.preserve_aspect_ratio:
            # Calculate scaling factor to fit within target size
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=self.interpolation_method)
            
            # Pad to target size if needed
            if new_width != target_width or new_height != target_height:
                # Create padded image
                if len(image.shape) == 3:
                    padded = np.zeros((target_height, target_width, image.shape[2]), 
                                    dtype=image.dtype)
                else:
                    padded = np.zeros((target_height, target_width), dtype=image.dtype)
                
                # Center the resized image
                y_offset = (target_height - new_height) // 2
                x_offset = (target_width - new_width) // 2
                
                if len(image.shape) == 3:
                    padded[y_offset:y_offset+new_height, 
                           x_offset:x_offset+new_width] = resized
                else:
                    padded[y_offset:y_offset+new_height, 
                           x_offset:x_offset+new_width] = resized
                
                return padded
            else:
                return resized
        else:
            # Direct resize without preserving aspect ratio
            return cv2.resize(image, target_size, 
                            interpolation=self.interpolation_method)
    
    def reduce_noise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Apply noise reduction algorithms.
        
        Args:
            image: Input image array
            method: Noise reduction method ('bilateral', 'gaussian', 'median', 'nlm')
            
        Returns:
            Denoised image array
        """
        if method == 'bilateral':
            # Bilateral filter preserves edges while reducing noise
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        elif method == 'gaussian':
            # Gaussian blur for general noise reduction
            return cv2.GaussianBlur(image, (5, 5), 0)
        
        elif method == 'median':
            # Median filter for salt-and-pepper noise
            return cv2.medianBlur(image, 5)
        
        elif method == 'nlm':
            # Non-local means denoising (more computationally expensive)
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        elif method == 'morphological':
            # Morphological opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if len(image.shape) == 3:
                denoised = np.zeros_like(image)
                for i in range(image.shape[2]):
                    denoised[:, :, i] = cv2.morphologyEx(image[:, :, i], 
                                                       cv2.MORPH_OPEN, kernel)
                return denoised
            else:
                return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        else:
            raise ValueError(f"Unsupported noise reduction method: {method}")
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Apply contrast enhancement algorithms.
        
        Args:
            image: Input image array
            method: Enhancement method ('clahe', 'histogram_eq', 'adaptive', 'gamma')
            
        Returns:
            Contrast-enhanced image array
        """
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            if len(image.shape) == 3:
                # Convert to LAB color space for better results
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        
        elif method == 'histogram_eq':
            # Global histogram equalization
            if len(image.shape) == 3:
                # Convert to YUV and equalize Y channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                return cv2.equalizeHist(image)
        
        elif method == 'adaptive':
            # Adaptive histogram equalization using scikit-image
            if len(image.shape) == 3:
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    enhanced[:, :, i] = exposure.equalize_adapthist(
                        image[:, :, i], clip_limit=0.03)
                return (enhanced * 255).astype(np.uint8)
            else:
                enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
                return (enhanced * 255).astype(np.uint8)
        
        elif method == 'gamma':
            # Gamma correction
            gamma = 1.2  # Adjust as needed
            gamma_corrected = np.power(image / 255.0, gamma)
            return (gamma_corrected * 255).astype(np.uint8)
        
        else:
            raise ValueError(f"Unsupported contrast enhancement method: {method}")
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image array
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized image array
        """
        image_float = image.astype(np.float32)
        
        if method == 'minmax':
            # Min-max normalization to [0, 255]
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            if max_val > min_val:
                normalized = (image_float - min_val) / (max_val - min_val) * 255
            else:
                normalized = image_float
            return normalized.astype(np.uint8)
        
        elif method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(image_float)
            std_val = np.std(image_float)
            if std_val > 0:
                normalized = (image_float - mean_val) / std_val
                # Scale to [0, 255] range
                normalized = (normalized - np.min(normalized))
                normalized = normalized / np.max(normalized) * 255
            else:
                normalized = image_float
            return normalized.astype(np.uint8)
        
        elif method == 'robust':
            # Robust normalization using percentiles
            p2, p98 = np.percentile(image_float, (2, 98))
            if p98 > p2:
                normalized = np.clip((image_float - p2) / (p98 - p2) * 255, 0, 255)
            else:
                normalized = image_float
            return normalized.astype(np.uint8)
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    def standardize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Standardize image to common format and value range.
        
        Args:
            image: Input image array
            
        Returns:
            Standardized image array (BGR, uint8, [0-255])
        """
        # Ensure proper data type
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume float images are in [0, 1] range
                if np.max(image) <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure 3-channel BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Remove alpha channel
            image = image[:, :, :3]
        
        return image
    
    def preprocess(self, image: np.ndarray) -> ProcessedImage:
        """
        Complete preprocessing pipeline for basic image processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ProcessedImage with metadata
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        preprocessing_steps = []
        
        # Step 1: Standardize format
        processed = self.standardize_image(image)
        preprocessing_steps.append("standardize_format")
        
        # Step 2: Noise reduction
        processed = self.reduce_noise(processed, method='bilateral')
        preprocessing_steps.append("noise_reduction_bilateral")
        
        # Step 3: Contrast enhancement
        processed = self.enhance_contrast(processed, method='clahe')
        preprocessing_steps.append("contrast_enhancement_clahe")
        
        # Step 4: Normalize resolution
        processed = self.normalize_resolution(processed)
        preprocessing_steps.append("resolution_normalization")
        
        # Step 5: Final normalization
        processed = self.normalize_image(processed, method='robust')
        preprocessing_steps.append("pixel_normalization")
        
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
        Basic text visibility enhancement.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        # Convert to grayscale for text processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply unsharp masking for text sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Convert back to original format
        if len(image.shape) == 3:
            return cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)
        else:
            return unsharp_mask
    
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Basic orientation correction (placeholder implementation).
        
        Args:
            image: Input image array
            
        Returns:
            Orientation-corrected image array
        """
        # Basic implementation - detect and correct 90-degree rotations
        # This is a simplified version; more advanced methods would use
        # text line detection or Hough transforms
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use edge detection to find dominant orientations
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate dominant angle
            angles = []
            for line in lines[:min(20, len(lines))]:
                rho, theta = line[0]  # Extract from nested array
                angle = theta * 180 / np.pi
                # Normalize to [-45, 45] range
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                angles.append(angle)
            
            if angles:
                # Use median angle for robustness
                rotation_angle = np.median(angles)
                
                # Only correct if angle is significant (> 2 degrees)
                if abs(rotation_angle) > 2:
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    corrected = cv2.warpAffine(image, rotation_matrix, 
                                             (image.shape[1], image.shape[0]),
                                             flags=cv2.INTER_LANCZOS4,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(255, 255, 255))
                    return corrected
        
        return image