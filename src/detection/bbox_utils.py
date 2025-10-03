"""
Utility functions for bounding box processing and validation.
"""

from typing import List, Tuple, Optional
import numpy as np
from ..models.data_structures import BoundingBox, Detection


class BoundingBoxUtils:
    """Utility class for bounding box operations."""
    
    @staticmethod
    def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1.coordinates
        x1_2, y1_2, x2_2, y2_2 = bbox2.coordinates
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = bbox1.area
        area2 = bbox2.area
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    @staticmethod
    def calculate_overlap_ratio(bbox1: BoundingBox, bbox2: BoundingBox) -> Tuple[float, float]:
        """
        Calculate overlap ratios for two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Tuple of (overlap_ratio_1, overlap_ratio_2) where overlap_ratio_i
            is the fraction of bbox_i that overlaps with the other box
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1.coordinates
        x1_2, y1_2, x2_2, y2_2 = bbox2.coordinates
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0, 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate overlap ratios
        area1 = bbox1.area
        area2 = bbox2.area
        
        overlap_ratio_1 = intersection_area / area1 if area1 > 0 else 0.0
        overlap_ratio_2 = intersection_area / area2 if area2 > 0 else 0.0
        
        return overlap_ratio_1, overlap_ratio_2
    
    @staticmethod
    def is_inside(inner_bbox: BoundingBox, outer_bbox: BoundingBox, 
                  tolerance: float = 0.0) -> bool:
        """
        Check if one bounding box is inside another.
        
        Args:
            inner_bbox: Bounding box to check if inside
            outer_bbox: Bounding box to check if contains inner
            tolerance: Tolerance for boundary checking (pixels)
            
        Returns:
            True if inner_bbox is inside outer_bbox
        """
        x1_i, y1_i, x2_i, y2_i = inner_bbox.coordinates
        x1_o, y1_o, x2_o, y2_o = outer_bbox.coordinates
        
        return (x1_i >= x1_o - tolerance and 
                y1_i >= y1_o - tolerance and
                x2_i <= x2_o + tolerance and 
                y2_i <= y2_o + tolerance)
    
    @staticmethod
    def merge_bboxes(bboxes: List[BoundingBox]) -> BoundingBox:
        """
        Merge multiple bounding boxes into one encompassing box.
        
        Args:
            bboxes: List of bounding boxes to merge
            
        Returns:
            Merged bounding box
            
        Raises:
            ValueError: If bboxes list is empty
        """
        if not bboxes:
            raise ValueError("Cannot merge empty list of bounding boxes")
        
        # Find extremes
        min_x = min(bbox.x for bbox in bboxes)
        min_y = min(bbox.y for bbox in bboxes)
        max_x = max(bbox.x + bbox.width for bbox in bboxes)
        max_y = max(bbox.y + bbox.height for bbox in bboxes)
        
        # Calculate average confidence
        avg_confidence = sum(bbox.confidence for bbox in bboxes) / len(bboxes)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence
        )
    
    @staticmethod
    def validate_bbox(bbox: BoundingBox, image_shape: Tuple[int, int], 
                     min_size: int = 5) -> bool:
        """
        Validate bounding box coordinates and size.
        
        Args:
            bbox: Bounding box to validate
            image_shape: Image dimensions (height, width)
            min_size: Minimum width/height for valid box
            
        Returns:
            True if bounding box is valid
        """
        height, width = image_shape
        
        # Check if bounding box is within image bounds
        if bbox.x < 0 or bbox.y < 0:
            return False
        
        if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
            return False
        
        # Check minimum size requirements
        if bbox.width < min_size or bbox.height < min_size:
            return False
        
        # Check confidence range
        if bbox.confidence < 0.0 or bbox.confidence > 1.0:
            return False
        
        return True
    
    @staticmethod
    def clip_bbox_to_image(bbox: BoundingBox, image_shape: Tuple[int, int]) -> BoundingBox:
        """
        Clip bounding box coordinates to image boundaries.
        
        Args:
            bbox: Bounding box to clip
            image_shape: Image dimensions (height, width)
            
        Returns:
            Clipped bounding box
        """
        height, width = image_shape
        
        # Clip coordinates
        x = max(0, min(bbox.x, width - 1))
        y = max(0, min(bbox.y, height - 1))
        
        # Adjust width and height
        max_width = width - x
        max_height = height - y
        
        clipped_width = min(bbox.width, max_width)
        clipped_height = min(bbox.height, max_height)
        
        return BoundingBox(
            x=x,
            y=y,
            width=clipped_width,
            height=clipped_height,
            confidence=bbox.confidence
        )
    
    @staticmethod
    def scale_bbox(bbox: BoundingBox, scale_x: float, scale_y: float) -> BoundingBox:
        """
        Scale bounding box coordinates.
        
        Args:
            bbox: Bounding box to scale
            scale_x: Scaling factor for x-axis
            scale_y: Scaling factor for y-axis
            
        Returns:
            Scaled bounding box
        """
        return BoundingBox(
            x=int(bbox.x * scale_x),
            y=int(bbox.y * scale_y),
            width=int(bbox.width * scale_x),
            height=int(bbox.height * scale_y),
            confidence=bbox.confidence
        )
    
    @staticmethod
    def expand_bbox(bbox: BoundingBox, expand_ratio: float = 0.1) -> BoundingBox:
        """
        Expand bounding box by a given ratio.
        
        Args:
            bbox: Bounding box to expand
            expand_ratio: Ratio to expand (0.1 = 10% expansion)
            
        Returns:
            Expanded bounding box
        """
        expand_w = int(bbox.width * expand_ratio)
        expand_h = int(bbox.height * expand_ratio)
        
        return BoundingBox(
            x=max(0, bbox.x - expand_w // 2),
            y=max(0, bbox.y - expand_h // 2),
            width=bbox.width + expand_w,
            height=bbox.height + expand_h,
            confidence=bbox.confidence
        )
    
    @staticmethod
    def filter_small_boxes(bboxes: List[BoundingBox], min_area: int = 25) -> List[BoundingBox]:
        """
        Filter out bounding boxes that are too small.
        
        Args:
            bboxes: List of bounding boxes
            min_area: Minimum area threshold
            
        Returns:
            Filtered list of bounding boxes
        """
        return [bbox for bbox in bboxes if bbox.area >= min_area]
    
    @staticmethod
    def sort_bboxes_reading_order(bboxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Sort bounding boxes in reading order (top to bottom, left to right).
        
        Args:
            bboxes: List of bounding boxes to sort
            
        Returns:
            Sorted list of bounding boxes
        """
        return sorted(bboxes, key=lambda bbox: (bbox.y, bbox.x))
    
    @staticmethod
    def group_bboxes_by_line(bboxes: List[BoundingBox], 
                           line_threshold: float = 0.5) -> List[List[BoundingBox]]:
        """
        Group bounding boxes by text lines based on vertical overlap.
        
        Args:
            bboxes: List of bounding boxes to group
            line_threshold: Minimum vertical overlap ratio to be on same line
            
        Returns:
            List of lists, each containing boxes on the same line
        """
        if not bboxes:
            return []
        
        # Sort by y-coordinate
        sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox.y)
        
        lines = []
        current_line = [sorted_bboxes[0]]
        
        for i in range(1, len(sorted_bboxes)):
            current_bbox = sorted_bboxes[i]
            
            # Check if current box overlaps vertically with any box in current line
            overlaps_with_line = False
            for line_bbox in current_line:
                # Calculate vertical overlap
                y1_1, y2_1 = line_bbox.y, line_bbox.y + line_bbox.height
                y1_2, y2_2 = current_bbox.y, current_bbox.y + current_bbox.height
                
                overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
                min_height = min(line_bbox.height, current_bbox.height)
                
                if overlap >= line_threshold * min_height:
                    overlaps_with_line = True
                    break
            
            if overlaps_with_line:
                current_line.append(current_bbox)
            else:
                # Sort current line by x-coordinate and add to lines
                current_line.sort(key=lambda bbox: bbox.x)
                lines.append(current_line)
                current_line = [current_bbox]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda bbox: bbox.x)
            lines.append(current_line)
        
        return lines


class NonMaximumSuppression:
    """Non-Maximum Suppression utilities for detection filtering."""
    
    @staticmethod
    def apply_nms(detections: List[Detection], nms_threshold: float = 0.5) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.
        
        Args:
            detections: List of detection results
            nms_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("NMS threshold must be between 0.0 and 1.0")
        
        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while sorted_detections:
            # Keep the detection with highest confidence
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU overlap
            remaining = []
            for detection in sorted_detections:
                iou = BoundingBoxUtils.calculate_iou(current.bbox, detection.bbox)
                if iou < nms_threshold:
                    remaining.append(detection)
            
            sorted_detections = remaining
        
        return keep
    
    @staticmethod
    def apply_soft_nms(detections: List[Detection], nms_threshold: float = 0.5,
                      sigma: float = 0.5, min_confidence: float = 0.1) -> List[Detection]:
        """
        Apply Soft Non-Maximum Suppression which reduces confidence instead of removing.
        
        Args:
            detections: List of detection results
            nms_threshold: IoU threshold for suppression
            sigma: Gaussian parameter for soft suppression
            min_confidence: Minimum confidence to keep detection
            
        Returns:
            Filtered list of detections with adjusted confidences
        """
        if not detections:
            return []
        
        # Create mutable copies
        detections_copy = []
        for det in detections:
            new_bbox = BoundingBox(
                x=det.bbox.x, y=det.bbox.y, 
                width=det.bbox.width, height=det.bbox.height,
                confidence=det.confidence
            )
            new_det = Detection(
                bbox=new_bbox,
                detection_type=det.detection_type,
                confidence=det.confidence,
                image_coordinates=det.image_coordinates
            )
            detections_copy.append(new_det)
        
        # Sort by confidence (descending)
        detections_copy.sort(key=lambda x: x.confidence, reverse=True)
        
        for i in range(len(detections_copy)):
            if detections_copy[i].confidence < min_confidence:
                continue
                
            for j in range(i + 1, len(detections_copy)):
                if detections_copy[j].confidence < min_confidence:
                    continue
                
                iou = BoundingBoxUtils.calculate_iou(
                    detections_copy[i].bbox, detections_copy[j].bbox
                )
                
                if iou > nms_threshold:
                    # Apply Gaussian decay
                    weight = np.exp(-(iou * iou) / sigma)
                    detections_copy[j].confidence *= weight
                    detections_copy[j].bbox.confidence = detections_copy[j].confidence
        
        # Filter by minimum confidence
        return [det for det in detections_copy if det.confidence >= min_confidence]