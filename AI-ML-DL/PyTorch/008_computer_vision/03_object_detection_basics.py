"""
PyTorch Object Detection Basics - Fundamental Object Detection Implementation
Comprehensive guide to basic object detection concepts and implementation in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.ops import box_iou, nms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import math

print("=== OBJECT DETECTION BASICS ===")

# 1. BOUNDING BOX OPERATIONS
print("\n1. BOUNDING BOX OPERATIONS")

class BoundingBox:
    """Basic bounding box operations"""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, format: str = 'xyxy'):
        """
        Initialize bounding box
        Args:
            x1, y1, x2, y2: Box coordinates
            format: 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,width,height)
        """
        if format == 'xyxy':
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        elif format == 'xywh':
            self.x1, self.y1 = x1, y1
            self.x2, self.y2 = x1 + x2, y1 + y2
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def area(self) -> float:
        """Calculate box area"""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    def center(self) -> Tuple[float, float]:
        """Get box center"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def width_height(self) -> Tuple[float, float]:
        """Get box width and height"""
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor format"""
        return torch.tensor([self.x1, self.y1, self.x2, self.y2])

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two sets of boxes"""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Calculate intersection area
    x_left = torch.max(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
    y_top = torch.max(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
    x_right = torch.min(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
    y_bottom = torch.min(y2_1.unsqueeze(1), y2_2.unsqueeze(0))
    
    intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area
    
    # Calculate IoU
    iou = intersection_area / (union_area + 1e-6)
    return iou

# Test bounding box operations
bbox1 = BoundingBox(10, 10, 50, 50)
bbox2 = BoundingBox(30, 30, 70, 70)

print(f"Box 1 area: {bbox1.area()}")
print(f"Box 1 center: {bbox1.center()}")
print(f"Box 1 w,h: {bbox1.width_height()}")

# Test IoU calculation
boxes1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]]).float()
boxes2 = torch.tensor([[30, 30, 70, 70], [15, 15, 55, 55]]).float()
iou_matrix = calculate_iou(boxes1, boxes2)
print(f"IoU matrix shape: {iou_matrix.shape}")
print(f"IoU values:\n{iou_matrix}")

# 2. ANCHOR GENERATION
print("\n2. ANCHOR GENERATION")

class AnchorGenerator:
    """Generate anchor boxes for object detection"""
    
    def __init__(self, sizes: List[int] = [32, 64, 128, 256, 512],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 stride: int = 16):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        
    def generate_single_level_anchors(self, size: int, 
                                    aspect_ratios: List[float]) -> torch.Tensor:
        """Generate anchors for a single scale level"""
        anchors = []
        
        for ratio in aspect_ratios:
            # Calculate width and height based on area and aspect ratio
            area = size ** 2
            width = math.sqrt(area / ratio)
            height = width * ratio
            
            # Create anchor centered at origin
            anchor = [-width/2, -height/2, width/2, height/2]
            anchors.append(anchor)
            
        return torch.tensor(anchors)
    
    def generate_anchors_for_feature_map(self, feature_height: int, 
                                       feature_width: int, 
                                       size: int) -> torch.Tensor:
        """Generate anchors for entire feature map"""
        base_anchors = self.generate_single_level_anchors(size, self.aspect_ratios)
        num_anchors = len(base_anchors)
        
        # Create grid of anchor centers
        shifts_x = torch.arange(feature_width) * self.stride
        shifts_y = torch.arange(feature_height) * self.stride
        
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
        
        # Apply shifts to base anchors
        anchors = base_anchors.view(1, num_anchors, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.view(-1, 4)
        
        return anchors
    
    def generate_multi_level_anchors(self, feature_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """Generate anchors for multiple feature map levels"""
        multi_level_anchors = []
        
        for i, (height, width) in enumerate(feature_shapes):
            size = self.sizes[i] if i < len(self.sizes) else self.sizes[-1]
            anchors = self.generate_anchors_for_feature_map(height, width, size)
            multi_level_anchors.append(anchors)
            
        return multi_level_anchors

# Test anchor generation
anchor_gen = AnchorGenerator()

# Generate anchors for different feature map sizes
feature_shapes = [(32, 32), (16, 16), (8, 8)]
multi_level_anchors = anchor_gen.generate_multi_level_anchors(feature_shapes)

for i, anchors in enumerate(multi_level_anchors):
    print(f"Level {i}: {anchors.shape[0]} anchors, shape {anchors.shape}")
    print(f"  Sample anchors: {anchors[:3]}")

# 3. NON-MAXIMUM SUPPRESSION
print("\n3. NON-MAXIMUM SUPPRESSION")

def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, 
               iou_threshold: float = 0.5) -> torch.Tensor:
    """Simple implementation of Non-Maximum Suppression"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while sorted_indices.numel() > 0:
        # Take the box with highest score
        current = sorted_indices[0]
        keep.append(current.item())
        
        if sorted_indices.numel() == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = calculate_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)

# Test NMS
test_boxes = torch.tensor([
    [10, 10, 50, 50],
    [15, 15, 55, 55],  # High overlap with first box
    [100, 100, 140, 140],
    [105, 105, 145, 145],  # High overlap with third box
    [200, 200, 240, 240]
]).float()

test_scores = torch.tensor([0.9, 0.8, 0.95, 0.7, 0.85])

keep_indices = simple_nms(test_boxes, test_scores, iou_threshold=0.5)
print(f"Original boxes: {len(test_boxes)}")
print(f"After NMS: {len(keep_indices)}")
print(f"Kept indices: {keep_indices}")
print(f"Kept scores: {test_scores[keep_indices]}")

# 4. BASIC OBJECT DETECTION NETWORK
print("\n4. BASIC OBJECT DETECTION NETWORK")

class SimpleDetectionBackbone(nn.Module):
    """Simple CNN backbone for object detection"""
    
    def __init__(self, in_channels: int = 3):
        super(SimpleDetectionBackbone, self).__init__()
        
        # Feature extraction layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 1/2 resolution
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 1/4 resolution
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 1/8 resolution
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return [x2, x3, x4]  # Multi-scale features

class DetectionHead(nn.Module):
    """Detection head for classification and regression"""
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super(DetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolution layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, 3, padding=1)
        
        # Regression head (4 coordinates per anchor)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, 3, padding=1)
        
        # Objectness head (1 score per anchor)
        self.obj_head = nn.Conv2d(256, num_anchors, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_features = self.shared_conv(x)
        
        # Classification predictions
        cls_pred = self.cls_head(shared_features)
        batch_size, _, height, width = cls_pred.shape
        cls_pred = cls_pred.view(batch_size, self.num_anchors, self.num_classes, height, width)
        cls_pred = cls_pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Regression predictions
        reg_pred = self.reg_head(shared_features)
        reg_pred = reg_pred.view(batch_size, self.num_anchors, 4, height, width)
        reg_pred = reg_pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Objectness predictions
        obj_pred = self.obj_head(shared_features)
        obj_pred = obj_pred.view(batch_size, self.num_anchors, height, width)
        
        return cls_pred, reg_pred, obj_pred

class SimpleObjectDetector(nn.Module):
    """Simple object detection network"""
    
    def __init__(self, num_classes: int = 80, num_anchors: int = 3):
        super(SimpleObjectDetector, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = SimpleDetectionBackbone()
        
        # Detection heads for different feature levels
        self.detection_heads = nn.ModuleList([
            DetectionHead(128, num_classes, num_anchors),  # For layer2 output
            DetectionHead(256, num_classes, num_anchors),  # For layer3 output
            DetectionHead(512, num_classes, num_anchors),  # For layer4 output
        ])
        
    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply detection heads
        detections = []
        for feature, head in zip(features, self.detection_heads):
            cls_pred, reg_pred, obj_pred = head(feature)
            detections.append((cls_pred, reg_pred, obj_pred))
            
        return detections

# Test detection network
detector = SimpleObjectDetector(num_classes=20, num_anchors=3)
test_input = torch.randn(2, 3, 416, 416)  # Batch of 2 images
detections = detector(test_input)

print(f"Number of detection levels: {len(detections)}")
for i, (cls_pred, reg_pred, obj_pred) in enumerate(detections):
    print(f"Level {i}:")
    print(f"  Classification: {cls_pred.shape}")
    print(f"  Regression: {reg_pred.shape}")
    print(f"  Objectness: {obj_pred.shape}")

# 5. LOSS FUNCTIONS FOR OBJECT DETECTION
print("\n5. LOSS FUNCTIONS FOR OBJECT DETECTION")

class DetectionLoss(nn.Module):
    """Combined loss for object detection"""
    
    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2.0):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for classification"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def smooth_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                      mask: torch.Tensor) -> torch.Tensor:
        """Smooth L1 loss for bounding box regression"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        diff = pred[mask] - target[mask]
        abs_diff = torch.abs(diff)
        
        loss = torch.where(abs_diff < 1.0, 
                          0.5 * diff ** 2,
                          abs_diff - 0.5)
        return loss.mean()
    
    def forward(self, predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate total detection loss
        
        Args:
            predictions: List of (cls_pred, reg_pred, obj_pred) for each level
            targets: Dictionary containing 'boxes', 'labels', 'objectness'
        """
        total_cls_loss = 0
        total_reg_loss = 0
        total_obj_loss = 0
        
        for cls_pred, reg_pred, obj_pred in predictions:
            # Flatten predictions
            batch_size = cls_pred.size(0)
            cls_pred_flat = cls_pred.view(batch_size, -1, self.num_classes)
            reg_pred_flat = reg_pred.view(batch_size, -1, 4)
            obj_pred_flat = obj_pred.view(batch_size, -1)
            
            # For demonstration, create dummy targets
            num_predictions = cls_pred_flat.size(1)
            dummy_cls_targets = torch.randint(0, self.num_classes, (batch_size, num_predictions))
            dummy_reg_targets = torch.randn(batch_size, num_predictions, 4)
            dummy_obj_targets = torch.randint(0, 2, (batch_size, num_predictions)).float()
            
            # Calculate losses
            cls_loss = self.focal_loss(cls_pred_flat.view(-1, self.num_classes),
                                     dummy_cls_targets.view(-1))
            
            # Only calculate regression loss for positive samples
            pos_mask = dummy_obj_targets.bool()
            reg_loss = self.smooth_l1_loss(reg_pred_flat, dummy_reg_targets, pos_mask)
            
            obj_loss = F.binary_cross_entropy_with_logits(obj_pred_flat, dummy_obj_targets)
            
            total_cls_loss += cls_loss
            total_reg_loss += reg_loss
            total_obj_loss += obj_loss
        
        return {
            'classification_loss': total_cls_loss,
            'regression_loss': total_reg_loss,
            'objectness_loss': total_obj_loss,
            'total_loss': total_cls_loss + total_reg_loss + total_obj_loss
        }

# Test detection loss
loss_fn = DetectionLoss(num_classes=20)
dummy_targets = {}  # Would contain actual targets in real scenario

loss_dict = loss_fn(detections, dummy_targets)
print("Detection losses:")
for loss_name, loss_value in loss_dict.items():
    print(f"  {loss_name}: {loss_value.item():.4f}")

# 6. POST-PROCESSING AND INFERENCE
print("\n6. POST-PROCESSING AND INFERENCE")

class DetectionPostProcessor:
    """Post-process detection outputs to get final predictions"""
    
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.5,
                 max_detections: int = 100):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
    def decode_predictions(self, cls_pred: torch.Tensor, reg_pred: torch.Tensor,
                          obj_pred: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode network predictions to bounding boxes"""
        device = cls_pred.device
        batch_size = cls_pred.size(0)
        
        # Apply sigmoid to objectness and classification
        obj_scores = torch.sigmoid(obj_pred)
        cls_scores = torch.sigmoid(cls_pred)
        
        # Decode bounding box predictions (simplified)
        # In practice, this would involve anchor-based decoding
        decoded_boxes = reg_pred  # Simplified for demonstration
        
        return decoded_boxes, cls_scores, obj_scores
    
    def filter_predictions(self, boxes: torch.Tensor, cls_scores: torch.Tensor,
                          obj_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter predictions based on confidence threshold"""
        # Get maximum class scores and corresponding class indices
        max_cls_scores, class_indices = torch.max(cls_scores, dim=-1)
        
        # Combine objectness and classification confidence
        confidence_scores = obj_scores * max_cls_scores
        
        # Filter by confidence threshold
        mask = confidence_scores > self.conf_threshold
        
        filtered_boxes = boxes[mask]
        filtered_scores = confidence_scores[mask]
        filtered_classes = class_indices[mask]
        
        return filtered_boxes, filtered_scores, filtered_classes
    
    def apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
                  classes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return boxes, scores, classes
        
        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_classes = []
        
        unique_classes = torch.unique(classes)
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            # Apply NMS using torchvision
            keep_indices = nms(cls_boxes, cls_scores, self.nms_threshold)
            
            final_boxes.append(cls_boxes[keep_indices])
            final_scores.append(cls_scores[keep_indices])
            final_classes.append(classes[cls_mask][keep_indices])
        
        if final_boxes:
            final_boxes = torch.cat(final_boxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
            final_classes = torch.cat(final_classes, dim=0)
            
            # Sort by scores and keep top detections
            sorted_indices = torch.argsort(final_scores, descending=True)[:self.max_detections]
            final_boxes = final_boxes[sorted_indices]
            final_scores = final_scores[sorted_indices]
            final_classes = final_classes[sorted_indices]
        else:
            final_boxes = torch.empty((0, 4))
            final_scores = torch.empty((0,))
            final_classes = torch.empty((0,))
        
        return final_boxes, final_scores, final_classes
    
    def process_batch(self, predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     anchors_list: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Process entire batch of predictions"""
        batch_results = []
        
        # For demonstration, process first level only
        cls_pred, reg_pred, obj_pred = predictions[0]
        anchors = anchors_list[0] if anchors_list else None
        
        batch_size = cls_pred.size(0)
        
        for i in range(batch_size):
            # Extract single image predictions
            single_cls = cls_pred[i]
            single_reg = reg_pred[i]
            single_obj = obj_pred[i]
            
            # Flatten spatial dimensions
            single_cls = single_cls.view(-1, single_cls.size(-1))
            single_reg = single_reg.view(-1, 4)
            single_obj = single_obj.view(-1)
            
            # Decode predictions
            if anchors is not None:
                boxes, cls_scores, obj_scores = self.decode_predictions(
                    single_cls, single_reg, single_obj, anchors)
            else:
                boxes, cls_scores, obj_scores = single_reg, single_cls, single_obj
            
            # Filter and apply NMS
            filtered_boxes, filtered_scores, filtered_classes = self.filter_predictions(
                boxes, cls_scores, obj_scores)
            
            final_boxes, final_scores, final_classes = self.apply_nms(
                filtered_boxes, filtered_scores, filtered_classes)
            
            batch_results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_classes
            })
        
        return batch_results

# Test post-processing
post_processor = DetectionPostProcessor(conf_threshold=0.3, nms_threshold=0.5)
results = post_processor.process_batch(detections, multi_level_anchors)

print(f"Batch results: {len(results)} images")
for i, result in enumerate(results):
    print(f"Image {i}: {len(result['boxes'])} detections")
    if len(result['boxes']) > 0:
        print(f"  Scores: {result['scores'][:5]}")  # Show first 5 scores

# 7. EVALUATION METRICS
print("\n7. EVALUATION METRICS")

class DetectionMetrics:
    """Calculate evaluation metrics for object detection"""
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75]):
        self.iou_thresholds = iou_thresholds
        
    def calculate_ap(self, precisions: torch.Tensor, recalls: torch.Tensor) -> float:
        """Calculate Average Precision using 11-point interpolation"""
        ap = 0.0
        for t in torch.arange(0, 1.1, 0.1):
            if torch.sum(recalls >= t) == 0:
                p = 0
            else:
                p = torch.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap.item()
    
    def evaluate_single_class(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                             gt_boxes: torch.Tensor, iou_threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate detection for single class"""
        if len(pred_boxes) == 0:
            return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Sort predictions by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # Calculate IoU matrix
        if len(gt_boxes) > 0:
            ious = calculate_iou(pred_boxes, gt_boxes)
        else:
            ious = torch.zeros(len(pred_boxes), 0)
        
        # Assign predictions to ground truth
        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool) if len(gt_boxes) > 0 else torch.tensor([])
        
        for i in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
                
            # Find best matching ground truth
            max_iou, max_idx = torch.max(ious[i], dim=0)
            
            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                tp[i] = 1
                gt_matched[max_idx] = True
            else:
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else tp_cumsum
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP
        ap = self.calculate_ap(precisions, recalls)
        
        final_precision = precisions[-1].item() if len(precisions) > 0 else 0.0
        final_recall = recalls[-1].item() if len(recalls) > 0 else 0.0
        
        return {
            'ap': ap,
            'precision': final_precision,
            'recall': final_recall
        }

# Test evaluation metrics
metrics = DetectionMetrics()

# Create dummy predictions and ground truth
dummy_pred_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 40, 40]]).float()
dummy_pred_scores = torch.tensor([0.9, 0.8, 0.7])
dummy_gt_boxes = torch.tensor([[15, 15, 45, 45], [65, 65, 95, 95]]).float()

evaluation_results = metrics.evaluate_single_class(dummy_pred_boxes, dummy_pred_scores, dummy_gt_boxes)
print("Evaluation results:")
for metric, value in evaluation_results.items():
    print(f"  {metric}: {value:.4f}")

print("\n=== OBJECT DETECTION BASICS COMPLETE ===")
print("Key concepts covered:")
print("- Bounding box operations and IoU calculation")
print("- Anchor generation for multiple scales")
print("- Non-Maximum Suppression (NMS)")
print("- Basic object detection network architecture")
print("- Detection loss functions (Focal loss, Smooth L1)")
print("- Post-processing and inference pipeline")
print("- Evaluation metrics (AP, Precision, Recall)")