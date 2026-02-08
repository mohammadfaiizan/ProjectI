"""
PyTorch Instance Segmentation - Mask R-CNN and Instance Detection
Comprehensive guide to instance segmentation implementation in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.ops import roi_align, nms
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import math

print("=== INSTANCE SEGMENTATION ===")

# 1. REGION PROPOSAL NETWORK (RPN)
print("\n1. REGION PROPOSAL NETWORK (RPN)")

class AnchorGenerator:
    """Generate anchor boxes for RPN"""
    
    def __init__(self, sizes: List[int] = [128, 256, 512],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0]):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
    def generate_anchors(self, feature_height: int, feature_width: int, 
                        stride: int = 16) -> torch.Tensor:
        """Generate anchors for given feature map size"""
        anchors_per_location = len(self.sizes) * len(self.aspect_ratios)
        
        # Create base anchors
        base_anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                area = size ** 2
                width = math.sqrt(area / ratio)
                height = width * ratio
                base_anchors.append([-width/2, -height/2, width/2, height/2])
        
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32)
        
        # Create grid centers
        shifts_x = torch.arange(feature_width) * stride
        shifts_y = torch.arange(feature_height) * stride
        
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2)
        shifts = shifts.view(-1, 4)
        
        # Apply shifts to base anchors
        anchors = base_anchors.view(1, anchors_per_location, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.view(-1, 4)
        
        return anchors

class RPNHead(nn.Module):
    """Region Proposal Network Head"""
    
    def __init__(self, in_channels: int, num_anchors: int):
        super(RPNHead, self).__init__()
        
        # Shared convolution
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        # Classification head (object/background)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        
        # Regression head (bbox refinement)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Initialize weights
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
            
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through RPN"""
        logits = []
        bbox_reg = []
        
        for feature in features:
            # Shared convolution
            x = F.relu(self.conv(feature))
            
            # Classification
            logit = self.cls_logits(x)
            logits.append(logit)
            
            # Regression
            bbox = self.bbox_pred(x)
            bbox_reg.append(bbox)
            
        return logits, bbox_reg

# Test RPN components
anchor_gen = AnchorGenerator()
anchors = anchor_gen.generate_anchors(32, 32, stride=16)
print(f"Generated {len(anchors)} anchors")
print(f"Sample anchors: {anchors[:3]}")

# Test RPN head
feature_maps = [torch.randn(1, 256, 32, 32), torch.randn(1, 256, 16, 16)]
rpn_head = RPNHead(in_channels=256, num_anchors=9)
logits, bbox_reg = rpn_head(feature_maps)

print(f"RPN outputs:")
for i, (logit, bbox) in enumerate(zip(logits, bbox_reg)):
    print(f"  Level {i}: Logits {logit.shape}, BBox {bbox.shape}")

# 2. ROI POOLING AND ALIGNMENT
print("\n2. ROI POOLING AND ALIGNMENT")

class ROIAlign(nn.Module):
    """ROI Align layer for extracting features from regions"""
    
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float, sampling_ratio: int = 2):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        
    def forward(self, features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Apply ROI Align
        Args:
            features: Feature map [N, C, H, W]
            boxes: ROI boxes [num_rois, 5] where first column is batch index
        """
        return roi_align(
            features, boxes, self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio
        )

# Test ROI Align
roi_align_layer = ROIAlign(output_size=(7, 7), spatial_scale=1/16.0)

# Create dummy feature map and ROI boxes
feature_map = torch.randn(2, 256, 32, 32)
roi_boxes = torch.tensor([
    [0, 100, 100, 200, 200],  # batch_idx=0
    [0, 150, 150, 250, 250],  # batch_idx=0
    [1, 50, 50, 150, 150],    # batch_idx=1
]).float()

roi_features = roi_align_layer(feature_map, roi_boxes)
print(f"ROI features shape: {roi_features.shape}")

# 3. MASK HEAD IMPLEMENTATION
print("\n3. MASK HEAD IMPLEMENTATION")

class MaskHead(nn.Module):
    """Mask prediction head for instance segmentation"""
    
    def __init__(self, in_channels: int, num_classes: int, dim_reduced: int = 256):
        super(MaskHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, dim_reduced, 3, padding=1)
        self.conv2 = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)
        self.conv3 = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)
        self.conv4 = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)
        
        # Upsampling
        self.deconv = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, stride=2)
        
        # Final prediction layer
        self.predictor = nn.Conv2d(dim_reduced, num_classes, 1)
        
        # Initialize weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_normal_(self.predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.predictor.bias, 0)
        
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through mask head
        Args:
            roi_features: ROI-aligned features [num_rois, in_channels, H, W]
        Returns:
            mask_logits: Mask predictions [num_rois, num_classes, 2*H, 2*W]
        """
        x = F.relu(self.conv1(roi_features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Upsample by 2x
        x = F.relu(self.deconv(x))
        
        # Final prediction
        mask_logits = self.predictor(x)
        
        return mask_logits

# Test mask head
mask_head = MaskHead(in_channels=256, num_classes=80)
mask_predictions = mask_head(roi_features)
print(f"Mask predictions shape: {mask_predictions.shape}")

# 4. FEATURE PYRAMID NETWORK (FPN)
print("\n4. FEATURE PYRAMID NETWORK (FPN)")

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super(FPN, self).__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
        
        # Output convolutions
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
        
        # Initialize weights
        for module in [self.lateral_convs, self.output_convs]:
            for layer in module:
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)
                
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through FPN
        Args:
            features: List of feature maps from backbone [low_res -> high_res]
        Returns:
            fpn_features: List of FPN feature maps
        """
        # Apply lateral connections
        laterals = []
        for feature, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feature))
        
        # Top-down pathway
        fpn_features = []
        prev_feature = laterals[-1]  # Start from highest level
        
        for i in range(len(laterals) - 1, -1, -1):
            if i < len(laterals) - 1:
                # Upsample and add
                upsampled = F.interpolate(
                    prev_feature, size=laterals[i].shape[2:], 
                    mode='nearest'
                )
                prev_feature = laterals[i] + upsampled
            else:
                prev_feature = laterals[i]
            
            # Apply output convolution
            fpn_feature = self.output_convs[i](prev_feature)
            fpn_features.insert(0, fpn_feature)
        
        return fpn_features

# Test FPN
backbone_features = [
    torch.randn(1, 256, 64, 64),   # C2
    torch.randn(1, 512, 32, 32),   # C3  
    torch.randn(1, 1024, 16, 16),  # C4
    torch.randn(1, 2048, 8, 8),    # C5
]

fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
fpn_features = fpn(backbone_features)

print(f"FPN features:")
for i, feature in enumerate(fpn_features):
    print(f"  Level {i}: {feature.shape}")

# 5. SIMPLE MASK R-CNN IMPLEMENTATION
print("\n5. SIMPLE MASK R-CNN IMPLEMENTATION")

class SimpleMaskRCNN(nn.Module):
    """Simplified Mask R-CNN implementation"""
    
    def __init__(self, num_classes: int = 80, backbone_channels: List[int] = [256, 512, 1024, 2048]):
        super(SimpleMaskRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature Pyramid Network
        self.fpn = FPN(backbone_channels, out_channels=256)
        
        # RPN
        self.rpn_head = RPNHead(in_channels=256, num_anchors=3)
        
        # ROI operations
        self.roi_align = ROIAlign(output_size=(7, 7), spatial_scale=1/16.0)
        
        # Classification and regression heads
        self.cls_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes + 1)  # +1 for background
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, (num_classes + 1) * 4)
        )
        
        # Mask head
        self.mask_head = MaskHead(in_channels=256, num_classes=num_classes + 1)
        
    def forward(self, features: List[torch.Tensor], proposals: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Mask R-CNN
        Args:
            features: Backbone features
            proposals: ROI proposals [num_rois, 5] (batch_idx, x1, y1, x2, y2)
        """
        # Extract FPN features
        fpn_features = self.fpn(features)
        
        # RPN predictions
        rpn_logits, rpn_bbox_reg = self.rpn_head(fpn_features)
        
        if proposals is None:
            # Use dummy proposals for demonstration
            batch_size = features[0].size(0)
            proposals = torch.tensor([
                [0, 100, 100, 200, 200],
                [0, 150, 150, 250, 250],
            ]).float()
        
        # ROI pooling from appropriate FPN level
        roi_features = self.roi_align(fpn_features[0], proposals)  # Use P2 for simplicity
        
        # Flatten ROI features
        roi_features_flat = roi_features.view(roi_features.size(0), -1)
        
        # Classification and regression
        cls_scores = self.cls_head(roi_features_flat)
        bbox_deltas = self.bbox_head(roi_features_flat)
        
        # Mask prediction
        mask_logits = self.mask_head(roi_features)
        
        return {
            'rpn_logits': rpn_logits,
            'rpn_bbox_reg': rpn_bbox_reg,
            'cls_scores': cls_scores,
            'bbox_deltas': bbox_deltas,
            'mask_logits': mask_logits
        }

# Test Mask R-CNN
mask_rcnn = SimpleMaskRCNN(num_classes=80)
outputs = mask_rcnn(backbone_features)

print("Mask R-CNN outputs:")
for key, value in outputs.items():
    if isinstance(value, list):
        print(f"  {key}: {len(value)} levels")
        for i, v in enumerate(value):
            print(f"    Level {i}: {v.shape}")
    else:
        print(f"  {key}: {value.shape}")

# 6. INSTANCE SEGMENTATION LOSS
print("\n6. INSTANCE SEGMENTATION LOSS")

class MaskRCNNLoss(nn.Module):
    """Combined loss for Mask R-CNN"""
    
    def __init__(self, num_classes: int):
        super(MaskRCNNLoss, self).__init__()
        self.num_classes = num_classes
        
    def rpn_loss(self, rpn_logits: List[torch.Tensor], rpn_bbox_reg: List[torch.Tensor],
                 targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """RPN loss computation"""
        # Simplified RPN loss for demonstration
        total_rpn_cls_loss = 0
        total_rpn_bbox_loss = 0
        
        for logits, bbox_reg in zip(rpn_logits, rpn_bbox_reg):
            batch_size, num_anchors, height, width = logits.shape
            
            # Create dummy targets
            rpn_labels = torch.randint(0, 2, (batch_size, num_anchors, height, width)).float()
            rpn_bbox_targets = torch.randn(batch_size, num_anchors * 4, height, width)
            
            # Classification loss
            rpn_cls_loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), rpn_labels.view(-1)
            )
            
            # Regression loss (only for positive samples)
            pos_mask = rpn_labels.view(-1) > 0
            if pos_mask.sum() > 0:
                rpn_bbox_loss = F.smooth_l1_loss(
                    bbox_reg.view(-1, 4)[pos_mask],
                    rpn_bbox_targets.view(-1, 4)[pos_mask]
                )
            else:
                rpn_bbox_loss = torch.tensor(0.0, device=logits.device)
            
            total_rpn_cls_loss += rpn_cls_loss
            total_rpn_bbox_loss += rpn_bbox_loss
        
        return {
            'rpn_cls_loss': total_rpn_cls_loss,
            'rpn_bbox_loss': total_rpn_bbox_loss
        }
    
    def rcnn_loss(self, cls_scores: torch.Tensor, bbox_deltas: torch.Tensor,
                  targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """RCNN classification and regression loss"""
        num_rois = cls_scores.size(0)
        
        # Create dummy targets
        labels = torch.randint(0, self.num_classes + 1, (num_rois,))
        bbox_targets = torch.randn(num_rois, 4)
        
        # Classification loss
        cls_loss = F.cross_entropy(cls_scores, labels)
        
        # Regression loss (only for foreground)
        fg_mask = labels > 0
        if fg_mask.sum() > 0:
            # Select bbox deltas for the correct class
            bbox_deltas_selected = bbox_deltas.view(num_rois, self.num_classes + 1, 4)
            bbox_deltas_selected = bbox_deltas_selected[torch.arange(num_rois), labels]
            
            bbox_loss = F.smooth_l1_loss(
                bbox_deltas_selected[fg_mask],
                bbox_targets[fg_mask]
            )
        else:
            bbox_loss = torch.tensor(0.0, device=cls_scores.device)
        
        return {
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss
        }
    
    def mask_loss(self, mask_logits: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mask segmentation loss"""
        num_rois, num_classes_plus_bg, mask_height, mask_width = mask_logits.shape
        
        # Create dummy mask targets
        mask_targets = torch.randint(0, 2, (num_rois, mask_height, mask_width)).float()
        labels = torch.randint(0, self.num_classes + 1, (num_rois,))
        
        # Only compute loss for foreground ROIs
        fg_mask = labels > 0
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=mask_logits.device)
        
        # Select mask predictions for the correct class
        fg_labels = labels[fg_mask]
        fg_mask_logits = mask_logits[fg_mask]
        fg_mask_targets = mask_targets[fg_mask]
        
        # Select the appropriate class channel
        selected_mask_logits = fg_mask_logits[torch.arange(fg_labels.size(0)), fg_labels]
        
        # Binary cross-entropy loss
        mask_loss = F.binary_cross_entropy_with_logits(
            selected_mask_logits, fg_mask_targets
        )
        
        return mask_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute total Mask R-CNN loss"""
        # RPN losses
        rpn_losses = self.rpn_loss(
            predictions['rpn_logits'], 
            predictions['rpn_bbox_reg'], 
            targets
        )
        
        # RCNN losses
        rcnn_losses = self.rcnn_loss(
            predictions['cls_scores'], 
            predictions['bbox_deltas'], 
            targets
        )
        
        # Mask loss
        mask_loss = self.mask_loss(predictions['mask_logits'], targets)
        
        # Combine all losses
        total_loss = (rpn_losses['rpn_cls_loss'] + 
                     rpn_losses['rpn_bbox_loss'] + 
                     rcnn_losses['cls_loss'] + 
                     rcnn_losses['bbox_loss'] + 
                     mask_loss)
        
        return {
            'total_loss': total_loss,
            'rpn_cls_loss': rpn_losses['rpn_cls_loss'],
            'rpn_bbox_loss': rpn_losses['rpn_bbox_loss'],
            'cls_loss': rcnn_losses['cls_loss'],
            'bbox_loss': rcnn_losses['bbox_loss'],
            'mask_loss': mask_loss
        }

# Test Mask R-CNN loss
loss_fn = MaskRCNNLoss(num_classes=80)
dummy_targets = {}  # Would contain actual targets in practice

loss_dict = loss_fn(outputs, dummy_targets)
print("Mask R-CNN losses:")
for loss_name, loss_value in loss_dict.items():
    print(f"  {loss_name}: {loss_value.item():.4f}")

# 7. POST-PROCESSING FOR INFERENCE
print("\n7. POST-PROCESSING FOR INFERENCE")

class MaskRCNNPostProcessor:
    """Post-processing for Mask R-CNN inference"""
    
    def __init__(self, score_threshold: float = 0.5, nms_threshold: float = 0.5,
                 max_detections: int = 100, mask_threshold: float = 0.5):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.mask_threshold = mask_threshold
        
    def process_boxes_and_scores(self, cls_scores: torch.Tensor, bbox_deltas: torch.Tensor,
                                proposals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process box predictions and scores"""
        # Apply softmax to get probabilities
        scores = F.softmax(cls_scores, dim=1)
        
        # Remove background class
        scores = scores[:, 1:]  # Exclude background (class 0)
        num_classes = scores.size(1)
        
        # Decode bounding box deltas (simplified)
        # In practice, this would apply the deltas to proposal boxes
        boxes = proposals[:, 1:]  # Remove batch index, keep [x1, y1, x2, y2]
        
        # Get the best class for each proposal
        max_scores, predicted_classes = torch.max(scores, dim=1)
        
        # Filter by score threshold
        keep_mask = max_scores > self.score_threshold
        boxes = boxes[keep_mask]
        scores = max_scores[keep_mask]
        labels = predicted_classes[keep_mask] + 1  # +1 to account for background removal
        
        return boxes, scores, labels
    
    def process_masks(self, mask_logits: torch.Tensor, boxes: torch.Tensor,
                     labels: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        """Process mask predictions"""
        if len(boxes) == 0:
            return torch.empty((0, *original_size), dtype=torch.bool)
        
        # Select masks for predicted classes
        num_rois = mask_logits.size(0)
        selected_masks = mask_logits[torch.arange(num_rois), labels]
        
        # Apply sigmoid and threshold
        mask_probs = torch.sigmoid(selected_masks)
        masks = mask_probs > self.mask_threshold
        
        # Resize masks to original image size (simplified)
        # In practice, this would involve proper resizing and cropping
        
        return masks.bool()
    
    def apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
                  labels: torch.Tensor) -> torch.Tensor:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        keep_indices = nms(boxes, scores, self.nms_threshold)
        
        # Limit to max detections
        if len(keep_indices) > self.max_detections:
            keep_indices = keep_indices[:self.max_detections]
        
        return keep_indices
    
    def __call__(self, predictions: Dict[str, torch.Tensor], proposals: torch.Tensor,
                 original_size: Tuple[int, int] = (512, 512)) -> Dict[str, torch.Tensor]:
        """Complete post-processing pipeline"""
        # Process boxes and scores
        boxes, scores, labels = self.process_boxes_and_scores(
            predictions['cls_scores'],
            predictions['bbox_deltas'], 
            proposals
        )
        
        if len(boxes) == 0:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long),
                'masks': torch.empty((0, *original_size), dtype=torch.bool)
            }
        
        # Apply NMS
        keep_indices = self.apply_nms(boxes, scores, labels)
        
        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices]
        final_labels = labels[keep_indices]
        
        # Process masks
        kept_mask_logits = predictions['mask_logits'][keep_indices]
        final_masks = self.process_masks(
            kept_mask_logits, final_boxes, final_labels, original_size
        )
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels,
            'masks': final_masks
        }

# Test post-processing
post_processor = MaskRCNNPostProcessor(score_threshold=0.3, nms_threshold=0.5)

# Create dummy proposals
dummy_proposals = torch.tensor([
    [0, 100, 100, 200, 200],
    [0, 150, 150, 250, 250],
]).float()

final_predictions = post_processor(outputs, dummy_proposals, original_size=(512, 512))

print("Final predictions:")
for key, value in final_predictions.items():
    print(f"  {key}: {value.shape}")

# 8. EVALUATION METRICS FOR INSTANCE SEGMENTATION
print("\n8. EVALUATION METRICS FOR INSTANCE SEGMENTATION")

class InstanceSegmentationMetrics:
    """Evaluation metrics for instance segmentation"""
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75], num_classes: int = 80):
        self.iou_thresholds = iou_thresholds
        self.num_classes = num_classes
        
    def calculate_mask_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Calculate IoU between two binary masks"""
        intersection = (pred_mask & gt_mask).float().sum()
        union = (pred_mask | gt_mask).float().sum()
        
        if union == 0:
            return 0.0
        
        return (intersection / union).item()
    
    def calculate_bbox_iou(self, pred_box: torch.Tensor, gt_box: torch.Tensor) -> float:
        """Calculate IoU between two bounding boxes"""
        # Extract coordinates
        x1_pred, y1_pred, x2_pred, y2_pred = pred_box
        x1_gt, y1_gt, x2_gt, y2_gt = gt_box
        
        # Calculate intersection
        x1_inter = max(x1_pred, x1_gt)
        y1_inter = max(y1_pred, y1_gt)
        x2_inter = min(x2_pred, x2_gt)
        y2_inter = min(y2_pred, y2_gt)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union = area_pred + area_gt - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_detection(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                          pred_labels: torch.Tensor, gt_boxes: torch.Tensor,
                          gt_labels: torch.Tensor, iou_threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate detection performance (AP for bounding boxes)"""
        if len(pred_boxes) == 0:
            return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Sort predictions by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]
        
        # Match predictions to ground truth
        num_gt = len(gt_boxes)
        gt_matched = torch.zeros(num_gt, dtype=torch.bool)
        
        tp = []
        fp = []
        
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if pred_label != gt_label:
                    continue
                    
                iou = self.calculate_bbox_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate precision and recall
        tp_cumsum = torch.cumsum(torch.tensor(tp), dim=0)
        fp_cumsum = torch.cumsum(torch.tensor(fp), dim=0)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / num_gt if num_gt > 0 else tp_cumsum
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in torch.arange(0, 1.1, 0.1):
            if torch.sum(recalls >= t) == 0:
                p = 0
            else:
                p = torch.max(precisions[recalls >= t])
            ap += p / 11.0
        
        final_precision = precisions[-1].item() if len(precisions) > 0 else 0.0
        final_recall = recalls[-1].item() if len(recalls) > 0 else 0.0
        
        return {
            'ap': ap.item() if isinstance(ap, torch.Tensor) else ap,
            'precision': final_precision,
            'recall': final_recall
        }
    
    def evaluate_segmentation(self, pred_masks: torch.Tensor, pred_labels: torch.Tensor,
                             gt_masks: torch.Tensor, gt_labels: torch.Tensor,
                             iou_threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate instance segmentation performance"""
        if len(pred_masks) == 0:
            return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Match predictions to ground truth based on mask IoU
        num_gt = len(gt_masks)
        gt_matched = torch.zeros(num_gt, dtype=torch.bool)
        
        tp = []
        fp = []
        
        for pred_mask, pred_label in zip(pred_masks, pred_labels):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, (gt_mask, gt_label) in enumerate(zip(gt_masks, gt_labels)):
                if pred_label != gt_label:
                    continue
                    
                iou = self.calculate_mask_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate metrics similar to detection
        tp_cumsum = torch.cumsum(torch.tensor(tp), dim=0)
        fp_cumsum = torch.cumsum(torch.tensor(fp), dim=0)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / num_gt if num_gt > 0 else tp_cumsum
        
        # Calculate AP
        ap = 0.0
        for t in torch.arange(0, 1.1, 0.1):
            if torch.sum(recalls >= t) == 0:
                p = 0
            else:
                p = torch.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return {
            'ap': ap.item() if isinstance(ap, torch.Tensor) else ap,
            'precision': precisions[-1].item() if len(precisions) > 0 else 0.0,
            'recall': recalls[-1].item() if len(recalls) > 0 else 0.0
        }

# Test evaluation metrics
metrics = InstanceSegmentationMetrics()

# Create dummy predictions and ground truth
dummy_pred_boxes = torch.tensor([[100, 100, 200, 200], [150, 150, 250, 250]]).float()
dummy_pred_scores = torch.tensor([0.9, 0.8])
dummy_pred_labels = torch.tensor([1, 2])
dummy_pred_masks = torch.randint(0, 2, (2, 64, 64)).bool()

dummy_gt_boxes = torch.tensor([[110, 110, 210, 210], [160, 160, 260, 260]]).float()
dummy_gt_labels = torch.tensor([1, 2])
dummy_gt_masks = torch.randint(0, 2, (2, 64, 64)).bool()

detection_results = metrics.evaluate_detection(
    dummy_pred_boxes, dummy_pred_scores, dummy_pred_labels,
    dummy_gt_boxes, dummy_gt_labels
)

segmentation_results = metrics.evaluate_segmentation(
    dummy_pred_masks, dummy_pred_labels,
    dummy_gt_masks, dummy_gt_labels
)

print("Detection evaluation:")
for metric, value in detection_results.items():
    print(f"  {metric}: {value:.4f}")

print("Segmentation evaluation:")
for metric, value in segmentation_results.items():
    print(f"  {metric}: {value:.4f}")

print("\n=== INSTANCE SEGMENTATION COMPLETE ===")
print("Key concepts covered:")
print("- Region Proposal Network (RPN)")
print("- ROI Pooling and Alignment")
print("- Mask prediction head")
print("- Feature Pyramid Network (FPN)")
print("- Mask R-CNN architecture")
print("- Instance segmentation loss functions")
print("- Post-processing for inference")
print("- Evaluation metrics for instance segmentation")