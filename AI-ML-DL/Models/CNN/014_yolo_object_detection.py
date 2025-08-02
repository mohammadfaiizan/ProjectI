"""
ERA 6: SPECIALIZED APPLICATIONS - YOLO Object Detection
======================================================

Year: 2015-Present (YOLOv1-v8 evolution)
Paper: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2015)
Innovation: Single-shot object detection with unified end-to-end training
Previous Limitation: Two-stage detectors (R-CNN family) too slow for real-time applications
Performance Gain: Real-time detection (45+ FPS), simpler pipeline, end-to-end optimization
Impact: Revolutionized real-time object detection, enabled practical deployment in robotics/autonomous systems

This file implements YOLO (You Only Look Once) object detection that transformed computer vision
by enabling real-time object detection through a unified single-shot approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time
import psutil
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2015-Present"
INNOVATION = "Single-shot object detection with unified end-to-end training"
PREVIOUS_LIMITATION = "Two-stage detectors too slow, complex pipeline, no end-to-end optimization"
IMPACT = "Revolutionized real-time detection, enabled practical deployment, simplified pipeline"

print(f"=== YOLO Object Detection ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# DATASET SIMULATION (CIFAR-10 with DETECTION LABELS)
# ============================================================================

def create_detection_dataset():
    """
    Create a simplified object detection dataset using CIFAR-10
    Simulate bounding boxes for demonstration purposes
    """
    print("Creating object detection dataset from CIFAR-10...")
    
    # YOLO-style transforms
    transform_train = transforms.Compose([
        transforms.Resize(416),  # YOLO input size
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(416),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # CIFAR-10 classes for object detection
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Detection dataset created:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Classes: {len(classes)}")
    print(f"  Image size: 416x416 (YOLO standard)")
    print(f"  Task: Object detection and classification")
    
    return train_loader, test_loader, classes

def generate_pseudo_boxes(batch_size, num_classes, grid_size=13):
    """
    Generate pseudo bounding boxes for demonstration
    In real YOLO, these would come from annotated data
    """
    # Simulate YOLO ground truth format
    # Each cell predicts: [x, y, w, h, confidence, class_probs...]
    target_size = grid_size * grid_size * (5 + num_classes)  # 5 = x,y,w,h,conf
    
    # Create random but realistic targets
    targets = torch.zeros(batch_size, target_size)
    
    for b in range(batch_size):
        # Simulate 1-2 objects per image
        num_objects = np.random.randint(1, 3)
        
        for obj in range(num_objects):
            # Random grid cell
            grid_x = np.random.randint(0, grid_size)
            grid_y = np.random.randint(0, grid_size)
            cell_idx = grid_y * grid_size + grid_x
            
            # Object properties (normalized 0-1)
            x_offset = np.random.uniform(0.2, 0.8)  # Within cell
            y_offset = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.3, 0.9)     # Relative to image
            height = np.random.uniform(0.3, 0.9)
            confidence = 1.0  # Object present
            
            # Random class
            obj_class = np.random.randint(0, num_classes)
            
            # Fill target tensor (simplified YOLO format)
            base_idx = cell_idx * (5 + num_classes)
            targets[b, base_idx:base_idx+5] = torch.tensor([
                x_offset, y_offset, width, height, confidence
            ])
            targets[b, base_idx+5+obj_class] = 1.0  # One-hot class
    
    return targets

# ============================================================================
# YOLO DETECTION HEAD
# ============================================================================

class YOLODetectionHead(nn.Module):
    """
    YOLO Detection Head - Core Innovation
    
    Single-shot detection:
    1. Divide image into SxS grid
    2. Each cell predicts B bounding boxes
    3. Each box predicts: (x, y, w, h, confidence, class_probs)
    4. Non-maximum suppression for final detections
    """
    
    def __init__(self, in_channels, num_classes=10, num_boxes=2, grid_size=13):
        super(YOLODetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid_size = grid_size
        
        # Each grid cell predicts B boxes, each with (x,y,w,h,conf) + class probs
        self.predictions_per_cell = num_boxes * 5 + num_classes
        
        # Detection layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        
        # Fully connected layers for final prediction
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, grid_size * grid_size * self.predictions_per_cell)
        )
        
        print(f"  YOLO Detection Head:")
        print(f"    Grid size: {grid_size}x{grid_size}")
        print(f"    Boxes per cell: {num_boxes}")
        print(f"    Classes: {num_classes}")
        print(f"    Predictions per cell: {self.predictions_per_cell}")
        print(f"    Output size: {grid_size * grid_size * self.predictions_per_cell}")
    
    def forward(self, x):
        """Forward pass through YOLO detection head"""
        # Feature extraction
        x = self.conv_layers(x)
        
        # Detection prediction
        predictions = self.fc_layers(x)
        
        # Reshape to grid format
        batch_size = x.size(0)
        predictions = predictions.view(
            batch_size, self.grid_size, self.grid_size, self.predictions_per_cell
        )
        
        return predictions
    
    def decode_predictions(self, predictions, confidence_threshold=0.5):
        """
        Decode YOLO predictions to bounding boxes
        
        Args:
            predictions: Raw YOLO output (B, S, S, predictions_per_cell)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detections per image [(boxes, scores, classes), ...]
        """
        batch_size = predictions.size(0)
        detections = []
        
        for b in range(batch_size):
            image_detections = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_pred = predictions[b, i, j]
                    
                    # Extract box predictions
                    for box_idx in range(self.num_boxes):
                        base_idx = box_idx * 5
                        
                        # Box parameters
                        x = cell_pred[base_idx + 0]
                        y = cell_pred[base_idx + 1]
                        w = cell_pred[base_idx + 2]
                        h = cell_pred[base_idx + 3]
                        conf = torch.sigmoid(cell_pred[base_idx + 4])
                        
                        if conf > confidence_threshold:
                            # Convert to absolute coordinates
                            x_abs = (j + torch.sigmoid(x)) / self.grid_size
                            y_abs = (i + torch.sigmoid(y)) / self.grid_size
                            w_abs = torch.sigmoid(w)
                            h_abs = torch.sigmoid(h)
                            
                            # Class probabilities
                            class_probs = torch.softmax(
                                cell_pred[self.num_boxes * 5:], dim=0
                            )
                            class_score, class_idx = torch.max(class_probs, dim=0)
                            
                            # Final confidence
                            final_conf = conf * class_score
                            
                            # Bounding box (center format)
                            box = [x_abs.item(), y_abs.item(), w_abs.item(), h_abs.item()]
                            
                            image_detections.append({
                                'box': box,
                                'confidence': final_conf.item(),
                                'class': class_idx.item()
                            })
            
            detections.append(image_detections)
        
        return detections

# ============================================================================
# YOLO BACKBONE NETWORKS
# ============================================================================

class DarkNet19Backbone(nn.Module):
    """
    DarkNet-19 Backbone (YOLOv2 style)
    Efficient backbone for feature extraction
    """
    
    def __init__(self):
        super(DarkNet19Backbone, self).__init__()
        
        print("Building DarkNet-19 Backbone for YOLO...")
        
        # Initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Dark blocks
        self.dark_block1 = self._make_dark_block(32, 64)
        self.dark_block2 = self._make_dark_block(64, 128, num_blocks=2)
        self.dark_block3 = self._make_dark_block(128, 256, num_blocks=8)
        self.dark_block4 = self._make_dark_block(256, 512, num_blocks=8)
        self.dark_block5 = self._make_dark_block(512, 1024, num_blocks=4)
        
        print("  DarkNet-19 feature extraction backbone built")
    
    def _make_dark_block(self, in_channels, out_channels, num_blocks=1):
        """Create a DarkNet block"""
        layers = []
        
        # First conv
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2)
        ])
        
        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels // 2, 1),
                nn.BatchNorm2d(out_channels // 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through DarkNet backbone"""
        x = self.initial(x)
        x = self.dark_block1(x)
        x = self.dark_block2(x)
        x = self.dark_block3(x)
        x = self.dark_block4(x)
        x = self.dark_block5(x)
        
        return x

class CSPDarkNetBackbone(nn.Module):
    """
    CSP-DarkNet Backbone (YOLOv4/v5 style)
    Cross Stage Partial connections for better gradient flow
    """
    
    def __init__(self):
        super(CSPDarkNetBackbone, self).__init__()
        
        print("Building CSP-DarkNet Backbone for YOLO...")
        
        # Focus layer (space to depth)
        self.focus = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),  # 4x space-to-depth gives 12 channels
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        
        # CSP stages
        self.csp1 = self._make_csp_stage(32, 64, num_blocks=1)
        self.csp2 = self._make_csp_stage(64, 128, num_blocks=3)
        self.csp3 = self._make_csp_stage(128, 256, num_blocks=15)
        self.csp4 = self._make_csp_stage(256, 512, num_blocks=15)
        self.csp5 = self._make_csp_stage(512, 1024, num_blocks=7)
        
        print("  CSP-DarkNet backbone with cross-stage partial connections built")
    
    def _make_csp_stage(self, in_channels, out_channels, num_blocks):
        """Create CSP stage with partial connections"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            # Simplified CSP block
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def _space_to_depth(self, x):
        """Space to depth transformation for focus layer"""
        B, C, H, W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C*4, H//2, W//2)
        return x
    
    def forward(self, x):
        """Forward pass through CSP-DarkNet"""
        # Apply space-to-depth focus
        x = self._space_to_depth(x)
        x = self.focus(x)
        
        # CSP stages
        x = self.csp1(x)
        x = self.csp2(x)
        x = self.csp3(x)
        x = self.csp4(x)
        x = self.csp5(x)
        
        return x

# ============================================================================
# COMPLETE YOLO ARCHITECTURE
# ============================================================================

class YOLO_ObjectDetection(nn.Module):
    """
    YOLO (You Only Look Once) Object Detection
    
    Revolutionary Innovation:
    - Single-shot detection (no region proposals)
    - Unified end-to-end training
    - Real-time performance (45+ FPS)
    - Direct coordinate prediction
    """
    
    def __init__(self, num_classes=10, backbone='darknet19', grid_size=13):
        super(YOLO_ObjectDetection, self).__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.backbone_type = backbone
        
        print(f"Building YOLO Object Detection Model...")
        
        # Backbone selection
        if backbone == 'darknet19':
            self.backbone = DarkNet19Backbone()
            backbone_channels = 1024
        elif backbone == 'csp_darknet':
            self.backbone = CSPDarkNetBackbone()
            backbone_channels = 1024
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Detection head
        self.detection_head = YOLODetectionHead(
            backbone_channels, num_classes, num_boxes=2, grid_size=grid_size
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"YOLO Architecture Summary:")
        print(f"  Backbone: {backbone}")
        print(f"  Grid size: {grid_size}x{grid_size}")
        print(f"  Classes: {num_classes}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Single-shot real-time detection")
    
    def _initialize_weights(self):
        """Initialize YOLO weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through YOLO"""
        # Feature extraction
        features = self.backbone(x)
        
        # Object detection
        detections = self.detection_head(features)
        
        return detections
    
    def detect(self, x, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Perform object detection with post-processing
        
        Args:
            x: Input images (B, C, H, W)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: IoU threshold for NMS
            
        Returns:
            List of detections per image
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            predictions = self.forward(x)
            
            # Decode predictions
            detections = self.detection_head.decode_predictions(
                predictions, confidence_threshold
            )
            
            # Apply NMS (simplified version)
            final_detections = []
            for image_dets in detections:
                # Sort by confidence
                image_dets.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Simple NMS (can be improved)
                nms_dets = []
                for det in image_dets:
                    keep = True
                    for existing_det in nms_dets:
                        if self._calculate_iou(det['box'], existing_det['box']) > nms_threshold:
                            keep = False
                            break
                    if keep:
                        nms_dets.append(det)
                
                final_detections.append(nms_dets)
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes (center format)"""
        # Convert center format to corner format
        def center_to_corner(box):
            x, y, w, h = box
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            return [x1, y1, x2, y2]
        
        box1_corner = center_to_corner(box1)
        box2_corner = center_to_corner(box2)
        
        # Calculate intersection
        x1 = max(box1_corner[0], box2_corner[0])
        y1 = max(box1_corner[1], box2_corner[1])
        x2 = min(box1_corner[2], box2_corner[2])
        y2 = min(box1_corner[3], box2_corner[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_corner[2] - box1_corner[0]) * (box1_corner[3] - box1_corner[1])
        area2 = (box2_corner[2] - box2_corner[0]) * (box2_corner[3] - box2_corner[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_detection_analysis(self):
        """Analyze YOLO detection capabilities"""
        grid_cells = self.grid_size ** 2
        max_detections = grid_cells * 2  # 2 boxes per cell
        
        return {
            'architecture': 'Single-shot detector',
            'grid_size': f'{self.grid_size}x{self.grid_size}',
            'grid_cells': grid_cells,
            'max_detections_per_image': max_detections,
            'backbone': self.backbone_type,
            'real_time_capable': True,
            'end_to_end_training': True,
            'innovation': 'Unified detection and classification'
        }

# ============================================================================
# YOLO LOSS FUNCTION
# ============================================================================

class YOLOLoss(nn.Module):
    """
    YOLO Loss Function - Multi-task loss for detection
    
    Components:
    1. Coordinate loss (x, y, w, h)
    2. Confidence loss (objectness)
    3. Classification loss (class probabilities)
    """
    
    def __init__(self, num_classes=10, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # Coordinate loss weight
        self.lambda_noobj = lambda_noobj  # No-object loss weight
        
        print(f"  YOLO Loss Configuration:")
        print(f"    Coordinate weight: {lambda_coord}")
        print(f"    No-object weight: {lambda_noobj}")
        print(f"    Classes: {num_classes}")
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        
        Args:
            predictions: YOLO predictions (B, S, S, predictions_per_cell)
            targets: Ground truth (B, S*S*(5+C))
            
        Returns:
            Total loss, loss components
        """
        batch_size, S, _, pred_size = predictions.shape
        
        # Reshape predictions and targets
        predictions = predictions.view(batch_size, S*S, pred_size)
        targets = targets.view(batch_size, S*S, -1)
        
        # Loss components
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0
        
        for b in range(batch_size):
            for cell in range(S*S):
                pred_cell = predictions[b, cell]
                target_cell = targets[b, cell]
                
                # Check if object exists in this cell
                if target_cell[4] > 0:  # Confidence > 0 means object exists
                    # Coordinate loss
                    pred_coords = pred_cell[:4]
                    target_coords = target_cell[:4]
                    
                    coord_loss += self.lambda_coord * F.mse_loss(
                        pred_coords, target_coords, reduction='sum'
                    )
                    
                    # Confidence loss (object present)
                    pred_conf = torch.sigmoid(pred_cell[4])
                    conf_loss += F.mse_loss(pred_conf, target_cell[4])
                    
                    # Classification loss
                    pred_classes = pred_cell[5:5+self.num_classes]
                    target_classes = target_cell[5:5+self.num_classes]
                    
                    class_loss += F.mse_loss(pred_classes, target_classes)
                
                else:
                    # No object - only penalize confidence
                    pred_conf = torch.sigmoid(pred_cell[4])
                    conf_loss += self.lambda_noobj * F.mse_loss(
                        pred_conf, torch.tensor(0.0).to(pred_conf.device)
                    )
        
        total_loss = coord_loss + conf_loss + class_loss
        
        return total_loss, {
            'coordinate_loss': coord_loss,
            'confidence_loss': conf_loss,
            'classification_loss': class_loss,
            'total_loss': total_loss
        }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_yolo_detection(model, train_loader, test_loader, epochs=50, learning_rate=1e-3):
    """Train YOLO object detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # YOLO training configuration
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )
    
    # Learning rate scheduling
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 40], gamma=0.1
    )
    
    # YOLO loss function
    criterion = YOLOLoss(num_classes=model.num_classes)
    
    # Training tracking
    train_losses = []
    loss_components = []
    
    model_name = model.__class__.__name__
    print(f"Training {model_name} on device: {device}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_loss_components = defaultdict(float)
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            
            # Generate pseudo detection targets
            targets = generate_pseudo_boxes(
                data.size(0), model.num_classes, model.grid_size
            ).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(data)
            
            # Calculate loss
            total_loss, loss_dict = criterion(predictions, targets)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Track statistics
            running_loss += total_loss.item()
            for key, value in loss_dict.items():
                epoch_loss_components[key] += value.item() if hasattr(value, 'item') else value
            
            if batch_idx % 200 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {total_loss.item():.4f}, LR: {current_lr:.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Average loss components
        avg_components = {k: v/len(train_loader) for k, v in epoch_loss_components.items()}
        loss_components.append(avg_components)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/yolo_detection_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}')
        print(f'  Coord: {avg_components["coordinate_loss"]:.4f}, '
              f'Conf: {avg_components["confidence_loss"]:.4f}, '
              f'Class: {avg_components["classification_loss"]:.4f}')
        
        # Early stopping for demonstration
        if epoch_loss < 0.5:
            print(f"Convergence reached at epoch {epoch+1}")
            break
    
    print(f"Best training loss: {best_loss:.4f}")
    return train_losses, loss_components

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_yolo_innovations():
    """Visualize YOLO's object detection innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Single-shot vs Two-stage comparison
    ax = axes[0, 0]
    ax.set_title('Detection Pipeline Comparison', fontsize=14)
    
    # Two-stage (R-CNN family)
    ax.text(0.5, 0.8, 'Two-Stage Detection (R-CNN)', ha='center', va='center', 
           fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    stages_2 = ['1. Region\nProposal', '2. Feature\nExtraction', '3. Classification\n& Regression']
    for i, stage in enumerate(stages_2):
        x_pos = 0.15 + i * 0.25
        ax.text(x_pos, 0.6, stage, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
        if i < len(stages_2) - 1:
            ax.annotate('', xy=(x_pos + 0.12, 0.6), xytext=(x_pos + 0.08, 0.6),
                       arrowprops=dict(arrowstyle='->', lw=2))
    
    # Single-shot (YOLO)
    ax.text(0.5, 0.4, 'Single-Shot Detection (YOLO)', ha='center', va='center', 
           fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.text(0.5, 0.2, 'Direct Detection\n(One Forward Pass)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgoldenrodyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Grid-based detection
    ax = axes[0, 1]
    ax.set_title('YOLO Grid-Based Detection', fontsize=14)
    
    # Draw 7x7 grid
    grid_size = 7
    for i in range(grid_size + 1):
        ax.axhline(y=i, color='black', linewidth=1)
        ax.axvline(x=i, color='black', linewidth=1)
    
    # Simulate object centers and bounding boxes
    objects = [(2.5, 3.5), (5.2, 1.8), (4.3, 5.1)]
    colors = ['red', 'blue', 'green']
    
    for (x, y), color in zip(objects, colors):
        # Object center
        ax.plot(x, y, 'o', color=color, markersize=8)
        
        # Responsible grid cell
        grid_x, grid_y = int(x), int(y)
        rect = plt.Rectangle((grid_x, grid_y), 1, 1, 
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Bounding box
        box_rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               facecolor='none', edgecolor=color, linewidth=2, linestyle='--')
        ax.add_patch(box_rect)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xlabel('Grid cells predict objects whose centers fall within them')
    ax.invert_yaxis()
    
    # Speed comparison
    ax = axes[1, 0]
    methods = ['R-CNN', 'Fast R-CNN', 'Faster R-CNN', 'YOLO', 'YOLOv3', 'YOLOv5']
    fps = [0.02, 0.5, 7, 45, 65, 140]  # Frames per second
    colors_speed = ['#E74C3C', '#E67E22', '#F39C12', '#27AE60', '#2ECC71', '#00E676']
    
    bars = ax.bar(methods, fps, color=colors_speed)
    ax.set_title('Real-Time Performance Comparison', fontsize=14)
    ax.set_ylabel('FPS (Frames Per Second)')
    ax.set_yscale('log')
    
    # Add FPS labels
    for bar, speed in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2, 
                f'{speed}', ha='center', va='bottom')
    
    # Add real-time threshold
    ax.axhline(y=30, color='red', linestyle='--', alpha=0.7)
    ax.text(2, 35, 'Real-time threshold (30 FPS)', color='red', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # YOLO evolution
    ax = axes[1, 1]
    yolo_versions = ['YOLOv1', 'YOLOv2', 'YOLOv3', 'YOLOv4', 'YOLOv5', 'YOLOv8']
    accuracies = [63.4, 76.8, 80.2, 85.1, 88.3, 91.7]  # Example mAP scores
    
    ax.plot(yolo_versions, accuracies, 'o-', linewidth=3, markersize=8, color='#27AE60')
    ax.set_title('YOLO Evolution: Accuracy Improvement', fontsize=14)
    ax.set_ylabel('mAP (Mean Average Precision)')
    ax.grid(True, alpha=0.3)
    
    # Annotate key improvements
    improvements = ['Original', 'Batch Norm\nAnchor Boxes', 'Multi-Scale\nFPN', 'CSP\nMosaic', 'AutoML\nOptimizations', 'Latest\nArchitecture']
    for i, (version, acc, improvement) in enumerate(zip(yolo_versions, accuracies, improvements)):
        ax.annotate(improvement, (i, acc), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/014_yolo_innovations.png', dpi=300, bbox_inches='tight')
    print("YOLO innovations visualization saved: 014_yolo_innovations.png")

def visualize_detection_grid():
    """Visualize YOLO's grid-based detection mechanism"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grid prediction visualization
    ax = axes[0]
    ax.set_title('YOLO Grid Predictions', fontsize=14, fontweight='bold')
    
    # Create 13x13 grid
    grid_size = 13
    
    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.7)
        ax.axvline(x=i, color='gray', linewidth=0.5, alpha=0.7)
    
    # Simulate detection confidence heatmap
    np.random.seed(42)
    confidence_map = np.random.random((grid_size, grid_size)) * 0.3
    
    # Add some high-confidence detections
    high_conf_cells = [(3, 4), (8, 2), (10, 9), (5, 11)]
    for x, y in high_conf_cells:
        confidence_map[y, x] = 0.8 + np.random.random() * 0.2
    
    # Draw confidence heatmap
    im = ax.imshow(confidence_map, cmap='Reds', alpha=0.7, extent=[0, grid_size, 0, grid_size])
    
    # Draw predicted bounding boxes for high-confidence cells
    for x, y in high_conf_cells:
        # Cell center
        center_x, center_y = x + 0.5, y + 0.5
        
        # Random box dimensions
        box_w = 2 + np.random.random() * 2
        box_h = 1.5 + np.random.random() * 2
        
        # Bounding box
        box_rect = plt.Rectangle(
            (center_x - box_w/2, center_y - box_h/2), box_w, box_h,
            facecolor='none', edgecolor='blue', linewidth=2
        )
        ax.add_patch(box_rect)
        
        # Confidence score
        ax.text(center_x, center_y, f'{confidence_map[y, x]:.2f}', 
               ha='center', va='center', fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.8))
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xlabel('Each cell predicts bounding boxes and confidence')
    ax.invert_yaxis()
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Detection Confidence')
    
    # Multi-scale detection (YOLOv3+ feature)
    ax = axes[1]
    ax.set_title('Multi-Scale Detection (YOLOv3+)', fontsize=14, fontweight='bold')
    
    # Draw feature pyramid
    scales = [(13, 13), (26, 26), (52, 52)]
    scale_names = ['Large Objects\n(13×13)', 'Medium Objects\n(26×26)', 'Small Objects\n(52×52)']
    colors_pyramid = ['#E74C3C', '#F39C12', '#27AE60']
    
    for i, ((h, w), name, color) in enumerate(zip(scales, scale_names, colors_pyramid)):
        # Draw grid representation
        y_offset = i * 0.3
        grid_size_viz = 0.2
        
        # Grid
        for row in range(min(h, 10)):  # Show max 10x10 for visualization
            for col in range(min(w, 10)):
                x_pos = 0.1 + col * grid_size_viz / 10
                y_pos = y_offset + row * grid_size_viz / 10
                
                rect = plt.Rectangle((x_pos, y_pos), grid_size_viz/10, grid_size_viz/10,
                                   facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.1)
                ax.add_patch(rect)
        
        # Label
        ax.text(0.35, y_offset + grid_size_viz/2, name, ha='left', va='center', 
               fontsize=11, fontweight='bold')
        
        # Anchor boxes representation
        anchor_sizes = [['Large', 'XL'], ['Medium', 'Large'], ['Small', 'Medium']]
        ax.text(0.6, y_offset + grid_size_viz/2, f'Anchors: {", ".join(anchor_sizes[i])}', 
               ha='left', va='center', fontsize=10, style='italic')
    
    # Draw connections between scales
    ax.text(0.1, 1.0, 'Feature Pyramid Network (FPN)', ha='left', va='bottom', 
           fontsize=12, fontweight='bold', color='purple')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/014_detection_grid.png', dpi=300, bbox_inches='tight')
    print("Detection grid mechanism saved: 014_detection_grid.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== YOLO Object Detection Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Create detection dataset
    train_loader, test_loader, classes = create_detection_dataset()
    
    # Initialize YOLO models
    yolo_darknet = YOLO_ObjectDetection(
        num_classes=len(classes), backbone='darknet19', grid_size=13
    )
    
    yolo_csp = YOLO_ObjectDetection(
        num_classes=len(classes), backbone='csp_darknet', grid_size=13
    )
    
    # Compare model complexities
    yolo_darknet_params = sum(p.numel() for p in yolo_darknet.parameters())
    yolo_csp_params = sum(p.numel() for p in yolo_csp.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  YOLO DarkNet-19: {yolo_darknet_params:,} parameters")
    print(f"  YOLO CSP-DarkNet: {yolo_csp_params:,} parameters")
    print(f"  Parameter ratio: {yolo_csp_params/yolo_darknet_params:.2f}x")
    
    # Analyze detection capabilities
    detection_analysis = yolo_darknet.get_detection_analysis()
    
    print(f"\nYOLO Detection Analysis:")
    for key, value in detection_analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating YOLO analysis...")
    visualize_yolo_innovations()
    visualize_detection_grid()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("YOLO OBJECT DETECTION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nYOLO REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. SINGLE-SHOT DETECTION:")
    print("   • Direct bounding box prediction (no region proposals)")
    print("   • Unified network for detection and classification")
    print("   • End-to-end differentiable training")
    print("   • One forward pass for complete detection")
    
    print("\n2. GRID-BASED APPROACH:")
    print("   • Divide image into SxS grid (typically 13x13)")
    print("   • Each cell predicts bounding boxes and confidence")
    print("   • Responsible cell concept for object assignment")
    print("   • Direct coordinate regression")
    
    print("\n3. REAL-TIME PERFORMANCE:")
    print("   • 45+ FPS on modern hardware")
    print("   • Efficient backbone architectures (DarkNet)")
    print("   • Optimized for speed-accuracy tradeoff")
    print("   • Practical deployment in real applications")
    
    print("\n4. UNIFIED MULTI-TASK LOSS:")
    print("   • Coordinate regression loss")
    print("   • Objectness confidence loss")
    print("   • Classification loss")
    print("   • Balanced training across all tasks")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First practical real-time object detector")
    print("• Simplified detection pipeline")
    print("• End-to-end optimization")
    print("• Strong performance across object scales")
    print("• Enabled real-world deployment")
    
    print(f"\nYOLO EVOLUTION MILESTONES:")
    print("="*40)
    print("• YOLOv1 (2015): Original single-shot concept")
    print("• YOLOv2 (2016): Batch normalization, anchor boxes")
    print("• YOLOv3 (2018): Multi-scale detection, FPN")
    print("• YOLOv4 (2020): CSP, Mosaic augmentation")
    print("• YOLOv5 (2020): AutoML optimizations")
    print("• YOLOv8 (2023): Latest architecture improvements")
    
    print(f"\nDETECTION CAPABILITIES:")
    for key, value in detection_analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nARCHITECTURAL COMPONENTS:")
    print("="*40)
    print("• Backbone: Feature extraction (DarkNet, CSP-DarkNet)")
    print("• Neck: Feature aggregation (FPN in later versions)")
    print("• Head: Detection prediction (bounding boxes + classes)")
    print("• Post-processing: NMS for final detections")
    
    print(f"\nREAL-TIME APPLICATIONS:")
    print("="*40)
    print("• Autonomous driving (vehicle/pedestrian detection)")
    print("• Security surveillance (person/object tracking)")
    print("• Robotics (object manipulation, navigation)")
    print("• Augmented reality (real-time object recognition)")
    print("• Industrial automation (quality control)")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Revolutionized object detection paradigm")
    print("• Made real-time detection practical")
    print("• Simplified complex detection pipelines")
    print("• Enabled widespread deployment of computer vision")
    print("• Inspired numerous single-shot detector variants")
    print("• Established speed-accuracy tradeoff benchmarks")
    
    print(f"\nYOLO VS TRADITIONAL DETECTION:")
    print("="*40)
    print("• Traditional: Multi-stage (proposals → classification)")
    print("• YOLO: Single-stage (direct detection)")
    print("• Traditional: Slow (< 1 FPS)")
    print("• YOLO: Fast (45+ FPS)")
    print("• Traditional: Complex pipeline")
    print("• YOLO: Unified architecture")
    
    return {
        'model': 'YOLO Object Detection',
        'year': YEAR,
        'innovation': INNOVATION,
        'detection_analysis': detection_analysis,
        'parameter_comparison': {
            'yolo_darknet': yolo_darknet_params,
            'yolo_csp': yolo_csp_params
        }
    }

if __name__ == "__main__":
    results = main()