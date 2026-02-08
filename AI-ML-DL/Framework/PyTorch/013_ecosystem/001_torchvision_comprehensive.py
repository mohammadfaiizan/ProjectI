import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as vutils
from torchvision.ops import nms, roi_align, roi_pool
from torchvision.io import read_image, write_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional, Any

# Comprehensive Transforms Pipeline
class AdvancedTransforms:
    """Advanced torchvision transforms for different scenarios"""
    
    @staticmethod
    def get_training_transforms(image_size: Tuple[int, int] = (224, 224),
                               augmentation_level: str = "medium") -> transforms.Compose:
        """Get training transforms with different augmentation levels"""
        
        base_transforms = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
        
        if augmentation_level == "light":
            augment_transforms = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ]
        elif augmentation_level == "medium":
            augment_transforms = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3)
            ]
        elif augmentation_level == "heavy":
            augment_transforms = [
                transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
                transforms.RandomRotation(degrees=30),
                transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ]
        else:
            augment_transforms = []
        
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(base_transforms + augment_transforms + final_transforms)
    
    @staticmethod
    def get_inference_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
        """Get inference transforms (no augmentation)"""
        
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_time_augmentation() -> List[transforms.Compose]:
        """Get multiple transforms for test-time augmentation"""
        
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tta_transforms = [
            # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                base_transform
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                base_transform
            ]),
            # Different crops
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                base_transform
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                base_transform
            ])
        ]
        
        return tta_transforms

# Custom Dataset Examples
class CustomImageDataset(torch.utils.data.Dataset):
    """Custom dataset using torchvision utilities"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Use torchvision.io for efficient image loading
        image = read_image(self.image_paths[idx])
        
        # Convert to PIL for transforms compatibility
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

# Model Zoo and Fine-tuning
class ModelZooManager:
    """Manage torchvision model zoo for various tasks"""
    
    def __init__(self):
        self.available_models = {
            'classification': [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'vgg11', 'vgg13', 'vgg16', 'vgg19',
                'densenet121', 'densenet169', 'densenet201',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                'squeezenet1_0', 'squeezenet1_1',
                'inception_v3', 'googlenet', 'shufflenet_v2_x1_0',
                'resnext50_32x4d', 'wide_resnet50_2',
                'regnet_y_400mf', 'regnet_x_400mf'
            ],
            'detection': [
                'fasterrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn',
                'retinanet_resnet50_fpn', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large'
            ],
            'segmentation': [
                'fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
                'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large'
            ]
        }
    
    def load_pretrained_model(self, model_name: str, 
                             num_classes: Optional[int] = None,
                             task_type: str = 'classification') -> nn.Module:
        """Load pretrained model from torchvision"""
        
        if task_type == 'classification':
            model = self._load_classification_model(model_name, num_classes)
        elif task_type == 'detection':
            model = self._load_detection_model(model_name, num_classes)
        elif task_type == 'segmentation':
            model = self._load_segmentation_model(model_name, num_classes)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        print(f"✓ Loaded {model_name} for {task_type}")
        return model
    
    def _load_classification_model(self, model_name: str, num_classes: Optional[int]) -> nn.Module:
        """Load classification model"""
        
        # Get the model function
        model_fn = getattr(models, model_name)
        
        # Load with pretrained weights
        model = model_fn(pretrained=True)
        
        # Modify for custom number of classes
        if num_classes is not None:
            if hasattr(model, 'classifier'):
                # Models like VGG, AlexNet
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'fc'):
                # Models like ResNet
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'heads'):
                # Models like EfficientNet
                in_features = model.heads.head.in_features
                model.heads.head = nn.Linear(in_features, num_classes)
        
        return model
    
    def _load_detection_model(self, model_name: str, num_classes: Optional[int]) -> nn.Module:
        """Load detection model"""
        
        model_fn = getattr(models.detection, model_name)
        model = model_fn(pretrained=True)
        
        if num_classes is not None and hasattr(model, 'roi_heads'):
            # Modify classifier head
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
            model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)
        
        return model
    
    def _load_segmentation_model(self, model_name: str, num_classes: Optional[int]) -> nn.Module:
        """Load segmentation model"""
        
        model_fn = getattr(models.segmentation, model_name)
        model = model_fn(pretrained=True)
        
        if num_classes is not None and hasattr(model, 'classifier'):
            # Modify classifier
            model.classifier[-1] = nn.Conv2d(
                model.classifier[-1].in_channels,
                num_classes,
                kernel_size=1
            )
        
        return model
    
    def setup_feature_extraction(self, model: nn.Module, 
                                freeze_backbone: bool = True) -> nn.Module:
        """Setup model for feature extraction"""
        
        if freeze_backbone:
            # Freeze all parameters except the final layer
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'fc' not in name and 'heads' not in name:
                    param.requires_grad = False
            
            print("✓ Backbone frozen for feature extraction")
        
        return model

# Advanced Visualization
class TorchvisionVisualizer:
    """Advanced visualization utilities using torchvision"""
    
    @staticmethod
    def visualize_augmentations(image: Image.Image, 
                               transform: transforms.Compose,
                               num_samples: int = 8) -> None:
        """Visualize different augmentations of the same image"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(num_samples):
            augmented = transform(image)
            
            # Convert tensor to PIL for display
            if isinstance(augmented, torch.Tensor):
                # Denormalize if normalized
                if augmented.min() < 0:
                    augmented = augmented * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                               torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                augmented = torch.clamp(augmented, 0, 1)
                augmented = transforms.ToPILImage()(augmented)
            
            axes[i].imshow(augmented)
            axes[i].axis('off')
            axes[i].set_title(f'Augmentation {i+1}')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_feature_maps(model: nn.Module, 
                              input_tensor: torch.Tensor,
                              layer_name: str) -> None:
        """Visualize feature maps from a specific layer"""
        
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Register hook
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(get_activation(name))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Visualize feature maps
        if layer_name in activation:
            feature_maps = activation[layer_name][0]  # First batch
            num_maps = min(16, feature_maps.size(0))  # Show up to 16 maps
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for i in range(num_maps):
                fmap = feature_maps[i].cpu().numpy()
                axes[i].imshow(fmap, cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f'Feature Map {i+1}')
            
            plt.suptitle(f'Feature Maps from {layer_name}')
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def create_image_grid(images: List[torch.Tensor], 
                         nrow: int = 8,
                         normalize: bool = True) -> torch.Tensor:
        """Create image grid using torchvision utilities"""
        
        # Stack images
        image_tensor = torch.stack(images)
        
        # Create grid
        grid = vutils.make_grid(image_tensor, nrow=nrow, normalize=normalize, padding=2)
        
        return grid
    
    @staticmethod
    def save_image_grid(images: List[torch.Tensor], 
                       filename: str,
                       nrow: int = 8) -> None:
        """Save image grid to file"""
        
        grid = TorchvisionVisualizer.create_image_grid(images, nrow=nrow)
        vutils.save_image(grid, filename)
        print(f"✓ Image grid saved to {filename}")

# Object Detection Utilities
class DetectionUtils:
    """Utilities for object detection with torchvision"""
    
    @staticmethod
    def apply_nms(boxes: torch.Tensor, 
                  scores: torch.Tensor,
                  iou_threshold: float = 0.5) -> torch.Tensor:
        """Apply Non-Maximum Suppression"""
        
        keep_indices = nms(boxes, scores, iou_threshold)
        return keep_indices
    
    @staticmethod
    def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes"""
        
        from torchvision.ops import box_iou
        return box_iou(boxes1, boxes2)
    
    @staticmethod
    def roi_align_features(features: torch.Tensor,
                          boxes: torch.Tensor,
                          output_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
        """Apply ROI Align to extract features"""
        
        # Prepare boxes for ROI align (add batch index)
        batch_indices = torch.zeros(boxes.size(0), 1)
        roi_boxes = torch.cat([batch_indices, boxes], dim=1)
        
        # Apply ROI align
        roi_features = roi_align(features, roi_boxes, output_size)
        
        return roi_features
    
    @staticmethod
    def visualize_detections(image: torch.Tensor,
                           boxes: torch.Tensor,
                           labels: torch.Tensor,
                           scores: torch.Tensor,
                           class_names: List[str] = None,
                           threshold: float = 0.5) -> None:
        """Visualize detection results"""
        
        from torchvision.utils import draw_bounding_boxes
        
        # Filter by score threshold
        keep = scores > threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # Create labels with scores
        if class_names:
            label_texts = [f"{class_names[label]}: {score:.2f}" 
                          for label, score in zip(labels, scores)]
        else:
            label_texts = [f"Class {label}: {score:.2f}" 
                          for label, score in zip(labels, scores)]
        
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(
            (image * 255).byte(),
            boxes,
            labels=label_texts,
            colors="red",
            width=2
        )
        
        # Convert to PIL and display
        image_pil = transforms.ToPILImage()(image_with_boxes)
        plt.figure(figsize=(12, 8))
        plt.imshow(image_pil)
        plt.axis('off')
        plt.title('Detection Results')
        plt.show()

# Advanced Data Loading
class AdvancedDataLoader:
    """Advanced data loading utilities with torchvision"""
    
    @staticmethod
    def create_stratified_sampler(dataset: torch.utils.data.Dataset,
                                 labels: List[int]) -> torch.utils.data.Sampler:
        """Create stratified sampler for balanced training"""
        
        from collections import Counter
        
        # Count samples per class
        class_counts = Counter(labels)
        
        # Calculate weights for each sample
        weights = [1.0 / class_counts[label] for label in labels]
        
        # Create weighted sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
        return sampler
    
    @staticmethod
    def create_multi_scale_dataloader(dataset: torch.utils.data.Dataset,
                                     scales: List[int] = [224, 256, 288, 320],
                                     batch_size: int = 32) -> torch.utils.data.DataLoader:
        """Create dataloader with multi-scale training"""
        
        def collate_fn(batch):
            # Randomly choose scale
            scale = np.random.choice(scales)
            
            # Resize all images in batch to chosen scale
            resize_transform = transforms.Resize((scale, scale))
            
            images, labels = zip(*batch)
            resized_images = [resize_transform(img) for img in images]
            
            return torch.stack(resized_images), torch.tensor(labels)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )

# Performance Optimization
class TorchvisionOptimizer:
    """Optimization utilities for torchvision workflows"""
    
    @staticmethod
    def optimize_data_loading(dataset: torch.utils.data.Dataset,
                             batch_size: int = 32,
                             num_workers: int = 4) -> torch.utils.data.DataLoader:
        """Create optimized dataloader"""
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False
        )
    
    @staticmethod
    def create_cached_dataset(dataset: torch.utils.data.Dataset,
                             cache_size: int = 1000) -> torch.utils.data.Dataset:
        """Create dataset with caching for frequently accessed items"""
        
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, cache_size):
                self.base_dataset = base_dataset
                self.cache = {}
                self.cache_size = cache_size
                self.access_count = {}
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                if idx in self.cache:
                    self.access_count[idx] = self.access_count.get(idx, 0) + 1
                    return self.cache[idx]
                
                # Load from base dataset
                item = self.base_dataset[idx]
                
                # Add to cache if there's space
                if len(self.cache) < self.cache_size:
                    self.cache[idx] = item
                    self.access_count[idx] = 1
                
                return item
        
        return CachedDataset(dataset, cache_size)

if __name__ == "__main__":
    print("Comprehensive Torchvision Usage")
    print("=" * 35)
    
    print("\n1. Advanced Transforms")
    print("-" * 24)
    
    # Create sample image
    sample_image = Image.new('RGB', (256, 256), color='red')
    
    # Get different transform pipelines
    train_transforms = AdvancedTransforms.get_training_transforms(
        image_size=(224, 224), 
        augmentation_level="medium"
    )
    
    inference_transforms = AdvancedTransforms.get_inference_transforms()
    tta_transforms = AdvancedTransforms.get_test_time_augmentation()
    
    print(f"✓ Training transforms created with medium augmentation")
    print(f"✓ Inference transforms created")
    print(f"✓ {len(tta_transforms)} TTA transforms created")
    
    # Test transforms
    train_tensor = train_transforms(sample_image)
    inference_tensor = inference_transforms(sample_image)
    
    print(f"Training output shape: {train_tensor.shape}")
    print(f"Inference output shape: {inference_tensor.shape}")
    
    print("\n2. Model Zoo Management")
    print("-" * 26)
    
    model_manager = ModelZooManager()
    
    print("Available models:")
    for task, models_list in model_manager.available_models.items():
        print(f"  {task.title()}: {len(models_list)} models")
    
    # Load different types of models
    try:
        # Classification model
        resnet50 = model_manager.load_pretrained_model(
            'resnet50', 
            num_classes=10, 
            task_type='classification'
        )
        
        # Setup for feature extraction
        resnet50_frozen = model_manager.setup_feature_extraction(resnet50, freeze_backbone=True)
        
        print("✓ ResNet50 loaded and configured for feature extraction")
        
        # Test forward pass
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = resnet50(test_input)
            print(f"Model output shape: {output.shape}")
    
    except Exception as e:
        print(f"Model loading demo: {e}")
    
    print("\n3. Detection Utilities")
    print("-" * 24)
    
    detection_utils = DetectionUtils()
    
    # Sample detection data
    sample_boxes = torch.tensor([
        [10, 10, 50, 50],
        [15, 15, 55, 55],
        [100, 100, 150, 150]
    ], dtype=torch.float32)
    
    sample_scores = torch.tensor([0.9, 0.8, 0.95])
    
    # Apply NMS
    keep_indices = detection_utils.apply_nms(sample_boxes, sample_scores, iou_threshold=0.5)
    print(f"NMS keep indices: {keep_indices}")
    
    # Compute IoU
    iou_matrix = detection_utils.compute_iou(sample_boxes, sample_boxes)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    print(f"IoU between box 0 and 1: {iou_matrix[0, 1]:.3f}")
    
    print("\n4. Advanced Visualization")
    print("-" * 29)
    
    visualizer = TorchvisionVisualizer()
    
    # Create sample images for grid
    sample_images = [torch.randn(3, 64, 64) for _ in range(16)]
    
    # Create image grid
    grid = visualizer.create_image_grid(sample_images, nrow=4)
    print(f"✓ Image grid created with shape: {grid.shape}")
    
    # Save grid (commented out to avoid file operations)
    # visualizer.save_image_grid(sample_images, "sample_grid.png")
    
    print("\n5. Advanced Data Loading")
    print("-" * 27)
    
    data_loader = AdvancedDataLoader()
    
    # Create sample dataset for stratified sampling
    sample_labels = [0] * 100 + [1] * 200 + [2] * 50  # Imbalanced classes
    
    # Note: Would need actual dataset for real implementation
    print(f"Sample labels distribution: Class 0: 100, Class 1: 200, Class 2: 50")
    print("✓ Stratified sampler concept demonstrated")
    
    print("\n6. Performance Optimization")
    print("-" * 30)
    
    optimizer = TorchvisionOptimizer()
    
    print("Optimization strategies:")
    print("  - Optimized data loading with pin_memory and persistent_workers")
    print("  - Cached dataset for frequently accessed items")
    print("  - Multi-scale training support")
    print("  - Efficient transform pipelines")
    
    print("\n7. Torchvision Best Practices")
    print("-" * 34)
    
    best_practices = [
        "Use appropriate data augmentation based on dataset size",
        "Leverage pretrained models for transfer learning",
        "Apply proper normalization (ImageNet stats for pretrained models)",
        "Use efficient data loading with multiple workers",
        "Implement test-time augmentation for better accuracy",
        "Cache frequently accessed dataset items",
        "Use mixed precision training for faster training",
        "Apply NMS for object detection post-processing",
        "Visualize augmentations to ensure reasonable transforms",
        "Use stratified sampling for imbalanced datasets",
        "Implement proper validation data loading",
        "Monitor GPU memory usage with large datasets"
    ]
    
    print("Torchvision Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Common Workflows")
    print("-" * 21)
    
    workflows = {
        "Image Classification": [
            "Load pretrained model",
            "Modify final layer for custom classes",
            "Apply data augmentation",
            "Use stratified sampling if imbalanced",
            "Train with frozen backbone initially",
            "Fine-tune end-to-end with lower learning rate"
        ],
        "Object Detection": [
            "Prepare COCO-format annotations",
            "Load pretrained detection model",
            "Modify number of classes",
            "Apply detection-specific transforms",
            "Train with appropriate loss functions",
            "Apply NMS during inference"
        ],
        "Transfer Learning": [
            "Load pretrained weights",
            "Freeze backbone layers",
            "Train classifier head",
            "Gradually unfreeze layers",
            "Use different learning rates for different parts",
            "Apply appropriate regularization"
        ]
    }
    
    for workflow, steps in workflows.items():
        print(f"\n{workflow}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    
    print("\n9. Integration with Other Libraries")
    print("-" * 38)
    
    integrations = [
        "Albumentation: Advanced augmentation library",
        "OpenCV: Computer vision preprocessing",
        "Pillow-SIMD: Faster image operations",
        "ONNX: Model export for deployment",
        "TensorBoard: Visualization and logging",
        "Weights & Biases: Experiment tracking",
        "PyTorch Lightning: Training framework",
        "Hydra: Configuration management"
    ]
    
    print("Common Integrations:")
    for integration in integrations:
        print(f"  - {integration}")
    
    print("\nTorchvision comprehensive usage demonstration completed!")
    print("Key components covered:")
    print("  - Advanced transforms and augmentation")
    print("  - Model zoo management and fine-tuning")
    print("  - Object detection utilities")
    print("  - Visualization tools")
    print("  - Performance optimization")
    print("  - Best practices and workflows")
    
    print("\nTorchvision is essential for:")
    print("  - Computer vision tasks")
    print("  - Pretrained model usage")
    print("  - Data preprocessing and augmentation")
    print("  - Transfer learning workflows")
    print("  - Object detection and segmentation")