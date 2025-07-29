import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# Basic Transformations
def basic_transforms():
    """Standard basic image transformations"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def advanced_transforms():
    """Advanced transformations with perspective and affine changes"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

# Custom Augmentation Classes
class GaussianNoise(nn.Module):
    """Add Gaussian noise to images"""
    def __init__(self, mean=0.0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, tensor):
        if isinstance(tensor, torch.Tensor):
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

class RandomMixup(nn.Module):
    """Mixup augmentation for training"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, batch_data):
        if len(batch_data) != 2:
            return batch_data
        
        x, y = batch_data
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

class CutMix(nn.Module):
    """CutMix augmentation"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, batch_data):
        x, y = batch_data
        batch_size = x.size(0)
        
        if self.beta > 0:
            lam = np.random.beta(self.beta, self.beta)
        else:
            lam = 1
        
        rand_index = torch.randperm(batch_size)
        y_a = y
        y_b = y[rand_index]
        
        # Generate random bounding box
        W, H = x.size(-1), x.size(-2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to actual area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return x, y_a, y_b, lam

class MosaicAugmentation(nn.Module):
    """Mosaic augmentation combining 4 images"""
    def __init__(self, size=224):
        super().__init__()
        self.size = size
    
    def forward(self, images):
        """images: list of 4 PIL Images"""
        if len(images) != 4:
            return images[0]  # Return first image if not exactly 4
        
        # Create mosaic image
        mosaic = Image.new('RGB', (self.size, self.size))
        
        # Resize images to half size
        half_size = self.size // 2
        resized_images = [img.resize((half_size, half_size)) for img in images]
        
        # Place images in quadrants
        mosaic.paste(resized_images[0], (0, 0))
        mosaic.paste(resized_images[1], (half_size, 0))
        mosaic.paste(resized_images[2], (0, half_size))
        mosaic.paste(resized_images[3], (half_size, half_size))
        
        return mosaic

class AutoAugment(nn.Module):
    """AutoAugment policy implementation"""
    def __init__(self):
        super().__init__()
        self.policies = [
            [('Posterize', 0.8, 6), ('Rotate', 0.9, 3)],
            [('Solarize', 0.5, 5), ('AutoContrast', 0.9, None)],
            [('Equalize', 0.2, None), ('Equalize', 0.6, None)],
            [('Posterize', 0.7, 7), ('Posterize', 0.6, 6)],
            [('Equalize', 0.1, None), ('Solarize', 0.6, 1)],
            [('Equalize', 0.8, None), ('Rotate', 0.2, 8)],
            [('Solarize', 0.9, 3), ('Equalize', 0.7, None)],
            [('Posterize', 0.5, 8), ('Equalize', 1.0, None)],
            [('Rotate', 0.3, 3), ('Posterize', 0.8, 8)],
            [('Rotate', 0.7, 2), ('Solarize', 0.8, 6)],
        ]
    
    def forward(self, img):
        policy = random.choice(self.policies)
        for operation, prob, magnitude in policy:
            if random.random() < prob:
                img = self._apply_operation(img, operation, magnitude)
        return img
    
    def _apply_operation(self, img, operation, magnitude):
        if operation == 'Rotate':
            return TF.rotate(img, magnitude * 30)
        elif operation == 'Posterize':
            return TF.posterize(img, magnitude)
        elif operation == 'Solarize':
            return TF.solarize(img, 256 - magnitude * 32)
        elif operation == 'AutoContrast':
            return TF.autocontrast(img)
        elif operation == 'Equalize':
            return TF.equalize(img)
        return img

class GridMask(nn.Module):
    """GridMask augmentation"""
    def __init__(self, d_range=(96, 224), r=0.6, rotate=1, ratio=0.5):
        super().__init__()
        self.d_range = d_range
        self.r = r
        self.rotate = rotate
        self.ratio = ratio
    
    def forward(self, img):
        if random.random() > self.ratio:
            return img
        
        h, w = img.shape[-2:]
        d = random.randint(self.d_range[0], self.d_range[1])
        
        # Create grid mask
        mask = torch.ones((h, w), dtype=torch.float32)
        
        # Grid spacing
        l = int(d * self.r)
        
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:i+l, j:j+l] = 0
        
        # Apply rotation if specified
        if self.rotate == 1:
            angle = random.randint(0, 360)
            mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)
        
        # Apply mask
        if len(img.shape) == 3:  # CHW format
            mask = mask.unsqueeze(0).expand_as(img)
        
        return img * mask

# Augmentation Pipeline Classes
class TrainingAugmentation:
    """Training augmentation pipeline"""
    def __init__(self, size=224, severity='medium'):
        if severity == 'light':
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif severity == 'medium':
            self.transform = transforms.Compose([
                transforms.Resize((int(size * 1.1), int(size * 1.1))),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                GaussianNoise(std=0.01),
                transforms.RandomErasing(p=0.3),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # heavy
            self.transform = transforms.Compose([
                transforms.Resize((int(size * 1.2), int(size * 1.2))),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                AutoAugment(),
                transforms.ToTensor(),
                GaussianNoise(std=0.02),
                GridMask(ratio=0.7),
                transforms.RandomErasing(p=0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, x):
        return self.transform(x)

class TestAugmentation:
    """Test time augmentation (TTA)"""
    def __init__(self, size=224, num_augmentations=5):
        self.size = size
        self.num_augmentations = num_augmentations
        self.base_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.augmentation_transforms = [
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((int(size * 1.1), int(size * 1.1))),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
    
    def __call__(self, x):
        augmented_images = []
        for i in range(min(self.num_augmentations, len(self.augmentation_transforms))):
            augmented = self.augmentation_transforms[i](x)
            augmented_images.append(augmented)
        return torch.stack(augmented_images)

# Augmentation Strategies
def get_augmentation_strategy(strategy_name, **kwargs):
    """Factory function for augmentation strategies"""
    strategies = {
        'basic': lambda: basic_transforms(),
        'advanced': lambda: advanced_transforms(),
        'autoaugment': lambda: transforms.Compose([
            transforms.Resize((224, 224)),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'training_light': lambda: TrainingAugmentation(severity='light'),
        'training_medium': lambda: TrainingAugmentation(severity='medium'),
        'training_heavy': lambda: TrainingAugmentation(severity='heavy'),
        'test': lambda: TestAugmentation(**kwargs)
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

# Training with augmentation
def train_with_augmentation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10)
    ).to(device)
    
    # Data with augmentation
    train_transform = get_augmentation_strategy('training_medium')
    train_dataset = datasets.FakeData(size=1000, image_size=(3, 224, 224), 
                                    num_classes=10, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(2):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # Test individual augmentations
    print("Testing individual augmentations...")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Test Gaussian Noise
    tensor_img = transforms.ToTensor()(dummy_image)
    noise_aug = GaussianNoise(std=0.1)
    noisy_img = noise_aug(tensor_img)
    print(f"Gaussian noise output shape: {noisy_img.shape}")
    
    # Test AutoAugment
    auto_aug = AutoAugment()
    auto_img = auto_aug(dummy_image)
    print(f"AutoAugment applied successfully")
    
    # Test GridMask
    grid_mask = GridMask()
    masked_img = grid_mask(tensor_img)
    print(f"GridMask output shape: {masked_img.shape}")
    
    # Test training augmentation
    train_aug = TrainingAugmentation(severity='medium')
    train_img = train_aug(dummy_image)
    print(f"Training augmentation shape: {train_img.shape}")
    
    # Test TTA
    tta = TestAugmentation(num_augmentations=3)
    tta_imgs = tta(dummy_image)
    print(f"TTA output shape: {tta_imgs.shape}")
    
    # Test augmentation strategies
    for strategy in ['basic', 'advanced', 'autoaugment', 'training_light']:
        try:
            aug = get_augmentation_strategy(strategy)
            if strategy == 'test':
                result = aug(dummy_image)
                print(f"{strategy} strategy shape: {result.shape}")
            else:
                result = aug(dummy_image)
                print(f"{strategy} strategy shape: {result.shape}")
        except Exception as e:
            print(f"Error with {strategy}: {e}")
    
    print("\nStarting training with augmentation...")
    train_with_augmentation()