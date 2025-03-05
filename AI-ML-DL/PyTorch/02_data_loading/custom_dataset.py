"""
custom_dataset.py

Learn to create custom datasets and use DataLoaders:
1. Create Dataset class
2. Implement transforms/preprocessing
3. Use DataLoader for batching
4. Handle different data types (images, CSVs, etc.)
"""

import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ================================================================== #
#                     Custom Dataset Class                          #
# ================================================================== #

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to CSV with image labels
            img_dir (string): Directory with all images
            transform (callable, optional): Optional transforms
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

        # Basic validation
        if len(self.img_labels) == 0:
            raise ValueError("Dataset is empty! Check annotations file and directory")
            
        self.classes = sorted(self.img_labels['label'].unique().tolist())

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        try:
            # Load image (two methods shown)
            # Method 1: Using torchvision.io (no PIL dependency)
            # image = read_image(img_path)
            
            # Method 2: Using PIL (common for transformations)
            image = Image.open(img_path).convert('RGB')
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ================================================================== #
#                     Transforms & Preprocessing                     #
# ================================================================== #

# Define training transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define validation transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ================================================================== #
#                     Create Dataset & DataLoader                    #
# ================================================================== #

# Example directory structure:
# data/
#   train/
#     images/
#       img001.jpg
#       img002.jpg
#     labels.csv

# Create dataset instances
train_dataset = CustomImageDataset(
    annotations_file='data/train/labels.csv',
    img_dir='data/train/images',
    transform=train_transform
)

val_dataset = CustomImageDataset(
    annotations_file='data/val/labels.csv',
    img_dir='data/val/images',
    transform=val_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# ================================================================== #
#                     Visualization Utilities                       #
# ================================================================== #

def show_batch(sample_batch):
    """Show images with labels for a batch"""
    images, labels = sample_batch
    fig = plt.figure(figsize=(12, 6))
    for i in range(min(4, len(images))):  # Show first 4 images
        ax = fig.add_subplot(1, 4, i+1)
        # Denormalize image
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)
        
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.show()

# ================================================================== #
#                     Error Handling & Debugging                    #
# ================================================================== #

def safe_data_loader(loader):
    """Handle corrupt/nonexistent files in DataLoader"""
    for batch in loader:
        if None in batch:  # Handle failed samples
            print("Found None in batch, filtering...")
            batch = [t for t in zip(*batch) if None not in t]
            if len(batch) == 0:
                continue
            yield torch.utils.data.default_collate(batch)
        else:
            yield batch

# ================================================================== #
#                     Custom Collate Function                       #
# ================================================================== #

def custom_collate(batch):
    """Handle variable-sized images"""
    batch = list(filter(lambda x: x is not None, batch))  # Remove None
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    
    images, labels = zip(*batch)
    
    # Pad images to same size
    max_width = max([img.shape[2] for img in images])
    max_height = max([img.shape[1] for img in images])
    
    padded_images = []
    for img in images:
        pad = (
            0, max_width - img.shape[2],
            0, max_height - img.shape[1]
        )
        padded_images.append(torch.nn.functional.pad(img, pad))
        
    return torch.stack(padded_images), torch.tensor(labels)

# ================================================================== #
#                     Testing the Dataset                           #
# ================================================================== #

if __name__ == "__main__":
    # Test dataset with mock data (for demonstration)
    class MockDataset(CustomImageDataset):
        def __getitem__(self, idx):
            # Generate fake image
            image = torch.rand(3, 224, 224)
            label = idx % 5  # 5 classes
            return image, label

    mock_loader = DataLoader(
        MockDataset(None, None, None),
        batch_size=4,
        collate_fn=custom_collate
    )

    print("Testing with mock data...")
    batch = next(iter(mock_loader))
    print(f"Batch shape: {batch[0].shape}, Labels: {batch[1]}")
    show_batch(batch)

    print("\nDataset implementation validated!")
    print("Next: Explore dataloaders.py for advanced DataLoader configurations")