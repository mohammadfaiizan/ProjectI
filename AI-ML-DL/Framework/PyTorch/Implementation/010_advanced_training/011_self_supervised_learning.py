import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple

# SimCLR Framework
class SimCLR(nn.Module):
    """Simple Framework for Contrastive Learning of Visual Representations"""
    
    def __init__(self, encoder, projection_dim=128, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            encoder_output = self.encoder(dummy_input)
            encoder_dim = encoder_output.size(1)
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
    
    def forward(self, x1, x2):
        """Forward pass for contrastive learning"""
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to contrastive space
        z1 = F.normalize(self.projection_head(h1), dim=1)
        z2 = F.normalize(self.projection_head(h2), dim=1)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2):
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss"""
        batch_size = z1.size(0)
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for contrastive learning
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0).to(z1.device)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=bool).to(z1.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

# Data Augmentation for Contrastive Learning
class ContrastiveAugmentation:
    """Data augmentation for contrastive learning"""
    
    def __init__(self, strength=0.5):
        self.strength = strength
    
    def random_crop_and_resize(self, x, size=(32, 32)):
        """Random crop and resize"""
        # Simple random crop (for demonstration)
        h, w = x.shape[-2:]
        crop_h = int(h * (0.8 + 0.2 * random.random()))
        crop_w = int(w * (0.8 + 0.2 * random.random()))
        
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        cropped = x[..., start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        # Resize back (simplified)
        return F.interpolate(cropped.unsqueeze(0), size=size, mode='bilinear').squeeze(0)
    
    def color_jitter(self, x):
        """Color jittering"""
        # Simple color jittering by adding noise
        noise = torch.randn_like(x) * 0.1 * self.strength
        return torch.clamp(x + noise, 0, 1)
    
    def gaussian_blur(self, x):
        """Gaussian blur (simplified)"""
        # Simple blur by averaging with neighbors
        kernel = torch.ones(1, 1, 3, 3) / 9
        kernel = kernel.to(x.device)
        
        blurred = F.conv2d(x.unsqueeze(0), kernel, padding=1, groups=1)
        return blurred.squeeze(0)
    
    def random_horizontal_flip(self, x):
        """Random horizontal flip"""
        if random.random() > 0.5:
            return torch.flip(x, [-1])
        return x
    
    def __call__(self, x):
        """Apply random augmentations"""
        # Random crop and resize
        x = self.random_crop_and_resize(x)
        
        # Random horizontal flip
        x = self.random_horizontal_flip(x)
        
        # Color jitter
        if random.random() > 0.2:
            x = self.color_jitter(x)
        
        # Gaussian blur
        if random.random() > 0.5:
            x = self.gaussian_blur(x)
        
        return x

# Pretext Task: Rotation Prediction
class RotationPredictor(nn.Module):
    """Rotation prediction pretext task"""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            encoder_output = self.encoder(dummy_input)
            encoder_dim = encoder_output.size(1)
        
        # Classifier for 4 rotation classes (0°, 90°, 180°, 270°)
        self.rotation_classifier = nn.Linear(encoder_dim, 4)
    
    def rotate_image(self, x, rotation):
        """Rotate image by specified degrees"""
        if rotation == 0:
            return x
        elif rotation == 90:
            return torch.rot90(x, k=1, dims=[-2, -1])
        elif rotation == 180:
            return torch.rot90(x, k=2, dims=[-2, -1])
        elif rotation == 270:
            return torch.rot90(x, k=3, dims=[-2, -1])
        else:
            raise ValueError(f"Invalid rotation: {rotation}")
    
    def forward(self, x):
        """Forward pass for rotation prediction"""
        batch_size = x.size(0)
        
        # Create rotated versions
        rotations = [0, 90, 180, 270]
        rotated_images = []
        rotation_labels = []
        
        for i, rotation in enumerate(rotations):
            rotated = self.rotate_image(x, rotation)
            rotated_images.append(rotated)
            rotation_labels.extend([i] * batch_size)
        
        # Stack all rotated images
        all_images = torch.cat(rotated_images, dim=0)
        rotation_labels = torch.tensor(rotation_labels).to(x.device)
        
        # Encode and classify
        features = self.encoder(all_images)
        predictions = self.rotation_classifier(features)
        
        return predictions, rotation_labels

# Pretext Task: Jigsaw Puzzle
class JigsawPuzzle(nn.Module):
    """Jigsaw puzzle pretext task"""
    
    def __init__(self, encoder, grid_size=3):
        super().__init__()
        self.encoder = encoder
        self.grid_size = grid_size
        self.num_patches = grid_size ** 2
        
        # Predefined permutations (simplified - using a few permutations)
        self.permutations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Original
            [0, 1, 2, 6, 7, 8, 3, 4, 5],  # Swap middle and bottom rows
            [2, 1, 0, 5, 4, 3, 8, 7, 6],  # Horizontal flip
            [6, 7, 8, 3, 4, 5, 0, 1, 2],  # Vertical flip
        ]
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            encoder_output = self.encoder(dummy_input)
            encoder_dim = encoder_output.size(1)
        
        # Classifier for permutation prediction
        self.permutation_classifier = nn.Linear(encoder_dim, len(self.permutations))
    
    def create_patches(self, x):
        """Split image into patches"""
        batch_size, channels, height, width = x.shape
        patch_height = height // self.grid_size
        patch_width = width // self.grid_size
        
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                h_start = i * patch_height
                h_end = (i + 1) * patch_height
                w_start = j * patch_width
                w_end = (j + 1) * patch_width
                
                patch = x[:, :, h_start:h_end, w_start:w_end]
                patches.append(patch)
        
        return patches
    
    def reconstruct_from_patches(self, patches, permutation):
        """Reconstruct image from permuted patches"""
        # Reorder patches according to permutation
        reordered_patches = [patches[permutation[i]] for i in range(len(patches))]
        
        # Reconstruct image
        rows = []
        for i in range(self.grid_size):
            row_patches = reordered_patches[i * self.grid_size:(i + 1) * self.grid_size]
            row = torch.cat(row_patches, dim=-1)
            rows.append(row)
        
        reconstructed = torch.cat(rows, dim=-2)
        return reconstructed
    
    def forward(self, x):
        """Forward pass for jigsaw puzzle"""
        batch_size = x.size(0)
        
        all_puzzles = []
        puzzle_labels = []
        
        for perm_idx, permutation in enumerate(self.permutations):
            patches = self.create_patches(x)
            puzzle = self.reconstruct_from_patches(patches, permutation)
            
            all_puzzles.append(puzzle)
            puzzle_labels.extend([perm_idx] * batch_size)
        
        # Stack all puzzles
        all_puzzles = torch.cat(all_puzzles, dim=0)
        puzzle_labels = torch.tensor(puzzle_labels).to(x.device)
        
        # Encode and classify
        features = self.encoder(all_puzzles)
        predictions = self.permutation_classifier(features)
        
        return predictions, puzzle_labels

# Self-Supervised Training Dataset
class SelfSupervisedDataset(Dataset):
    """Dataset for self-supervised learning"""
    
    def __init__(self, size=1000, input_shape=(3, 32, 32), augmentation=None):
        self.size = size
        self.data = torch.randn(size, *input_shape)
        self.augmentation = augmentation
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.augmentation:
            # Create two augmented views for contrastive learning
            x1 = self.augmentation(x)
            x2 = self.augmentation(x)
            return x1, x2
        
        return x

# Encoder Network
class SimpleEncoder(nn.Module):
    """Simple encoder for self-supervised learning"""
    
    def __init__(self, input_channels=3, hidden_dim=64, output_dim=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Self-Supervised Trainer
class SelfSupervisedTrainer:
    """Trainer for self-supervised learning"""
    
    def __init__(self, model, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_contrastive(self, dataloader, epochs=5):
        """Train with contrastive learning (SimCLR)"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (x1, x2) in enumerate(dataloader):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                z1, z2 = self.model(x1, x2)
                loss = self.model.contrastive_loss(z1, z2)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.4f}')
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}')
    
    def train_pretext_task(self, dataloader, epochs=5):
        """Train with pretext task"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, x in enumerate(dataloader):
                if isinstance(x, tuple):
                    x = x[0]  # Take first element if tuple
                
                x = x.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions, labels = self.model(x)
                loss = self.criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.4f}')
            
            avg_loss = total_loss / len(dataloader)
            accuracy = 100. * correct / total
            print(f'Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')

# Linear Evaluation Protocol
def linear_evaluation(encoder, train_dataset, test_dataset, num_classes=10, device='cuda'):
    """Linear evaluation of learned representations"""
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Create linear classifier
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        encoder_output = encoder(dummy_input)
        feature_dim = encoder_output.size(1)
    
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train classifier
    classifier.train()
    for epoch in range(10):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            with torch.no_grad():
                features = encoder(data)
            
            # Classify
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
    
    # Evaluate
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            features = encoder(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# Supervised Dataset (for linear evaluation)
class SupervisedDataset(Dataset):
    """Supervised dataset for linear evaluation"""
    
    def __init__(self, size=500, input_shape=(3, 32, 32), num_classes=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

if __name__ == "__main__":
    print("Self-Supervised Learning")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    augmentation = ContrastiveAugmentation(strength=0.5)
    ssl_dataset = SelfSupervisedDataset(size=500, augmentation=augmentation)
    ssl_loader = DataLoader(ssl_dataset, batch_size=16, shuffle=True)
    
    # Test SimCLR
    print("\n1. Testing SimCLR")
    print("-" * 20)
    
    encoder = SimpleEncoder(output_dim=128)
    simclr = SimCLR(encoder, projection_dim=64, temperature=0.07)
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    trainer = SelfSupervisedTrainer(simclr, device)
    trainer.train_contrastive(ssl_loader, epochs=3)
    
    # Test Rotation Prediction
    print("\n2. Testing Rotation Prediction")
    print("-" * 30)
    
    encoder_rot = SimpleEncoder(output_dim=128)
    rotation_model = RotationPredictor(encoder_rot)
    
    # Create dataset without augmentation for pretext tasks
    pretext_dataset = SelfSupervisedDataset(size=300, augmentation=None)
    pretext_loader = DataLoader(pretext_dataset, batch_size=8, shuffle=True)
    
    rotation_trainer = SelfSupervisedTrainer(rotation_model, device)
    rotation_trainer.train_pretext_task(pretext_loader, epochs=3)
    
    # Test Jigsaw Puzzle
    print("\n3. Testing Jigsaw Puzzle")
    print("-" * 25)
    
    encoder_jigsaw = SimpleEncoder(output_dim=128)
    jigsaw_model = JigsawPuzzle(encoder_jigsaw, grid_size=3)
    
    jigsaw_trainer = SelfSupervisedTrainer(jigsaw_model, device)
    jigsaw_trainer.train_pretext_task(pretext_loader, epochs=2)
    
    # Linear Evaluation
    print("\n4. Linear Evaluation")
    print("-" * 20)
    
    # Create supervised datasets for evaluation
    train_supervised = SupervisedDataset(size=300, num_classes=5)
    test_supervised = SupervisedDataset(size=100, num_classes=5)
    
    # Evaluate SimCLR encoder
    simclr_accuracy = linear_evaluation(
        simclr.encoder, train_supervised, test_supervised, 
        num_classes=5, device=device
    )
    print(f"SimCLR linear evaluation accuracy: {simclr_accuracy:.2f}%")
    
    # Evaluate rotation encoder
    rotation_accuracy = linear_evaluation(
        rotation_model.encoder, train_supervised, test_supervised,
        num_classes=5, device=device
    )
    print(f"Rotation prediction linear evaluation accuracy: {rotation_accuracy:.2f}%")
    
    # Random encoder baseline
    random_encoder = SimpleEncoder(output_dim=128)
    random_accuracy = linear_evaluation(
        random_encoder, train_supervised, test_supervised,
        num_classes=5, device=device
    )
    print(f"Random encoder baseline accuracy: {random_accuracy:.2f}%")
    
    # Summary
    print("\n5. Results Summary")
    print("-" * 20)
    print(f"Random baseline: {random_accuracy:.2f}%")
    print(f"Rotation prediction: {rotation_accuracy:.2f}%")
    print(f"SimCLR: {simclr_accuracy:.2f}%")
    
    print("\nSelf-supervised learning demonstrations completed!") 