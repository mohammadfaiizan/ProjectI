"""
ERA 5: LARGE-SCALE TEXT-TO-IMAGE - CLIP-Guided Generation
=========================================================

Year: 2021
Paper: "Learning Transferable Visual Representations from Natural Language Supervision" (Radford et al., OpenAI)
Innovation: Contrastive vision-language training for zero-shot classification and guided generation
Previous Limitation: Poor text-image alignment and limited controllability in generation
Performance Gain: Strong semantic understanding enabling precise text-guided generation
Impact: Foundation for controllable generation and modern text-to-image guidance systems

This file implements CLIP-guided generation that revolutionized controllable image generation
through vision-language alignment and established the foundation for guided diffusion models.
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

YEAR = "2021"
INNOVATION = "Contrastive vision-language training for zero-shot classification and guided generation"
PREVIOUS_LIMITATION = "Poor text-image alignment and limited controllability in generation"
IMPACT = "Foundation for controllable generation and modern text-to-image guidance systems"

print(f"=== CLIP-Guided Generation ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# CLIP PRINCIPLES
# ============================================================================

CLIP_PRINCIPLES = {
    "contrastive_learning": "Learn vision-language alignment through contrastive objectives",
    "natural_supervision": "Train on natural language descriptions without manual annotations",
    "zero_shot_transfer": "Enable zero-shot image classification without task-specific training",
    "scalable_training": "Train on 400M image-text pairs from internet",
    "multimodal_embeddings": "Create shared embedding space for images and text",
    "generation_guidance": "Guide generation models toward desired text descriptions",
    "compositionality": "Understand complex compositional concepts and relationships",
    "robust_representations": "Learn robust features generalizing across domains and tasks"
}

print("CLIP Principles:")
for key, principle in CLIP_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10 + Text)
# ============================================================================

def load_cifar10_clip():
    """Load CIFAR-10 dataset for CLIP and guided generation study"""
    print("Loading CIFAR-10 dataset for CLIP-guided generation study...")
    print("Note: CLIP trained on 400M internet image-text pairs")
    
    # CLIP preprocessing (standard for vision models)
    transform_train = transforms.Compose([
        transforms.Resize(224),  # CLIP uses 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # CIFAR-10 class names with detailed descriptions
    class_descriptions = {
        0: ["a photo of an airplane", "an aircraft in flight", "a commercial airliner", "a military aircraft"],
        1: ["a photo of an automobile", "a car on the road", "a sedan vehicle", "a passenger car"],
        2: ["a photo of a bird", "a flying bird", "a small songbird", "a colorful bird"],
        3: ["a photo of a cat", "a domestic cat", "a feline pet", "a cute kitten"],
        4: ["a photo of a deer", "a wild deer", "a forest deer", "a graceful deer"],
        5: ["a photo of a dog", "a domestic dog", "a loyal pet", "a friendly puppy"],
        6: ["a photo of a frog", "an amphibian", "a green frog", "a pond frog"],
        7: ["a photo of a horse", "an equine animal", "a riding horse", "a wild stallion"],
        8: ["a photo of a ship", "a naval vessel", "a cargo ship", "a sailing boat"],
        9: ["a photo of a truck", "a commercial truck", "a delivery vehicle", "a heavy truck"]
    }
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(class_names)}")
    print(f"Image size: 224x224 RGB (CLIP standard)")
    print(f"Focus: Vision-language alignment and guided generation")
    
    return train_loader, test_loader, class_names, class_descriptions

# ============================================================================
# TEXT ENCODER
# ============================================================================

class SimpleTextEncoder(nn.Module):
    """
    Simplified text encoder for CLIP demonstration
    
    In practice, CLIP uses a Transformer encoder
    """
    
    def __init__(self, vocab_size=10000, embed_dim=512, hidden_dim=512, num_layers=6):
        super(SimpleTextEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding (for sequences up to 77 tokens like CLIP)
        self.position_embedding = nn.Embedding(77, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Simple vocabulary for demonstration
        self.vocab = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2,
            'a': 3, 'photo': 4, 'of': 5, 'an': 6, 'the': 7,
            'airplane': 10, 'automobile': 11, 'bird': 12, 'cat': 13, 'deer': 14,
            'dog': 15, 'frog': 16, 'horse': 17, 'ship': 18, 'truck': 19,
            'flying': 20, 'car': 21, 'small': 22, 'cute': 23, 'wild': 24,
            'domestic': 25, 'green': 26, 'commercial': 27, 'red': 28, 'blue': 29
        }
        
        print(f"  Text Encoder: {embed_dim}D, {num_layers} layers, {len(self.vocab)} vocab")
    
    def encode_text(self, text_list):
        """Encode list of text strings"""
        encoded = []
        for text in text_list:
            tokens = text.lower().split()
            indices = [self.vocab.get(token, 0) for token in tokens]  # Use 0 for unknown
            indices = [self.vocab['<SOS>']] + indices + [self.vocab['<EOS>']]
            encoded.append(indices)
        
        # Pad sequences
        max_len = min(77, max(len(seq) for seq in encoded))
        padded = []
        for seq in encoded:
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [self.vocab['<PAD>']] * (max_len - len(seq))
            padded.append(seq)
        
        return torch.tensor(padded)
    
    def forward(self, text_tokens):
        """
        Forward pass through text encoder
        
        Args:
            text_tokens: Tokenized text (batch_size, seq_len)
        
        Returns:
            Text features (batch_size, embed_dim)
        """
        batch_size, seq_len = text_tokens.shape
        device = text_tokens.device
        
        # Token embeddings
        token_embeds = self.token_embedding(text_tokens)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Create attention mask (ignore padding)
        attn_mask = (text_tokens == self.vocab['<PAD>'])
        
        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Use <EOS> token representation or pooling
        # Find <EOS> positions
        eos_positions = (text_tokens == self.vocab['<EOS>']).float()
        eos_positions = eos_positions / (eos_positions.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted pooling
        text_features = torch.sum(x * eos_positions.unsqueeze(-1), dim=1)
        
        # Layer norm
        text_features = self.ln_final(text_features)
        
        return text_features

# ============================================================================
# VISION ENCODER
# ============================================================================

class VisionTransformer(nn.Module):
    """
    Vision Transformer for CLIP image encoding
    
    Simplified version of ViT used in CLIP
    """
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, 
                 embed_dim=512, num_layers=12, num_heads=8):
        super(VisionTransformer, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        print(f"  Vision Transformer: {image_size}x{image_size}, {patch_size}x{patch_size} patches")
        print(f"    {self.num_patches} patches, {embed_dim}D, {num_layers} layers")
    
    def forward(self, images):
        """
        Forward pass through vision encoder
        
        Args:
            images: Input images (batch_size, channels, height, width)
        
        Returns:
            Image features (batch_size, embed_dim)
        """
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # (batch_size, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Use class token for image representation
        image_features = self.ln_final(x[:, 0])  # (batch_size, embed_dim)
        
        return image_features

# ============================================================================
# CLIP MODEL
# ============================================================================

class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) Model
    
    Revolutionary Innovations:
    - Contrastive learning on image-text pairs
    - Zero-shot transfer to downstream tasks
    - Natural language supervision at scale
    - Multimodal embedding space
    - Foundation for guided generation
    """
    
    def __init__(self, embed_dim=512, temperature=0.07):
        super(CLIP, self).__init__()
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        print(f"Building CLIP Model...")
        
        # Vision encoder
        self.vision_encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            embed_dim=embed_dim,
            num_layers=12,
            num_heads=8
        )
        
        # Text encoder
        self.text_encoder = SimpleTextEncoder(
            embed_dim=embed_dim,
            num_layers=6
        )
        
        # Projection layers
        self.vision_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # Calculate statistics
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        total_params = vision_params + text_params
        
        print(f"CLIP Model Summary:")
        print(f"  Vision encoder parameters: {vision_params:,}")
        print(f"  Text encoder parameters: {text_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Key innovation: Contrastive vision-language learning")
    
    def encode_image(self, images):
        """Encode images to embedding space"""
        image_features = self.vision_encoder(images)
        image_features = self.vision_projection(image_features)
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, text_list):
        """Encode text to embedding space"""
        if isinstance(text_list, list):
            text_tokens = self.text_encoder.encode_text(text_list)
        else:
            text_tokens = text_list
        
        text_features = self.text_encoder(text_tokens.to(next(self.parameters()).device))
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)
    
    def forward(self, images, text_list):
        """
        Forward pass for contrastive learning
        
        Args:
            images: Batch of images
            text_list: List of text descriptions
        
        Returns:
            Logits for contrastive loss
        """
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_list)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def get_similarity(self, images, text_list):
        """Get similarity scores between images and text"""
        with torch.no_grad():
            image_features = self.encode_image(images)
            text_features = self.encode_text(text_list)
            
            # Cosine similarity
            similarity = image_features @ text_features.t()
            return similarity
    
    def zero_shot_classify(self, images, class_descriptions):
        """
        Zero-shot classification using text descriptions
        
        Args:
            images: Batch of images
            class_descriptions: List of text descriptions for each class
        
        Returns:
            Classification probabilities
        """
        with torch.no_grad():
            # Encode images
            image_features = self.encode_image(images)
            
            # Encode all class descriptions
            all_texts = []
            for descriptions in class_descriptions.values():
                all_texts.extend(descriptions)
            
            text_features = self.encode_text(all_texts)
            
            # Reshape text features by class
            num_classes = len(class_descriptions)
            descriptions_per_class = len(list(class_descriptions.values())[0])
            text_features = text_features.view(num_classes, descriptions_per_class, -1)
            
            # Average over descriptions per class
            text_features = text_features.mean(dim=1)  # (num_classes, embed_dim)
            
            # Compute similarities
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            
            return F.softmax(logits, dim=-1)

# ============================================================================
# GUIDED GENERATION COMPONENTS
# ============================================================================

class SimpleGenerator(nn.Module):
    """
    Simple generator for demonstrating CLIP guidance
    
    In practice, this would be a sophisticated generator like StyleGAN or diffusion model
    """
    
    def __init__(self, latent_dim=128, image_size=224):
        super(SimpleGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Simple deconvolutional generator
        self.generator = nn.Sequential(
            # Start with 4x4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 224x224 (approximately)
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        print(f"  Simple Generator: {latent_dim}D -> {image_size}x{image_size}")
    
    def forward(self, z):
        """Generate images from latent codes"""
        # Reshape for deconvolution
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.generator(z)

class CLIPGuidedGenerator(nn.Module):
    """
    CLIP-Guided Generator
    
    Uses CLIP to guide generation toward desired text descriptions
    """
    
    def __init__(self, clip_model, generator, latent_dim=128):
        super(CLIPGuidedGenerator, self).__init__()
        
        self.clip_model = clip_model
        self.generator = generator
        self.latent_dim = latent_dim
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print(f"CLIP-Guided Generator initialized")
    
    def clip_loss(self, images, target_text, additional_texts=None):
        """
        Compute CLIP loss for guiding generation
        
        Args:
            images: Generated images
            target_text: Target text description
            additional_texts: Additional negative texts for contrastive learning
        
        Returns:
            CLIP guidance loss
        """
        # Encode generated images
        image_features = self.clip_model.encode_image(images)
        
        # Encode target text
        target_text_features = self.clip_model.encode_text(target_text)
        
        # Positive similarity (maximize)
        positive_similarity = torch.mean(image_features * target_text_features)
        
        loss = -positive_similarity  # Negative because we want to maximize
        
        # Optional: Add contrastive loss with negative texts
        if additional_texts is not None:
            negative_text_features = self.clip_model.encode_text(additional_texts)
            negative_similarity = torch.mean(image_features @ negative_text_features.t())
            loss += negative_similarity  # Minimize similarity to negative texts
        
        return loss
    
    def generate_guided(self, target_text, num_samples=4, num_iterations=100, 
                       learning_rate=0.1, additional_texts=None):
        """
        Generate images guided by CLIP toward target text
        
        Args:
            target_text: Target text description
            num_samples: Number of images to generate
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for latent optimization
            additional_texts: Additional texts for contrastive guidance
        
        Returns:
            Generated images and latent codes
        """
        device = next(self.parameters()).device
        
        # Initialize random latent codes
        z = torch.randn(num_samples, self.latent_dim, device=device, requires_grad=True)
        
        # Optimizer for latent codes
        optimizer = optim.Adam([z], lr=learning_rate)
        
        # Target text (replicate for batch)
        if isinstance(target_text, str):
            target_text = [target_text] * num_samples
        
        best_images = None
        best_loss = float('inf')
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Generate images
            images = self.generator(z)
            
            # Compute CLIP loss
            loss = self.clip_loss(images, target_text, additional_texts)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # Track best results
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_images = images.detach().clone()
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/{num_iterations}, CLIP Loss: {loss.item():.4f}")
        
        return best_images, z.detach()

# ============================================================================
# CLIP TRAINING FUNCTION
# ============================================================================

def train_clip(model, train_loader, class_descriptions, epochs=50, learning_rate=1e-4):
    """Train CLIP model with contrastive loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training tracking
    losses = []
    accuracies = []
    
    print(f"Training CLIP on device: {device}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get text descriptions for this batch
            batch_texts = []
            for label in labels:
                # Randomly sample from available descriptions
                descriptions = class_descriptions[label.item()]
                text = np.random.choice(descriptions)
                batch_texts.append(text)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits_per_image, logits_per_text = model(images, batch_texts)
            
            # Contrastive loss (symmetric)
            batch_size = images.shape[0]
            labels_contrastive = torch.arange(batch_size, device=device)
            
            loss_img = F.cross_entropy(logits_per_image, labels_contrastive)
            loss_txt = F.cross_entropy(logits_per_text, labels_contrastive)
            loss = (loss_img + loss_txt) / 2
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            pred_img = logits_per_image.argmax(dim=1)
            pred_txt = logits_per_text.argmax(dim=1)
            acc = ((pred_img == labels_contrastive).float().mean() + 
                   (pred_txt == labels_contrastive).float().mean()) / 2
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch averages
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_acc
            }, f'AI-ML-DL/Models/Generative_AI/clip_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_acc > 0.8:
            print(f"Good accuracy reached at epoch {epoch+1}")
            break
    
    return losses, accuracies

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_clip_concept():
    """Visualize CLIP core concepts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Contrastive learning concept
    ax = axes[0, 0]
    ax.set_title('CLIP Contrastive Learning', fontsize=14, fontweight='bold')
    
    # Show image-text pair matching
    images_x = [0.2, 0.2, 0.8, 0.8]
    images_y = [0.8, 0.6, 0.8, 0.6]
    texts_x = [0.2, 0.2, 0.8, 0.8]
    texts_y = [0.4, 0.2, 0.4, 0.2]
    
    # Correct pairs
    correct_pairs = [(0, 0), (1, 1)]
    incorrect_pairs = [(0, 1), (1, 0), (0, 2), (0, 3), (1, 2), (1, 3)]
    
    # Draw images
    for i, (x, y) in enumerate(zip(images_x, images_y)):
        color = 'lightblue' if i < 2 else 'lightcoral'
        rect = plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1, 
                           facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, f'Image {i+1}', ha='center', va='center', fontweight='bold')
    
    # Draw texts
    for i, (x, y) in enumerate(zip(texts_x, texts_y)):
        color = 'lightgreen' if i < 2 else 'lightyellow'
        rect = plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1, 
                           facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, f'Text {i+1}', ha='center', va='center', fontweight='bold')
    
    # Draw connections
    # Positive pairs (green)
    for img_idx, txt_idx in correct_pairs:
        ax.annotate('', xy=(texts_x[txt_idx], texts_y[txt_idx] + 0.05), 
                   xytext=(images_x[img_idx], images_y[img_idx] - 0.05),
                   arrowprops=dict(arrowstyle='<->', lw=3, color='green'))
        ax.text((images_x[img_idx] + texts_x[txt_idx])/2, 
               (images_y[img_idx] + texts_y[txt_idx])/2, '+', 
               ha='center', va='center', fontsize=16, fontweight='bold', color='green')
    
    # Negative pairs (red) - show a few
    for img_idx, txt_idx in incorrect_pairs[:2]:
        ax.annotate('', xy=(texts_x[txt_idx], texts_y[txt_idx] + 0.05), 
                   xytext=(images_x[img_idx], images_y[img_idx] - 0.05),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='red', alpha=0.5))
        ax.text((images_x[img_idx] + texts_x[txt_idx])/2, 
               (images_y[img_idx] + texts_y[txt_idx])/2, '−', 
               ha='center', va='center', fontsize=16, fontweight='bold', color='red')
    
    ax.text(0.5, 0.05, 'Maximize similarity for correct pairs, minimize for incorrect pairs', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Zero-shot classification
    ax = axes[0, 1]
    ax.set_title('Zero-Shot Classification', fontsize=14)
    
    # Test image
    test_rect = plt.Rectangle((0.1, 0.7), 0.2, 0.2, 
                            facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(test_rect)
    ax.text(0.2, 0.8, 'Test\nImage', ha='center', va='center', fontweight='bold')
    
    # Class descriptions
    class_names = ['airplane', 'car', 'bird', 'cat']
    class_colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        y_pos = 0.7 - i * 0.15
        rect = plt.Rectangle((0.6, y_pos), 0.25, 0.1, 
                           facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.725, y_pos + 0.05, f'"a photo of {name}"', 
               ha='center', va='center', fontweight='bold')
        
        # Similarity score
        similarity = [0.9, 0.3, 0.1, 0.2][i]  # Airplane is highest
        ax.text(0.9, y_pos + 0.05, f'{similarity:.1f}', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='green' if i == 0 else 'black')
        
        # Arrow with similarity
        ax.annotate('', xy=(0.6, y_pos + 0.05), xytext=(0.3, 0.8),
                   arrowprops=dict(arrowstyle='->', lw=2, 
                                 color='green' if i == 0 else 'gray',
                                 alpha=similarity))
    
    ax.text(0.5, 0.05, 'No task-specific training needed!', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Guided generation process
    ax = axes[1, 0]
    ax.set_title('CLIP-Guided Generation Process', fontsize=14)
    
    # Show iterative optimization
    steps = ['Random\nImage', 'CLIP\nScore', 'Gradient\nUpdate', 'Improved\nImage', 'Target\nText']
    positions = [(0.1, 0.7), (0.3, 0.5), (0.5, 0.7), (0.7, 0.5), (0.9, 0.7)]
    colors = ['lightgray', 'yellow', 'orange', 'lightgreen', 'lightblue']
    
    for i, ((x, y), step, color) in enumerate(zip(positions, steps, colors)):
        rect = plt.Rectangle((x-0.06, y-0.08), 0.12, 0.16, 
                           facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, step, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Arrows
        if i < len(positions) - 1 and i != 3:  # Skip arrow from improved image
            next_pos = positions[i+1]
            ax.annotate('', xy=(next_pos[0]-0.06, next_pos[1]), 
                       xytext=(x+0.06, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Special arrows
    # From CLIP score back to update
    ax.annotate('', xy=(0.44, 0.7), xytext=(0.36, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # From target text to CLIP score
    ax.annotate('', xy=(0.36, 0.5), xytext=(0.84, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple',
                             connectionstyle="arc3,rad=-0.3"))
    
    # Iteration loop
    ax.annotate('', xy=(0.1, 0.6), xytext=(0.7, 0.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='green',
                             connectionstyle="arc3,rad=0.5"))
    ax.text(0.4, 0.3, 'Iterate', ha='center', va='center', fontweight='bold', color='green')
    
    ax.text(0.5, 0.1, 'Optimize latent code to maximize CLIP similarity with target text', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # CLIP embedding space
    ax = axes[1, 1]
    ax.set_title('Multimodal Embedding Space', fontsize=14)
    
    # Create 2D visualization of embedding space
    np.random.seed(42)
    
    # Image embeddings
    img_embeddings = np.random.randn(20, 2) * 0.5
    img_embeddings[:5] += [1, 1]    # Cars
    img_embeddings[5:10] += [-1, 1]  # Birds
    img_embeddings[10:15] += [1, -1] # Cats
    img_embeddings[15:20] += [-1, -1] # Dogs
    
    # Text embeddings (close to corresponding images)
    text_embeddings = img_embeddings + np.random.randn(20, 2) * 0.1
    
    # Plot embeddings
    colors = ['red', 'blue', 'green', 'purple']
    labels = ['Cars', 'Birds', 'Cats', 'Dogs']
    
    for i in range(4):
        start_idx = i * 5
        end_idx = (i + 1) * 5
        
        # Image embeddings
        ax.scatter(img_embeddings[start_idx:end_idx, 0], 
                  img_embeddings[start_idx:end_idx, 1],
                  c=colors[i], marker='s', s=100, alpha=0.7, 
                  label=f'{labels[i]} (Images)')
        
        # Text embeddings
        ax.scatter(text_embeddings[start_idx:end_idx, 0], 
                  text_embeddings[start_idx:end_idx, 1],
                  c=colors[i], marker='o', s=100, alpha=0.7, 
                  label=f'{labels[i]} (Text)')
        
        # Connection lines
        for j in range(5):
            ax.plot([img_embeddings[start_idx + j, 0], text_embeddings[start_idx + j, 0]],
                   [img_embeddings[start_idx + j, 1], text_embeddings[start_idx + j, 1]],
                   c=colors[i], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0, -2, 'Images and text with similar semantics\nare close in embedding space', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/014_clip_concept.png', dpi=300, bbox_inches='tight')
    print("CLIP concept visualization saved: 014_clip_concept.png")

def visualize_clip_applications():
    """Visualize CLIP applications and capabilities"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Zero-shot performance comparison
    ax = axes[0, 0]
    ax.set_title('Zero-Shot vs Supervised Performance', fontsize=14, fontweight='bold')
    
    datasets = ['ImageNet', 'CIFAR-10', 'Food101', 'Pets', 'Cars']
    clip_scores = [76.2, 95.3, 88.9, 87.0, 65.1]
    supervised_scores = [88.5, 98.7, 92.4, 91.2, 78.3]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clip_scores, width, label='CLIP Zero-Shot', 
                  color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, supervised_scores, width, label='Supervised', 
                  color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Highlight key insight
    ax.text(len(datasets)/2, 50, 'CLIP achieves competitive performance\nwithout task-specific training!', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Scaling behavior
    ax = axes[0, 1]
    ax.set_title('CLIP Scaling: Data vs Performance', fontsize=14)
    
    # Data points for scaling
    data_sizes = [1e6, 5e6, 1e7, 5e7, 1e8, 4e8]  # Training data size
    performance = [45, 55, 62, 68, 72, 76]  # ImageNet zero-shot accuracy
    
    # Log scale
    log_data = np.log10(data_sizes)
    
    ax.semilogx(data_sizes, performance, 'bo-', linewidth=3, markersize=8)
    
    # Fit trend line
    z = np.polyfit(log_data, performance, 1)
    p = np.poly1d(z)
    ax.semilogx(data_sizes, p(log_data), 'r--', linewidth=2, alpha=0.7, label='Trend')
    
    ax.set_xlabel('Training Data Size')
    ax.set_ylabel('ImageNet Zero-Shot Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotations
    ax.text(1e7, 50, 'Performance scales\nwith data size', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.text(4e8, 76, 'CLIP\n400M pairs', ha='center', va='bottom', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    # Generation guidance applications
    ax = axes[1, 0]
    ax.set_title('CLIP Guidance Applications', fontsize=14)
    
    applications = ['Text-to-\nImage', 'Style\nTransfer', 'Image\nEditing', 'Video\nGeneration', '3D\nGeneration']
    effectiveness = [9, 8, 7, 6, 5]  # Relative effectiveness scores
    
    bars = ax.bar(applications, effectiveness, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'])
    
    # Add value labels
    for bar, score in zip(bars, effectiveness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{score}/10', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Effectiveness Score')
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Examples
    examples = [
        '"a red sports car"',
        '"in Van Gogh style"',
        '"change to winter"',
        '"a dog running"',
        '"3D model of car"'
    ]
    
    for i, (bar, example) in enumerate(zip(bars, examples)):
        ax.text(bar.get_x() + bar.get_width()/2, 1 + i*0.5,
               example, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.7))
    
    # Robustness and generalization
    ax = axes[1, 1]
    ax.set_title('CLIP Robustness Capabilities', fontsize=14)
    
    # Create radar chart for robustness
    capabilities = ['Distribution\nShift', 'Adversarial\nAttacks', 'Out-of-\nDomain', 
                   'Few-Shot\nLearning', 'Compositional\nReasoning', 'Fine-Grained\nClassification']
    
    # Scores compared to supervised models
    clip_scores = [8, 6, 9, 8, 7, 6]
    supervised_scores = [5, 4, 4, 6, 5, 8]
    
    # Convert to radar chart
    angles = np.linspace(0, 2*np.pi, len(capabilities), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    clip_scores = np.concatenate((clip_scores, [clip_scores[0]]))
    supervised_scores = np.concatenate((supervised_scores, [supervised_scores[0]]))
    
    # Create radar plot
    ax.plot(angles, clip_scores, 'bo-', linewidth=2, label='CLIP', alpha=0.7)
    ax.fill(angles, clip_scores, alpha=0.25, color='blue')
    
    ax.plot(angles, supervised_scores, 'ro-', linewidth=2, label='Supervised', alpha=0.7)
    ax.fill(angles, supervised_scores, alpha=0.25, color='red')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/014_clip_applications.png', dpi=300, bbox_inches='tight')
    print("CLIP applications visualization saved: 014_clip_applications.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== CLIP-Guided Generation Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset with descriptions
    train_loader, test_loader, class_names, class_descriptions = load_cifar10_clip()
    
    # Initialize CLIP model
    clip_model = CLIP(embed_dim=512, temperature=0.07)
    
    # Initialize simple generator for demonstration
    simple_generator = SimpleGenerator(latent_dim=128, image_size=224)
    
    # Initialize CLIP-guided generator
    guided_generator = CLIPGuidedGenerator(clip_model, simple_generator)
    
    # Analyze model properties
    clip_params = sum(p.numel() for p in clip_model.parameters())
    generator_params = sum(p.numel() for p in simple_generator.parameters())
    
    print(f"\nCLIP-Guided Generation Analysis:")
    print(f"  CLIP parameters: {clip_params:,}")
    print(f"  Generator parameters: {generator_params:,}")
    print(f"  Total parameters: {clip_params + generator_params:,}")
    
    # Get CLIP analysis
    clip_analysis = {
        'clip_principles': CLIP_PRINCIPLES,
        'architectural_innovations': [
            'Contrastive learning on image-text pairs',
            'Vision Transformer for image encoding',
            'Transformer for text encoding',
            'Shared multimodal embedding space',
            'Natural language supervision at scale'
        ],
        'training_methodology': [
            '400M image-text pairs from internet',
            'Contrastive loss for alignment',
            'Large-scale distributed training',
            'Careful data curation and filtering',
            'Zero-shot transfer evaluation'
        ],
        'guidance_capabilities': [
            'Text-to-image generation guidance',
            'Style transfer control',
            'Image editing direction',
            'Semantic manipulation',
            'Compositional understanding'
        ]
    }
    
    print(f"\nCLIP Innovations:")
    for key, value in clip_analysis.items():
        if isinstance(value, list):
            print(f"  {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"    • {item}")
        elif isinstance(value, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating CLIP analysis...")
    visualize_clip_concept()
    visualize_clip_applications()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("CLIP-GUIDED GENERATION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nCLIP REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. CONTRASTIVE VISION-LANGUAGE LEARNING:")
    print("   • Learn from natural language supervision")
    print("   • Contrastive loss: maximize similarity for correct pairs")
    print("   • 400M image-text pairs from internet")
    print("   • No manual annotation or task-specific labels")
    
    print("\n2. ZERO-SHOT TRANSFER:")
    print("   • Classify images without task-specific training")
    print("   • 'A photo of a [class]' text templates")
    print("   • Competitive with supervised methods")
    print("   • Generalizes across domains and distributions")
    
    print("\n3. MULTIMODAL EMBEDDING SPACE:")
    print("   • Joint representation for images and text")
    print("   • Cosine similarity for semantic alignment")
    print("   • Enables cross-modal reasoning and search")
    print("   • Foundation for guidance applications")
    
    print("\n4. GENERATION GUIDANCE:")
    print("   • Guide generation toward text descriptions")
    print("   • Optimize latent codes using CLIP loss")
    print("   • Enable controllable text-to-image generation")
    print("   • Foundation for guided diffusion models")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Strong zero-shot classification performance")
    print("• Robust cross-domain generalization")
    print("• Effective generation guidance capabilities")
    print("• Scalable contrastive learning framework")
    print("• Foundation for modern controllable generation")
    
    print(f"\nCLIP PRINCIPLES:")
    for key, principle in CLIP_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nARCHITECTURAL INNOVATIONS:")
    for innovation in clip_analysis['architectural_innovations']:
        print(f"  • {innovation}")
    
    print(f"\nTRAINING METHODOLOGY:")
    for method in clip_analysis['training_methodology']:
        print(f"  • {method}")
    
    print(f"\nGUIDANCE CAPABILITIES:")
    for capability in clip_analysis['guidance_capabilities']:
        print(f"  • {capability}")
    
    print(f"\nCONTRASTIVE LEARNING FRAMEWORK:")
    print("="*40)
    print("• Positive pairs: Matching image-text pairs")
    print("• Negative pairs: Non-matching image-text pairs")
    print("• Symmetric loss: Both image→text and text→image")
    print("• Temperature scaling: Control similarity distribution")
    print("• Large batch sizes: Many negatives per positive")
    
    print(f"\nVISION TRANSFORMER ENCODER:")
    print("="*40)
    print("• Input: 224×224 RGB images")
    print("• Patch size: 16×16 (196 patches)")
    print("• Architecture: ViT-B/16 or ViT-L/14")
    print("• Output: Global image representation")
    print("• Layer normalization and GELU activation")
    
    print(f"\nTEXT TRANSFORMER ENCODER:")
    print("="*40)
    print("• Input: Tokenized text (max 77 tokens)")
    print("• Architecture: Transformer encoder")
    print("• Vocabulary: 49,152 BPE tokens")
    print("• Output: Text representation from [EOS] token")
    print("• Causal masking for autoregressive training")
    
    print(f"\nZERO-SHOT CLASSIFICATION:")
    print("="*40)
    print("• No task-specific training data needed")
    print("• Text templates: 'A photo of a [class]'")
    print("• Compute similarity with all class descriptions")
    print("• Softmax over similarities for prediction")
    print("• Robust to distribution shift and domain changes")
    
    print(f"\nGENERATION GUIDANCE PROCESS:")
    print("="*40)
    print("• Start with random latent code")
    print("• Generate image from latent code")
    print("• Compute CLIP similarity with target text")
    print("• Backpropagate through CLIP to update latent")
    print("• Iterate until satisfactory alignment")
    
    print(f"\nGUIDANCE APPLICATIONS:")
    print("="*40)
    print("• Text-to-image generation (DALL-E 2)")
    print("• Image editing and manipulation")
    print("• Style transfer and artistic control")
    print("• Video generation and editing")
    print("• 3D generation and modeling")
    
    print(f"\nCOMPARISON TO SUPERVISED LEARNING:")
    print("="*40)
    print("• Supervised: Task-specific labels and training")
    print("• CLIP: Natural language supervision")
    print("• Supervised: Limited to predefined classes")
    print("• CLIP: Open vocabulary classification")
    print("• Supervised: Poor generalization across domains")
    print("• CLIP: Robust cross-domain transfer")
    
    print(f"\nSCALING INSIGHTS:")
    print("="*40)
    print("• Performance scales with dataset size")
    print("• 400M pairs → 76% ImageNet zero-shot")
    print("• Model size scaling also improves performance")
    print("• Data quality matters more than quantity")
    print("• Diverse internet data enables generalization")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Enabled practical zero-shot image classification")
    print("• Foundation for controllable generation systems")
    print("• Inspired numerous vision-language models")
    print("• Standard component in modern generative AI")
    print("• Democratized computer vision through natural language")
    print("• Bridge between computer vision and NLP")
    
    print(f"\nLIMITATIONS AND CHALLENGES:")
    print("="*40)
    print("• Computational requirements for large-scale training")
    print("• Bias from internet training data")
    print("• Performance gap with specialized supervised models")
    print("• Limited fine-grained understanding")
    print("• Requires careful prompt engineering")
    
    print(f"\nFOLLOW-UP INNOVATIONS:")
    print("="*40)
    print("• DALL-E 2: CLIP-guided diffusion generation")
    print("• GLIDE: Classifier-free guidance with CLIP")
    print("• Stable Diffusion: Open-source CLIP guidance")
    print("• FLAMINGO: Few-shot learning with CLIP")
    print("• GPT-4V: Multimodal understanding")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• Standard component in text-to-image models")
    print("• Foundation for multimodal foundation models")
    print("• Enables natural language control of AI systems")
    print("• Used in image search and recommendation")
    print("• Basis for content moderation and safety")
    
    return {
        'model': 'CLIP-Guided Generation',
        'year': YEAR,
        'innovation': INNOVATION,
        'clip_params': clip_params,
        'generator_params': generator_params,
        'total_params': clip_params + generator_params,
        'clip_analysis': clip_analysis
    }

if __name__ == "__main__":
    results = main()