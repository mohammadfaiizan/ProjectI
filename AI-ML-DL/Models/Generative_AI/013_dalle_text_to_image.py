"""
ERA 5: LARGE-SCALE TEXT-TO-IMAGE - DALL-E Text-to-Image
=======================================================

Year: 2021
Paper: "Zero-Shot Text-to-Image Generation" (Ramesh et al., OpenAI)
Innovation: Transformer-based text-to-image generation with discrete VAE tokenization
Previous Limitation: Limited text conditioning and poor text-image alignment
Performance Gain: High-quality text-to-image generation with strong semantic understanding
Impact: Demonstrated feasibility of large-scale text-to-image and launched consumer AI art era

This file implements DALL-E that revolutionized text-to-image generation through transformer
architectures and established the foundation for modern text-to-image systems.
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
INNOVATION = "Transformer-based text-to-image generation with discrete VAE tokenization"
PREVIOUS_LIMITATION = "Limited text conditioning and poor text-image alignment"
IMPACT = "Demonstrated feasibility of large-scale text-to-image and launched consumer AI art era"

print(f"=== DALL-E Text-to-Image ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# DALL-E PRINCIPLES
# ============================================================================

DALLE_PRINCIPLES = {
    "discrete_tokenization": "Convert images to discrete tokens using VQ-VAE for transformer processing",
    "autoregressive_generation": "Generate image tokens autoregressively conditioned on text",
    "transformer_architecture": "Use large transformer decoder for text-to-image modeling",
    "bpe_text_encoding": "Encode text using Byte-Pair Encoding (BPE) for subword tokenization",
    "joint_text_image_training": "Train on large-scale text-image pairs for semantic alignment",
    "zero_shot_generation": "Generate novel images from text without specific training examples",
    "compositional_understanding": "Combine concepts in novel ways through transformer attention",
    "large_scale_training": "12B parameter model trained on 250M text-image pairs"
}

print("DALL-E Principles:")
for key, principle in DALLE_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10 + Synthetic Text)
# ============================================================================

def load_cifar10_with_text():
    """Load CIFAR-10 dataset with synthetic text descriptions for DALL-E study"""
    print("Loading CIFAR-10 dataset with synthetic text for DALL-E study...")
    print("Note: DALL-E uses large-scale internet text-image pairs")
    
    # DALL-E preprocessing
    transform_train = transforms.Compose([
        transforms.Resize(64),  # Smaller for demonstration
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # CIFAR-10 class names and synthetic descriptions
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Synthetic text descriptions (simplified for demonstration)
    text_templates = {
        0: ["a photo of an airplane", "an airplane flying in the sky", "a commercial aircraft"],
        1: ["a photo of a car", "an automobile on the road", "a red car"],
        2: ["a photo of a bird", "a colorful bird", "a small bird flying"],
        3: ["a photo of a cat", "a cute cat", "a cat sitting"],
        4: ["a photo of a deer", "a deer in the forest", "a brown deer"],
        5: ["a photo of a dog", "a friendly dog", "a dog playing"],
        6: ["a photo of a frog", "a green frog", "a frog on a lily pad"],
        7: ["a photo of a horse", "a brown horse", "a horse running"],
        8: ["a photo of a ship", "a ship on the ocean", "a large ship"],
        9: ["a photo of a truck", "a big truck", "a delivery truck"]
    }
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(class_names)}")
    print(f"Image size: 64x64 RGB (DALL-E uses 256x256)")
    print(f"Focus: Transformer-based text-to-image generation")
    
    return train_loader, test_loader, class_names, text_templates

# ============================================================================
# DISCRETE VAE (VQ-VAE) FOR IMAGE TOKENIZATION
# ============================================================================

class VectorQuantization(nn.Module):
    """
    Vector Quantization layer for VQ-VAE
    
    Converts continuous representations to discrete tokens
    for transformer processing
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantization, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        print(f"    VectorQuantization: {num_embeddings} codes, {embedding_dim}D")
    
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1])

class VQVAEEncoder(nn.Module):
    """VQVAE Encoder for image tokenization"""
    
    def __init__(self, in_channels=3, hidden_channels=128, embedding_dim=64):
        super(VQVAEEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embedding_dim, 3, stride=1, padding=1)
        )
        
        print(f"  VQVAE Encoder: {in_channels} -> {embedding_dim}")
    
    def forward(self, x):
        return self.conv_layers(x)

class VQVAEDecoder(nn.Module):
    """VQVAE Decoder for image reconstruction"""
    
    def __init__(self, embedding_dim=64, hidden_channels=128, out_channels=3):
        super(VQVAEDecoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 4, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        print(f"  VQVAE Decoder: {embedding_dim} -> {out_channels}")
    
    def forward(self, x):
        return self.conv_layers(x)

class VQVAE(nn.Module):
    """
    Vector Quantized VAE for image tokenization
    
    Converts images to discrete tokens for transformer processing
    """
    
    def __init__(self, num_embeddings=1024, embedding_dim=64):
        super(VQVAE, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        print(f"Building VQ-VAE...")
        
        self.encoder = VQVAEEncoder(embedding_dim=embedding_dim)
        self.vq_layer = VectorQuantization(num_embeddings, embedding_dim)
        self.decoder = VQVAEDecoder(embedding_dim=embedding_dim)
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"VQ-VAE Summary:")
        print(f"  Embeddings: {num_embeddings}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Total parameters: {total_params:,}")
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, perplexity, indices
    
    def encode(self, x):
        """Encode images to discrete tokens"""
        z_e = self.encoder(x)
        _, _, _, indices = self.vq_layer(z_e)
        return indices
    
    def decode_tokens(self, indices):
        """Decode discrete tokens to images"""
        batch_size, height, width = indices.shape
        
        # Convert indices to embeddings
        embeddings = self.vq_layer.embedding(indices)
        embeddings = embeddings.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Decode
        return self.decoder(embeddings)

# ============================================================================
# TEXT TOKENIZATION
# ============================================================================

class SimpleTextTokenizer:
    """
    Simplified text tokenizer for DALL-E demonstration
    
    In practice, DALL-E uses BPE (Byte-Pair Encoding)
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        
        # Simple vocabulary (in practice, learned from data)
        self.vocab = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3,
            'a': 4, 'photo': 5, 'of': 6, 'an': 7, 'the': 8,
            'airplane': 10, 'automobile': 11, 'bird': 12, 'cat': 13, 'deer': 14,
            'dog': 15, 'frog': 16, 'horse': 17, 'ship': 18, 'truck': 19,
            'car': 20, 'flying': 21, 'sky': 22, 'commercial': 23, 'aircraft': 24,
            'road': 25, 'red': 26, 'colorful': 27, 'small': 28, 'cute': 29,
            'sitting': 30, 'forest': 31, 'brown': 32, 'friendly': 33, 'playing': 34,
            'green': 35, 'lily': 36, 'pad': 37, 'running': 38, 'ocean': 39,
            'large': 40, 'big': 41, 'delivery': 42, 'in': 43, 'on': 44, 'is': 45
        }
        
        # Reverse mapping
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"  Text Tokenizer: {len(self.vocab)} vocab size")
    
    def encode(self, text):
        """Encode text to token indices"""
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return [self.vocab['<BOS>']] + indices + [self.vocab['<EOS>']]
    
    def decode(self, indices):
        """Decode token indices to text"""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        return ' '.join(tokens).replace('<BOS>', '').replace('<EOS>', '').strip()
    
    def pad_sequence(self, sequences, max_length=32):
        """Pad sequences to max length"""
        padded = []
        for seq in sequences:
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq + [self.vocab['<PAD>']] * (max_length - len(seq))
            padded.append(seq)
        return torch.tensor(padded)

# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention for DALL-E transformer"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        print(f"    MultiHeadAttention: {embed_dim}D, {num_heads} heads")
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    """Transformer block for DALL-E"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        print(f"    TransformerBlock: {embed_dim}D, FF {ff_dim}D")
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

# ============================================================================
# DALL-E TRANSFORMER
# ============================================================================

class DALLETransformer(nn.Module):
    """
    DALL-E Transformer for autoregressive text-to-image generation
    
    Generates image tokens autoregressively conditioned on text
    """
    
    def __init__(self, text_vocab_size=1000, image_vocab_size=1024, 
                 embed_dim=512, num_layers=12, num_heads=8, 
                 max_text_len=32, max_image_len=64):
        super(DALLETransformer, self).__init__()
        
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.embed_dim = embed_dim
        self.max_text_len = max_text_len
        self.max_image_len = max_image_len
        self.total_vocab_size = text_vocab_size + image_vocab_size
        
        print(f"Building DALL-E Transformer...")
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.total_vocab_size, embed_dim)
        
        # Position embeddings
        max_seq_len = max_text_len + max_image_len
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, image_vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"DALL-E Transformer Summary:")
        print(f"  Text vocab: {text_vocab_size}")
        print(f"  Image vocab: {image_vocab_size}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Total parameters: {total_params:,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, text_tokens, image_tokens=None, training=True):
        """
        Forward pass for DALL-E transformer
        
        Args:
            text_tokens: Text token indices (batch_size, text_len)
            image_tokens: Image token indices (batch_size, image_len)
            training: Whether in training mode
        
        Returns:
            Logits for next image token prediction
        """
        device = text_tokens.device
        batch_size, text_len = text_tokens.shape
        
        if training and image_tokens is not None:
            # Training: concatenate text and image tokens
            # Offset image tokens by text vocab size
            image_tokens_offset = image_tokens + self.text_vocab_size
            tokens = torch.cat([text_tokens, image_tokens_offset], dim=1)
            seq_len = text_len + image_tokens.shape[1]
        else:
            # Inference: only text tokens
            tokens = text_tokens
            seq_len = text_len
        
        # Token embeddings
        token_embeds = self.token_embedding(tokens)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(device)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        if training:
            # Return logits for image tokens only
            return logits[:, text_len-1:-1]  # Shift for next token prediction
        else:
            return logits
    
    @torch.no_grad()
    def generate(self, text_tokens, temperature=1.0, top_k=50, max_image_len=64):
        """
        Generate image tokens autoregressively
        
        Args:
            text_tokens: Text conditioning tokens
            temperature: Sampling temperature
            top_k: Top-k sampling
            max_image_len: Maximum image sequence length
        
        Returns:
            Generated image tokens
        """
        self.eval()
        device = text_tokens.device
        batch_size = text_tokens.shape[0]
        
        # Start with text tokens
        generated_tokens = text_tokens.clone()
        
        for _ in range(max_image_len):
            # Forward pass
            logits = self.forward(text_tokens, 
                                generated_tokens[:, text_tokens.shape[1]:], 
                                training=False)
            
            # Get logits for next token
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Offset by text vocab size and append
            next_token_offset = next_token + self.text_vocab_size
            generated_tokens = torch.cat([generated_tokens, next_token_offset], dim=1)
        
        # Return only image tokens
        return generated_tokens[:, text_tokens.shape[1]:] - self.text_vocab_size

# ============================================================================
# DALL-E MODEL
# ============================================================================

class DALLE_TextToImage(nn.Module):
    """
    DALL-E Text-to-Image Model
    
    Revolutionary Innovations:
    - Transformer-based autoregressive image generation
    - Discrete VAE tokenization for image-text joint modeling
    - Large-scale training on text-image pairs
    - Zero-shot compositional generation
    - Foundation for modern text-to-image systems
    """
    
    def __init__(self, vqvae_embeddings=1024, transformer_layers=12):
        super(DALLE_TextToImage, self).__init__()
        
        print(f"Building DALL-E Text-to-Image Model...")
        
        # VQ-VAE for image tokenization
        self.vqvae = VQVAE(num_embeddings=vqvae_embeddings)
        
        # Text tokenizer
        self.text_tokenizer = SimpleTextTokenizer()
        
        # DALL-E Transformer
        self.transformer = DALLETransformer(
            text_vocab_size=self.text_tokenizer.vocab_size,
            image_vocab_size=vqvae_embeddings,
            embed_dim=512,
            num_layers=transformer_layers,
            num_heads=8,
            max_text_len=32,
            max_image_len=64  # 8x8 image tokens
        )
        
        # Calculate statistics
        vqvae_params = sum(p.numel() for p in self.vqvae.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        total_params = vqvae_params + transformer_params
        
        print(f"DALL-E Model Summary:")
        print(f"  VQ-VAE parameters: {vqvae_params:,}")
        print(f"  Transformer parameters: {transformer_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Transformer-based text-to-image generation")
    
    def encode_text(self, text_list):
        """Encode text to tokens"""
        encoded = [self.text_tokenizer.encode(text) for text in text_list]
        return self.text_tokenizer.pad_sequence(encoded)
    
    def forward(self, images, text_list):
        """
        Training forward pass
        
        Args:
            images: Input images
            text_list: List of text descriptions
        
        Returns:
            Combined loss (VQ-VAE + Transformer)
        """
        device = images.device
        
        # VQ-VAE forward pass
        recon_images, vq_loss, perplexity, image_tokens = self.vqvae(images)
        
        # VQ-VAE reconstruction loss
        recon_loss = F.mse_loss(recon_images, images)
        
        # Encode text
        text_tokens = self.encode_text(text_list).to(device)
        
        # Flatten image tokens for transformer
        batch_size, h, w = image_tokens.shape
        image_tokens_flat = image_tokens.view(batch_size, h * w)
        
        # Transformer forward pass
        logits = self.transformer(text_tokens, image_tokens_flat)
        
        # Transformer loss (next token prediction)
        transformer_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            image_tokens_flat[:, 1:].reshape(-1)  # Shift for next token prediction
        )
        
        # Combined loss
        total_loss = recon_loss + vq_loss + transformer_loss
        
        return total_loss, recon_loss, vq_loss, transformer_loss, perplexity
    
    @torch.no_grad()
    def generate(self, text_list, temperature=1.0, top_k=50):
        """
        Generate images from text
        
        Args:
            text_list: List of text descriptions
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            Generated images
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Encode text
        text_tokens = self.encode_text(text_list).to(device)
        
        # Generate image tokens
        image_tokens = self.transformer.generate(
            text_tokens, temperature=temperature, top_k=top_k
        )
        
        # Reshape to spatial dimensions
        batch_size = image_tokens.shape[0]
        image_tokens_spatial = image_tokens.view(batch_size, 8, 8)
        
        # Decode to images
        generated_images = self.vqvae.decode_tokens(image_tokens_spatial)
        
        return generated_images
    
    def get_dalle_analysis(self):
        """Analyze DALL-E innovations"""
        return {
            'dalle_principles': DALLE_PRINCIPLES,
            'architectural_innovations': [
                'Discrete VAE tokenization for transformer compatibility',
                'Autoregressive image generation with text conditioning',
                'Large-scale transformer architecture (12B parameters)',
                'Joint text-image vocabulary and training',
                'Zero-shot compositional understanding'
            ],
            'training_methodology': [
                'Large-scale internet text-image pairs (250M)',
                'Two-stage training: VQ-VAE then Transformer',
                'Autoregressive next-token prediction objective',
                'BPE text encoding for subword tokenization',
                'Careful data curation and filtering'
            ],
            'generation_capabilities': [
                'Zero-shot text-to-image generation',
                'Compositional understanding and novel combinations',
                'Style transfer and artistic control',
                'Object manipulation and scene composition',
                'Abstract concept visualization'
            ]
        }

# ============================================================================
# DALL-E TRAINING FUNCTION
# ============================================================================

def train_dalle(model, train_loader, text_templates, epochs=100, learning_rate=3e-4):
    """Train DALL-E model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training tracking
    losses = {
        'total': [], 'recon': [], 'vq': [], 'transformer': []
    }
    
    print(f"Training DALL-E on device: {device}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0.0, 'recon': 0.0, 'vq': 0.0, 'transformer': 0.0}
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            
            # Get text descriptions for this batch
            batch_texts = []
            for label in labels:
                # Randomly sample from available descriptions
                text_options = text_templates[label.item()]
                text = np.random.choice(text_options)
                batch_texts.append(text)
            
            optimizer.zero_grad()
            
            # Forward pass
            total_loss, recon_loss, vq_loss, transformer_loss, perplexity = model(images, batch_texts)
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['vq'] += vq_loss.item()
            epoch_losses['transformer'] += transformer_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Total: {total_loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'VQ: {vq_loss.item():.4f}, '
                      f'Trans: {transformer_loss.item():.4f}, '
                      f'Perplexity: {perplexity.item():.1f}')
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch averages
        num_batches = len(train_loader)
        for key in epoch_losses:
            avg_loss = epoch_losses[key] / num_batches
            losses[key].append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Total: {losses["total"][-1]:.4f}, '
              f'Recon: {losses["recon"][-1]:.4f}, '
              f'VQ: {losses["vq"][-1]:.4f}, '
              f'Trans: {losses["transformer"][-1]:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'losses': losses
            }, f'AI-ML-DL/Models/Generative_AI/dalle_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if losses['total'][-1] < 0.5:
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_dalle_architecture():
    """Visualize DALL-E architecture and process"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # DALL-E pipeline overview
    ax = axes[0, 0]
    ax.set_title('DALL-E Pipeline Overview', fontsize=14, fontweight='bold')
    
    # Pipeline stages
    stages = [
        ('Text\nInput', 'lightgreen', 0.05),
        ('BPE\nTokenizer', 'lightblue', 0.2),
        ('Text\nTokens', 'lightcyan', 0.35),
        ('Transformer\nDecoder', 'yellow', 0.5),
        ('Image\nTokens', 'orange', 0.65),
        ('VQ-VAE\nDecoder', 'lightpink', 0.8),
        ('Generated\nImage', 'lightgreen', 0.95)
    ]
    
    for i, (stage, color, x_pos) in enumerate(stages):
        if i < len(stages) - 1:
            width = 0.12
        else:
            width = 0.1
        
        rect = plt.Rectangle((x_pos - width/2, 0.6), width, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos, 0.75, stage, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Arrows between stages
        if i < len(stages) - 1:
            next_x = stages[i+1][2]
            ax.annotate('', xy=(next_x - 0.06, 0.75), xytext=(x_pos + width/2, 0.75),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Process details
    ax.text(0.2, 0.4, 'BPE encoding\nto subwords', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    
    ax.text(0.5, 0.3, 'Autoregressive\ngeneration', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.text(0.8, 0.4, 'Discrete tokens\nto pixels', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightpink', alpha=0.7))
    
    ax.text(0.5, 0.1, '"A red car on the highway" → Generated image', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # VQ-VAE tokenization process
    ax = axes[0, 1]
    ax.set_title('VQ-VAE Image Tokenization', fontsize=14)
    
    # Show tokenization process
    tokenization_stages = [
        ('64×64×3\nImage', 'lightgreen', 0.15, 0.8),
        ('8×8×64\nFeatures', 'lightblue', 0.15, 0.5),
        ('8×8\nTokens', 'orange', 0.15, 0.2),
        ('Codebook\n1024 codes', 'lightcoral', 0.6, 0.65),
        ('8×8×64\nQuantized', 'lightblue', 0.85, 0.5),
        ('64×64×3\nReconstructed', 'lightgreen', 0.85, 0.2)
    ]
    
    for stage, color, x, y in tokenization_stages:
        rect = plt.Rectangle((x-0.08, y-0.1), 0.16, 0.2, 
                           facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, stage, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        ((0.23, 0.8), (0.23, 0.6)),  # Encode
        ((0.23, 0.4), (0.23, 0.3)),  # Tokenize
        ((0.31, 0.2), (0.52, 0.6)),  # To codebook
        ((0.68, 0.65), (0.77, 0.55)),  # From codebook
        ((0.85, 0.4), (0.85, 0.3))   # Decode
    ]
    
    for (start_x, start_y), (end_x, end_y) in arrows:
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Labels
    ax.text(0.05, 0.65, 'Encoder', ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)
    ax.text(0.95, 0.35, 'Decoder', ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Transformer autoregressive generation
    ax = axes[1, 0]
    ax.set_title('Autoregressive Image Generation', fontsize=14)
    
    # Show token generation sequence
    sequence_length = 8
    x_positions = np.linspace(0.1, 0.9, sequence_length)
    
    # Text tokens (given)
    for i in range(3):
        rect = plt.Rectangle((x_positions[i]-0.03, 0.7), 0.06, 0.2, 
                           facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect)
        ax.text(x_positions[i], 0.8, f'T{i+1}', ha='center', va='center', fontweight='bold')
    
    # Image tokens (generated)
    colors = ['lightgreen', 'yellow', 'orange', 'lightcoral', 'lightpink']
    for i in range(5):
        alpha = 1.0 if i < 2 else 0.5  # Show already generated vs future
        rect = plt.Rectangle((x_positions[i+3]-0.03, 0.7), 0.06, 0.2, 
                           facecolor=colors[i], edgecolor='black', alpha=alpha)
        ax.add_patch(rect)
        ax.text(x_positions[i+3], 0.8, f'I{i+1}', ha='center', va='center', fontweight='bold')
        
        if i >= 2:  # Future tokens
            ax.text(x_positions[i+3], 0.6, '?', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='red')
    
    # Current prediction
    ax.annotate('Next token\nprediction', xy=(x_positions[5], 0.9), xytext=(x_positions[5], 1.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    # Attention mechanism
    for i in range(5):
        for j in range(i+4):
            alpha = 0.3 if i >= 2 else 0.7
            ax.annotate('', xy=(x_positions[j], 0.65), xytext=(x_positions[i+3], 0.65),
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=alpha))
    
    ax.text(0.5, 0.4, 'Causal Attention: Each token attends to previous tokens', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.text(0.5, 0.1, 'Autoregressive Generation: P(I₁,I₂,...|T₁,T₂,T₃) = ∏P(Iᵢ|I₁...Iᵢ₋₁,T)', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.axis('off')
    
    # DALL-E vs traditional approaches
    ax = axes[1, 1]
    ax.set_title('DALL-E vs Traditional Approaches', fontsize=14)
    
    approaches = ['GAN +\nText', 'VAE +\nText', 'DALL-E\nTransformer']
    metrics = ['Quality', 'Diversity', 'Text\nAlignment', 'Training\nStability']
    
    # Scores (relative)
    scores = np.array([
        [7, 5, 4, 3],  # GAN + Text
        [5, 7, 5, 8],  # VAE + Text
        [8, 8, 9, 7]   # DALL-E
    ])
    
    # Create bar chart
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    for i, (approach, color) in enumerate(zip(approaches, colors)):
        bars = ax.bar(x + i*width, scores[i], width, label=approach, 
                     color=color, alpha=0.8)
        
        # Add value labels
        for bar, score in zip(bars, scores[i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{score}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Performance Score')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight DALL-E advantages
    ax.text(len(metrics)/2, 9.5, 'DALL-E Advantages:\n• Strong text alignment\n• Compositional understanding\n• Zero-shot generation', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/013_dalle_architecture.png', dpi=300, bbox_inches='tight')
    print("DALL-E architecture visualization saved: 013_dalle_architecture.png")

def visualize_dalle_innovations():
    """Visualize DALL-E key innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Discrete tokenization concept
    ax = axes[0, 0]
    ax.set_title('Discrete Tokenization for Images', fontsize=14, fontweight='bold')
    
    # Show continuous vs discrete representation
    # Continuous (traditional)
    ax.text(0.25, 0.9, 'Traditional Approach', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Continuous space visualization
    x_cont = np.linspace(0.1, 0.4, 100)
    y_cont = 0.7 + 0.1 * np.sin(10 * x_cont)
    ax.plot(x_cont, y_cont, 'b-', linewidth=3, alpha=0.7)
    ax.fill_between(x_cont, y_cont-0.02, y_cont+0.02, alpha=0.3, color='blue')
    ax.text(0.25, 0.55, 'Continuous\nPixel Values', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    # DALL-E approach
    ax.text(0.75, 0.9, 'DALL-E Approach', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Discrete tokens
    x_discrete = np.linspace(0.6, 0.9, 8)
    y_discrete = [0.7] * 8
    colors = plt.cm.Set3(np.arange(8))
    
    for x, y, color in zip(x_discrete, y_discrete, colors):
        circle = plt.Circle((x, y), 0.02, facecolor=color, edgecolor='black')
        ax.add_patch(circle)
    
    ax.text(0.75, 0.55, 'Discrete\nImage Tokens', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    # Advantages
    ax.text(0.5, 0.3, 'Discrete Advantages:\n• Transformer compatibility\n• Autoregressive modeling\n• Stable training\n• Language-like processing', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Autoregressive vs parallel generation
    ax = axes[0, 1]
    ax.set_title('Autoregressive vs Parallel Generation', fontsize=14)
    
    # Autoregressive (DALL-E)
    ax.text(0.25, 0.9, 'Autoregressive (DALL-E)', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='blue')
    
    # Show sequential generation
    steps = 6
    step_positions = np.linspace(0.05, 0.45, steps)
    
    for i, x_pos in enumerate(step_positions):
        alpha = 1.0 if i < 3 else 0.3
        color = 'lightgreen' if i < 3 else 'lightgray'
        rect = plt.Rectangle((x_pos, 0.65), 0.05, 0.1, 
                           facecolor=color, edgecolor='black', alpha=alpha)
        ax.add_patch(rect)
        ax.text(x_pos + 0.025, 0.7, f't{i+1}', ha='center', va='center', fontsize=8)
        
        if i > 0:
            # Arrow from previous
            ax.annotate('', xy=(x_pos, 0.7), xytext=(step_positions[i-1] + 0.05, 0.7),
                       arrowprops=dict(arrowstyle='->', lw=1, color='blue'))
    
    ax.text(0.25, 0.55, 'Sequential\nO(n) steps', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    # Parallel (traditional)
    ax.text(0.75, 0.9, 'Parallel (Traditional)', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='red')
    
    # Show parallel generation
    parallel_positions = np.linspace(0.55, 0.95, steps)
    
    for i, x_pos in enumerate(parallel_positions):
        rect = plt.Rectangle((x_pos, 0.65), 0.05, 0.1, 
                           facecolor='orange', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos + 0.025, 0.7, f't{i+1}', ha='center', va='center', fontsize=8)
    
    # Central process
    central_rect = plt.Rectangle((0.7, 0.45), 0.1, 0.1, 
                               facecolor='yellow', edgecolor='black')
    ax.add_patch(central_rect)
    ax.text(0.75, 0.5, 'Model', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows to all positions
    for x_pos in parallel_positions:
        ax.annotate('', xy=(x_pos + 0.025, 0.65), xytext=(0.75, 0.55),
                   arrowprops=dict(arrowstyle='->', lw=1, color='red'))
    
    ax.text(0.75, 0.35, 'Parallel\nO(1) steps', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    
    # Trade-offs
    ax.text(0.5, 0.15, 'Trade-offs:\nAutoregressive: Slower but better quality and control\nParallel: Faster but limited expressiveness', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Training data scale
    ax = axes[1, 0]
    ax.set_title('Training Data Scale Evolution', fontsize=14)
    
    # Show evolution of dataset sizes
    models = ['Early\nGANs', 'StyleGAN', 'DALL-E', 'Future\nModels']
    dataset_sizes = [1e4, 1e5, 2.5e8, 1e12]  # Approximate sizes
    param_counts = [1e6, 1e8, 1.2e10, 1e12]  # Approximate parameter counts
    
    # Log scale
    log_dataset = np.log10(dataset_sizes)
    log_params = np.log10(param_counts)
    
    # Create scatter plot
    colors = ['red', 'blue', 'green', 'purple']
    sizes = [50, 100, 200, 300]
    
    for i, (model, color, size) in enumerate(zip(models, colors, sizes)):
        ax.scatter(log_dataset[i], log_params[i], c=color, s=size, alpha=0.7, edgecolors='black')
        ax.text(log_dataset[i], log_params[i] + 0.3, model, ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # Trend line
    ax.plot(log_dataset, log_params, 'k--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Log₁₀(Dataset Size)')
    ax.set_ylabel('Log₁₀(Parameters)')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(6, 11, 'DALL-E:\n250M images\n12B parameters', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.text(4, 8, 'Scaling Law:\nLarger datasets +\nLarger models =\nBetter performance', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Zero-shot capabilities
    ax = axes[1, 1]
    ax.set_title('Zero-Shot Generation Capabilities', fontsize=14)
    
    # Create capability radar chart
    capabilities = ['Object\nRecognition', 'Style\nTransfer', 'Scene\nComposition', 
                   'Abstract\nConcepts', 'Novel\nCombinations', 'Artistic\nStyles']
    
    # Scores for different approaches
    traditional_scores = [6, 4, 3, 2, 2, 3]
    dalle_scores = [8, 7, 8, 6, 9, 7]
    
    # Convert to radar chart
    angles = np.linspace(0, 2*np.pi, len(capabilities), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    traditional_scores = np.concatenate((traditional_scores, [traditional_scores[0]]))
    dalle_scores = np.concatenate((dalle_scores, [dalle_scores[0]]))
    
    # Create radar plot
    ax.plot(angles, traditional_scores, 'ro-', linewidth=2, label='Traditional Approaches', alpha=0.7)
    ax.fill(angles, traditional_scores, alpha=0.25, color='red')
    
    ax.plot(angles, dalle_scores, 'go-', linewidth=2, label='DALL-E', alpha=0.7)
    ax.fill(angles, dalle_scores, alpha=0.25, color='green')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/013_dalle_innovations.png', dpi=300, bbox_inches='tight')
    print("DALL-E innovations visualization saved: 013_dalle_innovations.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== DALL-E Text-to-Image Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset with text
    train_loader, test_loader, class_names, text_templates = load_cifar10_with_text()
    
    # Initialize DALL-E model
    dalle_model = DALLE_TextToImage(vqvae_embeddings=1024, transformer_layers=8)
    
    # Analyze model properties
    total_params = sum(p.numel() for p in dalle_model.parameters())
    dalle_analysis = dalle_model.get_dalle_analysis()
    
    print(f"\nDALL-E Model Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  VQ-VAE embeddings: 1024")
    print(f"  Transformer layers: 8")
    
    print(f"\nDALL-E Innovations:")
    for key, value in dalle_analysis.items():
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
    print("\nGenerating DALL-E analysis...")
    visualize_dalle_architecture()
    visualize_dalle_innovations()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("DALL-E TEXT-TO-IMAGE SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nDALL-E REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. DISCRETE IMAGE TOKENIZATION:")
    print("   • VQ-VAE converts images to discrete tokens")
    print("   • 256×256 image → 32×32 tokens (1024 vocabulary)")
    print("   • Enables transformer-style autoregressive modeling")
    print("   • Bridging computer vision and natural language processing")
    
    print("\n2. TRANSFORMER-BASED GENERATION:")
    print("   • Autoregressive image generation: P(image|text) = ∏P(token_i|previous_tokens, text)")
    print("   • Large-scale transformer decoder (12B parameters)")
    print("   • Causal attention for sequential generation")
    print("   • Joint text-image vocabulary and processing")
    
    print("\n3. LARGE-SCALE TRAINING:")
    print("   • 250 million text-image pairs from internet")
    print("   • Careful data curation and filtering")
    print("   • Two-stage training: VQ-VAE then Transformer")
    print("   • Massive computational requirements (OpenAI scale)")
    
    print("\n4. ZERO-SHOT CAPABILITIES:")
    print("   • Generate novel images from text without specific training")
    print("   • Compositional understanding: 'a red cube on a blue sphere'")
    print("   • Style transfer: 'in the style of Van Gogh'")
    print("   • Abstract concept visualization")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First large-scale text-to-image transformer")
    print("• Demonstrated feasibility of autoregressive image generation")
    print("• Strong semantic understanding and text-image alignment")
    print("• Compositional generation capabilities")
    print("• Foundation for modern text-to-image systems")
    
    print(f"\nDALL-E PRINCIPLES:")
    for key, principle in DALLE_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nARCHITECTURAL INNOVATIONS:")
    for innovation in dalle_analysis['architectural_innovations']:
        print(f"  • {innovation}")
    
    print(f"\nTRAINING METHODOLOGY:")
    for method in dalle_analysis['training_methodology']:
        print(f"  • {method}")
    
    print(f"\nGENERATION CAPABILITIES:")
    for capability in dalle_analysis['generation_capabilities']:
        print(f"  • {capability}")
    
    print(f"\nVQ-VAE IMAGE TOKENIZATION:")
    print("="*40)
    print("• Encoder: 256×256×3 → 32×32×256 features")
    print("• Vector Quantization: Features → Discrete tokens (1024 vocab)")
    print("• Decoder: Discrete tokens → 256×256×3 reconstructed image")
    print("• Two-stage training: Reconstruct images, then generate tokens")
    print("• Enables transformer processing of visual information")
    
    print(f"\nTRANSFORMER ARCHITECTURE:")
    print("="*40)
    print("• Decoder-only transformer (like GPT)")
    print("• 12 billion parameters (much larger than GPT-3)")
    print("• Text tokens + Image tokens in joint vocabulary")
    print("• Causal attention: each token attends to previous tokens")
    print("• Autoregressive generation of image tokens")
    
    print(f"\nTRAINING PROCESS:")
    print("="*40)
    print("• Stage 1: Train VQ-VAE on images for tokenization")
    print("• Stage 2: Train transformer on text-image token pairs")
    print("• Loss: Next token prediction (cross-entropy)")
    print("• Data: 250M text-image pairs from internet")
    print("• Compute: Massive scale training (expensive)")
    
    print(f"\nGENERATION PROCESS:")
    print("="*40)
    print("• Input text → BPE tokens")
    print("• Transformer generates image tokens autoregressively")
    print("• 32×32 = 1024 tokens generated sequentially")
    print("• VQ-VAE decoder converts tokens to pixels")
    print("• Optional: CLIP reranking for best samples")
    
    print(f"\nKEY TECHNICAL DETAILS:")
    print("="*40)
    print("• BPE text encoding: Subword tokenization")
    print("• Temperature sampling: Control generation randomness")
    print("• Top-k sampling: Limit to most likely tokens")
    print("• Attention masking: Prevent future token access")
    print("• Mixed precision training: FP16 for efficiency")
    
    print(f"\nCOMPOSITIONAL UNDERSTANDING:")
    print("="*40)
    print("• Object combinations: 'a red car and blue truck'")
    print("• Spatial relationships: 'a cat sitting on a chair'")
    print("• Style specifications: 'a photo vs a painting'")
    print("• Abstract concepts: 'the concept of democracy'")
    print("• Novel combinations: Never-seen object arrangements")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Demonstrated viability of text-to-image generation")
    print("• Launched consumer interest in AI art")
    print("• Inspired follow-up work (DALL-E 2, Midjourney, etc.)")
    print("• Established transformer approach for multimodal AI")
    print("• Showed importance of scale in generative modeling")
    print("• Created new field of AI-assisted creativity")
    
    print(f"\nLIMITATIONS AND CHALLENGES:")
    print("="*40)
    print("• Autoregressive generation: Slow (1024 sequential steps)")
    print("• Computational requirements: Massive training cost")
    print("• Limited resolution: 256×256 pixels")
    print("• Occasional semantic errors and artifacts")
    print("• Training data bias and safety concerns")
    
    print(f"\nCOMPARISON TO PREVIOUS WORK:")
    print("="*40)
    print("• GANs + Text: Better quality but limited text understanding")
    print("• DALL-E: Superior text understanding and composition")
    print("• VAEs + Text: Good diversity but lower quality")
    print("• DALL-E: Better quality through large-scale training")
    print("• Previous: Limited to specific domains")
    print("• DALL-E: General-purpose text-to-image generation")
    
    print(f"\nFOLLOW-UP INNOVATIONS:")
    print("="*40)
    print("• DALL-E 2: Diffusion-based with CLIP guidance")
    print("• Parti: Improved transformer architecture")
    print("• Imagen: Text-to-image diffusion models")
    print("• Midjourney: Commercial artistic applications")
    print("• Stable Diffusion: Open-source deployment")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• Established transformer viability for text-to-image")
    print("• Demonstrated importance of scale and data")
    print("• Foundation for modern multimodal models")
    print("• Inspired entire industry of AI art tools")
    print("• Set expectations for AI creativity capabilities")
    
    return {
        'model': 'DALL-E Text-to-Image',
        'year': YEAR,
        'innovation': INNOVATION,
        'total_params': total_params,
        'vqvae_embeddings': 1024,
        'transformer_layers': 8,
        'dalle_analysis': dalle_analysis
    }

if __name__ == "__main__":
    results = main()