"""
ERA 6: GPT SCALING REVOLUTION (2019-2020)
=========================================

Year: 2019-2020
Papers: GPT-1 (2018), GPT-2 (2019), GPT-3 (2020)
Innovation: Scaling laws and emergent capabilities through parameter and data scaling
Previous Limitation: BERT focused on understanding, limited autoregressive generation
Performance Gain: Emergent few-shot learning, human-like text generation
Impact: Demonstrated scaling laws, foundation for modern LLMs and ChatGPT

This file implements the GPT family that demonstrated how scaling Transformer
decoders leads to emergent capabilities and established the foundation for
modern large language models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import math
from collections import defaultdict, Counter
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2019-2020"
INNOVATION = "GPT Scaling Laws: Parameter scaling → Emergent capabilities"
PREVIOUS_LIMITATION = "BERT focused on understanding, limited generation capabilities"
IMPACT = "Scaling laws discovery, few-shot learning, foundation for modern LLMs"

print(f"=== {INNOVATION} ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING
# ============================================================================

def load_wikitext2_dataset():
    """Load WikiText-2 dataset with consistent preprocessing"""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    
    def preprocess_text(texts):
        sentences = []
        for text in texts:
            if text.strip():
                text = text.lower().replace('\n', ' ')
                text_sentences = text.split('.')
                for sentence in text_sentences:
                    tokens = word_tokenize(sentence.strip())
                    if 8 <= len(tokens) <= 35:  # Longer sequences for GPT
                        sentences.append(tokens)
        return sentences
    
    train_sentences = preprocess_text(dataset['train']['text'])
    val_sentences = preprocess_text(dataset['validation']['text'])
    test_sentences = preprocess_text(dataset['test']['text'])
    
    print(f"Train sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(val_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, vocab_size=8000):
    """Build vocabulary with GPT special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 4)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<BOS>': 2,  # Beginning of sequence
        '<EOS>': 3   # End of sequence
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# GPT DATASET CLASS
# ============================================================================

class GPTDataset(Dataset):
    """
    Dataset for GPT-style autoregressive language modeling
    Creates sequences for next-token prediction
    """
    
    def __init__(self, sentences, vocab, max_length=128, task='language_modeling'):
        self.vocab = vocab
        self.max_length = max_length
        self.task = task
        self.sequences = []
        
        if task == 'language_modeling':
            self._create_language_modeling_sequences(sentences)
        elif task == 'few_shot':
            self._create_few_shot_sequences(sentences)
        
        print(f"Created {len(self.sequences)} GPT sequences")
    
    def _create_language_modeling_sequences(self, sentences):
        """Create sequences for standard language modeling"""
        for sentence in sentences:
            if len(sentence) <= self.max_length - 2:
                # Add BOS and EOS tokens
                sequence = ['<BOS>'] + sentence + ['<EOS>']
                
                # For GPT, input and target are shifted by one position
                input_seq = sequence[:-1]  # All but last token
                target_seq = sequence[1:]  # All but first token
                
                self.sequences.append((input_seq, target_seq))
    
    def _create_few_shot_sequences(self, sentences):
        """Create few-shot learning examples"""
        # Create simple few-shot tasks
        # Task: Complete sentence based on pattern
        
        patterns = [
            ("The color of grass is", "green"),
            ("The color of sky is", "blue"), 
            ("The color of snow is", "white"),
            ("The capital of France is", "paris"),
            ("The capital of England is", "london")
        ]
        
        for i, sentence in enumerate(sentences[:100]):  # Limited for demo
            if len(sentence) >= 3:
                # Create completion task
                prompt_len = len(sentence) // 2
                prompt = sentence[:prompt_len]
                completion = sentence[prompt_len:]
                
                # Format as few-shot example
                few_shot_text = f"Complete: {' '.join(prompt)} -> {' '.join(completion[:3])}"
                tokens = few_shot_text.split()
                
                if len(tokens) <= self.max_length - 2:
                    sequence = ['<BOS>'] + tokens + ['<EOS>']
                    input_seq = sequence[:-1]
                    target_seq = sequence[1:]
                    
                    self.sequences.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        
        # Convert to indices
        input_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in input_seq]
        target_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in target_seq]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_gpt_fn(batch):
    """Collate function for GPT data"""
    input_sequences, target_sequences = zip(*batch)
    
    # Pad sequences to same length
    input_lengths = [len(seq) for seq in input_sequences]
    target_lengths = [len(seq) for seq in target_sequences]
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    
    padded_inputs = []
    padded_targets = []
    attention_masks = []
    
    for input_seq, target_seq in zip(input_sequences, target_sequences):
        # Pad input
        input_pad_len = max_input_len - len(input_seq)
        padded_input = torch.cat([input_seq, torch.zeros(input_pad_len, dtype=torch.long)])
        padded_inputs.append(padded_input)
        
        # Pad target  
        target_pad_len = max_target_len - len(target_seq)
        padded_target = torch.cat([target_seq, torch.zeros(target_pad_len, dtype=torch.long)])
        padded_targets.append(padded_target)
        
        # Create attention mask
        attention_mask = torch.cat([torch.ones(len(input_seq)), torch.zeros(input_pad_len)])
        attention_masks.append(attention_mask)
    
    return {
        'input_ids': torch.stack(padded_inputs),
        'labels': torch.stack(padded_targets),
        'attention_mask': torch.stack(attention_masks)
    }

# ============================================================================
# GPT ATTENTION WITH CAUSAL MASKING
# ============================================================================

class GPTMultiHeadAttention(nn.Module):
    """
    Multi-head attention with causal masking for autoregressive generation
    Core component of GPT architecture
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(GPTMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Single linear layer for Q, K, V (GPT optimization)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            attention_mask: (batch_size, seq_length) 
            
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, d_model = x.size()
        
        # Linear projection to Q, K, V
        qkv = self.c_attn(x)  # (batch_size, seq_length, 3 * d_model)
        
        # Split into Q, K, V and reshape for multi-head
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device))
        causal_mask = causal_mask.view(1, 1, seq_length, seq_length)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)
        
        # Apply padding mask if provided
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = (1.0 - padding_mask) * -1e9
            attention_scores = attention_scores + padding_mask
        
        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.c_proj(output)
        
        return output, attention_weights

# ============================================================================
# GPT TRANSFORMER BLOCK
# ============================================================================

class GPTBlock(nn.Module):
    """
    GPT Transformer block with causal self-attention and feed-forward
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(GPTBlock, self).__init__()
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPTMultiHeadAttention(d_model, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT uses GELU activation
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            attention_mask: (batch_size, seq_length)
            
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: attention weights from self-attention
        """
        # Pre-norm architecture (GPT-2 style)
        # Attention block
        norm_x = self.ln_1(x)
        attn_output, attention_weights = self.attn(norm_x, attention_mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward block
        norm_x = self.ln_2(x)
        mlp_output = self.mlp(norm_x)
        output = x + mlp_output
        
        return output, attention_weights

# ============================================================================
# GPT MODEL ARCHITECTURES
# ============================================================================

class GPTModel(nn.Module):
    """
    GPT model for autoregressive language modeling
    Supports different scales (GPT-1, GPT-2, GPT-3 sizes)
    """
    
    def __init__(self, vocab_size, max_length=1024, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, dropout=0.1):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between input and output embeddings (GPT optimization)
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT initialization scheme"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            labels: (batch_size, seq_length) for loss computation
            
        Returns:
            loss: language modeling loss (if labels provided)
            logits: (batch_size, seq_length, vocab_size)
            all_attention_weights: list of attention weights from each layer
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Position indices
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = self.dropout(token_embeds + position_embeds)
        
        # Apply transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attention_weights = block(x, attention_mask)
            all_attention_weights.append(attention_weights)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, logits, all_attention_weights
    
    def generate(self, input_ids, vocab, idx_to_word, max_new_tokens=20, 
                 temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using the trained GPT model
        Supports different decoding strategies
        """
        self.eval()
        device = input_ids.device
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                _, logits, _ = self.forward(generated)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                if next_token[0].item() == vocab.get('<EOS>', -1):
                    break
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Limit sequence length
                if generated.size(1) >= self.max_length:
                    break
        
        # Convert to words
        generated_words = []
        for token_idx in generated[0].tolist():
            if token_idx in idx_to_word and token_idx not in [vocab.get('<PAD>', 0), vocab.get('<BOS>', 1)]:
                if token_idx == vocab.get('<EOS>', -1):
                    break
                generated_words.append(idx_to_word[token_idx])
        
        return generated_words

# ============================================================================
# GPT MODEL CONFIGURATIONS (Scaling)
# ============================================================================

def get_gpt_config(model_size='small'):
    """
    Get GPT configuration for different model sizes
    Demonstrates scaling from GPT-1 to GPT-3 sizes
    """
    configs = {
        'tiny': {  # For demo purposes
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 4,
            'd_ff': 256,
            'max_length': 256
        },
        'small': {  # GPT-1 scale
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 512,
            'max_length': 512
        },
        'medium': {  # GPT-2 small
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 8,
            'd_ff': 1024,
            'max_length': 1024
        },
        'large': {  # GPT-2 large scale (scaled down for demo)
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 12,
            'd_ff': 2048,
            'max_length': 1024
        }
    }
    
    return configs.get(model_size, configs['small'])

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_gpt(model, train_loader, val_loader, epochs=10, learning_rate=6e-4):
    """Train GPT model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # AdamW optimizer with weight decay (GPT optimization)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           betas=(0.9, 0.95), weight_decay=0.1)
    
    # Learning rate schedule (cosine with warmup)
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    train_losses = []
    val_perplexities = []
    
    print(f"Training GPT on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, logits, _ = model(input_ids, attention_mask, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_perplexity = evaluate_gpt(model, val_loader, device)
        val_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    return train_losses, val_perplexities

def evaluate_gpt(model, data_loader, device):
    """Evaluate GPT model perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            loss, _, _ = model(input_ids, attention_mask, labels)
            
            # Count non-padding tokens
            num_tokens = (labels != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()

# ============================================================================
# SCALING LAW ANALYSIS
# ============================================================================

def analyze_scaling_laws(models_configs, results):
    """
    Analyze scaling laws: relationship between model size and performance
    Demonstrates the key insight from GPT research
    """
    model_sizes = []
    parameters = []
    perplexities = []
    
    for config_name, result in zip(models_configs, results):
        config = get_gpt_config(config_name)
        
        # Calculate approximate parameter count
        vocab_size = 3000  # From our demo
        param_count = (
            vocab_size * config['d_model'] +  # Token embeddings
            config['max_length'] * config['d_model'] +  # Position embeddings
            config['num_layers'] * (
                4 * config['d_model']**2 +  # Attention weights
                2 * config['d_model'] * config['d_ff']  # FFN weights
            ) +
            config['d_model'] * vocab_size  # Output projection
        )
        
        model_sizes.append(config_name)
        parameters.append(param_count)
        perplexities.append(result.get('final_perplexity', 100))
    
    return model_sizes, parameters, perplexities

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

def track_computational_metrics(model_name, train_function, *args):
    """Track computational metrics during training"""
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.time()
    result = train_function(*args)
    training_time = time.time() - start_time
    
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    
    return {
        'model_name': model_name,
        'training_time_minutes': training_time / 60,
        'memory_usage_mb': memory_used,
        'result': result
    }

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== GPT Scaling Revolution ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:800]
    val_subset = val_sentences[:160]
    test_subset = test_sentences[:160]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    results = []
    training_histories = {}
    model_configs = ['tiny', 'small']  # Test different scales
    
    for config_name in model_configs:
        print(f"\n" + "="*50)
        print(f"Training GPT-{config_name.upper()}")
        
        # Get model configuration
        config = get_gpt_config(config_name)
        
        # Create dataset
        max_length = min(config['max_length'], 64)  # Limit for demo
        train_dataset = GPTDataset(train_subset, vocab, max_length=max_length)
        val_dataset = GPTDataset(val_subset, vocab, max_length=max_length)
        
        # Create data loaders
        batch_size = 8 if config_name == 'tiny' else 4  # Smaller batch for larger models
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_gpt_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_gpt_fn)
        
        # Initialize GPT model
        gpt_model = GPTModel(
            vocab_size=len(vocab),
            max_length=max_length,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=0.1
        )
        
        print(f"GPT-{config_name} parameters: {count_parameters(gpt_model):,}")
        
        # Train model
        model_name = f'GPT-{config_name}'
        epochs = 6 if config_name == 'tiny' else 5  # Fewer epochs for larger models
        
        metrics = track_computational_metrics(
            model_name,
            train_gpt,
            gpt_model, train_loader, val_loader, epochs, 6e-4
        )
        
        train_losses, val_perplexities = metrics['result']
        training_histories[model_name] = (train_losses, val_perplexities)
        
        result = {
            'model': model_name,
            'config': config_name,
            'year': '2019-2020',
            'final_perplexity': val_perplexities[-1] if val_perplexities else 0,
            'parameters': count_parameters(gpt_model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': 'Autoregressive scaling and emergent capabilities'
        }
        results.append(result)
        
        # Demonstrate generation
        print(f"\n{config_name.upper()} GENERATION EXAMPLES:")
        print("="*30)
        
        gpt_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpt_model.to(device)
        
        # Generate with different prompts
        prompts = [
            ['the', 'quick'],
            ['in', 'the', 'beginning'],
            ['artificial', 'intelligence']
        ]
        
        for prompt in prompts:
            # Convert prompt to indices
            prompt_indices = [vocab.get(word, vocab['<UNK>']) for word in prompt]
            prompt_tensor = torch.tensor([prompt_indices], device=device)
            
            # Generate with different strategies
            generated_greedy = gpt_model.generate(
                prompt_tensor, vocab, idx_to_word, max_new_tokens=10, temperature=1.0
            )
            
            generated_creative = gpt_model.generate(
                prompt_tensor, vocab, idx_to_word, max_new_tokens=10, 
                temperature=1.2, top_k=20
            )
            
            print(f"  Prompt: {' '.join(prompt)}")
            print(f"  Greedy: {' '.join(generated_greedy)}")
            print(f"  Creative: {' '.join(generated_creative)}")
            print()
    
    # Analyze scaling laws
    model_sizes, parameters, perplexities = analyze_scaling_laws(model_configs, results)
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        if train_losses:
            ax.plot(train_losses, label=model_name, linewidth=2)
    ax.set_title('GPT Training Loss by Model Size', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Perplexity comparison
    ax = axes[0, 1]
    for model_name, (_, val_perplexities) in training_histories.items():
        if val_perplexities:
            ax.plot(val_perplexities, label=model_name, linewidth=2)
    ax.set_title('GPT Validation Perplexity', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scaling law visualization
    ax = axes[1, 0]
    if len(parameters) > 1:
        ax.loglog(parameters, perplexities, 'o-', linewidth=2, markersize=8, color='#E74C3C')
        for i, size in enumerate(model_sizes):
            ax.annotate(f'GPT-{size}', (parameters[i], perplexities[i]), 
                       xytext=(5, 5), textcoords='offset points')
    ax.set_title('GPT Scaling Laws: Parameters vs Perplexity', fontsize=14)
    ax.set_xlabel('Parameters (log scale)')
    ax.set_ylabel('Perplexity (log scale)')
    ax.grid(True, alpha=0.3)
    
    # GPT evolution timeline
    ax = axes[1, 1]
    gpt_versions = ['GPT-1\n(2018)', 'GPT-2\n(2019)', 'GPT-3\n(2020)']
    param_counts_real = [117e6, 1.5e9, 175e9]  # Real parameter counts
    capabilities = [3, 7, 10]  # Relative capability scores
    
    bars = ax.bar(gpt_versions, capabilities, color=['#3498DB', '#E67E22', '#E74C3C'])
    ax.set_title('GPT Evolution: Emergent Capabilities', fontsize=14)
    ax.set_ylabel('Capability Score')
    
    # Add parameter counts as text
    for i, (bar, params) in enumerate(zip(bars, param_counts_real)):
        height = bar.get_height()
        if params >= 1e9:
            param_text = f'{params/1e9:.1f}B'
        else:
            param_text = f'{params/1e6:.0f}M'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{param_text} params', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/016_gpt_scaling_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 016_gpt_scaling_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("🚀 GPT SCALING REVOLUTION SUMMARY 🚀")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  📊 Final Perplexity: {result['final_perplexity']:.2f}")
        print(f"  🔢 Parameters: {result['parameters']:,}")
        print(f"  ⏱️  Training Time: {result['training_time']:.2f} minutes")
        print(f"  💾 Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  💡 Innovation: {result['innovation']}")
    
    print("\n🧠 GPT ARCHITECTURAL INNOVATIONS:")
    print("="*50)
    print("1. 🎯 AUTOREGRESSIVE DECODING:")
    print("   • Unidirectional (left-to-right) attention")
    print("   • Causal masking prevents future information leakage")
    print("   • Optimized for text generation tasks")
    
    print("\n2. 📏 SCALING LAWS:")
    print("   • Larger models consistently outperform smaller ones")
    print("   • Performance scales predictably with parameters")
    print("   • Emergent capabilities appear at scale")
    
    print("\n3. ⚡ TRAINING OPTIMIZATIONS:")
    print("   • AdamW optimizer with weight decay")
    print("   • Cosine learning rate schedule with warmup")
    print("   • Gradient clipping for stability")
    print("   • Pre-norm architecture (GPT-2+)")
    
    print("\n4. 🎨 GENERATION STRATEGIES:")
    print("   • Temperature sampling for creativity control")
    print("   • Top-k and top-p (nucleus) sampling")
    print("   • Beam search for quality generation")
    
    print("\n📈 SCALING LAW DISCOVERIES:")
    print("="*50)
    print("• 📐 Power Law: Loss ∝ N^(-α) where N = parameters")
    print("• 📊 Data Scaling: Performance improves with dataset size")
    print("• ⚡ Compute Scaling: Optimal allocation of compute budget")
    print("• 🎯 Transfer: Larger models transfer better to new tasks")
    print("• 🔍 Few-shot: Emergent in-context learning abilities")
    
    print("\n🌟 EMERGENT CAPABILITIES:")
    print("="*50)
    print("GPT-1 (117M params):")
    print("  • Basic language modeling")
    print("  • Simple text completion")
    print("  • Demonstrated transfer learning")
    
    print("\nGPT-2 (1.5B params):")
    print("  • Coherent long-form generation")
    print("  • Zero-shot task performance")
    print("  • Diverse writing styles")
    print("  • 'Too dangerous to release' controversy")
    
    print("\nGPT-3 (175B params):")
    print("  • Few-shot in-context learning")
    print("  • Meta-learning capabilities")
    print("  • Code generation (Codex)")
    print("  • Human-like conversation")
    print("  • Creative writing and reasoning")
    
    print("\n🔬 TECHNICAL BREAKTHROUGHS:")
    print("="*50)
    print("• Causal Self-Attention: P(w_t | w_1, ..., w_{t-1})")
    print("• Weight Tying: Input embeddings = Output weights")
    print("• Pre-norm: LayerNorm before attention/FFN")
    print("• GELU Activation: Smooth, probabilistic activation")
    print("• Byte-Pair Encoding: Efficient tokenization")
    print("• Position Embeddings: Learned absolute positions")
    
    print("\n💡 WHY SCALING WORKED:")
    print("="*50)
    print("1. 📈 SMOOTH SCALING: No performance plateaus observed")
    print("2. 🎯 UNIVERSAL TASKS: Single architecture for all text tasks")
    print("3. 🧠 EMERGENT ABILITIES: New capabilities appear unpredictably")
    print("4. 📊 DATA ABUNDANCE: Internet-scale text availability")
    print("5. ⚡ COMPUTE GROWTH: GPU/TPU computational improvements")
    print("6. 🔧 OPTIMIZATION: Better training techniques and stability")
    
    print("\n🌈 GENERATION QUALITY PROGRESSION:")
    print("="*50)
    print("GPT-1: 'The dog ran fast and'")
    print("       → 'then he stopped running'")
    print("")
    print("GPT-2: 'In a shocking finding, scientists discovered'")
    print("       → 'a herd of unicorns living in a remote valley...'")
    print("")
    print("GPT-3: 'Explain quantum computing like I'm 5:'")
    print("       → 'Imagine you have a magic box that can try...'")
    
    print("\n🔄 BERT vs GPT PARADIGMS:")
    print("="*50)
    print("BERT (Bidirectional Encoder):")
    print("  ✅ Understanding tasks (classification, QA)")
    print("  ✅ Bidirectional context")
    print("  ❌ Limited generation capabilities")
    print("  🎯 Pre-train → Fine-tune")
    
    print("\nGPT (Autoregressive Decoder):")
    print("  ✅ Generation tasks (completion, creative writing)")
    print("  ✅ Zero/few-shot learning")
    print("  ❌ Unidirectional context")
    print("  🎯 Pre-train → Prompt")
    
    print("\n🎓 KEY INSIGHTS FROM GPT:")
    print("="*50)
    print("• Scale is all you need: Bigger is consistently better")
    print("• Emergent abilities: New capabilities appear at scale")
    print("• Generative pre-training: Powerful alternative to BERT")
    print("• In-context learning: Few-shot without parameter updates")
    print("• Universal architecture: One model for all text tasks")
    print("• Autoregressive modeling: Simple but incredibly powerful")
    
    print("\n🚀 IMPACT ON MODERN AI:")
    print("="*50)
    print("• 🏗️  Foundation for GPT-4, ChatGPT, and modern LLMs")
    print("• 📏 Established scaling laws as core AI principle")
    print("• 🎯 Enabled few-shot and zero-shot learning paradigms")
    print("• 💬 Made conversational AI practical and accessible")
    print("• 🧠 Demonstrated emergent intelligence from scale")
    print("• 🌐 Changed public perception of AI capabilities")
    
    print(f"\n{'='*70}")
    print("📈 GPT: SCALING TO ARTIFICIAL GENERAL INTELLIGENCE 📈")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()