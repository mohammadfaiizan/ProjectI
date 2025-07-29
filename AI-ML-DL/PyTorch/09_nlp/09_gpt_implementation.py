import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List

# GPT Configuration
class GPTConfig:
    """Configuration class for GPT model"""
    
    def __init__(self,
                 vocab_size=50257,
                 block_size=1024,
                 n_layer=12,
                 n_head=12,
                 n_embd=768,
                 dropout=0.1,
                 bias=True):
        self.vocab_size = vocab_size
        self.block_size = block_size  # Maximum sequence length
        self.n_layer = n_layer        # Number of transformer layers
        self.n_head = n_head          # Number of attention heads
        self.n_embd = n_embd          # Embedding dimension
        self.dropout = dropout
        self.bias = bias

# Causal Self-Attention
class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention mechanism"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Causal mask to ensure autoregressive property
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# MLP (Feed-Forward Network)
class MLP(nn.Module):
    """GPT MLP (feed-forward) block"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# GPT Block (Transformer Layer)
class Block(nn.Module):
    """GPT transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT Model
class GPTModel(nn.Module):
    """GPT Language Model"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Share weights between embedding and output layers (weight tying)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        
        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss
    
    def crop_block_size(self, block_size):
        """Crop the model's block size (for fine-tuning on shorter sequences)"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens using the model
        
        Args:
            idx: (B, T) array of indices in the current context
            max_new_tokens: number of new tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# GPT Trainer
class GPTTrainer:
    """Trainer for GPT models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def get_lr(self, step, config):
        """Learning rate schedule with warmup and cosine decay"""
        # 1) Linear warmup for warmup_iters steps
        if step < config.warmup_iters:
            return config.learning_rate * step / config.warmup_iters
        # 2) If step > lr_decay_iters, return min learning rate
        if step > config.lr_decay_iters:
            return config.min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    def estimate_loss(self, eval_iters, get_batch_fn):
        """Estimate validation loss"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch_fn(split)
                X, Y = X.to(self.device), Y.to(self.device)
                with torch.no_grad():
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train(self, config, get_batch_fn):
        """Train the GPT model"""
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        for iter_num in range(config.max_iters):
            # Determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num, config) if config.decay_lr else config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate the loss on train/val sets and write checkpoints
            if iter_num % config.eval_interval == 0:
                losses = self.estimate_loss(config.eval_iters, get_batch_fn)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Sample a batch of data
            xb, yb = get_batch_fn('train')
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

# Training Configuration
class TrainingConfig:
    """Training configuration for GPT"""
    
    def __init__(self):
        # I/O
        self.eval_interval = 250
        self.eval_iters = 200
        
        # Optimizer
        self.learning_rate = 6e-4  # max learning rate
        self.max_iters = 5000      # total number of training iterations
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        # Learning rate decay
        self.decay_lr = True       # whether to decay the learning rate
        self.warmup_iters = 1000   # how many steps to warm up for
        self.lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
        self.min_lr = 6e-5         # minimum learning rate, should be ~= learning_rate/10

# Text Generation Utilities
class TextGenerator:
    """Utilities for text generation with GPT"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def encode(self, text):
        """Encode text to token indices"""
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        """Decode token indices to text"""
        return self.tokenizer.decode(tokens)
    
    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=None, top_p=None):
        """Generate text from a prompt"""
        self.model.eval()
        
        # Encode the prompt
        context = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            if top_p is not None:
                generated = self.generate_with_nucleus_sampling(context, max_length, temperature, top_p)
            else:
                generated = self.model.generate(context, max_length, temperature, top_k)
        
        # Decode and return
        generated_text = self.decode(generated[0].tolist())
        return generated_text
    
    def generate_with_nucleus_sampling(self, context, max_length, temperature, top_p):
        """Generate text using nucleus (top-p) sampling"""
        generated = context
        
        for _ in range(max_length):
            # Get model predictions
            logits, _ = self.model(generated)
            logits = logits[:, -1, :] / temperature
            
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set logits to -inf for tokens to remove
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def beam_search(self, context, max_length, beam_size=5):
        """Generate text using beam search"""
        self.model.eval()
        device = context.device
        
        # Initialize beams
        beams = [(context, 0.0)]  # (sequence, log_probability)
        
        for _ in range(max_length):
            new_beams = []
            
            for sequence, log_prob in beams:
                # Get model predictions
                with torch.no_grad():
                    logits, _ = self.model(sequence)
                    logits = logits[:, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                
                for i in range(beam_size):
                    new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_sequence = torch.cat([sequence, new_token], dim=1)
                    new_log_prob = log_prob + top_log_probs[0, i].item()
                    new_beams.append((new_sequence, new_log_prob))
            
            # Keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Return the best sequence
        return beams[0][0]

# Simple Tokenizer for testing
class SimpleTokenizer:
    """Simple character-level tokenizer for testing"""
    
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])

# Data loading utilities
def get_batch(data, block_size, batch_size, device='cpu'):
    """Generate a batch of data for training"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print("Testing GPT implementation...")
    
    # Create a simple dataset
    text = """The quick brown fox jumps over the lazy dog. This is a sample text for testing the GPT model implementation. We will use this text to train a small GPT model and see how it performs in generating new text sequences."""
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Data size: {len(data)}, Vocab size: {tokenizer.vocab_size}")
    
    # Create model configuration
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    
    # Create model
    model = GPTModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    block_size = 32
    
    def get_batch_fn(split):
        return get_batch(data, block_size, batch_size)
    
    x, y = get_batch_fn('train')
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    # Forward pass
    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test generation
    print("\nTesting text generation...")
    
    # Generate some text
    prompt = "The quick"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    
    generated = model.generate(context, max_new_tokens=50, temperature=1.0, top_k=10)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated text: {generated_text}")
    
    # Test text generator utilities
    generator = TextGenerator(model, tokenizer)
    
    # Test beam search
    beam_result = generator.beam_search(context, max_length=30, beam_size=3)
    beam_text = tokenizer.decode(beam_result[0].tolist())
    print(f"Beam search result: {beam_text}")
    
    # Test nucleus sampling
    nucleus_result = generator.generate_with_nucleus_sampling(context, max_length=30, temperature=0.8, top_p=0.9)
    nucleus_text = tokenizer.decode(nucleus_result[0].tolist())
    print(f"Nucleus sampling result: {nucleus_text}")
    
    print("GPT implementation testing completed!")