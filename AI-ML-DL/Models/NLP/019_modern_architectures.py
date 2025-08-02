"""
ERA 7: MODERN ARCHITECTURES (2020-Present)
===========================================

Year: 2020-Present
Papers: InstructGPT, ChatGPT, GPT-4, CLIP, DALL-E, PaLM, LaMDA
Innovation: Instruction following, human alignment, multimodal capabilities, conversational AI
Previous Limitation: Models not aligned with human preferences, limited instruction following
Performance Gain: Human-aligned responses, multimodal understanding, conversational abilities
Impact: Democratized AI through ChatGPT, established human feedback training, multimodal AI

This file implements modern transformer architectures that brought AI to mainstream adoption
through instruction following, human alignment via RLHF, and multimodal capabilities.
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

YEAR = "2020-Present"
INNOVATION = "Human-aligned AI: Instruction following, RLHF, and multimodal capabilities"
PREVIOUS_LIMITATION = "Models not aligned with human preferences, poor instruction following"
IMPACT = "Democratized AI through ChatGPT, established RLHF paradigm, multimodal AI"

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
                    if 5 <= len(tokens) <= 25:  # Good length for modern models
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
    """Build vocabulary with modern special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 10)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<BOS>': 2,
        '<EOS>': 3,
        '<HUMAN>': 4,      # Human message marker
        '<ASSISTANT>': 5,  # Assistant response marker
        '<SYSTEM>': 6,     # System instruction marker
        '<REWARD>': 7,     # Reward signal marker
        '<INSTRUCT>': 8,   # Instruction prefix
        '<RESPONSE>': 9    # Response prefix
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# INSTRUCTION FOLLOWING DATASET
# ============================================================================

class InstructionDataset(Dataset):
    """
    Dataset for instruction following training
    Creates instruction-response pairs for training modern AI assistants
    """
    
    def __init__(self, sentences, vocab, task='instruction_following', max_length=128):
        self.vocab = vocab
        self.max_length = max_length
        self.task = task
        self.examples = []
        
        if task == 'instruction_following':
            self._create_instruction_examples(sentences)
        elif task == 'conversational':
            self._create_conversation_examples(sentences)
        elif task == 'helpful_harmless':
            self._create_helpful_harmless_examples(sentences)
        
        print(f"Created {len(self.examples)} {task} examples")
    
    def _create_instruction_examples(self, sentences):
        """Create instruction-following examples"""
        instruction_templates = [
            ("Explain what {concept} means", "explanation"),
            ("Summarize this text: {text}", "summary"),
            ("Answer this question: {question}", "answer"),
            ("Complete this sentence: {prefix}", "completion"),
            ("Rewrite this more formally: {text}", "rewrite")
        ]
        
        for sentence in sentences:
            if len(sentence) >= 3:
                # Create instruction-response pairs
                concept = random.choice(sentence)
                text_snippet = ' '.join(sentence[:5])
                
                # Random instruction type
                template_type = random.choice(instruction_templates)
                
                if template_type[1] == 'explanation':
                    instruction = f"Explain what {concept} means"
                    response = f"{concept} refers to " + ' '.join(sentence[:3])
                elif template_type[1] == 'summary':
                    instruction = f"Summarize this text: {text_snippet}"
                    response = concept  # Key concept as summary
                elif template_type[1] == 'completion':
                    instruction = f"Complete this sentence: {sentence[0]} {sentence[1]}"
                    response = ' '.join(sentence[2:4]) if len(sentence) > 3 else sentence[-1]
                else:
                    instruction = f"Answer this question about: {text_snippet}"
                    response = f"The answer involves {concept}"
                
                # Format as conversation
                full_input = f"<HUMAN> {instruction} <ASSISTANT>"
                full_target = f"{response} <EOS>"
                
                input_tokens = full_input.split()
                target_tokens = full_target.split()
                
                if len(input_tokens) + len(target_tokens) <= self.max_length:
                    self.examples.append({
                        'input': input_tokens,
                        'target': target_tokens,
                        'instruction': instruction,
                        'response': response
                    })
    
    def _create_conversation_examples(self, sentences):
        """Create conversational examples"""
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                human_msg = ' '.join(sentences[i][:8])
                assistant_msg = ' '.join(sentences[i + 1][:8])
                
                # Format as conversation turn
                conversation = f"<HUMAN> {human_msg} <ASSISTANT> {assistant_msg} <EOS>"
                tokens = conversation.split()
                
                if len(tokens) <= self.max_length:
                    # Input is everything except the last few tokens
                    input_tokens = tokens[:-len(assistant_msg.split()) - 1]  # Stop before assistant response
                    target_tokens = assistant_msg.split() + ['<EOS>']
                    
                    self.examples.append({
                        'input': input_tokens,
                        'target': target_tokens,
                        'conversation': conversation
                    })
    
    def _create_helpful_harmless_examples(self, sentences):
        """Create helpful and harmless response examples"""
        helpful_prefixes = [
            "I'd be happy to help you with",
            "Here's what I know about",
            "Let me explain",
            "I can assist you with"
        ]
        
        for sentence in sentences:
            if len(sentence) >= 4:
                topic = ' '.join(sentence[:3])
                helpful_prefix = random.choice(helpful_prefixes)
                
                instruction = f"Tell me about {topic}"
                response = f"{helpful_prefix} {topic}. " + ' '.join(sentence[3:6])
                
                full_input = f"<HUMAN> {instruction} <ASSISTANT>"
                full_target = f"{response} <EOS>"
                
                input_tokens = full_input.split()
                target_tokens = full_target.split()
                
                if len(input_tokens) + len(target_tokens) <= self.max_length:
                    self.examples.append({
                        'input': input_tokens,
                        'target': target_tokens,
                        'instruction': instruction,
                        'response': response
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to indices
        input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['input']]
        target_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['target']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'example': example
        }

def collate_instruction_fn(batch):
    """Collate function for instruction data"""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    # Pad sequences
    max_input_len = max(len(seq) for seq in input_ids)
    max_target_len = max(len(seq) for seq in target_ids)
    
    padded_input_ids = []
    padded_target_ids = []
    attention_masks = []
    
    for input_seq, target_seq in zip(input_ids, target_ids):
        # Pad input
        input_pad_len = max_input_len - len(input_seq)
        padded_input = torch.cat([input_seq, torch.zeros(input_pad_len, dtype=torch.long)])
        padded_input_ids.append(padded_input)
        
        # Pad target
        target_pad_len = max_target_len - len(target_seq)
        padded_target = torch.cat([target_seq, torch.zeros(target_pad_len, dtype=torch.long)])
        padded_target_ids.append(padded_target)
        
        # Attention mask
        attention_mask = torch.cat([torch.ones(len(input_seq)), torch.zeros(input_pad_len)])
        attention_masks.append(attention_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'target_ids': torch.stack(padded_target_ids),
        'attention_mask': torch.stack(attention_masks)
    }

# ============================================================================
# MODERN ATTENTION WITH IMPROVEMENTS
# ============================================================================

class ModernMultiHeadAttention(nn.Module):
    """
    Modern multi-head attention with optimizations from recent models
    Includes RoPE, flash attention concepts, and improved efficiency
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, use_rope=True):
        super(ModernMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        
        # QKV projections
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE (Rotary Position Embedding) components
        if use_rope:
            self.rope_cache = {}
            
        self.dropout = nn.Dropout(dropout)
        
    def get_rope_embeddings(self, seq_len, device):
        """Generate RoPE embeddings"""
        if seq_len in self.rope_cache:
            return self.rope_cache[seq_len]
        
        # Generate rotation matrices for RoPE
        dim = self.d_k
        position = torch.arange(seq_len, device=device).float()
        
        # Create frequency bands
        freqs = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
        
        # Position encodings
        pos_enc = torch.outer(position, freqs)
        
        # Rotary embeddings (simplified)
        cos_pos = torch.cos(pos_enc)
        sin_pos = torch.sin(pos_enc)
        
        self.rope_cache[seq_len] = (cos_pos, sin_pos)
        return cos_pos, sin_pos
    
    def apply_rope(self, x, cos_pos, sin_pos):
        """Apply rotary position embedding"""
        # Simplified RoPE application
        # In practice, this would be more complex rotation operations
        x_rope = x * cos_pos.unsqueeze(0).unsqueeze(0) + x * sin_pos.unsqueeze(0).unsqueeze(0)
        return x_rope
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass with modern optimizations
        """
        batch_size, seq_length, d_model = x.size()
        
        # QKV projection
        qkv = self.w_qkv(x)  # (batch_size, seq_length, 3 * d_model)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, d_k)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if enabled
        if self.use_rope:
            cos_pos, sin_pos = self.get_rope_embeddings(seq_length, x.device)
            q = self.apply_rope(q, cos_pos, sin_pos)
            k = self.apply_rope(k, cos_pos, sin_pos)
        
        # Attention computation (optimized)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - attention_mask) * -1e9
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.w_o(out)
        
        return output, attention_weights

# ============================================================================
# MODERN TRANSFORMER BLOCK
# ============================================================================

class ModernTransformerBlock(nn.Module):
    """
    Modern transformer block with latest optimizations
    Includes RMSNorm, SwiGLU activation, and improved architectures
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_rmsnorm=True):
        super(ModernTransformerBlock, self).__init__()
        
        self.use_rmsnorm = use_rmsnorm
        
        # Attention
        self.attention = ModernMultiHeadAttention(d_model, num_heads, dropout)
        
        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # SwiGLU feed-forward (modern activation)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture (modern standard)
        # Attention
        norm_x = self.norm1(x)
        attn_output, attention_weights = self.attention(norm_x, attention_mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        output = x + ff_output
        
        return output, attention_weights

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    More efficient alternative to LayerNorm used in modern models
    """
    
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU activation function used in modern transformer models
    Combines Swish activation with GLU (Gated Linear Unit)
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(SwiGLUFeedForward, self).__init__()
        
        # SwiGLU requires different dimensions
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: Swish(xW_gate) * (xW_up) * W_down
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        return self.w_down(self.dropout(gate * up))

# ============================================================================
# MODERN LANGUAGE MODEL
# ============================================================================

class ModernLanguageModel(nn.Module):
    """
    Modern language model with latest architectural improvements
    Incorporates techniques from GPT-4, PaLM, and other recent models
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12,
                 d_ff=3072, max_length=2048, dropout=0.1):
        super(ModernLanguageModel, self).__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Token embeddings (no positional embeddings - using RoPE instead)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (modern practice)
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Modern weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with modern optimizations
        """
        batch_size, seq_length = input_ids.size()
        
        # Token embeddings (no position embeddings - using RoPE)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Apply transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attention_weights = block(x, attention_mask)
            all_attention_weights.append(attention_weights)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, logits, all_attention_weights
    
    def generate_with_instruction(self, input_ids, vocab, idx_to_word, max_new_tokens=50,
                                temperature=0.7, top_p=0.9, do_sample=True):
        """
        Generate text following instructions with modern sampling
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
                
                if do_sample:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Check for end token
                if next_token[0].item() == vocab.get('<EOS>', -1):
                    break
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Limit length
                if generated.size(1) >= self.max_length:
                    break
        
        # Convert to words
        generated_words = []
        for token_idx in generated[0].tolist():
            if token_idx in idx_to_word and token_idx not in [vocab.get('<PAD>', 0)]:
                if token_idx == vocab.get('<EOS>', -1):
                    break
                generated_words.append(idx_to_word[token_idx])
        
        return generated_words

# ============================================================================
# REWARD MODEL FOR RLHF
# ============================================================================

class RewardModel(nn.Module):
    """
    Reward model for Reinforcement Learning from Human Feedback (RLHF)
    Predicts human preference scores for generated text
    """
    
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        
        # Use base model as backbone
        self.base_model = base_model
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.d_model, base_model.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_model.d_model // 2, 1)  # Single reward score
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Compute reward score for input sequence
        """
        # Get final hidden state from base model
        with torch.no_grad():
            _, logits, _ = self.base_model(input_ids, attention_mask)
        
        # Use final token representation
        final_hidden = logits[:, -1, :]  # (batch_size, d_model)
        
        # Compute reward score
        reward = self.reward_head(final_hidden)  # (batch_size, 1)
        
        return reward.squeeze(-1)  # (batch_size,)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_instruction_model(model, train_loader, val_loader, epochs=5, learning_rate=1e-4):
    """Train modern instruction-following model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Cosine learning rate schedule
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    train_losses = []
    val_losses = []
    
    print(f"Training modern instruction model on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Combine input and target for autoregressive training
            full_sequence = torch.cat([input_ids, target_ids], dim=1)
            labels = full_sequence.clone()
            
            # Mask input tokens in loss computation
            input_length = input_ids.size(1)
            labels[:, :input_length] = -100  # Ignore input tokens in loss
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, logits, _ = model(full_sequence, labels=labels)
            
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
        val_loss = evaluate_instruction_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_instruction_model(model, data_loader, device):
    """Evaluate instruction-following model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Combine sequences
            full_sequence = torch.cat([input_ids, target_ids], dim=1)
            labels = full_sequence.clone()
            
            # Mask input tokens
            input_length = input_ids.size(1)
            labels[:, :input_length] = -100
            
            loss, _, _ = model(full_sequence, labels=labels)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

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
    print(f"=== Modern Architectures ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:600]
    val_subset = val_sentences[:120]
    test_subset = test_sentences[:120]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    results = []
    training_histories = {}
    
    # Test modern models on different tasks
    tasks = ['instruction_following', 'conversational']
    
    for task in tasks[:1]:  # Test one task for demo
        print(f"\n" + "="*50)
        print(f"Training Modern Model for {task.upper()}")
        
        # Create dataset
        train_dataset = InstructionDataset(train_subset, vocab, task=task)
        val_dataset = InstructionDataset(val_subset, vocab, task=task)
        
        batch_size = 4  # Small batch size for modern models
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_instruction_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_instruction_fn)
        
        # Initialize modern model
        modern_model = ModernLanguageModel(
            vocab_size=len(vocab),
            d_model=512,       # Reduced for demo
            num_heads=8,
            num_layers=6,      # Reduced for demo
            d_ff=1024,
            max_length=256,
            dropout=0.1
        )
        
        print(f"Modern model parameters: {count_parameters(modern_model):,}")
        
        # Train model
        model_name = f'Modern-{task}'
        metrics = track_computational_metrics(
            model_name,
            train_instruction_model,
            modern_model, train_loader, val_loader, 5, 1e-4
        )
        
        train_losses, val_losses = metrics['result']
        training_histories[model_name] = (train_losses, val_losses)
        
        result = {
            'model': model_name,
            'task': task,
            'year': '2020-Present',
            'final_loss': val_losses[-1] if val_losses else 0,
            'parameters': count_parameters(modern_model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': 'Human-aligned instruction following'
        }
        results.append(result)
        
        # Demonstrate modern generation
        print(f"\n{task.upper()} EXAMPLES:")
        print("="*30)
        
        modern_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modern_model.to(device)
        
        # Test instruction following
        instructions = [
            "Explain artificial intelligence",
            "Summarize machine learning",
            "Complete this thought: The future of AI"
        ]
        
        for instruction in instructions:
            # Format as instruction
            input_text = f"<HUMAN> {instruction} <ASSISTANT>"
            input_tokens = [vocab.get(token, vocab['<UNK>']) for token in input_text.split()]
            input_tensor = torch.tensor([input_tokens], device=device)
            
            # Generate response
            generated = modern_model.generate_with_instruction(
                input_tensor, vocab, idx_to_word, max_new_tokens=15, temperature=0.8
            )
            
            # Extract response part
            response_start = len(input_text.split())
            response_tokens = generated[response_start:] if len(generated) > response_start else generated
            
            print(f"  Instruction: {instruction}")
            print(f"  Response:    {' '.join(response_tokens)}")
            print()
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        if train_losses:
            ax.plot(train_losses, label=model_name, linewidth=2)
    ax.set_title('Modern Model Training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss curves
    ax = axes[0, 1]
    for model_name, (_, val_losses) in training_histories.items():
        if val_losses:
            ax.plot(val_losses, label=model_name, linewidth=2)
    ax.set_title('Modern Model Validation Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Modern AI evolution
    ax = axes[1, 0]
    models = ['GPT-3\n(2020)', 'InstructGPT\n(2022)', 'ChatGPT\n(2022)', 'GPT-4\n(2023)']
    capabilities = [6, 8, 9, 10]
    bars = ax.bar(models, capabilities, color=['#3498DB', '#E67E22', '#E74C3C', '#9B59B6'])
    ax.set_title('Modern AI Capability Evolution', fontsize=14)
    ax.set_ylabel('Capability Score')
    ax.tick_params(axis='x', rotation=45)
    
    # Modern techniques
    ax = axes[1, 1]
    techniques = ['Instruction\nFollowing', 'RLHF', 'Chain of\nThought', 'Multimodal']
    impact_scores = [10, 9, 8, 9]
    bars = ax.bar(techniques, impact_scores, color=['#1ABC9C', '#F39C12', '#E74C3C', '#8E44AD'])
    ax.set_title('Modern AI Techniques', fontsize=14)
    ax.set_ylabel('Impact Score')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/019_modern_architectures_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 019_modern_architectures_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("MODERN ARCHITECTURES SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Task: {result['task']}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\n" + "="*70)
    print("MODERN AI REVOLUTION COMPONENTS")
    print("="*70)
    
    print("\n1. INSTRUCTION FOLLOWING:")
    print("   • Models trained to follow natural language instructions")
    print("   • Conversational interfaces like ChatGPT")
    print("   • Zero-shot task performance through instructions")
    
    print("\n2. REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF):")
    print("   • Train reward models on human preferences")
    print("   • Use RL to optimize for human-preferred responses")
    print("   • Improves helpfulness, harmlessness, honesty")
    
    print("\n3. ADVANCED ARCHITECTURES:")
    print("   • RoPE (Rotary Position Embedding)")
    print("   • RMSNorm instead of LayerNorm")
    print("   • SwiGLU activation functions")
    print("   • Flash Attention for efficiency")
    
    print("\n4. MULTIMODAL CAPABILITIES:")
    print("   • Vision-language models (CLIP)")
    print("   • Text-to-image generation (DALL-E)")
    print("   • Multimodal understanding")
    
    print("\nMODERN AI ACHIEVEMENTS:")
    print("="*50)
    print("• ChatGPT: Democratized AI through conversation")
    print("• GPT-4: Multimodal capabilities and reasoning")
    print("• InstructGPT: Human alignment through RLHF")
    print("• CLIP: Vision-language understanding")
    print("• Codex: Code generation capabilities")
    print("• DALL-E: Text-to-image generation")
    
    print("\nTECHNICAL INNOVATIONS:")
    print("="*50)
    print("• Chain-of-Thought prompting")
    print("• In-context learning optimization")
    print("• Constitutional AI for safety")
    print("• Tool use and API integration")
    print("• Long context extensions")
    print("• Efficient inference optimizations")
    
    print("\nIMPACT ON SOCIETY:")
    print("="*50)
    print("• Mainstream AI adoption")
    print("• New interaction paradigms")
    print("• Educational applications")
    print("• Creative AI assistance")
    print("• Coding and programming help")
    print("• Scientific research acceleration")
    
    print(f"\n{'='*70}")
    print("THE ERA OF HUMAN-ALIGNED AI")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()