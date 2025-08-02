"""
ERA 6: EFFICIENT TRANSFORMERS (2019-2020)
==========================================

Year: 2019-2020
Papers: DistilBERT, ALBERT, RoBERTa, MobileBERT
Innovation: Efficient architectures and training procedures for practical deployment
Previous Limitation: BERT and GPT models too large/slow for many real-world applications
Performance Gain: 40-90% size reduction with minimal performance loss
Impact: Made transformers practical for production, mobile, and resource-constrained environments

This file implements efficient transformer variants that made BERT-like performance
accessible for production deployment through knowledge distillation, parameter
sharing, and optimized training procedures.
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
INNOVATION = "Efficient Transformers: Practical deployment through compression and optimization"
PREVIOUS_LIMITATION = "BERT/GPT too large and slow for many real-world applications"
IMPACT = "Made transformers practical for production, mobile, and edge deployment"

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
                    if 5 <= len(tokens) <= 25:  # Good length for efficient models
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
    """Build vocabulary with special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 6)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<CLS>': 2,
        '<SEP>': 3,
        '<MASK>': 4,
        '<EOS>': 5
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# EFFICIENT TRANSFORMER DATASET
# ============================================================================

class EfficientTransformerDataset(Dataset):
    """Dataset for training efficient transformer variants"""
    
    def __init__(self, sentences, vocab, task='masked_lm', max_length=128):
        self.vocab = vocab
        self.max_length = max_length
        self.task = task
        self.examples = []
        
        if task == 'masked_lm':
            self._create_masked_lm_examples(sentences)
        elif task == 'classification':
            self._create_classification_examples(sentences)
        
        print(f"Created {len(self.examples)} {task} examples")
    
    def _create_masked_lm_examples(self, sentences):
        """Create masked language modeling examples"""
        for sentence in sentences:
            if len(sentence) <= self.max_length - 2:
                # Add CLS and SEP tokens
                tokens = ['<CLS>'] + sentence + ['<SEP>']
                
                # Apply masking
                masked_tokens, labels = self._apply_masking(tokens)
                
                self.examples.append({
                    'tokens': masked_tokens,
                    'labels': labels
                })
    
    def _create_classification_examples(self, sentences):
        """Create classification examples"""
        for sentence in sentences:
            if len(sentence) <= self.max_length - 2:
                tokens = ['<CLS>'] + sentence + ['<SEP>']
                
                # Simple sentiment classification based on heuristics
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worse']
                
                pos_count = sum(1 for word in sentence if word in positive_words)
                neg_count = sum(1 for word in sentence if word in negative_words)
                
                if pos_count > neg_count:
                    label = 1
                elif neg_count > pos_count:
                    label = 0
                else:
                    label = random.randint(0, 1)
                
                self.examples.append({
                    'tokens': tokens,
                    'label': label
                })
    
    def _apply_masking(self, tokens, mask_prob=0.15):
        """Apply BERT-style masking"""
        masked_tokens = tokens[:]
        labels = [-100] * len(tokens)  # -100 ignored in loss
        
        for i, token in enumerate(tokens):
            if token in ['<CLS>', '<SEP>']:
                continue
                
            if random.random() < mask_prob:
                labels[i] = self.vocab.get(token, self.vocab['<UNK>'])
                
                if random.random() < 0.8:
                    masked_tokens[i] = '<MASK>'
                elif random.random() < 0.5:
                    masked_tokens[i] = random.choice(list(self.vocab.keys()))
        
        return masked_tokens, labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.task == 'masked_lm':
            input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['tokens']]
            labels = example['labels']
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:  # classification
            input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['tokens']]
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(example['label'], dtype=torch.long)
            }

def collate_efficient_fn(batch):
    """Collate function for efficient transformer data"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    max_len = max(len(seq) for seq in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for i in range(len(batch)):
        seq_len = len(input_ids[i])
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input = torch.cat([input_ids[i], torch.zeros(pad_len, dtype=torch.long)])
        padded_input_ids.append(padded_input)
        
        # Pad labels
        if labels[i].dim() == 0:  # Classification labels
            padded_labels.append(labels[i])
        else:  # MLM labels
            padded_label = torch.cat([labels[i], torch.full((pad_len,), -100, dtype=torch.long)])
            padded_labels.append(padded_label)
        
        # Attention mask
        attention_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        attention_masks.append(attention_mask)
    
    result = {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks)
    }
    
    if labels[0].dim() == 0:  # Classification
        result['labels'] = torch.stack(padded_labels)
    else:  # MLM
        result['labels'] = torch.stack(padded_labels)
    
    return result

# ============================================================================
# DISTILBERT: KNOWLEDGE DISTILLATION
# ============================================================================

class DistilBERTAttention(nn.Module):
    """Simplified attention mechanism for DistilBERT"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(DistilBERTAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_length, hidden_size = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores += (1.0 - attention_mask) * -10000.0
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        
        output = self.out_proj(context)
        return output, attention_probs

class DistilBERTLayer(nn.Module):
    """DistilBERT transformer layer - simplified version of BERT layer"""
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super(DistilBERTLayer, self).__init__()
        
        self.attention = DistilBERTAttention(hidden_size, num_heads, dropout)
        self.sa_layer_norm = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        # Self-attention
        sa_output, attention_weights = self.attention(x, attention_mask)
        sa_output = self.sa_layer_norm(x + self.dropout(sa_output))
        
        # Feed-forward
        ffn_output = self.ffn(sa_output)
        output = self.output_layer_norm(sa_output + ffn_output)
        
        return output, attention_weights

class DistilBERT(nn.Module):
    """
    DistilBERT: Distilled version of BERT
    50% smaller, 60% faster, retains 95% of BERT's performance
    """
    
    def __init__(self, vocab_size, hidden_size=512, num_heads=8, num_layers=6, 
                 intermediate_size=1024, max_position_embeddings=512, dropout=0.1):
        super(DistilBERT, self).__init__()
        
        # Embeddings (no token type embeddings - simplification)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers (fewer than BERT)
        self.transformer = nn.ModuleList([
            DistilBERTLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Register position ids
        self.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transformer layers
        hidden_states = embeddings
        all_attention_weights = []
        
        for layer in self.transformer:
            hidden_states, attention_weights = layer(hidden_states, attention_mask)
            all_attention_weights.append(attention_weights)
        
        return hidden_states, all_attention_weights

# ============================================================================
# ALBERT: PARAMETER SHARING AND FACTORIZATION
# ============================================================================

class ALBERTEmbeddings(nn.Module):
    """
    ALBERT embeddings with factorized embedding parameterization
    Reduces embedding parameters by factorizing into smaller matrices
    """
    
    def __init__(self, vocab_size, embedding_size, hidden_size, max_position_embeddings=512, dropout=0.1):
        super(ALBERTEmbeddings, self).__init__()
        
        # Factorized embeddings: vocab_size -> embedding_size -> hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.word_embedding_projection = nn.Linear(embedding_size, hidden_size)
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        # Factorized word embeddings
        word_embeds = self.word_embeddings(input_ids)
        word_embeds = self.word_embedding_projection(word_embeds)
        
        position_embeds = self.position_embeddings(position_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class ALBERTLayer(nn.Module):
    """
    ALBERT layer with cross-layer parameter sharing
    All layers share the same parameters
    """
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super(ALBERTLayer, self).__init__()
        
        self.attention = DistilBERTAttention(hidden_size, num_heads, dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output, attention_weights = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_layer_norm(hidden_states + self.dropout(attention_output))
        
        # Feed-forward
        ffn_output = self.ffn(attention_output)
        layer_output = self.ffn_layer_norm(attention_output + ffn_output)
        
        return layer_output, attention_weights

class ALBERT(nn.Module):
    """
    ALBERT: A Lite BERT
    Uses parameter sharing and factorized embeddings for efficiency
    """
    
    def __init__(self, vocab_size, embedding_size=128, hidden_size=512, num_heads=8,
                 num_layers=12, intermediate_size=1024, dropout=0.1):
        super(ALBERT, self).__init__()
        
        self.embeddings = ALBERTEmbeddings(vocab_size, embedding_size, hidden_size, dropout=dropout)
        
        # Single shared layer (parameter sharing across all layers)
        self.shared_layer = ALBERTLayer(hidden_size, num_heads, intermediate_size, dropout)
        self.num_layers = num_layers
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Apply the same layer multiple times (parameter sharing)
        all_attention_weights = []
        for _ in range(self.num_layers):
            hidden_states, attention_weights = self.shared_layer(hidden_states, attention_mask)
            all_attention_weights.append(attention_weights)
        
        return hidden_states, all_attention_weights

# ============================================================================
# ROBERTA: OPTIMIZED TRAINING
# ============================================================================

class RoBERTa(nn.Module):
    """
    RoBERTa: Robustly Optimized BERT Pretraining Approach
    Same architecture as BERT but with optimized training procedure
    """
    
    def __init__(self, vocab_size, hidden_size=512, num_heads=8, num_layers=6,
                 intermediate_size=1024, max_position_embeddings=514, dropout=0.1):
        super(RoBERTa, self).__init__()
        
        # Similar to BERT but without NSP and with optimizations
        self.embeddings = ALBERTEmbeddings(vocab_size, hidden_size, hidden_size, 
                                         max_position_embeddings, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ALBERTLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        
        all_attention_weights = []
        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, attention_mask)
            all_attention_weights.append(attention_weights)
        
        return hidden_states, all_attention_weights

# ============================================================================
# EFFICIENT TRANSFORMER HEADS
# ============================================================================

class EfficientMLMHead(nn.Module):
    """Efficient masked language modeling head"""
    
    def __init__(self, hidden_size, vocab_size):
        super(EfficientMLMHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits

class EfficientClassificationHead(nn.Module):
    """Efficient classification head"""
    
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(EfficientClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states):
        # Use [CLS] token
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ============================================================================
# COMPLETE EFFICIENT MODELS
# ============================================================================

class DistilBERTForMaskedLM(nn.Module):
    """DistilBERT with MLM head"""
    
    def __init__(self, vocab_size, hidden_size=512, num_heads=8, num_layers=6):
        super(DistilBERTForMaskedLM, self).__init__()
        self.distilbert = DistilBERT(vocab_size, hidden_size, num_heads, num_layers)
        self.mlm_head = EfficientMLMHead(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states, attention_weights = self.distilbert(input_ids, attention_mask)
        logits = self.mlm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss, logits, attention_weights

class ALBERTForClassification(nn.Module):
    """ALBERT with classification head"""
    
    def __init__(self, vocab_size, num_labels=2, hidden_size=512):
        super(ALBERTForClassification, self).__init__()
        self.albert = ALBERT(vocab_size, hidden_size=hidden_size)
        self.classifier = EfficientClassificationHead(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states, attention_weights = self.albert(input_ids, attention_mask)
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return loss, logits, attention_weights

# ============================================================================
# KNOWLEDGE DISTILLATION
# ============================================================================

def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """
    Compute knowledge distillation loss
    Combines soft targets from teacher and hard targets from labels
    """
    # Soft target loss (distillation)
    student_log_softmax = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_softmax = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (temperature ** 2)
    
    # Hard target loss (original task)
    hard_loss = F.cross_entropy(student_logits, labels, ignore_index=-100)
    
    # Combine losses
    total_loss = alpha * kd_loss + (1 - alpha) * hard_loss
    
    return total_loss, kd_loss, hard_loss

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_efficient_model(model, train_loader, val_loader, epochs=5, learning_rate=5e-5, 
                         teacher_model=None):
    """Train efficient transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training efficient model on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if teacher_model is not None:
                # Knowledge distillation
                with torch.no_grad():
                    teacher_loss, teacher_logits, _ = teacher_model(input_ids, attention_mask, labels)
                
                student_loss, student_logits, _ = model(input_ids, attention_mask, labels)
                
                if labels.dim() > 1:  # MLM task
                    # Flatten for distillation
                    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
                    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
                    labels_flat = labels.view(-1)
                    
                    # Only distill on masked positions
                    mask = labels_flat != -100
                    if mask.sum() > 0:
                        loss, kd_loss, hard_loss = knowledge_distillation_loss(
                            student_logits_flat[mask], teacher_logits_flat[mask], labels_flat[mask]
                        )
                    else:
                        loss = student_loss
                else:  # Classification task
                    loss, kd_loss, hard_loss = knowledge_distillation_loss(
                        student_logits, teacher_logits, labels
                    )
            else:
                # Standard training
                loss, logits, _ = model(input_ids, attention_mask, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        if hasattr(model, 'classifier'):  # Classification model
            val_accuracy = evaluate_classification(model, val_loader, device)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:  # MLM model
            val_loss = evaluate_mlm(model, val_loader, device)
            val_accuracies.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_accuracies

def evaluate_classification(model, data_loader, device):
    """Evaluate classification accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            _, logits, _ = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def evaluate_mlm(model, data_loader, device):
    """Evaluate MLM loss"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _, _ = model(input_ids, attention_mask, labels)
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
    print(f"=== Efficient Transformers ({YEAR}) ===")
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
    
    # Test efficient models
    models_to_test = [
        ('DistilBERT', 'mlm'),
        ('ALBERT', 'classification'),
        ('RoBERTa', 'classification')
    ]
    
    for model_name, task in models_to_test[:2]:  # Test 2 models for demo
        print(f"\n" + "="*50)
        print(f"Training {model_name} for {task.upper()}")
        
        # Create dataset
        train_dataset = EfficientTransformerDataset(train_subset, vocab, task=task)
        val_dataset = EfficientTransformerDataset(val_subset, vocab, task=task)
        
        batch_size = 8
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_efficient_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_efficient_fn)
        
        # Initialize model
        if model_name == 'DistilBERT':
            model = DistilBERTForMaskedLM(vocab_size=len(vocab), hidden_size=256, num_layers=4)
        elif model_name == 'ALBERT':
            model = ALBERTForClassification(vocab_size=len(vocab), hidden_size=256)
        else:  # RoBERTa
            model = ALBERTForClassification(vocab_size=len(vocab), hidden_size=256)
        
        print(f"{model_name} parameters: {count_parameters(model):,}")
        
        # Train model
        metrics = track_computational_metrics(
            model_name,
            train_efficient_model,
            model, train_loader, val_loader, 5, 5e-5
        )
        
        train_losses, val_metrics = metrics['result']
        training_histories[model_name] = (train_losses, val_metrics)
        
        result = {
            'model': model_name,
            'task': task,
            'year': '2019-2020',
            'final_metric': val_metrics[-1] if val_metrics else 0,
            'parameters': count_parameters(model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'efficiency_gain': 'Parameter reduction and faster inference'
        }
        results.append(result)
    
    # Create efficiency comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        if train_losses:
            ax.plot(train_losses, label=model_name, linewidth=2)
    ax.set_title('Efficient Models Training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance metrics
    ax = axes[0, 1]
    for model_name, (_, val_metrics) in training_histories.items():
        if val_metrics:
            ax.plot(val_metrics, label=model_name, linewidth=2)
    ax.set_title('Validation Metrics', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric (Accuracy/Loss)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Efficiency comparison
    ax = axes[1, 0]
    model_names = [r['model'] for r in results]
    parameters = [r['parameters'] for r in results]
    
    # Compare with hypothetical full BERT
    full_bert_params = 110_000_000  # Approximate BERT-base parameters
    model_names.append('BERT-base')
    parameters.append(full_bert_params)
    
    bars = ax.bar(model_names, parameters, color=['#3498DB', '#E67E22', '#95A5A6'])
    ax.set_title('Parameter Count Comparison', fontsize=14)
    ax.set_ylabel('Parameters')
    ax.tick_params(axis='x', rotation=45)
    
    # Add efficiency labels
    for i, (bar, params) in enumerate(zip(bars[:-1], parameters[:-1])):
        efficiency = (1 - params / full_bert_params) * 100
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + full_bert_params * 0.02,
                f'{efficiency:.0f}% smaller', ha='center', va='bottom', fontsize=10)
    
    # Efficiency techniques
    ax = axes[1, 1]
    techniques = ['Knowledge\nDistillation', 'Parameter\nSharing', 'Factorized\nEmbeddings', 
                 'Optimized\nTraining']
    impact_scores = [9, 8, 7, 6]
    bars = ax.bar(techniques, impact_scores, color=['#E74C3C', '#9B59B6', '#1ABC9C', '#F39C12'])
    ax.set_title('Efficiency Techniques Impact', fontsize=14)
    ax.set_ylabel('Impact Score')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/017_efficient_transformers_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 017_efficient_transformers_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print(" EFFICIENT TRANSFORMERS SUMMARY ")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  📊 Task: {result['task']}")
        print(f"  📈 Final Metric: {result['final_metric']:.4f}")
        print(f"  🔢 Parameters: {result['parameters']:,}")
        print(f"  ⏱️  Training Time: {result['training_time']:.2f} minutes")
        print(f"  💾 Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"   Efficiency: {result['efficiency_gain']}")
    
    print("\n EFFICIENCY INNOVATIONS:")
    print("="*50)
    print("1. 🎓 KNOWLEDGE DISTILLATION (DistilBERT):")
    print("   • Train small student model to mimic large teacher")
    print("   • Soft targets from teacher + hard targets from data")
    print("   • 60% faster, 50% smaller, 95% performance retained")
    
    print("\n2. 🔄 PARAMETER SHARING (ALBERT):")
    print("   • Share parameters across all layers")
    print("   • Factorized embedding parameterization")
    print("   • 18x fewer parameters than BERT-large")
    
    print("\n3. 📏 FACTORIZED EMBEDDINGS (ALBERT):")
    print("   • Decompose large embedding matrix")
    print("   • vocab_size × embedding_size + embedding_size × hidden_size")
    print("   • Significant parameter reduction for large vocabularies")
    
    print("\n4. 🎯 OPTIMIZED TRAINING (RoBERTa):")
    print("   • Remove Next Sentence Prediction (NSP)")
    print("   • Dynamic masking instead of static")
    print("   • Larger batch sizes and learning rates")
    print("   • More training data and longer training")
    
    print("\n📊 EFFICIENCY COMPARISONS:")
    print("="*50)
    print("BERT-base (2018):     110M parameters, 100% performance")
    print("DistilBERT (2019):     66M parameters,  95% performance (40% reduction)")
    print("ALBERT-base (2019):    12M parameters,  97% performance (89% reduction)")
    print("RoBERTa-base (2019):  125M parameters, 104% performance (optimized)")
    print("MobileBERT (2020):     25M parameters,  92% performance (77% reduction)")
    
    print("\n SPEED IMPROVEMENTS:")
    print("="*50)
    print("• DistilBERT: 60% faster inference than BERT")
    print("• ALBERT: Parameter sharing reduces memory footprint")
    print("• RoBERTa: Better convergence, fewer training steps needed")
    print("• MobileBERT: Optimized for mobile/edge deployment")
    
    print("\n🧠 EFFICIENCY TECHNIQUES:")
    print("="*50)
    print("KNOWLEDGE DISTILLATION:")
    print("  Loss = α × KL(student || teacher) + (1-α) × CE(student, labels)")
    print("  • Temperature scaling for soft targets")
    print("  • Attention transfer from teacher to student")
    print("  • Hidden state matching losses")
    
    print("\nPARAMETER SHARING:")
    print("  • Universal Transformer concept")
    print("  • Same layer applied multiple times")
    print("  • Reduces parameters but maintains capacity")
    
    print("\nFACTORIZED EMBEDDINGS:")
    print("  • E = vocab_size × embedding_size")
    print("  • H = embedding_size × hidden_size")
    print("  • Total: E + H << vocab_size × hidden_size")
    
    print("\n🎯 DEPLOYMENT CONSIDERATIONS:")
    print("="*50)
    print("• 📱 MOBILE: MobileBERT for on-device inference")
    print("• ☁️  CLOUD: DistilBERT for cost-effective serving")
    print("• 🏭 EDGE: Quantized models for resource constraints")
    print("• ⚖️ TRADE-OFFS: Speed vs accuracy vs memory")
    
    print("\n🔬 TRAINING OPTIMIZATIONS:")
    print("="*50)
    print("RoBERTa Improvements:")
    print("• Remove NSP task (shown to be unnecessary)")
    print("• Dynamic masking (different masks each epoch)")
    print("• Larger batch sizes (8K vs 256)")
    print("• Byte-level BPE tokenization")
    print("• More data (160GB vs 16GB)")
    print("• Longer training (500K vs 100K steps)")
    
    print("\n💡 KEY INSIGHTS:")
    print("="*50)
    print("• Task-specific heads can be very lightweight")
    print("• Most BERT capacity is in hidden layers")
    print("• NSP pre-training task often unnecessary")
    print("• Dynamic masking better than static")
    print("• Parameter sharing maintains surprising capacity")
    print("• Knowledge distillation extremely effective")
    
    print("\n🌟 IMPACT ON INDUSTRY:")
    print("="*50)
    print("• Made transformer deployment economically viable")
    print("• Enabled real-time inference applications")
    print("• Reduced computational barriers for adoption")
    print("• Democratized access to BERT-level performance")
    print("• Established efficiency as key research direction")
    
    print("\n🔮 EFFICIENCY EVOLUTION:")
    print("="*50)
    print("2019: DistilBERT → First practical distillation")
    print("2019: ALBERT → Parameter sharing breakthrough")
    print("2019: RoBERTa → Training optimization insights")
    print("2020: MobileBERT → Mobile-specific optimizations")
    print("2020+: Sparse, quantized, and neural architecture search")
    
    print(f"\n{'='*70}")
    print(" MAKING TRANSFORMERS PRACTICAL FOR EVERYONE ")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()