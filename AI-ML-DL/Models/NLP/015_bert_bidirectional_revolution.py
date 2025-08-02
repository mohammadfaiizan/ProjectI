"""
ERA 5: BERT - BIDIRECTIONAL REVOLUTION (2018)
==============================================

Year: 2018
Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
Innovation: Bidirectional encoding + masked language modeling + pre-train/fine-tune
Previous Limitation: Unidirectional models (GPT), task-specific architectures
Performance Gain: 11 new SOTA results, universal pre-trained representations
Impact: Established pre-train/fine-tune paradigm, foundation for modern NLP

This file implements BERT that revolutionized NLP by introducing bidirectional
context encoding and the pre-training + fine-tuning paradigm that became
the standard approach for virtually all NLP tasks.
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

YEAR = "2018"
INNOVATION = "BERT: Bidirectional Encoder Representations from Transformers"
PREVIOUS_LIMITATION = "Unidirectional language models, task-specific architectures"
IMPACT = "Pre-train/fine-tune paradigm, 11 new SOTA results, universal representations"

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
                    if 5 <= len(tokens) <= 25:  # Good length for BERT
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
    """Build vocabulary with BERT special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 6)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<CLS>': 2,  # Classification token
        '<SEP>': 3,  # Separator token  
        '<MASK>': 4, # Mask token for MLM
        '<EOS>': 5   # End of sequence
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# BERT DATASET CLASSES
# ============================================================================

class BERTPretrainingDataset(Dataset):
    """
    Dataset for BERT pre-training with Masked Language Modeling (MLM)
    and Next Sentence Prediction (NSP)
    """
    
    def __init__(self, sentences, vocab, max_length=128, mlm_probability=0.15):
        self.vocab = vocab
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.sentences = sentences
        self.examples = []
        
        # Create sentence pairs for NSP
        self._create_pretraining_examples()
        
        print(f"Created {len(self.examples)} pre-training examples")
    
    def _create_pretraining_examples(self):
        """Create examples for MLM + NSP pre-training"""
        for i in range(len(self.sentences) - 1):
            sentence_a = self.sentences[i]
            
            # 50% chance of using next sentence (positive example)
            # 50% chance of using random sentence (negative example)
            if random.random() < 0.5:
                sentence_b = self.sentences[i + 1]
                is_next = 1
            else:
                sentence_b = self.sentences[random.randint(0, len(self.sentences) - 1)]
                is_next = 0
            
            # Combine sentences with special tokens
            # [CLS] sentence_a [SEP] sentence_b [SEP]
            tokens = ['<CLS>'] + sentence_a + ['<SEP>'] + sentence_b + ['<SEP>']
            
            # Truncate if too long
            if len(tokens) > self.max_length:
                # Keep CLS and truncate proportionally
                remaining = self.max_length - 3  # CLS + 2 SEP
                len_a = len(sentence_a)
                len_b = len(sentence_b)
                
                if len_a + len_b > remaining:
                    if len_a > remaining // 2:
                        len_a = remaining // 2
                    len_b = remaining - len_a
                    
                    sentence_a = sentence_a[:len_a]
                    sentence_b = sentence_b[:len_b]
                    
                tokens = ['<CLS>'] + sentence_a + ['<SEP>'] + sentence_b + ['<SEP>']
            
            # Create segment IDs (0 for sentence A, 1 for sentence B)
            segment_ids = ([0] * (len(sentence_a) + 2) +  # CLS + sentence_a + SEP
                          [1] * (len(sentence_b) + 1))     # sentence_b + SEP
            
            self.examples.append({
                'tokens': tokens,
                'segment_ids': segment_ids,
                'is_next': is_next
            })
    
    def _apply_masking(self, tokens):
        """Apply masked language modeling to tokens"""
        masked_tokens = tokens[:]
        mlm_labels = [-100] * len(tokens)  # -100 will be ignored in loss
        
        for i, token in enumerate(tokens):
            # Don't mask special tokens
            if token in ['<CLS>', '<SEP>', '<PAD>']:
                continue
                
            if random.random() < self.mlm_probability:
                mlm_labels[i] = self.vocab.get(token, self.vocab['<UNK>'])
                
                # 80% of time, replace with [MASK]
                if random.random() < 0.8:
                    masked_tokens[i] = '<MASK>'
                # 10% of time, replace with random token
                elif random.random() < 0.5:
                    random_token = random.choice(list(self.vocab.keys()))
                    masked_tokens[i] = random_token
                # 10% of time, keep original token
                # (already in masked_tokens)
        
        return masked_tokens, mlm_labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Apply masking
        masked_tokens, mlm_labels = self._apply_masking(example['tokens'])
        
        # Convert to indices
        input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in masked_tokens]
        mlm_labels_ids = mlm_labels
        segment_ids = example['segment_ids']
        is_next = example['is_next']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels_ids, dtype=torch.long),
            'nsp_labels': torch.tensor(is_next, dtype=torch.long)
        }

class BERTClassificationDataset(Dataset):
    """Dataset for BERT fine-tuning on classification tasks"""
    
    def __init__(self, sentences, vocab, task='sentiment', max_length=128):
        self.vocab = vocab
        self.max_length = max_length
        self.examples = []
        
        # Create classification examples
        for sentence in sentences:
            # Simple heuristic for sentiment classification
            if task == 'sentiment':
                # Positive if contains positive words, negative otherwise
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worse', 'worst']
                
                pos_count = sum(1 for word in sentence if word in positive_words)
                neg_count = sum(1 for word in sentence if word in negative_words)
                
                if pos_count > neg_count:
                    label = 1  # Positive
                elif neg_count > pos_count:
                    label = 0  # Negative
                else:
                    label = random.randint(0, 1)  # Random if neutral
            
            # Create input sequence: [CLS] sentence [SEP]
            tokens = ['<CLS>'] + sentence[:self.max_length-2] + ['<SEP>']
            
            self.examples.append({
                'tokens': tokens,
                'label': label
            })
        
        print(f"Created {len(self.examples)} classification examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to indices
        input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['tokens']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(example['label'], dtype=torch.long)
        }

def collate_bert_pretraining(batch):
    """Collate function for BERT pre-training"""
    input_ids = [item['input_ids'] for item in batch]
    segment_ids = [item['segment_ids'] for item in batch]
    mlm_labels = [item['mlm_labels'] for item in batch]
    nsp_labels = [item['nsp_labels'] for item in batch]
    
    # Pad sequences
    max_len = max(len(seq) for seq in input_ids)
    
    padded_input_ids = []
    padded_segment_ids = []
    padded_mlm_labels = []
    attention_masks = []
    
    for i in range(len(batch)):
        seq_len = len(input_ids[i])
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input = torch.cat([input_ids[i], torch.zeros(pad_len, dtype=torch.long)])
        padded_input_ids.append(padded_input)
        
        # Pad segment_ids
        padded_segment = torch.cat([segment_ids[i], torch.zeros(pad_len, dtype=torch.long)])
        padded_segment_ids.append(padded_segment)
        
        # Pad MLM labels
        padded_mlm = torch.cat([mlm_labels[i], torch.full((pad_len,), -100, dtype=torch.long)])
        padded_mlm_labels.append(padded_mlm)
        
        # Create attention mask
        attention_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        attention_masks.append(attention_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks),
        'segment_ids': torch.stack(padded_segment_ids),
        'mlm_labels': torch.stack(padded_mlm_labels),
        'nsp_labels': torch.stack(nsp_labels)
    }

def collate_bert_classification(batch):
    """Collate function for BERT classification"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    max_len = max(len(seq) for seq in input_ids)
    
    padded_input_ids = []
    attention_masks = []
    
    for seq in input_ids:
        seq_len = len(seq)
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        padded_input_ids.append(padded_input)
        
        # Create attention mask
        attention_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        attention_masks.append(attention_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }

# ============================================================================
# BERT EMBEDDINGS
# ============================================================================

class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings: Token + Segment + Position embeddings
    """
    
    def __init__(self, vocab_size, hidden_size=768, max_position_embeddings=512,
                 type_vocab_size=2, dropout=0.1):
        super(BERTEmbeddings, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize position ids
        self.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length) segment IDs
            
        Returns:
            embeddings: (batch_size, seq_length, hidden_size)
        """
        seq_length = input_ids.size(1)
        
        # Token embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Position embeddings
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        
        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Sum all embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# ============================================================================
# BERT SELF-ATTENTION
# ============================================================================

class BERTSelfAttention(nn.Module):
    """BERT Self-Attention mechanism"""
    
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1):
        super(BERTSelfAttention, self).__init__()
        
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        """Transpose tensor for multi-head attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length) 
            
        Returns:
            context_layer: (batch_size, seq_length, hidden_size)
            attention_probs: (batch_size, num_heads, seq_length, seq_length)
        """
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert attention mask to attention bias
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

# ============================================================================
# BERT LAYER
# ============================================================================

class BERTLayer(nn.Module):
    """Single BERT transformer layer"""
    
    def __init__(self, hidden_size=768, num_attention_heads=12, 
                 intermediate_size=3072, dropout=0.1):
        super(BERTLayer, self).__init__()
        
        self.attention = BERTSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length)
            
        Returns:
            layer_output: (batch_size, seq_length, hidden_size)
            attention_probs: attention weights
        """
        # Self-attention
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        
        # Add & Norm
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        
        # Add & Norm
        layer_output = self.output_layer_norm(attention_output + layer_output)
        
        return layer_output, attention_probs

# ============================================================================
# BERT ENCODER
# ============================================================================

class BERTEncoder(nn.Module):
    """BERT encoder with multiple transformer layers"""
    
    def __init__(self, hidden_size=768, num_attention_heads=12, 
                 num_hidden_layers=12, intermediate_size=3072, dropout=0.1):
        super(BERTEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length)
            
        Returns:
            sequence_output: (batch_size, seq_length, hidden_size)
            all_attention_probs: list of attention weights from each layer
        """
        all_attention_probs = []
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            all_attention_probs.append(attention_probs)
        
        return hidden_states, all_attention_probs

# ============================================================================
# BERT POOLER
# ============================================================================

class BERTPooler(nn.Module):
    """BERT pooler for extracting [CLS] representation"""
    
    def __init__(self, hidden_size=768):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        """
        Extract and transform the [CLS] token representation
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = torch.tanh(pooled_output)
        
        return pooled_output

# ============================================================================
# BERT MODEL
# ============================================================================

class BERTModel(nn.Module):
    """
    Complete BERT model for pre-training and fine-tuning
    """
    
    def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12,
                 num_hidden_layers=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout=0.1):
        super(BERTModel, self).__init__()
        
        self.embeddings = BERTEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout
        )
        self.encoder = BERTEncoder(
            hidden_size, num_attention_heads, num_hidden_layers, intermediate_size, dropout
        )
        self.pooler = BERTPooler(hidden_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            
        Returns:
            sequence_output: (batch_size, seq_length, hidden_size)
            pooled_output: (batch_size, hidden_size)
            all_attention_probs: list of attention weights
        """
        # Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Encoder
        sequence_output, all_attention_probs = self.encoder(embedding_output, attention_mask)
        
        # Pooler
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output, all_attention_probs

# ============================================================================
# BERT FOR PRE-TRAINING
# ============================================================================

class BERTForPreTraining(nn.Module):
    """BERT model for pre-training with MLM and NSP heads"""
    
    def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12,
                 num_hidden_layers=12, intermediate_size=3072, dropout=0.1):
        super(BERTForPreTraining, self).__init__()
        
        self.bert = BERTModel(
            vocab_size, hidden_size, num_attention_heads, 
            num_hidden_layers, intermediate_size, dropout=dropout
        )
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # NSP head
        self.nsp_head = nn.Linear(hidden_size, 2)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                mlm_labels=None, nsp_labels=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            mlm_labels: (batch_size, seq_length) MLM labels
            nsp_labels: (batch_size,) NSP labels
            
        Returns:
            loss: total pre-training loss (if labels provided)
            mlm_logits: (batch_size, seq_length, vocab_size)
            nsp_logits: (batch_size, 2)
        """
        # BERT forward pass
        sequence_output, pooled_output, attention_probs = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        
        # MLM head
        mlm_logits = self.mlm_head(sequence_output)
        
        # NSP head
        nsp_logits = self.nsp_head(pooled_output)
        
        # Compute losses if labels provided
        total_loss = None
        if mlm_labels is not None and nsp_labels is not None:
            # MLM loss
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            
            # NSP loss
            nsp_loss_fn = nn.CrossEntropyLoss()
            nsp_loss = nsp_loss_fn(nsp_logits, nsp_labels)
            
            total_loss = mlm_loss + nsp_loss
        
        return total_loss, mlm_logits, nsp_logits, attention_probs

# ============================================================================
# BERT FOR CLASSIFICATION
# ============================================================================

class BERTForSequenceClassification(nn.Module):
    """BERT model for sequence classification tasks"""
    
    def __init__(self, bert_model, num_labels=2, dropout=0.1):
        super(BERTForSequenceClassification, self).__init__()
        
        self.bert = bert_model.bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.bert.pooler.dense.out_features, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            labels: (batch_size,) classification labels
            
        Returns:
            loss: classification loss (if labels provided)
            logits: (batch_size, num_labels)
        """
        # BERT forward pass
        sequence_output, pooled_output, attention_probs = self.bert(input_ids, attention_mask)
        
        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return loss, logits, attention_probs

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_bert_pretraining(model, train_loader, val_loader, epochs=5, learning_rate=5e-5):
    """Train BERT for pre-training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    train_losses = []
    val_losses = []
    
    print(f"Training BERT pre-training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            nsp_labels = batch['nsp_labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, mlm_logits, nsp_logits, _ = model(
                input_ids, attention_mask, segment_ids, mlm_labels, nsp_labels
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = evaluate_bert_pretraining(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_bert_pretraining(model, data_loader, device):
    """Evaluate BERT pre-training"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            nsp_labels = batch['nsp_labels'].to(device)
            
            loss, _, _, _ = model(
                input_ids, attention_mask, segment_ids, mlm_labels, nsp_labels
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_bert_classification(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    """Fine-tune BERT for classification"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Fine-tuning BERT classification on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, logits, _ = model(input_ids, attention_mask, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_accuracy = evaluate_bert_classification(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return train_losses, val_accuracies

def evaluate_bert_classification(model, data_loader, device):
    """Evaluate BERT classification accuracy"""
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
    print(f"=== BERT Bidirectional Revolution ({YEAR}) ===")
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
    
    # Test BERT Pre-training
    print("\n" + "="*50)
    print("BERT PRE-TRAINING (MLM + NSP)")
    
    # Create pre-training dataset
    pretrain_dataset = BERTPretrainingDataset(train_subset, vocab, max_length=64)
    pretrain_val_dataset = BERTPretrainingDataset(val_subset, vocab, max_length=64)
    
    # Create data loaders
    batch_size = 8  # Small batch size for demo
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_bert_pretraining)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_bert_pretraining)
    
    # Initialize BERT for pre-training
    bert_pretrain = BERTForPreTraining(
        vocab_size=len(vocab),
        hidden_size=256,       # Reduced for demo
        num_attention_heads=8,
        num_hidden_layers=4,   # Reduced for demo
        intermediate_size=512, # Reduced for demo
        dropout=0.1
    )
    
    print(f"BERT pre-training parameters: {count_parameters(bert_pretrain):,}")
    
    # Train BERT pre-training
    metrics = track_computational_metrics(
        'BERT-PreTraining',
        train_bert_pretraining,
        bert_pretrain, pretrain_loader, pretrain_val_loader, 3, 5e-5
    )
    
    train_losses, val_losses = metrics['result']
    training_histories['BERT-PreTraining'] = (train_losses, val_losses)
    
    result1 = {
        'model': 'BERT-PreTraining',
        'stage': 'pre-training',
        'year': '2018',
        'final_loss': val_losses[-1] if val_losses else 0,
        'parameters': count_parameters(bert_pretrain),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Bidirectional encoding + MLM + NSP'
    }
    results.append(result1)
    
    # Test BERT Fine-tuning
    print("\n" + "="*50)
    print("BERT FINE-TUNING (Classification)")
    
    # Create classification dataset
    class_dataset = BERTClassificationDataset(train_subset, vocab, task='sentiment')
    class_val_dataset = BERTClassificationDataset(val_subset, vocab, task='sentiment')
    
    class_loader = DataLoader(class_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_bert_classification)
    class_val_loader = DataLoader(class_val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_bert_classification)
    
    # Initialize BERT for classification (using pre-trained weights)
    bert_classifier = BERTForSequenceClassification(bert_pretrain, num_labels=2)
    
    # Fine-tune BERT
    metrics = track_computational_metrics(
        'BERT-Classification',
        train_bert_classification,
        bert_classifier, class_loader, class_val_loader, 3, 2e-5
    )
    
    train_losses, val_accuracies = metrics['result']
    training_histories['BERT-Classification'] = (train_losses, val_accuracies)
    
    result2 = {
        'model': 'BERT-Classification',
        'stage': 'fine-tuning',
        'year': '2018',
        'final_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'parameters': count_parameters(bert_classifier),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Pre-train + fine-tune paradigm'
    }
    results.append(result2)
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pre-training loss
    ax = axes[0, 0]
    if 'BERT-PreTraining' in training_histories:
        pretrain_losses, _ = training_histories['BERT-PreTraining']
        ax.plot(pretrain_losses, label='Pre-training Loss', linewidth=2, color='#E74C3C')
    ax.set_title('BERT Pre-training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fine-tuning accuracy
    ax = axes[0, 1]
    if 'BERT-Classification' in training_histories:
        _, ft_accuracies = training_histories['BERT-Classification']
        ax.plot(ft_accuracies, label='Fine-tuning Accuracy', linewidth=2, color='#27AE60')
    ax.set_title('BERT Fine-tuning Accuracy', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # BERT innovations
    ax = axes[1, 0]
    innovations = ['Bidirectional\nEncoding', 'Masked LM', 'Next Sentence\nPrediction', 
                  'Pre-train +\nFine-tune']
    impact_scores = [10, 9, 6, 10]
    bars = ax.bar(innovations, impact_scores, 
                  color=['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C'])
    ax.set_title('BERT Revolutionary Innovations', fontsize=14)
    ax.set_ylabel('Impact Score')
    ax.tick_params(axis='x', rotation=45)
    
    # Before vs After BERT
    ax = axes[1, 1]
    eras = ['Pre-BERT\n(2017)', 'Post-BERT\n(2019)']
    sota_tasks = [3, 11]  # Number of SOTA results
    bars = ax.bar(eras, sota_tasks, color=['#95A5A6', '#F39C12'])
    ax.set_title('BERT Impact on NLP Tasks', fontsize=14)
    ax.set_ylabel('Number of SOTA Results')
    
    # Add value labels
    for bar, value in zip(bars, sota_tasks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/015_bert_revolution_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 015_bert_revolution_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("🚀 BERT BIDIRECTIONAL REVOLUTION SUMMARY 🚀")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}) - {result['stage'].upper()}:")
        if 'final_accuracy' in result:
            print(f"  🎯 Final Accuracy: {result['final_accuracy']:.4f}")
        if 'final_loss' in result:
            print(f"  📉 Final Loss: {result['final_loss']:.4f}")
        print(f"  🔢 Parameters: {result['parameters']:,}")
        print(f"  ⏱️  Training Time: {result['training_time']:.2f} minutes")
        print(f"  💾 Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  💡 Innovation: {result['innovation']}")
    
    print("\n🧠 BERT REVOLUTIONARY CONCEPTS:")
    print("="*50)
    print("1. 🔄 BIDIRECTIONAL ENCODING:")
    print("   • Uses context from both left AND right")
    print("   • Deeper understanding than unidirectional models")
    print("   • Enabled by masked language modeling")
    
    print("\n2. 🎭 MASKED LANGUAGE MODELING (MLM):")
    print("   • Randomly mask 15% of input tokens")
    print("   • Predict masked tokens using bidirectional context")
    print("   • Forces model to learn rich representations")
    
    print("\n3. 📝 NEXT SENTENCE PREDICTION (NSP):")
    print("   • Predict if two sentences are consecutive")
    print("   • Learns sentence-level relationships")
    print("   • Important for tasks like QA and NLI")
    
    print("\n4. 🔄 PRE-TRAIN + FINE-TUNE PARADIGM:")
    print("   • Pre-train on large unlabeled corpus")
    print("   • Fine-tune on specific downstream tasks")
    print("   • Universal representations for all NLP tasks")
    
    print("\n⚡ TECHNICAL INNOVATIONS:")
    print("="*50)
    print("• WordPiece tokenization for handling OOV words")
    print("• [CLS] token for classification tasks")
    print("• [SEP] token for separating sentences")
    print("• Segment embeddings for multi-sentence input")
    print("• GELU activation instead of ReLU")
    print("• Layer normalization with specific placement")
    
    print("\n📊 BERT'S REVOLUTIONARY RESULTS:")
    print("="*50)
    print("• 📈 GLUE: 7.7% improvement over previous SOTA")
    print("• 🎯 SQuAD v1.1: Human-level performance (F1: 88.5→93.2)")
    print("• 📚 MultiNLI: 4.6% accuracy improvement")
    print("• 🔍 Named Entity Recognition: New SOTA")
    print("• ❓ Question Answering: Dramatic improvements")
    print("• 📝 Text Classification: Universal improvements")
    
    print("\n🌟 WHY BERT WAS REVOLUTIONARY:")
    print("="*50)
    print("1. 🎯 BIDIRECTIONAL CONTEXT: First truly bidirectional model")
    print("2. 🌐 UNIVERSAL ARCHITECTURE: One model for all NLP tasks")
    print("3. 📈 MASSIVE IMPROVEMENTS: 11 new SOTA results")
    print("4. 🔄 NEW PARADIGM: Established pre-train/fine-tune standard")
    print("5. 🚀 SCALABILITY: Showed importance of model size")
    print("6. 💡 SIMPLICITY: Elegant solution to complex problems")
    
    print("\n🔬 TECHNICAL COMPARISON:")
    print("="*50)
    print("GPT (Unidirectional):     Token₍ᵢ₎ = f(Token₍₁₎...Token₍ᵢ₋₁₎)")
    print("BERT (Bidirectional):     Token₍ᵢ₎ = f(Token₍₁₎...Token₍ᵢ₋₁₎, Token₍ᵢ₊₁₎...Token₍ₙ₎)")
    print("")
    print("ELMo: Shallow bidirectional (concat of forward/backward)")
    print("BERT: Deep bidirectional (joint conditioning)")
    
    print("\n🌈 BEFORE vs AFTER BERT:")
    print("="*50)
    print("BEFORE BERT (2017):")
    print("  • Task-specific architectures")
    print("  • Limited transfer learning")
    print("  • Unidirectional or shallow bidirectional")
    print("  • Feature-based approaches")
    print("  • Modest improvements on benchmarks")
    
    print("\nAFTER BERT (2018+):")
    print("  • Universal pre-trained models")
    print("  • Massive transfer learning")
    print("  • Deep bidirectional representations")
    print("  • Fine-tuning approaches")
    print("  • Revolutionary benchmark improvements")
    
    print("\n🎓 BERT'S LASTING IMPACT:")
    print("="*50)
    print("• 📖 Most influential NLP paper after Transformer")
    print("• 🏗️  Template for all future language models")
    print("• 🔄 Established pre-train/fine-tune as standard")
    print("• 🎯 Showed importance of bidirectional context")
    print("• 🚀 Enabled the era of large language models")
    print("• 🌐 Made NLP accessible to non-experts")
    
    print("\n🚀 LINEAGE OF INFLUENCE:")
    print("="*50)
    print("BERT → RoBERTa → DeBERTa → ELECTRA")
    print("BERT → DistilBERT → ALBERT → MobileBERT")
    print("BERT + GPT → T5 → BART → UnifiedQA")
    print("BERT concepts → Modern LLMs architecture")
    
    print("\n💡 KEY LESSONS FROM BERT:")
    print("="*50)
    print("• Bidirectional context is crucial for understanding")
    print("• Pre-training on large corpora creates powerful representations")
    print("• Simple objectives (MLM, NSP) can drive complex learning")
    print("• Transfer learning is more powerful than task-specific training")
    print("• Scale and compute can lead to qualitative improvements")
    print("• Elegant architectures often outperform complex ones")
    
    print(f"\n{'='*70}")
    print("🏆 BERT: THE MODEL THAT CHANGED EVERYTHING 🏆")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()