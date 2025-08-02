"""
ERA 6: T5 - TEXT-TO-TEXT TRANSFER TRANSFORMER (2019)
====================================================

Year: 2019
Paper: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
Innovation: Unified text-to-text framework for all NLP tasks
Previous Limitation: Task-specific architectures and different training procedures
Performance Gain: Single model for all tasks, state-of-the-art across multiple benchmarks
Impact: Established text-to-text as universal NLP paradigm, influenced GPT-3 and modern LLMs

This file implements T5 that revolutionized NLP by treating every task as text-to-text,
unifying classification, generation, translation, and more under a single framework.
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

YEAR = "2019"
INNOVATION = "T5: Text-to-Text Transfer Transformer - Everything is text-to-text"
PREVIOUS_LIMITATION = "Task-specific architectures, different training procedures for each task"
IMPACT = "Universal text-to-text paradigm, single model for all NLP tasks"

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
                    if 5 <= len(tokens) <= 20:  # Good length for T5 tasks
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
    """Build vocabulary with T5 special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 10)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<EOS>': 2,
        '</s>': 3,    # T5 end token
        '<s>': 4,     # T5 start token
        '<extra_id_0>': 5,  # T5 sentinel tokens for span corruption
        '<extra_id_1>': 6,
        '<extra_id_2>': 7,
        '<extra_id_3>': 8,
        '<extra_id_4>': 9
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# T5 DATASET CLASS
# ============================================================================

class T5Dataset(Dataset):
    """
    Dataset for T5 text-to-text tasks
    Converts various NLP tasks into text-to-text format
    """
    
    def __init__(self, sentences, vocab, task='span_corruption', max_input_length=64, max_target_length=32):
        self.vocab = vocab
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.task = task
        self.examples = []
        
        if task == 'span_corruption':
            self._create_span_corruption_examples(sentences)
        elif task == 'classification':
            self._create_classification_examples(sentences)
        elif task == 'summarization':
            self._create_summarization_examples(sentences)
        elif task == 'translation':
            self._create_translation_examples(sentences)
        elif task == 'question_answering':
            self._create_qa_examples(sentences)
        
        print(f"Created {len(self.examples)} T5 {task} examples")
    
    def _create_span_corruption_examples(self, sentences):
        """
        Create span corruption examples (T5 pre-training task)
        Corrupts spans of text and asks model to reconstruct them
        """
        for sentence in sentences:
            if len(sentence) >= 5:
                # Corrupt approximately 15% of tokens
                tokens = sentence[:]
                num_corruptions = max(1, int(0.15 * len(tokens)))
                
                # Identify corruption spans
                corrupted_tokens = []
                target_tokens = []
                sentinel_id = 0
                i = 0
                
                corruption_positions = sorted(random.sample(range(len(tokens)), 
                                                          min(num_corruptions, len(tokens))))
                
                for pos in corruption_positions:
                    if i <= pos:
                        # Add tokens before corruption
                        corrupted_tokens.extend(tokens[i:pos])
                        
                        # Add sentinel token
                        sentinel_token = f'<extra_id_{sentinel_id}>'
                        corrupted_tokens.append(sentinel_token)
                        
                        # Add to target
                        target_tokens.append(sentinel_token)
                        
                        # Determine span length (1-3 tokens)
                        span_length = random.randint(1, min(3, len(tokens) - pos))
                        corrupted_span = tokens[pos:pos + span_length]
                        target_tokens.extend(corrupted_span)
                        
                        i = pos + span_length
                        sentinel_id += 1
                
                # Add remaining tokens
                if i < len(tokens):
                    corrupted_tokens.extend(tokens[i:])
                
                # Add final EOS to target
                target_tokens.append('</s>')
                
                if len(corrupted_tokens) <= self.max_input_length and len(target_tokens) <= self.max_target_length:
                    self.examples.append({
                        'input_text': corrupted_tokens,
                        'target_text': target_tokens
                    })
    
    def _create_classification_examples(self, sentences):
        """Create classification examples in text-to-text format"""
        for sentence in sentences:
            if len(sentence) <= self.max_input_length - 2:
                # Create sentiment classification task
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worse', 'worst']
                
                pos_count = sum(1 for word in sentence if word in positive_words)
                neg_count = sum(1 for word in sentence if word in negative_words)
                
                if pos_count > neg_count:
                    label = 'positive'
                elif neg_count > pos_count:
                    label = 'negative'
                else:
                    label = random.choice(['positive', 'negative'])
                
                # T5 format: "sentiment: [sentence]" -> "positive/negative"
                input_text = ['sentiment:'] + sentence
                target_text = [label, '</s>']
                
                self.examples.append({
                    'input_text': input_text,
                    'target_text': target_text
                })
    
    def _create_summarization_examples(self, sentences):
        """Create summarization examples"""
        for sentence in sentences:
            if len(sentence) >= 6:
                # Extract key words as summary (simplified)
                summary_length = min(3, len(sentence) // 2)
                
                # Take first word, longest word, and last word
                summary = [sentence[0]]
                if len(sentence) > 2:
                    longest = max(sentence[1:-1], key=len, default='')
                    if longest:
                        summary.append(longest)
                summary.append(sentence[-1])
                
                summary = summary[:summary_length]
                
                # T5 format: "summarize: [sentence]" -> "[summary]"
                input_text = ['summarize:'] + sentence
                target_text = summary + ['</s>']
                
                if len(input_text) <= self.max_input_length and len(target_text) <= self.max_target_length:
                    self.examples.append({
                        'input_text': input_text,
                        'target_text': target_text
                    })
    
    def _create_translation_examples(self, sentences):
        """Create simple translation examples"""
        # Simple word-level translation rules
        translation_dict = {
            'the': 'le', 'a': 'un', 'is': 'est', 'and': 'et',
            'good': 'bon', 'bad': 'mauvais', 'cat': 'chat', 'dog': 'chien'
        }
        
        for sentence in sentences:
            if len(sentence) <= self.max_input_length - 3:
                # Apply simple translation
                translated = []
                for word in sentence:
                    translated.append(translation_dict.get(word, word))
                
                # T5 format: "translate English to French: [sentence]" -> "[translation]"
                input_text = ['translate', 'english', 'to', 'french:'] + sentence
                target_text = translated + ['</s>']
                
                if len(input_text) <= self.max_input_length and len(target_text) <= self.max_target_length:
                    self.examples.append({
                        'input_text': input_text,
                        'target_text': target_text
                    })
    
    def _create_qa_examples(self, sentences):
        """Create question answering examples"""
        for sentence in sentences:
            if len(sentence) >= 4:
                # Create simple QA pairs
                # Question: "What is mentioned?" Answer: [key word]
                key_word = max(sentence, key=len)  # Longest word as answer
                
                question = ['question:', 'what', 'is', 'mentioned?', 'context:'] + sentence
                answer = [key_word, '</s>']
                
                if len(question) <= self.max_input_length and len(answer) <= self.max_target_length:
                    self.examples.append({
                        'input_text': question,
                        'target_text': answer
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to indices
        input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['input_text']]
        target_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in example['target_text']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def collate_t5_fn(batch):
    """Collate function for T5 data"""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    # Pad sequences
    max_input_len = max(len(seq) for seq in input_ids)
    max_target_len = max(len(seq) for seq in target_ids)
    
    padded_input_ids = []
    padded_target_ids = []
    input_attention_masks = []
    target_attention_masks = []
    
    for input_seq, target_seq in zip(input_ids, target_ids):
        # Pad input
        input_pad_len = max_input_len - len(input_seq)
        padded_input = torch.cat([input_seq, torch.zeros(input_pad_len, dtype=torch.long)])
        padded_input_ids.append(padded_input)
        
        # Input attention mask
        input_attention_mask = torch.cat([torch.ones(len(input_seq)), torch.zeros(input_pad_len)])
        input_attention_masks.append(input_attention_mask)
        
        # Pad target
        target_pad_len = max_target_len - len(target_seq)
        padded_target = torch.cat([target_seq, torch.zeros(target_pad_len, dtype=torch.long)])
        padded_target_ids.append(padded_target)
        
        # Target attention mask
        target_attention_mask = torch.cat([torch.ones(len(target_seq)), torch.zeros(target_pad_len)])
        target_attention_masks.append(target_attention_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'input_attention_mask': torch.stack(input_attention_masks),
        'target_ids': torch.stack(padded_target_ids),
        'target_attention_mask': torch.stack(target_attention_masks)
    }

# ============================================================================
# T5 RELATIVE POSITION ENCODING
# ============================================================================

class T5RelativePositionBias(nn.Module):
    """
    T5 relative position bias
    Adds learned biases based on relative positions
    """
    
    def __init__(self, num_heads, relative_attention_num_buckets=32, max_distance=128):
        super(T5RelativePositionBias, self).__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
    
    def _relative_position_bucket(self, relative_position):
        """Convert relative position to bucket index"""
        ret = 0
        n = -relative_position
        
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.max_distance
        
        # Half buckets for exact distances
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        
        # Small distances use exact buckets
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        # Large distances use logarithmic buckets
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, query_length, key_length):
        """Compute relative position bias"""
        device = self.relative_attention_bias.weight.device
        
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return values

# ============================================================================
# T5 ATTENTION
# ============================================================================

class T5Attention(nn.Module):
    """T5 multi-head attention with relative position bias"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1, has_relative_bias=False):
        super(T5Attention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.has_relative_bias = has_relative_bias
        
        # Linear projections
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Relative position bias
        if has_relative_bias:
            self.relative_attention_bias = T5RelativePositionBias(num_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attention_mask=None, key_value_states=None):
        """
        T5 attention forward pass
        
        Args:
            query: (batch_size, query_length, hidden_size)
            key: (batch_size, key_length, hidden_size) 
            value: (batch_size, key_length, hidden_size)
            attention_mask: (batch_size, key_length)
            key_value_states: For cross-attention
        """
        batch_size, query_length, _ = query.size()
        
        if key_value_states is not None:
            # Cross-attention (decoder attending to encoder)
            key_length = key_value_states.size(1)
            k_states = self.k(key_value_states)
            v_states = self.v(key_value_states)
        else:
            # Self-attention
            key_length = key.size(1)
            k_states = self.k(key)
            v_states = self.v(value)
        
        q_states = self.q(query)
        
        # Reshape for multi-head attention
        q_states = q_states.view(batch_size, query_length, self.num_heads, self.head_size).transpose(1, 2)
        k_states = k_states.view(batch_size, key_length, self.num_heads, self.head_size).transpose(1, 2)
        v_states = v_states.view(batch_size, key_length, self.num_heads, self.head_size).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q_states, k_states.transpose(-1, -2))
        
        # Add relative position bias
        if self.has_relative_bias:
            position_bias = self.relative_attention_bias(query_length, key_length)
            scores += position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores += (1.0 - attention_mask) * -10000.0
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v_states)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, query_length, self.hidden_size)
        output = self.o(context)
        
        return output, attention_weights

# ============================================================================
# T5 LAYER
# ============================================================================

class T5Layer(nn.Module):
    """T5 transformer layer (encoder or decoder)"""
    
    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.1, is_decoder=False):
        super(T5Layer, self).__init__()
        
        self.is_decoder = is_decoder
        
        # Self-attention
        self.self_attention = T5Attention(hidden_size, num_heads, dropout, has_relative_bias=True)
        self.self_attention_layer_norm = nn.LayerNorm(hidden_size)
        
        # Cross-attention (only for decoder)
        if is_decoder:
            self.cross_attention = T5Attention(hidden_size, num_heads, dropout, has_relative_bias=False)
            self.cross_attention_layer_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.feed_forward_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        """
        T5 layer forward pass
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Self-attention mask
            encoder_hidden_states: For cross-attention (decoder only)
            encoder_attention_mask: Encoder attention mask (decoder only)
        """
        # Self-attention
        attention_output, self_attention_weights = self.self_attention(
            hidden_states, hidden_states, hidden_states, attention_mask
        )
        hidden_states = self.self_attention_layer_norm(hidden_states + self.dropout(attention_output))
        
        cross_attention_weights = None
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_output, cross_attention_weights = self.cross_attention(
                hidden_states, encoder_hidden_states, encoder_hidden_states, 
                encoder_attention_mask, encoder_hidden_states
            )
            hidden_states = self.cross_attention_layer_norm(hidden_states + self.dropout(cross_attention_output))
        
        # Feed-forward
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(hidden_states + feed_forward_output)
        
        return hidden_states, self_attention_weights, cross_attention_weights

# ============================================================================
# T5 MODEL
# ============================================================================

class T5Model(nn.Module):
    """
    T5: Text-to-Text Transfer Transformer
    Complete encoder-decoder model with relative position biases
    """
    
    def __init__(self, vocab_size, hidden_size=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, ff_size=1024, max_length=512, dropout=0.1):
        super(T5Model, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Shared embedding layer
        self.shared_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            T5Layer(hidden_size, num_heads, ff_size, dropout, is_decoder=False)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_final_layer_norm = nn.LayerNorm(hidden_size)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            T5Layer(hidden_size, num_heads, ff_size, dropout, is_decoder=True)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_final_layer_norm = nn.LayerNorm(hidden_size)
        
        # Output head (tied with embedding weights)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def create_causal_mask(self, seq_length, device):
        """Create causal mask for decoder"""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def encode(self, input_ids, attention_mask=None):
        """Encode input sequence"""
        hidden_states = self.shared_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        for layer in self.encoder_layers:
            hidden_states, _, _ = layer(hidden_states, attention_mask)
        
        hidden_states = self.encoder_final_layer_norm(hidden_states)
        return hidden_states
    
    def decode(self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        """Decode with cross-attention to encoder"""
        seq_length = input_ids.size(1)
        device = input_ids.device
        
        hidden_states = self.shared_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal mask for decoder
        causal_mask = self.create_causal_mask(seq_length, device)
        if attention_mask is not None:
            # Combine causal mask with padding mask
            combined_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
        else:
            combined_mask = causal_mask
        
        all_self_attention_weights = []
        all_cross_attention_weights = []
        
        for layer in self.decoder_layers:
            hidden_states, self_attn_weights, cross_attn_weights = layer(
                hidden_states, combined_mask, encoder_hidden_states, encoder_attention_mask
            )
            all_self_attention_weights.append(self_attn_weights)
            all_cross_attention_weights.append(cross_attn_weights)
        
        hidden_states = self.decoder_final_layer_norm(hidden_states)
        
        return hidden_states, all_self_attention_weights, all_cross_attention_weights
    
    def forward(self, input_ids, input_attention_mask=None, target_ids=None, target_attention_mask=None):
        """
        T5 forward pass
        
        Args:
            input_ids: (batch_size, input_length) encoder inputs
            input_attention_mask: (batch_size, input_length) encoder attention mask
            target_ids: (batch_size, target_length) decoder inputs
            target_attention_mask: (batch_size, target_length) decoder attention mask
        
        Returns:
            logits: (batch_size, target_length, vocab_size)
            loss: if target_ids provided
        """
        # Encode
        encoder_hidden_states = self.encode(input_ids, input_attention_mask)
        
        if target_ids is not None:
            # Decode (teacher forcing)
            decoder_input_ids = target_ids[:, :-1]  # Shift right
            decoder_targets = target_ids[:, 1:]     # Shift left
            
            decoder_hidden_states, self_attention_weights, cross_attention_weights = self.decode(
                decoder_input_ids, encoder_hidden_states, None, input_attention_mask
            )
            
            # Generate logits
            logits = self.lm_head(decoder_hidden_states)
            
            # Compute loss
            loss = None
            if decoder_targets is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fn(logits.view(-1, logits.size(-1)), decoder_targets.view(-1))
            
            return loss, logits, self_attention_weights, cross_attention_weights
        else:
            return encoder_hidden_states
    
    def generate(self, input_ids, input_attention_mask, vocab, idx_to_word, max_length=32, temperature=1.0):
        """Generate text using the trained T5 model"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Encode input
        encoder_hidden_states = self.encode(input_ids, input_attention_mask)
        
        # Initialize decoder with start token
        decoder_input = torch.tensor([[vocab.get('<s>', vocab['<EOS>'])]] * batch_size, device=device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Decode one step
                decoder_hidden_states, _, _ = self.decode(
                    decoder_input, encoder_hidden_states, None, input_attention_mask
                )
                
                # Get next token logits
                next_token_logits = self.lm_head(decoder_hidden_states[:, -1, :]) / temperature
                next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                # Check for end token
                if next_token_id[0].item() == vocab.get('</s>', vocab['<EOS>']):
                    break
                
                generated_tokens.append(next_token_id[0].item())
                
                # Append to decoder input
                decoder_input = torch.cat([decoder_input, next_token_id], dim=1)
        
        # Convert to words
        generated_words = []
        for token_id in generated_tokens:
            if token_id in idx_to_word:
                generated_words.append(idx_to_word[token_id])
        
        return generated_words

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_t5(model, train_loader, val_loader, epochs=5, learning_rate=1e-4):
    """Train T5 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    train_losses = []
    val_losses = []
    
    print(f"Training T5 on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            input_attention_mask = batch['input_attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, logits, _, _ = model(input_ids, input_attention_mask, target_ids)
            
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
        val_loss = evaluate_t5(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_t5(model, data_loader, device):
    """Evaluate T5 model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            input_attention_mask = batch['input_attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            loss, _, _, _ = model(input_ids, input_attention_mask, target_ids)
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
    print(f"=== T5 Text-to-Text Framework ({YEAR}) ===")
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
    
    # Test T5 on different tasks
    tasks = ['span_corruption', 'classification', 'summarization']
    
    for task in tasks[:2]:  # Test 2 tasks for demo
        print(f"\n" + "="*50)
        print(f"Training T5 for {task.upper()}")
        
        # Create dataset
        train_dataset = T5Dataset(train_subset, vocab, task=task)
        val_dataset = T5Dataset(val_subset, vocab, task=task)
        
        batch_size = 4  # Smaller batch size for T5
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_t5_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_t5_fn)
        
        # Initialize T5 model
        t5_model = T5Model(
            vocab_size=len(vocab),
            hidden_size=256,       # Reduced for demo
            num_heads=8,
            num_encoder_layers=4,  # Reduced for demo
            num_decoder_layers=4,
            ff_size=512,
            dropout=0.1
        )
        
        print(f"T5 parameters: {count_parameters(t5_model):,}")
        
        # Train model
        model_name = f'T5-{task}'
        metrics = track_computational_metrics(
            model_name,
            train_t5,
            t5_model, train_loader, val_loader, 5, 1e-4
        )
        
        train_losses, val_losses = metrics['result']
        training_histories[model_name] = (train_losses, val_losses)
        
        result = {
            'model': model_name,
            'task': task,
            'year': '2019',
            'final_loss': val_losses[-1] if val_losses else 0,
            'parameters': count_parameters(t5_model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': 'Text-to-text unified framework'
        }
        results.append(result)
        
        # Demonstrate generation
        print(f"\n{task.upper()} EXAMPLES:")
        print("="*30)
        
        t5_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t5_model.to(device)
        
        # Create test dataset for demonstration
        test_dataset = T5Dataset(test_subset, vocab, task=task)
        
        # Test generation
        for i in range(2):
            if i < len(test_dataset):
                example = test_dataset[i]
                input_text = example['input_text']
                target_text = example['target_text']
                
                # Convert to tensor
                input_ids = torch.tensor([[vocab.get(token, vocab['<UNK>']) for token in input_text]], device=device)
                input_attention_mask = torch.ones_like(input_ids, device=device)
                
                # Generate
                generated = t5_model.generate(input_ids, input_attention_mask, vocab, idx_to_word, max_length=10)
                
                print(f"  Input:     {' '.join(input_text)}")
                print(f"  Target:    {' '.join(target_text[:-1])}")  # Exclude </s>
                print(f"  Generated: {' '.join(generated)}")
                print()
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        if train_losses:
            ax.plot(train_losses, label=model_name, linewidth=2)
    ax.set_title('T5 Training Loss by Task', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss curves
    ax = axes[0, 1]
    for model_name, (_, val_losses) in training_histories.items():
        if val_losses:
            ax.plot(val_losses, label=model_name, linewidth=2)
    ax.set_title('T5 Validation Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Task unification concept
    ax = axes[1, 0]
    tasks_unified = ['Classification\n→ Text', 'Translation\n→ Text', 'Summarization\n→ Text', 
                    'QA\n→ Text', 'Generation\n→ Text']
    unification_scores = [10, 10, 10, 10, 10]  # All equally unified
    bars = ax.bar(tasks_unified, unification_scores, color=['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C', '#E74C3C'])
    ax.set_title('T5 Task Unification', fontsize=14)
    ax.set_ylabel('Unification Level')
    ax.tick_params(axis='x', rotation=45)
    
    # T5 innovations
    ax = axes[1, 1]
    innovations = ['Text-to-Text\nFramework', 'Relative\nPosition Bias', 'Span\nCorruption', 
                  'Unified\nArchitecture']
    impact_scores = [10, 8, 7, 9]
    bars = ax.bar(innovations, impact_scores, color=['#E74C3C', '#3498DB', '#F39C12', '#27AE60'])
    ax.set_title('T5 Key Innovations', fontsize=14)
    ax.set_ylabel('Impact Score')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/018_t5_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 018_t5_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("T5 TEXT-TO-TEXT FRAMEWORK SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Task: {result['task']}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nT5 REVOLUTIONARY CONCEPT:")
    print("="*50)
    print("EVERYTHING IS TEXT-TO-TEXT:")
    print("   • Classification: 'sentiment: [text]' → 'positive'")
    print("   • Translation: 'translate English to French: [text]' → '[french_text]'")
    print("   • Summarization: 'summarize: [article]' → '[summary]'")
    print("   • QA: 'question: [question] context: [context]' → '[answer]'")
    print("   • Generation: '[prefix]' → '[completion]'")
    
    print("\nT5 ARCHITECTURE INNOVATIONS:")
    print("="*50)
    print("1. UNIFIED FRAMEWORK:")
    print("   • Single model architecture for all NLP tasks")
    print("   • Task specified through input text prefixes")
    print("   • Same training procedure for all tasks")
    
    print("\n2.  RELATIVE POSITION BIAS:")
    print("   • Learned relative position biases instead of absolute positions")
    print("   • More efficient than full position embeddings")
    print("   • Better generalization to different sequence lengths")
    
    print("\n3.  SPAN CORRUPTION PRE-TRAINING:")
    print("   • Corrupt consecutive spans of input text")
    print("   • Model must predict corrupted tokens")
    print("   • More challenging than BERT's random masking")
    
    print("\n4.  ENCODER-DECODER ARCHITECTURE:")
    print("   • Encoder for understanding, decoder for generation")
    print("   • Cross-attention between encoder and decoder")
    print("   • Flexible for both understanding and generation tasks")
    
    print("\n T5 TASK FORMATTING EXAMPLES:")
    print("="*50)
    print("SENTIMENT ANALYSIS:")
    print("  Input:  'sentiment: I love this movie'")
    print("  Output: 'positive'")
    print("")
    print("TRANSLATION:")
    print("  Input:  'translate English to German: Hello world'")
    print("  Output: 'Hallo Welt'")
    print("")
    print("SUMMARIZATION:")
    print("  Input:  'summarize: [long article text...]'")
    print("  Output: '[summary text]'")
    print("")
    print("QUESTION ANSWERING:")
    print("  Input:  'question: What is AI? context: Artificial intelligence...'")
    print("  Output: 'Machine intelligence'")
    
    print("\n T5 TECHNICAL DETAILS:")
    print("="*50)
    print("• Encoder-Decoder: Full Transformer with cross-attention")
    print("• Position Encoding: Relative position bias (learned)")
    print("• Pre-training: Span corruption on C4 dataset (750GB)")
    print("• Tokenization: SentencePiece subword tokenization")
    print("• Training: Multi-task learning with task prefixes")
    print("• Scaling: T5-Small (60M) to T5-11B (11B parameters)")
    
    print("\n WHY TEXT-TO-TEXT WORKS:")
    print("="*50)
    print("1. 🌐 UNIVERSAL INTERFACE: Text is natural interface for all tasks")
    print("2.  UNIFIED TRAINING: Same loss function and optimization")
    print("3. 🎨 FLEXIBLE PROMPTING: Tasks specified through natural language")
    print("4.  TRANSFER LEARNING: Shared representations across tasks")
    print("5.  SIMPLE ARCHITECTURE: One model architecture for everything")
    print("6.  HUMAN-LIKE: Mirrors how humans understand instructions")
    
    print("\n T5 PERFORMANCE ACHIEVEMENTS:")
    print("="*50)
    print("• GLUE: State-of-the-art on 8/9 tasks")
    print("• SuperGLUE: 89.7 score (human baseline: 89.8)")
    print("• SQuAD: 91.2 F1 (new state-of-the-art)")
    print("• WMT Translation: Competitive with specialized models")
    print("• CNN/DM Summarization: 43.5 ROUGE-L")
    print("• CoQA: 83.7 F1 score on conversational QA")
    
    print("\n T5'S INFLUENCE ON MODERN AI:")
    print("="*50)
    print("• 🤖 FOUNDATION FOR GPT-3: Text-to-text paradigm adopted")
    print("• 💬 INSTRUCTION FOLLOWING: Natural language task specification")
    print("• 🔧 PROMPT ENGINEERING: Task formatting through prompts")
    print("•  ZERO-SHOT LEARNING: Tasks via natural language descriptions")
    print("• 🌐 MULTIMODAL: Extended to vision-language tasks")
    print("•  ARCHITECTURE STANDARD: Encoder-decoder for many LLMs")
    
    print("\n TASK UNIFICATION IMPACT:")
    print("="*50)
    print("BEFORE T5:")
    print("  • Separate models for each task")
    print("  • Task-specific architectures and training")
    print("  • Limited transfer between tasks")
    print("  • Complex deployment pipelines")
    
    print("\nAFTER T5:")
    print("  • Single model for all tasks")
    print("  • Unified training and architecture")
    print("  • Natural transfer learning")
    print("  • Simple deployment (one model)")
    
    print("\n KEY INSIGHTS FROM T5:")
    print("="*50)
    print("• Text-to-text is a powerful universal framework")
    print("• Task specification through natural language works")
    print("• Span corruption better than random token masking")
    print("• Encoder-decoder superior for many tasks")
    print("• Relative position bias more efficient than absolute")
    print("• Multi-task learning improves individual task performance")
    
    print("\n T5 LEGACY:")
    print("="*50)
    print("• Established text-to-text as standard paradigm")
    print("• Influenced GPT-3, PaLM, and modern LLMs")
    print("• Showed power of unified architectures")
    print("• Pioneered instruction-following through text")
    print("• Bridge between BERT-style and GPT-style models")
    print("• Foundation for modern prompt engineering")
    
    print(f"\n{'='*70}")
    print(" T5: EVERYTHING IS TEXT-TO-TEXT! ")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()