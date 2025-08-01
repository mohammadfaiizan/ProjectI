import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Optional, Tuple

# BERT Configuration
class BERTConfig:
    """Configuration class for BERT model"""
    
    def __init__(self, 
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

# BERT Embeddings
class BERTEmbeddings(nn.Module):
    """BERT embeddings: token + position + segment embeddings"""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register position_ids as buffer
        self.register_buffer("position_ids", 
                           torch.arange(config.max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# BERT Self-Attention
class BERTSelfAttention(nn.Module):
    """BERT self-attention mechanism"""
    
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} not divisible by num heads {config.num_attention_heads}")
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

# BERT Attention Output
class BERTSelfOutput(nn.Module):
    """BERT self-attention output layer"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

# BERT Attention Layer
class BERTAttention(nn.Module):
    """Complete BERT attention layer"""
    
    def __init__(self, config):
        super().__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]  # Add attentions if we output them

# BERT Intermediate Layer
class BERTIntermediate(nn.Module):
    """BERT intermediate (feed-forward) layer"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu  # BERT uses GELU activation
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# BERT Output Layer
class BERTOutput(nn.Module):
    """BERT output layer"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

# BERT Layer
class BERTLayer(nn.Module):
    """Single BERT transformer layer"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        
        # Feed-forward
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

# BERT Encoder
class BERTEncoder(nn.Module):
    """BERT encoder with multiple transformer layers"""
    
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        all_hidden_states = ()
        all_attentions = ()
        
        for layer_module in self.layer:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states,)
        
        outputs = (hidden_states, all_hidden_states)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        
        return outputs

# BERT Pooler
class BERTPooler(nn.Module):
    """BERT pooler layer for [CLS] token"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# BERT Model
class BERTModel(nn.Module):
    """Complete BERT model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        self.apply(_init_weights)
    
    def get_extended_attention_mask(self, attention_mask):
        """Create extended attention mask for multi-head attention"""
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")
        
        # Convert to float and create additive mask
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, output_attentions=False):
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        # Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        
        # Encoder
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, output_attentions)
        sequence_output = encoder_outputs[0]
        
        # Pooler
        pooled_output = self.pooler(sequence_output)
        
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs

# BERT for Masked Language Modeling
class BERTForMaskedLM(nn.Module):
    """BERT model for masked language modeling pre-training"""
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.cls = BERTLMPredictionHead(config)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs[0]
        
        prediction_scores = self.cls(sequence_output)
        
        outputs = (prediction_scores,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs
        
        return outputs

# BERT LM Prediction Head
class BERTLMPredictionHead(nn.Module):
    """BERT language modeling prediction head"""
    
    def __init__(self, config):
        super().__init__()
        self.transform = BERTLMPredictionHead_Transform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        
        # Link decoder weights to embedding weights
        # self.decoder.weight = self.bert.embeddings.word_embeddings.weight
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BERTLMPredictionHead_Transform(nn.Module):
    """Transform layer in BERT LM prediction head"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = F.gelu
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

# BERT for Next Sentence Prediction
class BERTForNextSentencePrediction(nn.Module):
    """BERT model for next sentence prediction pre-training"""
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.cls = nn.Linear(config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        
        seq_relationship_score = self.cls(pooled_output)
        
        outputs = (seq_relationship_score,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), labels.view(-1))
            outputs = (next_sentence_loss,) + outputs
        
        return outputs

# BERT for Pre-training (MLM + NSP)
class BERTForPreTraining(nn.Module):
    """BERT model for pre-training with both MLM and NSP"""
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.cls = BERTPreTrainingHeads(config)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                masked_lm_labels=None, next_sentence_label=None):
        
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output, pooled_output = outputs[:2]
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        outputs = (prediction_scores, seq_relationship_score) + outputs[2:]
        
        total_loss = 0
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.bert.config.vocab_size), 
                                    masked_lm_labels.view(-1))
            total_loss += masked_lm_loss
        
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), 
                                        next_sentence_label.view(-1))
            total_loss += next_sentence_loss
        
        if masked_lm_labels is not None or next_sentence_label is not None:
            outputs = (total_loss,) + outputs
        
        return outputs

class BERTPreTrainingHeads(nn.Module):
    """BERT pre-training heads for MLM and NSP"""
    
    def __init__(self, config):
        super().__init__()
        self.predictions = BERTLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

# BERT for Sequence Classification
class BERTForSequenceClassification(nn.Module):
    """BERT model for sequence classification"""
    
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BERTModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

# Data processing utilities
def create_masked_lm_predictions(tokens, vocab_size, mask_prob=0.15, 
                                mask_token_id=103, vocab_list=None):
    """Create masked language modeling data"""
    
    tokens = tokens.clone()
    labels = tokens.clone()
    
    # Create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(tokens.shape)
    
    # Create mask array
    mask_arr = (rand < mask_prob) & (tokens != 0) & (tokens != 101) & (tokens != 102)
    
    selection = torch.flatten(mask_arr.nonzero()).tolist()
    
    for idx in selection:
        # Get position
        pos = (idx // tokens.size(1), idx % tokens.size(1))
        
        # 80% of the time, replace with [MASK] token
        if torch.rand(1) < 0.8:
            tokens[pos] = mask_token_id
        # 10% of the time, replace with random token
        elif torch.rand(1) < 0.5:
            tokens[pos] = torch.randint(0, vocab_size, (1,)).item()
        # 10% of the time, keep original token
    
    # Only compute loss on masked tokens
    labels[~mask_arr] = -100
    
    return tokens, labels

if __name__ == "__main__":
    print("Testing BERT implementation...")
    
    # Create config
    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=128
    )
    
    # Test BERT model
    model = BERTModel(config)
    print(f"BERT model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test input
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    
    # Forward pass
    outputs = model(input_ids, attention_mask, token_type_ids)
    sequence_output, pooled_output = outputs[:2]
    
    print(f"Sequence output shape: {sequence_output.shape}")
    print(f"Pooled output shape: {pooled_output.shape}")
    
    # Test BERT for pre-training
    pretraining_model = BERTForPreTraining(config)
    print(f"Pre-training model parameters: {sum(p.numel() for p in pretraining_model.parameters()):,}")
    
    # Test MLM data creation
    masked_tokens, mlm_labels = create_masked_lm_predictions(input_ids, config.vocab_size)
    print(f"Original tokens: {input_ids[0][:10]}")
    print(f"Masked tokens: {masked_tokens[0][:10]}")
    print(f"MLM labels: {mlm_labels[0][:10]}")
    
    # Test pre-training forward pass
    next_sentence_labels = torch.randint(0, 2, (batch_size,))
    pretraining_outputs = pretraining_model(
        masked_tokens, attention_mask, token_type_ids, 
        mlm_labels, next_sentence_labels
    )
    
    if len(pretraining_outputs) > 2:
        loss = pretraining_outputs[0]
        print(f"Pre-training loss: {loss.item():.4f}")
    
    # Test sequence classification
    classifier = BERTForSequenceClassification(config, num_labels=3)
    classification_outputs = classifier(input_ids, attention_mask, token_type_ids)
    logits = classification_outputs[0]
    print(f"Classification logits shape: {logits.shape}")
    
    print("BERT implementation testing completed!")