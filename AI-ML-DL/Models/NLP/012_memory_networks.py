"""
ERA 4: MEMORY NETWORKS (2014-2016)
==================================

Year: 2014-2016
Innovation: Memory Networks and Dynamic Memory Networks with external memory
Previous Limitation: Fixed internal memory in RNNs, limited reasoning capabilities
Performance Gain: External memory for complex reasoning, explicit memory operations
Impact: Influenced attention mechanisms, foundation for memory-augmented networks

This file demonstrates Memory Networks that introduced explicit memory operations
and influenced the development of attention mechanisms and external memory systems.
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

YEAR = "2014-2016"
INNOVATION = "Memory Networks and Dynamic Memory Networks"
PREVIOUS_LIMITATION = "RNNs have limited internal memory, poor at multi-step reasoning"
IMPACT = "External memory for reasoning, influenced attention mechanisms and memory-augmented networks"

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
                    if 5 <= len(tokens) <= 20:  # Good length for memory tasks
                        sentences.append(tokens)
        return sentences
    
    train_sentences = preprocess_text(dataset['train']['text'])
    val_sentences = preprocess_text(dataset['validation']['text'])
    test_sentences = preprocess_text(dataset['test']['text'])
    
    print(f"Train sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(val_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, vocab_size=5000):
    """Build vocabulary with special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 4)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<SOS>': 2,
        '<EOS>': 3
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# DATASET CLASS FOR MEMORY NETWORK EXPERIMENTS
# ============================================================================

class MemoryNetworkDataset(Dataset):
    """
    Dataset for memory network experiments
    Creates tasks that require memory and reasoning
    """
    
    def __init__(self, sentences, vocab, task='qa', max_sentences=5, max_length=15):
        self.vocab = vocab
        self.task = task
        self.max_sentences = max_sentences
        self.max_length = max_length
        self.examples = []
        
        if task == 'qa':
            # Simple question answering based on multiple sentences
            self._create_qa_examples(sentences)
        elif task == 'story_completion':
            # Story completion requiring memory of previous sentences
            self._create_story_examples(sentences)
        elif task == 'fact_retrieval':
            # Fact retrieval from memory
            self._create_fact_examples(sentences)
        
        print(f"Created {len(self.examples)} {task} examples")
    
    def _create_qa_examples(self, sentences):
        """Create simple QA examples"""
        question_templates = [
            ("what is {word}", "answer_word"),
            ("where is {word}", "location"),
            ("who has {word}", "person")
        ]
        
        for i in range(0, len(sentences) - self.max_sentences, self.max_sentences):
            context_sentences = sentences[i:i + self.max_sentences]
            
            # Ensure sentences are not too long
            context_sentences = [s for s in context_sentences if len(s) <= self.max_length]
            
            if len(context_sentences) >= 2:
                # Create simple questions based on words in context
                all_words = set()
                for sentence in context_sentences:
                    all_words.update(sentence)
                
                # Pick a random word for question
                if all_words:
                    target_word = random.choice(list(all_words))
                    
                    # Simple question: "What sentence contains {word}?"
                    question = ["what", "sentence", "contains", target_word]
                    
                    # Find answer (sentence index that contains the word)
                    answer_idx = 0
                    for idx, sentence in enumerate(context_sentences):
                        if target_word in sentence:
                            answer_idx = idx
                            break
                    
                    self.examples.append({
                        'context': context_sentences,
                        'question': question,
                        'answer': answer_idx,
                        'target_word': target_word
                    })
    
    def _create_story_examples(self, sentences):
        """Create story completion examples"""
        for i in range(0, len(sentences) - 3, 3):
            story_sentences = sentences[i:i + 3]
            
            if all(len(s) <= self.max_length for s in story_sentences):
                # Use first two sentences as context, predict third
                context = story_sentences[:2]
                target = story_sentences[2]
                
                self.examples.append({
                    'context': context,
                    'target': target
                })
    
    def _create_fact_examples(self, sentences):
        """Create fact retrieval examples"""
        for sentence in sentences:
            if len(sentence) <= self.max_length and len(sentence) >= 5:
                # Create fact: first part is query, second part is answer
                mid_point = len(sentence) // 2
                query = sentence[:mid_point]
                answer = sentence[mid_point:]
                
                self.examples.append({
                    'context': [sentence],  # Full sentence as context
                    'query': query,
                    'answer': answer
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.task == 'qa':
            # Convert context sentences to indices
            context_indices = []
            for sentence in example['context']:
                sentence_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence]
                context_indices.append(sentence_indices)
            
            # Convert question to indices
            question_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in example['question']]
            
            return {
                'context': context_indices,
                'question': question_indices,
                'answer': example['answer'],
                'target_word': example['target_word']
            }
        
        elif self.task == 'story_completion':
            context_indices = []
            for sentence in example['context']:
                sentence_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence]
                context_indices.append(sentence_indices)
            
            target_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in example['target']]
            
            return {
                'context': context_indices,
                'target': target_indices
            }
        
        else:  # fact_retrieval
            context_indices = []
            for sentence in example['context']:
                sentence_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence]
                context_indices.append(sentence_indices)
            
            query_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in example['query']]
            answer_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in example['answer']]
            
            return {
                'context': context_indices,
                'query': query_indices,
                'answer': answer_indices
            }

def collate_memory_fn(batch):
    """Custom collate function for memory network data"""
    if not batch:
        return None
    
    # Handle different task types
    if 'question' in batch[0]:  # QA task
        max_context_sentences = max(len(item['context']) for item in batch)
        max_sentence_length = max(max(len(sentence) for sentence in item['context']) for item in batch)
        max_question_length = max(len(item['question']) for item in batch)
        
        batch_contexts = []
        batch_questions = []
        batch_answers = []
        
        for item in batch:
            # Pad context sentences
            padded_context = []
            for sentence in item['context']:
                padded_sentence = sentence + [0] * (max_sentence_length - len(sentence))
                padded_context.append(padded_sentence)
            
            # Pad number of sentences
            while len(padded_context) < max_context_sentences:
                padded_context.append([0] * max_sentence_length)
            
            batch_contexts.append(padded_context)
            
            # Pad question
            padded_question = item['question'] + [0] * (max_question_length - len(item['question']))
            batch_questions.append(padded_question)
            
            batch_answers.append(item['answer'])
        
        return {
            'context': torch.tensor(batch_contexts, dtype=torch.long),
            'question': torch.tensor(batch_questions, dtype=torch.long),
            'answer': torch.tensor(batch_answers, dtype=torch.long)
        }
    
    else:
        # Handle other task types (simplified)
        return batch

# ============================================================================
# MEMORY MODULE
# ============================================================================

class MemoryModule(nn.Module):
    """
    Basic memory module that stores and retrieves information
    Core component of memory networks
    """
    
    def __init__(self, memory_size, memory_dim):
        super(MemoryModule, self).__init__()
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Initialize memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim) * 0.1)
        
        # Attention mechanism for memory retrieval
        self.query_projection = nn.Linear(memory_dim, memory_dim)
        self.key_projection = nn.Linear(memory_dim, memory_dim)
        self.value_projection = nn.Linear(memory_dim, memory_dim)
        
    def forward(self, query, write_vector=None, write_gate=None):
        """
        Memory read and optional write operation
        
        Args:
            query: (batch_size, memory_dim) query vector
            write_vector: (batch_size, memory_dim) optional vector to write
            write_gate: (batch_size, memory_size) optional write weights
            
        Returns:
            retrieved: (batch_size, memory_dim) retrieved memory content
            attention_weights: (batch_size, memory_size) attention weights
        """
        batch_size = query.size(0)
        
        # Project query for attention
        projected_query = self.query_projection(query)  # (batch_size, memory_dim)
        
        # Compute attention scores with memory
        memory_keys = self.key_projection(self.memory)  # (memory_size, memory_dim)
        
        # Attention scores
        attention_scores = torch.matmul(projected_query, memory_keys.t())  # (batch_size, memory_size)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Retrieve from memory
        memory_values = self.value_projection(self.memory)  # (memory_size, memory_dim)
        retrieved = torch.matmul(attention_weights, memory_values)  # (batch_size, memory_dim)
        
        # Optional write to memory
        if write_vector is not None and write_gate is not None:
            # Update memory using write gate
            write_weights = write_gate.unsqueeze(-1)  # (batch_size, memory_size, 1)
            write_content = write_vector.unsqueeze(1)  # (batch_size, 1, memory_dim)
            
            # Weighted update
            memory_update = torch.sum(write_weights * write_content, dim=0)  # (memory_size, memory_dim)
            self.memory.data = self.memory.data + memory_update
        
        return retrieved, attention_weights

# ============================================================================
# END-TO-END MEMORY NETWORK
# ============================================================================

class EndToEndMemoryNetwork(nn.Module):
    """
    End-to-End Memory Network for question answering
    Based on the MemN2N paper
    """
    
    def __init__(self, vocab_size, embed_dim=128, memory_dim=128, num_hops=3, max_sentences=10):
        super(EndToEndMemoryNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.num_hops = num_hops
        self.max_sentences = max_sentences
        
        # Embedding layers for different roles
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.question_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position encoding for sentences
        self.position_encoding = nn.Parameter(torch.randn(max_sentences, embed_dim))
        
        # Linear layers for each hop
        self.hop_projections = nn.ModuleList([
            nn.Linear(embed_dim, memory_dim) for _ in range(num_hops)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(memory_dim, max_sentences)  # Predict sentence index
        
    def forward(self, context, question):
        """
        Forward pass of memory network
        
        Args:
            context: (batch_size, max_sentences, max_sentence_length)
            question: (batch_size, question_length)
            
        Returns:
            logits: (batch_size, max_sentences) answer logits
            attention_weights: List of attention weights for each hop
        """
        batch_size, max_sentences, max_sentence_length = context.size()
        
        # Embed context sentences
        context_embedded = self.input_embedding(context)  # (batch_size, max_sentences, max_sentence_length, embed_dim)
        
        # Sum embeddings within each sentence
        context_encoded = context_embedded.sum(dim=2)  # (batch_size, max_sentences, embed_dim)
        
        # Add position encoding
        position_encoded = self.position_encoding[:max_sentences].unsqueeze(0).expand(batch_size, -1, -1)
        context_encoded = context_encoded + position_encoded
        
        # Embed question
        question_embedded = self.question_embedding(question)  # (batch_size, question_length, embed_dim)
        question_encoded = question_embedded.sum(dim=1)  # (batch_size, embed_dim)
        
        # Memory network hops
        query = question_encoded
        all_attention_weights = []
        
        for hop in range(self.num_hops):
            # Project query for this hop
            projected_query = self.hop_projections[hop](query)  # (batch_size, memory_dim)
            
            # Compute attention over context sentences
            # Use context as both keys and values
            context_keys = context_encoded  # (batch_size, max_sentences, embed_dim)
            
            # Attention scores
            attention_scores = torch.bmm(
                projected_query.unsqueeze(1),  # (batch_size, 1, memory_dim)
                context_keys.transpose(1, 2)  # (batch_size, embed_dim, max_sentences)
            ).squeeze(1)  # (batch_size, max_sentences)
            
            attention_weights = F.softmax(attention_scores, dim=1)
            all_attention_weights.append(attention_weights)
            
            # Retrieve from memory
            retrieved = torch.bmm(
                attention_weights.unsqueeze(1),  # (batch_size, 1, max_sentences)
                context_encoded  # (batch_size, max_sentences, embed_dim)
            ).squeeze(1)  # (batch_size, embed_dim)
            
            # Update query for next hop
            query = query + retrieved  # Residual connection
        
        # Final output
        logits = self.output_projection(query)  # (batch_size, max_sentences)
        
        return logits, all_attention_weights

# ============================================================================
# DYNAMIC MEMORY NETWORK
# ============================================================================

class DynamicMemoryNetwork(nn.Module):
    """
    Dynamic Memory Network with episodic memory
    More sophisticated memory mechanism with dynamic updates
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, memory_dim=128, num_episodes=3):
        super(DynamicMemoryNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_episodes = num_episodes
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Input module (encode context)
        self.input_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Question module
        self.question_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Episodic memory module
        self.attention_gru = nn.GRU(hidden_dim * 4, hidden_dim, batch_first=True)
        self.episode_projection = nn.Linear(hidden_dim, memory_dim)
        
        # Memory update gate
        self.memory_update = nn.Linear(memory_dim + hidden_dim, memory_dim)
        
        # Answer module
        self.answer_gru = nn.GRU(memory_dim + hidden_dim, hidden_dim, batch_first=True)
        self.answer_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, context, question):
        """
        Forward pass of Dynamic Memory Network
        
        Args:
            context: (batch_size, max_sentences, max_sentence_length)
            question: (batch_size, question_length)
            
        Returns:
            answer_logits: (batch_size, vocab_size)
            episode_attentions: List of attention weights for each episode
        """
        batch_size = context.size(0)
        
        # Encode context sentences
        context_representations = []
        for i in range(context.size(1)):
            sentence = context[:, i, :]  # (batch_size, max_sentence_length)
            sentence_embedded = self.embedding(sentence)
            sentence_output, sentence_hidden = self.input_gru(sentence_embedded)
            context_representations.append(sentence_hidden[-1])  # Use final hidden state
        
        context_encoded = torch.stack(context_representations, dim=1)  # (batch_size, max_sentences, hidden_dim)
        
        # Encode question
        question_embedded = self.embedding(question)
        question_output, question_hidden = self.question_gru(question_embedded)
        question_representation = question_hidden[-1]  # (batch_size, hidden_dim)
        
        # Initialize memory
        memory = question_representation  # Start with question representation
        
        episode_attentions = []
        
        # Episodic memory updates
        for episode in range(self.num_episodes):
            # Compute attention over context based on current memory and question
            episode_attention = self._compute_episode_attention(
                context_encoded, question_representation, memory
            )
            episode_attentions.append(episode_attention)
            
            # Update memory based on attended context
            attended_context = torch.bmm(
                episode_attention.unsqueeze(1),  # (batch_size, 1, max_sentences)
                context_encoded  # (batch_size, max_sentences, hidden_dim)
            ).squeeze(1)  # (batch_size, hidden_dim)
            
            # Memory update
            memory_input = torch.cat([memory, attended_context], dim=1)
            memory = torch.tanh(self.memory_update(memory_input))
        
        # Generate answer
        answer_input = torch.cat([memory, question_representation], dim=1)
        answer_hidden = answer_input.unsqueeze(1)  # (batch_size, 1, memory_dim + hidden_dim)
        
        answer_output, _ = self.answer_gru(answer_hidden)
        answer_logits = self.answer_projection(answer_output.squeeze(1))
        
        return answer_logits, episode_attentions
    
    def _compute_episode_attention(self, context_encoded, question_representation, memory):
        """Compute attention for episodic memory update"""
        batch_size, max_sentences, hidden_dim = context_encoded.size()
        
        # Prepare features for attention computation
        question_expanded = question_representation.unsqueeze(1).expand(-1, max_sentences, -1)
        memory_expanded = memory.unsqueeze(1).expand(-1, max_sentences, -1)
        
        # Concatenate features
        attention_features = torch.cat([
            context_encoded,
            question_expanded,
            memory_expanded,
            context_encoded * question_expanded  # Element-wise product
        ], dim=2)  # (batch_size, max_sentences, hidden_dim * 4)
        
        # Compute attention scores
        attention_output, _ = self.attention_gru(attention_features)
        attention_scores = attention_output.sum(dim=2)  # Sum over hidden dimensions
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_memory_network(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train memory network model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            if batch is None:
                continue
                
            context = batch['context'].to(device)
            question = batch['question'].to(device)
            answer = batch['answer'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, EndToEndMemoryNetwork):
                logits, attention_weights = model(context, question)
            else:  # DynamicMemoryNetwork
                logits, episode_attentions = model(context, question)
            
            loss = criterion(logits, answer)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_accuracy = evaluate_memory_network(model, val_loader, device)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: No valid batches")
    
    return train_losses, val_accuracies

def evaluate_memory_network(model, data_loader, device):
    """Evaluate memory network accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
                
            context = batch['context'].to(device)
            question = batch['question'].to(device)
            answer = batch['answer'].to(device)
            
            # Forward pass
            if isinstance(model, EndToEndMemoryNetwork):
                logits, _ = model(context, question)
            else:  # DynamicMemoryNetwork
                logits, _ = model(context, question)
            
            predictions = logits.argmax(dim=1)
            correct += (predictions == answer).sum().item()
            total += answer.size(0)
    
    return correct / total if total > 0 else 0.0

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
    print(f"=== Memory Networks ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:800]
    val_subset = val_sentences[:160]
    test_subset = test_sentences[:160]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=2000)
    
    results = []
    training_histories = {}
    
    # Test End-to-End Memory Network
    print("\n" + "="*50)
    print("Training End-to-End Memory Network")
    
    # Create QA dataset
    train_dataset = MemoryNetworkDataset(train_subset, vocab, task='qa', max_sentences=5)
    val_dataset = MemoryNetworkDataset(val_subset, vocab, task='qa', max_sentences=5)
    
    # Create data loaders
    batch_size = 16  # Smaller batch size for memory networks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_memory_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_memory_fn)
    
    # Initialize End-to-End Memory Network
    memn2n = EndToEndMemoryNetwork(
        vocab_size=len(vocab),
        embed_dim=128,
        memory_dim=128,
        num_hops=3,
        max_sentences=5
    )
    
    # Train model
    metrics = track_computational_metrics(
        'MemN2N',
        train_memory_network,
        memn2n, train_loader, val_loader, 8, 0.001
    )
    
    train_losses, val_accuracies = metrics['result']
    training_histories['MemN2N'] = (train_losses, val_accuracies)
    
    result1 = {
        'model': 'MemN2N',
        'year': '2015',
        'final_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'parameters': count_parameters(memn2n),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'End-to-end memory network with multiple hops'
    }
    results.append(result1)
    
    # Test Dynamic Memory Network
    print("\n" + "="*50)
    print("Training Dynamic Memory Network")
    
    # Initialize Dynamic Memory Network
    dmn = DynamicMemoryNetwork(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        memory_dim=128,
        num_episodes=3
    )
    
    # Train DMN
    metrics = track_computational_metrics(
        'DMN',
        train_memory_network,
        dmn, train_loader, val_loader, 8, 0.001
    )
    
    train_losses, val_accuracies = metrics['result']
    training_histories['DMN'] = (train_losses, val_accuracies)
    
    result2 = {
        'model': 'DMN',
        'year': '2016',
        'final_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'parameters': count_parameters(dmn),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Dynamic memory with episodic updates'
    }
    results.append(result2)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        if train_losses:
            ax.plot(train_losses, label=model_name)
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation accuracy curves
    ax = axes[0, 1]
    for model_name, (_, val_accuracies) in training_histories.items():
        if val_accuracies:
            ax.plot(val_accuracies, label=model_name)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Model comparison
    ax = axes[1, 0]
    models = [r['model'] for r in results]
    accuracies = [r['final_accuracy'] for r in results]
    ax.bar(models, accuracies)
    ax.set_title('Final Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    
    # Parameter comparison
    ax = axes[1, 1]
    parameters = [r['parameters'] for r in results]
    ax.bar(models, parameters)
    ax.set_title('Model Size (Parameters)')
    ax.set_ylabel('Number of Parameters')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/012_memory_networks_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 012_memory_networks_results.png")
    
    # Print summary
    print("\n" + "="*60)
    print("MEMORY NETWORKS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nMemory Network Components:")
    print("- External Memory: Explicit memory slots for information storage")
    print("- Memory Addressing: Attention-based retrieval mechanism")
    print("- Multiple Hops: Iterative reasoning over memory")
    print("- Episodic Updates: Dynamic memory modification")
    print("- End-to-End Training: Differentiable memory operations")
    
    print("\nKey Innovations:")
    print("- Explicit memory separate from parameters")
    print("- Attention-based memory retrieval")
    print("- Multi-step reasoning capabilities")
    print("- Influenced attention mechanism development")
    print("- Foundation for memory-augmented networks")
    
    print("\nInfluence on Attention Mechanisms:")
    print("- Memory addressing → attention over sequences")
    print("- Multiple hops → multi-head attention")
    print("- External memory → key-value attention")
    print("- Dynamic updates → self-attention")
    print("- Reasoning capabilities → transformer reasoning")
    
    print("\nHistorical Impact:")
    print("- Showed importance of external memory")
    print("- Demonstrated attention-based retrieval")
    print("- Influenced Transformer development")
    print("- Foundation for Neural Turing Machines")
    print("- Bridge to modern memory-augmented networks")
    
    return results

if __name__ == "__main__":
    results = main()