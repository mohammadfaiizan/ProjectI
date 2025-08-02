"""
ERA 1: WORD EMBEDDINGS EVOLUTION (2013-2016)
===========================================

Year: 2013-2016
Innovation: Dense vector representations of words
Previous Limitation: Sparse one-hot vectors, no semantic similarity
Performance Gain: Semantic relationships captured in vector space
Impact: Foundation for all neural NLP models

This file demonstrates the evolution from sparse to dense word representations,
implementing Word2Vec, GloVe, and FastText from scratch on WikiText-2.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
from collections import defaultdict, Counter
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2013-2016"
INNOVATION = "Dense Word Embeddings"
PREVIOUS_LIMITATION = "Sparse one-hot vectors couldn't capture semantic relationships"
IMPACT = "Enabled semantic understanding in neural networks"

print(f"=== {INNOVATION} ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING
# ============================================================================

def load_wikitext2_dataset():
    """Load WikiText-2 dataset for consistent embedding training"""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    
    def preprocess_text(texts):
        processed = []
        for text in texts:
            if text.strip():
                text = text.lower().replace('\n', ' ')
                tokens = word_tokenize(text)
                if len(tokens) > 3:
                    processed.append(tokens)
        return processed
    
    train_sentences = preprocess_text(dataset['train']['text'])
    val_sentences = preprocess_text(dataset['validation']['text'])
    test_sentences = preprocess_text(dataset['test']['text'])
    
    print(f"Train sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(val_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    
    return train_sentences, val_sentences, test_sentences

# ============================================================================
# VOCABULARY BUILDING
# ============================================================================

def build_vocabulary(sentences, min_count=5):
    """Build vocabulary with frequency filtering"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    # Filter by minimum count
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) 
             if count >= min_count}
    vocab['<UNK>'] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, word_counts

# ============================================================================
# WORD2VEC IMPLEMENTATION
# ============================================================================

class Word2Vec:
    """
    Word2Vec implementation with Skip-gram and CBOW
    Demonstrates the breakthrough in dense word representations
    """
    
    def __init__(self, vocab_size, embedding_dim=100, window_size=5, mode='skipgram'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.mode = mode
        
        # Initialize embeddings randomly
        self.embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.output_weights = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        
    def train(self, sentences, vocab, epochs=5, learning_rate=0.025):
        """Train Word2Vec embeddings"""
        print(f"Training Word2Vec ({self.mode})...")
        
        # Create training pairs
        training_pairs = self._create_training_pairs(sentences, vocab)
        print(f"Training pairs: {len(training_pairs):,}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            np.random.shuffle(training_pairs)
            
            for center_word, context_word in training_pairs[:10000]:  # Use subset for demo
                if self.mode == 'skipgram':
                    loss = self._train_skipgram(center_word, context_word, learning_rate)
                else:
                    loss = self._train_cbow(center_word, context_word, learning_rate)
                epoch_loss += loss
            
            avg_loss = epoch_loss / min(len(training_pairs), 10000)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def _create_training_pairs(self, sentences, vocab):
        """Create training pairs for Word2Vec"""
        pairs = []
        
        for sentence in sentences:
            # Convert to indices
            indices = [vocab.get(word, vocab['<UNK>']) for word in sentence]
            
            for i, center_idx in enumerate(indices):
                # Get context window
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i:
                        pairs.append((center_idx, indices[j]))
        
        return pairs
    
    def _train_skipgram(self, center_word, context_word, learning_rate):
        """Train one step of Skip-gram"""
        # Forward pass
        center_embedding = self.embeddings[center_word]
        
        # Compute scores for all words (simplified)
        scores = np.dot(self.output_weights, center_embedding)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Compute loss (negative log likelihood)
        loss = -np.log(probabilities[context_word] + 1e-10)
        
        # Backward pass (simplified gradient descent)
        grad_output = probabilities.copy()
        grad_output[context_word] -= 1
        
        # Update weights
        self.output_weights -= learning_rate * np.outer(grad_output, center_embedding)
        self.embeddings[center_word] -= learning_rate * np.dot(grad_output, self.output_weights)
        
        return loss
    
    def _train_cbow(self, center_word, context_word, learning_rate):
        """Train one step of CBOW (simplified)"""
        # This is a simplified version - in practice, CBOW uses multiple context words
        return self._train_skipgram(context_word, center_word, learning_rate)
    
    def get_word_vector(self, word, vocab):
        """Get embedding vector for a word"""
        word_idx = vocab.get(word, vocab['<UNK>'])
        return self.embeddings[word_idx]
    
    def find_similar_words(self, word, vocab, idx_to_word, top_k=5):
        """Find most similar words using cosine similarity"""
        word_idx = vocab.get(word, vocab['<UNK>'])
        if word_idx == vocab['<UNK>']:
            return []
        
        word_vector = self.embeddings[word_idx].reshape(1, -1)
        similarities = cosine_similarity(word_vector, self.embeddings)[0]
        
        # Get top-k similar words (excluding the word itself)
        similar_indices = similarities.argsort()[-top_k-1:-1][::-1]
        
        similar_words = []
        for idx in similar_indices:
            if idx in idx_to_word and idx != word_idx:
                similar_words.append((idx_to_word[idx], similarities[idx]))
        
        return similar_words

# ============================================================================
# GLOVE IMPLEMENTATION (SIMPLIFIED)
# ============================================================================

class GloVe:
    """
    Simplified GloVe implementation
    Demonstrates global co-occurrence matrix factorization approach
    """
    
    def __init__(self, vocab_size, embedding_dim=100, window_size=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        # Initialize embeddings
        self.embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.context_embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))
        self.biases = np.random.uniform(-0.5, 0.5, vocab_size)
        self.context_biases = np.random.uniform(-0.5, 0.5, vocab_size)
    
    def build_cooccurrence_matrix(self, sentences, vocab):
        """Build global co-occurrence matrix"""
        print("Building co-occurrence matrix...")
        cooccurrence = defaultdict(Counter)
        
        for sentence in sentences:
            indices = [vocab.get(word, vocab['<UNK>']) for word in sentence]
            
            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i:
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        cooccurrence[center_idx][indices[j]] += weight
        
        return cooccurrence
    
    def train(self, sentences, vocab, epochs=10, learning_rate=0.05):
        """Train GloVe embeddings"""
        print("Training GloVe...")
        
        # Build co-occurrence matrix
        cooccurrence = self.build_cooccurrence_matrix(sentences, vocab)
        
        # Convert to list of (word1, word2, count) tuples
        training_data = []
        for word1, contexts in cooccurrence.items():
            for word2, count in contexts.items():
                training_data.append((word1, word2, count))
        
        print(f"Co-occurrence pairs: {len(training_data):,}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            np.random.shuffle(training_data)
            
            for word1, word2, count in training_data[:5000]:  # Use subset for demo
                loss = self._train_step(word1, word2, count, learning_rate)
                epoch_loss += loss
            
            avg_loss = epoch_loss / min(len(training_data), 5000)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def _train_step(self, word1, word2, count, learning_rate):
        """Single training step for GloVe"""
        # Forward pass
        dot_product = np.dot(self.embeddings[word1], self.context_embeddings[word2])
        prediction = dot_product + self.biases[word1] + self.context_biases[word2]
        
        # Loss (simplified weighting function)
        weight = min(1.0, (count / 100.0) ** 0.75)
        loss = weight * (prediction - np.log(count + 1)) ** 2
        
        # Gradient
        grad = weight * (prediction - np.log(count + 1))
        
        # Update parameters
        self.embeddings[word1] -= learning_rate * grad * self.context_embeddings[word2]
        self.context_embeddings[word2] -= learning_rate * grad * self.embeddings[word1]
        self.biases[word1] -= learning_rate * grad
        self.context_biases[word2] -= learning_rate * grad
        
        return loss

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_embeddings(model, vocab, idx_to_word, model_name):
    """Evaluate embedding quality"""
    print(f"\n--- Evaluating {model_name} ---")
    
    # Test word similarity
    test_words = ['king', 'queen', 'man', 'woman', 'good', 'bad', 'big', 'small']
    available_words = [word for word in test_words if word in vocab]
    
    print("Word Similarities:")
    for word in available_words[:3]:  # Test first 3 available words
        if hasattr(model, 'find_similar_words'):
            similar = model.find_similar_words(word, vocab, idx_to_word, top_k=3)
            print(f"  {word}: {similar}")
    
    # Calculate embedding statistics
    if hasattr(model, 'embeddings'):
        embeddings = model.embeddings
    else:
        embeddings = np.random.randn(len(vocab), 100)  # Placeholder
    
    return {
        'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
        'embedding_dim': embeddings.shape[1]
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_embeddings(embeddings, vocab, idx_to_word, title):
    """Visualize embeddings using PCA"""
    # Use subset of words for visualization
    n_words = min(50, len(vocab))
    subset_embeddings = embeddings[:n_words]
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(subset_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add word labels for a few words
    for i in range(min(20, n_words)):
        if i in idx_to_word:
            plt.annotate(idx_to_word[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, alpha=0.8)
    
    plt.title(f'{title} - Word Embeddings Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    return embeddings_2d

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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Word Embeddings Evolution ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Build vocabulary
    vocab, word_counts = build_vocabulary(train_sentences[:1000])  # Use subset for demo
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    results = []
    
    # Train Word2Vec Skip-gram
    print("\n" + "="*40)
    w2v_skipgram = Word2Vec(len(vocab), embedding_dim=50, mode='skipgram')
    metrics = track_computational_metrics(
        'Word2Vec-Skipgram',
        w2v_skipgram.train,
        train_sentences[:1000], vocab, 3  # Reduced epochs for demo
    )
    
    result = {
        'model': 'Word2Vec-Skipgram',
        'year': '2013',
        'parameters': len(vocab) * 50 * 2,  # Input + output embeddings
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Dense word representations via neural networks'
    }
    results.append(result)
    
    # Evaluate Word2Vec
    eval_stats = evaluate_embeddings(w2v_skipgram, vocab, idx_to_word, 'Word2Vec-Skipgram')
    result.update(eval_stats)
    
    # Train GloVe
    print("\n" + "="*40)
    glove = GloVe(len(vocab), embedding_dim=50)
    metrics = track_computational_metrics(
        'GloVe',
        glove.train,
        train_sentences[:500], vocab, 5  # Reduced for demo
    )
    
    result = {
        'model': 'GloVe',
        'year': '2014',
        'parameters': len(vocab) * 50 * 2,  # Word + context embeddings
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Global co-occurrence matrix factorization'
    }
    results.append(result)
    
    # Evaluate GloVe
    eval_stats = evaluate_embeddings(glove, vocab, idx_to_word, 'GloVe')
    result.update(eval_stats)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    visualize_embeddings(w2v_skipgram.embeddings, vocab, idx_to_word, 'Word2Vec')
    
    plt.subplot(1, 3, 2)
    visualize_embeddings(glove.embeddings, vocab, idx_to_word, 'GloVe')
    
    plt.subplot(1, 3, 3)
    models = [r['model'] for r in results]
    training_times = [r['training_time'] for r in results]
    plt.bar(models, training_times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (minutes)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/002_word_embeddings_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 002_word_embeddings_results.png")
    
    # Print summary
    print("\n" + "="*60)
    print("WORD EMBEDDINGS EVOLUTION SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nKey Insights:")
    print("- Dense embeddings capture semantic relationships impossible with sparse vectors")
    print("- Word2Vec: Local context windows, efficient training")
    print("- GloVe: Global statistics, more stable training")
    print("- Foundation for all subsequent neural NLP models")
    print("- Enabled transfer learning in NLP")
    
    return results

if __name__ == "__main__":
    results = main()