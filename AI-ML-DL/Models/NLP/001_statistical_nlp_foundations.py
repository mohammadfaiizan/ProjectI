"""
ERA 1: STATISTICAL NLP FOUNDATIONS (2000-2010)
==============================================

Year: 1950s-2000s
Innovation: Statistical language modeling foundations
Previous Limitation: Rule-based systems couldn't handle language variation
Performance Gain: First quantitative approach to language modeling
Impact: Established probabilistic framework for NLP

This file demonstrates the foundational statistical approaches to language modeling
that preceded neural methods, using N-gram models and basic feature extraction.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import pickle
from collections import defaultdict, Counter
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "1950s-2000s"
INNOVATION = "Statistical Language Modeling"
PREVIOUS_LIMITATION = "Rule-based systems couldn't capture language probability"
IMPACT = "Foundation for all modern language modeling"

print(f"=== {INNOVATION} ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING
# ============================================================================

def load_wikitext2_dataset():
    """
    Load WikiText-2 dataset with consistent preprocessing
    Returns processed text for statistical modeling
    """
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    
    # Extract text from dataset
    train_texts = [text for text in dataset['train']['text'] if text.strip()]
    val_texts = [text for text in dataset['validation']['text'] if text.strip()]
    test_texts = [text for text in dataset['test']['text'] if text.strip()]
    
    # Basic preprocessing for statistical models
    def preprocess_text(texts):
        processed = []
        for text in texts:
            # Basic cleaning
            text = text.lower()
            text = text.replace('\n', ' ')
            # Tokenization
            tokens = word_tokenize(text)
            # Filter out very short sequences
            if len(tokens) > 5:
                processed.extend(tokens)
        return processed
    
    train_tokens = preprocess_text(train_texts)
    val_tokens = preprocess_text(val_texts)
    test_tokens = preprocess_text(test_texts)
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Test tokens: {len(test_tokens):,}")
    
    return train_tokens, val_tokens, test_tokens

# ============================================================================
# N-GRAM LANGUAGE MODEL
# ============================================================================

class NgramLanguageModel:
    """
    N-gram statistical language model implementation
    Demonstrates the foundation of statistical NLP
    """
    
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0
        
    def train(self, tokens):
        """Train the n-gram model on tokenized text"""
        print(f"Training {self.n}-gram model...")
        
        # Add special tokens
        padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
        
        # Build vocabulary
        self.vocab = set(padded_tokens)
        self.vocab_size = len(self.vocab)
        
        # Count n-grams
        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i + self.n])
            context = ngram[:-1]
            word = ngram[-1]
            
            self.ngram_counts[context][word] += 1
            self.context_counts[context] += 1
    
    def get_probability(self, context, word):
        """Calculate probability P(word|context) with smoothing"""
        if context not in self.context_counts:
            return 1.0 / self.vocab_size  # Uniform distribution for unseen context
        
        # Add-one smoothing (Laplace smoothing)
        word_count = self.ngram_counts[context][word]
        context_count = self.context_counts[context]
        
        return (word_count + 1) / (context_count + self.vocab_size)
    
    def calculate_perplexity(self, tokens):
        """Calculate perplexity on test tokens"""
        padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
        
        log_prob_sum = 0
        token_count = 0
        
        for i in range(self.n - 1, len(padded_tokens)):
            context = tuple(padded_tokens[i - self.n + 1:i])
            word = padded_tokens[i]
            
            prob = self.get_probability(context, word)
            log_prob_sum += np.log(prob)
            token_count += 1
        
        # Calculate perplexity
        avg_log_prob = log_prob_sum / token_count
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    def generate_text(self, prompt, max_length=20):
        """Generate text using the n-gram model"""
        # Convert prompt to tokens and pad
        prompt_tokens = word_tokenize(prompt.lower())
        context = (['<START>'] * (self.n - 1) + prompt_tokens)[-(self.n - 1):]
        
        generated = prompt_tokens[:]
        
        for _ in range(max_length):
            context_tuple = tuple(context)
            
            if context_tuple not in self.ngram_counts:
                break
            
            # Sample next word based on probabilities
            candidates = self.ngram_counts[context_tuple]
            if not candidates:
                break
            
            # Simple sampling (could be improved with temperature)
            words, counts = zip(*candidates.items())
            probabilities = np.array(counts, dtype=float)
            probabilities /= probabilities.sum()
            
            next_word = np.random.choice(words, p=probabilities)
            
            if next_word == '<END>':
                break
            
            generated.append(next_word)
            context = context[1:] + [next_word]
        
        return ' '.join(generated)

# ============================================================================
# TRADITIONAL CLASSIFICATION MODEL
# ============================================================================

class TraditionalNLPClassifier:
    """
    Traditional NLP classification using TF-IDF and Naive Bayes
    Demonstrates statistical approach to text classification
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.trained = False
    
    def prepare_classification_data(self, texts):
        """Prepare synthetic classification data from language modeling data"""
        # Create simple binary classification: long vs short sentences
        X, y = [], []
        
        for text in texts[:1000]:  # Use subset for demo
            tokens = word_tokenize(text.lower())
            if len(tokens) > 3:
                X.append(text)
                # Binary classification: long (>10 words) vs short sentences
                y.append(1 if len(tokens) > 10 else 0)
        
        return X, y
    
    def train(self, X_train, y_train):
        """Train the traditional classifier"""
        print("Training traditional NLP classifier...")
        
        # Transform text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train Naive Bayes classifier
        self.classifier.fit(X_train_tfidf, y_train)
        self.trained = True
    
    def evaluate(self, X_test, y_test):
        """Evaluate the classifier"""
        if not self.trained:
            return 0.5
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

def track_computational_metrics(model_name, train_function, *args):
    """Track computational metrics during training"""
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
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
# EVALUATION AND COMPARISON
# ============================================================================

def evaluate_statistical_models(train_tokens, val_tokens, test_tokens):
    """Evaluate different statistical models"""
    results = []
    
    # Test different n-gram orders
    for n in [2, 3, 4]:
        print(f"\n--- {n}-gram Model ---")
        
        model = NgramLanguageModel(n=n)
        
        # Track training metrics
        metrics = track_computational_metrics(
            f"{n}-gram", 
            model.train, 
            train_tokens
        )
        
        # Calculate perplexity
        test_perplexity = model.calculate_perplexity(test_tokens[:1000])  # Use subset for speed
        val_perplexity = model.calculate_perplexity(val_tokens[:500])
        
        result = {
            'model': f"{n}-gram",
            'year': YEAR,
            'perplexity': test_perplexity,
            'val_perplexity': val_perplexity,
            'parameters': len(model.ngram_counts),  # Number of n-gram entries
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': f"{n}-gram statistical modeling"
        }
        
        results.append(result)
        
        print(f"Test Perplexity: {test_perplexity:.2f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")
        print(f"Training Time: {metrics['training_time_minutes']:.2f} minutes")
        print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
        print(f"Model Size: {len(model.ngram_counts):,} n-grams")
        
        # Generate sample text
        print("\nSample Generation:")
        for prompt in ["the quick brown", "in the beginning", "scientists have"]:
            generated = model.generate_text(prompt, max_length=10)
            print(f"  '{prompt}' -> '{generated}'")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results):
    """Create standard visualizations for statistical models"""
    
    # Performance comparison
    plt.figure(figsize=(12, 8))
    
    # Perplexity comparison
    plt.subplot(2, 2, 1)
    models = [r['model'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    plt.bar(models, perplexities)
    plt.title('Perplexity Comparison (Lower is Better)')
    plt.ylabel('Perplexity')
    
    # Training time comparison
    plt.subplot(2, 2, 2)
    training_times = [r['training_time'] for r in results]
    plt.bar(models, training_times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (minutes)')
    
    # Model size comparison
    plt.subplot(2, 2, 3)
    model_sizes = [r['parameters'] for r in results]
    plt.bar(models, model_sizes)
    plt.title('Model Size (N-gram entries)')
    plt.ylabel('Number of N-grams')
    
    # Memory usage comparison
    plt.subplot(2, 2, 4)
    memory_usage = [r['memory_usage'] for r in results]
    plt.bar(models, memory_usage)
    plt.title('Memory Usage')
    plt.ylabel('Memory (MB)')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/001_statistical_nlp_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 001_statistical_nlp_results.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Statistical NLP Foundations ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_tokens, val_tokens, test_tokens = load_wikitext2_dataset()
    
    # Evaluate statistical models
    results = evaluate_statistical_models(train_tokens, val_tokens, test_tokens)
    
    # Create visualizations
    create_visualizations(results)
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL NLP FOUNDATIONS SUMMARY")
    print("="*60)
    
    best_model = min(results, key=lambda x: x['perplexity'])
    print(f"Best Model: {best_model['model']}")
    print(f"Best Perplexity: {best_model['perplexity']:.2f}")
    print(f"Training Time: {best_model['training_time']:.2f} minutes")
    
    print("\nKey Insights:")
    print("- Higher-order n-grams generally perform better but require more memory")
    print("- Statistical models establish baseline performance for language modeling")
    print("- Sparse data problem becomes severe with higher n-gram orders")
    print("- Foundation for understanding why neural approaches were needed")
    
    return results

if __name__ == "__main__":
    results = main()