import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import collections
from typing import List, Dict, Tuple, Optional
import numpy as np

# Basic Text Preprocessing
class TextTokenizer:
    """Basic text tokenizer with vocabulary management"""
    
    def __init__(self, vocab_size=10000, min_freq=1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.vocab_built = False
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation (optional)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        word_counts = collections.Counter()
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            word_counts.update(words)
        
        # Sort words by frequency
        sorted_words = word_counts.most_common(self.vocab_size - 4)
        
        # Add words to vocabulary if they meet minimum frequency
        for word, count in sorted_words:
            if count >= self.min_freq:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def tokenize(self, text):
        """Tokenize text into list of tokens"""
        cleaned_text = self.clean_text(text)
        return cleaned_text.split()
    
    def encode(self, text, max_length=None, add_special_tokens=True):
        """Encode text to list of token indices"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        
        # Convert to indices
        indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
        # Truncate or pad
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
                if add_special_tokens:
                    indices[-1] = self.word_to_idx['<EOS>']  # Ensure EOS at the end
            else:
                indices.extend([self.word_to_idx['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode list of indices back to text"""
        tokens = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        
        if skip_special_tokens:
            special_tokens = {'<PAD>', '<BOS>', '<EOS>', '<UNK>'}
            tokens = [token for token in tokens if token not in special_tokens]
        
        return ' '.join(tokens)

# Tensor-based Text Processing
class TensorTextProcessor:
    """Text processing operations using PyTorch tensors"""
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def create_padding_mask(self, sequences, pad_token_id=0):
        """Create padding mask for sequences"""
        # sequences: [batch_size, seq_len]
        return (sequences != pad_token_id).float()
    
    def create_attention_mask(self, sequences, pad_token_id=0):
        """Create attention mask (True for tokens to attend to)"""
        return sequences != pad_token_id
    
    def create_causal_mask(self, seq_len, device='cpu'):
        """Create causal (lower triangular) mask for autoregressive models"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for positions to attend to
    
    def one_hot_encode(self, sequences, vocab_size=None):
        """One-hot encode sequences"""
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        # sequences: [batch_size, seq_len]
        batch_size, seq_len = sequences.size()
        one_hot = torch.zeros(batch_size, seq_len, vocab_size)
        one_hot.scatter_(2, sequences.unsqueeze(2), 1)
        return one_hot
    
    def compute_sequence_lengths(self, sequences, pad_token_id=0):
        """Compute actual sequence lengths (excluding padding)"""
        # sequences: [batch_size, seq_len]
        mask = sequences != pad_token_id
        lengths = mask.sum(dim=1)
        return lengths
    
    def truncate_or_pad(self, sequences, max_length, pad_token_id=0):
        """Truncate or pad sequences to fixed length"""
        current_length = sequences.size(1)
        
        if current_length > max_length:
            # Truncate
            return sequences[:, :max_length]
        elif current_length < max_length:
            # Pad
            batch_size = sequences.size(0)
            padding = torch.full((batch_size, max_length - current_length), 
                               pad_token_id, dtype=sequences.dtype, device=sequences.device)
            return torch.cat([sequences, padding], dim=1)
        else:
            return sequences
    
    def sliding_window(self, sequences, window_size, stride=1):
        """Create sliding windows from sequences"""
        # sequences: [batch_size, seq_len]
        batch_size, seq_len = sequences.size()
        
        if seq_len < window_size:
            return sequences.unsqueeze(1)
        
        # Calculate number of windows
        num_windows = (seq_len - window_size) // stride + 1
        
        # Create windows
        windows = []
        for i in range(0, num_windows * stride, stride):
            if i + window_size <= seq_len:
                window = sequences[:, i:i + window_size]
                windows.append(window)
        
        return torch.stack(windows, dim=1)  # [batch_size, num_windows, window_size]

# Advanced Text Processing
class AdvancedTextProcessor:
    """Advanced text processing with tensor operations"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def ngram_features(self, text, n=2):
        """Extract n-gram features from text"""
        tokens = self.tokenizer.tokenize(text)
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def create_ngram_tensor(self, texts, n=2, max_features=1000):
        """Create n-gram feature tensor"""
        # Collect all n-grams
        all_ngrams = set()
        for text in texts:
            ngrams = self.ngram_features(text, n)
            all_ngrams.update(ngrams)
        
        # Create vocabulary
        ngram_vocab = {ngram: idx for idx, ngram in enumerate(list(all_ngrams)[:max_features])}
        
        # Create feature tensor
        features = torch.zeros(len(texts), len(ngram_vocab))
        
        for i, text in enumerate(texts):
            ngrams = self.ngram_features(text, n)
            for ngram in ngrams:
                if ngram in ngram_vocab:
                    features[i, ngram_vocab[ngram]] += 1
        
        return features, ngram_vocab
    
    def tf_idf_features(self, texts, max_features=1000):
        """Compute TF-IDF features using tensors"""
        # Simple TF-IDF implementation
        vocab = {}
        word_counts = []
        
        # Build vocabulary and count words
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            text_counts = collections.Counter(tokens)
            word_counts.append(text_counts)
            
            for word in text_counts:
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        # Limit vocabulary size
        if len(vocab) > max_features:
            # Keep most frequent words
            all_word_counts = collections.Counter()
            for counts in word_counts:
                all_word_counts.update(counts)
            
            top_words = [word for word, _ in all_word_counts.most_common(max_features)]
            vocab = {word: idx for idx, word in enumerate(top_words)}
        
        # Create TF matrix
        tf_matrix = torch.zeros(len(texts), len(vocab))
        
        for i, counts in enumerate(word_counts):
            total_words = sum(counts.values())
            for word, count in counts.items():
                if word in vocab:
                    tf_matrix[i, vocab[word]] = count / total_words
        
        # Compute IDF
        doc_freq = torch.zeros(len(vocab))
        for counts in word_counts:
            for word in counts:
                if word in vocab:
                    doc_freq[vocab[word]] += 1
        
        idf = torch.log(len(texts) / (doc_freq + 1))
        
        # Compute TF-IDF
        tfidf_matrix = tf_matrix * idf.unsqueeze(0)
        
        return tfidf_matrix, vocab
    
    def batch_encode(self, texts, max_length=128, return_tensors=True):
        """Batch encode multiple texts"""
        encoded_texts = []
        lengths = []
        
        for text in texts:
            indices = self.tokenizer.encode(text, max_length=max_length)
            encoded_texts.append(indices)
            lengths.append(len([idx for idx in indices if idx != self.tokenizer.word_to_idx['<PAD>']]))
        
        if return_tensors:
            # Convert to tensors
            encoded_texts = torch.tensor(encoded_texts, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)
            
            # Create attention mask
            attention_mask = (encoded_texts != self.tokenizer.word_to_idx['<PAD>']).long()
            
            return {
                'input_ids': encoded_texts,
                'attention_mask': attention_mask,
                'lengths': lengths
            }
        
        return encoded_texts, lengths

# Text Augmentation
class TextAugmentation:
    """Text augmentation techniques"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def random_word_dropout(self, text, dropout_prob=0.1):
        """Randomly drop words from text"""
        tokens = self.tokenizer.tokenize(text)
        
        # Keep tokens with probability (1 - dropout_prob)
        augmented_tokens = [token for token in tokens if torch.rand(1).item() > dropout_prob]
        
        # Ensure at least one token remains
        if not augmented_tokens:
            augmented_tokens = [tokens[0]] if tokens else ['<UNK>']
        
        return ' '.join(augmented_tokens)
    
    def random_word_replacement(self, text, replacement_prob=0.1, vocab_list=None):
        """Randomly replace words with random words from vocabulary"""
        tokens = self.tokenizer.tokenize(text)
        
        if vocab_list is None:
            vocab_list = list(self.tokenizer.word_to_idx.keys())
        
        augmented_tokens = []
        for token in tokens:
            if torch.rand(1).item() < replacement_prob:
                # Replace with random word
                random_word = torch.randint(0, len(vocab_list), (1,)).item()
                augmented_tokens.append(vocab_list[random_word])
            else:
                augmented_tokens.append(token)
        
        return ' '.join(augmented_tokens)
    
    def random_word_insertion(self, text, insertion_prob=0.1, vocab_list=None):
        """Randomly insert words into text"""
        tokens = self.tokenizer.tokenize(text)
        
        if vocab_list is None:
            vocab_list = list(self.tokenizer.word_to_idx.keys())
        
        augmented_tokens = []
        for token in tokens:
            augmented_tokens.append(token)
            
            # Insert random word with probability
            if torch.rand(1).item() < insertion_prob:
                random_word = torch.randint(0, len(vocab_list), (1,)).item()
                augmented_tokens.append(vocab_list[random_word])
        
        return ' '.join(augmented_tokens)
    
    def back_translation_simulation(self, text, noise_level=0.05):
        """Simulate back-translation by adding noise"""
        tokens = self.tokenizer.tokenize(text)
        
        # Simulate translation noise by shuffling adjacent words occasionally
        augmented_tokens = tokens.copy()
        
        for i in range(len(augmented_tokens) - 1):
            if torch.rand(1).item() < noise_level:
                # Swap adjacent tokens
                augmented_tokens[i], augmented_tokens[i + 1] = augmented_tokens[i + 1], augmented_tokens[i]
        
        return ' '.join(augmented_tokens)

# Utility functions
def create_text_tensor_dataset(texts, labels, tokenizer, max_length=128):
    """Create tensor dataset from texts and labels"""
    encoded_data = []
    
    for text, label in zip(texts, labels):
        indices = tokenizer.encode(text, max_length=max_length)
        encoded_data.append({
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        })
    
    return encoded_data

def collate_text_batch(batch):
    """Collate function for text batches"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Create attention mask
    attention_mask = (input_ids != 0).long()  # Assuming 0 is PAD token
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def compute_text_statistics(texts, tokenizer):
    """Compute various statistics about the text data"""
    stats = {
        'num_texts': len(texts),
        'vocab_size': len(tokenizer.word_to_idx) if tokenizer.vocab_built else 0,
        'avg_length': 0,
        'max_length': 0,
        'min_length': float('inf'),
        'word_frequencies': collections.Counter()
    }
    
    lengths = []
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        length = len(tokens)
        lengths.append(length)
        
        stats['max_length'] = max(stats['max_length'], length)
        stats['min_length'] = min(stats['min_length'], length)
        stats['word_frequencies'].update(tokens)
    
    stats['avg_length'] = np.mean(lengths) if lengths else 0
    stats['std_length'] = np.std(lengths) if lengths else 0
    stats['length_percentiles'] = {
        'p25': np.percentile(lengths, 25) if lengths else 0,
        'p50': np.percentile(lengths, 50) if lengths else 0,
        'p75': np.percentile(lengths, 75) if lengths else 0,
        'p95': np.percentile(lengths, 95) if lengths else 0
    }
    
    return stats

if __name__ == "__main__":
    print("Testing text preprocessing with tensors...")
    
    # Sample texts
    texts = [
        "Hello world! This is a test sentence.",
        "Natural language processing with PyTorch is amazing.",
        "We are learning about text preprocessing.",
        "Tokenization and encoding are important steps."
    ]
    
    # Test basic tokenizer
    tokenizer = TextTokenizer(vocab_size=1000, min_freq=1)
    tokenizer.build_vocab(texts)
    
    # Test encoding and decoding
    sample_text = texts[0]
    encoded = tokenizer.encode(sample_text, max_length=20)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test tensor operations
    tensor_processor = TensorTextProcessor(len(tokenizer.word_to_idx))
    
    # Create batch of sequences
    batch_encoded = [tokenizer.encode(text, max_length=15) for text in texts]
    batch_tensor = torch.tensor(batch_encoded)
    
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    # Test masks
    padding_mask = tensor_processor.create_padding_mask(batch_tensor)
    attention_mask = tensor_processor.create_attention_mask(batch_tensor)
    
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test advanced processing
    advanced_processor = AdvancedTextProcessor(tokenizer)
    
    # Test batch encoding
    batch_result = advanced_processor.batch_encode(texts, max_length=15)
    print(f"Batch encoded input_ids shape: {batch_result['input_ids'].shape}")
    print(f"Batch encoded attention_mask shape: {batch_result['attention_mask'].shape}")
    
    # Test n-gram features
    ngram_features, ngram_vocab = advanced_processor.create_ngram_tensor(texts, n=2, max_features=50)
    print(f"N-gram features shape: {ngram_features.shape}")
    print(f"N-gram vocabulary size: {len(ngram_vocab)}")
    
    # Test text augmentation
    augmenter = TextAugmentation(tokenizer)
    
    # Test different augmentation techniques
    original = "This is a sample sentence for testing"
    
    dropout_aug = augmenter.random_word_dropout(original, dropout_prob=0.2)
    replacement_aug = augmenter.random_word_replacement(original, replacement_prob=0.2)
    insertion_aug = augmenter.random_word_insertion(original, insertion_prob=0.2)
    
    print(f"Original: {original}")
    print(f"Dropout augmentation: {dropout_aug}")
    print(f"Replacement augmentation: {replacement_aug}")
    print(f"Insertion augmentation: {insertion_aug}")
    
    # Test text statistics
    stats = compute_text_statistics(texts, tokenizer)
    print(f"Text statistics: {stats}")
    
    print("Text preprocessing with tensors testing completed!")