#!/usr/bin/env python3
"""PyTorch Data Transforms NLP - Text preprocessing transforms"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
import re
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union, Callable
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

print("=== NLP Transforms Overview ===")

print("NLP transform types covered:")
print("1. Text cleaning and normalization")
print("2. Tokenization strategies")
print("3. Vocabulary building and encoding")
print("4. Sequence transformations")
print("5. Text augmentation techniques")
print("6. Custom text transforms")
print("7. Multi-language processing")
print("8. Performance optimization")

print("\n=== Text Cleaning and Normalization ===")

class TextCleaning:
    """Collection of text cleaning transforms"""
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text: str, keep_chars: str = "") -> str:
        """Remove punctuation from text"""
        translator = str.maketrans("", "", string.punctuation.replace(keep_chars, ""))
        return text.translate(translator)
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove all numbers from text"""
        return re.sub(r'\d+', '', text)
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace and normalize spacing"""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters"""
        import unicodedata
        return unicodedata.normalize('NFKD', text)
    
    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand common contractions"""
        contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text

# Test text cleaning
print("Testing text cleaning transforms:")

sample_texts = [
    "Hello World! This is a TEST... with NUMBERS 123 and URLs http://example.com",
    "  Extra   spaces    everywhere   ",
    "HTML tags <b>bold</b> and <i>italic</i> text",
    "Contractions: I'm can't won't they're"
]

cleaning_methods = [
    ('lowercase', TextCleaning.lowercase),
    ('remove_punctuation', TextCleaning.remove_punctuation),
    ('remove_numbers', TextCleaning.remove_numbers),
    ('remove_extra_whitespace', TextCleaning.remove_extra_whitespace),
    ('remove_urls', TextCleaning.remove_urls),
    ('remove_html_tags', TextCleaning.remove_html_tags),
    ('expand_contractions', TextCleaning.expand_contractions),
]

for text_idx, text in enumerate(sample_texts[:2]):  # Test first 2 samples
    print(f"\nSample {text_idx + 1}: '{text}'")
    for method_name, method in cleaning_methods:
        try:
            cleaned = method(text)
            print(f"  {method_name:20}: '{cleaned}'")
        except Exception as e:
            print(f"  {method_name:20}: Error - {e}")

print("\n=== Tokenization Strategies ===")

class Tokenizers:
    """Different tokenization approaches"""
    
    @staticmethod
    def whitespace_tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.split()
    
    @staticmethod
    def punctuation_aware_tokenize(text: str) -> List[str]:
        """Tokenize while preserving punctuation as separate tokens"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
    
    @staticmethod
    def subword_tokenize(text: str, vocab_size: int = 1000) -> List[str]:
        """Simple subword tokenization (BPE-like)"""
        # This is a simplified version for demonstration
        words = text.split()
        subwords = []
        
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Split into smaller parts
                mid = len(word) // 2
                subwords.extend([word[:mid], word[mid:]])
        
        return subwords
    
    @staticmethod
    def ngram_tokenize(text: str, n: int = 2) -> List[str]:
        """Create n-gram tokens"""
        words = text.split()
        if len(words) < n:
            return words
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    @staticmethod
    def character_tokenize(text: str) -> List[str]:
        """Character-level tokenization"""
        return list(text)

# Test tokenization
print("Testing tokenization strategies:")

sample_text = "Hello world! This is a test sentence with punctuation."

tokenization_methods = [
    ('Whitespace', Tokenizers.whitespace_tokenize),
    ('Punctuation-aware', Tokenizers.punctuation_aware_tokenize),
    ('Subword', Tokenizers.subword_tokenize),
    ('Bigram', lambda x: Tokenizers.ngram_tokenize(x, 2)),
    ('Character', Tokenizers.character_tokenize),
]

print(f"Original text: '{sample_text}'")
for method_name, method in tokenization_methods:
    try:
        tokens = method(sample_text)
        print(f"  {method_name:15}: {len(tokens)} tokens - {tokens[:5]}...")
    except Exception as e:
        print(f"  {method_name:15}: Error - {e}")

print("\n=== Vocabulary Building and Encoding ===")

class VocabularyBuilder:
    """Build and manage vocabularies for text data"""
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
    
    def build_vocab(self, texts: List[str], tokenizer: Callable = None):
        """Build vocabulary from list of texts"""
        if tokenizer is None:
            tokenizer = str.split
        
        # Count word frequencies
        for text in texts:
            tokens = tokenizer(text)
            self.word_counts.update(tokens)
        
        # Create word-to-index mapping
        self.word_to_idx = self.special_tokens.copy()
        
        # Add words based on frequency
        most_common = self.word_counts.most_common(self.max_vocab_size - len(self.special_tokens))
        for word, count in most_common:
            if count >= self.min_freq:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
        
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"  Built vocabulary with {len(self.word_to_idx)} words")
        return self
    
    def encode(self, text: str, tokenizer: Callable = None, max_length: int = None) -> List[int]:
        """Encode text to list of indices"""
        if tokenizer is None:
            tokenizer = str.split
        
        tokens = tokenizer(text)
        indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
        # Add special tokens
        indices = [self.word_to_idx['<BOS>']] + indices + [self.word_to_idx['<EOS>']]
        
        # Truncate or pad
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.word_to_idx['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decode list of indices back to text"""
        tokens = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        
        if remove_special:
            special_token_set = set(self.special_tokens.keys())
            tokens = [token for token in tokens if token not in special_token_set]
        
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        return len(self.word_to_idx)

# Test vocabulary building
print("Testing vocabulary building:")

sample_corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog was very lazy and slept all day",
    "a quick brown fox is a beautiful animal",
    "dogs and foxes are both animals that run fast"
]

vocab_builder = VocabularyBuilder(max_vocab_size=20, min_freq=1)
vocab_builder.build_vocab(sample_corpus)

print(f"  Vocabulary size: {vocab_builder.get_vocab_size()}")
print(f"  Word to index mapping (first 10): {list(vocab_builder.word_to_idx.items())[:10]}")

# Test encoding/decoding
test_text = "the quick fox runs fast"
encoded = vocab_builder.encode(test_text, max_length=10)
decoded = vocab_builder.decode(encoded)

print(f"  Original: '{test_text}'")
print(f"  Encoded: {encoded}")
print(f"  Decoded: '{decoded}'")

print("\n=== Sequence Transformations ===")

class SequenceTransforms:
    """Transforms for sequence data"""
    
    @staticmethod
    def pad_sequences(sequences: List[List[int]], max_length: int = None, 
                     pad_value: int = 0, truncate: str = 'post') -> torch.Tensor:
        """Pad sequences to uniform length"""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) > max_length:
                if truncate == 'post':
                    seq = seq[:max_length]
                else:  # 'pre'
                    seq = seq[-max_length:]
            else:
                padding = [pad_value] * (max_length - len(seq))
                seq = seq + padding
            
            padded.append(seq)
        
        return torch.tensor(padded, dtype=torch.long)
    
    @staticmethod
    def sliding_window(sequence: List, window_size: int, stride: int = 1) -> List[List]:
        """Create sliding windows over sequence"""
        windows = []
        for i in range(0, len(sequence) - window_size + 1, stride):
            windows.append(sequence[i:i + window_size])
        return windows
    
    @staticmethod
    def random_crop(sequence: List, crop_size: int) -> List:
        """Randomly crop sequence"""
        if len(sequence) <= crop_size:
            return sequence
        
        start_idx = random.randint(0, len(sequence) - crop_size)
        return sequence[start_idx:start_idx + crop_size]
    
    @staticmethod
    def reverse_sequence(sequence: List) -> List:
        """Reverse sequence order"""
        return sequence[::-1]
    
    @staticmethod
    def shuffle_sequence(sequence: List, preserve_length: bool = True) -> List:
        """Shuffle sequence elements"""
        shuffled = sequence.copy()
        random.shuffle(shuffled)
        return shuffled

# Test sequence transforms
print("Testing sequence transformations:")

# Create sample sequences
sample_sequences = [
    [1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14, 15, 16],
    [20, 21, 22]
]

print("Original sequences:")
for i, seq in enumerate(sample_sequences):
    print(f"  Sequence {i}: {seq}")

# Pad sequences
padded = SequenceTransforms.pad_sequences(sample_sequences, max_length=8)
print(f"\nPadded sequences (max_length=8):")
print(f"  Shape: {padded.shape}")
print(f"  Content:\n{padded}")

# Test other transforms
test_sequence = list(range(1, 11))  # [1, 2, 3, ..., 10]
print(f"\nTest sequence: {test_sequence}")

transforms_to_test = [
    ('Sliding window (size=3)', lambda x: SequenceTransforms.sliding_window(x, 3)),
    ('Random crop (size=5)', lambda x: SequenceTransforms.random_crop(x, 5)),
    ('Reverse', SequenceTransforms.reverse_sequence),
    ('Shuffle', SequenceTransforms.shuffle_sequence),
]

for name, transform in transforms_to_test:
    result = transform(test_sequence)
    print(f"  {name:25}: {result}")

print("\n=== Text Augmentation Techniques ===")

class TextAugmentation:
    """Text augmentation for data enhancement"""
    
    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace words with synonyms (simplified version)"""
        # This is a simplified implementation
        # In practice, you'd use WordNet or word embeddings
        simple_synonyms = {
            'good': ['great', 'excellent', 'wonderful'],
            'bad': ['terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'miniature'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['sluggish', 'gradual', 'leisurely']
        }
        
        words = text.split()
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower()
            if word in simple_synonyms:
                words[idx] = random.choice(simple_synonyms[word])
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert words"""
        words = text.split()
        
        for _ in range(n):
            # Insert a random word from the text
            if words:
                new_word = random.choice(words)
                idx = random.randint(0, len(words))
                words.insert(idx, new_word)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap words"""
        words = text.split()
        
        for _ in range(n):
            if len(words) < 2:
                break
            
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        return ' '.join(new_words) if new_words else text
    
    def back_translation(self, text: str) -> str:
        """Simulate back translation (placeholder)"""
        # In practice, this would use translation APIs
        # For demo, we'll just paraphrase slightly
        paraphrases = {
            'the': 'a',
            'is': 'was',
            'are': 'were',
            'good': 'nice',
            'bad': 'poor'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in paraphrases and random.random() < 0.3:
                words[i] = paraphrases[word.lower()]
        
        return ' '.join(words)
    
    def character_level_noise(self, text: str, p: float = 0.05) -> str:
        """Add character-level noise"""
        chars = list(text)
        
        for i in range(len(chars)):
            if random.random() < p:
                if chars[i].isalpha():
                    # Replace with random character
                    chars[i] = random.choice(string.ascii_lowercase)
        
        return ''.join(chars)

# Test text augmentation
print("Testing text augmentation techniques:")

augmenter = TextAugmentation()
original_text = "The quick brown fox jumps over the lazy dog"

augmentation_methods = [
    ('Synonym replacement', lambda x: augmenter.synonym_replacement(x, n=2)),
    ('Random insertion', lambda x: augmenter.random_insertion(x, n=1)),
    ('Random swap', lambda x: augmenter.random_swap(x, n=2)),
    ('Random deletion', lambda x: augmenter.random_deletion(x, p=0.2)),
    ('Back translation', augmenter.back_translation),
    ('Character noise', lambda x: augmenter.character_level_noise(x, p=0.1)),
]

print(f"Original: '{original_text}'")
for method_name, method in augmentation_methods:
    try:
        augmented = method(original_text)
        print(f"  {method_name:18}: '{augmented}'")
    except Exception as e:
        print(f"  {method_name:18}: Error - {e}")

print("\n=== Custom Text Transform Pipeline ===")

class TextTransformPipeline:
    """Customizable text transform pipeline"""
    
    def __init__(self, transforms: List[Callable] = None):
        self.transforms = transforms or []
    
    def add_transform(self, transform: Callable):
        """Add transform to pipeline"""
        self.transforms.append(transform)
        return self
    
    def __call__(self, text: str) -> str:
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            text = transform(text)
        return text
    
    def apply_with_probability(self, text: str, probabilities: List[float] = None) -> str:
        """Apply transforms with given probabilities"""
        if probabilities is None:
            probabilities = [1.0] * len(self.transforms)
        
        for transform, prob in zip(self.transforms, probabilities):
            if random.random() < prob:
                text = transform(text)
        
        return text

# Test custom pipeline
print("Testing custom text transform pipeline:")

# Create preprocessing pipeline
preprocess_pipeline = TextTransformPipeline([
    TextCleaning.lowercase,
    TextCleaning.expand_contractions,
    TextCleaning.remove_extra_whitespace,
    lambda x: TextCleaning.remove_punctuation(x, keep_chars="'")
])

# Create augmentation pipeline
augment_pipeline = TextTransformPipeline([
    lambda x: augmenter.synonym_replacement(x, n=1),
    lambda x: augmenter.random_swap(x, n=1),
])

test_text = "Hello! I'm testing THIS pipeline... It's quite INTERESTING!!!"

print(f"Original: '{test_text}'")
preprocessed = preprocess_pipeline(test_text)
print(f"Preprocessed: '{preprocessed}'")

augmented = augment_pipeline.apply_with_probability(preprocessed, [0.7, 0.5])
print(f"Augmented: '{augmented}'")

print("\n=== Multi-Language Processing ===")

class MultiLanguageProcessor:
    """Handle multiple languages in text processing"""
    
    def __init__(self):
        self.language_patterns = {
            'english': re.compile(r'[a-zA-Z]'),
            'chinese': re.compile(r'[\u4e00-\u9fff]'),
            'japanese': re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),
            'arabic': re.compile(r'[\u0600-\u06ff]'),
            'russian': re.compile(r'[\u0400-\u04ff]'),
        }
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = len(pattern.findall(text))
            scores[lang] = matches / max(len(text), 1)
        
        return max(scores, key=scores.get)
    
    def normalize_text(self, text: str, language: str = None) -> str:
        """Language-specific text normalization"""
        if language is None:
            language = self.detect_language(text)
        
        if language == 'chinese':
            # Remove spaces between Chinese characters
            text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        elif language == 'arabic':
            # Normalize Arabic text (simplified)
            text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
        
        return text
    
    def segment_mixed_text(self, text: str) -> List[Tuple[str, str]]:
        """Segment text by language"""
        segments = []
        current_segment = ""
        current_lang = None
        
        for char in text:
            char_lang = self.detect_language(char)
            
            if char_lang != current_lang:
                if current_segment:
                    segments.append((current_segment, current_lang))
                current_segment = char
                current_lang = char_lang
            else:
                current_segment += char
        
        if current_segment:
            segments.append((current_segment, current_lang))
        
        return segments

# Test multi-language processing
print("Testing multi-language processing:")

multi_processor = MultiLanguageProcessor()

test_texts = [
    "Hello world this is English text",
    "这是中文文本示例",
    "こんにちは世界",
    "Mixed text with English and 中文 characters"
]

for text in test_texts:
    detected_lang = multi_processor.detect_language(text)
    normalized = multi_processor.normalize_text(text)
    print(f"  Text: '{text}'")
    print(f"    Detected language: {detected_lang}")
    print(f"    Normalized: '{normalized}'")

print("\n=== Performance Optimization ===")

class OptimizedTextProcessor:
    """Optimized text processing for large datasets"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.transform_cache = {}
        self.vocab_cache = {}
    
    def cached_transform(self, text: str, transform_fn: Callable) -> str:
        """Apply transform with caching"""
        cache_key = (text, transform_fn.__name__)
        
        if cache_key in self.transform_cache:
            return self.transform_cache[cache_key]
        
        result = transform_fn(text)
        
        # Maintain cache size
        if len(self.transform_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.transform_cache))
            del self.transform_cache[oldest_key]
        
        self.transform_cache[cache_key] = result
        return result
    
    def batch_process(self, texts: List[str], transform_fn: Callable, 
                     batch_size: int = 100) -> List[str]:
        """Process texts in batches for memory efficiency"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [transform_fn(text) for text in batch]
            results.extend(batch_results)
        
        return results
    
    def parallel_tokenize(self, texts: List[str], tokenizer: Callable) -> List[List[str]]:
        """Tokenize texts in parallel (simplified simulation)"""
        # In practice, you'd use multiprocessing or threading
        return [tokenizer(text) for text in texts]

# Test performance optimization
print("Testing performance optimization:")

import time

optimizer = OptimizedTextProcessor(cache_size=100)

# Create test data
test_texts = [f"This is test sentence number {i}" for i in range(50)]

# Test cached transforms
print("Benchmarking cached vs non-cached transforms:")

# Non-cached
start_time = time.time()
for text in test_texts:
    _ = TextCleaning.lowercase(text)
non_cached_time = time.time() - start_time

# Cached (run twice to see caching effect)
start_time = time.time()
for text in test_texts:
    _ = optimizer.cached_transform(text, TextCleaning.lowercase)
for text in test_texts:  # Second pass should be faster
    _ = optimizer.cached_transform(text, TextCleaning.lowercase)
cached_time = time.time() - start_time

print(f"  Non-cached time: {non_cached_time*1000:.2f} ms")
print(f"  Cached time: {cached_time*1000:.2f} ms")
print(f"  Cache hit ratio: {len(optimizer.transform_cache)} entries")

print("\n=== NLP Transforms Best Practices ===")

print("Text Preprocessing Guidelines:")
print("1. Clean text consistently across train/test sets")
print("2. Handle edge cases (empty strings, special characters)")
print("3. Consider domain-specific preprocessing needs")
print("4. Preserve important information during cleaning")
print("5. Validate preprocessing results on sample data")

print("\nTokenization Best Practices:")
print("1. Choose tokenization strategy based on task")
print("2. Handle out-of-vocabulary words appropriately")
print("3. Consider subword tokenization for rare words")
print("4. Maintain consistency in tokenization")
print("5. Balance vocabulary size vs coverage")

print("\nAugmentation Guidelines:")
print("1. Apply augmentation only to training data")
print("2. Preserve semantic meaning during augmentation")
print("3. Test augmentation impact on model performance")
print("4. Use appropriate augmentation strength")
print("5. Consider task-specific augmentation strategies")

print("\nPerformance Tips:")
print("1. Cache expensive transformations")
print("2. Process texts in batches for efficiency")
print("3. Use vectorized operations when possible")
print("4. Consider parallel processing for large datasets")
print("5. Profile text processing pipelines")

print("\nCommon Pitfalls:")
print("1. Over-aggressive text cleaning")
print("2. Inconsistent preprocessing between stages")
print("3. Information loss during normalization")
print("4. Poor handling of special tokens")
print("5. Memory issues with large vocabularies")

print("\n=== NLP Transforms Complete ===")

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None