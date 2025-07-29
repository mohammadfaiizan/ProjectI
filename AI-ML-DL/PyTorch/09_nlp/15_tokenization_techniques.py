import torch
import torch.nn as nn
import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union
import unicodedata

# Character-Level Tokenizer
class CharacterTokenizer:
    """Character-level tokenizer"""
    
    def __init__(self, vocab_size=None, special_tokens=None):
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_built = False
    
    def build_vocab(self, texts):
        """Build character vocabulary from texts"""
        char_counts = Counter()
        
        for text in texts:
            char_counts.update(text)
        
        # Start with special tokens
        self.char_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.idx_to_char = {i: token for i, token in enumerate(self.special_tokens)}
        
        # Add most frequent characters
        most_common = char_counts.most_common(self.vocab_size - len(self.special_tokens) if self.vocab_size else None)
        
        for char, _ in most_common:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        self.vocab_built = True
        print(f"Character vocabulary built with {len(self.char_to_idx)} characters")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to character indices"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        chars = list(text)
        
        if add_special_tokens:
            chars = ['<BOS>'] + chars + ['<EOS>']
        
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in chars]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode indices back to text"""
        chars = [self.idx_to_char.get(idx, '<UNK>') for idx in indices]
        
        if skip_special_tokens:
            chars = [char for char in chars if char not in self.special_tokens]
        
        return ''.join(chars)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.char_to_idx = {k: int(v) for k, v in data['char_to_idx'].items()}
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.special_tokens = data['special_tokens']
        self.vocab_size = data['vocab_size']
        self.vocab_built = True

# Word-Level Tokenizer
class WordTokenizer:
    """Word-level tokenizer with preprocessing"""
    
    def __init__(self, vocab_size=None, min_freq=1, special_tokens=None, lowercase=True, 
                 remove_punctuation=False, normalize_unicode=True):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False
    
    def preprocess_text(self, text):
        """Preprocess text before tokenization"""
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        text = self.preprocess_text(text)
        # Simple whitespace tokenization (can be enhanced with regex)
        words = text.split()
        return words
    
    def build_vocab(self, texts):
        """Build word vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
        
        # Start with special tokens
        self.word_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.idx_to_word = {i: token for i, token in enumerate(self.special_tokens)}
        
        # Add words that meet minimum frequency
        filtered_words = [(word, count) for word, count in word_counts.items() if count >= self.min_freq]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        if self.vocab_size:
            max_words = self.vocab_size - len(self.special_tokens)
            filtered_words = filtered_words[:max_words]
        
        for word, _ in filtered_words:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"Word vocabulary built with {len(self.word_to_idx)} words")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to word indices"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        words = self.tokenize(text)
        
        if add_special_tokens:
            words = ['<BOS>'] + words + ['<EOS>']
        
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode indices back to text"""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        
        if skip_special_tokens:
            words = [word for word in words if word not in self.special_tokens]
        
        return ' '.join(words)

# Byte Pair Encoding (BPE) Tokenizer
class BPETokenizer:
    """Byte Pair Encoding tokenizer"""
    
    def __init__(self, vocab_size=1000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.merges = []
        self.vocab_built = False
    
    def get_word_tokens(self, text):
        """Get initial word tokens (characters with end-of-word marker)"""
        words = text.split()
        word_tokens = {}
        
        for word in words:
            # Add end-of-word marker
            token = ' '.join(list(word)) + ' </w>'
            word_tokens[token] = word_tokens.get(token, 0) + 1
        
        return word_tokens
    
    def get_pairs(self, word_tokens):
        """Get all adjacent pairs in the vocabulary"""
        pairs = defaultdict(int)
        
        for word, freq in word_tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return pairs
    
    def merge_vocab(self, pair, word_tokens):
        """Merge the most frequent pair in vocabulary"""
        new_word_tokens = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
        
        for word in word_tokens:
            new_word = p.sub(''.join(pair), word)
            new_word_tokens[new_word] = word_tokens[word]
        
        return new_word_tokens
    
    def build_vocab(self, texts):
        """Build BPE vocabulary"""
        # Combine all texts
        text = ' '.join(texts)
        
        # Get initial word tokens
        word_tokens = self.get_word_tokens(text)
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in word_tokens:
            vocab.update(word.split())
        
        # Start with special tokens
        self.token_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add initial characters
        for token in sorted(vocab):
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.token_to_idx)
        
        for i in range(num_merges):
            pairs = self.get_pairs(word_tokens)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_tokens = self.merge_vocab(best_pair, word_tokens)
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            new_token = ''.join(best_pair)
            if new_token not in self.token_to_idx:
                self.token_to_idx[new_token] = len(self.token_to_idx)
        
        # Create reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_built = True
        
        print(f"BPE vocabulary built with {len(self.token_to_idx)} tokens and {len(self.merges)} merges")
    
    def encode_word(self, word):
        """Encode a single word using BPE"""
        word = ' '.join(list(word)) + ' </w>'
        
        # Apply learned merges
        for pair in self.merges:
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
            word = p.sub(''.join(pair), word)
        
        return word.split()
    
    def encode(self, text, add_special_tokens=True):
        """Encode text using BPE"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        words = text.split()
        tokens = []
        
        if add_special_tokens:
            tokens.append('<BOS>')
        
        for word in words:
            word_tokens = self.encode_word(word)
            tokens.extend(word_tokens)
        
        if add_special_tokens:
            tokens.append('<EOS>')
        
        # Convert to indices
        indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) for token in tokens]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode BPE indices back to text"""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        # Join tokens and handle end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

# WordPiece Tokenizer (simplified version)
class WordPieceTokenizer:
    """Simplified WordPiece tokenizer"""
    
    def __init__(self, vocab_size=1000, special_tokens=None, unk_token='[UNK]', 
                 max_input_chars_per_word=200):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        
        self.vocab = {}
        self.ids_to_tokens = {}
        self.vocab_built = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Basic cleaning and normalization
        text = text.strip().lower()
        return text
    
    def build_vocab(self, texts):
        """Build WordPiece vocabulary (simplified)"""
        # For a full implementation, you would need the WordPiece algorithm
        # This is a simplified version that uses character n-grams
        
        char_counts = Counter()
        
        # Collect character statistics
        for text in texts:
            text = self.preprocess_text(text)
            char_counts.update(text)
        
        # Start with special tokens
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add single characters
        for char, count in char_counts.most_common():
            if len(self.vocab) >= self.vocab_size:
                break
            if char not in self.vocab and char.strip():
                self.vocab[char] = len(self.vocab)
        
        # Add common bigrams (simplified)
        bigram_counts = Counter()
        for text in texts:
            text = self.preprocess_text(text)
            for i in range(len(text) - 1):
                bigram = text[i:i+2]
                bigram_counts[bigram] += 1
        
        for bigram, count in bigram_counts.most_common():
            if len(self.vocab) >= self.vocab_size:
                break
            if bigram not in self.vocab and count > 1:
                # Add with WordPiece prefix
                token = '##' + bigram if not bigram.startswith(' ') else bigram
                self.vocab[token] = len(self.vocab)
        
        # Create reverse mapping
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        self.vocab_built = True
        
        print(f"WordPiece vocabulary built with {len(self.vocab)} tokens")
    
    def tokenize_word(self, word):
        """Tokenize a single word using greedy longest-match"""
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # Greedy longest-match
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                tokens.append(self.unk_token)
                start += 1
            else:
                tokens.append(cur_substr)
                start = end
        
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        """Encode text using WordPiece"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        text = self.preprocess_text(text)
        words = text.split()
        
        tokens = []
        if add_special_tokens:
            tokens.append('[CLS]')
        
        for word in words:
            word_tokens = self.tokenize_word(word)
            tokens.extend(word_tokens)
        
        if add_special_tokens:
            tokens.append('[SEP]')
        
        # Convert to indices
        indices = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode WordPiece indices back to text"""
        tokens = [self.ids_to_tokens.get(idx, self.unk_token) for idx in indices]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        # Join tokens and handle WordPiece markers
        text = ''
        for token in tokens:
            if token.startswith('##'):
                text += token[2:]
            else:
                if text:
                    text += ' '
                text += token
        
        return text

# SentencePiece-style Tokenizer (simplified)
class SentencePieceTokenizer:
    """Simplified SentencePiece-style tokenizer"""
    
    def __init__(self, vocab_size=1000, model_type='bpe', special_tokens=None):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.special_tokens = special_tokens or ['<unk>', '<s>', '</s>', '<pad>']
        
        self.sp_model = {}
        self.vocab_built = False
    
    def preprocess_text(self, text):
        """Preprocess text for SentencePiece"""
        # Add space prefix (SentencePiece treats space as a special character)
        return '▁' + text.replace(' ', '▁')
    
    def build_vocab(self, texts):
        """Build SentencePiece vocabulary (simplified BPE version)"""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Use BPE algorithm
        if self.model_type == 'bpe':
            bpe_tokenizer = BPETokenizer(self.vocab_size, self.special_tokens)
            bpe_tokenizer.build_vocab(processed_texts)
            
            # Adapt to SentencePiece format
            self.sp_model = {
                'token_to_idx': bpe_tokenizer.token_to_idx,
                'idx_to_token': bpe_tokenizer.idx_to_token,
                'merges': bpe_tokenizer.merges
            }
        
        self.vocab_built = True
        print(f"SentencePiece vocabulary built with {len(self.sp_model['token_to_idx'])} tokens")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text using SentencePiece"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Use BPE-style encoding
        tokens = []
        if add_special_tokens:
            tokens.append('<s>')
        
        # Simple tokenization (for full implementation, use trained SentencePiece model)
        words = processed_text.split('▁')[1:]  # Remove empty first element
        for word in words:
            if word:
                tokens.append('▁' + word)
        
        if add_special_tokens:
            tokens.append('</s>')
        
        # Convert to indices
        indices = [self.sp_model['token_to_idx'].get(token, self.sp_model['token_to_idx']['<unk>']) 
                  for token in tokens]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode SentencePiece indices back to text"""
        tokens = [self.sp_model['idx_to_token'].get(idx, '<unk>') for idx in indices]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        # Join tokens and handle space markers
        text = ''.join(tokens).replace('▁', ' ')
        return text.strip()

# Regex-based Tokenizer
class RegexTokenizer:
    """Regex-based tokenizer for flexible pattern matching"""
    
    def __init__(self, patterns=None, vocab_size=None, special_tokens=None):
        self.patterns = patterns or [
            r'\b\w+\b',  # Words
            r'\d+',      # Numbers
            r'[^\w\s]',  # Punctuation
        ]
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab_built = False
    
    def tokenize(self, text):
        """Tokenize text using regex patterns"""
        tokens = []
        
        for pattern in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            tokens.extend(matches)
            # Remove matched parts to avoid double tokenization
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        return [token for token in tokens if token.strip()]
    
    def build_vocab(self, texts):
        """Build vocabulary from tokenized texts"""
        token_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # Start with special tokens
        self.token_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.idx_to_token = {i: token for i, token in enumerate(self.special_tokens)}
        
        # Add most frequent tokens
        most_common = token_counts.most_common(
            self.vocab_size - len(self.special_tokens) if self.vocab_size else None
        )
        
        for token, _ in most_common:
            if token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
        
        self.vocab_built = True
        print(f"Regex vocabulary built with {len(self.token_to_idx)} tokens")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to indices"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        
        indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) for token in tokens]
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode indices back to text"""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        return ' '.join(tokens)

# Utility functions
def compare_tokenizers(text, tokenizers, names):
    """Compare different tokenizers on the same text"""
    print(f"Original text: {text}")
    print("=" * 80)
    
    for tokenizer, name in zip(tokenizers, names):
        if hasattr(tokenizer, 'vocab_built') and tokenizer.vocab_built:
            try:
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)
                
                print(f"{name}:")
                print(f"  Encoded: {encoded[:20]}...")  # Show first 20 tokens
                print(f"  Decoded: {decoded}")
                print(f"  Vocab size: {len(getattr(tokenizer, 'token_to_idx', getattr(tokenizer, 'char_to_idx', getattr(tokenizer, 'word_to_idx', {}))))}")
                print(f"  Token count: {len(encoded)}")
                print()
            except Exception as e:
                print(f"{name}: Error - {e}")
                print()

if __name__ == "__main__":
    print("Testing tokenization techniques...")
    
    # Sample texts for training
    texts = [
        "Hello world! This is a test sentence.",
        "Natural language processing is fascinating.",
        "Tokenization is an important preprocessing step.",
        "Different tokenizers have different advantages.",
        "Character-level models can handle out-of-vocabulary words.",
        "Subword tokenizers like BPE are very popular.",
        "WordPiece is used in BERT and related models.",
        "SentencePiece is language-agnostic and powerful."
    ]
    
    # Test text
    test_text = "This is a new sentence for testing tokenization!"
    
    # Initialize tokenizers
    char_tokenizer = CharacterTokenizer(vocab_size=100)
    word_tokenizer = WordTokenizer(vocab_size=50, min_freq=1)
    bpe_tokenizer = BPETokenizer(vocab_size=80)
    wordpiece_tokenizer = WordPieceTokenizer(vocab_size=80)
    regex_tokenizer = RegexTokenizer(vocab_size=60)
    
    # Build vocabularies
    print("Building vocabularies...")
    char_tokenizer.build_vocab(texts)
    word_tokenizer.build_vocab(texts)
    bpe_tokenizer.build_vocab(texts)
    wordpiece_tokenizer.build_vocab(texts)
    regex_tokenizer.build_vocab(texts)
    
    # Compare tokenizers
    tokenizers = [char_tokenizer, word_tokenizer, bpe_tokenizer, wordpiece_tokenizer, regex_tokenizer]
    names = ['Character', 'Word', 'BPE', 'WordPiece', 'Regex']
    
    compare_tokenizers(test_text, tokenizers, names)
    
    # Test saving and loading
    print("Testing save/load functionality...")
    char_tokenizer.save('char_tokenizer.json')
    
    new_char_tokenizer = CharacterTokenizer()
    new_char_tokenizer.load('char_tokenizer.json')
    
    # Verify it works
    original_encoded = char_tokenizer.encode(test_text)
    loaded_encoded = new_char_tokenizer.encode(test_text)
    
    print(f"Original encoding: {original_encoded[:10]}...")
    print(f"Loaded encoding:   {loaded_encoded[:10]}...")
    print(f"Encodings match: {original_encoded == loaded_encoded}")
    
    print("Tokenization techniques testing completed!")