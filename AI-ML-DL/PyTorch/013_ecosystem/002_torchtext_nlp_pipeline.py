import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from collections import Counter, defaultdict
import re
import string
import pickle

# Note: TorchText API has changed significantly. This demonstrates both legacy and modern approaches
# Install with: pip install torchtext

try:
    import torchtext
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    from torchtext.datasets import IMDB, AG_NEWS
    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False
    print("Warning: TorchText not available. Install with: pip install torchtext")

# Text Preprocessing Pipeline
class TextPreprocessor:
    """Comprehensive text preprocessing for NLP tasks"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.tokenizer = get_tokenizer('basic_english') if TORCHTEXT_AVAILABLE else self._basic_tokenizer
        self.vocab = None
        self.special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    def _basic_tokenizer(self, text: str) -> List[str]:
        """Basic tokenizer fallback when torchtext is not available"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        cleaned_text = self.clean_text(text)
        return self.tokenizer(cleaned_text)
    
    def build_vocabulary(self, text_iterator: Iterator[List[str]], 
                        min_freq: int = 2,
                        max_tokens: int = None) -> None:
        """Build vocabulary from text iterator"""
        
        if TORCHTEXT_AVAILABLE:
            self.vocab = build_vocab_from_iterator(
                text_iterator,
                min_freq=min_freq,
                max_tokens=max_tokens,
                specials=self.special_tokens
            )
            self.vocab.set_default_index(self.vocab['<unk>'])
        else:
            # Fallback vocabulary building
            word_counts = Counter()
            for tokens in text_iterator:
                word_counts.update(tokens)
            
            # Filter by minimum frequency
            filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
            
            # Limit vocabulary size
            if max_tokens:
                filtered_words = filtered_words[:max_tokens - len(self.special_tokens)]
            
            # Create vocabulary mapping
            vocab_dict = {token: idx for idx, token in enumerate(self.special_tokens)}
            vocab_dict.update({word: idx + len(self.special_tokens) 
                              for idx, word in enumerate(filtered_words)})
            
            self.vocab = vocab_dict
        
        print(f"✓ Vocabulary built with {len(self.vocab)} tokens")
    
    def text_to_indices(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices using vocabulary"""
        if TORCHTEXT_AVAILABLE and hasattr(self.vocab, '__call__'):
            return self.vocab(tokens)
        else:
            return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
    
    def indices_to_text(self, indices: List[int]) -> List[str]:
        """Convert indices back to tokens"""
        if TORCHTEXT_AVAILABLE and hasattr(self.vocab, 'get_itos'):
            itos = self.vocab.get_itos()
            return [itos[idx] for idx in indices]
        else:
            # Create reverse mapping
            itos = {idx: token for token, idx in self.vocab.items()}
            return [itos.get(idx, '<unk>') for idx in indices]
    
    def pad_sequences(self, sequences: List[List[int]], 
                     max_length: int = None,
                     padding_value: int = None) -> torch.Tensor:
        """Pad sequences to same length"""
        if padding_value is None:
            padding_value = self.vocab['<pad>'] if '<pad>' in self.vocab else 0
        
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) >= max_length:
                padded.append(seq[:max_length])
            else:
                padded.append(seq + [padding_value] * (max_length - len(seq)))
        
        return torch.tensor(padded, dtype=torch.long)

# Custom Dataset Classes
class TextClassificationDataset(torch.utils.data.Dataset):
    """Dataset for text classification tasks"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 preprocessor: TextPreprocessor,
                 max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        # Preprocess all texts
        self.processed_texts = []
        for text in texts:
            tokens = self.preprocessor.tokenize(text)
            indices = self.preprocessor.text_to_indices(tokens)
            self.processed_texts.append(indices[:max_length])  # Truncate if too long
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.processed_texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class LanguageModelingDataset(torch.utils.data.Dataset):
    """Dataset for language modeling tasks"""
    
    def __init__(self, text: str, preprocessor: TextPreprocessor, 
                 sequence_length: int = 128):
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        
        # Tokenize entire text
        tokens = self.preprocessor.tokenize(text)
        self.indices = self.preprocessor.text_to_indices(tokens)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.indices) - sequence_length, sequence_length):
            self.sequences.append(self.indices[i:i + sequence_length + 1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# Text Classification Models
class CNNTextClassifier(nn.Module):
    """CNN-based text classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_classes: int, num_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate all convolution outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class RNNTextClassifier(nn.Module):
    """RNN-based text classifier with attention"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 bidirectional: bool = True, rnn_type: str = 'LSTM'):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout,
                              bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout,
                             bidirectional=bidirectional)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Attention mechanism
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(rnn_output_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_output_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN forward pass
        rnn_output, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(rnn_output), dim=1)  # (batch_size, seq_len, 1)
        attended_output = torch.sum(attention_weights * rnn_output, dim=1)  # (batch_size, hidden_dim * 2)
        
        attended_output = self.dropout(attended_output)
        output = self.fc(attended_output)
        
        return output

class TransformerTextClassifier(nn.Module):
    """Transformer-based text classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 num_heads: int, num_layers: int, num_classes: int,
                 max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._create_positional_encoding(max_seq_length, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_seq_length: int, embedding_dim: int):
        """Create positional encoding"""
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                           -(np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_seq_length, embedding_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, seq_len, embedding_dim)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, embedding_dim)
        
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

# Advanced Text Processing
class TextAugmentation:
    """Text augmentation techniques"""
    
    @staticmethod
    def synonym_replacement(text: str, synonym_dict: Dict[str, List[str]], 
                          replacement_prob: float = 0.1) -> str:
        """Replace words with synonyms"""
        words = text.split()
        augmented_words = []
        
        for word in words:
            if word in synonym_dict and np.random.random() < replacement_prob:
                synonyms = synonym_dict[word]
                replacement = np.random.choice(synonyms)
                augmented_words.append(replacement)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    @staticmethod
    def random_insertion(text: str, vocab: List[str], 
                        insertion_prob: float = 0.1) -> str:
        """Randomly insert words"""
        words = text.split()
        num_insertions = max(1, int(len(words) * insertion_prob))
        
        for _ in range(num_insertions):
            random_word = np.random.choice(vocab)
            random_idx = np.random.randint(0, len(words) + 1)
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text: str, deletion_prob: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        if len(words) == 1:
            return text
        
        filtered_words = [word for word in words 
                         if np.random.random() > deletion_prob]
        
        if len(filtered_words) == 0:
            return words[np.random.randint(0, len(words))]
        
        return ' '.join(filtered_words)
    
    @staticmethod
    def random_swap(text: str, swap_prob: float = 0.1) -> str:
        """Randomly swap words"""
        words = text.split()
        if len(words) < 2:
            return text
        
        num_swaps = max(1, int(len(words) * swap_prob))
        
        for _ in range(num_swaps):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)

# Named Entity Recognition
class NERModel(nn.Module):
    """Named Entity Recognition model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, num_tags: int, num_layers: int = 2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

# Sentiment Analysis Pipeline
class SentimentAnalyzer:
    """Complete sentiment analysis pipeline"""
    
    def __init__(self, model_type: str = 'rnn'):
        self.preprocessor = TextPreprocessor()
        self.model_type = model_type
        self.model = None
        self.trained = False
    
    def prepare_data(self, texts: List[str], labels: List[int],
                    train_split: float = 0.8) -> Tuple:
        """Prepare data for training"""
        
        # Build vocabulary
        tokenized_texts = [self.preprocessor.tokenize(text) for text in texts]
        self.preprocessor.build_vocabulary(iter(tokenized_texts))
        
        # Create dataset
        dataset = TextClassificationDataset(texts, labels, self.preprocessor)
        
        # Split data
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        return train_dataset, val_dataset
    
    def build_model(self, vocab_size: int, num_classes: int,
                   embedding_dim: int = 128, hidden_dim: int = 128) -> None:
        """Build model based on specified type"""
        
        if self.model_type == 'cnn':
            self.model = CNNTextClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_classes=num_classes
            )
        elif self.model_type == 'rnn':
            self.model = RNNTextClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
        elif self.model_type == 'transformer':
            self.model = TransformerTextClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_heads=8,
                num_layers=4,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print(f"✓ {self.model_type.upper()} model created")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 10, lr: float = 0.001) -> None:
        """Train the sentiment analysis model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx >= 10:  # Limit for demo
                    break
            
            # Validation
            val_accuracy = self._evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss = {epoch_loss/num_batches:.4f}, "
                  f"Val Acc = {val_accuracy:.4f}")
        
        self.trained = True
        print("✓ Training completed")
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model accuracy"""
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                if batch_idx >= 5:  # Limit for demo
                    break
        
        self.model.train()
        return correct / total if total > 0 else 0.0
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment for new texts"""
        
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                tokens = self.preprocessor.tokenize(text)
                indices = self.preprocessor.text_to_indices(tokens)
                input_tensor = torch.tensor([indices], dtype=torch.long)
                
                output = self.model(input_tensor)
                predicted = torch.argmax(output, dim=1).item()
                predictions.append(predicted)
        
        return predictions

# Data Collation Functions
def collate_classification(batch):
    """Collate function for text classification"""
    texts, labels = zip(*batch)
    
    # Pad sequences
    max_len = max(len(text) for text in texts)
    padded_texts = []
    
    for text in texts:
        if len(text) < max_len:
            padded = torch.cat([text, torch.zeros(max_len - len(text), dtype=torch.long)])
        else:
            padded = text[:max_len]
        padded_texts.append(padded)
    
    return torch.stack(padded_texts), torch.tensor(labels)

def collate_language_modeling(batch):
    """Collate function for language modeling"""
    inputs, targets = zip(*batch)
    
    # All sequences should be the same length for language modeling
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    return inputs, targets

if __name__ == "__main__":
    print("Comprehensive TorchText NLP Pipeline")
    print("=" * 40)
    
    if not TORCHTEXT_AVAILABLE:
        print("TorchText not available. Demonstrating with fallback implementations.")
    
    print("\n1. Text Preprocessing")
    print("-" * 24)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Sample texts
    sample_texts = [
        "This is a great movie with excellent acting!",
        "The film was terrible and boring.",
        "I loved the storyline and characters.",
        "Worst movie I've ever seen.",
        "Amazing cinematography and soundtrack.",
        "Not worth watching at all.",
        "Brilliant performances by all actors.",
        "Very disappointing and overrated."
    ]
    
    # Clean and tokenize
    tokenized_texts = []
    for text in sample_texts:
        cleaned = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(text)
        tokenized_texts.append(tokens)
        print(f"Original: {text}")
        print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
        print()
    
    # Build vocabulary
    preprocessor.build_vocabulary(iter(tokenized_texts), min_freq=1)
    
    print(f"Vocabulary size: {len(preprocessor.vocab)}")
    
    # Convert to indices
    sample_indices = preprocessor.text_to_indices(tokenized_texts[0])
    print(f"Sample text as indices: {sample_indices}")
    
    # Convert back to text
    recovered_tokens = preprocessor.indices_to_text(sample_indices)
    print(f"Recovered tokens: {recovered_tokens}")
    
    print("\n2. Model Architectures")
    print("-" * 24)
    
    vocab_size = len(preprocessor.vocab)
    num_classes = 2  # Binary sentiment
    
    # Create different model types
    models = {
        'CNN': CNNTextClassifier(vocab_size, embedding_dim=64, num_classes=num_classes),
        'RNN': RNNTextClassifier(vocab_size, embedding_dim=64, hidden_dim=64, num_classes=num_classes),
        'Transformer': TransformerTextClassifier(vocab_size, embedding_dim=64, num_heads=4, 
                                               num_layers=2, num_classes=num_classes)
    }
    
    # Test forward pass
    sample_input = torch.randint(0, vocab_size, (2, 20))  # Batch of 2, sequence length 20
    
    for name, model in models.items():
        try:
            output = model(sample_input)
            print(f"{name} output shape: {output.shape}")
        except Exception as e:
            print(f"{name} error: {e}")
    
    print("\n3. Text Augmentation")
    print("-" * 22)
    
    augmentation = TextAugmentation()
    
    original_text = "This movie is really amazing and entertaining"
    
    # Sample synonym dictionary
    synonym_dict = {
        'amazing': ['fantastic', 'wonderful', 'great'],
        'movie': ['film', 'picture', 'flick'],
        'really': ['very', 'extremely', 'quite']
    }
    
    # Apply different augmentations
    synonym_aug = augmentation.synonym_replacement(original_text, synonym_dict, 0.5)
    deletion_aug = augmentation.random_deletion(original_text, 0.2)
    swap_aug = augmentation.random_swap(original_text, 0.3)
    
    print(f"Original: {original_text}")
    print(f"Synonym replacement: {synonym_aug}")
    print(f"Random deletion: {deletion_aug}")
    print(f"Random swap: {swap_aug}")
    
    print("\n4. Sentiment Analysis Pipeline")
    print("-" * 33)
    
    # Create sentiment analyzer
    analyzer = SentimentAnalyzer(model_type='rnn')
    
    # Sample data
    texts = sample_texts
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    
    # Prepare data
    try:
        train_dataset, val_dataset = analyzer.prepare_data(texts, labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_classification)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_classification)
        
        # Build model
        analyzer.build_model(
            vocab_size=len(analyzer.preprocessor.vocab),
            num_classes=2,
            embedding_dim=32,
            hidden_dim=32
        )
        
        # Train model
        analyzer.train(train_loader, val_loader, epochs=3, lr=0.01)
        
        # Test prediction
        test_texts = ["This movie is fantastic!", "I hate this film."]
        predictions = analyzer.predict(test_texts)
        
        for text, pred in zip(test_texts, predictions):
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {text}")
            print(f"Predicted sentiment: {sentiment}")
    
    except Exception as e:
        print(f"Sentiment analysis demo: {e}")
    
    print("\n5. Advanced Features")
    print("-" * 22)
    
    # Named Entity Recognition model
    ner_model = NERModel(vocab_size=vocab_size, embedding_dim=64, 
                        hidden_dim=64, num_tags=5)  # 5 NER tags
    
    # Test NER model
    sample_ner_input = torch.randint(0, vocab_size, (1, 15))
    ner_output = ner_model(sample_ner_input)
    print(f"NER model output shape: {ner_output.shape}")
    
    print("\n6. TorchText Best Practices")
    print("-" * 31)
    
    best_practices = [
        "Use appropriate tokenization for your language/domain",
        "Build vocabulary with proper frequency filtering",
        "Handle out-of-vocabulary words with <unk> tokens",
        "Apply proper text cleaning and normalization",
        "Use padding for batch processing",
        "Implement text augmentation for data scarce scenarios",
        "Choose appropriate model architecture for your task",
        "Use attention mechanisms for longer sequences",
        "Apply dropout and regularization for generalization",
        "Monitor training/validation metrics carefully",
        "Use pretrained embeddings when available",
        "Handle class imbalance in classification tasks"
    ]
    
    print("TorchText NLP Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Common NLP Tasks")
    print("-" * 21)
    
    nlp_tasks = {
        "Text Classification": "Categorize text into predefined classes",
        "Sentiment Analysis": "Determine emotional tone of text",
        "Named Entity Recognition": "Identify and classify named entities",
        "Language Modeling": "Predict next words in sequence",
        "Machine Translation": "Translate text between languages",
        "Text Summarization": "Generate concise summaries",
        "Question Answering": "Answer questions based on context",
        "Text Generation": "Generate coherent text sequences"
    }
    
    print("Common NLP Tasks:")
    for task, description in nlp_tasks.items():
        print(f"  {task}: {description}")
    
    print("\n8. Integration with Other Libraries")
    print("-" * 38)
    
    integrations = [
        "Hugging Face Transformers: Pretrained transformer models",
        "spaCy: Advanced NLP processing and pipelines",
        "NLTK: Natural language processing toolkit",
        "Gensim: Topic modeling and word embeddings",
        "FastText: Efficient text classification and embeddings",
        "SentencePiece: Subword tokenization",
        "Weights & Biases: Experiment tracking for NLP",
        "TensorBoard: Visualization and monitoring"
    ]
    
    print("Common NLP Integrations:")
    for integration in integrations:
        print(f"  - {integration}")
    
    print("\nTorchText NLP pipeline demonstration completed!")
    print("Key components covered:")
    print("  - Text preprocessing and tokenization")
    print("  - Vocabulary building and management")
    print("  - Multiple model architectures (CNN, RNN, Transformer)")
    print("  - Text augmentation techniques")
    print("  - Complete sentiment analysis pipeline")
    print("  - Named entity recognition")
    print("  - Best practices and integration patterns")
    
    print("\nTorchText enables:")
    print("  - Efficient text preprocessing")
    print("  - Vocabulary management")
    print("  - Dataset creation and batching")
    print("  - Integration with PyTorch models")
    print("  - End-to-end NLP pipelines")