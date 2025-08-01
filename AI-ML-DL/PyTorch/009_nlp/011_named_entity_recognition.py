import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import zip_longest

# CRF Layer
class CRF(nn.Module):
    """Conditional Random Field for sequence labeling"""
    
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition parameters: transitions[i, j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end tag indices
        self.start_tag_idx = num_tags
        self.end_tag_idx = num_tags + 1
        
        # Expand transitions to include start and end tags
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        
        # Initialize transitions
        self.init_weights()
    
    def init_weights(self):
        """Initialize transition weights"""
        # Don't allow transitions to start tag
        self.transitions.data[:, self.start_tag_idx] = -10000
        # Don't allow transitions from end tag
        self.transitions.data[self.end_tag_idx, :] = -10000
    
    def _compute_partition_function(self, emissions, mask):
        """Compute the partition function in log space using forward algorithm"""
        batch_size, seq_length, num_tags = emissions.shape
        
        # Initialize forward variables
        forward_var = emissions[:, 0].clone()  # [batch_size, num_tags]
        forward_var[:, :] = -10000
        # Start tag
        forward_var[:, self.start_tag_idx] = 0
        
        for i in range(1, seq_length):
            emit_score = emissions[:, i]  # [batch_size, num_tags]
            
            # Broadcast for transition scores
            forward_var_expanded = forward_var.unsqueeze(2)  # [batch_size, num_tags, 1]
            transition_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            
            # Compute forward scores
            next_tag_var = forward_var_expanded + transition_scores + emit_score.unsqueeze(1)
            forward_var = torch.logsumexp(next_tag_var, dim=1)
            
            # Handle masking
            if mask is not None:
                mask_i = mask[:, i].unsqueeze(1)  # [batch_size, 1]
                forward_var = torch.where(mask_i, forward_var, forward_var)
        
        # Add transition to end tag
        terminal_var = forward_var + self.transitions[self.end_tag_idx, :].unsqueeze(0)
        partition_function = torch.logsumexp(terminal_var, dim=1)
        
        return partition_function
    
    def _compute_score(self, emissions, tags, mask):
        """Compute the score of a given tag sequence"""
        batch_size, seq_length = tags.shape
        
        # Add start and end tags
        start_tag = torch.full((batch_size, 1), self.start_tag_idx, 
                              dtype=torch.long, device=tags.device)
        end_tag = torch.full((batch_size, 1), self.end_tag_idx,
                            dtype=torch.long, device=tags.device)
        
        padded_tags = torch.cat([start_tag, tags, end_tag], dim=1)
        
        score = torch.zeros(batch_size, device=tags.device)
        
        # Emission scores
        for i in range(seq_length):
            if mask is None or mask[:, i].all():
                score += emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
        
        # Transition scores
        for i in range(seq_length + 1):
            if mask is None or (i == 0 or mask[:, i-1].all()):
                current_tag = padded_tags[:, i]
                next_tag = padded_tags[:, i + 1]
                score += self.transitions[current_tag, next_tag]
        
        return score
    
    def forward(self, emissions, tags, mask=None):
        """Compute CRF loss"""
        # Compute partition function
        partition = self._compute_partition_function(emissions, mask)
        
        # Compute score of given tag sequence
        score = self._compute_score(emissions, tags, mask)
        
        # Return negative log likelihood
        return (partition - score).mean()
    
    def decode(self, emissions, mask=None):
        """Viterbi decoding to find the best tag sequence"""
        batch_size, seq_length, num_tags = emissions.shape
        
        # Initialize Viterbi variables
        viterbi_vars = emissions[:, 0].clone()  # [batch_size, num_tags]
        viterbi_vars[:, :] = -10000
        viterbi_vars[:, self.start_tag_idx] = 0
        
        # Store the best paths
        paths = []
        
        for i in range(1, seq_length):
            emit_score = emissions[:, i]  # [batch_size, num_tags]
            
            # Expand for broadcasting
            viterbi_vars_expanded = viterbi_vars.unsqueeze(2)  # [batch_size, num_tags, 1]
            transition_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            
            # Compute scores
            next_tag_var = viterbi_vars_expanded + transition_scores
            best_tag_ids = torch.argmax(next_tag_var, dim=1)  # [batch_size, num_tags]
            paths.append(best_tag_ids)
            
            viterbi_vars = torch.max(next_tag_var, dim=1)[0] + emit_score
        
        # Transition to end tag
        terminal_var = viterbi_vars + self.transitions[self.end_tag_idx, :].unsqueeze(0)
        best_tag_id = torch.argmax(terminal_var, dim=1)
        
        # Backtrack
        best_paths = []
        for batch_idx in range(batch_size):
            best_path = [best_tag_id[batch_idx].item()]
            
            for path in reversed(paths):
                best_tag_id = path[batch_idx, best_tag_id]
                best_path.append(best_tag_id.item())
            
            # Remove start tag and reverse
            best_path = best_path[1:]
            best_path.reverse()
            best_paths.append(best_path)
        
        return best_paths

# BiLSTM-CRF Model
class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for Named Entity Recognition"""
    
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, 
                 num_layers=1, dropout=0.5, char_vocab_size=None, char_embedding_dim=25):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Character-level embeddings (optional)
        self.use_char_embeddings = char_vocab_size is not None
        if self.use_char_embeddings:
            self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
            self.char_lstm = nn.LSTM(char_embedding_dim, char_embedding_dim, 
                                   batch_first=True, bidirectional=True)
            self.char_output_dim = char_embedding_dim * 2
            lstm_input_dim = embedding_dim + self.char_output_dim
        else:
            lstm_input_dim = embedding_dim
        
        # BiLSTM
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        
        # CRF layer
        self.crf = CRF(tagset_size)
    
    def _get_char_features(self, char_ids):
        """Extract character-level features"""
        batch_size, seq_len, max_word_len = char_ids.shape
        
        # Reshape for processing
        char_ids = char_ids.view(-1, max_word_len)  # [batch_size * seq_len, max_word_len]
        
        # Character embeddings
        char_embeds = self.char_embeddings(char_ids)  # [batch_size * seq_len, max_word_len, char_embed_dim]
        
        # Character LSTM
        char_lstm_out, _ = self.char_lstm(char_embeds)
        
        # Take last output (or you could use max pooling)
        char_features = char_lstm_out[:, -1, :]  # [batch_size * seq_len, char_output_dim]
        
        # Reshape back
        char_features = char_features.view(batch_size, seq_len, -1)
        
        return char_features
    
    def _get_lstm_features(self, word_ids, char_ids=None, lengths=None):
        """Get BiLSTM features"""
        # Word embeddings
        word_embeds = self.word_embeddings(word_ids)
        
        # Character features
        if self.use_char_embeddings and char_ids is not None:
            char_features = self._get_char_features(char_ids)
            embeds = torch.cat([word_embeds, char_features], dim=2)
        else:
            embeds = word_embeds
        
        embeds = self.dropout(embeds)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embeds)
        
        # Unpack if necessary
        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = self.dropout(lstm_out)
        
        # Map to tag space
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats
    
    def forward(self, word_ids, tags, char_ids=None, lengths=None, mask=None):
        """Forward pass for training"""
        # Get LSTM features
        lstm_feats = self._get_lstm_features(word_ids, char_ids, lengths)
        
        # CRF loss
        loss = self.crf(lstm_feats, tags, mask)
        
        return loss
    
    def predict(self, word_ids, char_ids=None, lengths=None, mask=None):
        """Predict tag sequences"""
        # Get LSTM features
        lstm_feats = self._get_lstm_features(word_ids, char_ids, lengths)
        
        # CRF decoding
        predictions = self.crf.decode(lstm_feats, mask)
        
        return predictions

# BERT-based NER
class BERTForNER(nn.Module):
    """BERT model for Named Entity Recognition"""
    
    def __init__(self, bert_model, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

# NER Dataset
class NERDataset(Dataset):
    """Dataset for Named Entity Recognition"""
    
    def __init__(self, sentences, labels, word_vocab, label_vocab, char_vocab=None, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]
        
        # Convert words to indices
        word_ids = [self.word_vocab.get(word.lower(), self.word_vocab['<UNK>']) for word in sentence]
        label_ids = [self.label_vocab[label] for label in labels]
        
        # Truncate if necessary
        if len(word_ids) > self.max_length:
            word_ids = word_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]
        
        # Character features
        char_ids = None
        if self.char_vocab:
            char_ids = []
            for word in sentence[:self.max_length]:
                word_chars = [self.char_vocab.get(c, self.char_vocab['<UNK>']) for c in word[:20]]  # Max 20 chars per word
                # Pad to fixed length
                while len(word_chars) < 20:
                    word_chars.append(self.char_vocab['<PAD>'])
                char_ids.append(word_chars)
        
        return {
            'word_ids': word_ids,
            'label_ids': label_ids,
            'char_ids': char_ids,
            'length': len(word_ids)
        }

def collate_fn_ner(batch):
    """Collate function for NER data"""
    word_ids = [torch.tensor(item['word_ids']) for item in batch]
    label_ids = [torch.tensor(item['label_ids']) for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences
    word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=0)
    label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=0)
    
    # Create attention mask
    attention_mask = (word_ids != 0).long()
    
    result = {
        'word_ids': word_ids,
        'label_ids': label_ids,
        'lengths': lengths,
        'attention_mask': attention_mask
    }
    
    # Handle character IDs if present
    if batch[0]['char_ids'] is not None:
        char_ids = [torch.tensor(item['char_ids']) for item in batch]
        char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)
        result['char_ids'] = char_ids
    
    return result

# NER Trainer
class NERTrainer:
    """Trainer for NER models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            word_ids = batch['word_ids'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            char_ids = batch.get('char_ids')
            if char_ids is not None:
                char_ids = char_ids.to(self.device)
            
            optimizer.zero_grad()
            
            if isinstance(self.model, BiLSTMCRF):
                loss = self.model(word_ids, label_ids, char_ids, lengths, attention_mask)
            else:
                # For BERT-based models
                outputs = self.model(word_ids, attention_mask, labels=label_ids)
                loss = outputs[0]
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, label_vocab):
        self.model.eval()
        predictions = []
        true_labels = []
        
        idx_to_label = {v: k for k, v in label_vocab.items()}
        
        with torch.no_grad():
            for batch in dataloader:
                word_ids = batch['word_ids'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                char_ids = batch.get('char_ids')
                if char_ids is not None:
                    char_ids = char_ids.to(self.device)
                
                if isinstance(self.model, BiLSTMCRF):
                    pred_tags = self.model.predict(word_ids, char_ids, lengths, attention_mask)
                else:
                    # For BERT-based models
                    outputs = self.model(word_ids, attention_mask)
                    logits = outputs[0]
                    pred_tags = torch.argmax(logits, dim=2).cpu().numpy()
                
                # Convert to label names
                for i, length in enumerate(lengths):
                    if isinstance(self.model, BiLSTMCRF):
                        pred_labels = [idx_to_label[tag] for tag in pred_tags[i][:length]]
                    else:
                        pred_labels = [idx_to_label[tag] for tag in pred_tags[i][:length]]
                    
                    true_labels_seq = [idx_to_label[tag.item()] for tag in label_ids[i][:length]]
                    
                    predictions.append(pred_labels)
                    true_labels.append(true_labels_seq)
        
        return predictions, true_labels
    
    def train(self, train_loader, val_loader, label_vocab, num_epochs=10, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # Evaluate
            predictions, true_labels = self.evaluate(val_loader, label_vocab)
            f1_score = self.calculate_f1(predictions, true_labels)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val F1: {f1_score:.4f}')
            print('-' * 50)
            
            scheduler.step()
    
    def calculate_f1(self, predictions, true_labels):
        """Calculate F1 score for NER"""
        true_entities = set()
        pred_entities = set()
        
        for pred_seq, true_seq in zip(predictions, true_labels):
            # Extract entities using BIO tagging
            true_entities.update(self.extract_entities(true_seq))
            pred_entities.update(self.extract_entities(pred_seq))
        
        if len(pred_entities) == 0:
            return 0.0
        
        correct = len(true_entities & pred_entities)
        precision = correct / len(pred_entities) if len(pred_entities) > 0 else 0
        recall = correct / len(true_entities) if len(true_entities) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def extract_entities(self, labels):
        """Extract entities from BIO-tagged sequence"""
        entities = []
        current_entity = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = (i, i, label[2:])
            elif label.startswith('I-') and current_entity and label[2:] == current_entity[2]:
                current_entity = (current_entity[0], i, current_entity[2])
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return set(entities)

# Utility functions
def build_vocab_from_tokens(sentences):
    """Build vocabulary from tokenized sentences"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            if word.lower() not in vocab:
                vocab[word.lower()] = len(vocab)
    return vocab

def build_label_vocab(label_sequences):
    """Build label vocabulary"""
    vocab = {}
    for sequence in label_sequences:
        for label in sequence:
            if label not in vocab:
                vocab[label] = len(vocab)
    return vocab

def build_char_vocab(sentences):
    """Build character vocabulary"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                if char not in vocab:
                    vocab[char] = len(vocab)
    return vocab

if __name__ == "__main__":
    print("Testing Named Entity Recognition models...")
    
    # Sample data (BIO tagging format)
    sentences = [
        ["John", "Smith", "works", "at", "Google", "in", "California"],
        ["Mary", "Johnson", "visited", "New", "York", "last", "week"],
        ["The", "company", "Apple", "was", "founded", "in", "Cupertino"]
    ]
    
    labels = [
        ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC"],
        ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"],
        ["O", "O", "B-ORG", "O", "O", "O", "B-LOC"]
    ]
    
    # Build vocabularies
    word_vocab = build_vocab_from_tokens(sentences)
    label_vocab = build_label_vocab(labels)
    char_vocab = build_char_vocab(sentences)
    
    print(f"Word vocab size: {len(word_vocab)}")
    print(f"Label vocab size: {len(label_vocab)}")
    print(f"Character vocab size: {len(char_vocab)}")
    
    # Create dataset
    dataset = NERDataset(sentences, labels, word_vocab, label_vocab, char_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_ner)
    
    # Test BiLSTM-CRF model
    model = BiLSTMCRF(
        vocab_size=len(word_vocab),
        tagset_size=len(label_vocab),
        embedding_dim=50,
        hidden_dim=64,
        char_vocab_size=len(char_vocab),
        char_embedding_dim=25
    )
    
    print(f"BiLSTM-CRF parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    for batch in dataloader:
        word_ids = batch['word_ids']
        label_ids = batch['label_ids']
        char_ids = batch['char_ids']
        lengths = batch['lengths']
        attention_mask = batch['attention_mask']
        
        # Training forward pass
        loss = model(word_ids, label_ids, char_ids, lengths, attention_mask)
        print(f"Training loss: {loss.item():.4f}")
        
        # Prediction
        predictions = model.predict(word_ids, char_ids, lengths, attention_mask)
        print(f"Predictions: {predictions}")
        
        break
    
    # Test trainer
    trainer = NERTrainer(model, device='cpu')
    
    # Test evaluation
    predictions, true_labels = trainer.evaluate(dataloader, label_vocab)
    f1_score = trainer.calculate_f1(predictions, true_labels)
    print(f"F1 Score: {f1_score:.4f}")
    
    # Test entity extraction
    test_labels = ["B-PER", "I-PER", "O", "B-ORG", "O"]
    entities = trainer.extract_entities(test_labels)
    print(f"Extracted entities: {entities}")
    
    print("Named Entity Recognition testing completed!")