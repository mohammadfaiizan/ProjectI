import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import re
import string
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

# Question Answering Dataset
class QADataset(Dataset):
    """Dataset for extractive question answering"""
    
    def __init__(self, contexts, questions, answers, tokenizer, max_length=512):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Tokenize
        context_tokens = self.tokenizer.tokenize(context.lower())
        question_tokens = self.tokenizer.tokenize(question.lower())
        
        # Find answer span in context
        answer_start, answer_end = self.find_answer_span(context_tokens, answer)
        
        # Combine question and context
        # Format: [CLS] question [SEP] context [SEP]
        tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
        
        # Adjust answer positions (account for question tokens and special tokens)
        if answer_start != -1 and answer_end != -1:
            answer_start += len(question_tokens) + 2  # +2 for [CLS] and [SEP]
            answer_end += len(question_tokens) + 2
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            # Check if answer span is still valid
            if answer_end >= self.max_length:
                answer_start = -1
                answer_end = -1
        
        # Convert to indices
        input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.vocab['[UNK]']) for token in tokens]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Create token type ids (0 for question, 1 for context)
        question_len = len(question_tokens) + 2  # +2 for [CLS] and [SEP]
        token_type_ids = [0] * question_len + [1] * (len(input_ids) - question_len)
        
        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'start_positions': torch.tensor(answer_start if answer_start != -1 else 0),
            'end_positions': torch.tensor(answer_end if answer_end != -1 else 0),
            'is_impossible': torch.tensor(1 if answer_start == -1 else 0)
        }
    
    def find_answer_span(self, context_tokens, answer_text):
        """Find the token span of the answer in the context"""
        if not answer_text or answer_text.lower() == 'unknown':
            return -1, -1
        
        answer_tokens = self.tokenizer.tokenize(answer_text.lower())
        
        # Find the best matching span
        best_start = -1
        best_end = -1
        best_score = 0
        
        for start in range(len(context_tokens) - len(answer_tokens) + 1):
            end = start + len(answer_tokens) - 1
            
            # Calculate overlap score
            score = 0
            for i, answer_token in enumerate(answer_tokens):
                if start + i < len(context_tokens) and context_tokens[start + i] == answer_token:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
        
        # Only accept if we have a reasonable match
        if best_score >= len(answer_tokens) * 0.5:
            return best_start, best_end
        else:
            return -1, -1

class SimpleTokenizer:
    """Simple tokenizer for QA"""
    
    def __init__(self):
        # Basic vocabulary with special tokens
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        }
        self.vocab_built = False
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Add frequent words to vocabulary
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.vocab)} tokens")
    
    def tokenize(self, text):
        """Simple tokenization"""
        # Basic tokenization: lowercase, split on whitespace and punctuation
        text = text.lower()
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        tokens = text.split()
        return tokens

# BiLSTM QA Model
class BiLSTMQA(nn.Module):
    """BiLSTM-based question answering model"""
    
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Span prediction layers
        self.start_linear = nn.Linear(hidden_dim * 2, 1)
        self.end_linear = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # BiLSTM
        lstm_output, _ = self.lstm(embedded)
        lstm_output = self.dropout(lstm_output)
        
        # Self-attention
        attended_output = self.attention(lstm_output, attention_mask)
        
        # Span prediction
        start_logits = self.start_linear(attended_output).squeeze(-1)
        end_logits = self.end_linear(attended_output).squeeze(-1)
        
        # Apply attention mask
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
            end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
        
        return start_logits, end_logits

class SelfAttention(nn.Module):
    """Self-attention mechanism for QA"""
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_dim = x.size()
        
        # Linear transformations
        Q = self.query_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask.bool(), -1e9)
        
        # Attention weights and output
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        output = self.output_linear(attention_output)
        return output

# BERT-based QA Model
class BERTForQuestionAnswering(nn.Module):
    """BERT-based question answering model"""
    
    def __init__(self, bert_model, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        
        # Span prediction layers
        self.qa_outputs = nn.Linear(bert_model.config.hidden_size, 2)  # start and end logits
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                start_positions=None, end_positions=None):
        
        # BERT forward pass
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # Span prediction
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits

# Reading Comprehension Model with Attention
class ReadingComprehensionModel(nn.Module):
    """Advanced reading comprehension model with co-attention"""
    
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Separate encoders for question and context
        self.question_encoder = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                       batch_first=True, bidirectional=True, dropout=dropout)
        self.context_encoder = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                      batch_first=True, bidirectional=True, dropout=dropout)
        
        # Co-attention mechanism
        self.co_attention = CoAttention(hidden_dim * 2)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.start_linear = nn.Linear(hidden_dim * 4, 1)  # *4 for concatenated representations
        self.end_linear = nn.Linear(hidden_dim * 4, 1)
    
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        # Separate question and context based on token_type_ids
        embedded = self.embedding(input_ids)
        
        # Find question and context boundaries
        question_mask = (token_type_ids == 0) & (attention_mask.bool() if attention_mask is not None else True)
        context_mask = (token_type_ids == 1) & (attention_mask.bool() if attention_mask is not None else True)
        
        # Encode question and context separately
        question_embedded = embedded * question_mask.unsqueeze(-1).float()
        context_embedded = embedded * context_mask.unsqueeze(-1).float()
        
        question_output, _ = self.question_encoder(question_embedded)
        context_output, _ = self.context_encoder(context_embedded)
        
        # Co-attention
        attended_context = self.co_attention(question_output, context_output, 
                                           question_mask, context_mask)
        
        # Span prediction (only on context part)
        attended_context = self.dropout(attended_context)
        start_logits = self.start_linear(attended_context).squeeze(-1)
        end_logits = self.end_linear(attended_context).squeeze(-1)
        
        # Mask out non-context positions
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~context_mask, -1e9)
            end_logits = end_logits.masked_fill(~context_mask, -1e9)
        
        return start_logits, end_logits

class CoAttention(nn.Module):
    """Co-attention mechanism between question and context"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_co = nn.Linear(hidden_dim, hidden_dim)
        
        self.linear_transform = nn.Linear(hidden_dim * 2, hidden_dim * 2)
    
    def forward(self, question_output, context_output, question_mask, context_mask):
        # question_output: [batch_size, seq_len, hidden_dim]
        # context_output: [batch_size, seq_len, hidden_dim]
        
        batch_size, seq_len, hidden_dim = context_output.size()
        
        # Project representations
        Q = self.W_q(question_output)  # [batch_size, seq_len, hidden_dim]
        C = self.W_c(context_output)   # [batch_size, seq_len, hidden_dim]
        
        # Compute attention between question and context
        attention_scores = torch.bmm(C, Q.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        # Apply masks
        if question_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~question_mask.unsqueeze(1), -1e9
            )
        
        # Attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Attended question representation for each context position
        attended_question = torch.bmm(attention_weights, question_output)  # [batch_size, seq_len, hidden_dim]
        
        # Combine context and attended question
        combined = torch.cat([context_output, attended_question], dim=-1)
        output = self.linear_transform(combined)
        
        return output

# QA Trainer
class QATrainer:
    """Trainer for question answering models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)
            
            optimizer.zero_grad()
            
            if hasattr(self.model, 'bert'):
                # BERT-based model
                token_type_ids = batch['token_type_ids'].to(self.device)
                loss, start_logits, end_logits = self.model(
                    input_ids, attention_mask, token_type_ids, start_positions, end_positions
                )
            else:
                # BiLSTM or other models
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    start_logits, end_logits = self.model(input_ids, token_type_ids, attention_mask)
                else:
                    start_logits, end_logits = self.model(input_ids, attention_mask)
                
                # Compute loss manually
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                if hasattr(self.model, 'bert'):
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    loss, start_logits, end_logits = self.model(
                        input_ids, attention_mask, token_type_ids, start_positions, end_positions
                    )
                else:
                    if 'token_type_ids' in batch:
                        token_type_ids = batch['token_type_ids'].to(self.device)
                        start_logits, end_logits = self.model(input_ids, token_type_ids, attention_mask)
                    else:
                        start_logits, end_logits = self.model(input_ids, attention_mask)
                    
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2
                
                total_loss += loss.item()
                
                # Calculate accuracy
                start_preds = start_logits.argmax(dim=1)
                end_preds = end_logits.argmax(dim=1)
                
                correct_start = (start_preds == start_positions).sum().item()
                correct_end = (end_preds == end_positions).sum().item()
                correct_both = ((start_preds == start_positions) & (end_preds == end_positions)).sum().item()
                
                correct_predictions += correct_both
                total_predictions += input_ids.size(0)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return total_loss / len(dataloader), accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            print('-' * 60)
            
            scheduler.step()

# Evaluation Metrics
def compute_exact_match(predictions, references):
    """Compute exact match score"""
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if normalize_answer(pred) == normalize_answer(ref):
            exact_matches += 1
    
    return exact_matches / len(predictions) if predictions else 0

def compute_f1_score(predictions, references):
    """Compute F1 score for QA"""
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()
        
        if not pred_tokens and not ref_tokens:
            f1_scores.append(1.0)
            continue
        elif not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        
        common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            f1_scores.append(0.0)
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

def normalize_answer(s):
    """Normalize answer for evaluation"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

# Answer Extraction
def extract_answer(start_logits, end_logits, input_tokens, max_answer_length=30):
    """Extract answer span from logits"""
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    
    # Find the best start and end positions
    best_score = 0
    best_start = 0
    best_end = 0
    
    for start in range(len(start_probs)):
        for end in range(start, min(start + max_answer_length, len(end_probs))):
            score = start_probs[start] * end_probs[end]
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
    
    # Extract answer tokens
    answer_tokens = input_tokens[best_start:best_end + 1]
    answer = ' '.join([token for token in answer_tokens if token not in ['[PAD]', '[CLS]', '[SEP]']])
    
    return answer, best_start, best_end

if __name__ == "__main__":
    print("Testing question answering models...")
    
    # Sample data
    contexts = [
        "The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree.",
        "Paris is the capital of France. It is known for the Eiffel Tower and good food.",
        "Machine learning is a subset of artificial intelligence. It involves training algorithms on data."
    ]
    
    questions = [
        "What jumps over the dog?",
        "What is Paris known for?", 
        "What does machine learning involve?"
    ]
    
    answers = [
        "The quick brown fox",
        "the Eiffel Tower and good food",
        "training algorithms on data"
    ]
    
    # Create simple tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    all_texts = contexts + questions + answers
    tokenizer.build_vocab(all_texts, min_freq=1)
    
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Create dataset
    dataset = QADataset(contexts, questions, answers, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Test BiLSTM QA model
    print("\nTesting BiLSTM QA model...")
    bilstm_model = BiLSTMQA(
        vocab_size=len(tokenizer.vocab),
        embed_dim=100,
        hidden_dim=64,
        num_layers=2
    )
    
    print(f"BiLSTM QA parameters: {sum(p.numel() for p in bilstm_model.parameters()):,}")
    
    # Test forward pass
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        start_logits, end_logits = bilstm_model(input_ids, attention_mask)
        print(f"Start logits shape: {start_logits.shape}")
        print(f"End logits shape: {end_logits.shape}")
        break
    
    # Test Reading Comprehension model
    print("\nTesting Reading Comprehension model...")
    rc_model = ReadingComprehensionModel(
        vocab_size=len(tokenizer.vocab),
        embed_dim=100,
        hidden_dim=64
    )
    
    print(f"RC model parameters: {sum(p.numel() for p in rc_model.parameters()):,}")
    
    # Test with token type ids
    for batch in dataloader:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        
        start_logits, end_logits = rc_model(input_ids, token_type_ids, attention_mask)
        print(f"RC Start logits shape: {start_logits.shape}")
        print(f"RC End logits shape: {end_logits.shape}")
        break
    
    # Test answer extraction
    print("\nTesting answer extraction...")
    with torch.no_grad():
        sample_tokens = ['[CLS]', 'what', 'jumps', '[SEP]', 'the', 'quick', 'brown', 'fox', 'jumps', '[SEP]']
        start_logits = torch.randn(10)
        end_logits = torch.randn(10)
        
        answer, start_pos, end_pos = extract_answer(start_logits, end_logits, sample_tokens)
        print(f"Extracted answer: '{answer}' (positions {start_pos}-{end_pos})")
    
    # Test evaluation metrics
    print("\nTesting evaluation metrics...")
    pred_answers = ["the quick brown fox", "eiffel tower and food", "training algorithms"]
    ref_answers = ["The quick brown fox", "the Eiffel Tower and good food", "training algorithms on data"]
    
    exact_match = compute_exact_match(pred_answers, ref_answers)
    f1_score = compute_f1_score(pred_answers, ref_answers)
    
    print(f"Exact Match: {exact_match:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Test trainer
    print("\nTesting QA trainer...")
    trainer = QATrainer(bilstm_model, device='cpu')
    
    # Quick evaluation test
    val_loss, val_accuracy = trainer.evaluate(dataloader)
    print(f"Initial validation loss: {val_loss:.4f}")
    print(f"Initial validation accuracy: {val_accuracy:.4f}")
    
    print("Question answering testing completed!")