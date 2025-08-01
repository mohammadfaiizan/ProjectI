import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Union
import math

# Base Text Generator
class TextGenerator:
    """Base class for text generation methods"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def encode(self, text):
        """Encode text to token indices"""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        else:
            # Simple character-level encoding
            return [self.tokenizer.char_to_idx.get(c, self.tokenizer.char_to_idx['<UNK>']) for c in text]
    
    def decode(self, tokens):
        """Decode token indices to text"""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens)
        else:
            # Simple character-level decoding
            return ''.join([self.tokenizer.idx_to_char.get(t, '<UNK>') for t in tokens])

# Greedy Search
class GreedyGenerator(TextGenerator):
    """Greedy decoding - always choose the most likely next token"""
    
    def generate(self, prompt, max_length=100, temperature=1.0):
        """Generate text using greedy search"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                if hasattr(self.model, 'generate'):
                    # For GPT-style models
                    input_ids = self.model.generate(input_ids, max_new_tokens=1, temperature=temperature, top_k=1)
                    break
                else:
                    # Generic approach
                    outputs = self.model(input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Get logits for last token and apply temperature
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Greedy selection (top-1)
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Append to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Multinomial Sampling
class MultinomialGenerator(TextGenerator):
    """Multinomial sampling from the probability distribution"""
    
    def generate(self, prompt, max_length=100, temperature=1.0):
        """Generate text using multinomial sampling"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from multinomial distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Top-K Sampling
class TopKGenerator(TextGenerator):
    """Top-K sampling - sample from top K most likely tokens"""
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50):
        """Generate text using top-k sampling"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Filter to top-k tokens
                if top_k > 0:
                    values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    indices_to_remove = next_token_logits < values[:, [-1]]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Top-P (Nucleus) Sampling
class TopPGenerator(TextGenerator):
    """Top-P (nucleus) sampling - sample from smallest set of tokens with cumulative probability >= p"""
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_p=0.9):
        """Generate text using top-p (nucleus) sampling"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create mask for original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Beam Search
class BeamSearchGenerator(TextGenerator):
    """Beam search for finding high-probability sequences"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        super().__init__(model, tokenizer, device)
    
    def generate(self, prompt, max_length=100, beam_size=5, length_penalty=1.0, early_stopping=True):
        """Generate text using beam search"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        batch_size, seq_len = input_ids.shape
        
        # Initialize beams: (sequence, log_probability, finished)
        beams = [(input_ids, 0.0, False)]
        finished_beams = []
        
        with torch.no_grad():
            for step in range(max_length):
                new_beams = []
                
                for sequence, log_prob, finished in beams:
                    if finished:
                        finished_beams.append((sequence, log_prob, True))
                        continue
                    
                    # Get model predictions
                    outputs = self.model(sequence)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    next_token_logits = logits[:, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top beam_size candidates
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                        new_sequence = torch.cat([sequence, new_token], dim=1)
                        new_log_prob = log_prob + top_log_probs[0, i].item()
                        
                        # Apply length penalty
                        length_normalized_score = new_log_prob / (new_sequence.size(1) ** length_penalty)
                        
                        # Check if sequence is finished (you might want to add EOS token check)
                        is_finished = False  # Add your EOS token check here
                        
                        new_beams.append((new_sequence, new_log_prob, is_finished))
                
                # Keep top beam_size beams
                all_beams = new_beams + finished_beams
                if length_penalty != 1.0:
                    # Sort by length-normalized score
                    all_beams.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
                else:
                    all_beams.sort(key=lambda x: x[1], reverse=True)
                
                beams = [beam for beam in all_beams[:beam_size] if not beam[2]]
                finished_beams = [beam for beam in all_beams if beam[2]]
                
                # Early stopping condition
                if early_stopping and len(finished_beams) >= beam_size:
                    break
                
                if not beams:  # All beams finished
                    break
        
        # Return best sequence
        all_final_beams = beams + finished_beams
        if length_penalty != 1.0:
            best_beam = max(all_final_beams, key=lambda x: x[1] / (x[0].size(1) ** length_penalty))
        else:
            best_beam = max(all_final_beams, key=lambda x: x[1])
        
        generated_text = self.decode(best_beam[0][0].cpu().tolist())
        return generated_text

# Diverse Beam Search
class DiverseBeamSearchGenerator(TextGenerator):
    """Diverse beam search to generate multiple diverse outputs"""
    
    def generate(self, prompt, max_length=100, beam_size=5, num_groups=2, diversity_penalty=0.5):
        """Generate diverse text using diverse beam search"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        group_size = beam_size // num_groups
        
        # Initialize groups
        groups = []
        for g in range(num_groups):
            groups.append([(input_ids.clone(), 0.0)])
        
        with torch.no_grad():
            for step in range(max_length):
                all_candidates = []
                
                for group_idx, group_beams in enumerate(groups):
                    group_candidates = []
                    
                    for sequence, log_prob in group_beams:
                        outputs = self.model(sequence)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        next_token_logits = logits[:, -1, :]
                        log_probs = F.log_softmax(next_token_logits, dim=-1)
                        
                        # Apply diversity penalty
                        if step > 0 and group_idx > 0:
                            # Penalize tokens chosen by previous groups
                            for prev_group_idx in range(group_idx):
                                for prev_seq, _ in groups[prev_group_idx]:
                                    if prev_seq.size(1) > sequence.size(1):
                                        penalty_token = prev_seq[0, sequence.size(1)]
                                        log_probs[0, penalty_token] -= diversity_penalty
                        
                        top_log_probs, top_indices = torch.topk(log_probs, group_size)
                        
                        for i in range(group_size):
                            new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                            new_sequence = torch.cat([sequence, new_token], dim=1)
                            new_log_prob = log_prob + top_log_probs[0, i].item()
                            group_candidates.append((new_sequence, new_log_prob))
                    
                    # Keep top candidates for this group
                    group_candidates.sort(key=lambda x: x[1], reverse=True)
                    groups[group_idx] = group_candidates[:group_size]
                    
                    all_candidates.extend(group_candidates[:group_size])
        
        # Return all diverse outputs
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for sequence, score in all_candidates[:beam_size]:
            text = self.decode(sequence[0].cpu().tolist())
            results.append((text, score))
        
        return results

# Contrastive Search
class ContrastiveSearchGenerator(TextGenerator):
    """Contrastive search balancing coherence and diversity"""
    
    def generate(self, prompt, max_length=100, alpha=0.6, k=4):
        """Generate text using contrastive search
        
        Args:
            alpha: weight for model confidence vs diversity penalty
            k: number of top tokens to consider
        """
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for step in range(max_length):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Get top-k candidates
                top_probs, top_indices = torch.topk(probs, k)
                
                # Calculate diversity penalty for each candidate
                scores = []
                for i in range(k):
                    candidate_token = top_indices[0, i]
                    model_confidence = top_probs[0, i].item()
                    
                    # Calculate similarity with previous tokens
                    if input_ids.size(1) > 1:
                        # Get embeddings (simplified - you might need model-specific approach)
                        similarity_penalty = 0.0  # Implement token similarity calculation
                        diversity_penalty = similarity_penalty
                    else:
                        diversity_penalty = 0.0
                    
                    # Combine model confidence and diversity
                    final_score = alpha * model_confidence - (1 - alpha) * diversity_penalty
                    scores.append((final_score, candidate_token))
                
                # Select token with highest combined score
                best_token = max(scores, key=lambda x: x[0])[1]
                input_ids = torch.cat([input_ids, best_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Typical Sampling
class TypicalSamplingGenerator(TextGenerator):
    """Typical sampling based on conditional entropy"""
    
    def generate(self, prompt, max_length=100, temperature=1.0, typical_p=0.9):
        """Generate text using typical sampling"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Calculate entropy and typical probability
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                # Calculate absolute difference from conditional entropy
                shifted_scores = torch.abs(log_probs + entropy.unsqueeze(-1))
                
                # Sort by typical probability
                sorted_scores, sorted_indices = torch.sort(shifted_scores)
                sorted_probs = probs.gather(-1, sorted_indices)
                
                # Get cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                last_ind = (cumulative_probs < typical_p).sum(dim=-1)
                last_ind = torch.clamp(last_ind, min=1, max=sorted_probs.size(-1) - 1)
                
                # Create mask
                sorted_indices_to_remove = torch.zeros_like(sorted_scores, dtype=torch.bool)
                for i, ind in enumerate(last_ind):
                    sorted_indices_to_remove[i, ind:] = True
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                filtered_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(filtered_probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Repetition Penalty
class RepetitionPenaltyGenerator(TextGenerator):
    """Text generation with repetition penalty"""
    
    def generate(self, prompt, max_length=100, temperature=1.0, repetition_penalty=1.2):
        """Generate text with repetition penalty"""
        self.model.eval()
        
        input_ids = torch.tensor(self.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(input_ids[0].tolist()):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                
                # Sample from penalized distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_text = self.decode(input_ids[0].cpu().tolist())
        return generated_text

# Unified Generator
class UnifiedTextGenerator:
    """Unified interface for all generation methods"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize all generators
        self.generators = {
            'greedy': GreedyGenerator(model, tokenizer, device),
            'multinomial': MultinomialGenerator(model, tokenizer, device),
            'top_k': TopKGenerator(model, tokenizer, device),
            'top_p': TopPGenerator(model, tokenizer, device),
            'beam_search': BeamSearchGenerator(model, tokenizer, device),
            'diverse_beam': DiverseBeamSearchGenerator(model, tokenizer, device),
            'contrastive': ContrastiveSearchGenerator(model, tokenizer, device),
            'typical': TypicalSamplingGenerator(model, tokenizer, device),
            'repetition_penalty': RepetitionPenaltyGenerator(model, tokenizer, device)
        }
    
    def generate(self, prompt, method='top_p', **kwargs):
        """Generate text using specified method"""
        if method not in self.generators:
            raise ValueError(f"Unknown generation method: {method}")
        
        return self.generators[method].generate(prompt, **kwargs)
    
    def compare_methods(self, prompt, methods=['greedy', 'top_k', 'top_p'], **kwargs):
        """Compare different generation methods"""
        results = {}
        for method in methods:
            try:
                result = self.generate(prompt, method=method, **kwargs)
                results[method] = result
            except Exception as e:
                results[method] = f"Error: {str(e)}"
        
        return results

# Generation utilities
def calculate_perplexity(model, text, tokenizer, device='cuda'):
    """Calculate perplexity of generated text"""
    model.eval()
    
    if hasattr(tokenizer, 'encode'):
        tokens = tokenizer.encode(text)
    else:
        tokens = [tokenizer.char_to_idx.get(c, 0) for c in text]
    
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        perplexity = torch.exp(loss).item()
    
    return perplexity

def evaluate_diversity(texts):
    """Evaluate diversity of generated texts using various metrics"""
    if not texts:
        return {}
    
    # Calculate distinct n-grams
    def distinct_ngrams(texts, n):
        all_ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            words = text.split()
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        return len(all_ngrams) / max(total_ngrams, 1)
    
    metrics = {
        'distinct_1': distinct_ngrams(texts, 1),
        'distinct_2': distinct_ngrams(texts, 2),
        'distinct_3': distinct_ngrams(texts, 3),
        'unique_texts': len(set(texts)) / len(texts)
    }
    
    return metrics

if __name__ == "__main__":
    print("Testing text generation methods...")
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(vocab_size, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.output = nn.Linear(128, vocab_size)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            logits = self.output(lstm_out)
            return logits
    
    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.vocab = ['<pad>', 'hello', 'world', 'test', 'generation', '.', ' ']
            self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
            self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        
        def encode(self, text):
            return [self.char_to_idx.get(c, 0) for c in text[:10]]  # Simplified
        
        def decode(self, tokens):
            return ''.join([self.idx_to_char.get(t, '<unk>') for t in tokens])
    
    # Create mock model and tokenizer
    vocab_size = 7
    model = MockModel(vocab_size)
    tokenizer = MockTokenizer()
    
    # Test unified generator
    generator = UnifiedTextGenerator(model, tokenizer, device='cpu')
    
    prompt = "hello"
    
    # Test different methods
    methods_to_test = ['greedy', 'multinomial', 'top_k', 'top_p']
    
    print("Comparing generation methods:")
    results = generator.compare_methods(prompt, methods=methods_to_test, max_length=20)
    
    for method, result in results.items():
        print(f"{method}: {result}")
    
    # Test beam search
    beam_generator = BeamSearchGenerator(model, tokenizer, device='cpu')
    beam_result = beam_generator.generate(prompt, max_length=15, beam_size=3)
    print(f"Beam search: {beam_result}")
    
    # Test diversity metrics
    test_texts = ["hello world", "hello test", "world generation"]
    diversity_metrics = evaluate_diversity(test_texts)
    print(f"Diversity metrics: {diversity_metrics}")
    
    print("Text generation methods testing completed!")