#!/usr/bin/env python3
"""
Context Window Manager: Optimizing Limited Attention for Maximum Relevance
=========================================================================

WHAT IS THE PROBLEM?
==================
AI models have limited context windows that constrain their memory and reasoning:
- Language models can only process a fixed number of tokens at once
- Important information gets lost when context exceeds the window limit
- No intelligent management of what information to keep vs. discard
- Context switching causes loss of important conversational state
- Inefficient use of available context space leads to poor performance
- No prioritization of information based on relevance or importance

Example: Conversation Context Overflow
WITHOUT CONTEXT MANAGEMENT (Traditional):
- User starts complex discussion about project planning
- Conversation grows beyond model's 4k token limit
- Early important requirements and decisions get pushed out
- Model forgets initial context and gives inconsistent advice
- User has to constantly re-explain previously discussed points
- Result: Broken continuity, repetitive conversations, poor user experience

REAL WORLD EXAMPLE:
=================
How does human working memory manage limited attention?

HUMAN ATTENTION MANAGEMENT:
1. SELECTIVE ATTENTION: Focus on most relevant information while filtering noise
2. CHUNKING: Group related information into meaningful units
3. COMPRESSION: Summarize detailed information into key points
4. PRIORITIZATION: Keep critical information active while backgrounding less important details
5. CONTEXTUAL RELEVANCE: Maintain information relevant to current goals
6. GRACEFUL DEGRADATION: Gradually forget least important details when capacity is reached
7. RETRIEVAL CUES: Use triggers to recall backgrounded information when needed

BENEFITS OF CONTEXT WINDOW MANAGEMENT:
- Maximizes effective use of limited model attention and memory
- Maintains conversational continuity even in long interactions
- Prioritizes most relevant information for current context
- Enables intelligent compression and summarization of historical context
- Supports seamless context switching without information loss
- Improves model performance through optimal context utilization

THE CONTEXT ADVANTAGE:
====================
UNMANAGED: Random truncation → Loss of important information
MANAGED: Intelligent curation → Maximum relevance and continuity

CONTEXT WINDOW COMPONENTS:
=========================
1. TOKEN COUNTING: Accurate tracking of context space utilization
2. RELEVANCE SCORING: Importance assessment for different content types
3. COMPRESSION STRATEGIES: Intelligent summarization of less critical content
4. SLIDING WINDOWS: Moving windows that maintain recent and relevant context
5. HIERARCHICAL CONTEXT: Multi-level context organization (immediate, recent, background)
6. CONTEXT SWITCHING: Seamless transitions between different conversation topics
7. RETRIEVAL INTEGRATION: Connection with external memory for context expansion

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI to maintain coherent long-form conversations and reasoning
- Critical for production AI systems with real-world conversation requirements
- Maximizes the effectiveness of limited model context windows
- Supports complex multi-turn interactions and task continuity
- Foundation for conversational AI that feels natural and consistent
- Enables AI to work within hardware and model constraints intelligently
"""

import asyncio
import time
import json
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
import tiktoken
import numpy as np
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ContentType(Enum):
    """Types of content in context window"""
    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESULT = "function_result"
    CONTEXT_SUMMARY = "context_summary"
    METADATA = "metadata"

class Priority(Enum):
    """Priority levels for context content"""
    CRITICAL = 5    # Must keep (system messages, current task)
    HIGH = 4        # Very important (recent user requests, key decisions)
    MEDIUM = 3      # Important (relevant history, context)
    LOW = 2         # Useful (background info, examples)
    MINIMAL = 1     # Can be compressed or removed

class CompressionStrategy(Enum):
    """Strategies for content compression"""
    NONE = "none"                    # No compression
    TRUNCATE = "truncate"            # Simple truncation
    SUMMARIZE = "summarize"          # Intelligent summarization
    EXTRACT_KEY_POINTS = "extract"   # Extract key information
    HIERARCHICAL = "hierarchical"    # Multi-level compression

@dataclass
class ContextItem:
    """Represents an item in the context window"""
    
    id: str
    content: str
    content_type: ContentType
    priority: Priority
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    
    # Relevance scoring
    relevance_score: float = 1.0
    decay_rate: float = 0.1
    
    # Compression
    is_compressed: bool = False
    original_content: Optional[str] = None
    compression_ratio: float = 1.0
    
    # Relationships
    related_items: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    
    # Usage tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def calculate_current_relevance(self) -> float:
        """Calculate current relevance considering time decay"""
        
        if not self.last_accessed:
            self.last_accessed = self.timestamp
        
        time_delta = datetime.now() - self.last_accessed
        decay_factor = max(0.1, 1.0 - (self.decay_rate * time_delta.total_seconds() / 3600))
        
        # Combine base relevance with decay and access frequency
        access_bonus = min(0.5, self.access_count * 0.1)
        current_relevance = (self.relevance_score * decay_factor) + access_bonus
        
        return min(1.0, current_relevance)
    
    def update_access(self) -> None:
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def compress(self, new_content: str, compression_ratio: float) -> None:
        """Apply compression to the item"""
        
        if not self.is_compressed:
            self.original_content = self.content
        
        self.content = new_content
        self.is_compressed = True
        self.compression_ratio = compression_ratio
        
        # Recalculate token count
        self.token_count = len(self.content.split())  # Simplified token counting

class TokenCounter:
    """Handles token counting for different models"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base if model not found
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        self.logger = logging.getLogger("TokenCounter")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed, using word approximation: {e}")
            # Fallback: approximate 1 token per 0.75 words
            return int(len(text.split()) / 0.75)
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages"""
        
        total_tokens = 0
        
        for message in messages:
            # Add tokens for message structure
            total_tokens += 3  # Role, content wrapper tokens
            
            for key, value in message.items():
                total_tokens += self.count_tokens(value)
        
        total_tokens += 3  # Priming tokens
        
        return total_tokens
    
    def estimate_response_tokens(self, prompt: str) -> int:
        """Estimate response tokens based on prompt"""
        
        # Simple heuristic: response is typically 10-50% of prompt length
        prompt_tokens = self.count_tokens(prompt)
        return int(prompt_tokens * 0.3)

class ContentCompressor:
    """Handles intelligent content compression"""
    
    def __init__(self):
        self.compression_strategies = {
            CompressionStrategy.SUMMARIZE: self._summarize_content,
            CompressionStrategy.EXTRACT_KEY_POINTS: self._extract_key_points,
            CompressionStrategy.TRUNCATE: self._truncate_content,
            CompressionStrategy.HIERARCHICAL: self._hierarchical_compression
        }
        
        self.logger = logging.getLogger("ContentCompressor")
    
    async def compress_content(self, content: str, target_ratio: float,
                             strategy: CompressionStrategy) -> Tuple[str, float]:
        """Compress content using specified strategy"""
        
        if strategy == CompressionStrategy.NONE:
            return content, 1.0
        
        if strategy not in self.compression_strategies:
            self.logger.warning(f"Unknown compression strategy: {strategy}")
            return self._truncate_content(content, target_ratio)
        
        try:
            compression_func = self.compression_strategies[strategy]
            return compression_func(content, target_ratio)
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return self._truncate_content(content, target_ratio)
    
    def _summarize_content(self, content: str, target_ratio: float) -> Tuple[str, float]:
        """Intelligent summarization of content"""
        
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return content, 1.0
        
        # Calculate how many sentences to keep
        target_sentences = max(1, int(len(sentences) * target_ratio))
        
        # Score sentences by length and position (simple heuristic)
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Favor longer sentences and earlier sentences
            length_score = len(sentence.split()) / 20.0  # Normalize by typical sentence length
            position_score = 1.0 - (i / len(sentences))  # Earlier sentences score higher
            
            total_score = (length_score + position_score) / 2
            scored_sentences.append((sentence, total_score, i))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = scored_sentences[:target_sentences]
        
        # Sort by original position to maintain order
        selected_sentences.sort(key=lambda x: x[2])
        
        summarized = '. '.join([s[0] for s in selected_sentences])
        actual_ratio = len(summarized) / len(content)
        
        return summarized, actual_ratio
    
    def _extract_key_points(self, content: str, target_ratio: float) -> Tuple[str, float]:
        """Extract key points from content"""
        
        # Look for structured content (lists, important phrases)
        lines = content.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Identify key points by patterns
            if (line.startswith('-') or line.startswith('*') or 
                line.startswith('•') or line.startswith(tuple('123456789'))):
                key_points.append(line)
            elif any(keyword in line.lower() for keyword in 
                    ['important', 'key', 'critical', 'note', 'remember', 'must']):
                key_points.append(f"• {line}")
            elif len(line.split()) < 15:  # Short, potentially important lines
                key_points.append(f"• {line}")
        
        if not key_points:
            # Fallback to first few sentences
            sentences = re.split(r'[.!?]+', content)
            target_sentences = max(1, int(len(sentences) * target_ratio))
            key_points = [f"• {s.strip()}" for s in sentences[:target_sentences] if s.strip()]
        
        extracted = '\n'.join(key_points[:int(len(key_points) * target_ratio * 1.5)])
        actual_ratio = len(extracted) / len(content) if content else 1.0
        
        return extracted, actual_ratio
    
    def _truncate_content(self, content: str, target_ratio: float) -> Tuple[str, float]:
        """Simple content truncation"""
        
        target_length = int(len(content) * target_ratio)
        
        if target_length >= len(content):
            return content, 1.0
        
        # Try to truncate at word boundary
        truncated = content[:target_length]
        last_space = truncated.rfind(' ')
        
        if last_space > target_length * 0.8:  # If we can find a reasonable word boundary
            truncated = truncated[:last_space] + "..."
        else:
            truncated = truncated + "..."
        
        actual_ratio = len(truncated) / len(content)
        
        return truncated, actual_ratio
    
    def _hierarchical_compression(self, content: str, target_ratio: float) -> Tuple[str, float]:
        """Multi-level hierarchical compression"""
        
        # First pass: extract structure
        structured_content = self._extract_structure(content)
        
        # Second pass: compress based on hierarchy
        compressed_parts = []
        
        for section_type, section_content in structured_content:
            if section_type == "header":
                # Keep headers with minimal compression
                compressed_parts.append(section_content)
            elif section_type == "list":
                # Compress lists by taking key items
                items = section_content.split('\n')
                keep_count = max(1, int(len(items) * target_ratio * 1.5))
                compressed_parts.append('\n'.join(items[:keep_count]))
            else:
                # Regular text: use summarization
                compressed_text, _ = self._summarize_content(section_content, target_ratio)
                compressed_parts.append(compressed_text)
        
        hierarchical_result = '\n\n'.join(compressed_parts)
        actual_ratio = len(hierarchical_result) / len(content) if content else 1.0
        
        return hierarchical_result, actual_ratio
    
    def _extract_structure(self, content: str) -> List[Tuple[str, str]]:
        """Extract structural elements from content"""
        
        lines = content.split('\n')
        structured = []
        current_section = []
        current_type = "text"
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Detect headers (lines with specific patterns)
            if (line_stripped.startswith('#') or 
                (len(line_stripped) < 50 and line_stripped.isupper()) or
                line_stripped.endswith(':')):
                
                # Save previous section
                if current_section:
                    structured.append((current_type, '\n'.join(current_section)))
                
                structured.append(("header", line_stripped))
                current_section = []
                current_type = "text"
            
            # Detect lists
            elif (line_stripped.startswith('-') or line_stripped.startswith('*') or
                  line_stripped.startswith('•') or 
                  re.match(r'^\d+\.', line_stripped)):
                
                if current_type != "list":
                    # Save previous section
                    if current_section:
                        structured.append((current_type, '\n'.join(current_section)))
                    current_section = []
                    current_type = "list"
                
                current_section.append(line_stripped)
            
            else:
                if current_type != "text":
                    # Save previous section
                    if current_section:
                        structured.append((current_type, '\n'.join(current_section)))
                    current_section = []
                    current_type = "text"
                
                current_section.append(line)
        
        # Save final section
        if current_section:
            structured.append((current_type, '\n'.join(current_section)))
        
        return structured

class ContextWindow:
    """Manages a context window with intelligent content curation"""
    
    def __init__(self, max_tokens: int = 4000, model_name: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(model_name)
        self.compressor = ContentCompressor()
        
        # Context storage
        self.items: Dict[str, ContextItem] = {}
        self.item_order: deque = deque()  # Maintains insertion order
        
        # Current token usage
        self.current_tokens = 0
        self.reserved_tokens = 500  # Reserve for response
        
        # Management strategies
        self.compression_threshold = 0.8  # Start compression at 80% capacity
        self.eviction_threshold = 0.95    # Start eviction at 95% capacity
        
        # Statistics
        self.stats = {
            'items_added': 0,
            'items_compressed': 0,
            'items_evicted': 0,
            'compressions_performed': 0,
            'total_tokens_processed': 0
        }
        
        self.logger = logging.getLogger("ContextWindow")
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content"""
        return self.max_tokens - self.current_tokens - self.reserved_tokens
    
    @property
    def utilization(self) -> float:
        """Get current token utilization ratio"""
        return self.current_tokens / (self.max_tokens - self.reserved_tokens)
    
    def add_item(self, content: str, content_type: ContentType, 
                priority: Priority = Priority.MEDIUM, 
                relevance_score: float = 1.0) -> str:
        """Add an item to the context window"""
        
        # Create context item
        item = ContextItem(
            id="",
            content=content,
            content_type=content_type,
            priority=priority,
            relevance_score=relevance_score,
            token_count=self.token_counter.count_tokens(content)
        )
        
        # Check if we need to make space
        if self.current_tokens + item.token_count > self.max_tokens - self.reserved_tokens:
            self._make_space(item.token_count)
        
        # Add item
        self.items[item.id] = item
        self.item_order.append(item.id)
        self.current_tokens += item.token_count
        
        self.stats['items_added'] += 1
        self.stats['total_tokens_processed'] += item.token_count
        
        self.logger.debug(f"Added item {item.id[:8]}... ({item.token_count} tokens)")
        
        return item.id
    
    def get_item(self, item_id: str) -> Optional[ContextItem]:
        """Get an item from the context window"""
        
        if item_id in self.items:
            item = self.items[item_id]
            item.update_access()
            return item
        
        return None
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the context window"""
        
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # Update token count
        self.current_tokens -= item.token_count
        
        # Remove from storage
        del self.items[item_id]
        
        # Remove from order queue
        try:
            self.item_order.remove(item_id)
        except ValueError:
            pass
        
        self.logger.debug(f"Removed item {item_id[:8]}...")
        
        return True
    
    async def compress_item(self, item_id: str, target_ratio: float = 0.5,
                           strategy: CompressionStrategy = CompressionStrategy.SUMMARIZE) -> bool:
        """Compress a specific item"""
        
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # Don't compress critical items or already compressed items with good ratio
        if (item.priority == Priority.CRITICAL or 
            (item.is_compressed and item.compression_ratio <= target_ratio)):
            return False
        
        # Perform compression
        compressed_content, actual_ratio = await self.compressor.compress_content(
            item.original_content or item.content, target_ratio, strategy
        )
        
        # Update token counts
        old_tokens = item.token_count
        new_tokens = self.token_counter.count_tokens(compressed_content)
        
        # Apply compression
        item.compress(compressed_content, actual_ratio)
        item.token_count = new_tokens
        
        # Update total token count
        self.current_tokens = self.current_tokens - old_tokens + new_tokens
        
        self.stats['items_compressed'] += 1
        self.stats['compressions_performed'] += 1
        
        self.logger.debug(f"Compressed item {item_id[:8]}... "
                         f"({old_tokens} -> {new_tokens} tokens, ratio: {actual_ratio:.2f})")
        
        return True
    
    def _make_space(self, needed_tokens: int) -> None:
        """Make space for new content"""
        
        # Strategy 1: Compress existing items if utilization is high
        if self.utilization > self.compression_threshold:
            self._compress_eligible_items()
        
        # Strategy 2: Evict items if still not enough space
        if self.current_tokens + needed_tokens > self.max_tokens - self.reserved_tokens:
            self._evict_items(needed_tokens)
    
    async def _compress_eligible_items(self) -> None:
        """Compress items that are eligible for compression"""
        
        # Get items sorted by compression priority
        compression_candidates = []
        
        for item in self.items.values():
            if (item.priority != Priority.CRITICAL and 
                not item.is_compressed and 
                item.token_count > 100):  # Only compress substantial items
                
                # Score for compression priority (lower priority, older items first)
                compression_score = (
                    (5 - item.priority.value) * 0.4 +  # Priority weight
                    (1.0 - item.calculate_current_relevance()) * 0.6  # Relevance weight
                )
                
                compression_candidates.append((item.id, compression_score))
        
        # Sort by compression priority
        compression_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Compress items until we're under threshold
        target_tokens = int((self.max_tokens - self.reserved_tokens) * self.compression_threshold)
        
        for item_id, _ in compression_candidates:
            if self.current_tokens <= target_tokens:
                break
            
            # Choose compression strategy based on content type and priority
            item = self.items[item_id]
            
            if item.content_type in [ContentType.USER_MESSAGE, ContentType.ASSISTANT_MESSAGE]:
                strategy = CompressionStrategy.SUMMARIZE
                target_ratio = 0.6
            elif item.content_type == ContentType.FUNCTION_RESULT:
                strategy = CompressionStrategy.EXTRACT_KEY_POINTS
                target_ratio = 0.4
            else:
                strategy = CompressionStrategy.TRUNCATE
                target_ratio = 0.5
            
            await self.compress_item(item_id, target_ratio, strategy)
    
    def _evict_items(self, needed_tokens: int) -> None:
        """Evict items to make space"""
        
        # Get eviction candidates (excluding critical items)
        eviction_candidates = []
        
        for item in self.items.values():
            if item.priority != Priority.CRITICAL:
                # Score for eviction (lower relevance, older items first)
                eviction_score = (
                    (5 - item.priority.value) * 0.3 +  # Priority weight
                    (1.0 - item.calculate_current_relevance()) * 0.5 +  # Relevance weight
                    (item.access_count / 100.0) * 0.2  # Access frequency (inverted)
                )
                
                eviction_candidates.append((item.id, eviction_score, item.token_count))
        
        # Sort by eviction priority (higher score = more likely to evict)
        eviction_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evict items until we have enough space
        tokens_freed = 0
        target_free = needed_tokens + 200  # Extra buffer
        
        for item_id, _, token_count in eviction_candidates:
            if tokens_freed >= target_free:
                break
            
            if self.remove_item(item_id):
                tokens_freed += token_count
                self.stats['items_evicted'] += 1
                
                self.logger.debug(f"Evicted item {item_id[:8]}... ({token_count} tokens)")
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get context formatted as messages for language model"""
        
        messages = []
        
        # Sort items by priority and timestamp
        sorted_items = sorted(
            self.items.values(),
            key=lambda x: (x.priority.value, x.timestamp),
            reverse=True
        )
        
        for item in sorted_items:
            # Convert to message format based on content type
            if item.content_type == ContentType.SYSTEM_MESSAGE:
                messages.append({"role": "system", "content": item.content})
            elif item.content_type == ContentType.USER_MESSAGE:
                messages.append({"role": "user", "content": item.content})
            elif item.content_type == ContentType.ASSISTANT_MESSAGE:
                messages.append({"role": "assistant", "content": item.content})
            elif item.content_type == ContentType.FUNCTION_CALL:
                # Include function calls in assistant messages
                messages.append({"role": "assistant", "content": f"[Function Call] {item.content}"})
            elif item.content_type == ContentType.FUNCTION_RESULT:
                # Include function results in user messages
                messages.append({"role": "user", "content": f"[Function Result] {item.content}"})
            elif item.content_type == ContentType.CONTEXT_SUMMARY:
                # Include summaries as system messages
                messages.append({"role": "system", "content": f"[Context Summary] {item.content}"})
        
        return messages
    
    def create_context_summary(self) -> Optional[str]:
        """Create a summary of current context"""
        
        if not self.items:
            return None
        
        # Group items by type
        grouped_content = defaultdict(list)
        
        for item in self.items.values():
            grouped_content[item.content_type].append(item.content)
        
        # Build summary
        summary_parts = []
        
        # Summarize user messages
        if ContentType.USER_MESSAGE in grouped_content:
            user_messages = grouped_content[ContentType.USER_MESSAGE]
            summary_parts.append(f"User discussed: {', '.join(user_messages[-3:])}")  # Last 3 messages
        
        # Summarize assistant messages
        if ContentType.ASSISTANT_MESSAGE in grouped_content:
            assistant_messages = grouped_content[ContentType.ASSISTANT_MESSAGE]
            summary_parts.append(f"Assistant provided: {', '.join(assistant_messages[-3:])}")
        
        # Include function calls if any
        if ContentType.FUNCTION_CALL in grouped_content:
            summary_parts.append(f"Functions used: {len(grouped_content[ContentType.FUNCTION_CALL])} calls")
        
        return " | ".join(summary_parts) if summary_parts else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context window statistics"""
        
        # Analyze current content
        content_distribution = defaultdict(int)
        priority_distribution = defaultdict(int)
        compression_stats = {'compressed': 0, 'uncompressed': 0}
        
        for item in self.items.values():
            content_distribution[item.content_type.value] += 1
            priority_distribution[item.priority.value] += 1
            
            if item.is_compressed:
                compression_stats['compressed'] += 1
            else:
                compression_stats['uncompressed'] += 1
        
        return {
            'capacity': {
                'max_tokens': self.max_tokens,
                'current_tokens': self.current_tokens,
                'available_tokens': self.available_tokens,
                'utilization': self.utilization,
                'reserved_tokens': self.reserved_tokens
            },
            'content': {
                'total_items': len(self.items),
                'content_type_distribution': dict(content_distribution),
                'priority_distribution': dict(priority_distribution),
                'compression_stats': compression_stats
            },
            'performance': self.stats
        }

class ContextWindowManager:
    """Complete context window management system"""
    
    def __init__(self, max_tokens: int = 4000, model_name: str = "gpt-3.5-turbo"):
        self.context_window = ContextWindow(max_tokens, model_name)
        
        # Context management
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_summaries: List[str] = []
        
        # Auto-management settings
        self.auto_compression = True
        self.auto_summarization = True
        self.summary_interval = 10  # Create summary every N interactions
        
        # Statistics
        self.stats = {
            'conversations_managed': 0,
            'summaries_created': 0,
            'context_switches': 0
        }
        
        self.logger = logging.getLogger("ContextWindowManager")
    
    async def initialize(self) -> None:
        """Initialize the context window manager"""
        self.logger.info("Context window manager initialized")
    
    def add_system_message(self, content: str) -> str:
        """Add a system message (highest priority)"""
        
        return self.context_window.add_item(
            content, ContentType.SYSTEM_MESSAGE, Priority.CRITICAL
        )
    
    def add_user_message(self, content: str, priority: Priority = Priority.HIGH) -> str:
        """Add a user message"""
        
        # Record in conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': content,
            'timestamp': datetime.now()
        })
        
        return self.context_window.add_item(
            content, ContentType.USER_MESSAGE, priority
        )
    
    def add_assistant_message(self, content: str, priority: Priority = Priority.HIGH) -> str:
        """Add an assistant message"""
        
        # Record in conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': content,
            'timestamp': datetime.now()
        })
        
        return self.context_window.add_item(
            content, ContentType.ASSISTANT_MESSAGE, priority
        )
    
    def add_function_call(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Add a function call to context"""
        
        content = f"{function_name}({json.dumps(arguments)})"
        
        return self.context_window.add_item(
            content, ContentType.FUNCTION_CALL, Priority.MEDIUM
        )
    
    def add_function_result(self, function_name: str, result: Any) -> str:
        """Add a function result to context"""
        
        content = f"{function_name} → {json.dumps(result) if result else 'None'}"
        
        return self.context_window.add_item(
            content, ContentType.FUNCTION_RESULT, Priority.MEDIUM
        )
    
    async def manage_conversation_turn(self, user_message: str) -> Dict[str, Any]:
        """Manage a full conversation turn"""
        
        # Add user message
        user_item_id = self.add_user_message(user_message)
        
        # Check if we need summarization
        if (self.auto_summarization and 
            len(self.conversation_history) % self.summary_interval == 0):
            await self.create_conversation_summary()
        
        # Prepare context for model
        context_messages = self.context_window.get_context_messages()
        
        self.stats['conversations_managed'] += 1
        
        return {
            'context_messages': context_messages,
            'user_item_id': user_item_id,
            'context_stats': self.context_window.get_statistics()
        }
    
    async def create_conversation_summary(self) -> Optional[str]:
        """Create a summary of recent conversation"""
        
        if len(self.conversation_history) < 5:  # Need minimum history
            return None
        
        # Get recent messages for summarization
        recent_messages = self.conversation_history[-self.summary_interval:]
        
        # Build summary content
        summary_parts = []
        
        user_topics = []
        assistant_responses = []
        
        for msg in recent_messages:
            if msg['role'] == 'user':
                user_topics.append(msg['content'][:100])  # Truncate for summary
            elif msg['role'] == 'assistant':
                assistant_responses.append(msg['content'][:100])
        
        if user_topics:
            summary_parts.append(f"User topics: {' | '.join(user_topics)}")
        
        if assistant_responses:
            summary_parts.append(f"Assistant responses: {' | '.join(assistant_responses)}")
        
        summary = " // ".join(summary_parts)
        
        # Add summary to context
        summary_item_id = self.context_window.add_item(
            summary, ContentType.CONTEXT_SUMMARY, Priority.LOW
        )
        
        self.context_summaries.append(summary)
        self.stats['summaries_created'] += 1
        
        self.logger.debug(f"Created conversation summary: {summary_item_id[:8]}...")
        
        return summary
    
    async def switch_context(self, new_topic: str) -> Dict[str, Any]:
        """Switch to a new conversation context"""
        
        # Create summary of current context before switching
        current_summary = await self.create_conversation_summary()
        
        # Clear low and medium priority items
        items_to_remove = []
        
        for item_id, item in self.context_window.items.items():
            if item.priority in [Priority.LOW, Priority.MEDIUM]:
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            self.context_window.remove_item(item_id)
        
        # Add new topic as system message
        new_topic_id = self.add_system_message(f"Switching context to: {new_topic}")
        
        self.stats['context_switches'] += 1
        
        self.logger.info(f"Switched context to: {new_topic}")
        
        return {
            'previous_summary': current_summary,
            'new_topic_id': new_topic_id,
            'items_cleared': len(items_to_remove)
        }
    
    def optimize_context(self) -> Dict[str, int]:
        """Manually optimize the context window"""
        
        initial_tokens = self.context_window.current_tokens
        initial_items = len(self.context_window.items)
        
        # Force compression of eligible items
        compression_count = 0
        
        for item_id in list(self.context_window.items.keys()):
            item = self.context_window.items[item_id]
            
            if (item.priority != Priority.CRITICAL and 
                not item.is_compressed and 
                item.token_count > 50):
                
                # Compress item
                asyncio.create_task(self.context_window.compress_item(
                    item_id, 0.5, CompressionStrategy.SUMMARIZE
                ))
                compression_count += 1
        
        final_tokens = self.context_window.current_tokens
        final_items = len(self.context_window.items)
        
        return {
            'tokens_saved': initial_tokens - final_tokens,
            'items_before': initial_items,
            'items_after': final_items,
            'compressions_performed': compression_count
        }
    
    def get_context_preview(self, max_items: int = 5) -> List[Dict[str, Any]]:
        """Get a preview of current context items"""
        
        # Get most recent and highest priority items
        sorted_items = sorted(
            self.context_window.items.values(),
            key=lambda x: (x.priority.value, x.timestamp.timestamp()),
            reverse=True
        )
        
        preview = []
        
        for item in sorted_items[:max_items]:
            preview.append({
                'id': item.id[:8],
                'type': item.content_type.value,
                'priority': item.priority.name,
                'tokens': item.token_count,
                'compressed': item.is_compressed,
                'content_preview': item.content[:100] + "..." if len(item.content) > 100 else item.content,
                'relevance': item.calculate_current_relevance()
            })
        
        return preview
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        context_stats = self.context_window.get_statistics()
        
        return {
            'manager_statistics': self.stats,
            'context_window_statistics': context_stats,
            'conversation_history': {
                'total_messages': len(self.conversation_history),
                'summaries_created': len(self.context_summaries)
            },
            'configuration': {
                'auto_compression': self.auto_compression,
                'auto_summarization': self.auto_summarization,
                'summary_interval': self.summary_interval
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_context_management():
    """Demo: Basic context window management"""
    print("\nDEMO 1: BASIC CONTEXT MANAGEMENT")
    print("=" * 50)
    
    manager = ContextWindowManager(max_tokens=1000, model_name="gpt-3.5-turbo")
    await manager.initialize()
    
    print("Setting up conversation context:")
    
    # Add system message
    system_id = manager.add_system_message(
        "You are a helpful AI assistant specializing in programming and technology."
    )
    print(f"  Added system message: {system_id[:8]}...")
    
    # Simulate conversation turns
    conversation_turns = [
        "Hello! I'm working on a Python project and need help with data structures.",
        "I need to choose between lists and dictionaries for storing user data.",
        "The data includes user ID, name, email, and preferences.",
        "I'll be doing frequent lookups by user ID and occasional updates.",
        "What would you recommend and why?"
    ]
    
    print(f"\nSimulating conversation turns:")
    
    for i, user_message in enumerate(conversation_turns, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {user_message[:50]}...")
        
        # Manage conversation turn
        turn_result = await manager.manage_conversation_turn(user_message)
        
        # Simulate assistant response
        assistant_response = f"Based on your requirements for turn {i}, I recommend using dictionaries for O(1) lookup performance by user ID. This is optimal for frequent ID-based queries."
        
        assistant_id = manager.add_assistant_message(assistant_response)
        print(f"  Assistant: {assistant_response[:50]}...")
        
        # Show context stats
        stats = turn_result['context_stats']
        print(f"  Context: {stats['capacity']['current_tokens']}/{stats['capacity']['max_tokens']} tokens "
              f"({stats['capacity']['utilization']:.1%} full)")
    
    # Show final context preview
    print(f"\nFinal context preview:")
    preview = manager.get_context_preview()
    
    for item in preview:
        print(f"  {item['type']} ({item['priority']}) - {item['tokens']} tokens")
        print(f"    {item['content_preview']}")

async def demo_context_compression():
    """Demo: Context compression strategies"""
    print("\nDEMO 2: CONTEXT COMPRESSION")
    print("=" * 50)
    
    manager = ContextWindowManager(max_tokens=800, model_name="gpt-3.5-turbo")
    await manager.initialize()
    
    print("Testing context compression with long content:")
    
    # Add a long system message
    long_system_message = """
    You are an expert software architect with 15 years of experience in building scalable systems.
    Your expertise includes:
    - Distributed systems architecture and microservices design patterns
    - Database design and optimization for high-performance applications
    - Cloud infrastructure deployment using AWS, Azure, and Google Cloud Platform
    - DevOps practices including CI/CD pipelines, containerization, and monitoring
    - Security best practices for web applications and API development
    - Performance optimization techniques for both frontend and backend systems
    - Team leadership and technical mentoring of junior developers
    
    When providing advice, always consider:
    - Scalability implications of design decisions
    - Security vulnerabilities and mitigation strategies
    - Performance impact on user experience
    - Maintenance costs and technical debt
    - Team skills and organizational constraints
    """
    
    system_id = manager.add_system_message(long_system_message)
    print(f"  Added long system message: {system_id[:8]}...")
    
    # Add several user messages to fill up context
    user_messages = [
        "I'm designing a new e-commerce platform that needs to handle 10,000 concurrent users during peak shopping periods. What architecture would you recommend?",
        "The platform needs to support multiple payment gateways including PayPal, Stripe, and custom credit card processing. How should I structure the payment service?",
        "We also need real-time inventory management that updates across multiple warehouses and shows accurate stock levels to customers browsing the website.",
        "The system needs to handle product recommendations based on user behavior, purchase history, and trending items. What ML approach would work best?",
        "We're considering implementing a mobile app alongside the web platform. Should we build separate APIs or can we reuse the same backend services?"
    ]
    
    print(f"\nAdding user messages to fill context:")
    
    for i, message in enumerate(user_messages, 1):
        user_id = manager.add_user_message(message)
        
        # Add corresponding assistant response
        assistant_response = f"For your e-commerce requirement {i}, I recommend implementing a microservices architecture with dedicated services for each major function. This provides scalability, maintainability, and allows independent deployment of components."
        
        assistant_id = manager.add_assistant_message(assistant_response)
        
        stats = manager.context_window.get_statistics()
        print(f"  Turn {i}: {stats['capacity']['current_tokens']}/{stats['capacity']['max_tokens']} tokens "
              f"({stats['capacity']['utilization']:.1%} full)")
        
        # Check if compression occurred
        if stats['content']['compression_stats']['compressed'] > 0:
            print(f"    Compression occurred: {stats['content']['compression_stats']['compressed']} items compressed")
    
    # Manually optimize context
    print(f"\nManually optimizing context:")
    optimization_result = manager.optimize_context()
    
    print(f"  Optimization results:")
    print(f"    Tokens saved: {optimization_result['tokens_saved']}")
    print(f"    Items before: {optimization_result['items_before']}")
    print(f"    Items after: {optimization_result['items_after']}")
    print(f"    Compressions: {optimization_result['compressions_performed']}")
    
    # Show context preview after optimization
    print(f"\nContext after optimization:")
    preview = manager.get_context_preview()
    
    for item in preview:
        compressed_indicator = " (compressed)" if item['compressed'] else ""
        print(f"  {item['type']} - {item['tokens']} tokens{compressed_indicator}")
        print(f"    {item['content_preview']}")

async def demo_context_switching():
    """Demo: Context switching between topics"""
    print("\nDEMO 3: CONTEXT SWITCHING")
    print("=" * 50)
    
    manager = ContextWindowManager(max_tokens=1200, model_name="gpt-3.5-turbo")
    await manager.initialize()
    
    print("Demonstrating context switching between topics:")
    
    # Initial topic: Programming
    print(f"\nTopic 1: Programming Discussion")
    
    programming_messages = [
        "I'm learning Python and need help with object-oriented programming.",
        "Can you explain the difference between classes and objects?",
        "How do inheritance and polymorphism work in Python?"
    ]
    
    for message in programming_messages:
        await manager.manage_conversation_turn(message)
        manager.add_assistant_message(
            f"Great question about Python! Let me explain the OOP concepts..."
        )
    
    stats_before_switch = manager.get_comprehensive_stats()
    print(f"  Context before switch: {stats_before_switch['context_window_statistics']['capacity']['current_tokens']} tokens")
    
    # Switch to new topic
    print(f"\nSwitching to Topic 2: Cooking")
    
    switch_result = await manager.switch_context("Cooking and Recipe Management")
    
    print(f"  Switch results:")
    print(f"    Previous summary: {switch_result['previous_summary'][:100] if switch_result['previous_summary'] else 'None'}...")
    print(f"    Items cleared: {switch_result['items_cleared']}")
    
    # New topic conversation
    cooking_messages = [
        "I want to learn how to make homemade pasta from scratch.",
        "What ingredients do I need for basic egg pasta?",
        "How long should I knead the dough?"
    ]
    
    for message in cooking_messages:
        await manager.manage_conversation_turn(message)
        manager.add_assistant_message(
            f"For pasta making, here's what you need to know..."
        )
    
    stats_after_switch = manager.get_comprehensive_stats()
    print(f"  Context after switch: {stats_after_switch['context_window_statistics']['capacity']['current_tokens']} tokens")
    
    # Switch back to programming
    print(f"\nSwitching back to Topic 1: Programming")
    
    await manager.switch_context("Back to Programming Discussion")
    
    # Continue programming discussion
    await manager.manage_conversation_turn("Now I want to learn about decorators in Python.")
    
    print(f"\nFinal context preview:")
    preview = manager.get_context_preview()
    
    for item in preview:
        print(f"  {item['type']} ({item['priority']}) - {item['content_preview'][:60]}...")

async def demo_function_call_management():
    """Demo: Managing function calls in context"""
    print("\nDEMO 4: FUNCTION CALL MANAGEMENT")
    print("=" * 50)
    
    manager = ContextWindowManager(max_tokens=1000, model_name="gpt-3.5-turbo")
    await manager.initialize()
    
    print("Managing function calls and results in context:")
    
    # Initial user request
    user_message = "I need to analyze the weather data for New York and London, then create a comparison report."
    
    await manager.manage_conversation_turn(user_message)
    
    # Simulate function calls
    print(f"\nSimulating function calls:")
    
    # Function call 1: Get weather data
    weather_call_id = manager.add_function_call(
        "get_weather_data",
        {"city": "New York", "days": 7}
    )
    print(f"  Called get_weather_data for New York: {weather_call_id[:8]}...")
    
    # Function result 1
    ny_weather_result = {
        "city": "New York",
        "temperature_avg": 72,
        "humidity": 65,
        "precipitation": 0.2,
        "forecast": ["sunny", "cloudy", "rainy", "sunny", "sunny", "cloudy", "sunny"]
    }
    
    ny_result_id = manager.add_function_result("get_weather_data", ny_weather_result)
    print(f"  Received NY weather result: {ny_result_id[:8]}...")
    
    # Function call 2: Get weather data for London
    london_call_id = manager.add_function_call(
        "get_weather_data",
        {"city": "London", "days": 7}
    )
    print(f"  Called get_weather_data for London: {london_call_id[:8]}...")
    
    # Function result 2
    london_weather_result = {
        "city": "London", 
        "temperature_avg": 58,
        "humidity": 78,
        "precipitation": 0.8,
        "forecast": ["rainy", "cloudy", "rainy", "cloudy", "sunny", "rainy", "cloudy"]
    }
    
    london_result_id = manager.add_function_result("get_weather_data", london_weather_result)
    print(f"  Received London weather result: {london_result_id[:8]}...")
    
    # Function call 3: Create comparison report
    comparison_call_id = manager.add_function_call(
        "create_weather_comparison",
        {
            "cities": ["New York", "London"],
            "metrics": ["temperature", "humidity", "precipitation"],
            "format": "detailed_report"
        }
    )
    print(f"  Called create_weather_comparison: {comparison_call_id[:8]}...")
    
    # Function result 3
    comparison_result = {
        "report_title": "Weather Comparison: New York vs London",
        "summary": "New York shows warmer and drier conditions compared to London",
        "key_differences": [
            "Temperature: NY 14°F warmer on average",
            "Humidity: London 13% more humid", 
            "Precipitation: London 4x more rainfall expected"
        ],
        "recommendations": "Pack lighter clothes for NY, waterproof gear for London"
    }
    
    comparison_result_id = manager.add_function_result("create_weather_comparison", comparison_result)
    print(f"  Received comparison report: {comparison_result_id[:8]}...")
    
    # Add assistant response using the function results
    assistant_response = """
    Based on the weather analysis, I've generated a comprehensive comparison report. 
    Key findings: New York is significantly warmer and drier than London this week. 
    New York averages 72°F with minimal precipitation, while London averages 58°F with 
    substantial rainfall expected. I recommend packing accordingly for your travels.
    """
    
    manager.add_assistant_message(assistant_response)
    
    # Show context with function calls
    print(f"\nContext including function calls:")
    
    context_messages = manager.context_window.get_context_messages()
    
    for i, message in enumerate(context_messages[-8:], 1):  # Show last 8 messages
        role = message['role']
        content = message['content'][:80] + "..." if len(message['content']) > 80 else message['content']
        print(f"  {i}. {role}: {content}")
    
    # Show context statistics
    stats = manager.get_comprehensive_stats()
    
    print(f"\nContext statistics with function calls:")
    print(f"  Total items: {stats['context_window_statistics']['content']['total_items']}")
    print(f"  Content distribution: {stats['context_window_statistics']['content']['content_type_distribution']}")
    print(f"  Token utilization: {stats['context_window_statistics']['capacity']['utilization']:.1%}")

async def demo_conversation_summarization():
    """Demo: Automatic conversation summarization"""
    print("\nDEMO 5: CONVERSATION SUMMARIZATION")
    print("=" * 50)
    
    manager = ContextWindowManager(max_tokens=1500, model_name="gpt-3.5-turbo")
    await manager.initialize()
    
    # Enable auto-summarization with short interval for demo
    manager.auto_summarization = True
    manager.summary_interval = 4  # Create summary every 4 turns
    
    print("Demonstrating automatic conversation summarization:")
    print(f"  Summary interval: {manager.summary_interval} turns")
    
    # Simulate extended conversation
    conversation_sequence = [
        ("I'm planning a vacation to Europe next month.", "That sounds exciting! Europe has many wonderful destinations..."),
        ("I'm considering Paris, Rome, and Barcelona.", "Excellent choices! Each city offers unique culture and attractions..."), 
        ("How many days should I spend in each city?", "I'd recommend 3-4 days per city to see the main highlights..."),
        ("What's the best way to travel between these cities?", "For Paris-Rome-Barcelona, consider flying or high-speed rail..."),
        # Summary should be created here (after 4 turns)
        ("I prefer trains over flying. Are there good connections?", "Yes! Europe has excellent rail networks, especially high-speed trains..."),
        ("What about accommodations? Hotels vs Airbnb?", "Both have advantages. Hotels offer convenience, Airbnb offers local experience..."),
        ("I'm traveling solo. Any safety tips?", "Solo travel in Europe is generally safe. Keep copies of documents..."),
        ("What about language barriers in these cities?", "In major tourist areas, English is widely spoken, but learning basics helps..."),
        # Another summary should be created here (after 8 turns total)
        ("Should I book attractions in advance?", "For popular attractions like the Louvre or Colosseum, advance booking is essential..."),
        ("What's the best time to visit these places?", "Spring and early fall offer good weather with fewer crowds...")
    ]
    
    summaries_created = []
    
    for i, (user_msg, assistant_msg) in enumerate(conversation_sequence, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {user_msg}")
        
        # Process user message
        turn_result = await manager.manage_conversation_turn(user_msg)
        
        # Add assistant response
        manager.add_assistant_message(assistant_msg)
        
        print(f"  Assistant: {assistant_msg[:60]}...")
        
        # Check if summary was created
        current_summaries = len(manager.context_summaries)
        if current_summaries > len(summaries_created):
            new_summary = manager.context_summaries[-1]
            summaries_created.append(new_summary)
            print(f"  📝 Summary created: {new_summary[:80]}...")
        
        # Show context stats
        stats = turn_result['context_stats']
        print(f"  Context: {stats['capacity']['current_tokens']} tokens, "
              f"{stats['content']['total_items']} items")
    
    # Show all summaries created
    print(f"\nAll conversation summaries created:")
    
    for i, summary in enumerate(summaries_created, 1):
        print(f"  Summary {i}: {summary}")
    
    # Show final context state
    print(f"\nFinal context preview:")
    preview = manager.get_context_preview(max_items=8)
    
    for item in preview:
        print(f"  {item['type']} ({item['priority']}) - {item['tokens']} tokens")
        print(f"    Relevance: {item['relevance']:.2f}")
        print(f"    {item['content_preview']}")
        print()
    
    # Show comprehensive statistics
    final_stats = manager.get_comprehensive_stats()
    
    print(f"Final conversation statistics:")
    print(f"  Total messages processed: {final_stats['conversation_history']['total_messages']}")
    print(f"  Summaries created: {final_stats['conversation_history']['summaries_created']}")
    print(f"  Context switches: {final_stats['manager_statistics']['context_switches']}")
    print(f"  Items compressed: {final_stats['context_window_statistics']['performance']['items_compressed']}")
    print(f"  Items evicted: {final_stats['context_window_statistics']['performance']['items_evicted']}")

async def main():
    """
    Demonstrate Context Window Manager for optimizing limited attention spans
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement intelligent context window management with token counting
    2. How to apply different compression strategies for various content types
    3. How to perform context switching while maintaining conversation continuity
    4. How to manage function calls and results within limited context space
    5. How to create automatic conversation summarization for long interactions
    6. How to build complete context management systems for production AI
    
    REAL WORLD APPLICATIONS:
    =======================
    - Conversational AI systems managing long-form dialogues
    - Customer service bots maintaining context across multiple interactions
    - Code assistance tools managing large codebases within context limits
    - Educational AI tutors tracking learning progress and conversation history
    - Creative writing assistants managing story context and character development
    - Research assistants organizing information across multiple sources and topics
    """
    
    print("CONTEXT WINDOW MANAGER DEMONSTRATION")
    print("Optimizing limited attention for maximum relevance!")
    
    await demo_basic_context_management()
    await demo_context_compression()
    await demo_context_switching()
    await demo_function_call_management()
    await demo_conversation_summarization()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Intelligent context management maximizes limited model attention")
    print("✓ Compression strategies preserve important information while saving space")
    print("✓ Context switching enables topic changes without losing continuity")
    print("✓ Function call management integrates tool usage into conversations")
    print("✓ Automatic summarization handles long conversations gracefully")
    print("✓ Complete systems enable production-ready conversational AI")
    print("\nTHE POWER OF CONTEXT WINDOW MANAGEMENT:")
    print("- Enables AI to maintain coherent long-form conversations")
    print("- Maximizes effectiveness of limited model context windows")
    print("- Supports complex multi-turn interactions and task continuity")
    print("- Creates natural and consistent conversational experiences")
    print("- Essential for production AI systems with real-world requirements")

if __name__ == "__main__":
    asyncio.run(main())
