#!/usr/bin/env python3
"""
Hybrid RAG Architectures: Combining Multiple RAG Approaches for Optimal Performance
=================================================================================

WHAT IS THE PROBLEM?
==================
Single RAG approaches have limitations:
- Dense retrieval misses exact keyword matches
- Sparse retrieval struggles with semantic similarity
- Vector search can't handle structured queries
- Graph RAG doesn't work well for simple questions
- Hierarchical RAG adds unnecessary complexity for basic tasks
- No single approach is optimal for all question types

Example: Enterprise Search Complexity
SINGLE APPROACH LIMITATIONS:
- Technical documentation: Needs exact code matches + semantic understanding
- Business analysis: Requires structured data + unstructured insights
- Customer support: Combines FAQ matching + contextual understanding
- Research queries: Needs recent data + historical context + expert knowledge
- Multi-lingual content: Different retrieval strategies per language

REAL WORLD EXAMPLE:
=================
How does Google Search actually work?

GOOGLE'S HYBRID SEARCH ARCHITECTURE:
1. KEYWORD MATCHING: Traditional TF-IDF for exact matches
2. SEMANTIC SEARCH: BERT/neural networks for intent understanding
3. KNOWLEDGE GRAPHS: Structured entity relationships
4. REAL-TIME INDEXING: Recent content integration
5. PERSONALIZATION: User history and preferences
6. MULTIPLE RANKING: PageRank + relevance + freshness + authority
7. RESULT FUSION: Combine different retrieval approaches

BENEFITS OF HYBRID APPROACH:
- Covers all types of search intents and content
- Optimizes for both precision and recall
- Adapts to different content types and user needs
- Provides backup when one approach fails
- Delivers consistently high-quality results
- Scales to billions of documents effectively

THE HYBRID ADVANTAGE:
===================
TRADITIONAL RAG: One retrieval method for all questions
HYBRID RAG: Intelligently combine multiple approaches based on:
- Question type and complexity
- Content characteristics
- Performance requirements
- Available computing resources
- Quality thresholds

HYBRID STRATEGIES:
================
1. PARALLEL FUSION: Run multiple retrievers simultaneously, merge results
2. SEQUENTIAL ROUTING: Route to different retrievers based on question analysis
3. ADAPTIVE SELECTION: Choose approach based on initial retrieval quality
4. HIERARCHICAL CASCADING: Start simple, escalate complexity as needed
5. ENSEMBLE VOTING: Multiple retrievers vote on document relevance
6. SPECIALIZED ROUTING: Domain-specific retrievers for different content types

WHY THIS IS REVOLUTIONARY:
========================
- Achieves best-of-all-worlds performance across diverse scenarios
- Provides robust fallback mechanisms for consistent quality
- Enables specialized optimization for different use cases
- Supports enterprise-scale deployment with reliability
- Critical for production RAG systems serving diverse needs
- Powers next-generation search and knowledge systems
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, Counter
import re
import math
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RetrieverType(Enum):
    """Types of retrievers in hybrid architecture"""
    DENSE_VECTOR = "dense_vector"         # Semantic similarity retrieval
    SPARSE_KEYWORD = "sparse_keyword"     # Keyword/TF-IDF retrieval
    GRAPH_TRAVERSAL = "graph_traversal"   # Knowledge graph retrieval
    HIERARCHICAL = "hierarchical"         # Multi-level retrieval
    RECENT_TEMPORAL = "recent_temporal"   # Time-based retrieval
    STRUCTURED_QUERY = "structured_query" # SQL-like retrieval
    HYBRID_FUSION = "hybrid_fusion"       # Meta-retriever combining others

class FusionStrategy(Enum):
    """Strategies for combining retrieval results"""
    PARALLEL_FUSION = "parallel_fusion"       # Run all, merge results
    SEQUENTIAL_ROUTING = "sequential_routing" # Route based on question type
    ADAPTIVE_SELECTION = "adaptive_selection" # Choose based on quality
    HIERARCHICAL_CASCADE = "hierarchical_cascade" # Escalate complexity
    ENSEMBLE_VOTING = "ensemble_voting"       # Multiple retrievers vote
    SPECIALIZED_ROUTING = "specialized_routing" # Domain-specific routing

class QuestionType(Enum):
    """Types of questions for routing decisions"""
    FACTUAL = "factual"                 # Simple fact lookup
    ANALYTICAL = "analytical"           # Complex analysis needed
    COMPARATIVE = "comparative"         # Comparing entities
    TEMPORAL = "temporal"               # Time-based questions
    PROCEDURAL = "procedural"           # How-to questions
    CONCEPTUAL = "conceptual"           # Understanding concepts
    MULTI_HOP = "multi_hop"            # Multi-step reasoning

class ContentType(Enum):
    """Types of content for specialized routing"""
    TECHNICAL_DOCS = "technical_docs"
    BUSINESS_REPORTS = "business_reports"
    NEWS_ARTICLES = "news_articles"
    ACADEMIC_PAPERS = "academic_papers"
    LEGAL_DOCUMENTS = "legal_documents"
    PRODUCT_INFO = "product_info"
    CUSTOMER_SUPPORT = "customer_support"

@dataclass
class RetrievalResult:
    """Result from a single retriever"""
    retriever_id: str
    retriever_type: RetrieverType
    documents: List[Dict[str, Any]]
    scores: List[float]
    
    # Retrieval metadata
    query_time: float = 0.0
    total_candidates: int = 0
    confidence_score: float = 0.0
    
    # Quality metrics
    precision_estimate: float = 0.0
    recall_estimate: float = 0.0
    coverage_score: float = 0.0
    
    def __post_init__(self):
        if not self.scores and self.documents:
            self.scores = [1.0] * len(self.documents)
        
        if len(self.scores) != len(self.documents):
            min_len = min(len(self.scores), len(self.documents))
            self.scores = self.scores[:min_len]
            self.documents = self.documents[:min_len]

@dataclass
class HybridRetrievalResult:
    """Combined result from hybrid retrieval"""
    fusion_strategy: FusionStrategy
    individual_results: List[RetrievalResult]
    fused_documents: List[Dict[str, Any]]
    fused_scores: List[float]
    
    # Fusion metadata
    total_retrieval_time: float = 0.0
    retrievers_used: List[str] = field(default_factory=list)
    fusion_confidence: float = 0.0
    
    # Quality metrics
    diversity_score: float = 0.0
    consensus_score: float = 0.0
    overall_quality: float = 0.0

class QuestionAnalyzer:
    """Analyzes questions to determine optimal retrieval strategy"""
    
    def __init__(self):
        # Pattern matching for question types
        self.factual_patterns = [
            r'what\s+is\s+(?:the\s+)?(?:definition|meaning)',
            r'who\s+(?:is|was|are)',
            r'when\s+(?:did|was|were)',
            r'where\s+(?:is|was|are)',
            r'how\s+many|how\s+much'
        ]
        
        self.analytical_patterns = [
            r'(?:analyze|analysis|assess|evaluate)',
            r'(?:implications|impact|effects?)',
            r'(?:advantages?|disadvantages?|pros?\s+and\s+cons?)',
            r'(?:strategy|strategic|approach)'
        ]
        
        self.comparative_patterns = [
            r'(?:compare|comparison|versus|vs\.?)',
            r'(?:difference|similar|alike)',
            r'(?:better|worse|superior|inferior)',
            r'(?:which\s+(?:is\s+)?(?:better|best))'
        ]
        
        self.temporal_patterns = [
            r'(?:trend|trends|trending)',
            r'(?:historical|history|past)',
            r'(?:future|forecast|prediction)',
            r'(?:recent|latest|current)'
        ]
        
        self.procedural_patterns = [
            r'how\s+(?:to|do\s+(?:i|you))',
            r'(?:steps?|process|procedure)',
            r'(?:guide|tutorial|instructions?)',
            r'(?:implement|setup|configure)'
        ]
        
        self.multi_hop_patterns = [
            r'(?:considering|given\s+that|assuming)',
            r'(?:and\s+(?:also\s+)?(?:how|what|why))',
            r'(?:furthermore|additionally|moreover)',
            r'(?:implications?\s+for|impact\s+on)'
        ]
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine type and complexity"""
        
        question_lower = question.lower()
        
        # Determine question type
        question_type = self._classify_question_type(question_lower)
        
        # Assess complexity
        complexity_score = self._assess_complexity(question)
        
        # Determine content requirements
        content_requirements = self._analyze_content_requirements(question_lower)
        
        # Estimate optimal retrievers
        optimal_retrievers = self._suggest_retrievers(question_type, complexity_score, content_requirements)
        
        return {
            'question_type': question_type,
            'complexity_score': complexity_score,
            'content_requirements': content_requirements,
            'optimal_retrievers': optimal_retrievers,
            'fusion_strategy': self._suggest_fusion_strategy(question_type, complexity_score),
            'estimated_difficulty': self._estimate_difficulty(complexity_score, question_type)
        }
    
    def _classify_question_type(self, question_lower: str) -> QuestionType:
        """Classify the type of question"""
        
        # Check for multi-hop first (most complex)
        if self._matches_patterns(question_lower, self.multi_hop_patterns):
            return QuestionType.MULTI_HOP
        
        if self._matches_patterns(question_lower, self.analytical_patterns):
            return QuestionType.ANALYTICAL
        
        if self._matches_patterns(question_lower, self.comparative_patterns):
            return QuestionType.COMPARATIVE
        
        if self._matches_patterns(question_lower, self.temporal_patterns):
            return QuestionType.TEMPORAL
        
        if self._matches_patterns(question_lower, self.procedural_patterns):
            return QuestionType.PROCEDURAL
        
        if self._matches_patterns(question_lower, self.factual_patterns):
            return QuestionType.FACTUAL
        
        # Default to conceptual
        return QuestionType.CONCEPTUAL
    
    def _assess_complexity(self, question: str) -> float:
        """Assess question complexity (0.0 to 1.0)"""
        
        complexity = 0.0
        
        # Length factor
        words = question.split()
        complexity += min(len(words) / 50.0, 0.3)
        
        # Question words (more = more complex)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        q_word_count = sum(1 for word in words if word.lower() in question_words)
        complexity += min(q_word_count * 0.1, 0.2)
        
        # Conjunctions (indicate multi-part questions)
        conjunctions = ['and', 'or', 'but', 'however', 'considering', 'given', 'while', 'whereas']
        conj_count = sum(1 for word in words if word.lower() in conjunctions)
        complexity += min(conj_count * 0.15, 0.3)
        
        # Technical terms (heuristic)
        technical_indicators = ['analysis', 'implementation', 'methodology', 'architecture', 'framework']
        tech_count = sum(1 for word in words if word.lower() in technical_indicators)
        complexity += min(tech_count * 0.1, 0.2)
        
        return min(complexity, 1.0)
    
    def _analyze_content_requirements(self, question_lower: str) -> List[ContentType]:
        """Determine what types of content are needed"""
        
        requirements = []
        
        # Technical content
        if any(term in question_lower for term in ['api', 'code', 'programming', 'software', 'technical', 'implementation']):
            requirements.append(ContentType.TECHNICAL_DOCS)
        
        # Business content
        if any(term in question_lower for term in ['business', 'market', 'strategy', 'financial', 'revenue', 'growth']):
            requirements.append(ContentType.BUSINESS_REPORTS)
        
        # News content
        if any(term in question_lower for term in ['recent', 'latest', 'current', 'news', 'announcement']):
            requirements.append(ContentType.NEWS_ARTICLES)
        
        # Academic content
        if any(term in question_lower for term in ['research', 'study', 'paper', 'academic', 'scholarly', 'peer-reviewed']):
            requirements.append(ContentType.ACADEMIC_PAPERS)
        
        # Legal content
        if any(term in question_lower for term in ['legal', 'law', 'regulation', 'compliance', 'policy']):
            requirements.append(ContentType.LEGAL_DOCUMENTS)
        
        # Product content
        if any(term in question_lower for term in ['product', 'feature', 'specification', 'manual', 'documentation']):
            requirements.append(ContentType.PRODUCT_INFO)
        
        # Support content
        if any(term in question_lower for term in ['how to', 'troubleshoot', 'problem', 'issue', 'support', 'help']):
            requirements.append(ContentType.CUSTOMER_SUPPORT)
        
        return requirements
    
    def _suggest_retrievers(self, question_type: QuestionType, complexity: float, 
                          content_requirements: List[ContentType]) -> List[RetrieverType]:
        """Suggest optimal retrievers for question"""
        
        retrievers = []
        
        # Base retrievers based on question type
        if question_type == QuestionType.FACTUAL:
            retrievers.extend([RetrieverType.SPARSE_KEYWORD, RetrieverType.DENSE_VECTOR])
        
        elif question_type == QuestionType.ANALYTICAL:
            retrievers.extend([RetrieverType.DENSE_VECTOR, RetrieverType.GRAPH_TRAVERSAL])
        
        elif question_type == QuestionType.COMPARATIVE:
            retrievers.extend([RetrieverType.DENSE_VECTOR, RetrieverType.STRUCTURED_QUERY])
        
        elif question_type == QuestionType.TEMPORAL:
            retrievers.extend([RetrieverType.RECENT_TEMPORAL, RetrieverType.DENSE_VECTOR])
        
        elif question_type == QuestionType.PROCEDURAL:
            retrievers.extend([RetrieverType.SPARSE_KEYWORD, RetrieverType.HIERARCHICAL])
        
        elif question_type == QuestionType.MULTI_HOP:
            retrievers.extend([RetrieverType.GRAPH_TRAVERSAL, RetrieverType.HIERARCHICAL, RetrieverType.DENSE_VECTOR])
        
        else:  # CONCEPTUAL
            retrievers.extend([RetrieverType.DENSE_VECTOR, RetrieverType.SPARSE_KEYWORD])
        
        # Add specialized retrievers based on content requirements
        if ContentType.TECHNICAL_DOCS in content_requirements:
            if RetrieverType.SPARSE_KEYWORD not in retrievers:
                retrievers.append(RetrieverType.SPARSE_KEYWORD)
        
        if ContentType.NEWS_ARTICLES in content_requirements:
            if RetrieverType.RECENT_TEMPORAL not in retrievers:
                retrievers.append(RetrieverType.RECENT_TEMPORAL)
        
        # Add complexity-based retrievers
        if complexity > 0.7:
            if RetrieverType.GRAPH_TRAVERSAL not in retrievers:
                retrievers.append(RetrieverType.GRAPH_TRAVERSAL)
        
        return list(set(retrievers))  # Remove duplicates
    
    def _suggest_fusion_strategy(self, question_type: QuestionType, complexity: float) -> FusionStrategy:
        """Suggest optimal fusion strategy"""
        
        if complexity > 0.8:
            return FusionStrategy.ENSEMBLE_VOTING
        
        elif question_type in [QuestionType.MULTI_HOP, QuestionType.ANALYTICAL]:
            return FusionStrategy.HIERARCHICAL_CASCADE
        
        elif question_type == QuestionType.FACTUAL:
            return FusionStrategy.ADAPTIVE_SELECTION
        
        elif complexity > 0.5:
            return FusionStrategy.PARALLEL_FUSION
        
        else:
            return FusionStrategy.SEQUENTIAL_ROUTING
    
    def _estimate_difficulty(self, complexity: float, question_type: QuestionType) -> str:
        """Estimate retrieval difficulty"""
        
        type_difficulty = {
            QuestionType.FACTUAL: 0.2,
            QuestionType.PROCEDURAL: 0.3,
            QuestionType.CONCEPTUAL: 0.4,
            QuestionType.TEMPORAL: 0.5,
            QuestionType.COMPARATIVE: 0.6,
            QuestionType.ANALYTICAL: 0.7,
            QuestionType.MULTI_HOP: 0.9
        }
        
        total_difficulty = (complexity + type_difficulty.get(question_type, 0.5)) / 2
        
        if total_difficulty < 0.3:
            return "easy"
        elif total_difficulty < 0.6:
            return "medium"
        elif total_difficulty < 0.8:
            return "hard"
        else:
            return "very_hard"
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns"""
        return any(re.search(pattern, text) for pattern in patterns)

class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    def __init__(self, retriever_id: str, retriever_type: RetrieverType):
        self.retriever_id = retriever_id
        self.retriever_type = retriever_type
        
        # Performance tracking
        self.total_queries = 0
        self.total_time = 0.0
        self.average_precision = 0.0
        self.success_rate = 0.0
        
        self.logger = logging.getLogger(f"Retriever-{retriever_id}")
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """Retrieve documents for query"""
        pass
    
    def update_performance_metrics(self, query_time: float, precision: float, success: bool) -> None:
        """Update performance metrics"""
        self.total_queries += 1
        self.total_time += query_time
        
        # Running average of precision
        self.average_precision = (
            (self.average_precision * (self.total_queries - 1) + precision) / self.total_queries
        )
        
        # Running average of success rate
        success_value = 1.0 if success else 0.0
        self.success_rate = (
            (self.success_rate * (self.total_queries - 1) + success_value) / self.total_queries
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / max(self.total_queries, 1)
        
        return {
            'retriever_id': self.retriever_id,
            'retriever_type': self.retriever_type.value,
            'total_queries': self.total_queries,
            'average_query_time': avg_time,
            'average_precision': self.average_precision,
            'success_rate': self.success_rate
        }

class DenseVectorRetriever(BaseRetriever):
    """Dense vector similarity retrieval using embeddings"""
    
    def __init__(self, retriever_id: str = "dense_vector"):
        super().__init__(retriever_id, RetrieverType.DENSE_VECTOR)
        
        # Simulated document embeddings and metadata
        self.document_embeddings = self._create_sample_embeddings()
        self.documents = self._create_sample_documents()
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """Retrieve using dense vector similarity"""
        
        start_time = time.time()
        
        try:
            # Simulate query embedding
            query_embedding = self._encode_query(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.document_embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Prepare results
            documents = []
            scores = []
            
            for doc_idx, score in top_results:
                documents.append(self.documents[doc_idx])
                scores.append(float(score))
            
            query_time = time.time() - start_time
            
            # Estimate quality metrics
            precision_estimate = min(1.0, sum(scores) / max(len(scores), 1))
            confidence_score = max(scores) if scores else 0.0
            
            result = RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=documents,
                scores=scores,
                query_time=query_time,
                total_candidates=len(self.documents),
                confidence_score=confidence_score,
                precision_estimate=precision_estimate,
                recall_estimate=min(1.0, len(documents) / 20),  # Assume 20 relevant docs
                coverage_score=min(1.0, len(documents) / top_k)
            )
            
            # Update performance metrics
            self.update_performance_metrics(query_time, precision_estimate, confidence_score > 0.5)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Dense vector retrieval failed: {e}")
            return RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=[],
                scores=[],
                query_time=time.time() - start_time
            )
    
    def _create_sample_embeddings(self) -> List[List[float]]:
        """Create sample document embeddings"""
        np.random.seed(42)
        num_docs = 100
        embedding_dim = 384
        
        embeddings = []
        for _ in range(num_docs):
            # Create diverse embeddings for different topics
            embedding = np.random.normal(0, 1, embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    def _create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents"""
        documents = []
        
        topics = [
            "artificial intelligence and machine learning",
            "business strategy and market analysis",
            "technology trends and innovation",
            "financial markets and investment",
            "healthcare and medical research",
            "environmental sustainability",
            "software development and programming",
            "data science and analytics"
        ]
        
        for i in range(100):
            topic = topics[i % len(topics)]
            doc = {
                'id': f'doc_{i:03d}',
                'title': f'Document about {topic} - Part {i // len(topics) + 1}',
                'content': f'This is comprehensive content about {topic}. '
                          f'It contains detailed information and analysis relevant to the topic. '
                          f'Document {i} provides specific insights and examples.',
                'topic': topic,
                'content_type': 'general',
                'timestamp': f'2024-01-{(i % 30) + 1:02d}',
                'source': f'source_{i % 10}'
            }
            documents.append(doc)
        
        return documents
    
    def _encode_query(self, query: str) -> List[float]:
        """Encode query into embedding (simulated)"""
        # Simulate query encoding based on content
        np.random.seed(hash(query) % (2**32))
        embedding = np.random.normal(0, 1, 384)
        
        # Add some topic-specific bias
        if 'ai' in query.lower() or 'artificial intelligence' in query.lower():
            embedding[:50] += 0.5  # Boost AI-related dimensions
        elif 'business' in query.lower() or 'market' in query.lower():
            embedding[50:100] += 0.5  # Boost business-related dimensions
        elif 'technology' in query.lower() or 'tech' in query.lower():
            embedding[100:150] += 0.5  # Boost tech-related dimensions
        
        return (embedding / np.linalg.norm(embedding)).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class SparseKeywordRetriever(BaseRetriever):
    """Sparse keyword retrieval using TF-IDF"""
    
    def __init__(self, retriever_id: str = "sparse_keyword"):
        super().__init__(retriever_id, RetrieverType.SPARSE_KEYWORD)
        
        # Create sample documents and build TF-IDF index
        self.documents = self._create_sample_documents()
        self.tfidf_index = self._build_tfidf_index()
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """Retrieve using TF-IDF keyword matching"""
        
        start_time = time.time()
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query.lower())
            
            # Calculate TF-IDF scores for each document
            doc_scores = []
            
            for i, doc in enumerate(self.documents):
                score = self._calculate_tfidf_score(query_tokens, i)
                if score > 0:
                    doc_scores.append((i, score))
            
            # Sort by score and get top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = doc_scores[:top_k]
            
            # Prepare results
            documents = []
            scores = []
            
            for doc_idx, score in top_results:
                documents.append(self.documents[doc_idx])
                scores.append(float(score))
            
            query_time = time.time() - start_time
            
            # Estimate quality metrics
            precision_estimate = 0.8 if scores else 0.0  # Keyword matching is precise
            confidence_score = max(scores) if scores else 0.0
            
            result = RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=documents,
                scores=scores,
                query_time=query_time,
                total_candidates=len(self.documents),
                confidence_score=confidence_score,
                precision_estimate=precision_estimate,
                recall_estimate=min(0.6, len(documents) / 15),  # Keyword matching has good precision, lower recall
                coverage_score=min(1.0, len(documents) / top_k)
            )
            
            # Update performance metrics
            self.update_performance_metrics(query_time, precision_estimate, confidence_score > 0.3)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sparse keyword retrieval failed: {e}")
            return RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=[],
                scores=[],
                query_time=time.time() - start_time
            )
    
    def _create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents with keyword-rich content"""
        documents = []
        
        content_templates = [
            "Artificial intelligence and machine learning applications in {domain}. "
            "Deep learning neural networks provide advanced capabilities for {task}. "
            "AI algorithms enable automated {process} with high accuracy and efficiency.",
            
            "Business strategy analysis for {domain} market expansion. "
            "Strategic planning and competitive analysis for {task} optimization. "
            "Market research and business intelligence for {process} improvement.",
            
            "Technology trends and innovation in {domain} sector. "
            "Technical implementation and software architecture for {task}. "
            "Development methodology and engineering practices for {process}.",
            
            "Financial analysis and investment strategy for {domain} markets. "
            "Portfolio management and risk assessment for {task} evaluation. "
            "Economic indicators and market trends affecting {process}.",
        ]
        
        domains = ["healthcare", "finance", "technology", "education", "manufacturing", "retail"]
        tasks = ["automation", "optimization", "analysis", "prediction", "classification", "monitoring"]
        processes = ["workflow", "decision-making", "data processing", "quality control", "reporting"]
        
        for i in range(100):
            template = content_templates[i % len(content_templates)]
            domain = domains[i % len(domains)]
            task = tasks[(i // len(domains)) % len(tasks)]
            process = processes[(i // (len(domains) * len(tasks))) % len(processes)]
            
            content = template.format(domain=domain, task=task, process=process)
            
            doc = {
                'id': f'doc_{i:03d}',
                'title': f'{domain.title()} {task.title()} Guide',
                'content': content,
                'domain': domain,
                'task': task,
                'process': process,
                'timestamp': f'2024-01-{(i % 30) + 1:02d}',
                'source': f'source_{i % 5}'
            }
            documents.append(doc)
        
        return documents
    
    def _build_tfidf_index(self) -> Dict[str, Dict[int, float]]:
        """Build TF-IDF index from documents"""
        
        # Calculate term frequencies
        doc_term_freq = []
        all_terms = set()
        
        for doc in self.documents:
            terms = self._tokenize(doc['content'].lower())
            term_freq = Counter(terms)
            doc_term_freq.append(term_freq)
            all_terms.update(terms)
        
        # Calculate document frequencies
        doc_freq = {}
        num_docs = len(self.documents)
        
        for term in all_terms:
            doc_freq[term] = sum(1 for tf in doc_term_freq if term in tf)
        
        # Build TF-IDF index
        tfidf_index = defaultdict(dict)
        
        for doc_idx, term_freq in enumerate(doc_term_freq):
            for term, tf in term_freq.items():
                df = doc_freq[term]
                idf = math.log(num_docs / df) if df > 0 else 0
                tfidf_score = tf * idf
                
                if tfidf_score > 0:
                    tfidf_index[term][doc_idx] = tfidf_score
        
        return dict(tfidf_index)
    
    def _calculate_tfidf_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate TF-IDF score for document given query tokens"""
        
        score = 0.0
        
        for token in query_tokens:
            if token in self.tfidf_index:
                if doc_idx in self.tfidf_index[token]:
                    score += self.tfidf_index[token][doc_idx]
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return filtered_tokens

class GraphTraversalRetriever(BaseRetriever):
    """Graph-based retrieval using entity relationships"""
    
    def __init__(self, retriever_id: str = "graph_traversal"):
        super().__init__(retriever_id, RetrieverType.GRAPH_TRAVERSAL)
        
        # Create sample knowledge graph
        self.knowledge_graph = self._create_sample_graph()
        self.documents = self._create_graph_documents()
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """Retrieve using graph traversal"""
        
        start_time = time.time()
        
        try:
            # Extract entities from query
            query_entities = self._extract_entities(query)
            
            # Find related entities through graph traversal
            related_entities = set()
            for entity in query_entities:
                related = self._traverse_graph(entity, max_depth=2)
                related_entities.update(related)
            
            # Score documents based on entity relevance
            doc_scores = []
            
            for i, doc in enumerate(self.documents):
                score = self._calculate_entity_score(doc, query_entities, related_entities)
                if score > 0:
                    doc_scores.append((i, score))
            
            # Sort by score and get top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = doc_scores[:top_k]
            
            # Prepare results
            documents = []
            scores = []
            
            for doc_idx, score in top_results:
                documents.append(self.documents[doc_idx])
                scores.append(float(score))
            
            query_time = time.time() - start_time
            
            # Estimate quality metrics
            precision_estimate = 0.75 if scores else 0.0  # Graph retrieval is good for connected concepts
            confidence_score = max(scores) if scores else 0.0
            
            result = RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=documents,
                scores=scores,
                query_time=query_time,
                total_candidates=len(self.documents),
                confidence_score=confidence_score,
                precision_estimate=precision_estimate,
                recall_estimate=min(0.8, len(documents) / 12),  # Good recall for connected concepts
                coverage_score=min(1.0, len(documents) / top_k)
            )
            
            # Update performance metrics
            self.update_performance_metrics(query_time, precision_estimate, confidence_score > 0.4)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Graph traversal retrieval failed: {e}")
            return RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=[],
                scores=[],
                query_time=time.time() - start_time
            )
    
    def _create_sample_graph(self) -> Dict[str, List[str]]:
        """Create sample knowledge graph"""
        
        graph = {
            # AI/ML entities
            'artificial_intelligence': ['machine_learning', 'deep_learning', 'neural_networks', 'automation'],
            'machine_learning': ['artificial_intelligence', 'data_science', 'algorithms', 'prediction'],
            'deep_learning': ['artificial_intelligence', 'neural_networks', 'computer_vision', 'nlp'],
            'neural_networks': ['deep_learning', 'artificial_intelligence', 'backpropagation'],
            
            # Business entities
            'business_strategy': ['market_analysis', 'competitive_advantage', 'growth', 'planning'],
            'market_analysis': ['business_strategy', 'research', 'competitors', 'trends'],
            'financial_planning': ['business_strategy', 'investment', 'budgeting', 'roi'],
            
            # Technology entities
            'software_development': ['programming', 'coding', 'applications', 'systems'],
            'programming': ['software_development', 'algorithms', 'languages', 'debugging'],
            'cloud_computing': ['aws', 'azure', 'scalability', 'infrastructure'],
            
            # Healthcare entities
            'healthcare': ['medical_research', 'patient_care', 'diagnosis', 'treatment'],
            'medical_research': ['healthcare', 'clinical_trials', 'pharmaceuticals', 'innovation'],
            
            # Finance entities
            'investment': ['portfolio', 'risk_management', 'returns', 'markets'],
            'portfolio': ['investment', 'diversification', 'assets', 'allocation'],
        }
        
        return graph
    
    def _create_graph_documents(self) -> List[Dict[str, Any]]:
        """Create documents connected to graph entities"""
        
        documents = []
        entities = list(self.knowledge_graph.keys())
        
        for i in range(80):
            # Choose 1-3 entities for this document
            num_entities = min(3, max(1, np.random.poisson(2)))
            doc_entities = np.random.choice(entities, size=num_entities, replace=False).tolist()
            
            # Create content based on entities
            content_parts = []
            for entity in doc_entities:
                entity_text = entity.replace('_', ' ').title()
                content_parts.append(f"This document covers {entity_text} and its applications.")
                
                # Add related entities
                if entity in self.knowledge_graph:
                    related = self.knowledge_graph[entity][:2]  # Take first 2 related
                    related_text = ', '.join(r.replace('_', ' ') for r in related)
                    content_parts.append(f"It is closely related to {related_text}.")
            
            content = ' '.join(content_parts)
            
            doc = {
                'id': f'graph_doc_{i:03d}',
                'title': f'Guide to {", ".join(e.replace("_", " ").title() for e in doc_entities)}',
                'content': content,
                'entities': doc_entities,
                'primary_entity': doc_entities[0],
                'timestamp': f'2024-01-{(i % 30) + 1:02d}',
                'source': f'knowledge_base_{i % 8}'
            }
            documents.append(doc)
        
        return documents
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query (simple matching)"""
        
        query_lower = query.lower()
        entities = []
        
        for entity in self.knowledge_graph.keys():
            entity_readable = entity.replace('_', ' ')
            if entity_readable in query_lower or entity in query_lower:
                entities.append(entity)
        
        return entities
    
    def _traverse_graph(self, start_entity: str, max_depth: int = 2) -> Set[str]:
        """Traverse graph to find related entities"""
        
        visited = set()
        queue = [(start_entity, 0)]
        
        while queue:
            entity, depth = queue.pop(0)
            
            if entity in visited or depth > max_depth:
                continue
            
            visited.add(entity)
            
            if entity in self.knowledge_graph:
                for related_entity in self.knowledge_graph[entity]:
                    if related_entity not in visited:
                        queue.append((related_entity, depth + 1))
        
        return visited
    
    def _calculate_entity_score(self, doc: Dict[str, Any], 
                              query_entities: List[str], 
                              related_entities: Set[str]) -> float:
        """Calculate score based on entity matches"""
        
        score = 0.0
        doc_entities = set(doc.get('entities', []))
        
        # Direct entity matches (high weight)
        direct_matches = len(set(query_entities) & doc_entities)
        score += direct_matches * 2.0
        
        # Related entity matches (lower weight)
        related_matches = len(related_entities & doc_entities)
        score += related_matches * 0.5
        
        return score

class RecentTemporalRetriever(BaseRetriever):
    """Retriever focusing on recent/temporal information"""
    
    def __init__(self, retriever_id: str = "recent_temporal"):
        super().__init__(retriever_id, RetrieverType.RECENT_TEMPORAL)
        
        self.documents = self._create_temporal_documents()
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> RetrievalResult:
        """Retrieve with temporal relevance"""
        
        start_time = time.time()
        
        try:
            # Check if query has temporal indicators
            temporal_weight = self._assess_temporal_relevance(query)
            
            # Score documents based on content relevance and recency
            doc_scores = []
            
            for i, doc in enumerate(self.documents):
                content_score = self._calculate_content_relevance(query, doc)
                temporal_score = self._calculate_temporal_score(doc)
                
                # Combine scores
                total_score = (content_score * (1 - temporal_weight) + 
                             temporal_score * temporal_weight)
                
                if total_score > 0:
                    doc_scores.append((i, total_score))
            
            # Sort by score and get top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = doc_scores[:top_k]
            
            # Prepare results
            documents = []
            scores = []
            
            for doc_idx, score in top_results:
                documents.append(self.documents[doc_idx])
                scores.append(float(score))
            
            query_time = time.time() - start_time
            
            # Estimate quality metrics
            precision_estimate = 0.7 if scores else 0.0
            confidence_score = max(scores) if scores else 0.0
            
            result = RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=documents,
                scores=scores,
                query_time=query_time,
                total_candidates=len(self.documents),
                confidence_score=confidence_score,
                precision_estimate=precision_estimate,
                recall_estimate=min(0.6, len(documents) / 18),
                coverage_score=min(1.0, len(documents) / top_k)
            )
            
            # Update performance metrics
            self.update_performance_metrics(query_time, precision_estimate, confidence_score > 0.3)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Temporal retrieval failed: {e}")
            return RetrievalResult(
                retriever_id=self.retriever_id,
                retriever_type=self.retriever_type,
                documents=[],
                scores=[],
                query_time=time.time() - start_time
            )
    
    def _create_temporal_documents(self) -> List[Dict[str, Any]]:
        """Create documents with temporal characteristics"""
        
        documents = []
        
        # Create documents with different ages
        import datetime
        base_date = datetime.datetime(2024, 1, 1)
        
        topics = [
            "latest AI developments and breakthroughs",
            "recent market trends and analysis", 
            "current technology innovations",
            "breaking news in healthcare research",
            "new financial regulations and policies",
            "emerging cybersecurity threats",
            "recent scientific discoveries",
            "updated business strategies"
        ]
        
        for i in range(60):
            # Create documents with varying recency
            days_old = i  # Recent to older
            doc_date = base_date + datetime.timedelta(days=days_old)
            
            topic = topics[i % len(topics)]
            
            # Add temporal keywords based on recency
            if days_old < 7:
                temporal_keywords = ["latest", "breaking", "just announced", "new"]
            elif days_old < 30:
                temporal_keywords = ["recent", "current", "updated", "fresh"]
            elif days_old < 90:
                temporal_keywords = ["current", "established", "ongoing"]
            else:
                temporal_keywords = ["historical", "established", "traditional"]
            
            keyword = temporal_keywords[i % len(temporal_keywords)]
            
            content = f"This is {keyword} information about {topic}. "
            content += f"Published on {doc_date.strftime('%Y-%m-%d')}, this content provides "
            content += f"{keyword} insights and analysis relevant to current developments."
            
            doc = {
                'id': f'temporal_doc_{i:03d}',
                'title': f'{keyword.title()} {topic.title()}',
                'content': content,
                'topic': topic,
                'publish_date': doc_date.isoformat(),
                'days_old': days_old,
                'temporal_keywords': temporal_keywords,
                'source': f'news_source_{i % 6}'
            }
            documents.append(doc)
        
        return documents
    
    def _assess_temporal_relevance(self, query: str) -> float:
        """Assess how much the query cares about temporal relevance"""
        
        query_lower = query.lower()
        
        # Strong temporal indicators
        strong_temporal = ['latest', 'recent', 'current', 'new', 'breaking', 'today', 'now', 'updated']
        
        # Moderate temporal indicators  
        moderate_temporal = ['trend', 'development', 'change', 'evolution', 'progress']
        
        strong_count = sum(1 for term in strong_temporal if term in query_lower)
        moderate_count = sum(1 for term in moderate_temporal if term in query_lower)
        
        # Calculate temporal weight (0.0 to 1.0)
        temporal_weight = min(1.0, strong_count * 0.4 + moderate_count * 0.2)
        
        return temporal_weight
    
    def _calculate_content_relevance(self, query: str, doc: Dict[str, Any]) -> float:
        """Calculate content relevance score"""
        
        query_words = set(query.lower().split())
        doc_words = set(doc['content'].lower().split())
        
        # Simple overlap scoring
        overlap = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        return overlap / max(union, 1)
    
    def _calculate_temporal_score(self, doc: Dict[str, Any]) -> float:
        """Calculate temporal relevance score (newer = higher)"""
        
        days_old = doc.get('days_old', 365)
        
        # Exponential decay with recency
        temporal_score = math.exp(-days_old / 30.0)  # 30-day half-life
        
        return temporal_score

class ResultFusionEngine:
    """Engine for fusing results from multiple retrievers"""
    
    def __init__(self):
        self.logger = logging.getLogger("ResultFusionEngine")
    
    async def fuse_results(self, results: List[RetrievalResult], 
                          fusion_strategy: FusionStrategy,
                          target_size: int = 10) -> HybridRetrievalResult:
        """Fuse results from multiple retrievers"""
        
        start_time = time.time()
        
        try:
            if fusion_strategy == FusionStrategy.PARALLEL_FUSION:
                fused_docs, fused_scores = await self._parallel_fusion(results, target_size)
            
            elif fusion_strategy == FusionStrategy.ENSEMBLE_VOTING:
                fused_docs, fused_scores = await self._ensemble_voting(results, target_size)
            
            elif fusion_strategy == FusionStrategy.ADAPTIVE_SELECTION:
                fused_docs, fused_scores = await self._adaptive_selection(results, target_size)
            
            elif fusion_strategy == FusionStrategy.HIERARCHICAL_CASCADE:
                fused_docs, fused_scores = await self._hierarchical_cascade(results, target_size)
            
            else:  # Default to parallel fusion
                fused_docs, fused_scores = await self._parallel_fusion(results, target_size)
            
            # Calculate fusion metrics
            diversity_score = self._calculate_diversity(fused_docs)
            consensus_score = self._calculate_consensus(results, fused_docs)
            fusion_confidence = self._calculate_fusion_confidence(results, fused_scores)
            
            fusion_time = time.time() - start_time
            
            hybrid_result = HybridRetrievalResult(
                fusion_strategy=fusion_strategy,
                individual_results=results,
                fused_documents=fused_docs,
                fused_scores=fused_scores,
                total_retrieval_time=fusion_time,
                retrievers_used=[r.retriever_id for r in results],
                fusion_confidence=fusion_confidence,
                diversity_score=diversity_score,
                consensus_score=consensus_score,
                overall_quality=(fusion_confidence + diversity_score + consensus_score) / 3
            )
            
            return hybrid_result
            
        except Exception as e:
            self.logger.error(f"Result fusion failed: {e}")
            return HybridRetrievalResult(
                fusion_strategy=fusion_strategy,
                individual_results=results,
                fused_documents=[],
                fused_scores=[],
                total_retrieval_time=time.time() - start_time
            )
    
    async def _parallel_fusion(self, results: List[RetrievalResult], 
                             target_size: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Fuse results by combining and re-ranking all documents"""
        
        # Collect all documents with metadata
        all_docs = []
        
        for result in results:
            retriever_weight = self._get_retriever_weight(result.retriever_type)
            
            for doc, score in zip(result.documents, result.scores):
                doc_entry = {
                    'document': doc,
                    'original_score': score,
                    'weighted_score': score * retriever_weight,
                    'retriever_id': result.retriever_id,
                    'retriever_type': result.retriever_type.value
                }
                all_docs.append(doc_entry)
        
        # Remove duplicates (based on document ID)
        seen_ids = set()
        unique_docs = []
        
        for doc_entry in all_docs:
            doc_id = doc_entry['document'].get('id', str(hash(doc_entry['document']['content'])))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc_entry)
        
        # Sort by weighted score
        unique_docs.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Take top results
        top_docs = unique_docs[:target_size]
        
        fused_documents = [entry['document'] for entry in top_docs]
        fused_scores = [entry['weighted_score'] for entry in top_docs]
        
        return fused_documents, fused_scores
    
    async def _ensemble_voting(self, results: List[RetrievalResult], 
                             target_size: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Fuse results using ensemble voting"""
        
        # Collect document votes
        doc_votes = defaultdict(list)
        
        for result in results:
            retriever_weight = self._get_retriever_weight(result.retriever_type)
            
            for doc, score in zip(result.documents, result.scores):
                doc_id = doc.get('id', str(hash(doc['content'])))
                doc_votes[doc_id].append({
                    'document': doc,
                    'score': score,
                    'weight': retriever_weight,
                    'retriever_type': result.retriever_type.value
                })
        
        # Calculate ensemble scores
        doc_ensemble_scores = []
        
        for doc_id, votes in doc_votes.items():
            # Weighted average of scores
            total_weighted_score = sum(vote['score'] * vote['weight'] for vote in votes)
            total_weight = sum(vote['weight'] for vote in votes)
            
            ensemble_score = total_weighted_score / max(total_weight, 1)
            
            # Bonus for consensus (more retrievers agreeing)
            consensus_bonus = len(votes) * 0.1
            final_score = ensemble_score + consensus_bonus
            
            doc_ensemble_scores.append({
                'document': votes[0]['document'],  # Same document from all votes
                'ensemble_score': final_score,
                'vote_count': len(votes),
                'retriever_types': [vote['retriever_type'] for vote in votes]
            })
        
        # Sort by ensemble score
        doc_ensemble_scores.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        # Take top results
        top_docs = doc_ensemble_scores[:target_size]
        
        fused_documents = [entry['document'] for entry in top_docs]
        fused_scores = [entry['ensemble_score'] for entry in top_docs]
        
        return fused_documents, fused_scores
    
    async def _adaptive_selection(self, results: List[RetrievalResult], 
                                target_size: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Adaptively select best retriever based on result quality"""
        
        # Evaluate retriever performance for this query
        best_result = None
        best_quality = 0.0
        
        for result in results:
            # Calculate overall quality score
            quality = (result.confidence_score * 0.4 + 
                      result.precision_estimate * 0.3 + 
                      result.recall_estimate * 0.3)
            
            if quality > best_quality:
                best_quality = quality
                best_result = result
        
        # Use best retriever's results
        if best_result:
            return best_result.documents[:target_size], best_result.scores[:target_size]
        
        # Fallback to first retriever
        if results:
            return results[0].documents[:target_size], results[0].scores[:target_size]
        
        return [], []
    
    async def _hierarchical_cascade(self, results: List[RetrievalResult], 
                                  target_size: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Use hierarchical cascade - start with simple, add complexity"""
        
        fused_documents = []
        fused_scores = []
        
        # Priority order for retrievers
        priority_order = [
            RetrieverType.SPARSE_KEYWORD,    # Start with precise keyword matches
            RetrieverType.DENSE_VECTOR,      # Add semantic similarity
            RetrieverType.RECENT_TEMPORAL,   # Add temporal relevance
            RetrieverType.GRAPH_TRAVERSAL    # Add conceptual connections
        ]
        
        # Add results in priority order
        for retriever_type in priority_order:
            # Find result from this retriever type
            matching_results = [r for r in results if r.retriever_type == retriever_type]
            
            if matching_results:
                result = matching_results[0]
                
                # Add documents not already included
                for doc, score in zip(result.documents, result.scores):
                    doc_id = doc.get('id', str(hash(doc['content'])))
                    
                    # Check if already included
                    already_included = any(
                        existing_doc.get('id', str(hash(existing_doc['content']))) == doc_id 
                        for existing_doc in fused_documents
                    )
                    
                    if not already_included and len(fused_documents) < target_size:
                        fused_documents.append(doc)
                        fused_scores.append(score)
        
        return fused_documents, fused_scores
    
    def _get_retriever_weight(self, retriever_type: RetrieverType) -> float:
        """Get weight for different retriever types"""
        
        weights = {
            RetrieverType.DENSE_VECTOR: 1.0,
            RetrieverType.SPARSE_KEYWORD: 0.9,
            RetrieverType.GRAPH_TRAVERSAL: 0.8,
            RetrieverType.RECENT_TEMPORAL: 0.7,
            RetrieverType.HIERARCHICAL: 0.8,
            RetrieverType.STRUCTURED_QUERY: 0.9,
        }
        
        return weights.get(retriever_type, 0.5)
    
    def _calculate_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate diversity of fused results"""
        
        if len(documents) < 2:
            return 0.0
        
        # Simple diversity based on different sources
        sources = set(doc.get('source', 'unknown') for doc in documents)
        
        # Normalize by number of documents
        diversity = len(sources) / len(documents)
        
        return min(diversity, 1.0)
    
    def _calculate_consensus(self, results: List[RetrievalResult], 
                           fused_documents: List[Dict[str, Any]]) -> float:
        """Calculate consensus among retrievers"""
        
        if not results or not fused_documents:
            return 0.0
        
        # Count how many retrievers found each document
        doc_retriever_counts = defaultdict(int)
        
        for doc in fused_documents:
            doc_id = doc.get('id', str(hash(doc['content'])))
            
            for result in results:
                for result_doc in result.documents:
                    result_doc_id = result_doc.get('id', str(hash(result_doc['content'])))
                    if result_doc_id == doc_id:
                        doc_retriever_counts[doc_id] += 1
                        break
        
        # Calculate average consensus
        total_consensus = sum(doc_retriever_counts.values())
        max_possible_consensus = len(fused_documents) * len(results)
        
        consensus = total_consensus / max(max_possible_consensus, 1)
        
        return min(consensus, 1.0)
    
    def _calculate_fusion_confidence(self, results: List[RetrievalResult], 
                                   fused_scores: List[float]) -> float:
        """Calculate confidence in fusion results"""
        
        if not results or not fused_scores:
            return 0.0
        
        # Average confidence of individual retrievers
        avg_retriever_confidence = sum(r.confidence_score for r in results) / len(results)
        
        # Quality of fused scores
        avg_fused_score = sum(fused_scores) / max(len(fused_scores), 1)
        
        # Combine metrics
        fusion_confidence = (avg_retriever_confidence + avg_fused_score) / 2
        
        return min(fusion_confidence, 1.0)

class HybridRAGSystem:
    """
    Complete Hybrid RAG System combining multiple retrieval approaches
    
    EXAMPLE USAGE:
    =============
    # Create hybrid RAG system
    rag = HybridRAGSystem()
    await rag.initialize()
    
    # Process query with automatic strategy selection
    query = "What are the latest developments in artificial intelligence for healthcare?"
    
    result = await rag.hybrid_retrieve(query)
    
    print(f"Fusion strategy: {result.fusion_strategy.value}")
    print(f"Retrievers used: {', '.join(result.retrievers_used)}")
    print(f"Total documents: {len(result.fused_documents)}")
    print(f"Overall quality: {result.overall_quality:.2f}")
    """
    
    def __init__(self):
        # Initialize retrievers
        self.retrievers: Dict[RetrieverType, BaseRetriever] = {}
        
        # Initialize components
        self.question_analyzer = QuestionAnalyzer()
        self.fusion_engine = ResultFusionEngine()
        
        # System statistics
        self.system_stats = {
            'total_queries': 0,
            'fusion_strategies_used': defaultdict(int),
            'retriever_usage': defaultdict(int),
            'average_quality': 0.0,
            'average_retrieval_time': 0.0
        }
        
        self.logger = logging.getLogger("HybridRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize hybrid RAG system"""
        
        # Create retrievers
        self.retrievers[RetrieverType.DENSE_VECTOR] = DenseVectorRetriever()
        self.retrievers[RetrieverType.SPARSE_KEYWORD] = SparseKeywordRetriever()
        self.retrievers[RetrieverType.GRAPH_TRAVERSAL] = GraphTraversalRetriever()
        self.retrievers[RetrieverType.RECENT_TEMPORAL] = RecentTemporalRetriever()
        
        self.logger.info(f"Hybrid RAG system initialized with {len(self.retrievers)} retrievers")
    
    async def hybrid_retrieve(self, query: str, top_k: int = 10) -> HybridRetrievalResult:
        """Perform hybrid retrieval with automatic strategy selection"""
        
        start_time = time.time()
        self.system_stats['total_queries'] += 1
        
        try:
            # Analyze query to determine optimal strategy
            analysis = self.question_analyzer.analyze_question(query)
            
            fusion_strategy = analysis['fusion_strategy']
            optimal_retrievers = analysis['optimal_retrievers']
            
            self.logger.info(f"Query analysis: type={analysis['question_type'].value}, "
                           f"complexity={analysis['complexity_score']:.2f}, "
                           f"strategy={fusion_strategy.value}")
            
            # Execute retrieval with selected retrievers
            retrieval_results = []
            
            for retriever_type in optimal_retrievers:
                if retriever_type in self.retrievers:
                    retriever = self.retrievers[retriever_type]
                    
                    try:
                        result = await retriever.retrieve(query, top_k)
                        retrieval_results.append(result)
                        
                        # Update usage statistics
                        self.system_stats['retriever_usage'][retriever_type.value] += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Retriever {retriever_type.value} failed: {e}")
            
            # Fuse results
            if retrieval_results:
                hybrid_result = await self.fusion_engine.fuse_results(
                    retrieval_results, 
                    fusion_strategy, 
                    top_k
                )
                
                # Update statistics
                self.system_stats['fusion_strategies_used'][fusion_strategy.value] += 1
                self._update_performance_stats(hybrid_result, time.time() - start_time)
                
                self.logger.info(f"Hybrid retrieval completed: {len(hybrid_result.fused_documents)} docs, "
                               f"quality={hybrid_result.overall_quality:.2f}")
                
                return hybrid_result
            
            else:
                # No successful retrievals
                return HybridRetrievalResult(
                    fusion_strategy=fusion_strategy,
                    individual_results=[],
                    fused_documents=[],
                    fused_scores=[],
                    total_retrieval_time=time.time() - start_time
                )
                
        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {e}")
            return HybridRetrievalResult(
                fusion_strategy=FusionStrategy.PARALLEL_FUSION,
                individual_results=[],
                fused_documents=[],
                fused_scores=[],
                total_retrieval_time=time.time() - start_time
            )
    
    async def retrieve_with_strategy(self, query: str, 
                                   fusion_strategy: FusionStrategy,
                                   retrievers: List[RetrieverType],
                                   top_k: int = 10) -> HybridRetrievalResult:
        """Retrieve with explicitly specified strategy and retrievers"""
        
        start_time = time.time()
        
        # Execute specified retrievers
        retrieval_results = []
        
        for retriever_type in retrievers:
            if retriever_type in self.retrievers:
                retriever = self.retrievers[retriever_type]
                result = await retriever.retrieve(query, top_k)
                retrieval_results.append(result)
        
        # Fuse with specified strategy
        hybrid_result = await self.fusion_engine.fuse_results(
            retrieval_results, 
            fusion_strategy, 
            top_k
        )
        
        hybrid_result.total_retrieval_time = time.time() - start_time
        
        return hybrid_result
    
    def _update_performance_stats(self, hybrid_result: HybridRetrievalResult, 
                                total_time: float) -> None:
        """Update system performance statistics"""
        
        # Update average quality
        current_avg_quality = self.system_stats['average_quality']
        query_count = self.system_stats['total_queries']
        
        self.system_stats['average_quality'] = (
            (current_avg_quality * (query_count - 1) + hybrid_result.overall_quality) / query_count
        )
        
        # Update average retrieval time
        current_avg_time = self.system_stats['average_retrieval_time']
        
        self.system_stats['average_retrieval_time'] = (
            (current_avg_time * (query_count - 1) + total_time) / query_count
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        retriever_stats = {}
        for retriever_type, retriever in self.retrievers.items():
            retriever_stats[retriever_type.value] = retriever.get_performance_stats()
        
        return {
            'system_stats': dict(self.system_stats),
            'retriever_performance': retriever_stats,
            'available_strategies': [s.value for s in FusionStrategy],
            'available_retrievers': [r.value for r in RetrieverType if r in self.retrievers]
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_question_analysis():
    """Demo: Question analysis for strategy selection"""
    print("\nDEMO 1: INTELLIGENT QUESTION ANALYSIS")
    print("=" * 50)
    
    analyzer = QuestionAnalyzer()
    
    test_questions = [
        "What is machine learning?",
        "Compare the advantages of Python versus Java for enterprise development",
        "What are the latest trends in artificial intelligence for healthcare?",
        "How do I implement a REST API using FastAPI?",
        "Analyze the strategic implications of cloud computing adoption considering security concerns and cost factors",
        "Who founded Google and when was it established?"
    ]
    
    print("Analyzing different question types:")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        analysis = analyzer.analyze_question(question)
        
        print(f"Type: {analysis['question_type'].value}")
        print(f"Complexity: {analysis['complexity_score']:.2f}")
        print(f"Difficulty: {analysis['estimated_difficulty']}")
        print(f"Strategy: {analysis['fusion_strategy'].value}")
        print(f"Optimal retrievers: {[r.value for r in analysis['optimal_retrievers']]}")
        
        if analysis['content_requirements']:
            content_types = [ct.value for ct in analysis['content_requirements']]
            print(f"Content needs: {', '.join(content_types)}")

async def demo_individual_retrievers():
    """Demo: Individual retriever performance"""
    print("\nDEMO 2: INDIVIDUAL RETRIEVER PERFORMANCE")
    print("=" * 50)
    
    # Create retrievers
    retrievers = [
        DenseVectorRetriever(),
        SparseKeywordRetriever(), 
        GraphTraversalRetriever(),
        RecentTemporalRetriever()
    ]
    
    test_query = "artificial intelligence applications in healthcare technology"
    
    print(f"Query: {test_query}")
    print(f"\nTesting {len(retrievers)} different retrievers:")
    
    for retriever in retrievers:
        print(f"\n--- {retriever.retriever_type.value.replace('_', ' ').title()} ---")
        
        result = await retriever.retrieve(test_query, top_k=5)
        
        print(f"Documents found: {len(result.documents)}")
        print(f"Query time: {result.query_time:.3f}s")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Precision estimate: {result.precision_estimate:.2f}")
        print(f"Recall estimate: {result.recall_estimate:.2f}")
        
        if result.documents:
            print("Top result:")
            top_doc = result.documents[0]
            print(f"  Title: {top_doc['title']}")
            print(f"  Score: {result.scores[0]:.3f}")
            print(f"  Content: {top_doc['content'][:100]}...")

async def demo_fusion_strategies():
    """Demo: Different fusion strategies"""
    print("\nDEMO 3: FUSION STRATEGIES COMPARISON")
    print("=" * 50)
    
    # Create retrievers and get results
    retrievers = [
        DenseVectorRetriever(),
        SparseKeywordRetriever(),
        GraphTraversalRetriever()
    ]
    
    query = "machine learning algorithms for business analysis"
    
    # Get individual results
    individual_results = []
    for retriever in retrievers:
        result = await retriever.retrieve(query, top_k=8)
        individual_results.append(result)
    
    print(f"Query: {query}")
    print(f"Individual results: {[len(r.documents) for r in individual_results]} documents")
    
    # Test different fusion strategies
    fusion_engine = ResultFusionEngine()
    strategies = [
        FusionStrategy.PARALLEL_FUSION,
        FusionStrategy.ENSEMBLE_VOTING,
        FusionStrategy.ADAPTIVE_SELECTION,
        FusionStrategy.HIERARCHICAL_CASCADE
    ]
    
    print(f"\nTesting {len(strategies)} fusion strategies:")
    
    for strategy in strategies:
        print(f"\n--- {strategy.value.replace('_', ' ').title()} ---")
        
        hybrid_result = await fusion_engine.fuse_results(
            individual_results, 
            strategy, 
            target_size=5
        )
        
        print(f"Fused documents: {len(hybrid_result.fused_documents)}")
        print(f"Fusion time: {hybrid_result.total_retrieval_time:.3f}s")
        print(f"Diversity score: {hybrid_result.diversity_score:.2f}")
        print(f"Consensus score: {hybrid_result.consensus_score:.2f}")
        print(f"Overall quality: {hybrid_result.overall_quality:.2f}")
        
        if hybrid_result.fused_documents:
            print(f"Top result score: {hybrid_result.fused_scores[0]:.3f}")

async def demo_hybrid_system():
    """Demo: Complete hybrid RAG system"""
    print("\nDEMO 4: COMPLETE HYBRID RAG SYSTEM")
    print("=" * 50)
    
    # Initialize hybrid RAG system
    rag_system = HybridRAGSystem()
    await rag_system.initialize()
    
    # Test different types of queries
    test_queries = [
        "What is deep learning?",  # Simple factual
        "Latest developments in renewable energy technology",  # Temporal
        "Compare cloud computing platforms for enterprise use",  # Comparative
        "How does machine learning relate to artificial intelligence and data science?",  # Graph/conceptual
        "Analyze the impact of AI on healthcare considering ethical implications and implementation challenges"  # Complex analytical
    ]
    
    print("Testing hybrid RAG with diverse queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}")
        print(f"{'='*60}")
        print(f"Query: {query}")
        
        result = await rag_system.hybrid_retrieve(query, top_k=5)
        
        print(f"\nHybrid Retrieval Results:")
        print(f"  Fusion strategy: {result.fusion_strategy.value}")
        print(f"  Retrievers used: {', '.join(result.retrievers_used)}")
        print(f"  Documents found: {len(result.fused_documents)}")
        print(f"  Retrieval time: {result.total_retrieval_time:.3f}s")
        print(f"  Diversity score: {result.diversity_score:.2f}")
        print(f"  Consensus score: {result.consensus_score:.2f}")
        print(f"  Overall quality: {result.overall_quality:.2f}")
        
        if result.fused_documents:
            print(f"\nTop Results:")
            for j, (doc, score) in enumerate(zip(result.fused_documents[:3], result.fused_scores[:3]), 1):
                print(f"  {j}. {doc['title']} (score: {score:.3f})")
                print(f"     {doc['content'][:80]}...")

async def demo_performance_analytics():
    """Demo: System performance analytics"""
    print("\nDEMO 5: PERFORMANCE ANALYTICS")
    print("=" * 50)
    
    rag_system = HybridRAGSystem()
    await rag_system.initialize()
    
    # Process multiple queries for analytics
    analytics_queries = [
        "artificial intelligence machine learning",
        "business strategy market analysis",
        "latest technology trends innovation",
        "healthcare medical research developments",
        "financial markets investment opportunities",
        "software development programming best practices",
        "data science analytics techniques",
        "cloud computing infrastructure solutions"
    ]
    
    print("Processing multiple queries for performance analysis...")
    
    results = []
    for query in analytics_queries:
        result = await rag_system.hybrid_retrieve(query, top_k=5)
        results.append(result)
        print(f"  ✓ Processed: {query[:40]}...")
    
    # Get comprehensive statistics
    stats = rag_system.get_system_statistics()
    
    print(f"\nHYBRID RAG SYSTEM ANALYTICS")
    print("=" * 40)
    
    print(f"\nSystem Performance:")
    system_stats = stats['system_stats']
    print(f"  Total queries: {system_stats['total_queries']}")
    print(f"  Average quality: {system_stats['average_quality']:.2f}")
    print(f"  Average retrieval time: {system_stats['average_retrieval_time']:.3f}s")
    
    print(f"\nFusion Strategy Usage:")
    for strategy, count in system_stats['fusion_strategies_used'].items():
        percentage = (count / system_stats['total_queries']) * 100
        print(f"  {strategy}: {count} ({percentage:.1f}%)")
    
    print(f"\nRetriever Usage:")
    for retriever, count in system_stats['retriever_usage'].items():
        print(f"  {retriever}: {count} queries")
    
    print(f"\nIndividual Retriever Performance:")
    for retriever_type, perf_stats in stats['retriever_performance'].items():
        print(f"  {retriever_type}:")
        print(f"    Average time: {perf_stats['average_query_time']:.3f}s")
        print(f"    Success rate: {perf_stats['success_rate']:.1%}")
        print(f"    Average precision: {perf_stats['average_precision']:.2f}")
    
    print(f"\nResult Quality Analysis:")
    successful_results = [r for r in results if r.fused_documents]
    if successful_results:
        avg_diversity = sum(r.diversity_score for r in successful_results) / len(successful_results)
        avg_consensus = sum(r.consensus_score for r in successful_results) / len(successful_results)
        avg_quality = sum(r.overall_quality for r in successful_results) / len(successful_results)
        
        print(f"  Average diversity: {avg_diversity:.2f}")
        print(f"  Average consensus: {avg_consensus:.2f}")
        print(f"  Average overall quality: {avg_quality:.2f}")
    
    print(f"\nSystem Capabilities:")
    print(f"  Available strategies: {len(stats['available_strategies'])}")
    print(f"  Available retrievers: {len(stats['available_retrievers'])}")
    print(f"  ✓ Automatic strategy selection")
    print(f"  ✓ Multi-retriever fusion")
    print(f"  ✓ Performance optimization")
    print(f"  ✓ Quality assurance")

async def main():
    """
    Demonstrate Hybrid RAG Architectures for optimal retrieval performance
    
    WHAT YOU'LL LEARN:
    ================
    1. How to analyze questions to select optimal retrieval strategies
    2. How to implement multiple specialized retrievers
    3. How to fuse results from different retrievers effectively
    4. How to build adaptive systems that optimize for different scenarios
    5. How to create production-ready RAG systems with reliability
    
    REAL WORLD APPLICATIONS:
    =======================
    - Enterprise search systems serving diverse content types
    - E-commerce product search with multiple ranking factors
    - Academic research platforms with varied content sources
    - Customer support systems with different question types
    - News and media platforms with temporal and topical content
    - Business intelligence systems requiring multi-modal analysis
    """
    
    print("HYBRID RAG ARCHITECTURES DEMONSTRATION")
    print("Building production-ready RAG systems with optimal performance!")
    
    await demo_question_analysis()
    await demo_individual_retrievers()
    await demo_fusion_strategies()
    await demo_hybrid_system()
    await demo_performance_analytics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Question analysis enables intelligent strategy selection")
    print("✓ Different retrievers excel at different types of content")
    print("✓ Fusion strategies combine strengths while mitigating weaknesses")
    print("✓ Hybrid systems achieve superior overall performance")
    print("✓ Analytics enable continuous optimization and improvement")
    print("✓ Production systems require robust fallback mechanisms")
    print("\nTHE POWER OF HYBRID RAG:")
    print("- Delivers best-of-all-worlds performance across scenarios")
    print("- Provides consistent quality through intelligent adaptation")
    print("- Enables enterprise-scale deployment with reliability")
    print("- Powers next-generation search and knowledge systems")

if __name__ == "__main__":
    asyncio.run(main())
