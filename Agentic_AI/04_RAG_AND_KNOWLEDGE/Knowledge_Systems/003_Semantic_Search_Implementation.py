#!/usr/bin/env python3
"""
Semantic Search Implementation: Understanding Meaning Beyond Keywords
====================================================================

WHAT IS THE PROBLEM?
==================
Traditional keyword-based search fails to understand meaning:
- Exact keyword matching misses semantically similar content
- Users struggle to find relevant information using different terminology
- Search results lack understanding of context and intent
- No understanding of synonyms, concepts, or relationships
- Cannot handle complex queries requiring reasoning
- Poor user experience due to irrelevant results

Example: Medical Information Search Failure
KEYWORD SEARCH (Traditional):
- Query: "heart attack symptoms"
- Misses documents mentioning "myocardial infarction"
- Ignores "cardiac arrest" related content
- Cannot understand "chest pain" as related symptom
- Fails to connect "shortness of breath" to heart conditions
- Result: Incomplete, potentially dangerous medical information

REAL WORLD EXAMPLE:
=================
How does Google's semantic search work?

GOOGLE'S SEMANTIC UNDERSTANDING:
1. ENTITY RECOGNITION: Identify people, places, things in queries
2. INTENT UNDERSTANDING: Determine what users are really looking for
3. CONTEXT ANALYSIS: Consider user location, history, and preferences
4. KNOWLEDGE GRAPH: Use structured knowledge to understand relationships
5. VECTOR EMBEDDINGS: Represent words and concepts in semantic space
6. NEURAL MATCHING: Match query intent with document meaning
7. RANKING SIGNALS: Combine semantic relevance with other factors

BENEFITS OF SEMANTIC SEARCH:
- Understanding user intent rather than just keywords
- Finding relevant content regardless of exact word matches
- Handling synonyms, abbreviations, and related concepts
- Supporting natural language queries and conversations
- Providing contextually relevant results
- Enabling discovery of conceptually related information

THE SEMANTIC ADVANTAGE:
=====================
KEYWORD SEARCH: Words → Exact Match → Limited Results
SEMANTIC SEARCH: Meaning → Understanding → Intelligent Results

SEMANTIC SEARCH COMPONENTS:
=========================
1. TEXT EMBEDDINGS: Convert text to dense vector representations
2. SIMILARITY COMPUTATION: Measure semantic similarity between vectors
3. QUERY UNDERSTANDING: Parse and interpret user intent
4. CONTEXT AWARENESS: Consider user and situational context
5. ENTITY LINKING: Connect mentions to knowledge base entities
6. RANKING ALGORITHMS: Score results by semantic relevance
7. RESULT EXPLANATION: Provide reasoning for search results

WHY THIS IS REVOLUTIONARY:
========================
- Enables intuitive, natural language search experiences
- Bridges the vocabulary gap between users and content
- Powers intelligent question answering systems
- Critical for AI assistants and conversational interfaces
- Enables discovery of insights hidden in text
- Provides foundation for truly intelligent information retrieval
"""

import asyncio
import time
import json
import uuid
import re
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class QueryType(Enum):
    """Types of search queries"""
    KEYWORD = "keyword"                 # Simple keyword search
    PHRASE = "phrase"                   # Exact phrase search
    QUESTION = "question"               # Natural language question
    SEMANTIC = "semantic"               # Meaning-based search
    CONTEXTUAL = "contextual"           # Context-aware search
    CONVERSATIONAL = "conversational"  # Multi-turn conversation

class EntityType(Enum):
    """Types of entities in text"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    PRODUCT = "product"
    EVENT = "event"
    DATE = "date"
    NUMBER = "number"

class SearchIntent(Enum):
    """User search intents"""
    INFORMATIONAL = "informational"     # Seeking information
    NAVIGATIONAL = "navigational"       # Finding specific page/resource
    TRANSACTIONAL = "transactional"     # Wanting to perform action
    COMPARISON = "comparison"            # Comparing options
    DEFINITIONAL = "definitional"       # Seeking definitions
    HOW_TO = "how_to"                   # Seeking instructions

@dataclass
class Document:
    """Represents a document in the search index"""
    
    id: str
    title: str
    content: str
    
    # Metadata
    url: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    category: str = ""
    
    # Processed content
    embedding: Optional[np.ndarray] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    summary: str = ""
    
    # Search metadata
    indexed_at: datetime = field(default_factory=datetime.now)
    index_version: str = "1.0"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def get_text(self) -> str:
        """Get full text content for processing"""
        return f"{self.title} {self.content}".strip()
    
    def add_entity(self, entity_type: str, entity_text: str, 
                  confidence: float = 1.0, metadata: Dict[str, Any] = None) -> None:
        """Add entity to document"""
        entity = {
            'type': entity_type,
            'text': entity_text,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        self.entities.append(entity)
    
    def add_concept(self, concept: str) -> None:
        """Add concept to document"""
        if concept not in self.concepts:
            self.concepts.append(concept)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        doc_dict = {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'category': self.category,
            'entities': self.entities,
            'concepts': self.concepts,
            'summary': self.summary,
            'indexed_at': self.indexed_at.isoformat(),
            'index_version': self.index_version
        }
        
        if self.embedding is not None:
            doc_dict['embedding'] = self.embedding.tolist()
        
        return doc_dict

@dataclass
class Query:
    """Represents a search query"""
    
    id: str
    text: str
    query_type: QueryType
    
    # Query analysis
    intent: Optional[SearchIntent] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    
    # Processing
    embedding: Optional[np.ndarray] = None
    expanded_terms: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    user_id: str = ""
    session_id: str = ""
    previous_queries: List[str] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    language: str = "en"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class SearchResult:
    """Represents a search result"""
    
    document: Document
    score: float
    rank: int
    
    # Explanation
    matched_terms: List[str] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)
    matched_concepts: List[str] = field(default_factory=list)
    explanation: str = ""
    
    # Metadata
    retrieval_method: str = "semantic"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'rank': self.rank,
            'matched_terms': self.matched_terms,
            'matched_entities': self.matched_entities,
            'matched_concepts': self.matched_concepts,
            'explanation': self.explanation,
            'retrieval_method': self.retrieval_method,
            'confidence': self.confidence
        }

class TextEmbedder:
    """Creates dense vector embeddings for text"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
        # Simple word-based embeddings (in practice, use pre-trained models)
        self.vocabulary = {}
        self.word_vectors = {}
        self.idf_scores = {}
        
        self.logger = logging.getLogger("TextEmbedder")
    
    async def build_vocabulary(self, documents: List[Document]) -> None:
        """Build vocabulary and IDF scores from documents"""
        
        self.logger.info("Building vocabulary from documents")
        
        # Collect all words and document frequencies
        word_doc_count = defaultdict(int)
        total_docs = len(documents)
        
        for doc in documents:
            words = set(self._tokenize(doc.get_text().lower()))
            for word in words:
                word_doc_count[word] += 1
        
        # Build vocabulary and calculate IDF scores
        self.vocabulary = {word: idx for idx, word in enumerate(word_doc_count.keys())}
        
        for word, doc_count in word_doc_count.items():
            self.idf_scores[word] = math.log(total_docs / (doc_count + 1))
        
        # Initialize random word vectors (in practice, use pre-trained embeddings)
        for word in self.vocabulary:
            self.word_vectors[word] = np.random.randn(self.embedding_dim)
            # Normalize vector
            self.word_vectors[word] = self.word_vectors[word] / np.linalg.norm(self.word_vectors[word])
        
        self.logger.info(f"Built vocabulary with {len(self.vocabulary)} words")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text"""
        
        words = self._tokenize(text.lower())
        
        if not words:
            return np.zeros(self.embedding_dim)
        
        # TF-IDF weighted average of word vectors
        word_weights = {}
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / len(words)
                idf = self.idf_scores.get(word, 0)
                word_weights[word] = tf * idf
        
        # Weighted average of word vectors
        if not word_weights:
            return np.zeros(self.embedding_dim)
        
        embedding = np.zeros(self.embedding_dim)
        total_weight = 0
        
        for word, weight in word_weights.items():
            if word in self.word_vectors:
                embedding += weight * self.word_vectors[word]
                total_weight += weight
        
        if total_weight > 0:
            embedding = embedding / total_weight
        
        # Normalize final embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter out short words and numbers
        words = [word for word in words if len(word) > 2 and not word.isdigit()]
        
        return words
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        
        if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

class EntityExtractor:
    """Extracts entities from text"""
    
    def __init__(self):
        # Simple pattern-based entity extraction
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\bDr\. [A-Z][a-z]+\b',         # Dr. Name
                r'\bProf\. [A-Z][a-z]+\b'        # Prof. Name
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|University|Institute|Company)\b',
                r'\bUniversity of [A-Z][a-z]+\b'
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+, [A-Z][A-Z]\b',  # City, State
                r'\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard)\b'
            ],
            EntityType.CONCEPT: [
                r'\b(machine learning|artificial intelligence|data science|deep learning)\b',
                r'\b[a-z]+ (algorithm|method|technique|approach)\b'
            ],
            EntityType.DATE: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
            ],
            EntityType.NUMBER: [
                r'\b\d+(\.\d+)?%\b',  # Percentages
                r'\$\d+(\.\d+)?\b'    # Money amounts
            ]
        }
        
        self.logger = logging.getLogger("EntityExtractor")
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group().strip()
                    
                    entity = {
                        'type': entity_type.value,
                        'text': entity_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8  # Simple confidence score
                    }
                    
                    entities.append(entity)
        
        # Remove duplicate entities
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            entity_key = (entity['type'], entity['text'].lower())
            if entity_key not in seen_texts:
                seen_texts.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities

class ConceptExtractor:
    """Extracts concepts and topics from text"""
    
    def __init__(self):
        # Concept keywords and patterns
        self.concept_keywords = {
            'artificial_intelligence': ['ai', 'artificial intelligence', 'machine intelligence', 'cognitive computing'],
            'machine_learning': ['machine learning', 'ml', 'statistical learning', 'predictive modeling'],
            'deep_learning': ['deep learning', 'neural networks', 'deep neural networks', 'dl'],
            'data_science': ['data science', 'data analysis', 'data mining', 'analytics'],
            'natural_language_processing': ['nlp', 'natural language processing', 'text processing', 'language understanding'],
            'computer_vision': ['computer vision', 'image processing', 'image recognition', 'visual perception'],
            'robotics': ['robotics', 'autonomous systems', 'robotic systems', 'automation'],
            'cloud_computing': ['cloud computing', 'cloud services', 'aws', 'azure', 'distributed computing'],
            'cybersecurity': ['cybersecurity', 'information security', 'network security', 'cyber defense'],
            'blockchain': ['blockchain', 'cryptocurrency', 'distributed ledger', 'bitcoin']
        }
        
        self.logger = logging.getLogger("ConceptExtractor")
    
    async def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text"""
        
        text_lower = text.lower()
        concepts = []
        
        for concept, keywords in self.concept_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if concept not in concepts:
                        concepts.append(concept)
                    break
        
        return concepts

class QueryAnalyzer:
    """Analyzes and understands search queries"""
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            SearchIntent.DEFINITIONAL: [
                r'\bwhat is\b', r'\bdefine\b', r'\bdefinition of\b', r'\bmeaning of\b'
            ],
            SearchIntent.HOW_TO: [
                r'\bhow to\b', r'\bhow do\b', r'\bhow can\b', r'\bsteps to\b'
            ],
            SearchIntent.COMPARISON: [
                r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bdifference between\b', r'\bbetter than\b'
            ],
            SearchIntent.INFORMATIONAL: [
                r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwhich\b', r'\btell me about\b'
            ]
        }
        
        # Query type detection
        self.query_type_patterns = {
            QueryType.QUESTION: [
                r'\?$', r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b'
            ],
            QueryType.PHRASE: [
                r'"[^"]+"'  # Quoted phrases
            ]
        }
        
        self.entity_extractor = EntityExtractor()
        self.concept_extractor = ConceptExtractor()
        
        self.logger = logging.getLogger("QueryAnalyzer")
    
    async def analyze_query(self, query: Query) -> Query:
        """Analyze query and extract intent, entities, concepts"""
        
        # Detect query type
        if query.query_type == QueryType.KEYWORD:  # Default type
            query.query_type = await self._detect_query_type(query.text)
        
        # Detect intent
        query.intent = await self._detect_intent(query.text)
        
        # Extract entities
        entities = await self.entity_extractor.extract_entities(query.text)
        query.entities = entities
        
        # Extract concepts
        concepts = await self.concept_extractor.extract_concepts(query.text)
        query.concepts = concepts
        
        # Expand query terms
        query.expanded_terms = await self._expand_query_terms(query.text, concepts)
        
        self.logger.debug(f"Analyzed query: type={query.query_type.value}, intent={query.intent.value if query.intent else 'unknown'}")
        
        return query
    
    async def _detect_query_type(self, text: str) -> QueryType:
        """Detect query type from text"""
        
        text_lower = text.lower()
        
        for query_type, patterns in self.query_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return query_type
        
        # Check if it's a conversational query
        if len(text.split()) > 10 and any(word in text_lower for word in ['please', 'can you', 'could you', 'i need', 'help me']):
            return QueryType.CONVERSATIONAL
        
        # Default to semantic for complex queries
        if len(text.split()) > 5:
            return QueryType.SEMANTIC
        
        return QueryType.KEYWORD
    
    async def _detect_intent(self, text: str) -> Optional[SearchIntent]:
        """Detect user intent from query text"""
        
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        # Default intent based on query characteristics
        if '?' in text:
            return SearchIntent.INFORMATIONAL
        elif any(word in text_lower for word in ['buy', 'purchase', 'order', 'download']):
            return SearchIntent.TRANSACTIONAL
        elif any(word in text_lower for word in ['login', 'sign in', 'website', 'homepage']):
            return SearchIntent.NAVIGATIONAL
        
        return SearchIntent.INFORMATIONAL
    
    async def _expand_query_terms(self, text: str, concepts: List[str]) -> List[str]:
        """Expand query with related terms"""
        
        expanded_terms = []
        
        # Add synonyms for detected concepts
        concept_synonyms = {
            'artificial_intelligence': ['ai', 'machine intelligence', 'cognitive computing'],
            'machine_learning': ['ml', 'statistical learning', 'predictive modeling'],
            'deep_learning': ['neural networks', 'deep neural networks'],
            'data_science': ['data analysis', 'analytics', 'data mining'],
            'natural_language_processing': ['nlp', 'text processing'],
            'computer_vision': ['image processing', 'image recognition']
        }
        
        for concept in concepts:
            if concept in concept_synonyms:
                expanded_terms.extend(concept_synonyms[concept])
        
        # Add related terms based on text analysis
        text_lower = text.lower()
        
        if 'python' in text_lower:
            expanded_terms.extend(['programming', 'coding', 'development'])
        elif 'database' in text_lower:
            expanded_terms.extend(['sql', 'data storage', 'data management'])
        elif 'web' in text_lower:
            expanded_terms.extend(['website', 'internet', 'html', 'css', 'javascript'])
        
        return list(set(expanded_terms))

class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self, embedding_dim: int = 384):
        # Core components
        self.embedder = TextEmbedder(embedding_dim)
        self.query_analyzer = QueryAnalyzer()
        
        # Document index
        self.documents: Dict[str, Document] = {}
        self.document_embeddings: Dict[str, np.ndarray] = {}
        
        # Search configuration
        self.default_results_limit = 10
        self.similarity_threshold = 0.1
        
        # Statistics
        self.stats = {
            'documents_indexed': 0,
            'queries_processed': 0,
            'total_search_time': 0.0,
            'average_search_time': 0.0
        }
        
        self.logger = logging.getLogger("SemanticSearchEngine")
    
    async def initialize(self) -> None:
        """Initialize search engine"""
        self.logger.info("Semantic search engine initialized")
    
    async def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Index documents for semantic search"""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Indexing {len(documents)} documents")
            
            # Build vocabulary for embeddings
            await self.embedder.build_vocabulary(documents)
            
            # Process each document
            indexed_count = 0
            
            for doc in documents:
                await self._process_document(doc)
                self.documents[doc.id] = doc
                indexed_count += 1
                
                if indexed_count % 100 == 0:
                    self.logger.debug(f"Indexed {indexed_count}/{len(documents)} documents")
            
            indexing_time = time.time() - start_time
            self.stats['documents_indexed'] = len(self.documents)
            
            result = {
                'success': True,
                'documents_indexed': indexed_count,
                'indexing_time': indexing_time,
                'total_documents': len(self.documents)
            }
            
            self.logger.info(f"Indexing completed: {indexed_count} documents, {indexing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Document indexing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'indexing_time': time.time() - start_time
            }
    
    async def search(self, query_text: str, limit: int = None,
                    user_id: str = "", session_id: str = "",
                    filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform semantic search"""
        
        start_time = time.time()
        self.stats['queries_processed'] += 1
        
        if limit is None:
            limit = self.default_results_limit
        
        try:
            # Create and analyze query
            query = Query(
                id="",
                text=query_text,
                query_type=QueryType.KEYWORD,
                user_id=user_id,
                session_id=session_id,
                filters=filters or {}
            )
            
            query = await self.query_analyzer.analyze_query(query)
            
            # Generate query embedding
            query.embedding = await self.embedder.embed_text(query.text)
            
            # Perform search
            results = await self._execute_search(query, limit)
            
            search_time = time.time() - start_time
            self.stats['total_search_time'] += search_time
            self.stats['average_search_time'] = self.stats['total_search_time'] / self.stats['queries_processed']
            
            search_result = {
                'success': True,
                'query': {
                    'id': query.id,
                    'text': query.text,
                    'type': query.query_type.value,
                    'intent': query.intent.value if query.intent else None,
                    'entities': query.entities,
                    'concepts': query.concepts,
                    'expanded_terms': query.expanded_terms
                },
                'results': [result.to_dict() for result in results],
                'total_results': len(results),
                'search_time': search_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Search completed: '{query_text}' -> {len(results)} results, {search_time:.3f}s")
            
            return search_result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'search_time': time.time() - start_time
            }
    
    async def _process_document(self, document: Document) -> None:
        """Process document for indexing"""
        
        # Generate embedding
        document.embedding = await self.embedder.embed_text(document.get_text())
        self.document_embeddings[document.id] = document.embedding
        
        # Extract entities
        entities = await self.query_analyzer.entity_extractor.extract_entities(document.get_text())
        document.entities = entities
        
        # Extract concepts
        concepts = await self.query_analyzer.concept_extractor.extract_concepts(document.get_text())
        document.concepts = concepts
        
        # Generate summary (simple truncation for demo)
        document.summary = document.content[:200] + "..." if len(document.content) > 200 else document.content
        
        # Update metadata
        document.indexed_at = datetime.now()
    
    async def _execute_search(self, query: Query, limit: int) -> List[SearchResult]:
        """Execute semantic search"""
        
        if query.embedding is None:
            return []
        
        # Calculate similarities with all documents
        similarities = []
        
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = self.embedder.compute_similarity(query.embedding, doc_embedding)
            
            if similarity >= self.similarity_threshold:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create search results
        results = []
        
        for rank, (doc_id, similarity) in enumerate(similarities[:limit], 1):
            document = self.documents.get(doc_id)
            if document:
                result = await self._create_search_result(query, document, similarity, rank)
                results.append(result)
        
        return results
    
    async def _create_search_result(self, query: Query, document: Document,
                                  similarity: float, rank: int) -> SearchResult:
        """Create search result with explanation"""
        
        # Find matched terms
        query_words = set(query.text.lower().split())
        doc_words = set(document.get_text().lower().split())
        matched_terms = list(query_words & doc_words)
        
        # Find matched entities
        query_entity_texts = {entity['text'].lower() for entity in query.entities}
        doc_entity_texts = {entity['text'].lower() for entity in document.entities}
        matched_entities = list(query_entity_texts & doc_entity_texts)
        
        # Find matched concepts
        matched_concepts = list(set(query.concepts) & set(document.concepts))
        
        # Generate explanation
        explanation = await self._generate_explanation(
            query, document, matched_terms, matched_entities, matched_concepts, similarity
        )
        
        return SearchResult(
            document=document,
            score=similarity,
            rank=rank,
            matched_terms=matched_terms,
            matched_entities=matched_entities,
            matched_concepts=matched_concepts,
            explanation=explanation,
            retrieval_method="semantic_embedding",
            confidence=similarity
        )
    
    async def _generate_explanation(self, query: Query, document: Document,
                                  matched_terms: List[str], matched_entities: List[str],
                                  matched_concepts: List[str], similarity: float) -> str:
        """Generate explanation for search result"""
        
        explanations = []
        
        # Term matching
        if matched_terms:
            explanations.append(f"Matched keywords: {', '.join(matched_terms[:3])}")
        
        # Entity matching
        if matched_entities:
            explanations.append(f"Matched entities: {', '.join(matched_entities[:2])}")
        
        # Concept matching
        if matched_concepts:
            concept_names = [concept.replace('_', ' ').title() for concept in matched_concepts[:2]]
            explanations.append(f"Related concepts: {', '.join(concept_names)}")
        
        # Semantic similarity
        if similarity > 0.8:
            explanations.append("High semantic similarity")
        elif similarity > 0.6:
            explanations.append("Good semantic match")
        elif similarity > 0.4:
            explanations.append("Moderate semantic relevance")
        
        # Query intent matching
        if query.intent == SearchIntent.DEFINITIONAL and any(word in document.content.lower() for word in ['definition', 'define', 'means', 'is']):
            explanations.append("Contains definitional content")
        elif query.intent == SearchIntent.HOW_TO and any(word in document.content.lower() for word in ['step', 'how', 'method', 'process']):
            explanations.append("Contains instructional content")
        
        return "; ".join(explanations) if explanations else f"Semantic similarity: {similarity:.2f}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        
        # Document statistics
        doc_categories = defaultdict(int)
        doc_entities = defaultdict(int)
        doc_concepts = defaultdict(int)
        
        for document in self.documents.values():
            if document.category:
                doc_categories[document.category] += 1
            
            for entity in document.entities:
                doc_entities[entity['type']] += 1
            
            for concept in document.concepts:
                doc_concepts[concept] += 1
        
        return {
            'index_statistics': {
                'total_documents': len(self.documents),
                'total_embeddings': len(self.document_embeddings),
                'vocabulary_size': len(self.embedder.vocabulary),
                'embedding_dimension': self.embedder.embedding_dim
            },
            'search_statistics': self.stats,
            'content_analysis': {
                'document_categories': dict(doc_categories),
                'entity_distribution': dict(doc_entities),
                'concept_distribution': dict(doc_concepts)
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_document_indexing():
    """Demo: Document indexing and processing"""
    print("\nDEMO 1: DOCUMENT INDEXING AND PROCESSING")
    print("=" * 50)
    
    search_engine = SemanticSearchEngine(embedding_dim=128)  # Smaller for demo
    await search_engine.initialize()
    
    # Create sample documents
    documents = [
        Document(
            id="",
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. Popular techniques include supervised learning, unsupervised learning, and reinforcement learning. Python is widely used for machine learning projects.",
            author="AI Researcher",
            category="Technology",
            tags=["AI", "ML", "Programming"]
        ),
        Document(
            id="",
            title="Deep Learning with Neural Networks",
            content="Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. TensorFlow and PyTorch are popular frameworks for deep learning. Applications include computer vision and natural language processing.",
            author="Data Scientist",
            category="Technology", 
            tags=["Deep Learning", "Neural Networks", "AI"]
        ),
        Document(
            id="",
            title="Python Programming Best Practices",
            content="Python is a versatile programming language known for its simplicity and readability. Best practices include writing clean code, using virtual environments, and following PEP 8 style guidelines. Python is excellent for data science and web development.",
            author="Software Engineer",
            category="Programming",
            tags=["Python", "Programming", "Best Practices"]
        ),
        Document(
            id="",
            title="Data Science Methodology",
            content="Data science involves extracting insights from data using statistical methods and machine learning algorithms. The process includes data collection, cleaning, analysis, and visualization. Tools like pandas, numpy, and matplotlib are essential for data scientists.",
            author="Data Analyst",
            category="Data Science",
            tags=["Data Science", "Statistics", "Analysis"]
        ),
        Document(
            id="",
            title="Natural Language Processing Applications",
            content="Natural language processing enables computers to understand and generate human language. Applications include sentiment analysis, machine translation, and chatbots. Modern NLP uses transformer models like BERT and GPT for better performance.",
            author="NLP Researcher",
            category="Technology",
            tags=["NLP", "Language", "AI"]
        )
    ]
    
    print(f"Indexing {len(documents)} documents...")
    
    # Index documents
    indexing_result = await search_engine.index_documents(documents)
    
    if indexing_result['success']:
        print(f"✓ Successfully indexed {indexing_result['documents_indexed']} documents")
        print(f"  Indexing time: {indexing_result['indexing_time']:.3f}s")
        print(f"  Total documents: {indexing_result['total_documents']}")
    else:
        print(f"✗ Indexing failed: {indexing_result['error']}")
        return
    
    # Show processed document details
    print(f"\nProcessed Document Details:")
    
    for i, (doc_id, doc) in enumerate(list(search_engine.documents.items())[:3], 1):
        print(f"\n{i}. {doc.title}")
        print(f"   Category: {doc.category}")
        print(f"   Author: {doc.author}")
        print(f"   Tags: {', '.join(doc.tags)}")
        
        if doc.entities:
            print(f"   Entities: {[e['text'] for e in doc.entities[:3]]}")
        
        if doc.concepts:
            print(f"   Concepts: {[c.replace('_', ' ').title() for c in doc.concepts[:3]]}")
        
        print(f"   Summary: {doc.summary}")
        print(f"   Embedding shape: {doc.embedding.shape if doc.embedding is not None else 'None'}")
    
    # Show indexing statistics
    stats = search_engine.get_statistics()
    
    print(f"\nIndexing Statistics:")
    index_stats = stats['index_statistics']
    print(f"  Total documents: {index_stats['total_documents']}")
    print(f"  Vocabulary size: {index_stats['vocabulary_size']}")
    print(f"  Embedding dimension: {index_stats['embedding_dimension']}")
    
    content_stats = stats['content_analysis']
    print(f"\nContent Analysis:")
    print(f"  Categories: {content_stats['document_categories']}")
    print(f"  Entity types: {list(content_stats['entity_distribution'].keys())}")
    print(f"  Top concepts: {list(content_stats['concept_distribution'].keys())[:5]}")

async def demo_query_analysis():
    """Demo: Query analysis and understanding"""
    print("\nDEMO 2: QUERY ANALYSIS AND UNDERSTANDING")
    print("=" * 50)
    
    analyzer = QueryAnalyzer()
    
    # Test different types of queries
    test_queries = [
        "machine learning algorithms",
        "What is artificial intelligence?",
        "How to implement neural networks in Python",
        "Compare TensorFlow vs PyTorch",
        "Deep learning techniques for computer vision",
        "Python programming best practices",
        "Can you help me understand data science methodology?",
        "Show me natural language processing applications"
    ]
    
    print("Analyzing different types of queries:")
    
    for query_text in test_queries:
        print(f"\nQuery: '{query_text}'")
        
        # Create and analyze query
        query = Query(
            id="",
            text=query_text,
            query_type=QueryType.KEYWORD
        )
        
        analyzed_query = await analyzer.analyze_query(query)
        
        print(f"  Type: {analyzed_query.query_type.value}")
        print(f"  Intent: {analyzed_query.intent.value if analyzed_query.intent else 'Unknown'}")
        
        if analyzed_query.entities:
            entities_text = [f"{e['text']} ({e['type']})" for e in analyzed_query.entities[:3]]
            print(f"  Entities: {', '.join(entities_text)}")
        
        if analyzed_query.concepts:
            concepts_text = [c.replace('_', ' ').title() for c in analyzed_query.concepts]
            print(f"  Concepts: {', '.join(concepts_text)}")
        
        if analyzed_query.expanded_terms:
            print(f"  Expanded terms: {', '.join(analyzed_query.expanded_terms[:5])}")
    
    print(f"\nQuery Type Classification:")
    print(f"  KEYWORD: Simple keyword searches")
    print(f"  QUESTION: Natural language questions") 
    print(f"  SEMANTIC: Complex meaning-based queries")
    print(f"  CONVERSATIONAL: Multi-turn dialogue queries")
    
    print(f"\nIntent Classification:")
    print(f"  DEFINITIONAL: Seeking definitions or explanations")
    print(f"  HOW_TO: Looking for instructions or tutorials")
    print(f"  COMPARISON: Comparing different options")
    print(f"  INFORMATIONAL: General information seeking")

async def demo_semantic_search():
    """Demo: Semantic search with different query types"""
    print("\nDEMO 3: SEMANTIC SEARCH WITH DIFFERENT QUERIES")
    print("=" * 50)
    
    # Initialize search engine with documents
    search_engine = SemanticSearchEngine(embedding_dim=128)
    await search_engine.initialize()
    
    # Create diverse documents
    documents = [
        Document(
            id="",
            title="Machine Learning Fundamentals",
            content="Machine learning algorithms enable computers to learn patterns from data without explicit programming. Supervised learning uses labeled data, unsupervised learning finds hidden patterns, and reinforcement learning learns through trial and error. Popular algorithms include linear regression, decision trees, and neural networks.",
            category="Education",
            tags=["ML", "Algorithms", "Learning"]
        ),
        Document(
            id="",
            title="Python for Data Science",
            content="Python is the most popular programming language for data science due to its simplicity and powerful libraries. NumPy provides numerical computing, Pandas handles data manipulation, Matplotlib creates visualizations, and Scikit-learn offers machine learning algorithms. Jupyter notebooks are ideal for interactive analysis.",
            category="Programming",
            tags=["Python", "Data Science", "Tools"]
        ),
        Document(
            id="",
            title="Neural Network Architecture Design",
            content="Designing effective neural network architectures requires understanding of layers, activation functions, and optimization techniques. Convolutional neural networks excel at image processing, recurrent networks handle sequential data, and transformer architectures power modern language models. Proper initialization and regularization prevent overfitting.",
            category="Deep Learning",
            tags=["Neural Networks", "Architecture", "Design"]
        ),
        Document(
            id="",
            title="Introduction to Artificial Intelligence", 
            content="Artificial intelligence aims to create machines that can perform tasks requiring human intelligence. AI includes machine learning, natural language processing, computer vision, and robotics. Applications range from recommendation systems to autonomous vehicles. AI development requires careful consideration of ethics and societal impact.",
            category="AI Overview",
            tags=["AI", "Overview", "Applications"]
        ),
        Document(
            id="",
            title="Data Preprocessing Techniques",
            content="Data preprocessing is crucial for successful machine learning projects. Steps include data cleaning to handle missing values, normalization to scale features, encoding categorical variables, and feature selection to identify relevant inputs. Quality preprocessing significantly improves model performance and reduces training time.",
            category="Data Processing",
            tags=["Data", "Preprocessing", "Quality"]
        )
    ]
    
    # Index documents
    await search_engine.index_documents(documents)
    
    # Test different search queries
    search_queries = [
        ("machine learning", "Simple keyword search"),
        ("What is artificial intelligence?", "Definitional question"),
        ("How to use Python for data analysis", "How-to query"),
        ("neural network design principles", "Technical concept search"),
        ("data cleaning and preparation", "Process-oriented search"),
        ("AI applications in real world", "Application-focused search")
    ]
    
    print("Testing semantic search with different query types:")
    
    for query_text, query_description in search_queries:
        print(f"\n--- {query_description} ---")
        print(f"Query: '{query_text}'")
        
        # Perform search
        search_result = await search_engine.search(
            query_text=query_text,
            limit=3,
            user_id="demo_user"
        )
        
        if search_result['success']:
            query_info = search_result['query']
            results = search_result['results']
            
            print(f"Query analysis:")
            print(f"  Type: {query_info['type']}")
            print(f"  Intent: {query_info['intent'] or 'Unknown'}")
            if query_info['concepts']:
                concepts = [c.replace('_', ' ').title() for c in query_info['concepts']]
                print(f"  Concepts: {', '.join(concepts)}")
            
            print(f"Results ({len(results)} found):")
            
            for result in results:
                doc = result['document']
                print(f"  {result['rank']}. {doc['title']} (Score: {result['score']:.3f})")
                print(f"     Category: {doc['category']}")
                print(f"     Explanation: {result['explanation']}")
                if result['matched_concepts']:
                    matched = [c.replace('_', ' ').title() for c in result['matched_concepts']]
                    print(f"     Matched concepts: {', '.join(matched)}")
        else:
            print(f"Search failed: {search_result['error']}")
    
    # Show search statistics
    stats = search_engine.get_statistics()
    search_stats = stats['search_statistics']
    
    print(f"\nSearch Performance:")
    print(f"  Queries processed: {search_stats['queries_processed']}")
    print(f"  Average search time: {search_stats['average_search_time']:.3f}s")

async def demo_similarity_computation():
    """Demo: Text similarity and embedding comparison"""
    print("\nDEMO 4: TEXT SIMILARITY AND EMBEDDING COMPARISON")
    print("=" * 50)
    
    embedder = TextEmbedder(embedding_dim=128)
    
    # Create sample texts with varying similarity
    text_pairs = [
        (
            "Machine learning algorithms learn from data",
            "AI systems can learn patterns from training data",
            "High similarity - same concept, different words"
        ),
        (
            "Python is a programming language",
            "Python programming for beginners",
            "Medium similarity - related but different focus"
        ),
        (
            "Deep learning neural networks",
            "Cooking recipes for dinner",
            "Low similarity - completely different topics"
        ),
        (
            "Artificial intelligence and machine learning",
            "AI and ML technologies",
            "High similarity - synonyms and abbreviations"
        ),
        (
            "Data science methodology and best practices",
            "Statistical analysis and data mining techniques",
            "Medium similarity - overlapping domain"
        )
    ]
    
    print("Building vocabulary from sample texts...")
    
    # Create documents for vocabulary building
    vocab_docs = []
    for text1, text2, _ in text_pairs:
        vocab_docs.append(Document(id="", title="", content=text1))
        vocab_docs.append(Document(id="", title="", content=text2))
    
    await embedder.build_vocabulary(vocab_docs)
    
    print(f"Vocabulary size: {len(embedder.vocabulary)}")
    
    print(f"\nComputing text similarities:")
    
    for text1, text2, description in text_pairs:
        print(f"\nComparison: {description}")
        print(f"  Text 1: '{text1}'")
        print(f"  Text 2: '{text2}'")
        
        # Generate embeddings
        embedding1 = await embedder.embed_text(text1)
        embedding2 = await embedder.embed_text(text2)
        
        # Compute similarity
        similarity = embedder.compute_similarity(embedding1, embedding2)
        
        print(f"  Similarity score: {similarity:.3f}")
        
        # Interpret similarity
        if similarity > 0.8:
            interpretation = "Very high similarity"
        elif similarity > 0.6:
            interpretation = "High similarity"
        elif similarity > 0.4:
            interpretation = "Moderate similarity"
        elif similarity > 0.2:
            interpretation = "Low similarity"
        else:
            interpretation = "Very low similarity"
        
        print(f"  Interpretation: {interpretation}")
    
    # Demonstrate embedding properties
    print(f"\nEmbedding Properties:")
    
    sample_text = "machine learning algorithms"
    embedding = await embedder.embed_text(sample_text)
    
    print(f"  Sample text: '{sample_text}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.3f}")
    print(f"  First 5 dimensions: {embedding[:5]}")
    
    # Show word-level analysis
    print(f"\nWord-level Analysis:")
    words = sample_text.split()
    
    for word in words:
        if word in embedder.vocabulary:
            word_vector = embedder.word_vectors[word]
            idf_score = embedder.idf_scores.get(word, 0)
            print(f"  '{word}': vocab_id={embedder.vocabulary[word]}, idf={idf_score:.3f}")

async def demo_advanced_search_features():
    """Demo: Advanced search features and explanations"""
    print("\nDEMO 5: ADVANCED SEARCH FEATURES")
    print("=" * 50)
    
    search_engine = SemanticSearchEngine(embedding_dim=128)
    await search_engine.initialize()
    
    # Create specialized documents
    documents = [
        Document(
            id="",
            title="Getting Started with TensorFlow",
            content="TensorFlow is an open-source machine learning framework developed by Google. It provides tools for building and training neural networks. TensorFlow supports both research and production environments. Key features include automatic differentiation, distributed computing, and model deployment capabilities.",
            author="Google AI Team",
            category="Tutorial",
            tags=["TensorFlow", "Google", "Tutorial", "ML Framework"]
        ),
        Document(
            id="",
            title="PyTorch Deep Learning Guide",
            content="PyTorch is a dynamic deep learning framework created by Facebook. It offers intuitive APIs and excellent debugging capabilities. PyTorch is popular in research communities due to its flexibility. The framework supports GPU acceleration and distributed training for large models.",
            author="Facebook AI Research",
            category="Tutorial", 
            tags=["PyTorch", "Facebook", "Deep Learning", "Research"]
        ),
        Document(
            id="",
            title="Comparing ML Frameworks",
            content="Machine learning frameworks each have unique strengths. TensorFlow excels in production deployment, PyTorch offers research flexibility, and scikit-learn provides classical algorithms. Choice depends on project requirements, team expertise, and performance needs. Consider factors like community support and documentation quality.",
            author="ML Engineer",
            category="Comparison",
            tags=["Comparison", "Frameworks", "Analysis"]
        ),
        Document(
            id="",
            title="Computer Vision with CNN",
            content="Convolutional Neural Networks are the foundation of computer vision. CNNs use convolution layers to detect features in images. Popular architectures include LeNet, AlexNet, VGG, and ResNet. Applications include image classification, object detection, and facial recognition. Transfer learning accelerates development.",
            author="CV Researcher",
            category="Technical",
            tags=["Computer Vision", "CNN", "Image Processing"]
        ),
        Document(
            id="",
            title="Natural Language Processing Basics",
            content="Natural Language Processing enables computers to understand human language. Core tasks include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. Modern NLP uses transformer architectures like BERT and GPT. Applications include chatbots, translation, and text summarization.",
            author="NLP Expert",
            category="Technical",
            tags=["NLP", "Language", "Text Processing"]
        )
    ]
    
    await search_engine.index_documents(documents)
    
    # Advanced search scenarios
    search_scenarios = [
        {
            'query': "TensorFlow vs PyTorch comparison",
            'description': "Comparison query with entity recognition",
            'expected': "Should find comparison document and individual framework docs"
        },
        {
            'query': "How to build neural networks for images",
            'description': "How-to query with concept mapping",
            'expected': "Should find computer vision and framework tutorials"
        },
        {
            'query': "Google machine learning tools",
            'description': "Entity-based search",
            'expected': "Should find TensorFlow document (Google product)"
        },
        {
            'query': "deep learning research framework",
            'description': "Concept-based semantic search", 
            'expected': "Should find PyTorch (research-focused) and related docs"
        },
        {
            'query': "What is computer vision CNN",
            'description': "Definitional query with technical terms",
            'expected': "Should find computer vision document with CNN explanation"
        }
    ]
    
    print("Testing advanced search features:")
    
    for scenario in search_scenarios:
        print(f"\n--- {scenario['description']} ---")
        print(f"Query: '{scenario['query']}'")
        print(f"Expected: {scenario['expected']}")
        
        search_result = await search_engine.search(
            query_text=scenario['query'],
            limit=3,
            user_id="advanced_user"
        )
        
        if search_result['success']:
            query_info = search_result['query']
            results = search_result['results']
            
            print(f"\nQuery Analysis:")
            print(f"  Type: {query_info['type']}")
            print(f"  Intent: {query_info['intent'] or 'Unknown'}")
            
            if query_info['entities']:
                entities = [f"{e['text']} ({e['type']})" for e in query_info['entities']]
                print(f"  Entities: {', '.join(entities)}")
            
            if query_info['concepts']:
                concepts = [c.replace('_', ' ').title() for c in query_info['concepts']]
                print(f"  Concepts: {', '.join(concepts)}")
            
            print(f"\nResults:")
            for result in results:
                doc = result['document']
                print(f"  {result['rank']}. {doc['title']} (Score: {result['score']:.3f})")
                print(f"     Author: {doc['author']}")
                print(f"     Category: {doc['category']}")
                print(f"     Tags: {', '.join(doc['tags'])}")
                print(f"     Explanation: {result['explanation']}")
                
                # Show specific matches
                if result['matched_terms']:
                    print(f"     Matched terms: {', '.join(result['matched_terms'])}")
                if result['matched_entities']:
                    print(f"     Matched entities: {', '.join(result['matched_entities'])}")
                if result['matched_concepts']:
                    concepts = [c.replace('_', ' ').title() for c in result['matched_concepts']]
                    print(f"     Matched concepts: {', '.join(concepts)}")
        else:
            print(f"Search failed: {search_result['error']}")
    
    # Show comprehensive statistics
    stats = search_engine.get_statistics()
    
    print(f"\nSearch Engine Statistics:")
    print(f"  Documents indexed: {stats['index_statistics']['total_documents']}")
    print(f"  Vocabulary size: {stats['index_statistics']['vocabulary_size']}")
    print(f"  Queries processed: {stats['search_statistics']['queries_processed']}")
    print(f"  Average search time: {stats['search_statistics']['average_search_time']:.3f}s")
    
    content_analysis = stats['content_analysis']
    print(f"\nContent Analysis:")
    print(f"  Categories: {content_analysis['document_categories']}")
    print(f"  Top entity types: {list(content_analysis['entity_distribution'].keys())[:5]}")
    print(f"  Top concepts: {list(content_analysis['concept_distribution'].keys())[:5]}")

async def main():
    """
    Demonstrate Semantic Search Implementation for understanding meaning beyond keywords
    
    WHAT YOU'LL LEARN:
    ================
    1. How to build dense vector embeddings for semantic understanding
    2. How to analyze queries and extract intent, entities, and concepts
    3. How to compute semantic similarity between texts
    4. How to implement ranking algorithms for semantic relevance
    5. How to provide explanations for search results
    6. How to handle different types of queries and search intents
    
    REAL WORLD APPLICATIONS:
    =======================
    - Intelligent search engines and information retrieval systems
    - Question answering and conversational AI systems
    - Content recommendation and discovery platforms
    - Knowledge management and enterprise search
    - Customer support and help desk automation
    - Research and scientific literature search
    """
    
    print("SEMANTIC SEARCH IMPLEMENTATION DEMONSTRATION")
    print("Understanding meaning and intent beyond simple keywords!")
    
    await demo_document_indexing()
    await demo_query_analysis()
    await demo_semantic_search()
    await demo_similarity_computation()
    await demo_advanced_search_features()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Document indexing creates semantic representations of content")
    print("✓ Query analysis extracts intent, entities, and concepts")
    print("✓ Semantic search finds relevant content beyond keyword matching")
    print("✓ Similarity computation measures meaning-based relevance")
    print("✓ Advanced features provide intelligent search experiences")
    print("✓ Result explanations help users understand search logic")
    print("\nTHE POWER OF SEMANTIC SEARCH:")
    print("- Understands user intent rather than just keywords")
    print("- Bridges vocabulary gaps between users and content")
    print("- Enables natural language search and conversation")
    print("- Provides foundation for intelligent AI systems")

if __name__ == "__main__":
    asyncio.run(main())
