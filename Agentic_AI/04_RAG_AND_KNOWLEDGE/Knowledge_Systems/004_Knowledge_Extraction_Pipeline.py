#!/usr/bin/env python3
"""
Knowledge Extraction Pipeline: Automated Knowledge Discovery from Text
=====================================================================

WHAT IS THE PROBLEM?
==================
Valuable knowledge is trapped in unstructured text:
- Millions of documents contain critical insights but remain unanalyzed
- Manual knowledge extraction is time-consuming and error-prone
- Important relationships and patterns are missed without systematic analysis
- Knowledge workers spend 80% of time searching rather than analyzing
- Organizations cannot scale human expertise to process all available information
- Critical insights remain buried in data silos

Example: Medical Literature Analysis Crisis
MANUAL APPROACH (Traditional):
- Researchers manually read thousands of medical papers
- Critical drug interactions discovered years after publication
- Treatment protocols based on incomplete evidence review
- Life-saving insights delayed due to information overload
- Systematic reviews take months or years to complete
- Result: Delayed medical breakthroughs, suboptimal patient care

REAL WORLD EXAMPLE:
=================
How does IBM Watson Discovery work?

IBM WATSON DISCOVERY PIPELINE:
1. DOCUMENT INGESTION: Automated import from multiple sources
2. TEXT PREPROCESSING: Cleaning, normalization, and formatting
3. ENTITY EXTRACTION: Identify people, organizations, concepts
4. RELATIONSHIP DETECTION: Discover connections between entities
5. CONCEPT MAPPING: Link to domain-specific knowledge bases
6. INSIGHT GENERATION: Automatic summarization and pattern detection
7. KNOWLEDGE GRAPH CONSTRUCTION: Build structured knowledge representation

BENEFITS OF AUTOMATED KNOWLEDGE EXTRACTION:
- Process thousands of documents in minutes instead of months
- Discover hidden patterns and relationships at scale
- Maintain consistent quality and reduce human bias
- Enable real-time knowledge updates as new information arrives
- Free human experts to focus on analysis rather than data processing
- Create structured knowledge assets for AI systems

THE AUTOMATION ADVANTAGE:
========================
MANUAL EXTRACTION: Documents → Human Reading → Slow, Limited Knowledge
AUTOMATED PIPELINE: Documents → AI Processing → Fast, Comprehensive Knowledge

PIPELINE COMPONENTS:
==================
1. DOCUMENT INGESTION: Multi-format document processing
2. TEXT PREPROCESSING: Cleaning and normalization
3. ENTITY RECOGNITION: Named entity extraction and linking
4. RELATION EXTRACTION: Relationship identification between entities
5. CONCEPT EXTRACTION: Domain-specific concept identification
6. FACT EXTRACTION: Structured fact discovery and validation
7. KNOWLEDGE INTEGRATION: Merging and consolidating extracted knowledge

WHY THIS IS REVOLUTIONARY:
========================
- Enables organizations to leverage all their textual knowledge
- Accelerates research and discovery across all domains
- Powers intelligent search and recommendation systems
- Critical for building comprehensive knowledge bases
- Enables AI systems to understand and reason about information
- Creates competitive advantage through better knowledge utilization
"""

import asyncio
import time
import json
import uuid
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DocumentType(Enum):
    """Types of documents to process"""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    DOCX = "docx"

class ExtractionMethod(Enum):
    """Methods for knowledge extraction"""
    RULE_BASED = "rule_based"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    STATISTICAL = "statistical"

class EntityCategory(Enum):
    """Categories of entities to extract"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    EVENT = "event"
    PROCESS = "process"

class RelationType(Enum):
    """Types of relationships to extract"""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    CAUSES = "causes"
    PREVENTS = "prevents"
    INFLUENCES = "influences"
    SIMILAR_TO = "similar_to"
    OPPOSITE_TO = "opposite_to"
    TEMPORAL = "temporal"
    CAUSAL = "causal"

@dataclass
class Document:
    """Represents a document for processing"""
    
    id: str
    title: str
    content: str
    doc_type: DocumentType
    
    # Metadata
    source: str = ""
    author: str = ""
    publish_date: Optional[datetime] = None
    url: str = ""
    language: str = "en"
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    processing_version: str = "1.0"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    
    id: str
    text: str
    category: EntityCategory
    
    # Position information
    start_pos: int
    end_pos: int
    sentence_id: int
    
    # Confidence and metadata
    confidence: float = 1.0
    extraction_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    normalized_form: str = ""
    
    # Linking information
    external_id: str = ""  # Link to external knowledge base
    aliases: List[str] = field(default_factory=list)
    
    # Context
    context_sentence: str = ""
    surrounding_entities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.normalized_form:
            self.normalized_form = self.text.lower().strip()

@dataclass
class ExtractedRelation:
    """Represents an extracted relationship"""
    
    id: str
    subject_entity_id: str
    object_entity_id: str
    relation_type: RelationType
    
    # Confidence and metadata
    confidence: float = 1.0
    extraction_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    
    # Context information
    sentence_id: int = 0
    context_sentence: str = ""
    trigger_phrase: str = ""  # Phrase that indicated this relationship
    
    # Temporal information
    temporal_context: str = ""  # When this relationship is true
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ExtractedFact:
    """Represents an extracted fact"""
    
    id: str
    subject: str
    predicate: str
    object: str
    
    # Confidence and metadata
    confidence: float = 1.0
    extraction_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    
    # Source information
    source_sentence: str = ""
    sentence_id: int = 0
    document_id: str = ""
    
    # Validation
    is_validated: bool = False
    validation_score: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ExtractionResult:
    """Complete extraction result for a document"""
    
    document_id: str
    
    # Extracted knowledge
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    facts: List[ExtractedFact] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    extraction_methods_used: List[ExtractionMethod] = field(default_factory=list)
    
    # Quality metrics
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    entity_count_by_category: Dict[str, int] = field(default_factory=dict)
    relation_count_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class TextPreprocessor:
    """Preprocesses text for knowledge extraction"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
        
        self.stop_words = set(stopwords.words('english'))
        self.logger = logging.getLogger("TextPreprocessor")
    
    async def preprocess_document(self, document: Document) -> Dict[str, Any]:
        """Preprocess document for extraction"""
        
        start_time = time.time()
        
        try:
            # Clean text
            cleaned_text = self._clean_text(document.content)
            
            # Sentence segmentation
            sentences = sent_tokenize(cleaned_text)
            
            # Tokenization and basic processing
            processed_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Tokenize
                tokens = word_tokenize(sentence)
                
                # Filter tokens
                filtered_tokens = [
                    token.lower() for token in tokens 
                    if token.isalpha() and token.lower() not in self.stop_words and len(token) > 2
                ]
                
                # POS tagging
                pos_tags = pos_tag(tokens)
                
                processed_sentences.append({
                    'id': i,
                    'text': sentence,
                    'tokens': tokens,
                    'filtered_tokens': filtered_tokens,
                    'pos_tags': pos_tags,
                    'word_count': len(tokens)
                })
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'original_text': document.content,
                'cleaned_text': cleaned_text,
                'sentences': processed_sentences,
                'sentence_count': len(sentences),
                'total_tokens': sum(len(s['tokens']) for s in processed_sentences),
                'processing_time': processing_time
            }
            
            self.logger.debug(f"Preprocessed document {document.id}: {len(sentences)} sentences")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for document {document.id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Fix common formatting issues
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)  # Space before punctuation
        
        return text.strip()

class EntityExtractor:
    """Extracts named entities from text"""
    
    def __init__(self):
        # Entity patterns for rule-based extraction
        self.entity_patterns = {
            EntityCategory.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\bDr\. [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b',  # Dr. Name
                r'\bProf\. [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b',  # Prof. Name
                r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2},\s*Ph\.?D\.?\b'  # Name, PhD
            ],
            EntityCategory.ORGANIZATION: [
                r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s+(Inc|Corp|LLC|Ltd|Company|Institute|University|Foundation)\b',
                r'\bUniversity\s+of\s+[A-Z][a-z]+\b',
                r'\b[A-Z][A-Z]+\s+(?:Inc|Corp|LLC|Ltd)\b'
            ],
            EntityCategory.LOCATION: [
                r'\b[A-Z][a-z]+,\s*[A-Z][A-Z]\b',  # City, State
                r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, Country
                r'\b\d+\s+[A-Z][a-z]+\s+(Street|Avenue|Road|Boulevard|Lane|Drive)\b'
            ],
            EntityCategory.DATE: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b'  # YYYY-MM-DD
            ],
            EntityCategory.MONEY: [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Dollar amounts
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?\b',
                r'\b\d+(?:,\d{3})*\s*USD\b'
            ],
            EntityCategory.TECHNOLOGY: [
                r'\b(Python|Java|JavaScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go)\b',
                r'\b(TensorFlow|PyTorch|Keras|scikit-learn|pandas|numpy)\b',
                r'\b(AWS|Azure|Google Cloud|Docker|Kubernetes)\b'
            ]
        }
        
        # Concept keywords for domain-specific extraction
        self.concept_keywords = {
            'machine_learning': [
                'machine learning', 'ml', 'supervised learning', 'unsupervised learning',
                'reinforcement learning', 'deep learning', 'neural networks'
            ],
            'artificial_intelligence': [
                'artificial intelligence', 'ai', 'cognitive computing', 'expert systems'
            ],
            'data_science': [
                'data science', 'data analysis', 'analytics', 'big data', 'data mining'
            ],
            'software_engineering': [
                'software engineering', 'programming', 'software development', 'coding'
            ]
        }
        
        self.logger = logging.getLogger("EntityExtractor")
    
    async def extract_entities(self, processed_doc: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract entities from preprocessed document"""
        
        entities = []
        sentences = processed_doc['sentences']
        
        try:
            # Rule-based extraction
            rule_entities = await self._extract_rule_based_entities(sentences)
            entities.extend(rule_entities)
            
            # NLTK-based extraction
            nltk_entities = await self._extract_nltk_entities(sentences)
            entities.extend(nltk_entities)
            
            # Concept extraction
            concept_entities = await self._extract_concept_entities(sentences)
            entities.extend(concept_entities)
            
            # Deduplicate entities
            entities = await self._deduplicate_entities(entities)
            
            self.logger.debug(f"Extracted {len(entities)} entities")
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    async def _extract_rule_based_entities(self, sentences: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Extract entities using rule-based patterns"""
        
        entities = []
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            sentence_id = sentence_info['id']
            
            for category, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence_text)
                    
                    for match in matches:
                        entity_text = match.group().strip()
                        
                        entity = ExtractedEntity(
                            id="",
                            text=entity_text,
                            category=category,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            sentence_id=sentence_id,
                            confidence=0.8,
                            extraction_method=ExtractionMethod.RULE_BASED,
                            context_sentence=sentence_text
                        )
                        
                        entities.append(entity)
        
        return entities
    
    async def _extract_nltk_entities(self, sentences: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Extract entities using NLTK named entity recognition"""
        
        entities = []
        
        # NLTK entity type mapping
        nltk_to_category = {
            'PERSON': EntityCategory.PERSON,
            'ORGANIZATION': EntityCategory.ORGANIZATION,
            'GPE': EntityCategory.LOCATION,  # Geopolitical entity
            'LOCATION': EntityCategory.LOCATION,
            'MONEY': EntityCategory.MONEY,
            'DATE': EntityCategory.DATE,
            'TIME': EntityCategory.DATE
        }
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            sentence_id = sentence_info['id']
            pos_tags = sentence_info['pos_tags']
            
            # Named entity chunking
            named_entities = ne_chunk(pos_tags)
            
            for chunk in named_entities:
                if hasattr(chunk, 'label'):  # It's a named entity
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    chunk_label = chunk.label()
                    
                    if chunk_label in nltk_to_category:
                        # Find position in sentence
                        start_pos = sentence_text.find(entity_text)
                        end_pos = start_pos + len(entity_text)
                        
                        if start_pos != -1:
                            entity = ExtractedEntity(
                                id="",
                                text=entity_text,
                                category=nltk_to_category[chunk_label],
                                start_pos=start_pos,
                                end_pos=end_pos,
                                sentence_id=sentence_id,
                                confidence=0.7,
                                extraction_method=ExtractionMethod.MACHINE_LEARNING,
                                context_sentence=sentence_text
                            )
                            
                            entities.append(entity)
        
        return entities
    
    async def _extract_concept_entities(self, sentences: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Extract domain-specific concept entities"""
        
        entities = []
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text'].lower()
            sentence_id = sentence_info['id']
            
            for concept, keywords in self.concept_keywords.items():
                for keyword in keywords:
                    if keyword in sentence_text:
                        # Find exact position
                        start_pos = sentence_text.find(keyword)
                        if start_pos != -1:
                            entity = ExtractedEntity(
                                id="",
                                text=keyword,
                                category=EntityCategory.CONCEPT,
                                start_pos=start_pos,
                                end_pos=start_pos + len(keyword),
                                sentence_id=sentence_id,
                                confidence=0.9,
                                extraction_method=ExtractionMethod.RULE_BASED,
                                context_sentence=sentence_info['text'],
                                normalized_form=concept
                            )
                            
                            entities.append(entity)
        
        return entities
    
    async def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        
        unique_entities = []
        seen_entities = set()
        
        for entity in entities:
            # Create key based on normalized form and category
            entity_key = (entity.normalized_form, entity.category.value, entity.sentence_id)
            
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_entities.append(entity)
            else:
                # Update confidence if higher
                for existing in unique_entities:
                    if (existing.normalized_form == entity.normalized_form and
                        existing.category == entity.category and
                        existing.sentence_id == entity.sentence_id):
                        
                        if entity.confidence > existing.confidence:
                            existing.confidence = entity.confidence
                            existing.extraction_method = entity.extraction_method
                        break
        
        return unique_entities

class RelationExtractor:
    """Extracts relationships between entities"""
    
    def __init__(self):
        # Relationship patterns
        self.relation_patterns = {
            RelationType.WORKS_FOR: [
                r'(\w+(?:\s+\w+)*)\s+works?\s+(?:for|at)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?employed\s+by\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*),?\s+(?:a\s+)?(?:researcher|scientist|engineer)\s+at\s+(\w+(?:\s+\w+)*)'
            ],
            RelationType.LOCATED_IN: [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:located|based|situated)\s+in\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+in\s+(\w+,\s*\w+)',
                r'(\w+(?:\s+\w+)*)\s+headquarters\s+in\s+(\w+(?:\s+\w+)*)'
            ],
            RelationType.PART_OF: [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:part\s+of|component\s+of|belongs\s+to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:in|within)\s+(?:the\s+)?(\w+(?:\s+\w+)*)\s+(?:department|division|group)'
            ],
            RelationType.CAUSES: [
                r'(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+leads?\s+to\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+results?\s+in\s+(\w+(?:\s+\w+)*)'
            ],
            RelationType.SIMILAR_TO: [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?similar\s+to\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:like|comparable\s+to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)\s+are\s+similar'
            ]
        }
        
        self.logger = logging.getLogger("RelationExtractor")
    
    async def extract_relations(self, entities: List[ExtractedEntity],
                              sentences: List[Dict[str, Any]]) -> List[ExtractedRelation]:
        """Extract relationships between entities"""
        
        relations = []
        
        try:
            # Create entity lookup by sentence
            entities_by_sentence = defaultdict(list)
            for entity in entities:
                entities_by_sentence[entity.sentence_id].append(entity)
            
            # Pattern-based relation extraction
            pattern_relations = await self._extract_pattern_relations(sentences, entities_by_sentence)
            relations.extend(pattern_relations)
            
            # Co-occurrence based relations
            cooccurrence_relations = await self._extract_cooccurrence_relations(entities_by_sentence)
            relations.extend(cooccurrence_relations)
            
            self.logger.debug(f"Extracted {len(relations)} relations")
            
        except Exception as e:
            self.logger.error(f"Relation extraction failed: {e}")
        
        return relations
    
    async def _extract_pattern_relations(self, sentences: List[Dict[str, Any]],
                                       entities_by_sentence: Dict[int, List[ExtractedEntity]]) -> List[ExtractedRelation]:
        """Extract relations using patterns"""
        
        relations = []
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            sentence_id = sentence_info['id']
            sentence_entities = entities_by_sentence.get(sentence_id, [])
            
            if len(sentence_entities) < 2:
                continue
            
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence_text, re.IGNORECASE)
                    
                    for match in matches:
                        if len(match.groups()) >= 2:
                            subject_text = match.group(1).strip()
                            object_text = match.group(2).strip()
                            
                            # Find matching entities
                            subject_entity = await self._find_matching_entity(
                                subject_text, sentence_entities
                            )
                            object_entity = await self._find_matching_entity(
                                object_text, sentence_entities
                            )
                            
                            if subject_entity and object_entity:
                                relation = ExtractedRelation(
                                    id="",
                                    subject_entity_id=subject_entity.id,
                                    object_entity_id=object_entity.id,
                                    relation_type=relation_type,
                                    confidence=0.8,
                                    extraction_method=ExtractionMethod.RULE_BASED,
                                    sentence_id=sentence_id,
                                    context_sentence=sentence_text,
                                    trigger_phrase=match.group()
                                )
                                
                                relations.append(relation)
        
        return relations
    
    async def _extract_cooccurrence_relations(self, entities_by_sentence: Dict[int, List[ExtractedEntity]]) -> List[ExtractedRelation]:
        """Extract relations based on entity co-occurrence"""
        
        relations = []
        
        for sentence_id, sentence_entities in entities_by_sentence.items():
            if len(sentence_entities) < 2:
                continue
            
            # Create relations between co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    # Determine relation type based on entity categories
                    relation_type = await self._infer_relation_type(entity1, entity2)
                    
                    if relation_type:
                        relation = ExtractedRelation(
                            id="",
                            subject_entity_id=entity1.id,
                            object_entity_id=entity2.id,
                            relation_type=relation_type,
                            confidence=0.5,  # Lower confidence for inferred relations
                            extraction_method=ExtractionMethod.STATISTICAL,
                            sentence_id=sentence_id,
                            context_sentence=entity1.context_sentence
                        )
                        
                        relations.append(relation)
        
        return relations
    
    async def _find_matching_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity that matches the given text"""
        
        text_lower = text.lower().strip()
        
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
            
            # Check aliases
            for alias in entity.aliases:
                if text_lower in alias.lower() or alias.lower() in text_lower:
                    return entity
        
        return None
    
    async def _infer_relation_type(self, entity1: ExtractedEntity, 
                                 entity2: ExtractedEntity) -> Optional[RelationType]:
        """Infer relation type based on entity categories"""
        
        category1 = entity1.category
        category2 = entity2.category
        
        # Common relation patterns
        if category1 == EntityCategory.PERSON and category2 == EntityCategory.ORGANIZATION:
            return RelationType.WORKS_FOR
        elif category1 == EntityCategory.ORGANIZATION and category2 == EntityCategory.LOCATION:
            return RelationType.LOCATED_IN
        elif category1 == EntityCategory.CONCEPT and category2 == EntityCategory.CONCEPT:
            return RelationType.SIMILAR_TO
        elif category1 == EntityCategory.TECHNOLOGY and category2 == EntityCategory.CONCEPT:
            return RelationType.PART_OF
        
        return None

class FactExtractor:
    """Extracts structured facts from text"""
    
    def __init__(self):
        # Fact patterns (subject-predicate-object)
        self.fact_patterns = [
            r'(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+was\s+(?:born|founded|created)\s+in\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+developed\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+contains\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+measures\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+costs\s+(\$?\w+(?:\s+\w+)*)'
        ]
        
        # Predicate mapping
        self.predicate_mapping = {
            'is': 'type',
            'has': 'has_property',
            'was born in': 'born_in',
            'was founded in': 'founded_in',
            'was created in': 'created_in',
            'developed': 'developed',
            'contains': 'contains',
            'measures': 'measures',
            'costs': 'costs'
        }
        
        self.logger = logging.getLogger("FactExtractor")
    
    async def extract_facts(self, sentences: List[Dict[str, Any]], 
                          document_id: str) -> List[ExtractedFact]:
        """Extract structured facts from sentences"""
        
        facts = []
        
        try:
            for sentence_info in sentences:
                sentence_text = sentence_info['text']
                sentence_id = sentence_info['id']
                
                # Extract facts using patterns
                for pattern in self.fact_patterns:
                    matches = re.finditer(pattern, sentence_text, re.IGNORECASE)
                    
                    for match in matches:
                        if len(match.groups()) >= 2:
                            subject = match.group(1).strip()
                            object_text = match.group(2).strip()
                            
                            # Determine predicate from pattern
                            predicate = await self._determine_predicate(match.group(), sentence_text)
                            
                            fact = ExtractedFact(
                                id="",
                                subject=subject,
                                predicate=predicate,
                                object=object_text,
                                confidence=0.7,
                                extraction_method=ExtractionMethod.RULE_BASED,
                                source_sentence=sentence_text,
                                sentence_id=sentence_id,
                                document_id=document_id
                            )
                            
                            facts.append(fact)
            
            self.logger.debug(f"Extracted {len(facts)} facts")
            
        except Exception as e:
            self.logger.error(f"Fact extraction failed: {e}")
        
        return facts
    
    async def _determine_predicate(self, matched_text: str, sentence: str) -> str:
        """Determine predicate from matched text"""
        
        matched_lower = matched_text.lower()
        
        for predicate_phrase, mapped_predicate in self.predicate_mapping.items():
            if predicate_phrase in matched_lower:
                return mapped_predicate
        
        # Extract verb from matched text
        words = matched_text.split()
        for word in words:
            if word.lower() in ['is', 'was', 'has', 'have', 'contains', 'includes', 'measures']:
                return word.lower()
        
        return 'related_to'

class KnowledgeExtractionPipeline:
    """Complete knowledge extraction pipeline"""
    
    def __init__(self):
        # Pipeline components
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.fact_extractor = FactExtractor()
        
        # Pipeline configuration
        self.extraction_methods = [ExtractionMethod.RULE_BASED, ExtractionMethod.MACHINE_LEARNING]
        self.min_confidence_threshold = 0.5
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'total_entities_extracted': 0,
            'total_relations_extracted': 0,
            'total_facts_extracted': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        self.logger = logging.getLogger("KnowledgeExtractionPipeline")
    
    async def initialize(self) -> None:
        """Initialize the pipeline"""
        self.logger.info("Knowledge extraction pipeline initialized")
    
    async def process_document(self, document: Document) -> ExtractionResult:
        """Process a single document through the complete pipeline"""
        
        start_time = time.time()
        
        result = ExtractionResult(document_id=document.id)
        
        try:
            self.logger.info(f"Processing document: {document.id}")
            
            # Step 1: Preprocess document
            preprocessed = await self.preprocessor.preprocess_document(document)
            
            if not preprocessed['success']:
                result.errors.append(f"Preprocessing failed: {preprocessed['error']}")
                return result
            
            # Step 2: Extract entities
            entities = await self.entity_extractor.extract_entities(preprocessed)
            
            # Filter entities by confidence
            entities = [e for e in entities if e.confidence >= self.min_confidence_threshold]
            result.entities = entities
            
            # Step 3: Extract relations
            relations = await self.relation_extractor.extract_relations(
                entities, preprocessed['sentences']
            )
            
            # Filter relations by confidence
            relations = [r for r in relations if r.confidence >= self.min_confidence_threshold]
            result.relations = relations
            
            # Step 4: Extract facts
            facts = await self.fact_extractor.extract_facts(
                preprocessed['sentences'], document.id
            )
            
            # Filter facts by confidence
            facts = [f for f in facts if f.confidence >= self.min_confidence_threshold]
            result.facts = facts
            
            # Step 5: Calculate metrics and metadata
            await self._calculate_result_metrics(result)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.extraction_methods_used = self.extraction_methods
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['total_entities_extracted'] += len(entities)
            self.stats['total_relations_extracted'] += len(relations)
            self.stats['total_facts_extracted'] += len(facts)
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['documents_processed']
            )
            
            self.logger.info(f"Document processed: {len(entities)} entities, "
                           f"{len(relations)} relations, {len(facts)} facts, {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
        
        return result
    
    async def process_documents(self, documents: List[Document]) -> List[ExtractionResult]:
        """Process multiple documents"""
        
        results = []
        
        self.logger.info(f"Processing {len(documents)} documents")
        
        for i, document in enumerate(documents, 1):
            self.logger.debug(f"Processing document {i}/{len(documents)}: {document.id}")
            
            result = await self.process_document(document)
            results.append(result)
            
            if i % 10 == 0:
                self.logger.info(f"Processed {i}/{len(documents)} documents")
        
        return results
    
    async def _calculate_result_metrics(self, result: ExtractionResult) -> None:
        """Calculate metrics for extraction result"""
        
        # Entity count by category
        entity_counts = defaultdict(int)
        entity_confidences = []
        
        for entity in result.entities:
            entity_counts[entity.category.value] += 1
            entity_confidences.append(entity.confidence)
        
        result.entity_count_by_category = dict(entity_counts)
        
        # Relation count by type
        relation_counts = defaultdict(int)
        relation_confidences = []
        
        for relation in result.relations:
            relation_counts[relation.relation_type.value] += 1
            relation_confidences.append(relation.confidence)
        
        result.relation_count_by_type = dict(relation_counts)
        
        # Fact confidences
        fact_confidences = [fact.confidence for fact in result.facts]
        
        # Overall confidence scores
        result.confidence_scores = {
            'entity_average': sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0.0,
            'relation_average': sum(relation_confidences) / len(relation_confidences) if relation_confidences else 0.0,
            'fact_average': sum(fact_confidences) / len(fact_confidences) if fact_confidences else 0.0,
            'overall_average': (
                sum(entity_confidences + relation_confidences + fact_confidences) /
                len(entity_confidences + relation_confidences + fact_confidences)
            ) if (entity_confidences + relation_confidences + fact_confidences) else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        
        return {
            'processing_statistics': self.stats,
            'configuration': {
                'extraction_methods': [method.value for method in self.extraction_methods],
                'min_confidence_threshold': self.min_confidence_threshold
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_text_preprocessing():
    """Demo: Text preprocessing and sentence analysis"""
    print("\nDEMO 1: TEXT PREPROCESSING AND ANALYSIS")
    print("=" * 50)
    
    preprocessor = TextPreprocessor()
    
    # Sample document
    sample_doc = Document(
        id="demo_doc_1",
        title="Machine Learning Research",
        content="""
        Dr. John Smith works at Stanford University in California. He specializes in machine learning 
        and artificial intelligence research. His team developed a new neural network architecture 
        called DeepNet in 2023. The project received $2.5 million in funding from Google Inc. 
        DeepNet achieves 95% accuracy on image classification tasks. The research was published 
        in Nature journal in March 2023.
        """,
        doc_type=DocumentType.TEXT
    )
    
    print(f"Original Document:")
    print(f"Title: {sample_doc.title}")
    print(f"Content: {sample_doc.content.strip()}")
    
    # Preprocess document
    preprocessed = await preprocessor.preprocess_document(sample_doc)
    
    if preprocessed['success']:
        print(f"\nPreprocessing Results:")
        print(f"  Sentence count: {preprocessed['sentence_count']}")
        print(f"  Total tokens: {preprocessed['total_tokens']}")
        print(f"  Processing time: {preprocessed['processing_time']:.3f}s")
        
        print(f"\nSentence Analysis:")
        for i, sentence in enumerate(preprocessed['sentences'][:3], 1):
            print(f"\n  Sentence {i}: {sentence['text']}")
            print(f"    Word count: {sentence['word_count']}")
            print(f"    Key tokens: {sentence['filtered_tokens'][:5]}")
            
            # Show POS tags for first few words
            pos_sample = sentence['pos_tags'][:5]
            pos_display = [f"{word}/{pos}" for word, pos in pos_sample]
            print(f"    POS tags: {', '.join(pos_display)}")
    else:
        print(f"Preprocessing failed: {preprocessed['error']}")

async def demo_entity_extraction():
    """Demo: Entity extraction from text"""
    print("\nDEMO 2: ENTITY EXTRACTION")
    print("=" * 50)
    
    preprocessor = TextPreprocessor()
    entity_extractor = EntityExtractor()
    
    # Sample document with various entity types
    sample_doc = Document(
        id="demo_doc_2",
        title="Technology Company Profile",
        content="""
        Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976. 
        The company is now headquartered at 1 Infinite Loop, Cupertino, CA. Tim Cook became CEO 
        in August 2011. Apple develops products like iPhone, iPad, and MacBook using technologies 
        such as iOS, macOS, and Swift programming language. The company's market value reached 
        $3 trillion in January 2023. Apple employs over 147,000 people worldwide and generates 
        annual revenue of $394 billion as of September 2023.
        """,
        doc_type=DocumentType.TEXT
    )
    
    print(f"Document: {sample_doc.title}")
    print(f"Content: {sample_doc.content.strip()}")
    
    # Preprocess and extract entities
    preprocessed = await preprocessor.preprocess_document(sample_doc)
    
    if preprocessed['success']:
        entities = await entity_extractor.extract_entities(preprocessed)
        
        print(f"\nEntity Extraction Results:")
        print(f"Total entities extracted: {len(entities)}")
        
        # Group entities by category
        entities_by_category = defaultdict(list)
        for entity in entities:
            entities_by_category[entity.category].append(entity)
        
        for category, category_entities in entities_by_category.items():
            print(f"\n{category.value.upper()} entities ({len(category_entities)}):")
            
            for entity in category_entities:
                print(f"  - '{entity.text}' (confidence: {entity.confidence:.2f})")
                print(f"    Method: {entity.extraction_method.value}")
                print(f"    Position: {entity.start_pos}-{entity.end_pos}")
                print(f"    Sentence: {entity.sentence_id}")
                if entity.normalized_form != entity.text.lower():
                    print(f"    Normalized: {entity.normalized_form}")
    else:
        print(f"Preprocessing failed: {preprocessed['error']}")

async def demo_relation_extraction():
    """Demo: Relationship extraction between entities"""
    print("\nDEMO 3: RELATIONSHIP EXTRACTION")
    print("=" * 50)
    
    preprocessor = TextPreprocessor()
    entity_extractor = EntityExtractor()
    relation_extractor = RelationExtractor()
    
    # Sample document with clear relationships
    sample_doc = Document(
        id="demo_doc_3",
        title="Research Collaboration",
        content="""
        Dr. Sarah Johnson works for MIT in Boston, Massachusetts. She collaborates with 
        researchers at Google DeepMind. Her team developed AlphaCode using Python and TensorFlow. 
        The project is part of the artificial intelligence research program. AlphaCode is similar 
        to GitHub Copilot in functionality. The research was funded by the National Science Foundation. 
        MIT's Computer Science Department is located in the Stata Center.
        """,
        doc_type=DocumentType.TEXT
    )
    
    print(f"Document: {sample_doc.title}")
    print(f"Content: {sample_doc.content.strip()}")
    
    # Process through pipeline
    preprocessed = await preprocessor.preprocess_document(sample_doc)
    
    if preprocessed['success']:
        entities = await entity_extractor.extract_entities(preprocessed)
        relations = await relation_extractor.extract_relations(entities, preprocessed['sentences'])
        
        print(f"\nExtraction Results:")
        print(f"  Entities: {len(entities)}")
        print(f"  Relations: {len(relations)}")
        
        # Show entities
        print(f"\nExtracted Entities:")
        entity_lookup = {entity.id: entity for entity in entities}
        
        for entity in entities:
            print(f"  {entity.id[:8]}: '{entity.text}' ({entity.category.value})")
        
        # Show relations
        print(f"\nExtracted Relations:")
        
        for relation in relations:
            subject_entity = entity_lookup.get(relation.subject_entity_id)
            object_entity = entity_lookup.get(relation.object_entity_id)
            
            if subject_entity and object_entity:
                print(f"\n  Relation: {relation.relation_type.value}")
                print(f"    Subject: '{subject_entity.text}' ({subject_entity.category.value})")
                print(f"    Object: '{object_entity.text}' ({object_entity.category.value})")
                print(f"    Confidence: {relation.confidence:.2f}")
                print(f"    Method: {relation.extraction_method.value}")
                if relation.trigger_phrase:
                    print(f"    Trigger: '{relation.trigger_phrase}'")
                print(f"    Context: {relation.context_sentence[:100]}...")
    else:
        print(f"Preprocessing failed: {preprocessed['error']}")

async def demo_fact_extraction():
    """Demo: Structured fact extraction"""
    print("\nDEMO 4: STRUCTURED FACT EXTRACTION")
    print("=" * 50)
    
    preprocessor = TextPreprocessor()
    fact_extractor = FactExtractor()
    
    # Sample document with factual statements
    sample_doc = Document(
        id="demo_doc_4",
        title="Product Specifications", 
        content="""
        The iPhone 15 is a smartphone developed by Apple. It has a 6.1-inch display and costs $799. 
        The device was released in September 2023. iPhone 15 contains an A16 Bionic chip and measures 
        147.6 x 71.6 x 7.80 mm. The battery capacity is 3349 mAh. The phone has a titanium frame and 
        was manufactured in China. Apple developed the device over 3 years.
        """,
        doc_type=DocumentType.TEXT
    )
    
    print(f"Document: {sample_doc.title}")
    print(f"Content: {sample_doc.content.strip()}")
    
    # Preprocess and extract facts
    preprocessed = await preprocessor.preprocess_document(sample_doc)
    
    if preprocessed['success']:
        facts = await fact_extractor.extract_facts(preprocessed['sentences'], sample_doc.id)
        
        print(f"\nFact Extraction Results:")
        print(f"Total facts extracted: {len(facts)}")
        
        print(f"\nExtracted Facts:")
        for i, fact in enumerate(facts, 1):
            print(f"\n  Fact {i}:")
            print(f"    Subject: '{fact.subject}'")
            print(f"    Predicate: '{fact.predicate}'")
            print(f"    Object: '{fact.object}'")
            print(f"    Confidence: {fact.confidence:.2f}")
            print(f"    Method: {fact.extraction_method.value}")
            print(f"    Source: {fact.source_sentence[:80]}...")
        
        # Group facts by predicate
        facts_by_predicate = defaultdict(list)
        for fact in facts:
            facts_by_predicate[fact.predicate].append(fact)
        
        print(f"\nFacts by Predicate Type:")
        for predicate, predicate_facts in facts_by_predicate.items():
            print(f"  {predicate}: {len(predicate_facts)} facts")
    else:
        print(f"Preprocessing failed: {preprocessed['error']}")

async def demo_complete_pipeline():
    """Demo: Complete knowledge extraction pipeline"""
    print("\nDEMO 5: COMPLETE KNOWLEDGE EXTRACTION PIPELINE")
    print("=" * 50)
    
    pipeline = KnowledgeExtractionPipeline()
    await pipeline.initialize()
    
    # Sample documents for pipeline testing
    documents = [
        Document(
            id="",
            title="AI Research Breakthrough",
            content="""
            OpenAI announced GPT-4 in March 2023, a large language model that represents a significant 
            advancement in artificial intelligence. The model was trained on diverse internet text and 
            demonstrates human-level performance on various professional benchmarks. GPT-4 is similar 
            to previous models but shows improved reasoning capabilities. The research team at OpenAI 
            includes scientists from Stanford University and other institutions. The model costs 
            approximately $20 per 1M tokens for API usage.
            """,
            doc_type=DocumentType.TEXT,
            author="AI News",
            source="tech_news"
        ),
        Document(
            id="",
            title="Quantum Computing Development",
            content="""
            IBM developed a 1000-qubit quantum processor called Condor in 2023. The chip is part of 
            IBM's quantum computing research program. Dr. Jay Gambetta leads the quantum team at IBM 
            Research in Yorktown Heights, New York. Quantum computers use quantum mechanics principles 
            and are fundamentally different from classical computers. The Condor processor measures 
            quantum states and contains superconducting qubits. IBM's quantum division was founded 
            in 2016 and has invested over $2 billion in quantum research.
            """,
            doc_type=DocumentType.TEXT,
            author="Science Journal", 
            source="research_papers"
        ),
        Document(
            id="",
            title="Renewable Energy Statistics",
            content="""
            Solar energy capacity reached 1,177 gigawatts globally in 2022. The International Energy 
            Agency reported this growth in their annual renewable energy report. China leads solar 
            installation with 261 GW of total capacity. Solar panels convert sunlight to electricity 
            and typically have 20-25 year warranties. The cost of solar energy has decreased 90% 
            since 2010. Tesla develops solar roof tiles and battery storage systems for homes.
            """,
            doc_type=DocumentType.TEXT,
            author="Energy Analyst",
            source="energy_reports"
        )
    ]
    
    print(f"Processing {len(documents)} documents through complete pipeline...")
    
    # Process documents
    results = await pipeline.process_documents(documents)
    
    print(f"\nPipeline Processing Results:")
    
    for i, result in enumerate(results, 1):
        doc = documents[i-1]
        print(f"\n--- Document {i}: {doc.title} ---")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
            continue
        
        print(f"  Processing time: {result.processing_time:.3f}s")
        print(f"  Entities extracted: {len(result.entities)}")
        print(f"  Relations extracted: {len(result.relations)}")
        print(f"  Facts extracted: {len(result.facts)}")
        
        # Show confidence scores
        confidence = result.confidence_scores
        print(f"  Confidence scores:")
        print(f"    Entities: {confidence.get('entity_average', 0):.2f}")
        print(f"    Relations: {confidence.get('relation_average', 0):.2f}")
        print(f"    Facts: {confidence.get('fact_average', 0):.2f}")
        print(f"    Overall: {confidence.get('overall_average', 0):.2f}")
        
        # Show entity distribution
        if result.entity_count_by_category:
            print(f"  Entity distribution: {result.entity_count_by_category}")
        
        # Show relation distribution
        if result.relation_count_by_type:
            print(f"  Relation distribution: {result.relation_count_by_type}")
        
        # Show sample extractions
        if result.entities:
            sample_entities = result.entities[:3]
            print(f"  Sample entities:")
            for entity in sample_entities:
                print(f"    - '{entity.text}' ({entity.category.value})")
        
        if result.relations:
            print(f"  Sample relations:")
            for relation in result.relations[:2]:
                # Find entity texts
                subject_entity = next((e for e in result.entities if e.id == relation.subject_entity_id), None)
                object_entity = next((e for e in result.entities if e.id == relation.object_entity_id), None)
                
                if subject_entity and object_entity:
                    print(f"    - {subject_entity.text} --[{relation.relation_type.value}]--> {object_entity.text}")
        
        if result.facts:
            print(f"  Sample facts:")
            for fact in result.facts[:2]:
                print(f"    - {fact.subject} {fact.predicate} {fact.object}")
    
    # Show overall statistics
    stats = pipeline.get_statistics()
    processing_stats = stats['processing_statistics']
    
    print(f"\nPipeline Statistics:")
    print(f"  Documents processed: {processing_stats['documents_processed']}")
    print(f"  Total entities: {processing_stats['total_entities_extracted']}")
    print(f"  Total relations: {processing_stats['total_relations_extracted']}")
    print(f"  Total facts: {processing_stats['total_facts_extracted']}")
    print(f"  Average processing time: {processing_stats['average_processing_time']:.3f}s")
    
    config = stats['configuration']
    print(f"\nConfiguration:")
    print(f"  Extraction methods: {config['extraction_methods']}")
    print(f"  Min confidence threshold: {config['min_confidence_threshold']}")

async def main():
    """
    Demonstrate Knowledge Extraction Pipeline for automated knowledge discovery
    
    WHAT YOU'LL LEARN:
    ================
    1. How to preprocess text for optimal knowledge extraction
    2. How to extract entities using multiple methods (rules, ML, patterns)
    3. How to discover relationships between extracted entities
    4. How to extract structured facts from unstructured text
    5. How to build end-to-end knowledge extraction pipelines
    6. How to evaluate and optimize extraction quality
    
    REAL WORLD APPLICATIONS:
    =======================
    - Scientific literature analysis and discovery
    - Legal document processing and compliance
    - Medical record analysis and clinical research
    - Business intelligence and competitive analysis
    - News monitoring and event detection
    - Patent analysis and innovation tracking
    """
    
    print("KNOWLEDGE EXTRACTION PIPELINE DEMONSTRATION")
    print("Automated discovery of structured knowledge from unstructured text!")
    
    await demo_text_preprocessing()
    await demo_entity_extraction()
    await demo_relation_extraction()
    await demo_fact_extraction()
    await demo_complete_pipeline()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Text preprocessing enables effective knowledge extraction")
    print("✓ Entity extraction identifies key concepts and objects")
    print("✓ Relation extraction discovers connections between entities")
    print("✓ Fact extraction creates structured knowledge from text")
    print("✓ Complete pipelines enable scalable knowledge discovery")
    print("✓ Multiple extraction methods improve coverage and accuracy")
    print("\nTHE POWER OF AUTOMATED EXTRACTION:")
    print("- Processes thousands of documents in minutes")
    print("- Discovers hidden patterns and relationships")
    print("- Creates structured knowledge for AI systems")
    print("- Enables real-time knowledge updates")

if __name__ == "__main__":
    asyncio.run(main())
