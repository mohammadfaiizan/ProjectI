#!/usr/bin/env python3
"""
Semantic RAG Systems: Deep Understanding and Knowledge-Based Retrieval
=====================================================================

WHAT IS THE PROBLEM?
==================
Traditional RAG relies on surface-level similarity:
- Keyword matching misses conceptual relationships
- Cannot understand semantic intent and context
- Misses implicit knowledge and reasoning chains
- Cannot handle synonyms, abstractions, and concept hierarchies
- Limited understanding of domain knowledge structures
- Cannot reason about relationships between concepts

Example: Medical Query Complexity
KEYWORD RAG (Traditional):
- Query: "Treatment for chest pain"
- Retrieves: Documents containing exact words "chest pain"
- Misses: Myocardial infarction, angina, cardiac arrest documents
- Missing: Symptom-disease relationships, treatment protocols
- Result: Incomplete medical guidance

REAL WORLD EXAMPLE:
=================
How does IBM Watson work in healthcare?

WATSON'S SEMANTIC UNDERSTANDING:
1. CONCEPT EXTRACTION: Identifies medical concepts and relationships
2. ONTOLOGY MAPPING: Maps to medical knowledge structures (SNOMED, ICD)
3. REASONING CHAINS: Connects symptoms → diagnosis → treatments
4. EVIDENCE GRADING: Understands strength of medical evidence
5. CONTEXT AWARENESS: Considers patient context and contraindications
6. KNOWLEDGE GRAPHS: Leverages structured medical knowledge
7. SEMANTIC SEARCH: Finds conceptually related information

BENEFITS OF SEMANTIC RAG:
- Deep understanding beyond keyword matching
- Conceptual reasoning and knowledge inference
- Domain-specific expertise and structured knowledge
- Context-aware and relationship-based retrieval
- Advanced reasoning and explanation capabilities
- Professional-grade accuracy and completeness

THE SEMANTIC ADVANTAGE:
======================
TRADITIONAL RAG: Keywords → Text similarity → Surface matches
SEMANTIC RAG: Concepts → Knowledge graphs → Deep understanding → Intelligent retrieval

SEMANTIC COMPONENTS:
==================
1. CONCEPT EXTRACTION: Identify entities, concepts, and relationships
2. KNOWLEDGE GRAPHS: Structured representation of domain knowledge
3. ONTOLOGY REASONING: Leverage formal knowledge structures
4. SEMANTIC SIMILARITY: Meaning-based rather than text-based matching
5. INFERENCE ENGINES: Derive implicit knowledge through reasoning
6. CONTEXT UNDERSTANDING: Consider semantic context and intent
7. EXPLANATION GENERATION: Provide reasoning chains and justifications

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI to understand meaning, not just text
- Provides expert-level domain reasoning
- Supports complex knowledge work and decision making
- Powers next-generation intelligent assistants
- Critical for professional applications requiring deep understanding
- Bridges the gap between information retrieval and knowledge reasoning
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import re
import math
import networkx as nx
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ConceptType(Enum):
    """Types of concepts in semantic knowledge"""
    ENTITY = "entity"               # Concrete entities (person, place, thing)
    ABSTRACT = "abstract"           # Abstract concepts (democracy, intelligence)
    PROCESS = "process"             # Processes and actions (treatment, analysis)
    PROPERTY = "property"           # Properties and attributes (red, fast, expensive)
    RELATION = "relation"           # Relationships between concepts
    EVENT = "event"                 # Events and occurrences

class RelationType(Enum):
    """Types of semantic relationships"""
    IS_A = "is_a"                   # Taxonomic relationship (cat is_a animal)
    PART_OF = "part_of"             # Part-whole relationship (wheel part_of car)
    CAUSES = "causes"               # Causal relationship (virus causes disease)
    TREATS = "treats"               # Treatment relationship (medicine treats disease)
    ASSOCIATED_WITH = "associated_with"  # General association
    ENABLES = "enables"             # Enablement relationship
    REQUIRES = "requires"           # Requirement relationship
    SIMILAR_TO = "similar_to"       # Similarity relationship

class ReasoningType(Enum):
    """Types of semantic reasoning"""
    DEDUCTIVE = "deductive"         # From general to specific
    INDUCTIVE = "inductive"         # From specific to general
    ABDUCTIVE = "abductive"         # Best explanation inference
    ANALOGICAL = "analogical"       # Reasoning by analogy
    CAUSAL = "causal"              # Causal reasoning
    TEMPORAL = "temporal"           # Temporal reasoning

@dataclass
class SemanticConcept:
    """Semantic concept with rich metadata"""
    concept_id: str
    name: str
    concept_type: ConceptType
    
    # Concept properties
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    # Domain context
    domain: str = "general"
    ontology_uris: List[str] = field(default_factory=list)
    
    # Semantic properties
    abstraction_level: float = 0.5  # 0=concrete, 1=abstract
    importance_score: float = 0.5
    frequency_score: float = 0.5
    
    # Relationships
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.concept_id:
            self.concept_id = str(uuid.uuid4())

@dataclass
class SemanticRelation:
    """Semantic relationship between concepts"""
    relation_id: str
    source_concept: str
    target_concept: str
    relation_type: RelationType
    
    # Relationship properties
    strength: float = 1.0           # Relationship strength (0-1)
    confidence: float = 1.0         # Confidence in relationship
    bidirectional: bool = False     # Whether relationship works both ways
    
    # Context
    domain: str = "general"
    source: str = ""                # Source of relationship (ontology, extraction, etc.)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.relation_id:
            self.relation_id = str(uuid.uuid4())

@dataclass
class SemanticQuery:
    """Semantic query with concept-level understanding"""
    query_id: str
    original_text: str
    
    # Extracted concepts
    identified_concepts: List[str] = field(default_factory=list)
    concept_weights: Dict[str, float] = field(default_factory=dict)
    
    # Query intent
    query_intent: str = "search"    # search, explain, compare, analyze, etc.
    reasoning_type: Optional[ReasoningType] = None
    
    # Context
    domain_context: str = "general"
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic expansion
    expanded_concepts: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())

class ConceptExtractor:
    """Extracts semantic concepts from text"""
    
    def __init__(self):
        # Domain-specific concept patterns
        self.concept_patterns = {
            'medical': {
                'diseases': r'\b(?:diabetes|hypertension|cancer|pneumonia|influenza)\b',
                'symptoms': r'\b(?:fever|pain|headache|nausea|fatigue)\b',
                'treatments': r'\b(?:medication|surgery|therapy|treatment|prescription)\b',
                'anatomy': r'\b(?:heart|lung|brain|liver|kidney)\b'
            },
            'technology': {
                'ai_concepts': r'\b(?:machine learning|deep learning|neural network|algorithm)\b',
                'programming': r'\b(?:python|java|javascript|database|api)\b',
                'systems': r'\b(?:cloud|server|network|security|encryption)\b'
            },
            'business': {
                'finance': r'\b(?:revenue|profit|investment|market|analysis)\b',
                'strategy': r'\b(?:strategy|planning|growth|competitive|advantage)\b',
                'operations': r'\b(?:process|efficiency|optimization|management)\b'
            }
        }
        
        # Common concept indicators
        self.concept_indicators = {
            'entities': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'technical_terms': r'\b[a-z]+(?:-[a-z]+)*(?:tion|sion|ment|ness|ity)\b',
            'processes': r'\b(?:ing|ed)\s+[a-z]+\b'
        }
        
        self.logger = logging.getLogger("ConceptExtractor")
    
    async def extract_concepts(self, text: str, domain: str = "general") -> List[SemanticConcept]:
        """Extract semantic concepts from text"""
        
        try:
            concepts = []
            text_lower = text.lower()
            
            # Domain-specific extraction
            if domain in self.concept_patterns:
                domain_patterns = self.concept_patterns[domain]
                
                for category, pattern in domain_patterns.items():
                    matches = re.findall(pattern, text_lower)
                    
                    for match in matches:
                        concept = SemanticConcept(
                            concept_id="",
                            name=match,
                            concept_type=self._infer_concept_type(match, category),
                            domain=domain,
                            importance_score=self._calculate_importance(match, text),
                            frequency_score=text_lower.count(match) / len(text.split())
                        )
                        
                        # Add category-specific properties
                        concept.attributes['category'] = category
                        concept.attributes['extraction_method'] = 'pattern_matching'
                        
                        concepts.append(concept)
            
            # General concept extraction
            general_concepts = await self._extract_general_concepts(text)
            concepts.extend(general_concepts)
            
            # Remove duplicates and rank
            unique_concepts = self._deduplicate_concepts(concepts)
            ranked_concepts = self._rank_concepts(unique_concepts, text)
            
            self.logger.debug(f"Extracted {len(ranked_concepts)} concepts from text")
            
            return ranked_concepts
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
            return []
    
    async def _extract_general_concepts(self, text: str) -> List[SemanticConcept]:
        """Extract general concepts using linguistic patterns"""
        
        concepts = []
        
        # Extract entities (proper nouns)
        entity_matches = re.findall(self.concept_indicators['entities'], text)
        for entity in entity_matches:
            concept = SemanticConcept(
                concept_id="",
                name=entity,
                concept_type=ConceptType.ENTITY,
                domain="general",
                abstraction_level=0.2
            )
            concepts.append(concept)
        
        # Extract technical terms
        tech_matches = re.findall(self.concept_indicators['technical_terms'], text.lower())
        for term in tech_matches:
            if len(term) > 4:  # Filter short terms
                concept = SemanticConcept(
                    concept_id="",
                    name=term,
                    concept_type=ConceptType.ABSTRACT,
                    domain="general",
                    abstraction_level=0.7
                )
                concepts.append(concept)
        
        return concepts
    
    def _infer_concept_type(self, concept: str, category: str) -> ConceptType:
        """Infer concept type from category and content"""
        
        type_mapping = {
            'diseases': ConceptType.ABSTRACT,
            'symptoms': ConceptType.PROPERTY,
            'treatments': ConceptType.PROCESS,
            'anatomy': ConceptType.ENTITY,
            'ai_concepts': ConceptType.ABSTRACT,
            'programming': ConceptType.ENTITY,
            'systems': ConceptType.ENTITY,
            'finance': ConceptType.ABSTRACT,
            'strategy': ConceptType.ABSTRACT,
            'operations': ConceptType.PROCESS
        }
        
        return type_mapping.get(category, ConceptType.ABSTRACT)
    
    def _calculate_importance(self, concept: str, text: str) -> float:
        """Calculate importance score for concept"""
        
        # Factors affecting importance
        frequency = text.lower().count(concept.lower())
        position_weight = 1.0
        
        # Check if concept appears early in text (higher importance)
        first_occurrence = text.lower().find(concept.lower())
        if first_occurrence != -1:
            position_weight = 1.0 - (first_occurrence / len(text))
        
        # Length bias (longer concepts tend to be more specific/important)
        length_weight = min(1.0, len(concept) / 20.0)
        
        # Combine factors
        importance = (frequency * 0.4 + position_weight * 0.3 + length_weight * 0.3)
        
        return min(1.0, importance)
    
    def _deduplicate_concepts(self, concepts: List[SemanticConcept]) -> List[SemanticConcept]:
        """Remove duplicate concepts"""
        
        seen_names = set()
        unique_concepts = []
        
        for concept in concepts:
            concept_key = concept.name.lower().strip()
            
            if concept_key not in seen_names:
                seen_names.add(concept_key)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _rank_concepts(self, concepts: List[SemanticConcept], text: str) -> List[SemanticConcept]:
        """Rank concepts by relevance and importance"""
        
        # Calculate combined score
        for concept in concepts:
            concept.importance_score = (
                concept.importance_score * 0.6 +
                concept.frequency_score * 0.4
            )
        
        # Sort by importance
        concepts.sort(key=lambda x: x.importance_score, reverse=True)
        
        return concepts

class KnowledgeGraph:
    """Semantic knowledge graph for concept relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, SemanticConcept] = {}
        self.relations: Dict[str, SemanticRelation] = {}
        
        # Graph statistics
        self.concept_count = 0
        self.relation_count = 0
        
        self.logger = logging.getLogger("KnowledgeGraph")
    
    async def add_concept(self, concept: SemanticConcept) -> None:
        """Add concept to knowledge graph"""
        
        try:
            self.concepts[concept.concept_id] = concept
            
            # Add node to graph
            self.graph.add_node(
                concept.concept_id,
                name=concept.name,
                type=concept.concept_type.value,
                domain=concept.domain,
                importance=concept.importance_score
            )
            
            self.concept_count += 1
            
            self.logger.debug(f"Added concept: {concept.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add concept {concept.name}: {e}")
    
    async def add_relation(self, relation: SemanticRelation) -> None:
        """Add relationship to knowledge graph"""
        
        try:
            # Ensure both concepts exist
            if (relation.source_concept not in self.concepts or 
                relation.target_concept not in self.concepts):
                self.logger.warning(f"Missing concepts for relation {relation.relation_id}")
                return
            
            self.relations[relation.relation_id] = relation
            
            # Add edge to graph
            self.graph.add_edge(
                relation.source_concept,
                relation.target_concept,
                relation_type=relation.relation_type.value,
                strength=relation.strength,
                confidence=relation.confidence,
                bidirectional=relation.bidirectional
            )
            
            # Add reverse edge if bidirectional
            if relation.bidirectional:
                self.graph.add_edge(
                    relation.target_concept,
                    relation.source_concept,
                    relation_type=relation.relation_type.value,
                    strength=relation.strength,
                    confidence=relation.confidence,
                    bidirectional=True
                )
            
            self.relation_count += 1
            
            self.logger.debug(f"Added relation: {relation.relation_type.value} "
                            f"between {relation.source_concept} and {relation.target_concept}")
            
        except Exception as e:
            self.logger.error(f"Failed to add relation {relation.relation_id}: {e}")
    
    async def find_related_concepts(self, concept_id: str, 
                                  max_distance: int = 2,
                                  min_strength: float = 0.1) -> List[Tuple[str, float, str]]:
        """Find concepts related to given concept"""
        
        if concept_id not in self.graph:
            return []
        
        related = []
        
        try:
            # Use shortest path algorithms to find related concepts
            for target_id in self.graph.nodes():
                if target_id == concept_id:
                    continue
                
                try:
                    # Check if path exists
                    if nx.has_path(self.graph, concept_id, target_id):
                        path_length = nx.shortest_path_length(self.graph, concept_id, target_id)
                        
                        if path_length <= max_distance:
                            # Calculate relationship strength
                            strength = self._calculate_path_strength(concept_id, target_id)
                            
                            if strength >= min_strength:
                                # Get relationship type
                                rel_type = self._get_relationship_type(concept_id, target_id)
                                related.append((target_id, strength, rel_type))
                
                except nx.NetworkXNoPath:
                    continue
            
            # Sort by strength
            related.sort(key=lambda x: x[1], reverse=True)
            
            return related
            
        except Exception as e:
            self.logger.error(f"Failed to find related concepts: {e}")
            return []
    
    async def find_reasoning_path(self, source_concept: str, 
                                target_concept: str) -> List[Dict[str, Any]]:
        """Find reasoning path between concepts"""
        
        if (source_concept not in self.graph or 
            target_concept not in self.graph):
            return []
        
        try:
            if not nx.has_path(self.graph, source_concept, target_concept):
                return []
            
            # Get shortest path
            path = nx.shortest_path(self.graph, source_concept, target_concept)
            
            # Build reasoning chain
            reasoning_chain = []
            
            for i in range(len(path) - 1):
                current_id = path[i]
                next_id = path[i + 1]
                
                # Get edge data
                edge_data = self.graph.get_edge_data(current_id, next_id)
                
                current_concept = self.concepts.get(current_id)
                next_concept = self.concepts.get(next_id)
                
                if current_concept and next_concept and edge_data:
                    reasoning_step = {
                        'from_concept': current_concept.name,
                        'to_concept': next_concept.name,
                        'relation_type': edge_data.get('relation_type', 'related'),
                        'strength': edge_data.get('strength', 0.5),
                        'confidence': edge_data.get('confidence', 0.5)
                    }
                    reasoning_chain.append(reasoning_step)
            
            return reasoning_chain
            
        except Exception as e:
            self.logger.error(f"Failed to find reasoning path: {e}")
            return []
    
    async def expand_concept_query(self, concept_ids: List[str], 
                                 expansion_depth: int = 1) -> List[str]:
        """Expand query concepts with related concepts"""
        
        expanded = set(concept_ids)
        
        for concept_id in concept_ids:
            related = await self.find_related_concepts(
                concept_id, 
                max_distance=expansion_depth,
                min_strength=0.3
            )
            
            # Add high-strength related concepts
            for related_id, strength, rel_type in related[:5]:  # Top 5
                if strength > 0.5:
                    expanded.add(related_id)
        
        return list(expanded)
    
    def _calculate_path_strength(self, source: str, target: str) -> float:
        """Calculate strength of relationship path"""
        
        try:
            path = nx.shortest_path(self.graph, source, target)
            
            if len(path) < 2:
                return 0.0
            
            total_strength = 1.0
            
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    edge_strength = edge_data.get('strength', 0.5)
                    edge_confidence = edge_data.get('confidence', 0.5)
                    total_strength *= (edge_strength * edge_confidence)
            
            # Apply distance penalty
            distance_penalty = 1.0 / len(path)
            
            return total_strength * distance_penalty
            
        except:
            return 0.0
    
    def _get_relationship_type(self, source: str, target: str) -> str:
        """Get relationship type between concepts"""
        
        try:
            if self.graph.has_edge(source, target):
                edge_data = self.graph.get_edge_data(source, target)
                return edge_data.get('relation_type', 'related')
            else:
                # Find path and get first relation type
                path = nx.shortest_path(self.graph, source, target)
                if len(path) >= 2:
                    edge_data = self.graph.get_edge_data(path[0], path[1])
                    return edge_data.get('relation_type', 'related')
        except:
            pass
        
        return 'related'
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        
        return {
            'concept_count': self.concept_count,
            'relation_count': self.relation_count,
            'graph_density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected()) if self.graph.nodes() else 0,
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'concepts_by_domain': self._get_domain_distribution(),
            'concepts_by_type': self._get_type_distribution()
        }
    
    def _get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of concepts by domain"""
        
        domain_counts = defaultdict(int)
        
        for concept in self.concepts.values():
            domain_counts[concept.domain] += 1
        
        return dict(domain_counts)
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of concepts by type"""
        
        type_counts = defaultdict(int)
        
        for concept in self.concepts.values():
            type_counts[concept.concept_type.value] += 1
        
        return dict(type_counts)

class SemanticRetriever:
    """Semantic-aware document retrieval"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.document_concepts: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger("SemanticRetriever")
    
    async def add_document(self, doc_id: str, document: Dict[str, Any], 
                         concepts: List[SemanticConcept]) -> None:
        """Add document with associated concepts"""
        
        try:
            self.documents[doc_id] = document
            
            # Store document-concept associations
            concept_ids = []
            for concept in concepts:
                await self.knowledge_graph.add_concept(concept)
                concept_ids.append(concept.concept_id)
            
            self.document_concepts[doc_id] = concept_ids
            
            self.logger.debug(f"Added document {doc_id} with {len(concepts)} concepts")
            
        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {e}")
    
    async def semantic_search(self, query: SemanticQuery, 
                            top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform semantic search using concept relationships"""
        
        try:
            if not query.identified_concepts:
                self.logger.warning("No concepts identified in query")
                return []
            
            # Expand query concepts using knowledge graph
            expanded_concepts = await self.knowledge_graph.expand_concept_query(
                query.identified_concepts,
                expansion_depth=1
            )
            
            # Score documents based on semantic similarity
            doc_scores = []
            
            for doc_id, doc_concept_ids in self.document_concepts.items():
                score = await self._calculate_semantic_score(
                    expanded_concepts,
                    doc_concept_ids,
                    query
                )
                
                if score > 0:
                    # Get reasoning explanation
                    explanation = await self._generate_relevance_explanation(
                        query.identified_concepts,
                        doc_concept_ids
                    )
                    
                    doc_scores.append((doc_id, score, explanation))
            
            # Sort by score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results with documents
            results = []
            for doc_id, score, explanation in doc_scores[:top_k]:
                if doc_id in self.documents:
                    doc_copy = self.documents[doc_id].copy()
                    doc_copy['semantic_score'] = score
                    doc_copy['relevance_explanation'] = explanation
                    results.append((doc_id, score, doc_copy))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _calculate_semantic_score(self, query_concepts: List[str], 
                                      doc_concepts: List[str],
                                      query: SemanticQuery) -> float:
        """Calculate semantic similarity score"""
        
        if not query_concepts or not doc_concepts:
            return 0.0
        
        total_score = 0.0
        
        for query_concept in query_concepts:
            query_weight = query.concept_weights.get(query_concept, 1.0)
            best_match_score = 0.0
            
            for doc_concept in doc_concepts:
                # Direct match
                if query_concept == doc_concept:
                    match_score = 1.0
                else:
                    # Semantic similarity through knowledge graph
                    match_score = await self._calculate_concept_similarity(
                        query_concept, doc_concept
                    )
                
                best_match_score = max(best_match_score, match_score)
            
            total_score += query_weight * best_match_score
        
        # Normalize by number of query concepts
        return total_score / len(query_concepts)
    
    async def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concepts"""
        
        try:
            # Find relationship path
            path = await self.knowledge_graph.find_reasoning_path(concept1, concept2)
            
            if not path:
                return 0.0
            
            # Calculate similarity based on path strength and length
            path_strength = 1.0
            for step in path:
                step_strength = step['strength'] * step['confidence']
                path_strength *= step_strength
            
            # Distance penalty
            distance_penalty = 1.0 / (len(path) + 1)
            
            similarity = path_strength * distance_penalty
            
            return min(1.0, similarity)
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate concept similarity: {e}")
            return 0.0
    
    async def _generate_relevance_explanation(self, query_concepts: List[str], 
                                            doc_concepts: List[str]) -> Dict[str, Any]:
        """Generate explanation for document relevance"""
        
        explanations = []
        concept_matches = []
        
        for query_concept in query_concepts:
            query_concept_obj = self.knowledge_graph.concepts.get(query_concept)
            if not query_concept_obj:
                continue
            
            for doc_concept in doc_concepts:
                doc_concept_obj = self.knowledge_graph.concepts.get(doc_concept)
                if not doc_concept_obj:
                    continue
                
                # Direct match
                if query_concept == doc_concept:
                    concept_matches.append({
                        'query_concept': query_concept_obj.name,
                        'doc_concept': doc_concept_obj.name,
                        'match_type': 'direct',
                        'strength': 1.0
                    })
                else:
                    # Semantic relationship
                    reasoning_path = await self.knowledge_graph.find_reasoning_path(
                        query_concept, doc_concept
                    )
                    
                    if reasoning_path:
                        path_description = " → ".join([
                            f"{step['from_concept']} {step['relation_type']} {step['to_concept']}"
                            for step in reasoning_path
                        ])
                        
                        explanations.append({
                            'reasoning_chain': path_description,
                            'strength': min(step['strength'] for step in reasoning_path),
                            'path_length': len(reasoning_path)
                        })
        
        return {
            'concept_matches': concept_matches,
            'reasoning_chains': explanations,
            'total_matches': len(concept_matches),
            'semantic_connections': len(explanations)
        }

class SemanticRAGSystem:
    """
    Complete Semantic RAG System with deep concept understanding
    
    EXAMPLE USAGE:
    =============
    # Create semantic RAG system
    rag = SemanticRAGSystem()
    await rag.initialize()
    
    # Add domain knowledge
    await rag.add_domain_knowledge("medical", {
        "concepts": [
            {"name": "myocardial_infarction", "type": "disease", "synonyms": ["heart_attack"]},
            {"name": "chest_pain", "type": "symptom"},
            {"name": "aspirin", "type": "medication"}
        ],
        "relations": [
            {"source": "myocardial_infarction", "target": "chest_pain", "type": "causes"},
            {"source": "aspirin", "target": "myocardial_infarction", "type": "treats"}
        ]
    })
    
    # Add documents
    await rag.add_document("doc1", {
        "title": "Heart Attack Treatment Guidelines",
        "content": "Myocardial infarction requires immediate treatment with aspirin..."
    })
    
    # Semantic search
    result = await rag.semantic_search("treatment for chest pain")
    
    print(f"Found {len(result['documents'])} semantically relevant documents")
    print(f"Reasoning: {result['semantic_explanation']}")
    """
    
    def __init__(self):
        # Core components
        self.concept_extractor = ConceptExtractor()
        self.knowledge_graph = KnowledgeGraph()
        self.semantic_retriever = SemanticRetriever(self.knowledge_graph)
        
        # System statistics
        self.system_stats = {
            'documents_processed': 0,
            'concepts_extracted': 0,
            'relations_created': 0,
            'semantic_queries': 0,
            'reasoning_chains_generated': 0,
            'average_reasoning_depth': 0.0
        }
        
        self.logger = logging.getLogger("SemanticRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize semantic RAG system"""
        
        # Load base ontologies and knowledge
        await self._load_base_knowledge()
        
        self.logger.info("Semantic RAG system initialized")
    
    async def add_domain_knowledge(self, domain: str, 
                                 knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add structured domain knowledge"""
        
        try:
            concepts_added = 0
            relations_added = 0
            
            # Add concepts
            if 'concepts' in knowledge_data:
                for concept_data in knowledge_data['concepts']:
                    concept = SemanticConcept(
                        concept_id="",
                        name=concept_data['name'],
                        concept_type=ConceptType(concept_data.get('type', 'abstract')),
                        domain=domain,
                        definition=concept_data.get('definition', ''),
                        synonyms=concept_data.get('synonyms', []),
                        importance_score=concept_data.get('importance', 0.5)
                    )
                    
                    await self.knowledge_graph.add_concept(concept)
                    concepts_added += 1
            
            # Add relations
            if 'relations' in knowledge_data:
                for relation_data in knowledge_data['relations']:
                    # Find concept IDs by name
                    source_id = self._find_concept_id_by_name(relation_data['source'])
                    target_id = self._find_concept_id_by_name(relation_data['target'])
                    
                    if source_id and target_id:
                        relation = SemanticRelation(
                            relation_id="",
                            source_concept=source_id,
                            target_concept=target_id,
                            relation_type=RelationType(relation_data['type']),
                            domain=domain,
                            strength=relation_data.get('strength', 1.0),
                            confidence=relation_data.get('confidence', 1.0)
                        )
                        
                        await self.knowledge_graph.add_relation(relation)
                        relations_added += 1
            
            # Update statistics
            self.system_stats['concepts_extracted'] += concepts_added
            self.system_stats['relations_created'] += relations_added
            
            result = {
                'success': True,
                'domain': domain,
                'concepts_added': concepts_added,
                'relations_added': relations_added
            }
            
            self.logger.info(f"Added domain knowledge for {domain}: "
                           f"{concepts_added} concepts, {relations_added} relations")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to add domain knowledge: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def add_document(self, doc_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Add document with semantic concept extraction"""
        
        start_time = time.time()
        
        try:
            # Extract concepts from document
            text_content = document.get('content', '') + ' ' + document.get('title', '')
            domain = document.get('domain', 'general')
            
            concepts = await self.concept_extractor.extract_concepts(text_content, domain)
            
            # Add document to semantic retriever
            await self.semantic_retriever.add_document(doc_id, document, concepts)
            
            # Create inter-concept relationships based on co-occurrence
            await self._create_cooccurrence_relations(concepts, domain)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.system_stats['documents_processed'] += 1
            self.system_stats['concepts_extracted'] += len(concepts)
            
            result = {
                'success': True,
                'document_id': doc_id,
                'concepts_extracted': len(concepts),
                'processing_time': processing_time,
                'extracted_concepts': [
                    {
                        'name': concept.name,
                        'type': concept.concept_type.value,
                        'importance': concept.importance_score
                    }
                    for concept in concepts[:10]  # Top 10
                ]
            }
            
            self.logger.info(f"Added document {doc_id} with {len(concepts)} concepts")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def semantic_search(self, query_text: str, 
                            domain: str = "general",
                            top_k: int = 10) -> Dict[str, Any]:
        """Perform semantic search with concept understanding"""
        
        start_time = time.time()
        self.system_stats['semantic_queries'] += 1
        
        try:
            # Extract concepts from query
            query_concepts = await self.concept_extractor.extract_concepts(query_text, domain)
            
            # Create semantic query
            semantic_query = SemanticQuery(
                query_id="",
                original_text=query_text,
                identified_concepts=[c.concept_id for c in query_concepts],
                concept_weights={c.concept_id: c.importance_score for c in query_concepts},
                domain_context=domain
            )
            
            # Determine query intent
            semantic_query.query_intent = self._analyze_query_intent(query_text)
            
            # Perform semantic search
            search_results = await self.semantic_retriever.semantic_search(semantic_query, top_k)
            
            # Generate semantic explanation
            semantic_explanation = await self._generate_semantic_explanation(
                semantic_query, search_results
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'query': query_text,
                'identified_concepts': [
                    {
                        'name': concept.name,
                        'type': concept.concept_type.value,
                        'importance': concept.importance_score
                    }
                    for concept in query_concepts
                ],
                'query_intent': semantic_query.query_intent,
                'documents_found': len(search_results),
                'documents': [
                    {
                        'document_id': doc_id,
                        'semantic_score': score,
                        'document': doc_data,
                        'relevance_explanation': doc_data.get('relevance_explanation', {})
                    }
                    for doc_id, score, doc_data in search_results
                ],
                'semantic_explanation': semantic_explanation,
                'processing_time': processing_time
            }
            
            self.logger.info(f"Semantic search completed: {len(search_results)} results in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def explain_reasoning(self, query_text: str, document_id: str) -> Dict[str, Any]:
        """Explain semantic reasoning for query-document relevance"""
        
        try:
            # Extract query concepts
            query_concepts = await self.concept_extractor.extract_concepts(query_text)
            query_concept_ids = [c.concept_id for c in query_concepts]
            
            # Get document concepts
            doc_concept_ids = self.semantic_retriever.document_concepts.get(document_id, [])
            
            if not query_concept_ids or not doc_concept_ids:
                return {
                    'success': False,
                    'error': 'Insufficient concepts for reasoning explanation'
                }
            
            # Generate detailed reasoning chains
            reasoning_chains = []
            
            for query_concept_id in query_concept_ids:
                for doc_concept_id in doc_concept_ids:
                    reasoning_path = await self.knowledge_graph.find_reasoning_path(
                        query_concept_id, doc_concept_id
                    )
                    
                    if reasoning_path:
                        query_concept = self.knowledge_graph.concepts.get(query_concept_id)
                        doc_concept = self.knowledge_graph.concepts.get(doc_concept_id)
                        
                        if query_concept and doc_concept:
                            reasoning_chains.append({
                                'from_query_concept': query_concept.name,
                                'to_document_concept': doc_concept.name,
                                'reasoning_steps': reasoning_path,
                                'path_strength': min(step['strength'] for step in reasoning_path),
                                'explanation': self._generate_reasoning_explanation(reasoning_path)
                            })
            
            # Update statistics
            self.system_stats['reasoning_chains_generated'] += len(reasoning_chains)
            
            if reasoning_chains:
                avg_depth = sum(len(chain['reasoning_steps']) for chain in reasoning_chains) / len(reasoning_chains)
                current_avg = self.system_stats['average_reasoning_depth']
                total_chains = self.system_stats['reasoning_chains_generated']
                
                self.system_stats['average_reasoning_depth'] = (
                    (current_avg * (total_chains - len(reasoning_chains)) + avg_depth * len(reasoning_chains)) / total_chains
                )
            
            return {
                'success': True,
                'query': query_text,
                'document_id': document_id,
                'reasoning_chains': reasoning_chains,
                'total_reasoning_paths': len(reasoning_chains)
            }
            
        except Exception as e:
            self.logger.error(f"Reasoning explanation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _load_base_knowledge(self) -> None:
        """Load base ontological knowledge"""
        
        # Basic conceptual relationships
        base_concepts = [
            {"name": "entity", "type": "abstract"},
            {"name": "process", "type": "abstract"},
            {"name": "property", "type": "abstract"},
            {"name": "relation", "type": "abstract"}
        ]
        
        base_relations = [
            {"source": "entity", "target": "property", "type": "associated_with"},
            {"source": "process", "target": "entity", "type": "involves"}
        ]
        
        # Add base knowledge
        await self.add_domain_knowledge("base", {
            "concepts": base_concepts,
            "relations": base_relations
        })
    
    async def _create_cooccurrence_relations(self, concepts: List[SemanticConcept], 
                                           domain: str) -> None:
        """Create relationships based on concept co-occurrence"""
        
        # Create associated_with relations for concepts appearing together
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Calculate association strength based on importance and proximity
                strength = (concept1.importance_score + concept2.importance_score) / 2
                
                if strength > 0.3:  # Minimum threshold
                    relation = SemanticRelation(
                        relation_id="",
                        source_concept=concept1.concept_id,
                        target_concept=concept2.concept_id,
                        relation_type=RelationType.ASSOCIATED_WITH,
                        domain=domain,
                        strength=strength,
                        confidence=0.7,
                        bidirectional=True,
                        source="cooccurrence"
                    )
                    
                    await self.knowledge_graph.add_relation(relation)
                    self.system_stats['relations_created'] += 1
    
    def _find_concept_id_by_name(self, name: str) -> Optional[str]:
        """Find concept ID by name"""
        
        for concept_id, concept in self.knowledge_graph.concepts.items():
            if concept.name.lower() == name.lower():
                return concept_id
            
            # Check synonyms
            if name.lower() in [syn.lower() for syn in concept.synonyms]:
                return concept_id
        
        return None
    
    def _analyze_query_intent(self, query_text: str) -> str:
        """Analyze query intent"""
        
        query_lower = query_text.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'definition', 'meaning']):
            return 'explain'
        elif any(word in query_lower for word in ['how', 'process', 'steps', 'procedure']):
            return 'process'
        elif any(word in query_lower for word in ['compare', 'versus', 'difference', 'similar']):
            return 'compare'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'explain_causation'
        elif any(word in query_lower for word in ['find', 'search', 'show', 'list']):
            return 'search'
        else:
            return 'search'
    
    async def _generate_semantic_explanation(self, query: SemanticQuery, 
                                           search_results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate explanation of semantic search process"""
        
        concept_names = []
        for concept_id in query.identified_concepts:
            concept = self.knowledge_graph.concepts.get(concept_id)
            if concept:
                concept_names.append(concept.name)
        
        # Analyze result diversity
        relevance_scores = [score for _, score, _ in search_results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return {
            'query_concepts_identified': concept_names,
            'query_intent': query.query_intent,
            'semantic_expansion_used': len(query.expanded_concepts) > len(query.identified_concepts),
            'average_semantic_score': avg_relevance,
            'reasoning_depth': self.system_stats['average_reasoning_depth'],
            'knowledge_graph_utilized': True
        }
    
    def _generate_reasoning_explanation(self, reasoning_path: List[Dict[str, Any]]) -> str:
        """Generate human-readable reasoning explanation"""
        
        if not reasoning_path:
            return "Direct concept match"
        
        explanations = []
        for step in reasoning_path:
            explanation = f"{step['from_concept']} {step['relation_type'].replace('_', ' ')} {step['to_concept']}"
            explanations.append(explanation)
        
        return " → ".join(explanations)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        kg_stats = self.knowledge_graph.get_graph_statistics()
        
        return {
            'system_stats': self.system_stats,
            'knowledge_graph_stats': kg_stats,
            'capabilities': {
                'concept_extraction': True,
                'semantic_reasoning': True,
                'knowledge_graph': True,
                'ontology_support': True,
                'reasoning_explanation': True,
                'multi_domain': True
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_concept_extraction():
    """Demo: Semantic concept extraction"""
    print("\nDEMO 1: SEMANTIC CONCEPT EXTRACTION")
    print("=" * 50)
    
    extractor = ConceptExtractor()
    
    # Test texts from different domains
    test_texts = [
        {
            'domain': 'medical',
            'text': "Patient presents with chest pain and shortness of breath. Symptoms suggest possible myocardial infarction. Recommended treatment includes aspirin and immediate cardiac monitoring."
        },
        {
            'domain': 'technology',
            'text': "Machine learning algorithms can improve prediction accuracy in artificial intelligence systems. Deep learning neural networks are particularly effective for computer vision applications."
        },
        {
            'domain': 'business',
            'text': "Our quarterly revenue analysis shows significant growth in market share. The strategic planning process identified competitive advantages that drive profitability improvement."
        }
    ]
    
    print("Extracting semantic concepts from different domains:")
    
    for i, test_case in enumerate(test_texts, 1):
        print(f"\n--- {test_case['domain'].title()} Domain ---")
        print(f"Text: {test_case['text'][:80]}...")
        
        concepts = await extractor.extract_concepts(test_case['text'], test_case['domain'])
        
        print(f"Extracted {len(concepts)} concepts:")
        for j, concept in enumerate(concepts[:5], 1):  # Show top 5
            print(f"  {j}. {concept.name}")
            print(f"     Type: {concept.concept_type.value}")
            print(f"     Importance: {concept.importance_score:.2f}")
            print(f"     Frequency: {concept.frequency_score:.2f}")
            if concept.attributes:
                print(f"     Category: {concept.attributes.get('category', 'N/A')}")

async def demo_knowledge_graph():
    """Demo: Knowledge graph construction and reasoning"""
    print("\nDEMO 2: KNOWLEDGE GRAPH AND REASONING")
    print("=" * 50)
    
    kg = KnowledgeGraph()
    
    # Create sample medical concepts
    concepts = [
        SemanticConcept("heart_attack", "myocardial infarction", ConceptType.ABSTRACT, domain="medical"),
        SemanticConcept("chest_pain", "chest pain", ConceptType.PROPERTY, domain="medical"),
        SemanticConcept("aspirin", "aspirin", ConceptType.ENTITY, domain="medical"),
        SemanticConcept("cardiac_enzyme", "cardiac enzyme", ConceptType.ENTITY, domain="medical"),
        SemanticConcept("ecg", "electrocardiogram", ConceptType.PROCESS, domain="medical")
    ]
    
    # Add concepts to knowledge graph
    print("Building medical knowledge graph:")
    for concept in concepts:
        await kg.add_concept(concept)
        print(f"  ✓ Added concept: {concept.name}")
    
    # Create relationships
    relations = [
        SemanticRelation("", "heart_attack", "chest_pain", RelationType.CAUSES, strength=0.9),
        SemanticRelation("", "aspirin", "heart_attack", RelationType.TREATS, strength=0.8),
        SemanticRelation("", "heart_attack", "cardiac_enzyme", RelationType.ASSOCIATED_WITH, strength=0.9),
        SemanticRelation("", "ecg", "heart_attack", RelationType.ASSOCIATED_WITH, strength=0.85),
        SemanticRelation("", "chest_pain", "cardiac_enzyme", RelationType.ASSOCIATED_WITH, strength=0.6)
    ]
    
    print(f"\nAdding {len(relations)} relationships:")
    for relation in relations:
        await kg.add_relation(relation)
        source_name = kg.concepts[relation.source_concept].name
        target_name = kg.concepts[relation.target_concept].name
        print(f"  ✓ {source_name} {relation.relation_type.value} {target_name} (strength: {relation.strength})")
    
    # Test concept relationships
    print(f"\nTesting concept relationships:")
    
    test_concept = "chest_pain"
    related = await kg.find_related_concepts(test_concept, max_distance=2)
    
    print(f"Concepts related to '{kg.concepts[test_concept].name}':")
    for related_id, strength, rel_type in related:
        related_name = kg.concepts[related_id].name
        print(f"  - {related_name}: {rel_type} (strength: {strength:.2f})")
    
    # Test reasoning paths
    print(f"\nTesting reasoning paths:")
    
    reasoning_path = await kg.find_reasoning_path("chest_pain", "aspirin")
    if reasoning_path:
        print(f"Reasoning from chest pain to aspirin:")
        for step in reasoning_path:
            print(f"  {step['from_concept']} {step['relation_type']} {step['to_concept']} "
                  f"(strength: {step['strength']:.2f})")
    else:
        print("No reasoning path found")
    
    # Show graph statistics
    stats = kg.get_graph_statistics()
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Concepts: {stats['concept_count']}")
    print(f"  Relations: {stats['relation_count']}")
    print(f"  Graph density: {stats['graph_density']:.3f}")
    print(f"  Connected components: {stats['connected_components']}")

async def demo_semantic_search():
    """Demo: Semantic search with concept understanding"""
    print("\nDEMO 3: SEMANTIC SEARCH")
    print("=" * 50)
    
    rag_system = SemanticRAGSystem()
    await rag_system.initialize()
    
    # Add medical domain knowledge
    medical_knowledge = {
        "concepts": [
            {"name": "myocardial_infarction", "type": "abstract", "synonyms": ["heart_attack", "MI"], "importance": 0.9},
            {"name": "chest_pain", "type": "property", "importance": 0.7},
            {"name": "aspirin", "type": "entity", "importance": 0.8},
            {"name": "cardiac_arrest", "type": "abstract", "importance": 0.9},
            {"name": "angina", "type": "abstract", "synonyms": ["chest_discomfort"], "importance": 0.7}
        ],
        "relations": [
            {"source": "myocardial_infarction", "target": "chest_pain", "type": "causes", "strength": 0.9},
            {"source": "angina", "target": "chest_pain", "type": "causes", "strength": 0.8},
            {"source": "aspirin", "target": "myocardial_infarction", "type": "treats", "strength": 0.8},
            {"source": "myocardial_infarction", "target": "cardiac_arrest", "type": "associated_with", "strength": 0.7}
        ]
    }
    
    print("Adding medical domain knowledge...")
    knowledge_result = await rag_system.add_domain_knowledge("medical", medical_knowledge)
    print(f"  ✓ Added {knowledge_result['concepts_added']} concepts")
    print(f"  ✓ Added {knowledge_result['relations_added']} relations")
    
    # Add medical documents
    medical_documents = [
        {
            "id": "heart_attack_guide",
            "document": {
                "title": "Heart Attack Treatment Guidelines",
                "content": "Myocardial infarction is a serious cardiac emergency requiring immediate medical attention. Primary treatment includes aspirin administration and cardiac monitoring.",
                "domain": "medical"
            }
        },
        {
            "id": "chest_pain_diagnosis",
            "document": {
                "title": "Chest Pain Diagnostic Protocols",
                "content": "Chest pain can indicate various cardiac conditions including angina and myocardial infarction. Proper diagnosis requires ECG and cardiac enzyme testing.",
                "domain": "medical"
            }
        },
        {
            "id": "aspirin_therapy",
            "document": {
                "title": "Aspirin in Cardiac Care",
                "content": "Aspirin therapy is a cornerstone treatment for acute myocardial infarction and secondary prevention of cardiac events.",
                "domain": "medical"
            }
        }
    ]
    
    print(f"\nAdding {len(medical_documents)} medical documents:")
    for doc_data in medical_documents:
        result = await rag_system.add_document(doc_data["id"], doc_data["document"])
        if result['success']:
            print(f"  ✓ Added: {doc_data['document']['title']}")
            print(f"    Concepts: {result['concepts_extracted']}")
    
    # Test semantic searches
    test_queries = [
        "treatment for chest pain",          # Should find heart attack and aspirin docs
        "heart attack medication",           # Should emphasize aspirin therapy
        "cardiac emergency protocols",       # Should find treatment guidelines
        "chest discomfort diagnosis"         # Should find chest pain diagnostic doc (synonym matching)
    ]
    
    print(f"\nTesting semantic search:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        result = await rag_system.semantic_search(query, domain="medical", top_k=3)
        
        if result['success']:
            print(f"Identified concepts: {', '.join([c['name'] for c in result['identified_concepts']])}")
            print(f"Query intent: {result['query_intent']}")
            print(f"Documents found: {result['documents_found']}")
            
            for j, doc_result in enumerate(result['documents'], 1):
                print(f"  {j}. {doc_result['document']['title']}")
                print(f"     Semantic score: {doc_result['semantic_score']:.3f}")
                explanation = doc_result['relevance_explanation']
                print(f"     Concept matches: {explanation['total_matches']}")
                print(f"     Reasoning chains: {explanation['semantic_connections']}")
        else:
            print(f"Search failed: {result['error']}")

async def demo_reasoning_explanation():
    """Demo: Semantic reasoning explanation"""
    print("\nDEMO 4: SEMANTIC REASONING EXPLANATION")
    print("=" * 50)
    
    rag_system = SemanticRAGSystem()
    await rag_system.initialize()
    
    # Build comprehensive knowledge base
    comprehensive_knowledge = {
        "concepts": [
            {"name": "cardiovascular_disease", "type": "abstract", "importance": 0.9},
            {"name": "myocardial_infarction", "type": "abstract", "synonyms": ["heart_attack"], "importance": 0.9},
            {"name": "chest_pain", "type": "property", "importance": 0.7},
            {"name": "aspirin", "type": "entity", "importance": 0.8},
            {"name": "antiplatelet_therapy", "type": "process", "importance": 0.8},
            {"name": "blood_clot", "type": "entity", "importance": 0.7},
            {"name": "coronary_artery", "type": "entity", "importance": 0.8}
        ],
        "relations": [
            {"source": "myocardial_infarction", "target": "cardiovascular_disease", "type": "is_a", "strength": 1.0},
            {"source": "myocardial_infarction", "target": "chest_pain", "type": "causes", "strength": 0.9},
            {"source": "blood_clot", "target": "myocardial_infarction", "type": "causes", "strength": 0.8},
            {"source": "blood_clot", "target": "coronary_artery", "type": "associated_with", "strength": 0.9},
            {"source": "aspirin", "target": "antiplatelet_therapy", "type": "enables", "strength": 0.9},
            {"source": "antiplatelet_therapy", "target": "blood_clot", "type": "treats", "strength": 0.8},
            {"source": "aspirin", "target": "myocardial_infarction", "type": "treats", "strength": 0.8}
        ]
    }
    
    print("Building comprehensive medical knowledge base...")
    await rag_system.add_domain_knowledge("medical", comprehensive_knowledge)
    
    # Add document with complex relationships
    complex_doc = {
        "title": "Comprehensive Cardiac Care Protocol",
        "content": "Cardiovascular disease management requires understanding of coronary artery pathophysiology. Blood clots in coronary arteries can cause myocardial infarction, presenting with chest pain. Aspirin provides antiplatelet therapy to prevent clot formation.",
        "domain": "medical"
    }
    
    doc_result = await rag_system.add_document("complex_cardiac", complex_doc)
    print(f"Added complex document with {doc_result['concepts_extracted']} concepts")
    
    # Test reasoning explanation
    test_query = "medication for chest pain"
    
    print(f"\nExplaining reasoning for query: '{test_query}'")
    
    # First perform search to see results
    search_result = await rag_system.semantic_search(test_query, domain="medical")
    
    if search_result['success'] and search_result['documents']:
        document_id = search_result['documents'][0]['document_id']
        
        # Get detailed reasoning explanation
        explanation_result = await rag_system.explain_reasoning(test_query, document_id)
        
        if explanation_result['success']:
            print(f"\nReasoning explanation for document: {document_id}")
            print(f"Total reasoning paths: {explanation_result['total_reasoning_paths']}")
            
            for i, chain in enumerate(explanation_result['reasoning_chains'], 1):
                print(f"\n  Reasoning Chain {i}:")
                print(f"    From query concept: {chain['from_query_concept']}")
                print(f"    To document concept: {chain['to_document_concept']}")
                print(f"    Path strength: {chain['path_strength']:.2f}")
                print(f"    Reasoning: {chain['explanation']}")
                
                print(f"    Detailed steps:")
                for j, step in enumerate(chain['reasoning_steps'], 1):
                    print(f"      {j}. {step['from_concept']} {step['relation_type']} {step['to_concept']}")
                    print(f"         (strength: {step['strength']:.2f}, confidence: {step['confidence']:.2f})")
        else:
            print(f"Reasoning explanation failed: {explanation_result['error']}")

async def demo_system_analytics():
    """Demo: Semantic RAG system analytics"""
    print("\nDEMO 5: SYSTEM ANALYTICS")
    print("=" * 50)
    
    rag_system = SemanticRAGSystem()
    await rag_system.initialize()
    
    # Build multi-domain knowledge base
    domains_data = {
        "medical": {
            "concepts": [
                {"name": "diabetes", "type": "abstract", "importance": 0.9},
                {"name": "insulin", "type": "entity", "importance": 0.8},
                {"name": "blood_sugar", "type": "property", "importance": 0.7}
            ],
            "relations": [
                {"source": "insulin", "target": "diabetes", "type": "treats", "strength": 0.9},
                {"source": "diabetes", "target": "blood_sugar", "type": "affects", "strength": 0.8}
            ]
        },
        "technology": {
            "concepts": [
                {"name": "artificial_intelligence", "type": "abstract", "importance": 0.9},
                {"name": "machine_learning", "type": "process", "importance": 0.8},
                {"name": "neural_network", "type": "entity", "importance": 0.8}
            ],
            "relations": [
                {"source": "machine_learning", "target": "artificial_intelligence", "type": "is_a", "strength": 0.9},
                {"source": "neural_network", "target": "machine_learning", "type": "enables", "strength": 0.8}
            ]
        },
        "business": {
            "concepts": [
                {"name": "market_analysis", "type": "process", "importance": 0.8},
                {"name": "competitive_advantage", "type": "abstract", "importance": 0.8},
                {"name": "revenue_growth", "type": "property", "importance": 0.7}
            ],
            "relations": [
                {"source": "market_analysis", "target": "competitive_advantage", "type": "enables", "strength": 0.7},
                {"source": "competitive_advantage", "target": "revenue_growth", "type": "causes", "strength": 0.8}
            ]
        }
    }
    
    print("Building multi-domain knowledge base...")
    
    total_concepts = 0
    total_relations = 0
    
    for domain, knowledge in domains_data.items():
        result = await rag_system.add_domain_knowledge(domain, knowledge)
        total_concepts += result['concepts_added']
        total_relations += result['relations_added']
        print(f"  ✓ {domain}: {result['concepts_added']} concepts, {result['relations_added']} relations")
    
    # Add diverse documents
    sample_documents = [
        {
            "id": "diabetes_management",
            "document": {
                "title": "Diabetes Management with Insulin",
                "content": "Type 1 diabetes requires insulin therapy to regulate blood sugar levels effectively.",
                "domain": "medical"
            }
        },
        {
            "id": "ai_healthcare",
            "document": {
                "title": "AI in Healthcare Applications",
                "content": "Artificial intelligence and machine learning are revolutionizing medical diagnosis and treatment planning.",
                "domain": "technology"
            }
        },
        {
            "id": "market_strategy",
            "document": {
                "title": "Strategic Market Analysis",
                "content": "Comprehensive market analysis provides competitive advantage leading to sustainable revenue growth.",
                "domain": "business"
            }
        },
        {
            "id": "cross_domain",
            "document": {
                "title": "AI-Powered Business Intelligence",
                "content": "Machine learning algorithms enable advanced market analysis and competitive intelligence for business growth.",
                "domain": "technology"
            }
        }
    ]
    
    print(f"\nAdding {len(sample_documents)} diverse documents:")
    
    document_results = []
    for doc_data in sample_documents:
        result = await rag_system.add_document(doc_data["id"], doc_data["document"])
        document_results.append(result)
        if result['success']:
            print(f"  ✓ {doc_data['document']['title']}: {result['concepts_extracted']} concepts")
    
    # Perform multiple semantic searches
    test_searches = [
        "diabetes treatment options",
        "AI machine learning applications",
        "business growth strategies",
        "artificial intelligence in healthcare",
        "market competitive analysis"
    ]
    
    print(f"\nPerforming {len(test_searches)} semantic searches:")
    
    search_results = []
    for query in test_searches:
        result = await rag_system.semantic_search(query)
        search_results.append(result)
        print(f"  ✓ '{query}': {result['documents_found']} results")
    
    # Get comprehensive analytics
    stats = rag_system.get_system_statistics()
    
    print(f"\nSEMANTIC RAG SYSTEM ANALYTICS")
    print("=" * 40)
    
    print(f"\nSystem Performance:")
    system_stats = stats['system_stats']
    print(f"  Documents processed: {system_stats['documents_processed']}")
    print(f"  Concepts extracted: {system_stats['concepts_extracted']}")
    print(f"  Relations created: {system_stats['relations_created']}")
    print(f"  Semantic queries: {system_stats['semantic_queries']}")
    print(f"  Reasoning chains generated: {system_stats['reasoning_chains_generated']}")
    print(f"  Average reasoning depth: {system_stats['average_reasoning_depth']:.2f}")
    
    print(f"\nKnowledge Graph Statistics:")
    kg_stats = stats['knowledge_graph_stats']
    print(f"  Total concepts: {kg_stats['concept_count']}")
    print(f"  Total relations: {kg_stats['relation_count']}")
    print(f"  Graph density: {kg_stats['graph_density']:.3f}")
    print(f"  Connected components: {kg_stats['connected_components']}")
    print(f"  Average clustering: {kg_stats['average_clustering']:.3f}")
    
    print(f"\nDomain Distribution:")
    domain_dist = kg_stats['concepts_by_domain']
    for domain, count in domain_dist.items():
        print(f"  {domain}: {count} concepts")
    
    print(f"\nConcept Type Distribution:")
    type_dist = kg_stats['concepts_by_type']
    for concept_type, count in type_dist.items():
        print(f"  {concept_type}: {count} concepts")
    
    print(f"\nSearch Performance Analysis:")
    successful_searches = [r for r in search_results if r['success']]
    if successful_searches:
        avg_processing_time = sum(r['processing_time'] for r in successful_searches) / len(successful_searches)
        avg_concepts_per_query = sum(len(r['identified_concepts']) for r in successful_searches) / len(successful_searches)
        avg_results_per_query = sum(r['documents_found'] for r in successful_searches) / len(successful_searches)
        
        print(f"  Search success rate: {len(successful_searches)}/{len(search_results)} ({len(successful_searches)/len(search_results)*100:.1f}%)")
        print(f"  Average processing time: {avg_processing_time:.3f}s")
        print(f"  Average concepts per query: {avg_concepts_per_query:.1f}")
        print(f"  Average results per query: {avg_results_per_query:.1f}")
    
    print(f"\nSystem Capabilities:")
    capabilities = stats['capabilities']
    for capability, enabled in capabilities.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {capability.replace('_', ' ').title()}")

async def main():
    """
    Demonstrate Semantic RAG Systems for deep understanding and knowledge-based retrieval
    
    WHAT YOU'LL LEARN:
    ================
    1. How to extract and represent semantic concepts from text
    2. How to build knowledge graphs with concept relationships
    3. How to perform meaning-based rather than keyword-based search
    4. How to generate reasoning explanations for retrieval decisions
    5. How to create AI systems that understand domain knowledge
    
    REAL WORLD APPLICATIONS:
    =======================
    - Medical diagnosis support with symptom-disease-treatment reasoning
    - Legal research with case law and statute relationships
    - Scientific literature analysis with concept hierarchies
    - Technical documentation with component relationships
    - Business intelligence with market-strategy connections
    - Educational content with learning concept dependencies
    """
    
    print("SEMANTIC RAG SYSTEMS DEMONSTRATION")
    print("Building AI systems that understand meaning, relationships, and domain knowledge!")
    
    await demo_concept_extraction()
    await demo_knowledge_graph()
    await demo_semantic_search()
    await demo_reasoning_explanation()
    await demo_system_analytics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Concept extraction identifies semantic meaning beyond keywords")
    print("✓ Knowledge graphs capture relationships and enable reasoning")
    print("✓ Semantic search finds conceptually relevant information")
    print("✓ Reasoning explanation provides transparent decision making")
    print("✓ Domain knowledge enables expert-level understanding")
    print("✓ Multi-domain systems handle complex cross-domain queries")
    print("\nTHE POWER OF SEMANTIC RAG:")
    print("- Enables AI to understand meaning, not just text")
    print("- Provides expert-level domain reasoning and knowledge")
    print("- Supports complex knowledge work and decision making")
    print("- Powers next-generation intelligent knowledge systems")

if __name__ == "__main__":
    asyncio.run(main())
