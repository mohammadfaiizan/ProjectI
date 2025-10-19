#!/usr/bin/env python3
"""
Knowledge Graph Construction: Building Structured Knowledge Representations
=========================================================================

WHAT IS THE PROBLEM?
==================
Information exists in unstructured, disconnected silos:
- Documents contain valuable knowledge but lack relationships
- Information is scattered across multiple sources and formats
- Knowledge connections are implicit and hard to discover
- Search and reasoning require understanding complex relationships
- Insights emerge from connections between seemingly unrelated concepts
- Traditional storage doesn't capture semantic relationships

Example: Medical Knowledge Chaos
UNSTRUCTURED APPROACH (Traditional):
- Drug information in separate documents
- Disease symptoms in medical textbooks
- Treatment protocols in guidelines
- Patient cases in records
- Research findings in papers
- Result: Disconnected information, missed connections, poor insights

REAL WORLD EXAMPLE:
=================
How does Google's Knowledge Graph work?

GOOGLE'S KNOWLEDGE GRAPH:
1. ENTITY EXTRACTION: Identify people, places, things from web content
2. RELATIONSHIP MAPPING: Connect entities with semantic relationships
3. FACT VERIFICATION: Validate information across multiple sources
4. SCHEMA ORGANIZATION: Structure knowledge using standardized ontologies
5. CONTINUOUS UPDATING: Incorporate new information and refine existing knowledge
6. QUERY UNDERSTANDING: Use graph structure to understand search intent
7. INTELLIGENT RESPONSES: Provide direct answers using graph knowledge

BENEFITS OF KNOWLEDGE GRAPHS:
- Rich semantic understanding of domain knowledge
- Discovery of hidden patterns and relationships
- Enhanced search through relationship traversal
- Reasoning capabilities through graph inference
- Consistent knowledge representation across systems
- Scalable knowledge management and evolution

THE KNOWLEDGE ADVANTAGE:
======================
DOCUMENT STORAGE: Information → Search → Results (limited context)
KNOWLEDGE GRAPH: Entities → Relationships → Insights → Intelligent Reasoning

KNOWLEDGE GRAPH COMPONENTS:
=========================
1. ENTITIES: Real-world objects, concepts, or abstract ideas
2. RELATIONSHIPS: Connections between entities (typed edges)
3. ATTRIBUTES: Properties and characteristics of entities
4. ONTOLOGY: Schema defining entity types and relationship rules
5. INFERENCE RULES: Logic for deriving new knowledge from existing facts
6. PROVENANCE: Source tracking and confidence scoring
7. VERSIONING: Evolution and change management of knowledge

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems to understand context and relationships
- Provides foundation for intelligent reasoning and inference
- Powers advanced search and recommendation systems
- Critical for building truly intelligent AI assistants
- Enables discovery of insights that aren't explicitly stated
- Creates reusable knowledge assets that compound in value
"""

import asyncio
import time
import json
import uuid
import re
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EntityType(Enum):
    """Types of entities in knowledge graph"""
    PERSON = "person"
    ORGANIZATION = "organization" 
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    DOCUMENT = "document"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    PROCESS = "process"
    ABSTRACT = "abstract"

class RelationshipType(Enum):
    """Types of relationships between entities"""
    IS_A = "is_a"                    # Inheritance/subclass
    PART_OF = "part_of"              # Composition
    RELATED_TO = "related_to"        # General association
    CAUSES = "causes"                # Causal relationship
    LOCATED_IN = "located_in"        # Spatial relationship
    WORKS_FOR = "works_for"          # Employment
    CREATED_BY = "created_by"        # Authorship/creation
    INFLUENCES = "influences"        # Influence relationship
    DEPENDS_ON = "depends_on"        # Dependency
    SIMILAR_TO = "similar_to"        # Similarity

class ConfidenceLevel(Enum):
    """Confidence levels for knowledge"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    
    id: str
    name: str
    entity_type: EntityType
    
    # Core properties
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    confidence: float = 0.8
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Provenance and quality
    sources: List[str] = field(default_factory=list)
    verification_status: str = "unverified"
    quality_score: float = 0.5
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Normalize name
        self.name = self.name.strip()
        
        # Ensure aliases don't include the main name
        self.aliases = [alias.strip() for alias in self.aliases if alias.strip() != self.name]
    
    def add_attribute(self, key: str, value: Any, confidence: float = 0.8) -> None:
        """Add or update an attribute"""
        self.attributes[key] = {
            'value': value,
            'confidence': confidence,
            'updated_at': datetime.now()
        }
        self.updated_at = datetime.now()
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get attribute value"""
        if key in self.attributes:
            return self.attributes[key].get('value', default)
        return default
    
    def add_alias(self, alias: str) -> None:
        """Add an alias for this entity"""
        alias = alias.strip()
        if alias and alias != self.name and alias not in self.aliases:
            self.aliases.append(alias)
            self.updated_at = datetime.now()
    
    def merge_with(self, other: 'Entity') -> 'Entity':
        """Merge this entity with another entity"""
        if self.entity_type != other.entity_type:
            raise ValueError(f"Cannot merge entities of different types: {self.entity_type} and {other.entity_type}")
        
        # Keep the entity with higher confidence as base
        if other.confidence > self.confidence:
            base, merge = other, self
        else:
            base, merge = self, other
        
        # Create merged entity
        merged = Entity(
            id=base.id,
            name=base.name,
            entity_type=base.entity_type,
            description=base.description or merge.description,
            confidence=max(base.confidence, merge.confidence)
        )
        
        # Merge aliases
        merged.aliases = list(set(base.aliases + merge.aliases + [merge.name]))
        
        # Merge attributes (prefer higher confidence)
        merged.attributes = base.attributes.copy()
        for key, attr in merge.attributes.items():
            if key not in merged.attributes or attr.get('confidence', 0) > merged.attributes[key].get('confidence', 0):
                merged.attributes[key] = attr
        
        # Merge sources
        merged.sources = list(set(base.sources + merge.sources))
        
        # Update metadata
        merged.quality_score = (base.quality_score + merge.quality_score) / 2
        merged.updated_at = datetime.now()
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type.value,
            'description': self.description,
            'aliases': self.aliases,
            'attributes': self.attributes,
            'confidence': self.confidence,
            'source': self.source,
            'sources': self.sources,
            'verification_status': self.verification_status,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    
    # Core properties
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    confidence: float = 0.8
    weight: float = 1.0
    bidirectional: bool = False
    
    # Provenance
    source: str = ""
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_attribute(self, key: str, value: Any, confidence: float = 0.8) -> None:
        """Add or update a relationship attribute"""
        self.attributes[key] = {
            'value': value,
            'confidence': confidence,
            'updated_at': datetime.now()
        }
        self.updated_at = datetime.now()
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get relationship attribute value"""
        if key in self.attributes:
            return self.attributes[key].get('value', default)
        return default
    
    def reverse(self) -> 'Relationship':
        """Create reverse relationship if bidirectional"""
        if not self.bidirectional:
            raise ValueError("Cannot reverse unidirectional relationship")
        
        # Map relationship types to their reverses
        reverse_mapping = {
            RelationshipType.IS_A: RelationshipType.IS_A,
            RelationshipType.PART_OF: RelationshipType.PART_OF,
            RelationshipType.WORKS_FOR: RelationshipType.WORKS_FOR,
            RelationshipType.LOCATED_IN: RelationshipType.LOCATED_IN,
            RelationshipType.CREATED_BY: RelationshipType.CREATED_BY,
            RelationshipType.RELATED_TO: RelationshipType.RELATED_TO,
            RelationshipType.SIMILAR_TO: RelationshipType.SIMILAR_TO
        }
        
        reverse_type = reverse_mapping.get(self.relationship_type, self.relationship_type)
        
        return Relationship(
            id=f"{self.id}_reverse",
            source_entity_id=self.target_entity_id,
            target_entity_id=self.source_entity_id,
            relationship_type=reverse_type,
            description=f"Reverse of: {self.description}",
            confidence=self.confidence,
            weight=self.weight,
            bidirectional=True,
            source=self.source,
            sources=self.sources.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary"""
        return {
            'id': self.id,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'relationship_type': self.relationship_type.value,
            'description': self.description,
            'attributes': self.attributes,
            'confidence': self.confidence,
            'weight': self.weight,
            'bidirectional': self.bidirectional,
            'source': self.source,
            'sources': self.sources,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class EntityExtractor:
    """Extracts entities from text content"""
    
    def __init__(self):
        # Entity recognition patterns
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\bDr\. [A-Z][a-z]+\b',         # Dr. Name
                r'\bProf\. [A-Z][a-z]+\b'        # Prof. Name
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd)\b',
                r'\b[A-Z][A-Z]+ (University|Institute|Company)\b',
                r'\bUniversity of [A-Z][a-z]+\b'
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+, [A-Z][A-Z]\b',  # City, State
                r'\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard)\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(Python|Java|JavaScript|AI|ML|API)\b',
                r'\b[A-Z][a-z]+ (Framework|Library|Platform)\b'
            ],
            EntityType.CONCEPT: [
                r'\b(machine learning|artificial intelligence|data science)\b',
                r'\b[a-z]+ (algorithm|method|technique|approach)\b'
            ]
        }
        
        # Common stopwords to filter out
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        self.logger = logging.getLogger("EntityExtractor")
    
    async def extract_entities(self, text: str, source: str = "") -> List[Entity]:
        """Extract entities from text"""
        
        entities = []
        
        try:
            # Extract entities by type
            for entity_type, patterns in self.entity_patterns.items():
                type_entities = await self._extract_entities_by_type(
                    text, entity_type, patterns, source
                )
                entities.extend(type_entities)
            
            # Deduplicate entities
            entities = await self._deduplicate_entities(entities)
            
            # Enhance entities with context
            entities = await self._enhance_entities_with_context(entities, text)
            
            self.logger.debug(f"Extracted {len(entities)} entities from text")
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    async def _extract_entities_by_type(self, text: str, entity_type: EntityType,
                                      patterns: List[str], source: str) -> List[Entity]:
        """Extract entities of specific type using patterns"""
        
        entities = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity_text = match.group().strip()
                
                # Filter out stopwords and short matches
                if (entity_text.lower() not in self.stopwords and 
                    len(entity_text) > 2):
                    
                    entity = Entity(
                        id="",
                        name=entity_text,
                        entity_type=entity_type,
                        confidence=0.7,  # Pattern-based extraction confidence
                        source=source
                    )
                    
                    entities.append(entity)
        
        return entities
    
    async def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        
        unique_entities = []
        seen_names = set()
        
        for entity in entities:
            # Normalize name for comparison
            normalized_name = entity.name.lower().strip()
            
            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                unique_entities.append(entity)
            else:
                # Find existing entity and merge if same type
                for existing in unique_entities:
                    if (existing.name.lower() == normalized_name and 
                        existing.entity_type == entity.entity_type):
                        
                        # Update confidence if higher
                        if entity.confidence > existing.confidence:
                            existing.confidence = entity.confidence
                        
                        # Merge sources
                        if entity.source and entity.source not in existing.sources:
                            existing.sources.append(entity.source)
                        
                        break
        
        return unique_entities
    
    async def _enhance_entities_with_context(self, entities: List[Entity], 
                                           text: str) -> List[Entity]:
        """Enhance entities with contextual information"""
        
        enhanced_entities = []
        
        for entity in entities:
            # Find context around entity mention
            context = await self._extract_context(entity.name, text)
            
            if context:
                entity.description = context[:200]  # First 200 characters
                
                # Extract attributes from context
                attributes = await self._extract_attributes_from_context(context, entity.entity_type)
                for key, value in attributes.items():
                    entity.add_attribute(key, value)
            
            # Calculate quality score based on context richness
            entity.quality_score = min(1.0, 0.5 + len(context) / 200.0)
            
            enhanced_entities.append(entity)
        
        return enhanced_entities
    
    async def _extract_context(self, entity_name: str, text: str, 
                             window_size: int = 100) -> str:
        """Extract context around entity mention"""
        
        # Find entity mention in text
        pattern = re.escape(entity_name)
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            context = text[start:end].strip()
            return context
        
        return ""
    
    async def _extract_attributes_from_context(self, context: str, 
                                             entity_type: EntityType) -> Dict[str, Any]:
        """Extract attributes from entity context"""
        
        attributes = {}
        
        # Type-specific attribute extraction
        if entity_type == EntityType.PERSON:
            # Extract titles
            titles = re.findall(r'\b(Dr|Prof|CEO|CTO|Director)\b', context, re.IGNORECASE)
            if titles:
                attributes['title'] = titles[0]
            
            # Extract affiliations
            affiliations = re.findall(r'\bat ([A-Z][a-z]+ (?:University|Institute|Company))\b', 
                                    context, re.IGNORECASE)
            if affiliations:
                attributes['affiliation'] = affiliations[0]
        
        elif entity_type == EntityType.ORGANIZATION:
            # Extract founding year
            years = re.findall(r'\b(19|20)\d{2}\b', context)
            if years:
                attributes['founded'] = int(years[0])
            
            # Extract organization type
            org_types = re.findall(r'\b(company|corporation|nonprofit|university|institute)\b', 
                                 context, re.IGNORECASE)
            if org_types:
                attributes['type'] = org_types[0].lower()
        
        elif entity_type == EntityType.TECHNOLOGY:
            # Extract version information
            versions = re.findall(r'\bversion (\d+\.\d+|\d+)\b', context, re.IGNORECASE)
            if versions:
                attributes['version'] = versions[0]
            
            # Extract programming language
            languages = re.findall(r'\b(Python|Java|JavaScript|C\+\+|C#)\b', context)
            if languages:
                attributes['language'] = languages[0]
        
        return attributes

class RelationshipExtractor:
    """Extracts relationships between entities"""
    
    def __init__(self):
        # Relationship patterns
        self.relationship_patterns = {
            RelationshipType.WORKS_FOR: [
                r'(\w+) works for (\w+)',
                r'(\w+) is employed by (\w+)',
                r'(\w+) at (\w+)'
            ],
            RelationshipType.CREATED_BY: [
                r'(\w+) created by (\w+)',
                r'(\w+) developed by (\w+)',
                r'(\w+) authored by (\w+)'
            ],
            RelationshipType.LOCATED_IN: [
                r'(\w+) in (\w+)',
                r'(\w+) located in (\w+)',
                r'(\w+) based in (\w+)'
            ],
            RelationshipType.IS_A: [
                r'(\w+) is a (\w+)',
                r'(\w+) is an (\w+)',
                r'(\w+) type of (\w+)'
            ],
            RelationshipType.PART_OF: [
                r'(\w+) part of (\w+)',
                r'(\w+) component of (\w+)',
                r'(\w+) belongs to (\w+)'
            ]
        }
        
        self.logger = logging.getLogger("RelationshipExtractor")
    
    async def extract_relationships(self, text: str, entities: List[Entity],
                                  source: str = "") -> List[Relationship]:
        """Extract relationships from text given known entities"""
        
        relationships = []
        
        try:
            # Create entity lookup
            entity_lookup = self._create_entity_lookup(entities)
            
            # Extract relationships using patterns
            pattern_relationships = await self._extract_pattern_relationships(
                text, entity_lookup, source
            )
            relationships.extend(pattern_relationships)
            
            # Extract co-occurrence relationships
            cooccurrence_relationships = await self._extract_cooccurrence_relationships(
                text, entities, source
            )
            relationships.extend(cooccurrence_relationships)
            
            # Deduplicate relationships
            relationships = await self._deduplicate_relationships(relationships)
            
            self.logger.debug(f"Extracted {len(relationships)} relationships")
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")
        
        return relationships
    
    def _create_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Create lookup dictionary for entities by name and aliases"""
        
        lookup = {}
        
        for entity in entities:
            # Add main name
            lookup[entity.name.lower()] = entity
            
            # Add aliases
            for alias in entity.aliases:
                lookup[alias.lower()] = entity
        
        return lookup
    
    async def _extract_pattern_relationships(self, text: str, 
                                           entity_lookup: Dict[str, Entity],
                                           source: str) -> List[Relationship]:
        """Extract relationships using predefined patterns"""
        
        relationships = []
        
        for relationship_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_name = groups[0].strip().lower()
                        target_name = groups[1].strip().lower()
                        
                        # Check if both entities exist
                        if source_name in entity_lookup and target_name in entity_lookup:
                            source_entity = entity_lookup[source_name]
                            target_entity = entity_lookup[target_name]
                            
                            relationship = Relationship(
                                id="",
                                source_entity_id=source_entity.id,
                                target_entity_id=target_entity.id,
                                relationship_type=relationship_type,
                                description=match.group(),
                                confidence=0.8,  # Pattern-based confidence
                                source=source
                            )
                            
                            relationships.append(relationship)
        
        return relationships
    
    async def _extract_cooccurrence_relationships(self, text: str, 
                                                entities: List[Entity],
                                                source: str,
                                                window_size: int = 50) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence"""
        
        relationships = []
        
        # Find entity mentions in text
        entity_mentions = []
        
        for entity in entities:
            # Search for entity name and aliases
            names_to_search = [entity.name] + entity.aliases
            
            for name in names_to_search:
                pattern = re.escape(name)
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_mentions.append({
                        'entity': entity,
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group()
                    })
        
        # Sort mentions by position
        entity_mentions.sort(key=lambda x: x['start'])
        
        # Find co-occurring entities within window
        for i, mention1 in enumerate(entity_mentions):
            for mention2 in entity_mentions[i+1:]:
                distance = mention2['start'] - mention1['end']
                
                # If within window, create relationship
                if distance <= window_size:
                    # Avoid self-relationships
                    if mention1['entity'].id != mention2['entity'].id:
                        relationship = Relationship(
                            id="",
                            source_entity_id=mention1['entity'].id,
                            target_entity_id=mention2['entity'].id,
                            relationship_type=RelationshipType.RELATED_TO,
                            description=f"Co-occurs with distance {distance}",
                            confidence=max(0.3, 0.7 - distance / window_size),
                            weight=1.0 / (1 + distance / 10),  # Closer = higher weight
                            source=source
                        )
                        
                        relationships.append(relationship)
                else:
                    break  # Beyond window, stop checking
        
        return relationships
    
    async def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships"""
        
        unique_relationships = []
        seen_relationships = set()
        
        for relationship in relationships:
            # Create relationship key
            key = (
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relationship_type.value
            )
            
            if key not in seen_relationships:
                seen_relationships.add(key)
                unique_relationships.append(relationship)
            else:
                # Find existing relationship and merge
                for existing in unique_relationships:
                    if (existing.source_entity_id == relationship.source_entity_id and
                        existing.target_entity_id == relationship.target_entity_id and
                        existing.relationship_type == relationship.relationship_type):
                        
                        # Update confidence with maximum
                        existing.confidence = max(existing.confidence, relationship.confidence)
                        
                        # Merge sources
                        if relationship.source and relationship.source not in existing.sources:
                            existing.sources.append(relationship.source)
                        
                        break
        
        return unique_relationships

class KnowledgeGraph:
    """Main knowledge graph implementation"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Graph structure using NetworkX
        self.graph = nx.MultiDiGraph()
        
        # Extractors
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        
        # Statistics
        self.stats = {
            'entities_added': 0,
            'relationships_added': 0,
            'documents_processed': 0,
            'last_updated': datetime.now()
        }
        
        self.logger = logging.getLogger("KnowledgeGraph")
    
    async def add_entity(self, entity: Entity) -> str:
        """Add entity to knowledge graph"""
        
        try:
            # Check for existing entity with same name
            existing_id = await self._find_existing_entity(entity)
            
            if existing_id:
                # Merge with existing entity
                existing_entity = self.entities[existing_id]
                merged_entity = existing_entity.merge_with(entity)
                self.entities[existing_id] = merged_entity
                
                # Update graph
                self.graph.add_node(existing_id, **merged_entity.to_dict())
                
                self.logger.debug(f"Merged entity: {merged_entity.name}")
                return existing_id
            else:
                # Add new entity
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **entity.to_dict())
                
                self.stats['entities_added'] += 1
                self.stats['last_updated'] = datetime.now()
                
                self.logger.debug(f"Added entity: {entity.name}")
                return entity.id
                
        except Exception as e:
            self.logger.error(f"Failed to add entity: {e}")
            return ""
    
    async def add_relationship(self, relationship: Relationship) -> str:
        """Add relationship to knowledge graph"""
        
        try:
            # Verify entities exist
            if (relationship.source_entity_id not in self.entities or
                relationship.target_entity_id not in self.entities):
                raise ValueError("Source or target entity not found")
            
            # Check for existing relationship
            existing_id = await self._find_existing_relationship(relationship)
            
            if existing_id:
                # Update existing relationship
                existing_rel = self.relationships[existing_id]
                existing_rel.confidence = max(existing_rel.confidence, relationship.confidence)
                existing_rel.weight = max(existing_rel.weight, relationship.weight)
                
                # Merge sources
                if relationship.source and relationship.source not in existing_rel.sources:
                    existing_rel.sources.append(relationship.source)
                
                existing_rel.updated_at = datetime.now()
                
                # Update graph edge
                self.graph.add_edge(
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    key=existing_id,
                    **existing_rel.to_dict()
                )
                
                self.logger.debug(f"Updated relationship: {existing_id}")
                return existing_id
            else:
                # Add new relationship
                self.relationships[relationship.id] = relationship
                
                # Add to graph
                self.graph.add_edge(
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    key=relationship.id,
                    **relationship.to_dict()
                )
                
                # Add reverse relationship if bidirectional
                if relationship.bidirectional:
                    reverse_rel = relationship.reverse()
                    self.relationships[reverse_rel.id] = reverse_rel
                    
                    self.graph.add_edge(
                        reverse_rel.source_entity_id,
                        reverse_rel.target_entity_id,
                        key=reverse_rel.id,
                        **reverse_rel.to_dict()
                    )
                
                self.stats['relationships_added'] += 1
                self.stats['last_updated'] = datetime.now()
                
                self.logger.debug(f"Added relationship: {relationship.id}")
                return relationship.id
                
        except Exception as e:
            self.logger.error(f"Failed to add relationship: {e}")
            return ""
    
    async def process_document(self, content: str, document_id: str = "",
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document and extract knowledge"""
        
        start_time = time.time()
        
        try:
            if not document_id:
                document_id = str(uuid.uuid4())
            
            self.logger.info(f"Processing document: {document_id}")
            
            # Extract entities
            entities = await self.entity_extractor.extract_entities(content, document_id)
            
            # Add entities to graph
            entity_ids = []
            for entity in entities:
                entity_id = await self.add_entity(entity)
                if entity_id:
                    entity_ids.append(entity_id)
            
            # Extract relationships
            relationships = await self.relationship_extractor.extract_relationships(
                content, entities, document_id
            )
            
            # Add relationships to graph
            relationship_ids = []
            for relationship in relationships:
                rel_id = await self.add_relationship(relationship)
                if rel_id:
                    relationship_ids.append(rel_id)
            
            # Create document entity
            document_entity = Entity(
                id=document_id,
                name=f"Document {document_id}",
                entity_type=EntityType.DOCUMENT,
                description=content[:200],
                source=document_id
            )
            
            if metadata:
                for key, value in metadata.items():
                    document_entity.add_attribute(key, value)
            
            await self.add_entity(document_entity)
            
            # Link document to extracted entities
            for entity_id in entity_ids:
                doc_relationship = Relationship(
                    id="",
                    source_entity_id=document_id,
                    target_entity_id=entity_id,
                    relationship_type=RelationshipType.RELATED_TO,
                    description="Document mentions entity",
                    confidence=0.9
                )
                await self.add_relationship(doc_relationship)
            
            processing_time = time.time() - start_time
            self.stats['documents_processed'] += 1
            
            result = {
                'success': True,
                'document_id': document_id,
                'entities_extracted': len(entity_ids),
                'relationships_extracted': len(relationship_ids),
                'processing_time': processing_time,
                'entity_ids': entity_ids,
                'relationship_ids': relationship_ids
            }
            
            self.logger.info(f"Document processed: {len(entity_ids)} entities, "
                           f"{len(relationship_ids)} relationships, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def query_entities(self, query: str, entity_types: List[EntityType] = None,
                           limit: int = 10) -> List[Entity]:
        """Query entities by name or description"""
        
        matching_entities = []
        query_lower = query.lower()
        
        for entity in self.entities.values():
            # Check entity type filter
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Calculate relevance score
            score = 0.0
            
            # Exact name match
            if entity.name.lower() == query_lower:
                score += 1.0
            # Partial name match
            elif query_lower in entity.name.lower():
                score += 0.8
            # Alias match
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                score += 0.7
            # Description match
            elif query_lower in entity.description.lower():
                score += 0.5
            
            if score > 0:
                matching_entities.append((entity, score))
        
        # Sort by relevance score
        matching_entities.sort(key=lambda x: x[1], reverse=True)
        
        return [entity for entity, score in matching_entities[:limit]]
    
    async def find_path(self, source_entity_id: str, target_entity_id: str,
                       max_length: int = 5) -> List[List[str]]:
        """Find paths between entities"""
        
        try:
            if source_entity_id not in self.graph or target_entity_id not in self.graph:
                return []
            
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph, 
                source_entity_id, 
                target_entity_id, 
                cutoff=max_length
            ))
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Path finding failed: {e}")
            return []
    
    async def get_neighbors(self, entity_id: str, relationship_types: List[RelationshipType] = None,
                          distance: int = 1) -> List[Dict[str, Any]]:
        """Get neighboring entities"""
        
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        
        if distance == 1:
            # Direct neighbors
            for neighbor_id in self.graph.neighbors(entity_id):
                # Get relationship information
                edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
                
                for edge_key, edge_attrs in edge_data.items():
                    rel_type = RelationshipType(edge_attrs.get('relationship_type'))
                    
                    # Filter by relationship type if specified
                    if relationship_types and rel_type not in relationship_types:
                        continue
                    
                    neighbor_entity = self.entities.get(neighbor_id)
                    if neighbor_entity:
                        neighbors.append({
                            'entity': neighbor_entity,
                            'relationship': self.relationships.get(edge_key),
                            'distance': 1
                        })
        else:
            # Multi-hop neighbors using BFS
            visited = set()
            queue = deque([(entity_id, 0)])
            
            while queue:
                current_id, current_distance = queue.popleft()
                
                if current_distance >= distance:
                    continue
                
                for neighbor_id in self.graph.neighbors(current_id):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, current_distance + 1))
                        
                        neighbor_entity = self.entities.get(neighbor_id)
                        if neighbor_entity:
                            neighbors.append({
                                'entity': neighbor_entity,
                                'distance': current_distance + 1
                            })
        
        return neighbors
    
    async def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get statistics for an entity"""
        
        if entity_id not in self.entities:
            return {}
        
        entity = self.entities[entity_id]
        
        # Count relationships
        outgoing_relationships = len(list(self.graph.successors(entity_id)))
        incoming_relationships = len(list(self.graph.predecessors(entity_id)))
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph).get(entity_id, 0)
        betweenness_centrality = nx.betweenness_centrality(self.graph).get(entity_id, 0)
        
        # Get relationship types
        relationship_types = defaultdict(int)
        for neighbor_id in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
            for edge_attrs in edge_data.values():
                rel_type = edge_attrs.get('relationship_type')
                relationship_types[rel_type] += 1
        
        return {
            'entity': entity.to_dict(),
            'outgoing_relationships': outgoing_relationships,
            'incoming_relationships': incoming_relationships,
            'total_relationships': outgoing_relationships + incoming_relationships,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'relationship_types': dict(relationship_types)
        }
    
    async def _find_existing_entity(self, entity: Entity) -> Optional[str]:
        """Find existing entity with same name or alias"""
        
        for existing_id, existing_entity in self.entities.items():
            # Check name match
            if existing_entity.name.lower() == entity.name.lower():
                return existing_id
            
            # Check alias match
            if entity.name.lower() in [alias.lower() for alias in existing_entity.aliases]:
                return existing_id
            
            # Check if new entity name matches existing alias
            if existing_entity.name.lower() in [alias.lower() for alias in entity.aliases]:
                return existing_id
        
        return None
    
    async def _find_existing_relationship(self, relationship: Relationship) -> Optional[str]:
        """Find existing relationship between same entities"""
        
        for existing_id, existing_rel in self.relationships.items():
            if (existing_rel.source_entity_id == relationship.source_entity_id and
                existing_rel.target_entity_id == relationship.target_entity_id and
                existing_rel.relationship_type == relationship.relationship_type):
                return existing_id
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        
        # Basic counts
        num_entities = len(self.entities)
        num_relationships = len(self.relationships)
        
        # Entity type distribution
        entity_types = defaultdict(int)
        for entity in self.entities.values():
            entity_types[entity.entity_type.value] += 1
        
        # Relationship type distribution
        relationship_types = defaultdict(int)
        for relationship in self.relationships.values():
            relationship_types[relationship.relationship_type.value] += 1
        
        # Graph metrics
        if num_entities > 0:
            density = nx.density(self.graph)
            try:
                avg_clustering = nx.average_clustering(self.graph.to_undirected())
            except:
                avg_clustering = 0.0
            
            # Connected components
            undirected_graph = self.graph.to_undirected()
            num_components = nx.number_connected_components(undirected_graph)
            largest_component_size = len(max(nx.connected_components(undirected_graph), 
                                           key=len, default=[]))
        else:
            density = 0.0
            avg_clustering = 0.0
            num_components = 0
            largest_component_size = 0
        
        return {
            'basic_statistics': {
                'total_entities': num_entities,
                'total_relationships': num_relationships,
                'documents_processed': self.stats['documents_processed'],
                'last_updated': self.stats['last_updated'].isoformat()
            },
            'entity_distribution': dict(entity_types),
            'relationship_distribution': dict(relationship_types),
            'graph_metrics': {
                'density': density,
                'average_clustering': avg_clustering,
                'connected_components': num_components,
                'largest_component_size': largest_component_size
            },
            'processing_statistics': self.stats
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_entity_extraction():
    """Demo: Entity extraction from text"""
    print("\nDEMO 1: ENTITY EXTRACTION")
    print("=" * 50)
    
    extractor = EntityExtractor()
    
    # Sample text with various entity types
    sample_text = """
    Dr. John Smith, a professor at Stanford University, has been working on machine learning 
    algorithms for the past decade. His research focuses on neural networks and deep learning 
    frameworks like TensorFlow and PyTorch. He recently published a paper in collaboration 
    with researchers from Google Inc. The study was conducted in Mountain View, California, 
    and explores applications of artificial intelligence in healthcare.
    """
    
    print("Sample Text:")
    print(sample_text.strip())
    
    print(f"\nExtracting entities...")
    
    entities = await extractor.extract_entities(sample_text, "demo_document")
    
    print(f"\nExtracted {len(entities)} entities:")
    
    # Group entities by type
    entities_by_type = defaultdict(list)
    for entity in entities:
        entities_by_type[entity.entity_type].append(entity)
    
    for entity_type, type_entities in entities_by_type.items():
        print(f"\n{entity_type.value.upper()} entities:")
        for entity in type_entities:
            print(f"  - {entity.name} (confidence: {entity.confidence:.2f})")
            if entity.description:
                print(f"    Description: {entity.description[:100]}...")
            if entity.attributes:
                print(f"    Attributes: {entity.attributes}")

async def demo_relationship_extraction():
    """Demo: Relationship extraction from text"""
    print("\nDEMO 2: RELATIONSHIP EXTRACTION")
    print("=" * 50)
    
    # First extract entities
    extractor = EntityExtractor()
    rel_extractor = RelationshipExtractor()
    
    sample_text = """
    Python is a programming language created by Guido van Rossum. The language is used by 
    Google for many of their applications. TensorFlow is a machine learning framework 
    developed by Google. Many researchers at Stanford University use TensorFlow for their 
    deep learning projects. Dr. Sarah Johnson works for Google and specializes in artificial 
    intelligence research.
    """
    
    print("Sample Text:")
    print(sample_text.strip())
    
    print(f"\nExtracting entities and relationships...")
    
    # Extract entities first
    entities = await extractor.extract_entities(sample_text, "demo_document")
    
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.name} ({entity.entity_type.value})")
    
    # Extract relationships
    relationships = await rel_extractor.extract_relationships(
        sample_text, entities, "demo_document"
    )
    
    print(f"\nExtracted {len(relationships)} relationships:")
    
    # Create entity lookup for display
    entity_lookup = {entity.id: entity.name for entity in entities}
    
    for relationship in relationships:
        source_name = entity_lookup.get(relationship.source_entity_id, "Unknown")
        target_name = entity_lookup.get(relationship.target_entity_id, "Unknown")
        
        print(f"  - {source_name} --[{relationship.relationship_type.value}]--> {target_name}")
        print(f"    Confidence: {relationship.confidence:.2f}")
        if relationship.description:
            print(f"    Context: {relationship.description}")

async def demo_knowledge_graph_construction():
    """Demo: Building a complete knowledge graph"""
    print("\nDEMO 3: KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 50)
    
    kg = KnowledgeGraph()
    
    # Sample documents to process
    documents = [
        {
            'id': 'doc1',
            'content': '''
            Machine learning is a subset of artificial intelligence that focuses on algorithms 
            that can learn from data. Python is widely used for machine learning due to 
            libraries like scikit-learn, TensorFlow, and PyTorch. Google developed TensorFlow 
            as an open-source framework for deep learning.
            ''',
            'metadata': {'title': 'Introduction to Machine Learning', 'author': 'AI Researcher'}
        },
        {
            'id': 'doc2', 
            'content': '''
            Dr. Geoffrey Hinton is known as the "Godfather of Deep Learning" and works at 
            Google. He developed backpropagation algorithm which is fundamental to neural 
            networks. The University of Toronto is where Dr. Hinton was a professor before 
            joining Google.
            ''',
            'metadata': {'title': 'Deep Learning Pioneers', 'author': 'Tech Journalist'}
        },
        {
            'id': 'doc3',
            'content': '''
            Stanford University has a strong artificial intelligence program. Many researchers 
            there work on computer vision and natural language processing. The Stanford AI Lab 
            has contributed significantly to the field of robotics and autonomous systems.
            ''',
            'metadata': {'title': 'AI Research at Universities', 'author': 'Academic Writer'}
        }
    ]
    
    print("Processing documents to build knowledge graph...")
    
    for doc in documents:
        print(f"\nProcessing: {doc['metadata']['title']}")
        
        result = await kg.process_document(
            doc['content'], 
            doc['id'], 
            doc['metadata']
        )
        
        if result['success']:
            print(f"  ✓ Extracted {result['entities_extracted']} entities")
            print(f"  ✓ Extracted {result['relationships_extracted']} relationships")
            print(f"  ✓ Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Show final statistics
    print(f"\nKnowledge Graph Statistics:")
    stats = kg.get_statistics()
    
    basic_stats = stats['basic_statistics']
    print(f"  Total entities: {basic_stats['total_entities']}")
    print(f"  Total relationships: {basic_stats['total_relationships']}")
    print(f"  Documents processed: {basic_stats['documents_processed']}")
    
    print(f"\nEntity Distribution:")
    for entity_type, count in stats['entity_distribution'].items():
        print(f"  {entity_type}: {count}")
    
    print(f"\nRelationship Distribution:")
    for rel_type, count in stats['relationship_distribution'].items():
        print(f"  {rel_type}: {count}")
    
    graph_metrics = stats['graph_metrics']
    print(f"\nGraph Metrics:")
    print(f"  Density: {graph_metrics['density']:.3f}")
    print(f"  Average clustering: {graph_metrics['average_clustering']:.3f}")
    print(f"  Connected components: {graph_metrics['connected_components']}")

async def demo_knowledge_graph_querying():
    """Demo: Querying the knowledge graph"""
    print("\nDEMO 4: KNOWLEDGE GRAPH QUERYING")
    print("=" * 50)
    
    # Build a knowledge graph first
    kg = KnowledgeGraph()
    
    # Add some sample entities manually for demonstration
    entities_data = [
        ("Python", EntityType.TECHNOLOGY, "Programming language"),
        ("Google", EntityType.ORGANIZATION, "Technology company"),
        ("TensorFlow", EntityType.TECHNOLOGY, "Machine learning framework"),
        ("Dr. Geoffrey Hinton", EntityType.PERSON, "Deep learning researcher"),
        ("Stanford University", EntityType.ORGANIZATION, "Educational institution"),
        ("Machine Learning", EntityType.CONCEPT, "AI subset"),
        ("Artificial Intelligence", EntityType.CONCEPT, "Computer science field")
    ]
    
    entity_objects = []
    for name, entity_type, description in entities_data:
        entity = Entity(
            id="",
            name=name,
            entity_type=entity_type,
            description=description,
            confidence=0.9
        )
        entity_id = await kg.add_entity(entity)
        entity_objects.append((entity_id, entity))
    
    # Add some relationships
    relationships_data = [
        (0, 5, RelationshipType.RELATED_TO, "Python used for ML"),  # Python -> ML
        (1, 2, RelationshipType.CREATED_BY, "Google created TensorFlow"),  # TensorFlow -> Google
        (3, 1, RelationshipType.WORKS_FOR, "Hinton works at Google"),  # Hinton -> Google
        (5, 6, RelationshipType.IS_A, "ML is subset of AI"),  # ML -> AI
        (2, 5, RelationshipType.RELATED_TO, "TensorFlow for ML")  # TensorFlow -> ML
    ]
    
    for source_idx, target_idx, rel_type, description in relationships_data:
        source_id = entity_objects[source_idx][0]
        target_id = entity_objects[target_idx][0]
        
        relationship = Relationship(
            id="",
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type=rel_type,
            description=description,
            confidence=0.8
        )
        await kg.add_relationship(relationship)
    
    print("Knowledge graph constructed with sample data.")
    
    # Demo queries
    queries = [
        ("Python", None),
        ("Google", [EntityType.ORGANIZATION]),
        ("learning", [EntityType.CONCEPT]),
        ("Dr", [EntityType.PERSON])
    ]
    
    print(f"\nQuerying entities:")
    
    for query, entity_types in queries:
        print(f"\nQuery: '{query}'" + 
              (f" (types: {[t.value for t in entity_types]})" if entity_types else ""))
        
        results = await kg.query_entities(query, entity_types, limit=5)
        
        if results:
            for entity in results:
                print(f"  - {entity.name} ({entity.entity_type.value})")
                print(f"    {entity.description}")
        else:
            print("  No results found")
    
    # Demo path finding
    print(f"\nFinding paths between entities:")
    
    # Find path from Python to Artificial Intelligence
    python_entity = entity_objects[0][1]  # Python
    ai_entity = entity_objects[6][1]      # AI
    
    paths = await kg.find_path(python_entity.id, ai_entity.id, max_length=4)
    
    if paths:
        print(f"\nPaths from '{python_entity.name}' to '{ai_entity.name}':")
        
        for i, path in enumerate(paths[:3], 1):  # Show first 3 paths
            print(f"  Path {i}: ", end="")
            path_names = []
            for entity_id in path:
                entity = kg.entities.get(entity_id)
                if entity:
                    path_names.append(entity.name)
            print(" -> ".join(path_names))
    else:
        print(f"No paths found between '{python_entity.name}' and '{ai_entity.name}'")
    
    # Demo neighbor finding
    google_entity = entity_objects[1][1]  # Google
    print(f"\nNeighbors of '{google_entity.name}':")
    
    neighbors = await kg.get_neighbors(google_entity.id, distance=1)
    
    for neighbor_info in neighbors:
        neighbor = neighbor_info['entity']
        relationship = neighbor_info.get('relationship')
        
        print(f"  - {neighbor.name} ({neighbor.entity_type.value})")
        if relationship:
            print(f"    Relationship: {relationship.relationship_type.value}")
            print(f"    Confidence: {relationship.confidence:.2f}")

async def demo_entity_statistics():
    """Demo: Entity statistics and analysis"""
    print("\nDEMO 5: ENTITY STATISTICS AND ANALYSIS")
    print("=" * 50)
    
    # Create knowledge graph with comprehensive data
    kg = KnowledgeGraph()
    
    # Process a comprehensive document
    comprehensive_text = """
    Artificial intelligence is a broad field that encompasses machine learning, deep learning, 
    and natural language processing. Python is the most popular programming language for AI 
    development, with libraries like TensorFlow, PyTorch, and scikit-learn.
    
    Google developed TensorFlow, which is widely used at Stanford University for research. 
    Dr. Geoffrey Hinton, who works at Google, is known for his contributions to deep learning. 
    The University of Toronto, where Dr. Hinton was previously a professor, has a strong AI program.
    
    Facebook (now Meta) developed PyTorch, another popular deep learning framework. Many companies 
    like Microsoft, Amazon, and Apple also invest heavily in artificial intelligence research.
    The MIT Computer Science and Artificial Intelligence Laboratory (CSAIL) is another leading 
    research institution in this field.
    """
    
    print("Processing comprehensive text to build detailed knowledge graph...")
    
    result = await kg.process_document(comprehensive_text, "comprehensive_doc")
    
    if result['success']:
        print(f"✓ Successfully processed document")
        print(f"  Entities: {result['entities_extracted']}")
        print(f"  Relationships: {result['relationships_extracted']}")
        
        # Get overall statistics
        stats = kg.get_statistics()
        
        print(f"\nOverall Knowledge Graph Statistics:")
        print(f"  Total entities: {stats['basic_statistics']['total_entities']}")
        print(f"  Total relationships: {stats['basic_statistics']['total_relationships']}")
        print(f"  Graph density: {stats['graph_metrics']['density']:.3f}")
        
        # Find and analyze top entities
        print(f"\nTop Entities by Connectivity:")
        
        entity_connectivity = []
        for entity_id, entity in kg.entities.items():
            entity_stats = await kg.get_entity_statistics(entity_id)
            connectivity = entity_stats.get('total_relationships', 0)
            entity_connectivity.append((entity, connectivity, entity_stats))
        
        # Sort by connectivity
        entity_connectivity.sort(key=lambda x: x[1], reverse=True)
        
        for i, (entity, connectivity, entity_stats) in enumerate(entity_connectivity[:5], 1):
            print(f"\n{i}. {entity.name} ({entity.entity_type.value})")
            print(f"   Total relationships: {connectivity}")
            print(f"   Degree centrality: {entity_stats.get('degree_centrality', 0):.3f}")
            print(f"   Betweenness centrality: {entity_stats.get('betweenness_centrality', 0):.3f}")
            
            rel_types = entity_stats.get('relationship_types', {})
            if rel_types:
                print(f"   Relationship types: {dict(rel_types)}")
        
        # Show entity type analysis
        print(f"\nEntity Type Analysis:")
        entity_dist = stats['entity_distribution']
        total_entities = sum(entity_dist.values())
        
        for entity_type, count in sorted(entity_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_entities) * 100
            print(f"  {entity_type}: {count} ({percentage:.1f}%)")
        
        # Show relationship type analysis
        print(f"\nRelationship Type Analysis:")
        rel_dist = stats['relationship_distribution']
        total_relationships = sum(rel_dist.values())
        
        for rel_type, count in sorted(rel_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_relationships) * 100
            print(f"  {rel_type}: {count} ({percentage:.1f}%)")
    
    else:
        print(f"✗ Failed to process document: {result['error']}")

async def main():
    """
    Demonstrate Knowledge Graph Construction for structured knowledge representation
    
    WHAT YOU'LL LEARN:
    ================
    1. How to extract entities and relationships from unstructured text
    2. How to build and maintain a structured knowledge graph
    3. How to query and traverse knowledge graphs effectively
    4. How to analyze entity importance and graph structure
    5. How to create reusable knowledge assets from documents
    6. How to handle entity disambiguation and relationship inference
    
    REAL WORLD APPLICATIONS:
    =======================
    - Enterprise knowledge management systems
    - Intelligent search and recommendation engines
    - Automated fact-checking and information verification
    - Research and discovery platforms
    - Customer support knowledge bases
    - Medical and scientific knowledge systems
    """
    
    print("KNOWLEDGE GRAPH CONSTRUCTION DEMONSTRATION")
    print("Building structured knowledge from unstructured information!")
    
    await demo_entity_extraction()
    await demo_relationship_extraction()
    await demo_knowledge_graph_construction()
    await demo_knowledge_graph_querying()
    await demo_entity_statistics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Entity extraction identifies key concepts and objects in text")
    print("✓ Relationship extraction discovers connections between entities")
    print("✓ Knowledge graphs provide structured representation of knowledge")
    print("✓ Graph querying enables intelligent search and discovery")
    print("✓ Statistical analysis reveals important entities and patterns")
    print("✓ Knowledge graphs enable reasoning and inference capabilities")
    print("\nTHE POWER OF KNOWLEDGE GRAPHS:")
    print("- Transform unstructured information into structured knowledge")
    print("- Enable discovery of hidden connections and insights")
    print("- Provide foundation for intelligent AI systems")
    print("- Create reusable and evolving knowledge assets")

if __name__ == "__main__":
    asyncio.run(main())
