#!/usr/bin/env python3
"""
Semantic Memory Manager: Organizing General Knowledge and Concepts
================================================================

WHAT IS THE PROBLEM?
==================
AI systems struggle with organizing and accessing general knowledge:
- Facts and concepts are stored as isolated pieces without meaningful connections
- No hierarchical organization of knowledge from general to specific
- Cannot distinguish between different types of knowledge (facts vs concepts vs procedures)
- Missing semantic relationships that enable intelligent reasoning and inference
- Lack of conceptual understanding that goes beyond simple keyword matching
- No way to organize knowledge by abstraction levels or domain categories

Example: Fragmented Knowledge Without Semantic Organization
WITHOUT SEMANTIC MEMORY (Traditional):
- System knows "Dog is an animal" but not that this is a taxonomic relationship
- Knows "Paris is in France" but not the geographic containment hierarchy
- Cannot infer that "If Fido is a dog, and dogs are mammals, then Fido is a mammal"
- Missing conceptual frameworks that organize related knowledge
- Cannot distinguish between facts, definitions, examples, and procedures
- Result: Fragmented knowledge, poor reasoning, inability to make logical connections

REAL WORLD EXAMPLE:
=================
How does human semantic memory organize knowledge?

HUMAN SEMANTIC MEMORY ORGANIZATION:
1. HIERARCHICAL CONCEPTS: Animal → Mammal → Dog → Golden Retriever
2. SEMANTIC NETWORKS: Concepts connected by meaningful relationships
3. CATEGORICAL KNOWLEDGE: Things grouped by shared properties and functions
4. ABSTRACTION LEVELS: From concrete instances to abstract principles
5. DOMAIN ORGANIZATION: Medical knowledge separate from cooking knowledge
6. PROPERTY INHERITANCE: Lower-level concepts inherit properties from higher levels
7. CONCEPTUAL FRAMEWORKS: Mental models that organize related knowledge

BENEFITS OF SEMANTIC MEMORY:
- Enables logical reasoning and inference from general knowledge
- Supports knowledge transfer between related domains and concepts
- Allows efficient knowledge organization and retrieval
- Enables understanding of conceptual relationships and hierarchies
- Supports learning by connecting new information to existing frameworks
- Provides foundation for intelligent question answering and reasoning

THE SEMANTIC ADVANTAGE:
=====================
ISOLATED FACTS: Dog, Animal, Mammal → Separate unconnected information
SEMANTIC NETWORK: Dog IS-A Mammal IS-A Animal → Connected knowledge hierarchy

SEMANTIC MEMORY COMPONENTS:
==========================
1. CONCEPT HIERARCHIES: Tree structures from general to specific concepts
2. SEMANTIC RELATIONSHIPS: IS-A, HAS-A, PART-OF, CAUSES, and other relations
3. PROPERTY INHERITANCE: Lower concepts inherit properties from higher concepts
4. CONCEPTUAL CATEGORIES: Grouping concepts by shared characteristics
5. DOMAIN ORGANIZATION: Separate knowledge spaces for different domains
6. KNOWLEDGE GRAPHS: Network representation of concepts and relationships
7. ABSTRACTION MANAGEMENT: Different levels of detail and generalization

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems to reason with structured knowledge like humans do
- Critical for intelligent question answering and logical reasoning
- Foundation for knowledge-based AI and expert systems
- Supports natural language understanding through conceptual frameworks
- Enables efficient knowledge sharing and transfer between domains
- Creates scalable knowledge organization for large-scale AI systems
"""

import asyncio
import time
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sqlite3
import threading
import numpy as np
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SemanticRelationType(Enum):
    """Types of semantic relationships"""
    IS_A = "is_a"                    # Taxonomic hierarchy (dog is_a mammal)
    HAS_A = "has_a"                  # Part-whole relationship (car has_a engine)
    PART_OF = "part_of"              # Reverse of has_a (engine part_of car)
    CAUSES = "causes"                # Causal relationship (virus causes disease)
    ENABLES = "enables"              # Enabling relationship (key enables access)
    REQUIRES = "requires"            # Dependency relationship (cooking requires heat)
    SIMILAR_TO = "similar_to"        # Similarity relationship (cat similar_to dog)
    OPPOSITE_OF = "opposite_of"      # Antonym relationship (hot opposite_of cold)
    EXAMPLE_OF = "example_of"        # Instantiation (Fido example_of dog)
    USED_FOR = "used_for"           # Purpose relationship (hammer used_for hitting)

class ConceptType(Enum):
    """Types of concepts in semantic memory"""
    ENTITY = "entity"                # Concrete objects (dog, car, person)
    ABSTRACT = "abstract"            # Abstract concepts (love, justice, freedom)
    CATEGORY = "category"            # Classifications (animal, vehicle, color)
    PROPERTY = "property"            # Attributes (red, large, intelligent)
    RELATION = "relation"            # Relationships (parent, friend, above)
    PROCESS = "process"              # Actions/procedures (cooking, learning, running)
    EVENT = "event"                  # Occurrences (birthday, meeting, earthquake)

class KnowledgeType(Enum):
    """Types of knowledge stored"""
    FACT = "fact"                    # Factual information (Paris is in France)
    DEFINITION = "definition"        # Concept definitions (Dog is a domesticated mammal)
    RULE = "rule"                    # General principles (Metals expand when heated)
    EXAMPLE = "example"              # Specific instances (Golden Retriever is a dog breed)
    PROCEDURE = "procedure"          # How-to knowledge (Steps to make coffee)

@dataclass
class Concept:
    """Represents a concept in semantic memory"""
    
    id: str
    name: str
    concept_type: ConceptType
    
    # Definition and description
    definition: str = ""
    description: str = ""
    
    # Properties and attributes
    properties: Dict[str, Any] = field(default_factory=dict)
    attributes: Set[str] = field(default_factory=set)
    
    # Hierarchical information
    parent_concepts: Set[str] = field(default_factory=set)   # More general concepts
    child_concepts: Set[str] = field(default_factory=set)    # More specific concepts
    
    # Domain and category
    domain: str = ""
    categories: Set[str] = field(default_factory=set)
    
    # Semantic relationships
    relationships: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Knowledge content
    facts: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    # Usage statistics
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_relationship(self, relation_type: SemanticRelationType, target_concept_id: str) -> None:
        """Add a relationship to another concept"""
        
        relation_key = relation_type.value
        if relation_key not in self.relationships:
            self.relationships[relation_key] = set()
        
        self.relationships[relation_key].add(target_concept_id)
        self.last_updated = datetime.now()
    
    def remove_relationship(self, relation_type: SemanticRelationType, target_concept_id: str) -> bool:
        """Remove a relationship to another concept"""
        
        relation_key = relation_type.value
        if relation_key in self.relationships:
            if target_concept_id in self.relationships[relation_key]:
                self.relationships[relation_key].remove(target_concept_id)
                
                # Clean up empty relationship sets
                if not self.relationships[relation_key]:
                    del self.relationships[relation_key]
                
                self.last_updated = datetime.now()
                return True
        
        return False
    
    def get_related_concepts(self, relation_type: SemanticRelationType) -> Set[str]:
        """Get concepts related by specific relationship type"""
        
        return self.relationships.get(relation_type.value, set())
    
    def add_property(self, property_name: str, property_value: Any) -> None:
        """Add a property to the concept"""
        
        self.properties[property_name] = property_value
        self.last_updated = datetime.now()
    
    def inherit_properties(self, parent_concept: 'Concept') -> None:
        """Inherit properties from parent concept"""
        
        for prop_name, prop_value in parent_concept.properties.items():
            if prop_name not in self.properties:
                self.properties[prop_name] = prop_value
        
        # Inherit attributes
        self.attributes.update(parent_concept.attributes)
        
        self.last_updated = datetime.now()
    
    def access(self) -> None:
        """Record concept access"""
        
        self.access_count += 1
        self.last_accessed = datetime.now()

@dataclass
class SemanticKnowledge:
    """Represents semantic knowledge (facts, rules, etc.)"""
    
    id: str
    knowledge_type: KnowledgeType
    content: str
    
    # Related concepts
    subject_concept_id: Optional[str] = None
    object_concept_id: Optional[str] = None
    related_concepts: Set[str] = field(default_factory=set)
    
    # Metadata
    domain: str = ""
    confidence: float = 1.0
    source: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class ConceptHierarchy:
    """Manages concept hierarchies and inheritance"""
    
    def __init__(self):
        self.hierarchy_graph: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.reverse_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # child -> parents
        
        self.logger = logging.getLogger("ConceptHierarchy")
    
    def add_is_a_relationship(self, child_concept_id: str, parent_concept_id: str) -> None:
        """Add IS-A relationship (child is a parent)"""
        
        self.hierarchy_graph[parent_concept_id].add(child_concept_id)
        self.reverse_hierarchy[child_concept_id].add(parent_concept_id)
        
        self.logger.debug(f"Added hierarchy: {child_concept_id} IS-A {parent_concept_id}")
    
    def remove_is_a_relationship(self, child_concept_id: str, parent_concept_id: str) -> None:
        """Remove IS-A relationship"""
        
        self.hierarchy_graph[parent_concept_id].discard(child_concept_id)
        self.reverse_hierarchy[child_concept_id].discard(parent_concept_id)
        
        # Clean up empty sets
        if not self.hierarchy_graph[parent_concept_id]:
            del self.hierarchy_graph[parent_concept_id]
        
        if not self.reverse_hierarchy[child_concept_id]:
            del self.reverse_hierarchy[child_concept_id]
    
    def get_ancestors(self, concept_id: str, max_depth: int = None) -> Set[str]:
        """Get all ancestor concepts (parents, grandparents, etc.)"""
        
        ancestors = set()
        to_visit = deque([(concept_id, 0)])
        visited = set()
        
        while to_visit:
            current_id, depth = to_visit.popleft()
            
            if current_id in visited:
                continue
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            visited.add(current_id)
            
            # Get direct parents
            parents = self.reverse_hierarchy.get(current_id, set())
            
            for parent_id in parents:
                if parent_id not in visited:
                    ancestors.add(parent_id)
                    to_visit.append((parent_id, depth + 1))
        
        return ancestors
    
    def get_descendants(self, concept_id: str, max_depth: int = None) -> Set[str]:
        """Get all descendant concepts (children, grandchildren, etc.)"""
        
        descendants = set()
        to_visit = deque([(concept_id, 0)])
        visited = set()
        
        while to_visit:
            current_id, depth = to_visit.popleft()
            
            if current_id in visited:
                continue
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            visited.add(current_id)
            
            # Get direct children
            children = self.hierarchy_graph.get(current_id, set())
            
            for child_id in children:
                if child_id not in visited:
                    descendants.add(child_id)
                    to_visit.append((child_id, depth + 1))
        
        return descendants
    
    def get_common_ancestors(self, concept_ids: List[str]) -> Set[str]:
        """Find common ancestors of multiple concepts"""
        
        if not concept_ids:
            return set()
        
        # Get ancestors for first concept
        common_ancestors = self.get_ancestors(concept_ids[0])
        
        # Intersect with ancestors of other concepts
        for concept_id in concept_ids[1:]:
            concept_ancestors = self.get_ancestors(concept_id)
            common_ancestors = common_ancestors.intersection(concept_ancestors)
        
        return common_ancestors
    
    def get_lowest_common_ancestor(self, concept_ids: List[str]) -> Optional[str]:
        """Find the lowest (most specific) common ancestor"""
        
        common_ancestors = self.get_common_ancestors(concept_ids)
        
        if not common_ancestors:
            return None
        
        # Find the ancestor with the fewest ancestors (most specific)
        most_specific = None
        min_ancestor_count = float('inf')
        
        for ancestor_id in common_ancestors:
            ancestor_count = len(self.get_ancestors(ancestor_id))
            
            if ancestor_count < min_ancestor_count:
                min_ancestor_count = ancestor_count
                most_specific = ancestor_id
        
        return most_specific
    
    def is_ancestor(self, potential_ancestor_id: str, concept_id: str) -> bool:
        """Check if one concept is an ancestor of another"""
        
        ancestors = self.get_ancestors(concept_id)
        return potential_ancestor_id in ancestors
    
    def get_hierarchy_depth(self, concept_id: str) -> int:
        """Get the depth of a concept in the hierarchy"""
        
        max_depth = 0
        
        def calculate_depth(current_id: str, current_depth: int) -> int:
            parents = self.reverse_hierarchy.get(current_id, set())
            
            if not parents:
                return current_depth
            
            max_parent_depth = current_depth
            
            for parent_id in parents:
                parent_depth = calculate_depth(parent_id, current_depth + 1)
                max_parent_depth = max(max_parent_depth, parent_depth)
            
            return max_parent_depth
        
        return calculate_depth(concept_id, 0)

class SemanticNetwork:
    """Manages semantic relationships between concepts"""
    
    def __init__(self):
        self.relationships: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.reverse_relationships: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        self.logger = logging.getLogger("SemanticNetwork")
    
    def add_relationship(self, source_concept_id: str, relation_type: SemanticRelationType,
                        target_concept_id: str) -> None:
        """Add a semantic relationship"""
        
        relation_key = relation_type.value
        
        # Add forward relationship
        self.relationships[source_concept_id][relation_key].add(target_concept_id)
        
        # Add reverse relationship
        self.reverse_relationships[target_concept_id][relation_key].add(source_concept_id)
        
        self.logger.debug(f"Added relationship: {source_concept_id} {relation_key} {target_concept_id}")
    
    def remove_relationship(self, source_concept_id: str, relation_type: SemanticRelationType,
                           target_concept_id: str) -> None:
        """Remove a semantic relationship"""
        
        relation_key = relation_type.value
        
        # Remove forward relationship
        if source_concept_id in self.relationships:
            if relation_key in self.relationships[source_concept_id]:
                self.relationships[source_concept_id][relation_key].discard(target_concept_id)
                
                # Clean up empty sets
                if not self.relationships[source_concept_id][relation_key]:
                    del self.relationships[source_concept_id][relation_key]
                
                if not self.relationships[source_concept_id]:
                    del self.relationships[source_concept_id]
        
        # Remove reverse relationship
        if target_concept_id in self.reverse_relationships:
            if relation_key in self.reverse_relationships[target_concept_id]:
                self.reverse_relationships[target_concept_id][relation_key].discard(source_concept_id)
                
                # Clean up empty sets
                if not self.reverse_relationships[target_concept_id][relation_key]:
                    del self.reverse_relationships[target_concept_id][relation_key]
                
                if not self.reverse_relationships[target_concept_id]:
                    del self.reverse_relationships[target_concept_id]
    
    def get_related_concepts(self, concept_id: str, relation_type: SemanticRelationType = None,
                           direction: str = "outgoing") -> Dict[str, Set[str]]:
        """Get concepts related to a given concept"""
        
        if direction == "outgoing":
            relationships = self.relationships.get(concept_id, {})
        elif direction == "incoming":
            relationships = self.reverse_relationships.get(concept_id, {})
        else:  # both
            outgoing = self.relationships.get(concept_id, {})
            incoming = self.reverse_relationships.get(concept_id, {})
            
            # Merge both directions
            relationships = defaultdict(set)
            
            for rel_type, targets in outgoing.items():
                relationships[rel_type].update(targets)
            
            for rel_type, sources in incoming.items():
                relationships[f"inverse_{rel_type}"].update(sources)
        
        if relation_type:
            relation_key = relation_type.value
            return {relation_key: relationships.get(relation_key, set())}
        
        return dict(relationships)
    
    def find_path(self, source_concept_id: str, target_concept_id: str,
                 max_depth: int = 5) -> Optional[List[Tuple[str, str]]]:
        """Find a path between two concepts through relationships"""
        
        # BFS to find shortest path
        queue = deque([(source_concept_id, [])])
        visited = set()
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_concept_id:
                return path
            
            if current_id in visited or len(path) >= max_depth:
                continue
            
            visited.add(current_id)
            
            # Explore all outgoing relationships
            for relation_type, targets in self.relationships.get(current_id, {}).items():
                for target_id in targets:
                    if target_id not in visited:
                        new_path = path + [(current_id, relation_type)]
                        queue.append((target_id, new_path))
        
        return None
    
    def get_semantic_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """Calculate semantic similarity between two concepts"""
        
        # Get all relationships for both concepts
        rel1 = self.get_related_concepts(concept_id1, direction="both")
        rel2 = self.get_related_concepts(concept_id2, direction="both")
        
        # Calculate Jaccard similarity of related concepts
        all_related1 = set()
        for targets in rel1.values():
            all_related1.update(targets)
        
        all_related2 = set()
        for targets in rel2.values():
            all_related2.update(targets)
        
        if not all_related1 and not all_related2:
            return 0.0
        
        intersection = len(all_related1.intersection(all_related2))
        union = len(all_related1.union(all_related2))
        
        return intersection / union if union > 0 else 0.0

class SemanticStorage:
    """Storage backend for semantic memory"""
    
    def __init__(self, db_path: str = "semantic_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger("SemanticStorage")
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create concepts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    concept_type TEXT NOT NULL,
                    definition TEXT,
                    description TEXT,
                    properties_json TEXT,
                    attributes_json TEXT,
                    parent_concepts_json TEXT,
                    child_concepts_json TEXT,
                    domain TEXT,
                    categories_json TEXT,
                    relationships_json TEXT,
                    facts_json TEXT,
                    examples_json TEXT,
                    created_at TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            ''')
            
            # Create knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_knowledge (
                    id TEXT PRIMARY KEY,
                    knowledge_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    subject_concept_id TEXT,
                    object_concept_id TEXT,
                    related_concepts_json TEXT,
                    domain TEXT,
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    created_at TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_type ON concepts(concept_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_domain ON concepts(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_type ON semantic_knowledge(knowledge_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON semantic_knowledge(domain)')
            
            conn.commit()
    
    async def store_concept(self, concept: Concept) -> bool:
        """Store a concept"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO concepts (
                            id, name, concept_type, definition, description,
                            properties_json, attributes_json, parent_concepts_json,
                            child_concepts_json, domain, categories_json,
                            relationships_json, facts_json, examples_json,
                            created_at, last_updated, confidence, access_count,
                            last_accessed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        concept.id,
                        concept.name,
                        concept.concept_type.value,
                        concept.definition,
                        concept.description,
                        json.dumps(concept.properties),
                        json.dumps(list(concept.attributes)),
                        json.dumps(list(concept.parent_concepts)),
                        json.dumps(list(concept.child_concepts)),
                        concept.domain,
                        json.dumps(list(concept.categories)),
                        json.dumps({k: list(v) for k, v in concept.relationships.items()}),
                        json.dumps(concept.facts),
                        json.dumps(concept.examples),
                        concept.created_at.isoformat(),
                        concept.last_updated.isoformat(),
                        concept.confidence,
                        concept.access_count,
                        concept.last_accessed.isoformat() if concept.last_accessed else None
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to store concept {concept.id}: {e}")
            return False
    
    async def retrieve_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieve a concept by ID"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT * FROM concepts WHERE id = ?', (concept_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_concept(row)
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to retrieve concept {concept_id}: {e}")
            return None
    
    async def search_concepts(self, query: str = "", concept_types: List[ConceptType] = None,
                            domain: str = "", limit: int = 20) -> List[Concept]:
        """Search for concepts"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Build search query
                    conditions = []
                    params = []
                    
                    if query:
                        conditions.append('(name LIKE ? OR definition LIKE ? OR description LIKE ?)')
                        params.extend([f'%{query}%', f'%{query}%', f'%{query}%'])
                    
                    if concept_types:
                        type_placeholders = ','.join('?' * len(concept_types))
                        conditions.append(f'concept_type IN ({type_placeholders})')
                        params.extend([ct.value for ct in concept_types])
                    
                    if domain:
                        conditions.append('domain = ?')
                        params.append(domain)
                    
                    where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
                    
                    sql = f'''
                        SELECT * FROM concepts
                        {where_clause}
                        ORDER BY access_count DESC, name ASC
                        LIMIT ?
                    '''
                    params.append(limit)
                    
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    return [self._row_to_concept(row) for row in rows]
                    
        except Exception as e:
            self.logger.error(f"Failed to search concepts: {e}")
            return []
    
    def _row_to_concept(self, row: Tuple) -> Concept:
        """Convert database row to Concept object"""
        
        (id, name, concept_type, definition, description, properties_json,
         attributes_json, parent_concepts_json, child_concepts_json, domain,
         categories_json, relationships_json, facts_json, examples_json,
         created_at, last_updated, confidence, access_count, last_accessed) = row
        
        # Parse JSON fields
        relationships_dict = json.loads(relationships_json or '{}')
        relationships = {k: set(v) for k, v in relationships_dict.items()}
        
        concept = Concept(
            id=id,
            name=name,
            concept_type=ConceptType(concept_type),
            definition=definition or "",
            description=description or "",
            properties=json.loads(properties_json or '{}'),
            attributes=set(json.loads(attributes_json or '[]')),
            parent_concepts=set(json.loads(parent_concepts_json or '[]')),
            child_concepts=set(json.loads(child_concepts_json or '[]')),
            domain=domain or "",
            categories=set(json.loads(categories_json or '[]')),
            relationships=relationships,
            facts=json.loads(facts_json or '[]'),
            examples=json.loads(examples_json or '[]'),
            created_at=datetime.fromisoformat(created_at),
            last_updated=datetime.fromisoformat(last_updated),
            confidence=confidence,
            access_count=access_count,
            last_accessed=datetime.fromisoformat(last_accessed) if last_accessed else None
        )
        
        return concept

class SemanticMemoryManager:
    """Complete semantic memory management system"""
    
    def __init__(self, db_path: str = "semantic_memory.db"):
        # Core components
        self.storage = SemanticStorage(db_path)
        self.hierarchy = ConceptHierarchy()
        self.network = SemanticNetwork()
        
        # Cache for frequently accessed concepts
        self.concept_cache: Dict[str, Concept] = {}
        self.cache_size_limit = 1000
        
        # Domain organization
        self.domains: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'concepts_created': 0,
            'concepts_accessed': 0,
            'relationships_created': 0,
            'inferences_made': 0
        }
        
        self.logger = logging.getLogger("SemanticMemoryManager")
    
    async def initialize(self) -> None:
        """Initialize the semantic memory system"""
        
        # Load existing concepts to rebuild in-memory structures
        await self._rebuild_memory_structures()
        
        self.logger.info("Semantic memory manager initialized")
    
    async def _rebuild_memory_structures(self) -> None:
        """Rebuild in-memory structures from stored data"""
        
        # Get all concepts
        all_concepts = await self.storage.search_concepts(limit=10000)
        
        for concept in all_concepts:
            # Rebuild hierarchy
            for parent_id in concept.parent_concepts:
                self.hierarchy.add_is_a_relationship(concept.id, parent_id)
            
            # Rebuild semantic network
            for relation_type, targets in concept.relationships.items():
                try:
                    semantic_relation = SemanticRelationType(relation_type)
                    for target_id in targets:
                        self.network.add_relationship(concept.id, semantic_relation, target_id)
                except ValueError:
                    # Skip unknown relation types
                    pass
            
            # Rebuild domain index
            if concept.domain:
                self.domains[concept.domain].add(concept.id)
    
    async def create_concept(self, name: str, concept_type: ConceptType,
                           definition: str = "", description: str = "",
                           domain: str = "", properties: Dict[str, Any] = None) -> str:
        """Create a new concept"""
        
        concept = Concept(
            id="",
            name=name,
            concept_type=concept_type,
            definition=definition,
            description=description,
            domain=domain,
            properties=properties or {}
        )
        
        success = await self.storage.store_concept(concept)
        
        if success:
            # Add to cache
            self.concept_cache[concept.id] = concept
            
            # Add to domain index
            if domain:
                self.domains[domain].add(concept.id)
            
            self.stats['concepts_created'] += 1
            
            self.logger.debug(f"Created concept: {name}")
            
            return concept.id
        else:
            raise Exception("Failed to create concept")
    
    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        
        # Check cache first
        if concept_id in self.concept_cache:
            concept = self.concept_cache[concept_id]
            concept.access()
            await self.storage.store_concept(concept)
            self.stats['concepts_accessed'] += 1
            return concept
        
        # Load from storage
        concept = await self.storage.retrieve_concept(concept_id)
        
        if concept:
            concept.access()
            await self.storage.store_concept(concept)
            
            # Add to cache
            self._add_to_cache(concept)
            
            self.stats['concepts_accessed'] += 1
        
        return concept
    
    async def find_concept_by_name(self, name: str, domain: str = "") -> Optional[Concept]:
        """Find a concept by name"""
        
        concepts = await self.storage.search_concepts(query=name, domain=domain, limit=1)
        
        if concepts and concepts[0].name.lower() == name.lower():
            return await self.get_concept(concepts[0].id)
        
        return None
    
    async def add_is_a_relationship(self, child_concept_id: str, parent_concept_id: str) -> bool:
        """Add IS-A relationship between concepts"""
        
        # Get both concepts
        child_concept = await self.get_concept(child_concept_id)
        parent_concept = await self.get_concept(parent_concept_id)
        
        if not child_concept or not parent_concept:
            return False
        
        # Add to hierarchy
        self.hierarchy.add_is_a_relationship(child_concept_id, parent_concept_id)
        
        # Update concept relationships
        child_concept.parent_concepts.add(parent_concept_id)
        parent_concept.child_concepts.add(child_concept_id)
        
        child_concept.add_relationship(SemanticRelationType.IS_A, parent_concept_id)
        
        # Inherit properties from parent
        child_concept.inherit_properties(parent_concept)
        
        # Store updated concepts
        await self.storage.store_concept(child_concept)
        await self.storage.store_concept(parent_concept)
        
        self.stats['relationships_created'] += 1
        
        return True
    
    async def add_semantic_relationship(self, source_concept_id: str, 
                                      relation_type: SemanticRelationType,
                                      target_concept_id: str) -> bool:
        """Add a semantic relationship between concepts"""
        
        # Get concepts
        source_concept = await self.get_concept(source_concept_id)
        target_concept = await self.get_concept(target_concept_id)
        
        if not source_concept or not target_concept:
            return False
        
        # Add to semantic network
        self.network.add_relationship(source_concept_id, relation_type, target_concept_id)
        
        # Update concept
        source_concept.add_relationship(relation_type, target_concept_id)
        
        # Store updated concept
        await self.storage.store_concept(source_concept)
        
        self.stats['relationships_created'] += 1
        
        return True
    
    async def search_concepts(self, query: str, concept_types: List[ConceptType] = None,
                            domain: str = "", limit: int = 20) -> List[Concept]:
        """Search for concepts"""
        
        concepts = await self.storage.search_concepts(query, concept_types, domain, limit)
        
        # Update access for found concepts
        for concept in concepts:
            concept.access()
            self._add_to_cache(concept)
        
        self.stats['concepts_accessed'] += len(concepts)
        
        return concepts
    
    async def get_concept_hierarchy(self, concept_id: str) -> Dict[str, Any]:
        """Get hierarchical information for a concept"""
        
        concept = await self.get_concept(concept_id)
        
        if not concept:
            return {}
        
        ancestors = self.hierarchy.get_ancestors(concept_id)
        descendants = self.hierarchy.get_descendants(concept_id)
        
        # Get concept details for hierarchy
        ancestor_concepts = []
        for ancestor_id in ancestors:
            ancestor = await self.get_concept(ancestor_id)
            if ancestor:
                ancestor_concepts.append({'id': ancestor_id, 'name': ancestor.name})
        
        descendant_concepts = []
        for descendant_id in descendants:
            descendant = await self.get_concept(descendant_id)
            if descendant:
                descendant_concepts.append({'id': descendant_id, 'name': descendant.name})
        
        return {
            'concept': {'id': concept.id, 'name': concept.name},
            'ancestors': ancestor_concepts,
            'descendants': descendant_concepts,
            'depth': self.hierarchy.get_hierarchy_depth(concept_id)
        }
    
    async def infer_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Infer new relationships for a concept"""
        
        concept = await self.get_concept(concept_id)
        
        if not concept:
            return []
        
        inferences = []
        
        # Infer from hierarchy (property inheritance)
        ancestors = self.hierarchy.get_ancestors(concept_id)
        
        for ancestor_id in ancestors:
            ancestor = await self.get_concept(ancestor_id)
            
            if ancestor:
                # Inherit properties that are missing
                for prop_name, prop_value in ancestor.properties.items():
                    if prop_name not in concept.properties:
                        inferences.append({
                            'type': 'property_inheritance',
                            'property': prop_name,
                            'value': prop_value,
                            'source': ancestor.name,
                            'confidence': 0.8
                        })
        
        # Infer from similar concepts
        similar_concepts = await self._find_similar_concepts(concept_id)
        
        for similar_concept_id, similarity in similar_concepts:
            if similarity > 0.7:  # High similarity threshold
                similar_concept = await self.get_concept(similar_concept_id)
                
                if similar_concept:
                    # Suggest similar relationships
                    for relation_type, targets in similar_concept.relationships.items():
                        if relation_type not in concept.relationships:
                            inferences.append({
                                'type': 'similarity_inference',
                                'relation_type': relation_type,
                                'suggested_targets': list(targets),
                                'source': similar_concept.name,
                                'confidence': similarity * 0.6
                            })
        
        self.stats['inferences_made'] += len(inferences)
        
        return inferences
    
    async def _find_similar_concepts(self, concept_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Find concepts similar to the given concept"""
        
        concept = await self.get_concept(concept_id)
        
        if not concept:
            return []
        
        # Get concepts from same domain
        domain_concepts = []
        
        if concept.domain:
            domain_concept_ids = self.domains.get(concept.domain, set())
            
            for other_id in domain_concept_ids:
                if other_id != concept_id:
                    similarity = self.network.get_semantic_similarity(concept_id, other_id)
                    domain_concepts.append((other_id, similarity))
        
        # Sort by similarity and return top results
        domain_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return domain_concepts[:limit]
    
    async def get_domain_overview(self, domain: str) -> Dict[str, Any]:
        """Get overview of concepts in a domain"""
        
        domain_concept_ids = self.domains.get(domain, set())
        
        if not domain_concept_ids:
            return {'domain': domain, 'concept_count': 0}
        
        # Get concept details
        concepts = []
        concept_types = defaultdict(int)
        
        for concept_id in domain_concept_ids:
            concept = await self.get_concept(concept_id)
            
            if concept:
                concepts.append({
                    'id': concept.id,
                    'name': concept.name,
                    'type': concept.concept_type.value
                })
                
                concept_types[concept.concept_type.value] += 1
        
        # Find top-level concepts (concepts with no parents in domain)
        top_level_concepts = []
        
        for concept_id in domain_concept_ids:
            ancestors = self.hierarchy.get_ancestors(concept_id)
            domain_ancestors = ancestors.intersection(domain_concept_ids)
            
            if not domain_ancestors:  # No ancestors in this domain
                concept = await self.get_concept(concept_id)
                if concept:
                    top_level_concepts.append({
                        'id': concept.id,
                        'name': concept.name
                    })
        
        return {
            'domain': domain,
            'concept_count': len(domain_concept_ids),
            'concept_type_distribution': dict(concept_types),
            'top_level_concepts': top_level_concepts,
            'all_concepts': concepts
        }
    
    def _add_to_cache(self, concept: Concept) -> None:
        """Add concept to cache with size management"""
        
        self.concept_cache[concept.id] = concept
        
        # Manage cache size
        if len(self.concept_cache) > self.cache_size_limit:
            # Remove least recently accessed concepts
            concepts_by_access = sorted(
                self.concept_cache.values(),
                key=lambda c: c.last_accessed or datetime.min
            )
            
            # Remove oldest 10% of concepts
            to_remove = concepts_by_access[:len(concepts_by_access) // 10]
            
            for old_concept in to_remove:
                del self.concept_cache[old_concept.id]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            'system_statistics': self.stats,
            'cache_statistics': {
                'cache_size': len(self.concept_cache),
                'cache_limit': self.cache_size_limit
            },
            'domain_statistics': {
                'total_domains': len(self.domains),
                'concepts_per_domain': {domain: len(concepts) for domain, concepts in self.domains.items()}
            },
            'hierarchy_statistics': {
                'total_hierarchy_relationships': len(self.hierarchy.hierarchy_graph),
                'concepts_with_parents': len(self.hierarchy.reverse_hierarchy)
            },
            'network_statistics': {
                'concepts_with_relationships': len(self.network.relationships),
                'total_relationship_instances': sum(
                    sum(len(targets) for targets in relations.values())
                    for relations in self.network.relationships.values()
                )
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_concept_creation_and_hierarchy():
    """Demo: Creating concepts and building hierarchies"""
    print("\nDEMO 1: CONCEPT CREATION AND HIERARCHY")
    print("=" * 50)
    
    memory_manager = SemanticMemoryManager("demo_semantic.db")
    await memory_manager.initialize()
    
    print("Creating a concept hierarchy for animals:")
    
    # Create top-level concept
    animal_id = await memory_manager.create_concept(
        "Animal",
        ConceptType.CATEGORY,
        definition="A living organism that feeds on organic matter",
        description="Multicellular organisms that can move and respond to environment",
        domain="biology",
        properties={
            "multicellular": True,
            "can_move": True,
            "needs_food": True,
            "has_metabolism": True
        }
    )
    print(f"  Created concept: Animal")
    
    # Create mammal concept
    mammal_id = await memory_manager.create_concept(
        "Mammal",
        ConceptType.CATEGORY,
        definition="Warm-blooded vertebrate animal with hair or fur",
        description="Animals that feed milk to their young and have hair",
        domain="biology",
        properties={
            "warm_blooded": True,
            "has_hair": True,
            "feeds_milk": True,
            "gives_live_birth": True
        }
    )
    print(f"  Created concept: Mammal")
    
    # Create dog concept
    dog_id = await memory_manager.create_concept(
        "Dog",
        ConceptType.ENTITY,
        definition="Domesticated mammal and companion animal",
        description="Loyal companion animals bred from wolves",
        domain="biology",
        properties={
            "domesticated": True,
            "loyal": True,
            "carnivorous": True,
            "pack_animal": True
        }
    )
    print(f"  Created concept: Dog")
    
    # Create specific dog breed
    golden_retriever_id = await memory_manager.create_concept(
        "Golden Retriever",
        ConceptType.ENTITY,
        definition="Large-sized breed of dog known for golden coat",
        description="Friendly and intelligent dog breed, great with families",
        domain="biology",
        properties={
            "size": "large",
            "coat_color": "golden",
            "temperament": "friendly",
            "intelligence": "high"
        }
    )
    print(f"  Created concept: Golden Retriever")
    
    # Build hierarchy relationships
    print(f"\nBuilding IS-A hierarchy:")
    
    await memory_manager.add_is_a_relationship(mammal_id, animal_id)
    print(f"  Mammal IS-A Animal")
    
    await memory_manager.add_is_a_relationship(dog_id, mammal_id)
    print(f"  Dog IS-A Mammal")
    
    await memory_manager.add_is_a_relationship(golden_retriever_id, dog_id)
    print(f"  Golden Retriever IS-A Dog")
    
    # Verify hierarchy and property inheritance
    print(f"\nVerifying hierarchy and property inheritance:")
    
    golden_retriever = await memory_manager.get_concept(golden_retriever_id)
    
    print(f"  Golden Retriever properties:")
    for prop, value in golden_retriever.properties.items():
        print(f"    {prop}: {value}")
    
    # Get hierarchy information
    hierarchy_info = await memory_manager.get_concept_hierarchy(golden_retriever_id)
    
    print(f"\n  Hierarchy for Golden Retriever:")
    print(f"    Depth: {hierarchy_info['depth']}")
    print(f"    Ancestors: {[a['name'] for a in hierarchy_info['ancestors']]}")
    print(f"    Descendants: {[d['name'] for d in hierarchy_info['descendants']]}")

async def demo_semantic_relationships():
    """Demo: Creating semantic relationships between concepts"""
    print("\nDEMO 2: SEMANTIC RELATIONSHIPS")
    print("=" * 50)
    
    memory_manager = SemanticMemoryManager("demo_relationships.db")
    await memory_manager.initialize()
    
    print("Creating concepts and semantic relationships:")
    
    # Create concepts for demonstration
    concepts_to_create = [
        ("Car", ConceptType.ENTITY, "Motorized vehicle for transportation", "automotive"),
        ("Engine", ConceptType.ENTITY, "Machine that converts fuel to mechanical energy", "automotive"),
        ("Wheel", ConceptType.ENTITY, "Circular object that revolves on an axle", "automotive"),
        ("Transportation", ConceptType.ABSTRACT, "Movement of people or goods from one place to another", "general"),
        ("Fuel", ConceptType.ENTITY, "Material used to produce energy", "automotive"),
        ("Road", ConceptType.ENTITY, "Path for vehicles to travel", "infrastructure")
    ]
    
    concept_ids = {}
    
    for name, concept_type, definition, domain in concepts_to_create:
        concept_id = await memory_manager.create_concept(
            name, concept_type, definition=definition, domain=domain
        )
        concept_ids[name] = concept_id
        print(f"  Created: {name}")
    
    # Create semantic relationships
    print(f"\nCreating semantic relationships:")
    
    relationships_to_create = [
        ("Car", SemanticRelationType.HAS_A, "Engine"),
        ("Car", SemanticRelationType.HAS_A, "Wheel"),
        ("Car", SemanticRelationType.USED_FOR, "Transportation"),
        ("Car", SemanticRelationType.REQUIRES, "Fuel"),
        ("Car", SemanticRelationType.REQUIRES, "Road"),
        ("Engine", SemanticRelationType.PART_OF, "Car"),
        ("Engine", SemanticRelationType.REQUIRES, "Fuel"),
        ("Wheel", SemanticRelationType.PART_OF, "Car")
    ]
    
    for source, relation, target in relationships_to_create:
        success = await memory_manager.add_semantic_relationship(
            concept_ids[source], relation, concept_ids[target]
        )
        if success:
            print(f"  {source} {relation.value} {target}")
    
    # Explore relationships
    print(f"\nExploring semantic relationships:")
    
    # Get all relationships for Car
    car_concept = await memory_manager.get_concept(concept_ids["Car"])
    
    print(f"\n  Car relationships:")
    for relation_type, targets in car_concept.relationships.items():
        for target_id in targets:
            target_concept = await memory_manager.get_concept(target_id)
            if target_concept:
                print(f"    {relation_type}: {target_concept.name}")
    
    # Find semantic path between concepts
    car_to_fuel_path = memory_manager.network.find_path(concept_ids["Car"], concept_ids["Fuel"])
    
    if car_to_fuel_path:
        print(f"\n  Semantic path from Car to Fuel:")
        for i, (concept_id, relation) in enumerate(car_to_fuel_path):
            concept = await memory_manager.get_concept(concept_id)
            if concept:
                print(f"    {i+1}. {concept.name} --{relation}-->")
        
        # Add final concept
        fuel_concept = await memory_manager.get_concept(concept_ids["Fuel"])
        print(f"    {len(car_to_fuel_path)+1}. {fuel_concept.name}")
    
    # Calculate semantic similarity
    car_engine_similarity = memory_manager.network.get_semantic_similarity(
        concept_ids["Car"], concept_ids["Engine"]
    )
    
    car_road_similarity = memory_manager.network.get_semantic_similarity(
        concept_ids["Car"], concept_ids["Road"]
    )
    
    print(f"\n  Semantic similarities:")
    print(f"    Car ↔ Engine: {car_engine_similarity:.3f}")
    print(f"    Car ↔ Road: {car_road_similarity:.3f}")

async def demo_knowledge_inference():
    """Demo: Knowledge inference and relationship discovery"""
    print("\nDEMO 3: KNOWLEDGE INFERENCE")
    print("=" * 50)
    
    memory_manager = SemanticMemoryManager("demo_inference.db")
    await memory_manager.initialize()
    
    print("Setting up knowledge base for inference:")
    
    # Create concepts for programming domain
    programming_concepts = [
        ("Programming Language", ConceptType.CATEGORY, "Formal language for writing computer programs", "computer_science"),
        ("Python", ConceptType.ENTITY, "High-level programming language", "computer_science"),
        ("Java", ConceptType.ENTITY, "Object-oriented programming language", "computer_science"),
        ("Object-Oriented Programming", ConceptType.ABSTRACT, "Programming paradigm based on objects", "computer_science"),
        ("Variable", ConceptType.ENTITY, "Storage location with an associated name", "computer_science"),
        ("Function", ConceptType.ENTITY, "Reusable block of code that performs a task", "computer_science"),
        ("Class", ConceptType.ENTITY, "Template for creating objects in OOP", "computer_science"),
        ("Library", ConceptType.ENTITY, "Collection of pre-written code", "computer_science")
    ]
    
    concept_ids = {}
    
    for name, concept_type, definition, domain in programming_concepts:
        concept_id = await memory_manager.create_concept(
            name, concept_type, definition=definition, domain=domain,
            properties={"complexity": "intermediate" if "Programming" in name else "basic"}
        )
        concept_ids[name] = concept_id
        print(f"  Created: {name}")
    
    # Build relationships
    print(f"\nBuilding knowledge relationships:")
    
    # Hierarchical relationships
    await memory_manager.add_is_a_relationship(concept_ids["Python"], concept_ids["Programming Language"])
    await memory_manager.add_is_a_relationship(concept_ids["Java"], concept_ids["Programming Language"])
    print(f"  Built language hierarchy")
    
    # Semantic relationships
    relationships = [
        ("Python", SemanticRelationType.HAS_A, "Variable"),
        ("Python", SemanticRelationType.HAS_A, "Function"),
        ("Java", SemanticRelationType.HAS_A, "Variable"),
        ("Java", SemanticRelationType.HAS_A, "Function"),
        ("Java", SemanticRelationType.HAS_A, "Class"),
        ("Java", SemanticRelationType.USED_FOR, "Object-Oriented Programming"),
        ("Class", SemanticRelationType.PART_OF, "Object-Oriented Programming"),
        ("Programming Language", SemanticRelationType.HAS_A, "Library")
    ]
    
    for source, relation, target in relationships:
        await memory_manager.add_semantic_relationship(
            concept_ids[source], relation, concept_ids[target]
        )
    
    print(f"  Built semantic relationships")
    
    # Perform inference
    print(f"\nPerforming knowledge inference:")
    
    # Infer relationships for Python
    python_inferences = await memory_manager.infer_relationships(concept_ids["Python"])
    
    print(f"\n  Inferences for Python:")
    for inference in python_inferences:
        print(f"    Type: {inference['type']}")
        
        if inference['type'] == 'property_inheritance':
            print(f"      Property: {inference['property']} = {inference['value']}")
            print(f"      Inherited from: {inference['source']}")
        
        elif inference['type'] == 'similarity_inference':
            print(f"      Suggested relation: {inference['relation_type']}")
            print(f"      Based on similarity to: {inference['source']}")
        
        print(f"      Confidence: {inference['confidence']:.3f}")
        print()
    
    # Test inference through hierarchy
    python_concept = await memory_manager.get_concept(concept_ids["Python"])
    
    print(f"  Python properties after inference:")
    for prop, value in python_concept.properties.items():
        print(f"    {prop}: {value}")

async def demo_domain_organization():
    """Demo: Domain-based knowledge organization"""
    print("\nDEMO 4: DOMAIN ORGANIZATION")
    print("=" * 50)
    
    memory_manager = SemanticMemoryManager("demo_domains.db")
    await memory_manager.initialize()
    
    print("Creating concepts across multiple domains:")
    
    # Create concepts in different domains
    domain_concepts = {
        "biology": [
            ("Cell", ConceptType.ENTITY, "Basic unit of life"),
            ("DNA", ConceptType.ENTITY, "Genetic material"),
            ("Protein", ConceptType.ENTITY, "Large molecule composed of amino acids"),
            ("Photosynthesis", ConceptType.PROCESS, "Process of converting light to energy")
        ],
        "chemistry": [
            ("Atom", ConceptType.ENTITY, "Smallest unit of matter"),
            ("Molecule", ConceptType.ENTITY, "Group of atoms bonded together"),
            ("Chemical Reaction", ConceptType.PROCESS, "Process that changes chemical composition"),
            ("Catalyst", ConceptType.ENTITY, "Substance that speeds up reactions")
        ],
        "physics": [
            ("Force", ConceptType.ABSTRACT, "Interaction that changes object motion"),
            ("Energy", ConceptType.ABSTRACT, "Capacity to do work"),
            ("Gravity", ConceptType.FORCE, "Attractive force between masses"),
            ("Electromagnetic Field", ConceptType.ABSTRACT, "Physical field produced by charges")
        ]
    }
    
    concept_ids = {}
    
    for domain, concepts in domain_concepts.items():
        print(f"\n{domain.title()} domain:")
        
        for name, concept_type, definition in concepts:
            concept_id = await memory_manager.create_concept(
                name, concept_type, definition=definition, domain=domain
            )
            concept_ids[f"{domain}_{name}"] = concept_id
            print(f"  Created: {name}")
    
    # Create cross-domain relationships
    print(f"\nCreating cross-domain relationships:")
    
    cross_domain_relationships = [
        ("chemistry_Molecule", SemanticRelationType.HAS_A, "chemistry_Atom"),
        ("biology_DNA", SemanticRelationType.IS_A, "chemistry_Molecule"),
        ("biology_Protein", SemanticRelationType.IS_A, "chemistry_Molecule"),
        ("biology_Photosynthesis", SemanticRelationType.REQUIRES, "physics_Energy"),
        ("chemistry_Chemical Reaction", SemanticRelationType.REQUIRES, "physics_Energy")
    ]
    
    for source_key, relation, target_key in cross_domain_relationships:
        success = await memory_manager.add_semantic_relationship(
            concept_ids[source_key], relation, concept_ids[target_key]
        )
        
        if success:
            source_name = source_key.split("_", 1)[1]
            target_name = target_key.split("_", 1)[1]
            print(f"  {source_name} {relation.value} {target_name}")
    
    # Analyze domains
    print(f"\nDomain analysis:")
    
    for domain in domain_concepts.keys():
        overview = await memory_manager.get_domain_overview(domain)
        
        print(f"\n{domain.title()} Domain Overview:")
        print(f"  Total concepts: {overview['concept_count']}")
        print(f"  Concept types: {overview['concept_type_distribution']}")
        print(f"  Top-level concepts: {[c['name'] for c in overview['top_level_concepts']]}")
    
    # Search within specific domain
    print(f"\nDomain-specific search:")
    
    biology_search = await memory_manager.search_concepts("molecule", domain="biology")
    chemistry_search = await memory_manager.search_concepts("molecule", domain="chemistry")
    
    print(f"  'molecule' in biology: {[c.name for c in biology_search]}")
    print(f"  'molecule' in chemistry: {[c.name for c in chemistry_search]}")

async def demo_concept_search_and_retrieval():
    """Demo: Advanced concept search and retrieval"""
    print("\nDEMO 5: CONCEPT SEARCH AND RETRIEVAL")
    print("=" * 50)
    
    memory_manager = SemanticMemoryManager("demo_search.db")
    await memory_manager.initialize()
    
    print("Creating a diverse knowledge base:")
    
    # Create concepts with rich metadata
    diverse_concepts = [
        {
            "name": "Artificial Intelligence",
            "type": ConceptType.ABSTRACT,
            "definition": "Intelligence demonstrated by machines",
            "domain": "computer_science",
            "properties": {"complexity": "high", "field": "technology", "impact": "transformative"},
            "facts": ["AI can perform tasks requiring human intelligence", "AI includes machine learning"]
        },
        {
            "name": "Machine Learning",
            "type": ConceptType.PROCESS,
            "definition": "Method of data analysis that automates analytical model building",
            "domain": "computer_science", 
            "properties": {"complexity": "high", "requires": "data", "output": "predictions"},
            "facts": ["ML is a subset of AI", "ML learns from data without explicit programming"]
        },
        {
            "name": "Neural Network",
            "type": ConceptType.ENTITY,
            "definition": "Computing system inspired by biological neural networks",
            "domain": "computer_science",
            "properties": {"inspired_by": "brain", "complexity": "very_high", "type": "model"},
            "facts": ["Neural networks have layers of interconnected nodes", "Deep learning uses neural networks"]
        },
        {
            "name": "Data Science",
            "type": ConceptType.ABSTRACT,
            "definition": "Interdisciplinary field using scientific methods to extract insights from data",
            "domain": "computer_science",
            "properties": {"interdisciplinary": True, "involves": "statistics", "goal": "insights"},
            "facts": ["Data science combines programming and statistics", "Data scientists analyze large datasets"]
        }
    ]
    
    concept_ids = {}
    
    for concept_data in diverse_concepts:
        concept_id = await memory_manager.create_concept(
            concept_data["name"],
            concept_data["type"], 
            definition=concept_data["definition"],
            domain=concept_data["domain"],
            properties=concept_data["properties"]
        )
        
        # Add facts to concept
        concept = await memory_manager.get_concept(concept_id)
        concept.facts = concept_data["facts"]
        await memory_manager.storage.store_concept(concept)
        
        concept_ids[concept_data["name"]] = concept_id
        print(f"  Created: {concept_data['name']}")
    
    # Build relationships
    print(f"\nBuilding relationships:")
    
    relationships = [
        ("Machine Learning", SemanticRelationType.PART_OF, "Artificial Intelligence"),
        ("Neural Network", SemanticRelationType.USED_FOR, "Machine Learning"),
        ("Data Science", SemanticRelationType.USES, "Machine Learning")
    ]
    
    for source, relation, target in relationships:
        await memory_manager.add_semantic_relationship(
            concept_ids[source], relation, concept_ids[target]
        )
        print(f"  {source} {relation.value} {target}")
    
    # Perform various searches
    print(f"\nPerforming various search queries:")
    
    # Search by keyword
    print(f"\n1. Keyword search for 'learning':")
    learning_concepts = await memory_manager.search_concepts("learning")
    for concept in learning_concepts:
        print(f"   {concept.name}: {concept.definition}")
    
    # Search by concept type
    print(f"\n2. Search for abstract concepts:")
    abstract_concepts = await memory_manager.search_concepts("", concept_types=[ConceptType.ABSTRACT])
    for concept in abstract_concepts:
        print(f"   {concept.name} ({concept.concept_type.value})")
    
    # Search by domain
    print(f"\n3. Search in computer science domain:")
    cs_concepts = await memory_manager.search_concepts("", domain="computer_science")
    for concept in cs_concepts:
        print(f"   {concept.name}")
    
    # Find concept by exact name
    print(f"\n4. Find specific concept:")
    ai_concept = await memory_manager.find_concept_by_name("Artificial Intelligence")
    if ai_concept:
        print(f"   Found: {ai_concept.name}")
        print(f"   Definition: {ai_concept.definition}")
        print(f"   Properties: {ai_concept.properties}")
        print(f"   Facts: {ai_concept.facts}")
    
    # Show system statistics
    print(f"\nSystem statistics:")
    stats = memory_manager.get_system_statistics()
    
    print(f"  Concepts created: {stats['system_statistics']['concepts_created']}")
    print(f"  Concepts accessed: {stats['system_statistics']['concepts_accessed']}")
    print(f"  Relationships created: {stats['system_statistics']['relationships_created']}")
    print(f"  Cache size: {stats['cache_statistics']['cache_size']}")
    print(f"  Total domains: {stats['domain_statistics']['total_domains']}")

async def main():
    """
    Demonstrate Semantic Memory Manager for organizing general knowledge and concepts
    
    WHAT YOU'LL LEARN:
    ================
    1. How to create and organize concepts with hierarchical relationships
    2. How to build semantic networks with meaningful relationships
    3. How to implement knowledge inference and property inheritance
    4. How to organize knowledge by domains and categories
    5. How to perform advanced concept search and retrieval
    6. How to create complete semantic memory systems for AI reasoning
    
    REAL WORLD APPLICATIONS:
    =======================
    - Knowledge-based AI systems for intelligent question answering
    - Educational platforms with structured domain knowledge
    - Expert systems for medical, legal, or technical domains
    - Semantic search engines understanding concept relationships
    - AI assistants with deep understanding of topic hierarchies
    - Research tools organizing scientific knowledge and discoveries
    """
    
    print("SEMANTIC MEMORY MANAGER DEMONSTRATION")
    print("Organizing general knowledge and concepts!")
    
    await demo_concept_creation_and_hierarchy()
    await demo_semantic_relationships()
    await demo_knowledge_inference()
    await demo_domain_organization()
    await demo_concept_search_and_retrieval()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Concept hierarchies enable logical knowledge organization")
    print("✓ Semantic relationships create meaningful knowledge networks")
    print("✓ Knowledge inference discovers implicit relationships and properties")
    print("✓ Domain organization enables scalable knowledge management")
    print("✓ Advanced search supports intelligent knowledge retrieval")
    print("✓ Complete systems enable AI reasoning with structured knowledge")
    print("\nTHE POWER OF SEMANTIC MEMORY:")
    print("- Enables AI systems to reason with structured knowledge like humans")
    print("- Supports natural language understanding through conceptual frameworks")
    print("- Provides foundation for intelligent question answering and inference")
    print("- Creates scalable knowledge organization for large-scale AI systems")
    print("- Enables efficient knowledge sharing and transfer between domains")

if __name__ == "__main__":
    asyncio.run(main())
