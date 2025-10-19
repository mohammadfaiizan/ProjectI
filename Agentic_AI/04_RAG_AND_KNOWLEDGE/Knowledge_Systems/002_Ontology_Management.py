#!/usr/bin/env python3
"""
Ontology Management: Structured Knowledge Schema and Reasoning
============================================================

WHAT IS THE PROBLEM?
==================
Knowledge without structure leads to chaos and inconsistency:
- Different systems use different terms for the same concepts
- Relationships lack formal definitions and constraints
- Knowledge cannot be validated for logical consistency
- Integration between systems becomes impossible
- Reasoning capabilities are severely limited
- Knowledge evolution becomes unmanageable

Example: Medical Terminology Chaos
UNSTRUCTURED APPROACH (Traditional):
- "Heart attack", "Myocardial infarction", "MI" used interchangeably
- No formal relationship between symptoms and diseases
- Different departments use incompatible terminologies
- Drug interactions cannot be automatically verified
- Treatment protocols lack formal logic
- Result: Medical errors, inconsistent care, missed diagnoses

REAL WORLD EXAMPLE:
=================
How does the Gene Ontology work?

GENE ONTOLOGY SYSTEM:
1. CONTROLLED VOCABULARY: Standardized terms for biological processes
2. HIERARCHICAL STRUCTURE: Parent-child relationships between concepts
3. FORMAL DEFINITIONS: Precise meaning of each term and relationship
4. LOGICAL CONSTRAINTS: Rules that prevent inconsistent annotations
5. EVIDENCE CODES: Tracking the source and quality of knowledge
6. CROSS-REFERENCES: Links to other biological databases
7. VERSIONING: Systematic evolution of the ontology over time

BENEFITS OF ONTOLOGY MANAGEMENT:
- Consistent terminology across systems and organizations
- Automated reasoning and inference capabilities
- Knowledge validation and consistency checking
- Seamless integration between different knowledge sources
- Systematic knowledge evolution and version control
- Enhanced search and discovery through semantic relationships

THE ONTOLOGY ADVANTAGE:
=====================
UNSTRUCTURED KNOWLEDGE: Terms → Confusion → Inconsistency
STRUCTURED ONTOLOGY: Concepts → Relationships → Reasoning → Intelligence

ONTOLOGY COMPONENTS:
==================
1. CONCEPTS: Abstract ideas or classes of entities
2. INSTANCES: Specific examples of concepts
3. PROPERTIES: Attributes and characteristics
4. RELATIONSHIPS: Connections between concepts
5. AXIOMS: Logical rules and constraints
6. INFERENCE RULES: Mechanisms for deriving new knowledge
7. CONSTRAINTS: Validity and consistency requirements

WHY THIS IS REVOLUTIONARY:
========================
- Enables machines to understand meaning, not just text
- Provides foundation for automated reasoning and inference
- Creates interoperable knowledge that works across systems
- Critical for building truly intelligent AI systems
- Enables systematic knowledge management at scale
- Powers semantic search and intelligent question answering
"""

import asyncio
import time
import json
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ConceptType(Enum):
    """Types of concepts in ontology"""
    ABSTRACT = "abstract"           # Abstract concepts
    CONCRETE = "concrete"           # Physical entities
    PROCESS = "process"             # Actions or procedures
    QUALITY = "quality"             # Attributes or properties
    ROLE = "role"                   # Functions or positions
    EVENT = "event"                 # Temporal occurrences
    RELATION = "relation"           # Relationship concepts

class PropertyType(Enum):
    """Types of properties"""
    DATATYPE = "datatype"           # Links to literal values
    OBJECT = "object"               # Links to other concepts
    ANNOTATION = "annotation"       # Metadata properties
    FUNCTIONAL = "functional"       # Single-valued properties
    INVERSE_FUNCTIONAL = "inverse_functional"  # Unique identifier properties

class RelationType(Enum):
    """Types of relationships in ontology"""
    IS_A = "is_a"                   # Subclass relationship
    PART_OF = "part_of"             # Composition relationship
    HAS_PART = "has_part"           # Reverse of part_of
    RELATED_TO = "related_to"       # General association
    DEPENDS_ON = "depends_on"       # Dependency relationship
    PRECEDES = "precedes"           # Temporal ordering
    CAUSES = "causes"               # Causal relationship
    EQUIVALENT_TO = "equivalent_to" # Equivalence relationship

class ConstraintType(Enum):
    """Types of logical constraints"""
    CARDINALITY = "cardinality"     # Number restrictions
    DOMAIN = "domain"               # Property domain restrictions
    RANGE = "range"                 # Property range restrictions
    DISJOINT = "disjoint"           # Mutual exclusion
    INVERSE = "inverse"             # Inverse properties
    TRANSITIVE = "transitive"       # Transitive properties
    SYMMETRIC = "symmetric"         # Symmetric properties

@dataclass
class Concept:
    """Represents a concept in the ontology"""
    
    id: str
    name: str
    concept_type: ConceptType
    
    # Core properties
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    description: str = ""
    
    # Hierarchy
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    
    # Properties and relationships
    properties: Dict[str, Any] = field(default_factory=dict)
    related_concepts: Dict[str, List[str]] = field(default_factory=dict)
    
    # Constraints and rules
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    axioms: List[str] = field(default_factory=list)
    
    # Metadata
    namespace: str = ""
    version: str = "1.0"
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Provenance
    sources: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.namespace}:{self.name.replace(' ', '_')}"
    
    def add_parent(self, parent_id: str) -> None:
        """Add parent concept"""
        if parent_id not in self.parent_concepts:
            self.parent_concepts.append(parent_id)
            self.updated_at = datetime.now()
    
    def add_child(self, child_id: str) -> None:
        """Add child concept"""
        if child_id not in self.child_concepts:
            self.child_concepts.append(child_id)
            self.updated_at = datetime.now()
    
    def add_property(self, property_name: str, property_value: Any, 
                    property_type: PropertyType = PropertyType.DATATYPE) -> None:
        """Add property to concept"""
        self.properties[property_name] = {
            'value': property_value,
            'type': property_type.value,
            'updated_at': datetime.now()
        }
        self.updated_at = datetime.now()
    
    def add_relationship(self, relation_type: str, target_concept_ids: List[str]) -> None:
        """Add relationship to other concepts"""
        if relation_type not in self.related_concepts:
            self.related_concepts[relation_type] = []
        
        for target_id in target_concept_ids:
            if target_id not in self.related_concepts[relation_type]:
                self.related_concepts[relation_type].append(target_id)
        
        self.updated_at = datetime.now()
    
    def add_constraint(self, constraint_type: ConstraintType, 
                      constraint_data: Dict[str, Any]) -> None:
        """Add logical constraint"""
        constraint = {
            'type': constraint_type.value,
            'data': constraint_data,
            'created_at': datetime.now()
        }
        self.constraints.append(constraint)
        self.updated_at = datetime.now()
    
    def add_axiom(self, axiom: str) -> None:
        """Add logical axiom"""
        if axiom not in self.axioms:
            self.axioms.append(axiom)
            self.updated_at = datetime.now()
    
    def get_ancestors(self, ontology: 'Ontology') -> Set[str]:
        """Get all ancestor concepts"""
        ancestors = set()
        to_visit = self.parent_concepts.copy()
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id not in ancestors:
                ancestors.add(current_id)
                
                # Add parents of current concept
                current_concept = ontology.get_concept(current_id)
                if current_concept:
                    to_visit.extend(current_concept.parent_concepts)
        
        return ancestors
    
    def get_descendants(self, ontology: 'Ontology') -> Set[str]:
        """Get all descendant concepts"""
        descendants = set()
        to_visit = self.child_concepts.copy()
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id not in descendants:
                descendants.add(current_id)
                
                # Add children of current concept
                current_concept = ontology.get_concept(current_id)
                if current_concept:
                    to_visit.extend(current_concept.child_concepts)
        
        return descendants
    
    def is_ancestor_of(self, other_concept_id: str, ontology: 'Ontology') -> bool:
        """Check if this concept is ancestor of another"""
        return other_concept_id in self.get_descendants(ontology)
    
    def is_descendant_of(self, other_concept_id: str, ontology: 'Ontology') -> bool:
        """Check if this concept is descendant of another"""
        return other_concept_id in self.get_ancestors(ontology)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert concept to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'concept_type': self.concept_type.value,
            'definition': self.definition,
            'synonyms': self.synonyms,
            'description': self.description,
            'parent_concepts': self.parent_concepts,
            'child_concepts': self.child_concepts,
            'properties': self.properties,
            'related_concepts': self.related_concepts,
            'constraints': self.constraints,
            'axioms': self.axioms,
            'namespace': self.namespace,
            'version': self.version,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'sources': self.sources,
            'evidence': self.evidence
        }

@dataclass
class Property:
    """Represents a property in the ontology"""
    
    id: str
    name: str
    property_type: PropertyType
    
    # Definition
    definition: str = ""
    description: str = ""
    
    # Domain and range
    domain: List[str] = field(default_factory=list)  # Concept IDs
    range: Union[List[str], str] = field(default_factory=list)  # Concept IDs or datatype
    
    # Property characteristics
    functional: bool = False          # Single-valued
    inverse_functional: bool = False  # Unique identifier
    transitive: bool = False         # A->B, B->C implies A->C
    symmetric: bool = False          # A->B implies B->A
    reflexive: bool = False          # A->A
    irreflexive: bool = False        # Not A->A
    
    # Relationships with other properties
    inverse_property: Optional[str] = None
    sub_properties: List[str] = field(default_factory=list)
    super_properties: List[str] = field(default_factory=list)
    
    # Metadata
    namespace: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.namespace}:{self.name.replace(' ', '_')}"
    
    def add_domain_concept(self, concept_id: str) -> None:
        """Add concept to domain"""
        if concept_id not in self.domain:
            self.domain.append(concept_id)
            self.updated_at = datetime.now()
    
    def add_range_concept(self, concept_id: str) -> None:
        """Add concept to range"""
        if isinstance(self.range, list) and concept_id not in self.range:
            self.range.append(concept_id)
            self.updated_at = datetime.now()
    
    def set_range_datatype(self, datatype: str) -> None:
        """Set range to a datatype"""
        self.range = datatype
        self.updated_at = datetime.now()
    
    def add_sub_property(self, property_id: str) -> None:
        """Add sub-property"""
        if property_id not in self.sub_properties:
            self.sub_properties.append(property_id)
            self.updated_at = datetime.now()
    
    def add_super_property(self, property_id: str) -> None:
        """Add super-property"""
        if property_id not in self.super_properties:
            self.super_properties.append(property_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert property to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'property_type': self.property_type.value,
            'definition': self.definition,
            'description': self.description,
            'domain': self.domain,
            'range': self.range,
            'functional': self.functional,
            'inverse_functional': self.inverse_functional,
            'transitive': self.transitive,
            'symmetric': self.symmetric,
            'reflexive': self.reflexive,
            'irreflexive': self.irreflexive,
            'inverse_property': self.inverse_property,
            'sub_properties': self.sub_properties,
            'super_properties': self.super_properties,
            'namespace': self.namespace,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class Instance:
    """Represents an instance of a concept"""
    
    id: str
    name: str
    concept_ids: List[str]  # Can be instance of multiple concepts
    
    # Properties
    property_values: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    namespace: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.namespace}:{self.name.replace(' ', '_')}_instance"
    
    def add_concept_type(self, concept_id: str) -> None:
        """Add concept type"""
        if concept_id not in self.concept_ids:
            self.concept_ids.append(concept_id)
            self.updated_at = datetime.now()
    
    def set_property_value(self, property_id: str, value: Any) -> None:
        """Set property value"""
        self.property_values[property_id] = {
            'value': value,
            'updated_at': datetime.now()
        }
        self.updated_at = datetime.now()
    
    def add_relationship(self, property_id: str, target_instance_id: str) -> None:
        """Add relationship to another instance"""
        if property_id not in self.relationships:
            self.relationships[property_id] = []
        
        if target_instance_id not in self.relationships[property_id]:
            self.relationships[property_id].append(target_instance_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'concept_ids': self.concept_ids,
            'property_values': self.property_values,
            'relationships': self.relationships,
            'namespace': self.namespace,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ValidationRule:
    """Represents a validation rule for ontology consistency"""
    
    def __init__(self, rule_id: str, name: str, description: str):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = "error"  # error, warning, info
    
    async def validate(self, ontology: 'Ontology') -> List[Dict[str, Any]]:
        """Validate ontology against this rule"""
        raise NotImplementedError("Subclasses must implement validate method")

class CircularityRule(ValidationRule):
    """Rule to detect circular inheritance"""
    
    def __init__(self):
        super().__init__(
            "circularity_check",
            "Circular Inheritance Detection",
            "Detects circular inheritance relationships in concept hierarchy"
        )
    
    async def validate(self, ontology: 'Ontology') -> List[Dict[str, Any]]:
        """Check for circular inheritance"""
        violations = []
        
        for concept_id, concept in ontology.concepts.items():
            if await self._has_circular_inheritance(concept, ontology, set()):
                violations.append({
                    'rule_id': self.rule_id,
                    'severity': self.severity,
                    'concept_id': concept_id,
                    'message': f"Circular inheritance detected for concept {concept.name}",
                    'details': {'concept_name': concept.name}
                })
        
        return violations
    
    async def _has_circular_inheritance(self, concept: Concept, ontology: 'Ontology', 
                                      visited: Set[str]) -> bool:
        """Recursively check for circular inheritance"""
        if concept.id in visited:
            return True
        
        visited.add(concept.id)
        
        for parent_id in concept.parent_concepts:
            parent_concept = ontology.get_concept(parent_id)
            if parent_concept and await self._has_circular_inheritance(parent_concept, ontology, visited.copy()):
                return True
        
        return False

class DomainRangeRule(ValidationRule):
    """Rule to validate property domain and range constraints"""
    
    def __init__(self):
        super().__init__(
            "domain_range_check",
            "Domain and Range Validation",
            "Validates that property usage respects domain and range constraints"
        )
    
    async def validate(self, ontology: 'Ontology') -> List[Dict[str, Any]]:
        """Check domain and range constraints"""
        violations = []
        
        # Check concept properties
        for concept_id, concept in ontology.concepts.items():
            for prop_name, prop_data in concept.properties.items():
                property_def = ontology.get_property(prop_name)
                
                if property_def:
                    # Check domain constraint
                    if property_def.domain and concept_id not in property_def.domain:
                        # Check if concept is subclass of domain concepts
                        is_valid_domain = False
                        for domain_concept_id in property_def.domain:
                            if concept.is_descendant_of(domain_concept_id, ontology):
                                is_valid_domain = True
                                break
                        
                        if not is_valid_domain:
                            violations.append({
                                'rule_id': self.rule_id,
                                'severity': 'warning',
                                'concept_id': concept_id,
                                'property_id': prop_name,
                                'message': f"Property {prop_name} used outside its domain",
                                'details': {
                                    'concept_name': concept.name,
                                    'property_domain': property_def.domain
                                }
                            })
        
        return violations

class CardinalityRule(ValidationRule):
    """Rule to validate cardinality constraints"""
    
    def __init__(self):
        super().__init__(
            "cardinality_check",
            "Cardinality Constraint Validation",
            "Validates cardinality constraints on properties"
        )
    
    async def validate(self, ontology: 'Ontology') -> List[Dict[str, Any]]:
        """Check cardinality constraints"""
        violations = []
        
        for concept_id, concept in ontology.concepts.items():
            for constraint in concept.constraints:
                if constraint['type'] == ConstraintType.CARDINALITY.value:
                    constraint_data = constraint['data']
                    property_id = constraint_data.get('property')
                    min_cardinality = constraint_data.get('min', 0)
                    max_cardinality = constraint_data.get('max', float('inf'))
                    
                    # Count actual values
                    actual_count = 0
                    
                    # Check in properties
                    if property_id in concept.properties:
                        actual_count = 1
                    
                    # Check in relationships
                    if property_id in concept.related_concepts:
                        actual_count = len(concept.related_concepts[property_id])
                    
                    # Validate cardinality
                    if actual_count < min_cardinality:
                        violations.append({
                            'rule_id': self.rule_id,
                            'severity': self.severity,
                            'concept_id': concept_id,
                            'property_id': property_id,
                            'message': f"Minimum cardinality violation: expected >= {min_cardinality}, found {actual_count}",
                            'details': {
                                'min_cardinality': min_cardinality,
                                'actual_count': actual_count
                            }
                        })
                    
                    if actual_count > max_cardinality:
                        violations.append({
                            'rule_id': self.rule_id,
                            'severity': self.severity,
                            'concept_id': concept_id,
                            'property_id': property_id,
                            'message': f"Maximum cardinality violation: expected <= {max_cardinality}, found {actual_count}",
                            'details': {
                                'max_cardinality': max_cardinality,
                                'actual_count': actual_count
                            }
                        })
        
        return violations

class ReasoningEngine:
    """Performs reasoning and inference on ontology"""
    
    def __init__(self, ontology: 'Ontology'):
        self.ontology = ontology
        self.inferred_facts = []
        self.logger = logging.getLogger("ReasoningEngine")
    
    async def perform_reasoning(self) -> Dict[str, Any]:
        """Perform comprehensive reasoning"""
        start_time = time.time()
        
        results = {
            'inheritance_inferences': await self._infer_inheritance(),
            'property_inferences': await self._infer_properties(),
            'relationship_inferences': await self._infer_relationships(),
            'equivalence_inferences': await self._infer_equivalences(),
            'disjointness_inferences': await self._infer_disjointness()
        }
        
        reasoning_time = time.time() - start_time
        results['reasoning_time'] = reasoning_time
        results['total_inferences'] = sum(len(inf) for inf in results.values() if isinstance(inf, list))
        
        self.logger.info(f"Reasoning completed: {results['total_inferences']} inferences, {reasoning_time:.3f}s")
        
        return results
    
    async def _infer_inheritance(self) -> List[Dict[str, Any]]:
        """Infer inheritance relationships"""
        inferences = []
        
        for concept_id, concept in self.ontology.concepts.items():
            # Infer transitive inheritance
            ancestors = concept.get_ancestors(self.ontology)
            
            for ancestor_id in ancestors:
                ancestor = self.ontology.get_concept(ancestor_id)
                if ancestor:
                    # Inherit properties from ancestors
                    for prop_name, prop_data in ancestor.properties.items():
                        if prop_name not in concept.properties:
                            inferences.append({
                                'type': 'property_inheritance',
                                'concept_id': concept_id,
                                'ancestor_id': ancestor_id,
                                'property': prop_name,
                                'inherited_value': prop_data['value'],
                                'confidence': 0.8
                            })
                    
                    # Inherit constraints
                    for constraint in ancestor.constraints:
                        if constraint not in concept.constraints:
                            inferences.append({
                                'type': 'constraint_inheritance',
                                'concept_id': concept_id,
                                'ancestor_id': ancestor_id,
                                'constraint': constraint,
                                'confidence': 0.9
                            })
        
        return inferences
    
    async def _infer_properties(self) -> List[Dict[str, Any]]:
        """Infer property characteristics and relationships"""
        inferences = []
        
        for prop_id, prop in self.ontology.properties.items():
            # Infer inverse properties
            if prop.inverse_property:
                inverse_prop = self.ontology.get_property(prop.inverse_property)
                if inverse_prop and inverse_prop.inverse_property != prop_id:
                    inferences.append({
                        'type': 'inverse_property',
                        'property_id': prop_id,
                        'inverse_property_id': prop.inverse_property,
                        'confidence': 1.0
                    })
            
            # Infer transitive closure for transitive properties
            if prop.transitive:
                inferences.extend(await self._infer_transitive_closure(prop))
        
        return inferences
    
    async def _infer_transitive_closure(self, prop: Property) -> List[Dict[str, Any]]:
        """Infer transitive closure for a transitive property"""
        inferences = []
        
        # Find all instances using this property
        property_instances = []
        
        for concept_id, concept in self.ontology.concepts.items():
            if prop.id in concept.related_concepts:
                for target_id in concept.related_concepts[prop.id]:
                    property_instances.append((concept_id, target_id))
        
        # Compute transitive closure
        changed = True
        while changed:
            changed = False
            new_instances = []
            
            for source, intermediate in property_instances:
                for intermediate2, target in property_instances:
                    if intermediate == intermediate2 and (source, target) not in property_instances:
                        new_instances.append((source, target))
                        inferences.append({
                            'type': 'transitive_inference',
                            'property_id': prop.id,
                            'source_concept': source,
                            'target_concept': target,
                            'via_concept': intermediate,
                            'confidence': 0.7
                        })
                        changed = True
            
            property_instances.extend(new_instances)
        
        return inferences
    
    async def _infer_relationships(self) -> List[Dict[str, Any]]:
        """Infer new relationships based on existing ones"""
        inferences = []
        
        # Example: If A part_of B and B part_of C, then A part_of C (transitivity)
        part_of_relations = []
        
        for concept_id, concept in self.ontology.concepts.items():
            if 'part_of' in concept.related_concepts:
                for target_id in concept.related_concepts['part_of']:
                    part_of_relations.append((concept_id, target_id))
        
        # Infer transitive part_of relationships
        for source, intermediate in part_of_relations:
            for intermediate2, target in part_of_relations:
                if intermediate == intermediate2 and (source, target) not in part_of_relations:
                    inferences.append({
                        'type': 'transitive_part_of',
                        'source_concept': source,
                        'target_concept': target,
                        'via_concept': intermediate,
                        'confidence': 0.8
                    })
        
        return inferences
    
    async def _infer_equivalences(self) -> List[Dict[str, Any]]:
        """Infer concept equivalences"""
        inferences = []
        
        # Find concepts with same definition or high synonym overlap
        concepts_list = list(self.ontology.concepts.values())
        
        for i, concept1 in enumerate(concepts_list):
            for concept2 in concepts_list[i+1:]:
                similarity_score = await self._calculate_concept_similarity(concept1, concept2)
                
                if similarity_score > 0.9:  # High similarity threshold
                    inferences.append({
                        'type': 'concept_equivalence',
                        'concept1_id': concept1.id,
                        'concept2_id': concept2.id,
                        'similarity_score': similarity_score,
                        'confidence': min(0.9, similarity_score)
                    })
        
        return inferences
    
    async def _infer_disjointness(self) -> List[Dict[str, Any]]:
        """Infer disjoint relationships"""
        inferences = []
        
        # Concepts with contradictory properties are likely disjoint
        for concept1_id, concept1 in self.ontology.concepts.items():
            for concept2_id, concept2 in self.ontology.concepts.items():
                if concept1_id != concept2_id:
                    disjointness_evidence = await self._check_disjointness_evidence(concept1, concept2)
                    
                    if disjointness_evidence['score'] > 0.7:
                        inferences.append({
                            'type': 'concept_disjointness',
                            'concept1_id': concept1_id,
                            'concept2_id': concept2_id,
                            'evidence': disjointness_evidence['evidence'],
                            'confidence': disjointness_evidence['score']
                        })
        
        return inferences
    
    async def _calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Calculate similarity between two concepts"""
        similarity_score = 0.0
        
        # Definition similarity (simplified)
        if concept1.definition and concept2.definition:
            common_words = set(concept1.definition.lower().split()) & set(concept2.definition.lower().split())
            total_words = set(concept1.definition.lower().split()) | set(concept2.definition.lower().split())
            if total_words:
                definition_similarity = len(common_words) / len(total_words)
                similarity_score += definition_similarity * 0.4
        
        # Synonym overlap
        concept1_terms = set([concept1.name.lower()] + [s.lower() for s in concept1.synonyms])
        concept2_terms = set([concept2.name.lower()] + [s.lower() for s in concept2.synonyms])
        
        if concept1_terms & concept2_terms:
            synonym_similarity = len(concept1_terms & concept2_terms) / len(concept1_terms | concept2_terms)
            similarity_score += synonym_similarity * 0.3
        
        # Property similarity
        common_properties = set(concept1.properties.keys()) & set(concept2.properties.keys())
        total_properties = set(concept1.properties.keys()) | set(concept2.properties.keys())
        
        if total_properties:
            property_similarity = len(common_properties) / len(total_properties)
            similarity_score += property_similarity * 0.3
        
        return min(1.0, similarity_score)
    
    async def _check_disjointness_evidence(self, concept1: Concept, concept2: Concept) -> Dict[str, Any]:
        """Check for evidence of disjointness between concepts"""
        evidence = []
        score = 0.0
        
        # Check for contradictory properties
        for prop_name in concept1.properties:
            if prop_name in concept2.properties:
                value1 = concept1.properties[prop_name]['value']
                value2 = concept2.properties[prop_name]['value']
                
                # Simple contradiction check
                if isinstance(value1, bool) and isinstance(value2, bool) and value1 != value2:
                    evidence.append(f"Contradictory boolean property: {prop_name}")
                    score += 0.3
                elif isinstance(value1, str) and isinstance(value2, str):
                    # Check for contradictory string values
                    contradictory_pairs = [
                        ('living', 'non-living'),
                        ('animate', 'inanimate'),
                        ('organic', 'inorganic'),
                        ('solid', 'liquid'),
                        ('positive', 'negative')
                    ]
                    
                    for pair in contradictory_pairs:
                        if (value1.lower() in pair[0] and value2.lower() in pair[1]) or \
                           (value1.lower() in pair[1] and value2.lower() in pair[0]):
                            evidence.append(f"Contradictory property values: {prop_name} ({value1} vs {value2})")
                            score += 0.4
        
        # Check explicit disjoint constraints
        for constraint in concept1.constraints:
            if constraint['type'] == ConstraintType.DISJOINT.value:
                if concept2.id in constraint['data'].get('concepts', []):
                    evidence.append("Explicit disjoint constraint")
                    score += 1.0
        
        return {
            'evidence': evidence,
            'score': min(1.0, score)
        }

class Ontology:
    """Main ontology management system"""
    
    def __init__(self, name: str, namespace: str = "", version: str = "1.0"):
        self.name = name
        self.namespace = namespace or name.lower().replace(' ', '_')
        self.version = version
        
        # Core components
        self.concepts: Dict[str, Concept] = {}
        self.properties: Dict[str, Property] = {}
        self.instances: Dict[str, Instance] = {}
        
        # Validation and reasoning
        self.validation_rules: List[ValidationRule] = []
        self.reasoning_engine = ReasoningEngine(self)
        
        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.created_by = ""
        self.description = ""
        
        # Statistics
        self.stats = {
            'concepts_added': 0,
            'properties_added': 0,
            'instances_added': 0,
            'validations_performed': 0,
            'reasoning_sessions': 0
        }
        
        # Initialize default validation rules
        self._initialize_validation_rules()
        
        self.logger = logging.getLogger("Ontology")
    
    def _initialize_validation_rules(self) -> None:
        """Initialize default validation rules"""
        self.validation_rules = [
            CircularityRule(),
            DomainRangeRule(),
            CardinalityRule()
        ]
    
    async def add_concept(self, concept: Concept) -> str:
        """Add concept to ontology"""
        try:
            # Set namespace if not provided
            if not concept.namespace:
                concept.namespace = self.namespace
                concept.id = f"{self.namespace}:{concept.name.replace(' ', '_')}"
            
            # Check for existing concept
            if concept.id in self.concepts:
                self.logger.warning(f"Concept {concept.id} already exists")
                return concept.id
            
            # Add concept
            self.concepts[concept.id] = concept
            
            # Update parent-child relationships
            for parent_id in concept.parent_concepts:
                parent_concept = self.get_concept(parent_id)
                if parent_concept:
                    parent_concept.add_child(concept.id)
            
            for child_id in concept.child_concepts:
                child_concept = self.get_concept(child_id)
                if child_concept:
                    child_concept.add_parent(concept.id)
            
            self.stats['concepts_added'] += 1
            self.updated_at = datetime.now()
            
            self.logger.debug(f"Added concept: {concept.name}")
            return concept.id
            
        except Exception as e:
            self.logger.error(f"Failed to add concept: {e}")
            return ""
    
    async def add_property(self, property: Property) -> str:
        """Add property to ontology"""
        try:
            # Set namespace if not provided
            if not property.namespace:
                property.namespace = self.namespace
                property.id = f"{self.namespace}:{property.name.replace(' ', '_')}"
            
            # Check for existing property
            if property.id in self.properties:
                self.logger.warning(f"Property {property.id} already exists")
                return property.id
            
            # Add property
            self.properties[property.id] = property
            
            # Update property hierarchy
            for sub_prop_id in property.sub_properties:
                sub_property = self.get_property(sub_prop_id)
                if sub_property:
                    sub_property.add_super_property(property.id)
            
            for super_prop_id in property.super_properties:
                super_property = self.get_property(super_prop_id)
                if super_property:
                    super_property.add_sub_property(property.id)
            
            self.stats['properties_added'] += 1
            self.updated_at = datetime.now()
            
            self.logger.debug(f"Added property: {property.name}")
            return property.id
            
        except Exception as e:
            self.logger.error(f"Failed to add property: {e}")
            return ""
    
    async def add_instance(self, instance: Instance) -> str:
        """Add instance to ontology"""
        try:
            # Set namespace if not provided
            if not instance.namespace:
                instance.namespace = self.namespace
                instance.id = f"{self.namespace}:{instance.name.replace(' ', '_')}_instance"
            
            # Validate concept types exist
            for concept_id in instance.concept_ids:
                if concept_id not in self.concepts:
                    raise ValueError(f"Concept {concept_id} not found in ontology")
            
            # Add instance
            self.instances[instance.id] = instance
            
            self.stats['instances_added'] += 1
            self.updated_at = datetime.now()
            
            self.logger.debug(f"Added instance: {instance.name}")
            return instance.id
            
        except Exception as e:
            self.logger.error(f"Failed to add instance: {e}")
            return ""
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID"""
        return self.concepts.get(concept_id)
    
    def get_property(self, property_id: str) -> Optional[Property]:
        """Get property by ID"""
        return self.properties.get(property_id)
    
    def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Get instance by ID"""
        return self.instances.get(instance_id)
    
    async def validate_ontology(self) -> Dict[str, Any]:
        """Validate ontology consistency"""
        start_time = time.time()
        
        all_violations = []
        
        for rule in self.validation_rules:
            try:
                violations = await rule.validate(self)
                all_violations.extend(violations)
            except Exception as e:
                self.logger.error(f"Validation rule {rule.rule_id} failed: {e}")
        
        # Group violations by severity
        violations_by_severity = defaultdict(list)
        for violation in all_violations:
            violations_by_severity[violation['severity']].append(violation)
        
        validation_time = time.time() - start_time
        self.stats['validations_performed'] += 1
        
        result = {
            'valid': len(violations_by_severity.get('error', [])) == 0,
            'total_violations': len(all_violations),
            'violations_by_severity': dict(violations_by_severity),
            'validation_time': validation_time,
            'rules_applied': len(self.validation_rules)
        }
        
        self.logger.info(f"Ontology validation: {result['total_violations']} violations, "
                        f"valid: {result['valid']}, {validation_time:.3f}s")
        
        return result
    
    async def perform_reasoning(self) -> Dict[str, Any]:
        """Perform reasoning and inference"""
        self.stats['reasoning_sessions'] += 1
        return await self.reasoning_engine.perform_reasoning()
    
    async def query_concepts(self, query: str, concept_types: List[ConceptType] = None) -> List[Concept]:
        """Query concepts by name or definition"""
        matching_concepts = []
        query_lower = query.lower()
        
        for concept in self.concepts.values():
            # Filter by concept type
            if concept_types and concept.concept_type not in concept_types:
                continue
            
            # Calculate relevance score
            score = 0.0
            
            # Exact name match
            if concept.name.lower() == query_lower:
                score += 1.0
            # Partial name match
            elif query_lower in concept.name.lower():
                score += 0.8
            # Synonym match
            elif any(query_lower in synonym.lower() for synonym in concept.synonyms):
                score += 0.7
            # Definition match
            elif query_lower in concept.definition.lower():
                score += 0.5
            # Description match
            elif query_lower in concept.description.lower():
                score += 0.3
            
            if score > 0:
                matching_concepts.append((concept, score))
        
        # Sort by relevance
        matching_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return [concept for concept, score in matching_concepts]
    
    async def get_concept_hierarchy(self, root_concept_id: str = None) -> Dict[str, Any]:
        """Get concept hierarchy starting from root"""
        
        if root_concept_id and root_concept_id not in self.concepts:
            return {}
        
        if not root_concept_id:
            # Find root concepts (concepts with no parents)
            root_concepts = [c for c in self.concepts.values() if not c.parent_concepts]
        else:
            root_concepts = [self.concepts[root_concept_id]]
        
        hierarchy = {}
        
        for root_concept in root_concepts:
            hierarchy[root_concept.id] = await self._build_concept_subtree(root_concept)
        
        return hierarchy
    
    async def _build_concept_subtree(self, concept: Concept) -> Dict[str, Any]:
        """Build subtree for a concept"""
        subtree = {
            'concept': concept.to_dict(),
            'children': {}
        }
        
        for child_id in concept.child_concepts:
            child_concept = self.get_concept(child_id)
            if child_concept:
                subtree['children'][child_id] = await self._build_concept_subtree(child_concept)
        
        return subtree
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ontology statistics"""
        
        # Basic counts
        num_concepts = len(self.concepts)
        num_properties = len(self.properties)
        num_instances = len(self.instances)
        
        # Concept type distribution
        concept_types = defaultdict(int)
        for concept in self.concepts.values():
            concept_types[concept.concept_type.value] += 1
        
        # Property type distribution
        property_types = defaultdict(int)
        for prop in self.properties.values():
            property_types[prop.property_type.value] += 1
        
        # Hierarchy depth
        max_depth = 0
        for concept in self.concepts.values():
            if not concept.parent_concepts:  # Root concept
                depth = self._calculate_hierarchy_depth(concept, 0)
                max_depth = max(max_depth, depth)
        
        return {
            'basic_statistics': {
                'total_concepts': num_concepts,
                'total_properties': num_properties,
                'total_instances': num_instances,
                'last_updated': self.updated_at.isoformat()
            },
            'concept_distribution': dict(concept_types),
            'property_distribution': dict(property_types),
            'hierarchy_metrics': {
                'max_depth': max_depth,
                'root_concepts': len([c for c in self.concepts.values() if not c.parent_concepts]),
                'leaf_concepts': len([c for c in self.concepts.values() if not c.child_concepts])
            },
            'processing_statistics': self.stats
        }
    
    def _calculate_hierarchy_depth(self, concept: Concept, current_depth: int) -> int:
        """Calculate maximum depth of concept hierarchy"""
        if not concept.child_concepts:
            return current_depth
        
        max_child_depth = current_depth
        for child_id in concept.child_concepts:
            child_concept = self.get_concept(child_id)
            if child_concept:
                child_depth = self._calculate_hierarchy_depth(child_concept, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_concept_creation():
    """Demo: Creating concepts and building hierarchy"""
    print("\nDEMO 1: CONCEPT CREATION AND HIERARCHY")
    print("=" * 50)
    
    ontology = Ontology("Medical Ontology", "medical", "1.0")
    
    # Create root concepts
    disease_concept = Concept(
        id="",
        name="Disease",
        concept_type=ConceptType.ABSTRACT,
        definition="A disorder or abnormal condition that affects the functioning of an organism",
        description="General category for all diseases and medical conditions",
        namespace="medical"
    )
    
    treatment_concept = Concept(
        id="",
        name="Treatment",
        concept_type=ConceptType.PROCESS,
        definition="Medical intervention designed to cure or alleviate a disease or condition",
        description="General category for all medical treatments and therapies",
        namespace="medical"
    )
    
    print("Creating root concepts:")
    disease_id = await ontology.add_concept(disease_concept)
    treatment_id = await ontology.add_concept(treatment_concept)
    print(f"  ✓ Disease concept: {disease_id}")
    print(f"  ✓ Treatment concept: {treatment_id}")
    
    # Create specific disease concepts
    cardiovascular_disease = Concept(
        id="",
        name="Cardiovascular Disease",
        concept_type=ConceptType.CONCRETE,
        definition="Disease affecting the heart and blood vessels",
        parent_concepts=[disease_id],
        synonyms=["Heart Disease", "Cardiac Disease"],
        namespace="medical"
    )
    
    heart_attack = Concept(
        id="",
        name="Myocardial Infarction",
        concept_type=ConceptType.CONCRETE,
        definition="Death of heart muscle due to insufficient blood supply",
        parent_concepts=[],  # Will be set after adding cardiovascular disease
        synonyms=["Heart Attack", "MI"],
        namespace="medical"
    )
    
    # Add properties to concepts
    heart_attack.add_property("severity", "high", PropertyType.DATATYPE)
    heart_attack.add_property("emergency_level", "critical", PropertyType.DATATYPE)
    heart_attack.add_property("typical_age_range", "50-80", PropertyType.DATATYPE)
    
    print(f"\nCreating specific disease concepts:")
    cardio_id = await ontology.add_concept(cardiovascular_disease)
    print(f"  ✓ Cardiovascular Disease: {cardio_id}")
    
    # Set heart attack as child of cardiovascular disease
    heart_attack.parent_concepts = [cardio_id]
    mi_id = await ontology.add_concept(heart_attack)
    print(f"  ✓ Myocardial Infarction: {mi_id}")
    
    # Create treatment concepts
    medication_treatment = Concept(
        id="",
        name="Medication Treatment",
        concept_type=ConceptType.PROCESS,
        definition="Treatment using pharmaceutical drugs",
        parent_concepts=[treatment_id],
        namespace="medical"
    )
    
    surgical_treatment = Concept(
        id="",
        name="Surgical Treatment", 
        concept_type=ConceptType.PROCESS,
        definition="Treatment involving surgical procedures",
        parent_concepts=[treatment_id],
        namespace="medical"
    )
    
    med_treatment_id = await ontology.add_concept(medication_treatment)
    surg_treatment_id = await ontology.add_concept(surgical_treatment)
    print(f"  ✓ Medication Treatment: {med_treatment_id}")
    print(f"  ✓ Surgical Treatment: {surg_treatment_id}")
    
    # Show hierarchy
    print(f"\nConcept Hierarchy:")
    hierarchy = await ontology.get_concept_hierarchy()
    
    def print_hierarchy(node, level=0):
        indent = "  " * level
        concept_data = node['concept']
        print(f"{indent}- {concept_data['name']} ({concept_data['concept_type']})")
        
        for child_id, child_node in node['children'].items():
            print_hierarchy(child_node, level + 1)
    
    for root_id, root_node in hierarchy.items():
        print_hierarchy(root_node)
    
    # Show statistics
    stats = ontology.get_statistics()
    print(f"\nOntology Statistics:")
    print(f"  Total concepts: {stats['basic_statistics']['total_concepts']}")
    print(f"  Hierarchy depth: {stats['hierarchy_metrics']['max_depth']}")
    print(f"  Root concepts: {stats['hierarchy_metrics']['root_concepts']}")

async def demo_property_management():
    """Demo: Creating and managing properties"""
    print("\nDEMO 2: PROPERTY MANAGEMENT")
    print("=" * 50)
    
    ontology = Ontology("Research Ontology", "research", "1.0")
    
    # Create concepts first
    person_concept = Concept(
        id="",
        name="Person",
        concept_type=ConceptType.CONCRETE,
        definition="An individual human being",
        namespace="research"
    )
    
    organization_concept = Concept(
        id="",
        name="Organization",
        concept_type=ConceptType.CONCRETE,
        definition="A structured group of people with shared objectives",
        namespace="research"
    )
    
    publication_concept = Concept(
        id="",
        name="Publication",
        concept_type=ConceptType.CONCRETE,
        definition="A published research work",
        namespace="research"
    )
    
    person_id = await ontology.add_concept(person_concept)
    org_id = await ontology.add_concept(organization_concept)
    pub_id = await ontology.add_concept(publication_concept)
    
    print("Created base concepts:")
    print(f"  ✓ Person: {person_id}")
    print(f"  ✓ Organization: {org_id}")
    print(f"  ✓ Publication: {pub_id}")
    
    # Create properties
    print(f"\nCreating properties:")
    
    # Object properties (relationships between concepts)
    works_for_property = Property(
        id="",
        name="works for",
        property_type=PropertyType.OBJECT,
        definition="Relationship between a person and their employer",
        domain=[person_id],
        range=[org_id],
        functional=True,  # Person can work for only one organization
        namespace="research"
    )
    
    authored_by_property = Property(
        id="",
        name="authored by",
        property_type=PropertyType.OBJECT,
        definition="Relationship between a publication and its author",
        domain=[pub_id],
        range=[person_id],
        inverse_property="",  # Will be set after creating "authors" property
        namespace="research"
    )
    
    authors_property = Property(
        id="",
        name="authors",
        property_type=PropertyType.OBJECT,
        definition="Relationship between a person and their publications",
        domain=[person_id],
        range=[pub_id],
        namespace="research"
    )
    
    # Datatype properties (attributes)
    age_property = Property(
        id="",
        name="age",
        property_type=PropertyType.DATATYPE,
        definition="Age of a person in years",
        domain=[person_id],
        functional=True,
        namespace="research"
    )
    age_property.set_range_datatype("integer")
    
    name_property = Property(
        id="",
        name="name",
        property_type=PropertyType.DATATYPE,
        definition="Name of an entity",
        domain=[person_id, org_id, pub_id],
        functional=True,
        namespace="research"
    )
    name_property.set_range_datatype("string")
    
    # Add properties to ontology
    works_for_id = await ontology.add_property(works_for_property)
    authored_by_id = await ontology.add_property(authored_by_property)
    authors_id = await ontology.add_property(authors_property)
    age_id = await ontology.add_property(age_property)
    name_id = await ontology.add_property(name_property)
    
    # Set inverse relationships
    authored_by_property.inverse_property = authors_id
    authors_property.inverse_property = authored_by_id
    
    print(f"  ✓ works for: {works_for_id}")
    print(f"  ✓ authored by: {authored_by_id}")
    print(f"  ✓ authors: {authors_id}")
    print(f"  ✓ age: {age_id}")
    print(f"  ✓ name: {name_id}")
    
    # Show property details
    print(f"\nProperty Details:")
    
    for prop_id, prop in ontology.properties.items():
        print(f"\n{prop.name} ({prop.property_type.value}):")
        print(f"  Definition: {prop.definition}")
        print(f"  Domain: {[ontology.get_concept(c_id).name for c_id in prop.domain if ontology.get_concept(c_id)]}")
        
        if isinstance(prop.range, list):
            range_concepts = [ontology.get_concept(c_id).name for c_id in prop.range if ontology.get_concept(c_id)]
            print(f"  Range: {range_concepts}")
        else:
            print(f"  Range: {prop.range}")
        
        print(f"  Functional: {prop.functional}")
        if prop.inverse_property:
            inverse_prop = ontology.get_property(prop.inverse_property)
            if inverse_prop:
                print(f"  Inverse: {inverse_prop.name}")
    
    # Show ontology statistics
    stats = ontology.get_statistics()
    print(f"\nOntology Statistics:")
    print(f"  Total properties: {stats['basic_statistics']['total_properties']}")
    print(f"  Property types: {stats['property_distribution']}")

async def demo_validation_and_reasoning():
    """Demo: Ontology validation and reasoning"""
    print("\nDEMO 3: VALIDATION AND REASONING")
    print("=" * 50)
    
    ontology = Ontology("AI Ontology", "ai", "1.0")
    
    # Create concepts with potential issues for validation
    print("Creating concepts with validation scenarios:")
    
    # Valid concepts
    ai_concept = Concept(
        id="",
        name="Artificial Intelligence",
        concept_type=ConceptType.ABSTRACT,
        definition="Intelligence demonstrated by machines",
        namespace="ai"
    )
    
    ml_concept = Concept(
        id="",
        name="Machine Learning",
        concept_type=ConceptType.ABSTRACT,
        definition="Subset of AI that learns from data",
        parent_concepts=[],  # Will be set after adding AI concept
        namespace="ai"
    )
    
    dl_concept = Concept(
        id="",
        name="Deep Learning",
        concept_type=ConceptType.ABSTRACT,
        definition="Subset of ML using neural networks",
        parent_concepts=[],  # Will be set after adding ML concept
        namespace="ai"
    )
    
    # Add concepts
    ai_id = await ontology.add_concept(ai_concept)
    
    ml_concept.parent_concepts = [ai_id]
    ml_id = await ontology.add_concept(ml_concept)
    
    dl_concept.parent_concepts = [ml_id]
    dl_id = await ontology.add_concept(dl_concept)
    
    print(f"  ✓ AI: {ai_id}")
    print(f"  ✓ ML: {ml_id}")
    print(f"  ✓ DL: {dl_id}")
    
    # Create concept with circular inheritance (for validation demo)
    circular_concept = Concept(
        id="",
        name="Circular Concept",
        concept_type=ConceptType.ABSTRACT,
        definition="Concept that creates circular inheritance",
        parent_concepts=[dl_id],  # This will create a circular reference
        namespace="ai"
    )
    
    circular_id = await ontology.add_concept(circular_concept)
    
    # Create the circular reference
    ai_concept.parent_concepts = [circular_id]  # AI -> Circular -> DL -> ML -> AI
    
    print(f"  ✓ Circular Concept: {circular_id} (creates circular inheritance)")
    
    # Add properties with constraints
    accuracy_property = Property(
        id="",
        name="accuracy",
        property_type=PropertyType.DATATYPE,
        definition="Accuracy score of an AI model",
        domain=[ml_id, dl_id],
        functional=True,
        namespace="ai"
    )
    accuracy_property.set_range_datatype("float")
    
    accuracy_id = await ontology.add_property(accuracy_property)
    
    # Add cardinality constraint
    ml_concept.add_constraint(
        ConstraintType.CARDINALITY,
        {
            'property': accuracy_id,
            'min': 1,
            'max': 1
        }
    )
    
    print(f"  ✓ Accuracy property with cardinality constraint: {accuracy_id}")
    
    # Perform validation
    print(f"\nPerforming ontology validation:")
    validation_result = await ontology.validate_ontology()
    
    print(f"  Validation result: {'VALID' if validation_result['valid'] else 'INVALID'}")
    print(f"  Total violations: {validation_result['total_violations']}")
    print(f"  Validation time: {validation_result['validation_time']:.3f}s")
    
    if validation_result['violations_by_severity']:
        print(f"\nViolations by severity:")
        for severity, violations in validation_result['violations_by_severity'].items():
            print(f"  {severity.upper()}: {len(violations)}")
            
            for violation in violations[:3]:  # Show first 3 violations
                print(f"    - {violation['message']}")
                if 'concept_id' in violation:
                    concept = ontology.get_concept(violation['concept_id'])
                    if concept:
                        print(f"      Concept: {concept.name}")
    
    # Fix circular inheritance for reasoning demo
    ai_concept.parent_concepts = []  # Remove circular reference
    
    print(f"\nFixed circular inheritance for reasoning demo")
    
    # Add more concepts for reasoning
    neural_network_concept = Concept(
        id="",
        name="Neural Network",
        concept_type=ConceptType.CONCRETE,
        definition="Network of artificial neurons",
        parent_concepts=[dl_id],
        namespace="ai"
    )
    
    cnn_concept = Concept(
        id="",
        name="Convolutional Neural Network",
        concept_type=ConceptType.CONCRETE,
        definition="Neural network for image processing",
        parent_concepts=[],  # Will be set after adding neural network
        synonyms=["CNN", "ConvNet"],
        namespace="ai"
    )
    
    nn_id = await ontology.add_concept(neural_network_concept)
    cnn_concept.parent_concepts = [nn_id]
    cnn_id = await ontology.add_concept(cnn_concept)
    
    # Add some properties for reasoning
    neural_network_concept.add_property("layer_count", 5, PropertyType.DATATYPE)
    neural_network_concept.add_property("activation_function", "ReLU", PropertyType.DATATYPE)
    
    # Perform reasoning
    print(f"\nPerforming reasoning and inference:")
    reasoning_result = await ontology.perform_reasoning()
    
    print(f"  Reasoning time: {reasoning_result['reasoning_time']:.3f}s")
    print(f"  Total inferences: {reasoning_result['total_inferences']}")
    
    print(f"\nInference types:")
    for inference_type, inferences in reasoning_result.items():
        if isinstance(inferences, list) and inferences:
            print(f"  {inference_type}: {len(inferences)}")
            
            # Show sample inferences
            for inference in inferences[:2]:  # Show first 2
                print(f"    - Type: {inference.get('type', 'unknown')}")
                if 'concept_id' in inference:
                    concept = ontology.get_concept(inference['concept_id'])
                    if concept:
                        print(f"      Concept: {concept.name}")
                print(f"      Confidence: {inference.get('confidence', 0):.2f}")

async def demo_instance_management():
    """Demo: Managing concept instances"""
    print("\nDEMO 4: INSTANCE MANAGEMENT")
    print("=" * 50)
    
    ontology = Ontology("University Ontology", "university", "1.0")
    
    # Create concepts
    person_concept = Concept(
        id="",
        name="Person",
        concept_type=ConceptType.CONCRETE,
        definition="An individual human being",
        namespace="university"
    )
    
    student_concept = Concept(
        id="",
        name="Student",
        concept_type=ConceptType.ROLE,
        definition="Person enrolled in educational institution",
        parent_concepts=[],  # Will be set after adding Person
        namespace="university"
    )
    
    professor_concept = Concept(
        id="",
        name="Professor",
        concept_type=ConceptType.ROLE,
        definition="Academic instructor and researcher",
        parent_concepts=[],  # Will be set after adding Person
        namespace="university"
    )
    
    course_concept = Concept(
        id="",
        name="Course",
        concept_type=ConceptType.ABSTRACT,
        definition="Educational program or class",
        namespace="university"
    )
    
    # Add concepts
    person_id = await ontology.add_concept(person_concept)
    
    student_concept.parent_concepts = [person_id]
    student_id = await ontology.add_concept(student_concept)
    
    professor_concept.parent_concepts = [person_id]
    professor_id = await ontology.add_concept(professor_concept)
    
    course_id = await ontology.add_concept(course_concept)
    
    print("Created concepts:")
    print(f"  ✓ Person: {person_id}")
    print(f"  ✓ Student: {student_id}")
    print(f"  ✓ Professor: {professor_id}")
    print(f"  ✓ Course: {course_id}")
    
    # Create properties
    enrolled_in_property = Property(
        id="",
        name="enrolled in",
        property_type=PropertyType.OBJECT,
        definition="Student is enrolled in a course",
        domain=[student_id],
        range=[course_id],
        namespace="university"
    )
    
    teaches_property = Property(
        id="",
        name="teaches",
        property_type=PropertyType.OBJECT,
        definition="Professor teaches a course",
        domain=[professor_id],
        range=[course_id],
        namespace="university"
    )
    
    enrolled_in_id = await ontology.add_property(enrolled_in_property)
    teaches_id = await ontology.add_property(teaches_property)
    
    print(f"\nCreated properties:")
    print(f"  ✓ enrolled in: {enrolled_in_id}")
    print(f"  ✓ teaches: {teaches_id}")
    
    # Create instances
    print(f"\nCreating instances:")
    
    # Student instances
    alice_student = Instance(
        id="",
        name="Alice Johnson",
        concept_ids=[student_id],
        namespace="university"
    )
    alice_student.set_property_value("age", 20)
    alice_student.set_property_value("major", "Computer Science")
    alice_student.set_property_value("gpa", 3.8)
    
    bob_student = Instance(
        id="",
        name="Bob Smith",
        concept_ids=[student_id],
        namespace="university"
    )
    bob_student.set_property_value("age", 22)
    bob_student.set_property_value("major", "Mathematics")
    bob_student.set_property_value("gpa", 3.6)
    
    # Professor instance
    dr_wilson = Instance(
        id="",
        name="Dr. Sarah Wilson",
        concept_ids=[professor_id],
        namespace="university"
    )
    dr_wilson.set_property_value("age", 45)
    dr_wilson.set_property_value("department", "Computer Science")
    dr_wilson.set_property_value("tenure", True)
    
    # Course instances
    ai_course = Instance(
        id="",
        name="Introduction to Artificial Intelligence",
        concept_ids=[course_id],
        namespace="university"
    )
    ai_course.set_property_value("course_code", "CS 461")
    ai_course.set_property_value("credits", 3)
    ai_course.set_property_value("semester", "Fall 2024")
    
    math_course = Instance(
        id="",
        name="Linear Algebra",
        concept_ids=[course_id],
        namespace="university"
    )
    math_course.set_property_value("course_code", "MATH 341")
    math_course.set_property_value("credits", 4)
    math_course.set_property_value("semester", "Fall 2024")
    
    # Add instances to ontology
    alice_id = await ontology.add_instance(alice_student)
    bob_id = await ontology.add_instance(bob_student)
    dr_wilson_id = await ontology.add_instance(dr_wilson)
    ai_course_id = await ontology.add_instance(ai_course)
    math_course_id = await ontology.add_instance(math_course)
    
    print(f"  ✓ Alice Johnson (Student): {alice_id}")
    print(f"  ✓ Bob Smith (Student): {bob_id}")
    print(f"  ✓ Dr. Sarah Wilson (Professor): {dr_wilson_id}")
    print(f"  ✓ AI Course: {ai_course_id}")
    print(f"  ✓ Math Course: {math_course_id}")
    
    # Create relationships between instances
    print(f"\nCreating relationships:")
    
    # Students enrolled in courses
    alice_student.add_relationship(enrolled_in_id, ai_course_id)
    alice_student.add_relationship(enrolled_in_id, math_course_id)
    bob_student.add_relationship(enrolled_in_id, math_course_id)
    
    # Professor teaches courses
    dr_wilson.add_relationship(teaches_id, ai_course_id)
    
    print(f"  ✓ Alice enrolled in AI and Math courses")
    print(f"  ✓ Bob enrolled in Math course")
    print(f"  ✓ Dr. Wilson teaches AI course")
    
    # Show instance details
    print(f"\nInstance Details:")
    
    for instance_id, instance in ontology.instances.items():
        print(f"\n{instance.name}:")
        
        # Show concept types
        concept_names = [ontology.get_concept(c_id).name for c_id in instance.concept_ids 
                        if ontology.get_concept(c_id)]
        print(f"  Type(s): {', '.join(concept_names)}")
        
        # Show properties
        if instance.property_values:
            print(f"  Properties:")
            for prop_name, prop_data in instance.property_values.items():
                print(f"    {prop_name}: {prop_data['value']}")
        
        # Show relationships
        if instance.relationships:
            print(f"  Relationships:")
            for rel_prop_id, target_ids in instance.relationships.items():
                prop = ontology.get_property(rel_prop_id)
                prop_name = prop.name if prop else rel_prop_id
                
                for target_id in target_ids:
                    target_instance = ontology.get_instance(target_id)
                    target_name = target_instance.name if target_instance else target_id
                    print(f"    {prop_name}: {target_name}")
    
    # Show final statistics
    stats = ontology.get_statistics()
    print(f"\nFinal Ontology Statistics:")
    print(f"  Concepts: {stats['basic_statistics']['total_concepts']}")
    print(f"  Properties: {stats['basic_statistics']['total_properties']}")
    print(f"  Instances: {stats['basic_statistics']['total_instances']}")

async def demo_ontology_querying():
    """Demo: Querying and searching the ontology"""
    print("\nDEMO 5: ONTOLOGY QUERYING AND SEARCH")
    print("=" * 50)
    
    # Build a comprehensive ontology for demonstration
    ontology = Ontology("Technology Ontology", "tech", "1.0")
    
    # Create technology concepts
    concepts_data = [
        ("Technology", ConceptType.ABSTRACT, "Applied science and engineering", []),
        ("Software", ConceptType.ABSTRACT, "Computer programs and applications", ["Technology"]),
        ("Hardware", ConceptType.CONCRETE, "Physical computing components", ["Technology"]),
        ("Programming Language", ConceptType.ABSTRACT, "Formal language for programming", ["Software"]),
        ("Database", ConceptType.CONCRETE, "Organized collection of data", ["Software"]),
        ("Operating System", ConceptType.CONCRETE, "System software managing resources", ["Software"]),
        ("Web Framework", ConceptType.CONCRETE, "Framework for web development", ["Software"]),
        ("Machine Learning", ConceptType.ABSTRACT, "AI subset that learns from data", ["Software"]),
        ("Python", ConceptType.CONCRETE, "High-level programming language", ["Programming Language"]),
        ("JavaScript", ConceptType.CONCRETE, "Dynamic programming language", ["Programming Language"]),
        ("MySQL", ConceptType.CONCRETE, "Relational database system", ["Database"]),
        ("Linux", ConceptType.CONCRETE, "Open-source operating system", ["Operating System"]),
        ("Django", ConceptType.CONCRETE, "Python web framework", ["Web Framework"]),
        ("TensorFlow", ConceptType.CONCRETE, "Machine learning framework", ["Machine Learning"])
    ]
    
    concept_map = {}
    
    print("Building comprehensive technology ontology...")
    
    for name, concept_type, definition, parent_names in concepts_data:
        concept = Concept(
            id="",
            name=name,
            concept_type=concept_type,
            definition=definition,
            namespace="tech"
        )
        
        # Add synonyms for some concepts
        if name == "Python":
            concept.synonyms = ["Python Programming Language"]
        elif name == "JavaScript":
            concept.synonyms = ["JS", "ECMAScript"]
        elif name == "Machine Learning":
            concept.synonyms = ["ML", "Statistical Learning"]
        elif name == "MySQL":
            concept.synonyms = ["MySQL Database"]
        
        concept_id = await ontology.add_concept(concept)
        concept_map[name] = concept_id
    
    # Set parent relationships
    for name, concept_type, definition, parent_names in concepts_data:
        if parent_names:
            concept = ontology.get_concept(concept_map[name])
            if concept:
                for parent_name in parent_names:
                    if parent_name in concept_map:
                        concept.add_parent(concept_map[parent_name])
                        parent_concept = ontology.get_concept(concept_map[parent_name])
                        if parent_concept:
                            parent_concept.add_child(concept_map[name])
    
    print(f"✓ Created {len(concepts_data)} concepts")
    
    # Add some properties to concepts
    python_concept = ontology.get_concept(concept_map["Python"])
    if python_concept:
        python_concept.add_property("paradigm", "multi-paradigm", PropertyType.DATATYPE)
        python_concept.add_property("first_appeared", 1991, PropertyType.DATATYPE)
        python_concept.add_property("typing", "dynamic", PropertyType.DATATYPE)
    
    tensorflow_concept = ontology.get_concept(concept_map["TensorFlow"])
    if tensorflow_concept:
        tensorflow_concept.add_property("developer", "Google", PropertyType.DATATYPE)
        tensorflow_concept.add_property("language", "Python", PropertyType.DATATYPE)
        tensorflow_concept.add_property("first_release", 2015, PropertyType.DATATYPE)
    
    # Perform various queries
    print(f"\nPerforming ontology queries:")
    
    # Query 1: Search for concepts containing "Python"
    print(f"\n1. Searching for 'Python':")
    python_results = await ontology.query_concepts("Python")
    for concept in python_results:
        print(f"   - {concept.name} ({concept.concept_type.value}): {concept.definition}")
        if concept.synonyms:
            print(f"     Synonyms: {', '.join(concept.synonyms)}")
    
    # Query 2: Search for programming languages only
    print(f"\n2. Searching for Programming Languages:")
    prog_lang_results = await ontology.query_concepts("", [ConceptType.CONCRETE])
    
    # Filter for programming languages by checking parent concepts
    programming_languages = []
    prog_lang_concept_id = concept_map.get("Programming Language")
    
    for concept in prog_lang_results:
        if prog_lang_concept_id in concept.get_ancestors(ontology):
            programming_languages.append(concept)
    
    for concept in programming_languages:
        print(f"   - {concept.name}: {concept.definition}")
        if concept.properties:
            print(f"     Properties: {list(concept.properties.keys())}")
    
    # Query 3: Search for machine learning related concepts
    print(f"\n3. Searching for 'machine learning' related concepts:")
    ml_results = await ontology.query_concepts("machine learning")
    for concept in ml_results:
        print(f"   - {concept.name} ({concept.concept_type.value}): {concept.definition}")
    
    # Query 4: Search by partial name
    print(f"\n4. Searching for concepts containing 'data':")
    data_results = await ontology.query_concepts("data")
    for concept in data_results:
        print(f"   - {concept.name}: {concept.definition}")
    
    # Show concept hierarchy
    print(f"\n5. Technology Concept Hierarchy:")
    hierarchy = await ontology.get_concept_hierarchy()
    
    def print_hierarchy(node, level=0):
        indent = "  " * level
        concept_data = node['concept']
        print(f"{indent}├─ {concept_data['name']}")
        
        for child_id, child_node in node['children'].items():
            print_hierarchy(child_node, level + 1)
    
    for root_id, root_node in hierarchy.items():
        print_hierarchy(root_node)
    
    # Show concept ancestors and descendants
    print(f"\n6. Python concept relationships:")
    python_concept = ontology.get_concept(concept_map["Python"])
    if python_concept:
        ancestors = python_concept.get_ancestors(ontology)
        descendants = python_concept.get_descendants(ontology)
        
        print(f"   Ancestors:")
        for ancestor_id in ancestors:
            ancestor = ontology.get_concept(ancestor_id)
            if ancestor:
                print(f"     - {ancestor.name}")
        
        print(f"   Descendants:")
        if descendants:
            for descendant_id in descendants:
                descendant = ontology.get_concept(descendant_id)
                if descendant:
                    print(f"     - {descendant.name}")
        else:
            print(f"     (No descendants)")
    
    # Final statistics
    stats = ontology.get_statistics()
    print(f"\nOntology Statistics:")
    print(f"  Total concepts: {stats['basic_statistics']['total_concepts']}")
    print(f"  Hierarchy depth: {stats['hierarchy_metrics']['max_depth']}")
    print(f"  Concept types: {stats['concept_distribution']}")

async def main():
    """
    Demonstrate Ontology Management for structured knowledge schema and reasoning
    
    WHAT YOU'LL LEARN:
    ================
    1. How to create and organize concepts in formal hierarchies
    2. How to define properties with domain, range, and constraints
    3. How to validate ontology consistency and detect logical errors
    4. How to perform automated reasoning and inference
    5. How to manage concept instances and their relationships
    6. How to query and search structured knowledge effectively
    
    REAL WORLD APPLICATIONS:
    =======================
    - Medical terminology systems (SNOMED CT, ICD)
    - Scientific classification systems (Gene Ontology)
    - Enterprise knowledge management
    - Semantic web and linked data applications
    - AI knowledge representation systems
    - Standards and compliance frameworks
    """
    
    print("ONTOLOGY MANAGEMENT DEMONSTRATION")
    print("Building structured knowledge schemas with formal reasoning!")
    
    await demo_concept_creation()
    await demo_property_management()
    await demo_validation_and_reasoning()
    await demo_instance_management()
    await demo_ontology_querying()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Concepts provide structured vocabulary with formal definitions")
    print("✓ Properties define relationships and constraints between concepts")
    print("✓ Validation ensures logical consistency and constraint compliance")
    print("✓ Reasoning derives new knowledge through logical inference")
    print("✓ Instances represent real-world examples of abstract concepts")
    print("✓ Querying enables intelligent search through structured knowledge")
    print("\nTHE POWER OF ONTOLOGY MANAGEMENT:")
    print("- Creates interoperable knowledge that works across systems")
    print("- Enables automated reasoning and intelligent inference")
    print("- Provides foundation for semantic understanding")
    print("- Ensures consistent terminology and meaning")

if __name__ == "__main__":
    asyncio.run(main())
