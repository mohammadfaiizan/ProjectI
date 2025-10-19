#!/usr/bin/env python3
"""
Knowledge Validation Systems: Ensuring Accuracy and Consistency
==============================================================

WHAT IS THE PROBLEM?
==================
Knowledge systems accumulate incorrect and inconsistent information:
- Extracted knowledge may contain factual errors and inaccuracies
- Information from different sources often contradicts each other
- Knowledge degrades over time as facts become outdated
- No systematic way to verify claims against reliable sources
- Inconsistent data leads to poor decision-making and unreliable AI
- Manual validation doesn't scale to large knowledge bases

Example: Medical Knowledge Validation Crisis
UNVALIDATED SYSTEM (Traditional):
- Medical AI recommends treatment based on outdated research
- Drug interaction database contains contradictory information
- Patient care decisions based on unverified medical claims
- Different medical systems provide conflicting diagnoses
- Life-threatening errors due to incorrect dosage information
- Result: Patient harm, medical malpractice, loss of trust

REAL WORLD EXAMPLE:
=================
How does Wikipedia maintain information quality?

WIKIPEDIA'S VALIDATION SYSTEM:
1. SOURCE VERIFICATION: All claims must cite reliable sources
2. CITATION REQUIREMENTS: Editorial guidelines for source quality
3. PEER REVIEW: Community editors review and fact-check content
4. VANDALISM DETECTION: Automated systems detect harmful edits
5. CONFLICT RESOLUTION: Dispute resolution processes for disagreements
6. EXPERTISE VALIDATION: Subject matter experts review complex topics
7. CONTINUOUS MONITORING: Real-time tracking of content changes

BENEFITS OF KNOWLEDGE VALIDATION:
- Ensures factual accuracy and reduces misinformation
- Maintains consistency across different information sources
- Builds trust in knowledge systems and AI recommendations
- Enables confident decision-making based on verified facts
- Prevents propagation of errors through dependent systems
- Provides audit trails for knowledge quality assurance

THE VALIDATION ADVANTAGE:
=======================
UNVALIDATED KNOWLEDGE: Information → Errors → Poor Decisions
VALIDATED KNOWLEDGE: Information → Verification → Reliable Decisions

VALIDATION COMPONENTS:
====================
1. SOURCE AUTHORITY ASSESSMENT: Evaluate credibility of information sources
2. FACT VERIFICATION: Check claims against authoritative references
3. CONSISTENCY CHECKING: Identify contradictions within knowledge base
4. TEMPORAL VALIDATION: Verify information currency and relevance
5. CROSS-REFERENCE VALIDATION: Compare information across multiple sources
6. EXPERT REVIEW SYSTEMS: Leverage human expertise for complex validation
7. AUTOMATED QUALITY SCORING: Continuous assessment of knowledge quality

WHY THIS IS REVOLUTIONARY:
========================
- Enables trustworthy AI systems that users can rely on
- Prevents the spread of misinformation and false claims
- Provides foundation for high-stakes decision support systems
- Critical for medical, legal, financial, and safety applications
- Enables automated quality assurance at scale
- Creates competitive advantage through superior information quality
"""

import asyncio
import time
import json
import uuid
import re
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ValidationMethod(Enum):
    """Methods for validating knowledge"""
    SOURCE_AUTHORITY = "source_authority"
    FACT_CHECKING = "fact_checking"
    CROSS_REFERENCE = "cross_reference"
    TEMPORAL_CHECK = "temporal_check"
    CONSISTENCY_CHECK = "consistency_check"
    EXPERT_REVIEW = "expert_review"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    AUTOMATED_SCORING = "automated_scoring"

class ValidationResult(Enum):
    """Results of validation checks"""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    OUTDATED = "outdated"
    CONFLICTING = "conflicting"
    NEEDS_REVIEW = "needs_review"

class SourceType(Enum):
    """Types of information sources"""
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    GOVERNMENT_DOCUMENT = "government_document"
    ENCYCLOPEDIA = "encyclopedia"
    WEBSITE = "website"
    BOOK = "book"
    EXPERT_STATEMENT = "expert_statement"
    DATABASE_RECORD = "database_record"

class FactType(Enum):
    """Types of facts to validate"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    DEFINITIONAL = "definitional"
    CAUSAL = "causal"
    BIOGRAPHICAL = "biographical"
    GEOGRAPHICAL = "geographical"

@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge to be validated"""
    
    id: str
    content: str
    fact_type: FactType
    
    # Structured representation
    subject: str = ""
    predicate: str = ""
    object: str = ""
    
    # Source information
    source_url: str = ""
    source_type: SourceType = SourceType.WEBSITE
    extraction_date: datetime = field(default_factory=datetime.now)
    
    # Context
    context: str = ""
    domain: str = ""
    
    # Current validation status
    validation_status: ValidationResult = ValidationResult.UNCERTAIN
    confidence_score: float = 0.5
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ValidationRule:
    """Represents a validation rule"""
    
    id: str
    name: str
    description: str
    validation_method: ValidationMethod
    
    # Rule parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Applicability
    applicable_fact_types: List[FactType] = field(default_factory=list)
    applicable_domains: List[str] = field(default_factory=list)
    
    # Rule quality
    accuracy: float = 0.9
    precision: float = 0.8
    recall: float = 0.7
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ValidationCheck:
    """Represents a single validation check"""
    
    id: str
    knowledge_item_id: str
    rule_id: str
    validation_method: ValidationMethod
    
    # Check results
    result: ValidationResult = ValidationResult.UNCERTAIN
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    
    # Additional information
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    # Metadata
    performed_at: datetime = field(default_factory=datetime.now)
    performed_by: str = "system"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ValidationReport:
    """Complete validation report for a knowledge item"""
    
    knowledge_item_id: str
    
    # Overall assessment
    overall_result: ValidationResult = ValidationResult.UNCERTAIN
    overall_confidence: float = 0.5
    
    # Individual checks
    validation_checks: List[ValidationCheck] = field(default_factory=list)
    
    # Aggregated results
    method_results: Dict[str, ValidationResult] = field(default_factory=dict)
    confidence_by_method: Dict[str, float] = field(default_factory=dict)
    
    # Issues and recommendations
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Quality metrics
    source_quality_score: float = 0.0
    consistency_score: float = 0.0
    temporal_relevance_score: float = 0.0
    
    # Metadata
    validation_date: datetime = field(default_factory=datetime.now)
    validation_duration: float = 0.0
    
    def add_check(self, check: ValidationCheck) -> None:
        """Add validation check to report"""
        self.validation_checks.append(check)
        self.method_results[check.validation_method.value] = check.result
        self.confidence_by_method[check.validation_method.value] = check.confidence

class SourceAuthorityValidator:
    """Validates the authority and credibility of information sources"""
    
    def __init__(self):
        # Authority scoring parameters
        self.domain_authority_scores = {
            # Academic and research domains
            '.edu': 0.9,
            '.ac.uk': 0.9,
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'arxiv.org': 0.8,
            'ieee.org': 0.9,
            'acm.org': 0.9,
            'nature.com': 0.95,
            'science.org': 0.95,
            
            # Government domains
            '.gov': 0.85,
            'who.int': 0.9,
            'cdc.gov': 0.9,
            'fda.gov': 0.9,
            'nih.gov': 0.9,
            
            # Established organizations
            'wikipedia.org': 0.7,
            'britannica.com': 0.8,
            'reuters.com': 0.8,
            'bbc.com': 0.8,
            'npr.org': 0.8,
            
            # Lower authority
            '.com': 0.5,
            '.org': 0.6,
            '.net': 0.4
        }
        
        self.logger = logging.getLogger("SourceAuthorityValidator")
    
    async def validate_source_authority(self, knowledge_item: KnowledgeItem) -> ValidationCheck:
        """Validate the authority of the source"""
        
        start_time = time.time()
        
        check = ValidationCheck(
            id="",
            knowledge_item_id=knowledge_item.id,
            rule_id="source_authority_rule",
            validation_method=ValidationMethod.SOURCE_AUTHORITY
        )
        
        try:
            if not knowledge_item.source_url:
                check.result = ValidationResult.UNCERTAIN
                check.confidence = 0.1
                check.evidence.append("No source URL provided")
                return check
            
            # Parse URL to extract domain
            parsed_url = urlparse(knowledge_item.source_url)
            domain = parsed_url.netloc.lower()
            
            # Calculate authority score
            authority_score = await self._calculate_authority_score(domain, knowledge_item.source_type)
            
            # Additional factors
            recency_score = await self._calculate_recency_score(knowledge_item.extraction_date)
            
            # Combined score
            combined_score = (authority_score * 0.7) + (recency_score * 0.3)
            
            # Determine result
            if combined_score >= 0.8:
                check.result = ValidationResult.VALID
            elif combined_score >= 0.6:
                check.result = ValidationResult.UNCERTAIN
            else:
                check.result = ValidationResult.INVALID
            
            check.confidence = combined_score
            check.evidence.append(f"Domain authority score: {authority_score:.2f}")
            check.evidence.append(f"Recency score: {recency_score:.2f}")
            check.evidence.append(f"Source type: {knowledge_item.source_type.value}")
            
            check.details = {
                'domain': domain,
                'authority_score': authority_score,
                'recency_score': recency_score,
                'combined_score': combined_score
            }
            
        except Exception as e:
            self.logger.error(f"Source authority validation failed: {e}")
            check.result = ValidationResult.UNCERTAIN
            check.confidence = 0.1
            check.evidence.append(f"Validation error: {str(e)}")
        
        check.execution_time = time.time() - start_time
        return check
    
    async def _calculate_authority_score(self, domain: str, source_type: SourceType) -> float:
        """Calculate authority score for domain"""
        
        # Check for exact domain matches
        for auth_domain, score in self.domain_authority_scores.items():
            if domain == auth_domain or domain.endswith(auth_domain):
                return score
        
        # Boost score based on source type
        source_type_boosts = {
            SourceType.ACADEMIC_PAPER: 0.2,
            SourceType.GOVERNMENT_DOCUMENT: 0.15,
            SourceType.ENCYCLOPEDIA: 0.1,
            SourceType.EXPERT_STATEMENT: 0.1,
            SourceType.NEWS_ARTICLE: 0.05,
            SourceType.WEBSITE: 0.0,
            SourceType.DATABASE_RECORD: 0.1
        }
        
        base_score = 0.5  # Default score
        boost = source_type_boosts.get(source_type, 0.0)
        
        return min(1.0, base_score + boost)
    
    async def _calculate_recency_score(self, extraction_date: datetime) -> float:
        """Calculate recency score based on extraction date"""
        
        days_old = (datetime.now() - extraction_date).days
        
        if days_old <= 30:
            return 1.0
        elif days_old <= 90:
            return 0.9
        elif days_old <= 180:
            return 0.8
        elif days_old <= 365:
            return 0.7
        elif days_old <= 730:  # 2 years
            return 0.5
        else:
            return 0.3

class FactCheckingValidator:
    """Validates facts against reference sources"""
    
    def __init__(self):
        # Reference sources for fact checking
        self.reference_sources = {
            'numerical_facts': [
                'worldbank.org',
                'data.gov',
                'census.gov',
                'statista.com'
            ],
            'biographical_facts': [
                'biography.com',
                'britannica.com',
                'wikipedia.org'
            ],
            'geographical_facts': [
                'cia.gov/the-world-factbook',
                'nationsonline.org',
                'worldatlas.com'
            ],
            'scientific_facts': [
                'pubmed.ncbi.nlm.nih.gov',
                'nature.com',
                'science.org'
            ]
        }
        
        # Fact patterns for extraction
        self.fact_patterns = {
            FactType.NUMERICAL: [
                r'\$?[\d,]+\.?\d*\s*(million|billion|trillion|thousand)?',
                r'\d+\.\d+\s*%',
                r'\d{4}\s*(?:years?|months?|days?)'
            ],
            FactType.TEMPORAL: [
                r'\b\d{4}\b',  # Years
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}',
                r'\d{1,2}/\d{1,2}/\d{4}'
            ],
            FactType.BIOGRAPHICAL: [
                r'born\s+(?:in\s+)?\d{4}',
                r'died\s+(?:in\s+)?\d{4}',
                r'age\s+\d+'
            ]
        }
        
        self.logger = logging.getLogger("FactCheckingValidator")
    
    async def validate_fact(self, knowledge_item: KnowledgeItem) -> ValidationCheck:
        """Validate a fact against reference sources"""
        
        start_time = time.time()
        
        check = ValidationCheck(
            id="",
            knowledge_item_id=knowledge_item.id,
            rule_id="fact_checking_rule",
            validation_method=ValidationMethod.FACT_CHECKING
        )
        
        try:
            # Extract fact claims from content
            claims = await self._extract_fact_claims(knowledge_item)
            
            if not claims:
                check.result = ValidationResult.UNCERTAIN
                check.confidence = 0.3
                check.evidence.append("No verifiable claims found")
                return check
            
            # Validate each claim
            claim_validations = []
            
            for claim in claims:
                validation_result = await self._validate_claim(claim, knowledge_item.fact_type)
                claim_validations.append(validation_result)
            
            # Aggregate results
            valid_claims = sum(1 for v in claim_validations if v['valid'])
            total_claims = len(claim_validations)
            
            accuracy_ratio = valid_claims / total_claims if total_claims > 0 else 0
            
            # Determine overall result
            if accuracy_ratio >= 0.8:
                check.result = ValidationResult.VALID
                check.confidence = 0.8 + (accuracy_ratio - 0.8) * 0.5
            elif accuracy_ratio >= 0.5:
                check.result = ValidationResult.UNCERTAIN
                check.confidence = 0.5 + (accuracy_ratio - 0.5) * 0.6
            else:
                check.result = ValidationResult.INVALID
                check.confidence = 0.2 + accuracy_ratio * 0.5
            
            check.evidence.append(f"Validated {valid_claims}/{total_claims} claims")
            check.evidence.extend([f"Claim: {v['claim']} - {'Valid' if v['valid'] else 'Invalid'}" 
                                 for v in claim_validations[:3]])  # Show first 3
            
            check.details = {
                'total_claims': total_claims,
                'valid_claims': valid_claims,
                'accuracy_ratio': accuracy_ratio,
                'claim_validations': claim_validations
            }
            
        except Exception as e:
            self.logger.error(f"Fact checking validation failed: {e}")
            check.result = ValidationResult.UNCERTAIN
            check.confidence = 0.1
            check.evidence.append(f"Validation error: {str(e)}")
        
        check.execution_time = time.time() - start_time
        return check
    
    async def _extract_fact_claims(self, knowledge_item: KnowledgeItem) -> List[str]:
        """Extract verifiable claims from knowledge item"""
        
        claims = []
        content = knowledge_item.content
        
        # Extract based on fact type
        patterns = self.fact_patterns.get(knowledge_item.fact_type, [])
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches)
        
        # For structured facts, use subject-predicate-object
        if knowledge_item.subject and knowledge_item.predicate and knowledge_item.object:
            claims.append(f"{knowledge_item.subject} {knowledge_item.predicate} {knowledge_item.object}")
        
        return list(set(claims))  # Remove duplicates
    
    async def _validate_claim(self, claim: str, fact_type: FactType) -> Dict[str, Any]:
        """Validate a specific claim"""
        
        # Simulate fact checking against reference sources
        # In a real implementation, this would query actual databases/APIs
        
        validation_result = {
            'claim': claim,
            'valid': False,
            'confidence': 0.5,
            'sources_checked': 0
        }
        
        # Get reference sources for fact type
        fact_type_mapping = {
            FactType.NUMERICAL: 'numerical_facts',
            FactType.BIOGRAPHICAL: 'biographical_facts',
            FactType.GEOGRAPHICAL: 'geographical_facts',
            FactType.TEMPORAL: 'numerical_facts'  # Often in statistical sources
        }
        
        source_category = fact_type_mapping.get(fact_type, 'numerical_facts')
        reference_sources = self.reference_sources.get(source_category, [])
        
        # Simulate checking against reference sources
        sources_supporting = 0
        sources_checked = min(3, len(reference_sources))  # Check up to 3 sources
        
        for source in reference_sources[:sources_checked]:
            # Simulate source checking with some randomness
            if await self._simulate_source_check(claim, source):
                sources_supporting += 1
        
        if sources_checked > 0:
            support_ratio = sources_supporting / sources_checked
            validation_result['valid'] = support_ratio >= 0.5
            validation_result['confidence'] = 0.3 + support_ratio * 0.7
            validation_result['sources_checked'] = sources_checked
        
        return validation_result
    
    async def _simulate_source_check(self, claim: str, source: str) -> bool:
        """Simulate checking claim against a reference source"""
        
        # This is a simulation - in real implementation, would query actual sources
        # Return random result biased towards truth for demonstration
        
        # Add some logic based on claim content and source
        claim_lower = claim.lower()
        
        # High-authority sources more likely to validate
        high_authority = any(domain in source for domain in ['gov', 'edu', 'nature', 'science'])
        
        if high_authority:
            return random.random() > 0.3  # 70% chance of validation
        else:
            return random.random() > 0.5  # 50% chance of validation

class ConsistencyValidator:
    """Validates consistency within knowledge base"""
    
    def __init__(self):
        # Consistency rules
        self.consistency_rules = [
            {
                'name': 'temporal_consistency',
                'description': 'Events should follow temporal logic',
                'pattern': r'born.*(\d{4}).*died.*(\d{4})',
                'validator': self._validate_temporal_order
            },
            {
                'name': 'numerical_consistency',
                'description': 'Numerical facts should be consistent',
                'pattern': r'(\d+(?:,\d{3})*(?:\.\d+)?)',
                'validator': self._validate_numerical_consistency
            },
            {
                'name': 'geographical_consistency',
                'description': 'Geographical relationships should be consistent',
                'pattern': r'([\w\s]+)\s+(?:in|located in)\s+([\w\s]+)',
                'validator': self._validate_geographical_consistency
            }
        ]
        
        self.logger = logging.getLogger("ConsistencyValidator")
    
    async def validate_consistency(self, knowledge_items: List[KnowledgeItem], 
                                 target_item: KnowledgeItem) -> ValidationCheck:
        """Validate consistency of target item against knowledge base"""
        
        start_time = time.time()
        
        check = ValidationCheck(
            id="",
            knowledge_item_id=target_item.id,
            rule_id="consistency_check_rule",
            validation_method=ValidationMethod.CONSISTENCY_CHECK
        )
        
        try:
            # Find related knowledge items
            related_items = await self._find_related_items(target_item, knowledge_items)
            
            if not related_items:
                check.result = ValidationResult.UNCERTAIN
                check.confidence = 0.5
                check.evidence.append("No related items found for consistency check")
                return check
            
            # Apply consistency rules
            consistency_results = []
            
            for rule in self.consistency_rules:
                result = await rule['validator'](target_item, related_items)
                if result:
                    consistency_results.append(result)
            
            # Evaluate overall consistency
            if not consistency_results:
                check.result = ValidationResult.UNCERTAIN
                check.confidence = 0.5
                check.evidence.append("No applicable consistency rules")
            else:
                consistent_checks = sum(1 for r in consistency_results if r['consistent'])
                total_checks = len(consistency_results)
                consistency_ratio = consistent_checks / total_checks
                
                if consistency_ratio >= 0.8:
                    check.result = ValidationResult.VALID
                    check.confidence = 0.7 + consistency_ratio * 0.3
                elif consistency_ratio >= 0.5:
                    check.result = ValidationResult.UNCERTAIN
                    check.confidence = 0.4 + consistency_ratio * 0.4
                else:
                    check.result = ValidationResult.CONFLICTING
                    check.confidence = 0.2 + consistency_ratio * 0.3
                
                check.evidence.append(f"Consistency: {consistent_checks}/{total_checks} rules passed")
                check.evidence.extend([f"{r['rule']}: {'Consistent' if r['consistent'] else 'Inconsistent'}" 
                                     for r in consistency_results])
            
            check.details = {
                'related_items_count': len(related_items),
                'consistency_checks': consistency_results,
                'consistency_ratio': consistency_ratio if consistency_results else 0
            }
            
        except Exception as e:
            self.logger.error(f"Consistency validation failed: {e}")
            check.result = ValidationResult.UNCERTAIN
            check.confidence = 0.1
            check.evidence.append(f"Validation error: {str(e)}")
        
        check.execution_time = time.time() - start_time
        return check
    
    async def _find_related_items(self, target_item: KnowledgeItem, 
                                knowledge_items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        """Find knowledge items related to target item"""
        
        related_items = []
        target_content = target_item.content.lower()
        target_subject = target_item.subject.lower() if target_item.subject else ""
        
        for item in knowledge_items:
            if item.id == target_item.id:
                continue
            
            # Check for content overlap
            item_content = item.content.lower()
            common_words = set(target_content.split()) & set(item_content.split())
            
            if len(common_words) >= 2:  # At least 2 common words
                related_items.append(item)
                continue
            
            # Check for subject overlap
            if target_subject and item.subject:
                if target_subject in item.subject.lower() or item.subject.lower() in target_subject:
                    related_items.append(item)
                    continue
            
            # Check for same domain
            if target_item.domain and item.domain == target_item.domain:
                related_items.append(item)
        
        return related_items
    
    async def _validate_temporal_order(self, target_item: KnowledgeItem, 
                                     related_items: List[KnowledgeItem]) -> Optional[Dict[str, Any]]:
        """Validate temporal consistency"""
        
        # Extract dates from target item
        content = target_item.content
        date_pattern = r'(\d{4})'
        dates = re.findall(date_pattern, content)
        
        if len(dates) < 2:
            return None
        
        # Check if dates are in logical order
        dates = [int(d) for d in dates]
        dates.sort()
        
        # For biographical data, birth should come before death
        if 'born' in content.lower() and 'died' in content.lower():
            birth_match = re.search(r'born.*?(\d{4})', content.lower())
            death_match = re.search(r'died.*?(\d{4})', content.lower())
            
            if birth_match and death_match:
                birth_year = int(birth_match.group(1))
                death_year = int(death_match.group(1))
                
                consistent = birth_year < death_year
                
                return {
                    'rule': 'temporal_order',
                    'consistent': consistent,
                    'details': f"Birth: {birth_year}, Death: {death_year}"
                }
        
        return None
    
    async def _validate_numerical_consistency(self, target_item: KnowledgeItem, 
                                            related_items: List[KnowledgeItem]) -> Optional[Dict[str, Any]]:
        """Validate numerical consistency"""
        
        # Extract numbers from target and related items
        number_pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)'
        target_numbers = re.findall(number_pattern, target_item.content)
        
        if not target_numbers:
            return None
        
        # Check for conflicting numbers in related items
        conflicts = 0
        comparisons = 0
        
        for related_item in related_items:
            related_numbers = re.findall(number_pattern, related_item.content)
            
            for target_num in target_numbers:
                for related_num in related_numbers:
                    # Convert to float for comparison
                    try:
                        target_val = float(target_num.replace(',', ''))
                        related_val = float(related_num.replace(',', ''))
                        
                        # Check if numbers are very different (more than 10% difference)
                        if abs(target_val - related_val) / max(target_val, related_val) > 0.1:
                            conflicts += 1
                        
                        comparisons += 1
                        
                    except ValueError:
                        continue
        
        if comparisons > 0:
            consistent = conflicts / comparisons < 0.3  # Less than 30% conflicts
            
            return {
                'rule': 'numerical_consistency',
                'consistent': consistent,
                'details': f"Conflicts: {conflicts}/{comparisons}"
            }
        
        return None
    
    async def _validate_geographical_consistency(self, target_item: KnowledgeItem, 
                                               related_items: List[KnowledgeItem]) -> Optional[Dict[str, Any]]:
        """Validate geographical consistency"""
        
        # Extract geographical relationships
        geo_pattern = r'([\w\s]+)\s+(?:in|located in)\s+([\w\s]+)'
        target_matches = re.findall(geo_pattern, target_item.content, re.IGNORECASE)
        
        if not target_matches:
            return None
        
        # Check for conflicting geographical information
        conflicts = 0
        total_checks = 0
        
        for related_item in related_items:
            related_matches = re.findall(geo_pattern, related_item.content, re.IGNORECASE)
            
            for target_place, target_location in target_matches:
                for related_place, related_location in related_matches:
                    # Check if same place has different locations
                    if target_place.strip().lower() == related_place.strip().lower():
                        if target_location.strip().lower() != related_location.strip().lower():
                            conflicts += 1
                        total_checks += 1
        
        if total_checks > 0:
            consistent = conflicts == 0
            
            return {
                'rule': 'geographical_consistency',
                'consistent': consistent,
                'details': f"Conflicts: {conflicts}/{total_checks}"
            }
        
        return None

class KnowledgeValidationSystem:
    """Complete knowledge validation system"""
    
    def __init__(self):
        # Validation components
        self.source_validator = SourceAuthorityValidator()
        self.fact_validator = FactCheckingValidator()
        self.consistency_validator = ConsistencyValidator()
        
        # Validation rules
        self.validation_rules: List[ValidationRule] = []
        
        # Knowledge base for consistency checking
        self.knowledge_base: List[KnowledgeItem] = []
        
        # System configuration
        self.min_confidence_threshold = 0.6
        self.require_multiple_validations = True
        
        # Statistics
        self.stats = {
            'items_validated': 0,
            'validation_checks_performed': 0,
            'valid_items': 0,
            'invalid_items': 0,
            'uncertain_items': 0,
            'total_validation_time': 0.0
        }
        
        self.logger = logging.getLogger("KnowledgeValidationSystem")
    
    async def initialize(self) -> None:
        """Initialize validation system"""
        await self._load_validation_rules()
        self.logger.info("Knowledge validation system initialized")
    
    async def validate_knowledge_item(self, knowledge_item: KnowledgeItem) -> ValidationReport:
        """Validate a single knowledge item"""
        
        start_time = time.time()
        
        report = ValidationReport(knowledge_item_id=knowledge_item.id)
        
        try:
            self.logger.info(f"Validating knowledge item: {knowledge_item.id}")
            
            # Perform source authority validation
            source_check = await self.source_validator.validate_source_authority(knowledge_item)
            report.add_check(source_check)
            
            # Perform fact checking validation
            fact_check = await self.fact_validator.validate_fact(knowledge_item)
            report.add_check(fact_check)
            
            # Perform consistency validation if knowledge base exists
            if self.knowledge_base:
                consistency_check = await self.consistency_validator.validate_consistency(
                    self.knowledge_base, knowledge_item
                )
                report.add_check(consistency_check)
            
            # Calculate overall assessment
            await self._calculate_overall_assessment(report)
            
            # Generate recommendations
            await self._generate_recommendations(report, knowledge_item)
            
            validation_time = time.time() - start_time
            report.validation_duration = validation_time
            
            # Update statistics
            self.stats['items_validated'] += 1
            self.stats['validation_checks_performed'] += len(report.validation_checks)
            self.stats['total_validation_time'] += validation_time
            
            if report.overall_result == ValidationResult.VALID:
                self.stats['valid_items'] += 1
            elif report.overall_result == ValidationResult.INVALID:
                self.stats['invalid_items'] += 1
            else:
                self.stats['uncertain_items'] += 1
            
            self.logger.info(f"Validation completed: {report.overall_result.value}, "
                           f"confidence: {report.overall_confidence:.3f}")
            
        except Exception as e:
            self.logger.error(f"Knowledge validation failed: {e}")
            report.issues_found.append(f"Validation error: {str(e)}")
            report.overall_result = ValidationResult.UNCERTAIN
            report.overall_confidence = 0.1
        
        return report
    
    async def validate_knowledge_batch(self, knowledge_items: List[KnowledgeItem]) -> List[ValidationReport]:
        """Validate multiple knowledge items"""
        
        reports = []
        
        self.logger.info(f"Validating {len(knowledge_items)} knowledge items")
        
        for i, item in enumerate(knowledge_items, 1):
            self.logger.debug(f"Validating item {i}/{len(knowledge_items)}: {item.id}")
            
            report = await self.validate_knowledge_item(item)
            reports.append(report)
            
            # Add to knowledge base for future consistency checks
            if report.overall_result == ValidationResult.VALID:
                self.knowledge_base.append(item)
            
            if i % 10 == 0:
                self.logger.info(f"Validated {i}/{len(knowledge_items)} items")
        
        return reports
    
    async def _load_validation_rules(self) -> None:
        """Load validation rules"""
        
        # Default validation rules
        rules = [
            ValidationRule(
                id="source_authority",
                name="Source Authority Check",
                description="Validates the authority and credibility of information sources",
                validation_method=ValidationMethod.SOURCE_AUTHORITY,
                applicable_fact_types=list(FactType),
                accuracy=0.85
            ),
            ValidationRule(
                id="fact_verification",
                name="Fact Verification",
                description="Verifies facts against authoritative reference sources",
                validation_method=ValidationMethod.FACT_CHECKING,
                applicable_fact_types=[FactType.NUMERICAL, FactType.TEMPORAL, FactType.BIOGRAPHICAL],
                accuracy=0.9
            ),
            ValidationRule(
                id="consistency_check",
                name="Consistency Check",
                description="Checks for consistency within the knowledge base",
                validation_method=ValidationMethod.CONSISTENCY_CHECK,
                applicable_fact_types=list(FactType),
                accuracy=0.8
            )
        ]
        
        self.validation_rules = rules
        self.logger.debug(f"Loaded {len(rules)} validation rules")
    
    async def _calculate_overall_assessment(self, report: ValidationReport) -> None:
        """Calculate overall assessment from individual checks"""
        
        if not report.validation_checks:
            report.overall_result = ValidationResult.UNCERTAIN
            report.overall_confidence = 0.1
            return
        
        # Weight different validation methods
        method_weights = {
            ValidationMethod.SOURCE_AUTHORITY: 0.3,
            ValidationMethod.FACT_CHECKING: 0.4,
            ValidationMethod.CONSISTENCY_CHECK: 0.3
        }
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        
        valid_checks = 0
        invalid_checks = 0
        uncertain_checks = 0
        
        for check in report.validation_checks:
            weight = method_weights.get(check.validation_method, 0.2)
            weighted_confidence += check.confidence * weight
            total_weight += weight
            
            if check.result == ValidationResult.VALID:
                valid_checks += 1
            elif check.result == ValidationResult.INVALID:
                invalid_checks += 1
            else:
                uncertain_checks += 1
        
        # Calculate overall confidence
        if total_weight > 0:
            report.overall_confidence = weighted_confidence / total_weight
        else:
            report.overall_confidence = 0.5
        
        # Determine overall result
        total_checks = len(report.validation_checks)
        
        if valid_checks / total_checks >= 0.7 and report.overall_confidence >= self.min_confidence_threshold:
            report.overall_result = ValidationResult.VALID
        elif invalid_checks / total_checks >= 0.5:
            report.overall_result = ValidationResult.INVALID
        elif any(check.result == ValidationResult.CONFLICTING for check in report.validation_checks):
            report.overall_result = ValidationResult.CONFLICTING
        else:
            report.overall_result = ValidationResult.UNCERTAIN
    
    async def _generate_recommendations(self, report: ValidationReport, 
                                      knowledge_item: KnowledgeItem) -> None:
        """Generate recommendations for improving validation"""
        
        recommendations = []
        
        # Check source authority issues
        source_checks = [c for c in report.validation_checks 
                        if c.validation_method == ValidationMethod.SOURCE_AUTHORITY]
        
        for check in source_checks:
            if check.result != ValidationResult.VALID:
                recommendations.append("Consider finding sources with higher authority scores")
                recommendations.append("Look for academic, government, or peer-reviewed sources")
        
        # Check fact validation issues
        fact_checks = [c for c in report.validation_checks 
                      if c.validation_method == ValidationMethod.FACT_CHECKING]
        
        for check in fact_checks:
            if check.result == ValidationResult.INVALID:
                recommendations.append("Verify facts against multiple authoritative sources")
                recommendations.append("Check for updated or corrected information")
        
        # Check consistency issues
        consistency_checks = [c for c in report.validation_checks 
                            if c.validation_method == ValidationMethod.CONSISTENCY_CHECK]
        
        for check in consistency_checks:
            if check.result == ValidationResult.CONFLICTING:
                recommendations.append("Resolve conflicting information with additional sources")
                recommendations.append("Consider temporal context - information may have changed")
        
        # Overall recommendations
        if report.overall_confidence < 0.5:
            recommendations.append("Seek additional validation from domain experts")
            recommendations.append("Consider marking this information as uncertain")
        
        report.recommendations = recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics"""
        
        avg_validation_time = (
            self.stats['total_validation_time'] / max(self.stats['items_validated'], 1)
        )
        
        return {
            'validation_statistics': self.stats,
            'performance_metrics': {
                'average_validation_time': avg_validation_time,
                'validation_throughput': self.stats['items_validated'] / max(self.stats['total_validation_time'], 1)
            },
            'validation_rules_count': len(self.validation_rules),
            'knowledge_base_size': len(self.knowledge_base)
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_source_authority_validation():
    """Demo: Source authority validation"""
    print("\nDEMO 1: SOURCE AUTHORITY VALIDATION")
    print("=" * 50)
    
    validator = SourceAuthorityValidator()
    
    # Test knowledge items with different source authorities
    test_items = [
        KnowledgeItem(
            id="",
            content="Machine learning is a subset of artificial intelligence",
            fact_type=FactType.DEFINITIONAL,
            source_url="https://nature.com/articles/ml-overview",
            source_type=SourceType.ACADEMIC_PAPER,
            extraction_date=datetime.now() - timedelta(days=30)
        ),
        KnowledgeItem(
            id="",
            content="The population of the United States is 331 million",
            fact_type=FactType.NUMERICAL,
            source_url="https://census.gov/population-data",
            source_type=SourceType.GOVERNMENT_DOCUMENT,
            extraction_date=datetime.now() - timedelta(days=60)
        ),
        KnowledgeItem(
            id="",
            content="Artificial intelligence will replace all jobs",
            fact_type=FactType.CATEGORICAL,
            source_url="https://random-blog.com/ai-predictions",
            source_type=SourceType.WEBSITE,
            extraction_date=datetime.now() - timedelta(days=800)
        ),
        KnowledgeItem(
            id="",
            content="COVID-19 vaccines are effective against severe disease",
            fact_type=FactType.DEFINITIONAL,
            source_url="https://cdc.gov/covid-vaccines",
            source_type=SourceType.GOVERNMENT_DOCUMENT,
            extraction_date=datetime.now() - timedelta(days=90)
        )
    ]
    
    print("Testing source authority validation:")
    
    for i, item in enumerate(test_items, 1):
        print(f"\n--- Test Item {i} ---")
        print(f"Content: {item.content}")
        print(f"Source: {item.source_url}")
        print(f"Type: {item.source_type.value}")
        print(f"Age: {(datetime.now() - item.extraction_date).days} days")
        
        check = await validator.validate_source_authority(item)
        
        print(f"\nValidation Result:")
        print(f"  Result: {check.result.value}")
        print(f"  Confidence: {check.confidence:.3f}")
        print(f"  Evidence:")
        for evidence in check.evidence:
            print(f"    - {evidence}")
        
        if check.details:
            details = check.details
            print(f"  Details:")
            print(f"    Domain: {details.get('domain', 'N/A')}")
            print(f"    Authority Score: {details.get('authority_score', 0):.3f}")
            print(f"    Recency Score: {details.get('recency_score', 0):.3f}")

async def demo_fact_checking_validation():
    """Demo: Fact checking validation"""
    print("\nDEMO 2: FACT CHECKING VALIDATION")
    print("=" * 50)
    
    validator = FactCheckingValidator()
    
    # Test knowledge items with different types of facts
    test_items = [
        KnowledgeItem(
            id="",
            content="The Earth's population reached 8 billion people in 2022",
            fact_type=FactType.NUMERICAL,
            subject="Earth's population",
            predicate="reached",
            object="8 billion people in 2022"
        ),
        KnowledgeItem(
            id="",
            content="Albert Einstein was born in 1879 and died in 1955",
            fact_type=FactType.BIOGRAPHICAL,
            subject="Albert Einstein",
            predicate="lived",
            object="1879-1955"
        ),
        KnowledgeItem(
            id="",
            content="Mount Everest is located in Nepal and China",
            fact_type=FactType.GEOGRAPHICAL,
            subject="Mount Everest",
            predicate="located_in",
            object="Nepal and China"
        ),
        KnowledgeItem(
            id="",
            content="The iPhone 15 costs $999 and was released in September 2023",
            fact_type=FactType.NUMERICAL,
            subject="iPhone 15",
            predicate="costs",
            object="$999"
        )
    ]
    
    print("Testing fact checking validation:")
    
    for i, item in enumerate(test_items, 1):
        print(f"\n--- Test Item {i} ---")
        print(f"Content: {item.content}")
        print(f"Fact Type: {item.fact_type.value}")
        print(f"Structured: {item.subject} {item.predicate} {item.object}")
        
        check = await validator.validate_fact(item)
        
        print(f"\nValidation Result:")
        print(f"  Result: {check.result.value}")
        print(f"  Confidence: {check.confidence:.3f}")
        print(f"  Evidence:")
        for evidence in check.evidence:
            print(f"    - {evidence}")
        
        if check.details:
            details = check.details
            print(f"  Details:")
            print(f"    Claims Found: {details.get('total_claims', 0)}")
            print(f"    Valid Claims: {details.get('valid_claims', 0)}")
            print(f"    Accuracy Ratio: {details.get('accuracy_ratio', 0):.3f}")

async def demo_consistency_validation():
    """Demo: Consistency validation"""
    print("\nDEMO 3: CONSISTENCY VALIDATION")
    print("=" * 50)
    
    validator = ConsistencyValidator()
    
    # Create knowledge base with some items
    knowledge_base = [
        KnowledgeItem(
            id="kb_1",
            content="Steve Jobs was born in 1955 and died in 2011",
            fact_type=FactType.BIOGRAPHICAL,
            subject="Steve Jobs"
        ),
        KnowledgeItem(
            id="kb_2", 
            content="Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak",
            fact_type=FactType.TEMPORAL,
            subject="Apple Inc."
        ),
        KnowledgeItem(
            id="kb_3",
            content="The iPhone was first released in 2007 with a price of $499",
            fact_type=FactType.NUMERICAL,
            subject="iPhone"
        ),
        KnowledgeItem(
            id="kb_4",
            content="Apple Inc. is located in Cupertino, California",
            fact_type=FactType.GEOGRAPHICAL,
            subject="Apple Inc."
        )
    ]
    
    # Test items for consistency
    test_items = [
        KnowledgeItem(
            id="",
            content="Steve Jobs was born in 1955 and founded Apple in 1976",
            fact_type=FactType.BIOGRAPHICAL,
            subject="Steve Jobs"
        ),
        KnowledgeItem(
            id="",
            content="Steve Jobs died in 1950 after founding Apple",  # Inconsistent - died before birth
            fact_type=FactType.BIOGRAPHICAL,
            subject="Steve Jobs"
        ),
        KnowledgeItem(
            id="",
            content="The original iPhone cost $599 when released in 2007",  # Slight price inconsistency
            fact_type=FactType.NUMERICAL,
            subject="iPhone"
        ),
        KnowledgeItem(
            id="",
            content="Apple Inc. is headquartered in New York City",  # Location inconsistency
            fact_type=FactType.GEOGRAPHICAL,
            subject="Apple Inc."
        )
    ]
    
    print("Testing consistency validation:")
    print(f"Knowledge base contains {len(knowledge_base)} items")
    
    for i, item in enumerate(test_items, 1):
        print(f"\n--- Test Item {i} ---")
        print(f"Content: {item.content}")
        print(f"Subject: {item.subject}")
        
        check = await validator.validate_consistency(knowledge_base, item)
        
        print(f"\nValidation Result:")
        print(f"  Result: {check.result.value}")
        print(f"  Confidence: {check.confidence:.3f}")
        print(f"  Evidence:")
        for evidence in check.evidence:
            print(f"    - {evidence}")
        
        if check.details:
            details = check.details
            print(f"  Details:")
            print(f"    Related Items: {details.get('related_items_count', 0)}")
            print(f"    Consistency Ratio: {details.get('consistency_ratio', 0):.3f}")

async def demo_complete_validation_system():
    """Demo: Complete knowledge validation system"""
    print("\nDEMO 4: COMPLETE VALIDATION SYSTEM")
    print("=" * 50)
    
    system = KnowledgeValidationSystem()
    await system.initialize()
    
    # Test knowledge items with various characteristics
    test_items = [
        KnowledgeItem(
            id="",
            content="The COVID-19 pandemic began in late 2019 and was declared a pandemic by WHO in March 2020",
            fact_type=FactType.TEMPORAL,
            source_url="https://who.int/covid-pandemic-timeline",
            source_type=SourceType.GOVERNMENT_DOCUMENT,
            extraction_date=datetime.now() - timedelta(days=180),
            domain="health"
        ),
        KnowledgeItem(
            id="",
            content="OpenAI's GPT-4 was released in March 2023 and has 175 billion parameters",
            fact_type=FactType.NUMERICAL,
            source_url="https://openai.com/gpt-4-announcement",
            source_type=SourceType.WEBSITE,
            extraction_date=datetime.now() - timedelta(days=60),
            domain="technology"
        ),
        KnowledgeItem(
            id="",
            content="The human brain contains approximately 86 billion neurons",
            fact_type=FactType.NUMERICAL,
            source_url="https://nature.com/articles/neuron-count-study",
            source_type=SourceType.ACADEMIC_PAPER,
            extraction_date=datetime.now() - timedelta(days=400),
            domain="neuroscience"
        ),
        KnowledgeItem(
            id="",
            content="Quantum computers will solve all computational problems instantly",
            fact_type=FactType.CATEGORICAL,
            source_url="https://tech-blog.example.com/quantum-hype",
            source_type=SourceType.WEBSITE,
            extraction_date=datetime.now() - timedelta(days=900),
            domain="technology"
        )
    ]
    
    print("Testing complete validation system:")
    
    # Validate items
    reports = await system.validate_knowledge_batch(test_items)
    
    print(f"\nValidation Results:")
    
    for i, (item, report) in enumerate(zip(test_items, reports), 1):
        print(f"\n--- Item {i} ---")
        print(f"Content: {item.content[:80]}...")
        print(f"Source: {item.source_url}")
        print(f"Domain: {item.domain}")
        
        print(f"\nOverall Assessment:")
        print(f"  Result: {report.overall_result.value}")
        print(f"  Confidence: {report.overall_confidence:.3f}")
        print(f"  Validation Time: {report.validation_duration:.3f}s")
        
        print(f"\nValidation Checks Performed: {len(report.validation_checks)}")
        for check in report.validation_checks:
            print(f"  - {check.validation_method.value}: {check.result.value} (confidence: {check.confidence:.3f})")
        
        if report.issues_found:
            print(f"\nIssues Found:")
            for issue in report.issues_found:
                print(f"  - {issue}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations[:3]:  # Show first 3
                print(f"  - {rec}")
    
    # Show system statistics
    stats = system.get_statistics()
    
    print(f"\nSystem Statistics:")
    validation_stats = stats['validation_statistics']
    print(f"  Items Validated: {validation_stats['items_validated']}")
    print(f"  Validation Checks: {validation_stats['validation_checks_performed']}")
    print(f"  Valid Items: {validation_stats['valid_items']}")
    print(f"  Invalid Items: {validation_stats['invalid_items']}")
    print(f"  Uncertain Items: {validation_stats['uncertain_items']}")
    
    performance = stats['performance_metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Average Validation Time: {performance['average_validation_time']:.3f}s")
    print(f"  Validation Throughput: {performance['validation_throughput']:.2f} items/second")

async def demo_validation_scenarios():
    """Demo: Various validation scenarios"""
    print("\nDEMO 5: VALIDATION SCENARIOS")
    print("=" * 50)
    
    system = KnowledgeValidationSystem()
    await system.initialize()
    
    # Different validation scenarios
    scenarios = [
        {
            'name': 'High Quality Academic Source',
            'item': KnowledgeItem(
                id="",
                content="The speed of light in vacuum is approximately 299,792,458 meters per second",
                fact_type=FactType.NUMERICAL,
                source_url="https://physics.nist.gov/constants",
                source_type=SourceType.GOVERNMENT_DOCUMENT,
                extraction_date=datetime.now() - timedelta(days=30),
                domain="physics"
            )
        },
        {
            'name': 'Questionable Blog Source',
            'item': KnowledgeItem(
                id="",
                content="Drinking 10 cups of coffee daily increases life expectancy by 50 years",
                fact_type=FactType.NUMERICAL,
                source_url="https://random-health-blog.com/coffee-miracle",
                source_type=SourceType.WEBSITE,
                extraction_date=datetime.now() - timedelta(days=1200),
                domain="health"
            )
        },
        {
            'name': 'Outdated Information',
            'item': KnowledgeItem(
                id="",
                content="The tallest building in the world is the Petronas Towers at 452 meters",
                fact_type=FactType.NUMERICAL,
                source_url="https://architecture-guide.com/tallest-buildings",
                source_type=SourceType.WEBSITE,
                extraction_date=datetime.now() - timedelta(days=2000),
                domain="architecture"
            )
        },
        {
            'name': 'Recent Government Data',
            'item': KnowledgeItem(
                id="",
                content="The US unemployment rate was 3.7% in September 2023",
                fact_type=FactType.NUMERICAL,
                source_url="https://bls.gov/employment-situation",
                source_type=SourceType.GOVERNMENT_DOCUMENT,
                extraction_date=datetime.now() - timedelta(days=15),
                domain="economics"
            )
        },
        {
            'name': 'Contradictory Claim',
            'item': KnowledgeItem(
                id="",
                content="Albert Einstein was born in 1879 and died in 1850",  # Impossible dates
                fact_type=FactType.BIOGRAPHICAL,
                source_url="https://biography-errors.com/einstein",
                source_type=SourceType.WEBSITE,
                extraction_date=datetime.now() - timedelta(days=300),
                domain="biography"
            )
        }
    ]
    
    print("Testing various validation scenarios:")
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        
        item = scenario['item']
        print(f"Content: {item.content}")
        print(f"Source: {item.source_url}")
        print(f"Type: {item.source_type.value}")
        print(f"Age: {(datetime.now() - item.extraction_date).days} days")
        print(f"Domain: {item.domain}")
        
        # Validate the item
        report = await system.validate_knowledge_item(item)
        
        print(f"\nValidation Report:")
        print(f"  Overall Result: {report.overall_result.value}")
        print(f"  Overall Confidence: {report.overall_confidence:.3f}")
        
        print(f"\nDetailed Checks:")
        for check in report.validation_checks:
            print(f"  {check.validation_method.value}:")
            print(f"    Result: {check.result.value}")
            print(f"    Confidence: {check.confidence:.3f}")
            print(f"    Evidence: {'; '.join(check.evidence[:2])}")  # First 2 pieces of evidence
        
        # Interpretation
        print(f"\nInterpretation:")
        if report.overall_result == ValidationResult.VALID:
            print("  ✅ This information appears to be reliable and trustworthy")
        elif report.overall_result == ValidationResult.INVALID:
            print("  ❌ This information has significant reliability issues")
        elif report.overall_result == ValidationResult.CONFLICTING:
            print("  ⚠️  This information conflicts with other sources")
        else:
            print("  ❓ The reliability of this information is uncertain")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations[:2]:  # Show first 2
                print(f"  • {rec}")

async def main():
    """
    Demonstrate Knowledge Validation Systems for ensuring accuracy and consistency
    
    WHAT YOU'LL LEARN:
    ================
    1. How to validate information source authority and credibility
    2. How to perform automated fact-checking against reference sources
    3. How to detect consistency issues within knowledge bases
    4. How to aggregate validation results and generate confidence scores
    5. How to build comprehensive validation pipelines
    6. How to provide actionable recommendations for information quality
    
    REAL WORLD APPLICATIONS:
    =======================
    - Medical information systems requiring high accuracy
    - Legal research platforms ensuring factual correctness
    - Financial data systems preventing misinformation
    - News and journalism fact-checking systems
    - Educational content quality assurance
    - Enterprise knowledge management systems
    """
    
    print("KNOWLEDGE VALIDATION SYSTEMS DEMONSTRATION")
    print("Ensuring accuracy, consistency, and reliability of information!")
    
    await demo_source_authority_validation()
    await demo_fact_checking_validation()
    await demo_consistency_validation()
    await demo_complete_validation_system()
    await demo_validation_scenarios()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Source authority validation assesses information credibility")
    print("✓ Fact checking validates claims against authoritative sources")
    print("✓ Consistency validation detects contradictions in knowledge")
    print("✓ Comprehensive systems provide reliable quality assessment")
    print("✓ Validation scenarios demonstrate real-world applicability")
    print("✓ Automated validation enables scalable quality assurance")
    print("\nTHE POWER OF VALIDATION SYSTEMS:")
    print("- Builds trust in AI systems through verified information")
    print("- Prevents the spread of misinformation and errors")
    print("- Enables confident decision-making in critical domains")
    print("- Provides competitive advantage through superior data quality")

if __name__ == "__main__":
    asyncio.run(main())
