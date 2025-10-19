#!/usr/bin/env python3
"""
Knowledge Lifecycle Management: Complete Lifecycle from Creation to Retirement
============================================================================

WHAT IS THE PROBLEM?
==================
Knowledge systems lack proper lifecycle management:
- Information is created but never updated as it becomes outdated
- Systems accumulate stale data that reduces overall quality and reliability
- No clear processes for validating, versioning, or retiring knowledge
- Knowledge conflicts arise when multiple versions exist without coordination
- Organizations lose track of knowledge provenance and quality over time
- Critical knowledge gets lost when key personnel leave without proper documentation
- Systems become slow and unreliable due to accumulation of irrelevant information

Example: Medical Knowledge Decay
UNMANAGED KNOWLEDGE LIFECYCLE (Traditional):
- Medical database contains treatment guidelines from 1990s alongside current protocols
- Doctors receive conflicting recommendations from outdated and current sources
- No process to retire superseded medical knowledge or update existing information
- Drug interaction data becomes incomplete as new medications are introduced
- Critical updates to treatment protocols don't propagate to all relevant systems
- Result: Patient safety risks, ineffective treatments, physician confusion

REAL WORLD EXAMPLE:
=================
How does Wikipedia manage knowledge lifecycle?

WIKIPEDIA'S KNOWLEDGE LIFECYCLE:
1. CREATION: Anyone can create new articles with proper sourcing
2. VALIDATION: Community editors review and verify information quality
3. VERSIONING: Complete edit history tracks all changes over time
4. UPDATING: Regular maintenance to keep information current and accurate
5. QUALITY CONTROL: Flagging systems identify articles needing improvement
6. RETIREMENT: Deletion processes remove inappropriate or outdated content
7. GOVERNANCE: Editorial policies and admin oversight ensure quality standards

BENEFITS OF KNOWLEDGE LIFECYCLE MANAGEMENT:
- Ensures information remains accurate, current, and reliable over time
- Prevents knowledge decay and accumulation of outdated information
- Provides clear audit trails for knowledge changes and decision-making
- Enables systematic quality improvement and knowledge optimization
- Reduces storage costs and system complexity through intelligent archiving
- Maintains institutional knowledge even as personnel change
- Supports compliance with regulatory requirements for data retention

THE LIFECYCLE ADVANTAGE:
======================
UNMANAGED: Create → Accumulate → Decay → Unreliable
MANAGED: Create → Validate → Update → Optimize → Retire → Reliable

KNOWLEDGE LIFECYCLE STAGES:
==========================
1. CREATION: Initial knowledge capture with proper metadata and provenance
2. VALIDATION: Quality assessment, fact-checking, and source verification
3. PUBLICATION: Making knowledge available to authorized users and systems
4. MAINTENANCE: Regular updates, corrections, and quality improvements
5. VERSIONING: Managing multiple versions and tracking changes over time
6. INTEGRATION: Linking and connecting knowledge across different domains
7. OPTIMIZATION: Performance tuning and redundancy elimination
8. ARCHIVAL: Moving infrequently accessed knowledge to long-term storage
9. RETIREMENT: Safely removing obsolete or incorrect knowledge

WHY THIS IS REVOLUTIONARY:
========================
- Transforms static knowledge repositories into dynamic, self-improving systems
- Ensures AI systems maintain high quality knowledge over extended periods
- Critical for enterprise knowledge management and institutional memory
- Enables compliance with regulatory requirements and audit processes
- Creates sustainable knowledge ecosystems that improve rather than degrade
- Provides foundation for trustworthy, long-lived AI systems
"""

import asyncio
import time
import json
import uuid
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pickle
import gzip
import shutil
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class KnowledgeState(Enum):
    """Lifecycle states of knowledge"""
    DRAFT = "draft"
    VALIDATION = "validation"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    RETIRED = "retired"

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    UNASSESSED = "unassessed"

class ChangeType(Enum):
    """Types of changes to knowledge"""
    CREATION = "creation"
    UPDATE = "update"
    CORRECTION = "correction"
    ENHANCEMENT = "enhancement"
    VALIDATION = "validation"
    DEPRECATION = "deprecation"
    ARCHIVAL = "archival"
    RETIREMENT = "retirement"

class AccessPattern(Enum):
    """Knowledge access patterns"""
    FREQUENT = "frequent"       # Accessed multiple times per day
    REGULAR = "regular"         # Accessed multiple times per week
    OCCASIONAL = "occasional"   # Accessed multiple times per month
    RARE = "rare"              # Accessed a few times per year
    DORMANT = "dormant"        # Not accessed in over a year

@dataclass
class KnowledgeMetadata:
    """Metadata for knowledge items"""
    
    # Identity
    id: str
    title: str
    description: str = ""
    
    # Lifecycle
    state: KnowledgeState = KnowledgeState.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Quality
    quality_level: QualityLevel = QualityLevel.UNASSESSED
    quality_score: float = 0.0
    
    # Provenance
    creator_id: str = ""
    source: str = ""
    authority: str = ""
    
    # Access
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    access_pattern: AccessPattern = AccessPattern.DORMANT
    
    # Relationships
    depends_on: Set[str] = field(default_factory=set)
    referenced_by: Set[str] = field(default_factory=set)
    related_to: Set[str] = field(default_factory=set)
    
    # Lifecycle management
    expiration_date: Optional[datetime] = None
    review_date: Optional[datetime] = None
    retention_policy: str = ""
    
    # Tags and classification
    tags: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class KnowledgeVersion:
    """Represents a version of knowledge"""
    
    version_id: str
    knowledge_id: str
    version_number: str
    
    # Content
    content: Any
    content_hash: str = ""
    
    # Version metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    change_type: ChangeType = ChangeType.UPDATE
    change_description: str = ""
    
    # Validation
    is_validated: bool = False
    validated_by: str = ""
    validation_notes: str = ""
    
    # Size and performance
    content_size: int = 0
    
    def __post_init__(self):
        if not self.version_id:
            self.version_id = str(uuid.uuid4())
        
        if not self.content_hash and self.content:
            self.content_hash = self._calculate_content_hash()
        
        if not self.content_size and self.content:
            self.content_size = self._calculate_content_size()
    
    def _calculate_content_hash(self) -> str:
        """Calculate hash of content for change detection"""
        content_str = json.dumps(self.content, sort_keys=True) if self.content else ""
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _calculate_content_size(self) -> int:
        """Calculate size of content in bytes"""
        if self.content is None:
            return 0
        
        try:
            content_bytes = pickle.dumps(self.content)
            return len(content_bytes)
        except:
            # Fallback to string representation
            return len(str(self.content).encode('utf-8'))

@dataclass
class QualityAssessment:
    """Quality assessment for knowledge"""
    
    id: str
    knowledge_id: str
    assessor_id: str
    
    # Assessment results
    quality_level: QualityLevel
    quality_score: float  # 0.0 to 1.0
    
    # Detailed metrics
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    currency_score: float = 0.0
    reliability_score: float = 0.0
    
    # Assessment details
    assessment_date: datetime = field(default_factory=datetime.now)
    assessment_method: str = ""
    notes: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    requires_update: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class LifecycleAction:
    """Represents an action in the knowledge lifecycle"""
    
    id: str
    knowledge_id: str
    action_type: ChangeType
    
    # Action details
    performed_by: str
    performed_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    # Context
    previous_state: Optional[KnowledgeState] = None
    new_state: Optional[KnowledgeState] = None
    
    # Impact
    affected_dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class QualityAssessor:
    """Assesses knowledge quality"""
    
    def __init__(self):
        self.assessment_methods = {
            'automated': self._automated_assessment,
            'peer_review': self._peer_review_assessment,
            'expert_review': self._expert_review_assessment,
            'crowd_sourced': self._crowd_sourced_assessment
        }
        
        self.logger = logging.getLogger("QualityAssessor")
    
    async def assess_quality(self, knowledge_item: Dict[str, Any], 
                           metadata: KnowledgeMetadata,
                           method: str = 'automated') -> QualityAssessment:
        """Assess quality of knowledge item"""
        
        if method not in self.assessment_methods:
            raise ValueError(f"Unknown assessment method: {method}")
        
        assessment_func = self.assessment_methods[method]
        return await assessment_func(knowledge_item, metadata)
    
    async def _automated_assessment(self, knowledge_item: Dict[str, Any],
                                   metadata: KnowledgeMetadata) -> QualityAssessment:
        """Perform automated quality assessment"""
        
        # Accuracy assessment (simplified)
        accuracy_score = self._assess_accuracy(knowledge_item, metadata)
        
        # Completeness assessment
        completeness_score = self._assess_completeness(knowledge_item, metadata)
        
        # Currency assessment
        currency_score = self._assess_currency(metadata)
        
        # Reliability assessment
        reliability_score = self._assess_reliability(metadata)
        
        # Overall quality score
        weights = {'accuracy': 0.3, 'completeness': 0.3, 'currency': 0.2, 'reliability': 0.2}
        
        quality_score = (
            accuracy_score * weights['accuracy'] +
            completeness_score * weights['completeness'] +
            currency_score * weights['currency'] +
            reliability_score * weights['reliability']
        )
        
        # Determine quality level
        if quality_score >= 0.9:
            quality_level = QualityLevel.EXCELLENT
        elif quality_score >= 0.7:
            quality_level = QualityLevel.GOOD
        elif quality_score >= 0.5:
            quality_level = QualityLevel.ADEQUATE
        else:
            quality_level = QualityLevel.POOR
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy_score, completeness_score, currency_score, reliability_score
        )
        
        assessment = QualityAssessment(
            id="",
            knowledge_id=metadata.id,
            assessor_id="automated_system",
            quality_level=quality_level,
            quality_score=quality_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            currency_score=currency_score,
            reliability_score=reliability_score,
            assessment_method=method,
            recommendations=recommendations,
            requires_update=quality_score < 0.7
        )
        
        return assessment
    
    def _assess_accuracy(self, knowledge_item: Dict[str, Any], 
                        metadata: KnowledgeMetadata) -> float:
        """Assess accuracy of knowledge"""
        
        accuracy_indicators = []
        
        # Source authority
        if metadata.authority:
            if metadata.authority in ['expert', 'authoritative_source', 'peer_reviewed']:
                accuracy_indicators.append(0.9)
            elif metadata.authority in ['reliable_source', 'verified']:
                accuracy_indicators.append(0.7)
            else:
                accuracy_indicators.append(0.5)
        else:
            accuracy_indicators.append(0.3)
        
        # Cross-references
        if len(metadata.referenced_by) > 5:
            accuracy_indicators.append(0.8)
        elif len(metadata.referenced_by) > 1:
            accuracy_indicators.append(0.6)
        else:
            accuracy_indicators.append(0.4)
        
        # Content structure
        if isinstance(knowledge_item, dict):
            if 'sources' in knowledge_item and knowledge_item['sources']:
                accuracy_indicators.append(0.8)
            else:
                accuracy_indicators.append(0.5)
        
        return np.mean(accuracy_indicators) if accuracy_indicators else 0.5
    
    def _assess_completeness(self, knowledge_item: Dict[str, Any],
                           metadata: KnowledgeMetadata) -> float:
        """Assess completeness of knowledge"""
        
        completeness_indicators = []
        
        # Metadata completeness
        metadata_fields = [
            metadata.title, metadata.description, metadata.creator_id,
            metadata.source, metadata.tags, metadata.categories
        ]
        
        filled_fields = sum(1 for field in metadata_fields if field)
        metadata_completeness = filled_fields / len(metadata_fields)
        completeness_indicators.append(metadata_completeness)
        
        # Content structure completeness
        if isinstance(knowledge_item, dict):
            expected_keys = ['content', 'summary', 'keywords']
            present_keys = sum(1 for key in expected_keys if key in knowledge_item)
            content_completeness = present_keys / len(expected_keys)
            completeness_indicators.append(content_completeness)
        
        # Relationship completeness
        total_relationships = len(metadata.depends_on) + len(metadata.related_to)
        if total_relationships > 3:
            completeness_indicators.append(0.8)
        elif total_relationships > 0:
            completeness_indicators.append(0.6)
        else:
            completeness_indicators.append(0.4)
        
        return np.mean(completeness_indicators) if completeness_indicators else 0.5
    
    def _assess_currency(self, metadata: KnowledgeMetadata) -> float:
        """Assess how current/fresh the knowledge is"""
        
        now = datetime.now()
        
        # Time since last update
        time_since_update = now - metadata.updated_at
        
        if time_since_update < timedelta(days=30):
            update_score = 1.0
        elif time_since_update < timedelta(days=90):
            update_score = 0.8
        elif time_since_update < timedelta(days=365):
            update_score = 0.6
        elif time_since_update < timedelta(days=730):
            update_score = 0.4
        else:
            update_score = 0.2
        
        # Review schedule adherence
        if metadata.review_date:
            if now <= metadata.review_date:
                review_score = 1.0
            elif now <= metadata.review_date + timedelta(days=30):
                review_score = 0.7
            else:
                review_score = 0.3
        else:
            review_score = 0.5  # No review schedule
        
        return (update_score + review_score) / 2
    
    def _assess_reliability(self, metadata: KnowledgeMetadata) -> float:
        """Assess reliability of knowledge"""
        
        reliability_indicators = []
        
        # Access pattern indicates reliability
        if metadata.access_pattern == AccessPattern.FREQUENT:
            reliability_indicators.append(0.9)
        elif metadata.access_pattern == AccessPattern.REGULAR:
            reliability_indicators.append(0.7)
        elif metadata.access_pattern == AccessPattern.OCCASIONAL:
            reliability_indicators.append(0.6)
        else:
            reliability_indicators.append(0.4)
        
        # Creator reliability (simplified)
        if metadata.creator_id:
            if 'expert' in metadata.creator_id.lower():
                reliability_indicators.append(0.9)
            elif 'admin' in metadata.creator_id.lower():
                reliability_indicators.append(0.7)
            else:
                reliability_indicators.append(0.6)
        else:
            reliability_indicators.append(0.3)
        
        # Source reliability
        if metadata.source:
            if any(term in metadata.source.lower() for term in ['peer-reviewed', 'academic', 'official']):
                reliability_indicators.append(0.9)
            elif any(term in metadata.source.lower() for term in ['news', 'report', 'study']):
                reliability_indicators.append(0.7)
            else:
                reliability_indicators.append(0.5)
        else:
            reliability_indicators.append(0.3)
        
        return np.mean(reliability_indicators) if reliability_indicators else 0.5
    
    def _generate_recommendations(self, accuracy: float, completeness: float,
                                currency: float, reliability: float) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        if accuracy < 0.7:
            recommendations.append("Verify accuracy with authoritative sources")
            recommendations.append("Add source citations and references")
        
        if completeness < 0.7:
            recommendations.append("Complete missing metadata fields")
            recommendations.append("Add comprehensive content description")
            recommendations.append("Establish relationships with related knowledge")
        
        if currency < 0.7:
            recommendations.append("Update content with latest information")
            recommendations.append("Establish regular review schedule")
        
        if reliability < 0.7:
            recommendations.append("Obtain validation from domain experts")
            recommendations.append("Improve source documentation")
        
        return recommendations
    
    async def _peer_review_assessment(self, knowledge_item: Dict[str, Any],
                                     metadata: KnowledgeMetadata) -> QualityAssessment:
        """Simulate peer review assessment"""
        
        # Simulate peer review process
        base_assessment = await self._automated_assessment(knowledge_item, metadata)
        
        # Peer review typically increases reliability
        base_assessment.reliability_score = min(1.0, base_assessment.reliability_score + 0.2)
        base_assessment.accuracy_score = min(1.0, base_assessment.accuracy_score + 0.1)
        
        # Recalculate overall score
        weights = {'accuracy': 0.3, 'completeness': 0.3, 'currency': 0.2, 'reliability': 0.2}
        
        base_assessment.quality_score = (
            base_assessment.accuracy_score * weights['accuracy'] +
            base_assessment.completeness_score * weights['completeness'] +
            base_assessment.currency_score * weights['currency'] +
            base_assessment.reliability_score * weights['reliability']
        )
        
        base_assessment.assessor_id = "peer_reviewer"
        base_assessment.assessment_method = "peer_review"
        base_assessment.notes = "Peer-reviewed by domain experts"
        
        return base_assessment
    
    async def _expert_review_assessment(self, knowledge_item: Dict[str, Any],
                                       metadata: KnowledgeMetadata) -> QualityAssessment:
        """Simulate expert review assessment"""
        
        # Expert review provides highest quality assessment
        assessment = QualityAssessment(
            id="",
            knowledge_id=metadata.id,
            assessor_id="domain_expert",
            quality_level=QualityLevel.EXCELLENT,
            quality_score=0.95,
            accuracy_score=0.95,
            completeness_score=0.90,
            currency_score=0.85,
            reliability_score=0.98,
            assessment_method="expert_review",
            notes="Validated by recognized domain expert",
            requires_update=False
        )
        
        return assessment
    
    async def _crowd_sourced_assessment(self, knowledge_item: Dict[str, Any],
                                       metadata: KnowledgeMetadata) -> QualityAssessment:
        """Simulate crowd-sourced assessment"""
        
        # Simulate multiple crowd assessments
        crowd_scores = []
        
        for _ in range(10):  # Simulate 10 crowd assessors
            individual_score = np.random.beta(4, 2)  # Skewed towards higher scores
            crowd_scores.append(individual_score)
        
        avg_score = np.mean(crowd_scores)
        confidence = 1.0 - np.std(crowd_scores)  # Lower std = higher confidence
        
        # Determine quality level
        if avg_score >= 0.8 and confidence >= 0.8:
            quality_level = QualityLevel.GOOD
        elif avg_score >= 0.6:
            quality_level = QualityLevel.ADEQUATE
        else:
            quality_level = QualityLevel.POOR
        
        assessment = QualityAssessment(
            id="",
            knowledge_id=metadata.id,
            assessor_id="crowd_sourced",
            quality_level=quality_level,
            quality_score=avg_score,
            accuracy_score=avg_score,
            completeness_score=avg_score * 0.9,
            currency_score=avg_score * 0.8,
            reliability_score=confidence,
            assessment_method="crowd_sourced",
            notes=f"Assessed by crowd with confidence {confidence:.2f}",
            requires_update=avg_score < 0.7
        )
        
        return assessment

class VersionManager:
    """Manages knowledge versions"""
    
    def __init__(self):
        self.versions: Dict[str, List[KnowledgeVersion]] = defaultdict(list)
        self.logger = logging.getLogger("VersionManager")
    
    def create_version(self, knowledge_id: str, content: Any, 
                      change_type: ChangeType, change_description: str,
                      created_by: str = "") -> KnowledgeVersion:
        """Create a new version of knowledge"""
        
        existing_versions = self.versions[knowledge_id]
        
        # Generate version number
        if not existing_versions:
            version_number = "1.0.0"
        else:
            latest_version = existing_versions[-1]
            version_parts = latest_version.version_number.split('.')
            
            if change_type in [ChangeType.CREATION, ChangeType.ENHANCEMENT]:
                # Major version
                version_number = f"{int(version_parts[0]) + 1}.0.0"
            elif change_type in [ChangeType.UPDATE, ChangeType.VALIDATION]:
                # Minor version
                version_number = f"{version_parts[0]}.{int(version_parts[1]) + 1}.0"
            else:
                # Patch version
                version_number = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
        
        # Create new version
        version = KnowledgeVersion(
            version_id="",
            knowledge_id=knowledge_id,
            version_number=version_number,
            content=content,
            created_by=created_by,
            change_type=change_type,
            change_description=change_description
        )
        
        # Store version
        self.versions[knowledge_id].append(version)
        
        self.logger.debug(f"Created version {version_number} for knowledge {knowledge_id}")
        
        return version
    
    def get_current_version(self, knowledge_id: str) -> Optional[KnowledgeVersion]:
        """Get the current (latest) version of knowledge"""
        
        if knowledge_id not in self.versions or not self.versions[knowledge_id]:
            return None
        
        return self.versions[knowledge_id][-1]
    
    def get_version(self, knowledge_id: str, version_number: str) -> Optional[KnowledgeVersion]:
        """Get a specific version of knowledge"""
        
        versions = self.versions.get(knowledge_id, [])
        
        for version in versions:
            if version.version_number == version_number:
                return version
        
        return None
    
    def get_version_history(self, knowledge_id: str) -> List[KnowledgeVersion]:
        """Get complete version history for knowledge"""
        
        return self.versions.get(knowledge_id, [])
    
    def compare_versions(self, knowledge_id: str, version1: str, 
                        version2: str) -> Dict[str, Any]:
        """Compare two versions of knowledge"""
        
        v1 = self.get_version(knowledge_id, version1)
        v2 = self.get_version(knowledge_id, version2)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': {
                'version_number': v1.version_number,
                'created_at': v1.created_at,
                'created_by': v1.created_by,
                'change_type': v1.change_type.value,
                'content_hash': v1.content_hash,
                'content_size': v1.content_size
            },
            'version2': {
                'version_number': v2.version_number,
                'created_at': v2.created_at,
                'created_by': v2.created_by,
                'change_type': v2.change_type.value,
                'content_hash': v2.content_hash,
                'content_size': v2.content_size
            },
            'differences': {
                'content_changed': v1.content_hash != v2.content_hash,
                'size_change': v2.content_size - v1.content_size,
                'time_difference': v2.created_at - v1.created_at
            }
        }
        
        return comparison
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version management statistics"""
        
        total_knowledge_items = len(self.versions)
        total_versions = sum(len(versions) for versions in self.versions.values())
        
        # Average versions per knowledge item
        avg_versions = total_versions / total_knowledge_items if total_knowledge_items > 0 else 0
        
        # Change type distribution
        change_types = defaultdict(int)
        for versions in self.versions.values():
            for version in versions:
                change_types[version.change_type.value] += 1
        
        return {
            'total_knowledge_items': total_knowledge_items,
            'total_versions': total_versions,
            'average_versions_per_item': avg_versions,
            'change_type_distribution': dict(change_types)
        }

class AccessAnalyzer:
    """Analyzes knowledge access patterns"""
    
    def __init__(self):
        self.access_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("AccessAnalyzer")
    
    def log_access(self, knowledge_id: str, user_id: str, 
                  access_type: str = "read") -> None:
        """Log knowledge access"""
        
        access_record = {
            'knowledge_id': knowledge_id,
            'user_id': user_id,
            'access_type': access_type,
            'timestamp': datetime.now()
        }
        
        self.access_log.append(access_record)
    
    def analyze_access_pattern(self, knowledge_id: str,
                             analysis_window: timedelta = timedelta(days=90)) -> AccessPattern:
        """Analyze access pattern for knowledge item"""
        
        cutoff_time = datetime.now() - analysis_window
        
        # Filter relevant access records
        relevant_accesses = [
            record for record in self.access_log
            if (record['knowledge_id'] == knowledge_id and 
                record['timestamp'] >= cutoff_time)
        ]
        
        if not relevant_accesses:
            return AccessPattern.DORMANT
        
        # Calculate access frequency
        access_count = len(relevant_accesses)
        days_in_window = analysis_window.days
        
        accesses_per_day = access_count / days_in_window
        
        # Classify access pattern
        if accesses_per_day >= 1.0:
            return AccessPattern.FREQUENT
        elif accesses_per_day >= 0.2:  # More than once every 5 days
            return AccessPattern.REGULAR
        elif accesses_per_day >= 0.033:  # More than once per month
            return AccessPattern.OCCASIONAL
        elif access_count > 0:
            return AccessPattern.RARE
        else:
            return AccessPattern.DORMANT
    
    def get_access_statistics(self, knowledge_id: str) -> Dict[str, Any]:
        """Get detailed access statistics"""
        
        knowledge_accesses = [
            record for record in self.access_log
            if record['knowledge_id'] == knowledge_id
        ]
        
        if not knowledge_accesses:
            return {
                'total_accesses': 0,
                'unique_users': 0,
                'first_access': None,
                'last_access': None,
                'access_pattern': AccessPattern.DORMANT.value
            }
        
        # Calculate statistics
        total_accesses = len(knowledge_accesses)
        unique_users = len(set(record['user_id'] for record in knowledge_accesses))
        
        timestamps = [record['timestamp'] for record in knowledge_accesses]
        first_access = min(timestamps)
        last_access = max(timestamps)
        
        access_pattern = self.analyze_access_pattern(knowledge_id)
        
        return {
            'total_accesses': total_accesses,
            'unique_users': unique_users,
            'first_access': first_access,
            'last_access': last_access,
            'access_pattern': access_pattern.value,
            'days_since_last_access': (datetime.now() - last_access).days
        }
    
    def identify_popular_knowledge(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify most popular knowledge items"""
        
        # Count accesses per knowledge item
        access_counts = defaultdict(int)
        
        for record in self.access_log:
            access_counts[record['knowledge_id']] += 1
        
        # Sort by access count
        sorted_items = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
        
        popular_items = []
        for knowledge_id, access_count in sorted_items[:top_n]:
            stats = self.get_access_statistics(knowledge_id)
            popular_items.append({
                'knowledge_id': knowledge_id,
                'access_count': access_count,
                'unique_users': stats['unique_users'],
                'access_pattern': stats['access_pattern']
            })
        
        return popular_items
    
    def identify_dormant_knowledge(self, dormant_threshold: timedelta = timedelta(days=365)) -> List[str]:
        """Identify knowledge items that haven't been accessed recently"""
        
        cutoff_time = datetime.now() - dormant_threshold
        
        # Get all knowledge IDs that have been accessed
        accessed_knowledge = set()
        recent_accesses = set()
        
        for record in self.access_log:
            accessed_knowledge.add(record['knowledge_id'])
            
            if record['timestamp'] >= cutoff_time:
                recent_accesses.add(record['knowledge_id'])
        
        # Knowledge that was accessed before but not recently
        dormant_knowledge = accessed_knowledge - recent_accesses
        
        return list(dormant_knowledge)

class LifecycleManager:
    """Complete knowledge lifecycle management"""
    
    def __init__(self):
        # Core components
        self.quality_assessor = QualityAssessor()
        self.version_manager = VersionManager()
        self.access_analyzer = AccessAnalyzer()
        
        # Knowledge storage
        self.knowledge_items: Dict[str, KnowledgeMetadata] = {}
        self.quality_assessments: Dict[str, List[QualityAssessment]] = defaultdict(list)
        self.lifecycle_actions: List[LifecycleAction] = []
        
        # Policies
        self.lifecycle_policies = {
            'auto_archive_threshold': timedelta(days=365),
            'auto_retire_threshold': timedelta(days=1095),
            'quality_review_interval': timedelta(days=180),
            'mandatory_update_threshold': timedelta(days=730)
        }
        
        # Statistics
        self.stats = {
            'knowledge_created': 0,
            'knowledge_updated': 0,
            'knowledge_archived': 0,
            'knowledge_retired': 0,
            'quality_assessments': 0
        }
        
        self.logger = logging.getLogger("LifecycleManager")
    
    async def initialize(self) -> None:
        """Initialize the lifecycle management system"""
        self.logger.info("Knowledge lifecycle management system initialized")
    
    async def create_knowledge(self, title: str, content: Any, creator_id: str,
                             source: str = "", tags: List[str] = None,
                             categories: List[str] = None) -> str:
        """Create new knowledge item"""
        
        if tags is None:
            tags = []
        if categories is None:
            categories = []
        
        # Create metadata
        metadata = KnowledgeMetadata(
            id="",
            title=title,
            creator_id=creator_id,
            source=source,
            tags=set(tags),
            categories=set(categories),
            state=KnowledgeState.DRAFT
        )
        
        # Store knowledge
        self.knowledge_items[metadata.id] = metadata
        
        # Create initial version
        version = self.version_manager.create_version(
            metadata.id, content, ChangeType.CREATION,
            "Initial creation", creator_id
        )
        
        # Log action
        action = LifecycleAction(
            id="",
            knowledge_id=metadata.id,
            action_type=ChangeType.CREATION,
            performed_by=creator_id,
            description=f"Created knowledge: {title}",
            new_state=KnowledgeState.DRAFT
        )
        self.lifecycle_actions.append(action)
        
        self.stats['knowledge_created'] += 1
        
        self.logger.info(f"Created knowledge: {metadata.id}")
        
        return metadata.id
    
    async def validate_knowledge(self, knowledge_id: str, validator_id: str,
                               assessment_method: str = 'automated') -> bool:
        """Validate knowledge quality"""
        
        if knowledge_id not in self.knowledge_items:
            return False
        
        metadata = self.knowledge_items[knowledge_id]
        current_version = self.version_manager.get_current_version(knowledge_id)
        
        if not current_version:
            return False
        
        # Perform quality assessment
        assessment = await self.quality_assessor.assess_quality(
            current_version.content, metadata, assessment_method
        )
        
        self.quality_assessments[knowledge_id].append(assessment)
        
        # Update metadata
        metadata.quality_level = assessment.quality_level
        metadata.quality_score = assessment.quality_score
        
        # Update version validation status
        current_version.is_validated = True
        current_version.validated_by = validator_id
        current_version.validation_notes = f"Quality score: {assessment.quality_score:.2f}"
        
        # Update state if validation successful
        if assessment.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
            await self._transition_state(knowledge_id, KnowledgeState.PUBLISHED, validator_id)
        
        # Log action
        action = LifecycleAction(
            id="",
            knowledge_id=knowledge_id,
            action_type=ChangeType.VALIDATION,
            performed_by=validator_id,
            description=f"Validated with quality level: {assessment.quality_level.value}"
        )
        self.lifecycle_actions.append(action)
        
        self.stats['quality_assessments'] += 1
        
        return True
    
    async def update_knowledge(self, knowledge_id: str, new_content: Any,
                             updater_id: str, change_description: str = "") -> bool:
        """Update existing knowledge"""
        
        if knowledge_id not in self.knowledge_items:
            return False
        
        metadata = self.knowledge_items[knowledge_id]
        
        # Create new version
        version = self.version_manager.create_version(
            knowledge_id, new_content, ChangeType.UPDATE,
            change_description, updater_id
        )
        
        # Update metadata
        metadata.updated_at = datetime.now()
        
        # Log action
        action = LifecycleAction(
            id="",
            knowledge_id=knowledge_id,
            action_type=ChangeType.UPDATE,
            performed_by=updater_id,
            description=change_description or "Knowledge updated"
        )
        self.lifecycle_actions.append(action)
        
        self.stats['knowledge_updated'] += 1
        
        return True
    
    async def access_knowledge(self, knowledge_id: str, user_id: str) -> Optional[Any]:
        """Access knowledge and update statistics"""
        
        if knowledge_id not in self.knowledge_items:
            return None
        
        metadata = self.knowledge_items[knowledge_id]
        current_version = self.version_manager.get_current_version(knowledge_id)
        
        if not current_version:
            return None
        
        # Update access statistics
        metadata.access_count += 1
        metadata.last_accessed = datetime.now()
        
        # Log access
        self.access_analyzer.log_access(knowledge_id, user_id)
        
        # Update access pattern
        metadata.access_pattern = self.access_analyzer.analyze_access_pattern(knowledge_id)
        
        return current_version.content
    
    async def archive_knowledge(self, knowledge_id: str, archiver_id: str,
                              reason: str = "") -> bool:
        """Archive knowledge item"""
        
        if knowledge_id not in self.knowledge_items:
            return False
        
        metadata = self.knowledge_items[knowledge_id]
        
        # Transition to archived state
        await self._transition_state(knowledge_id, KnowledgeState.ARCHIVED, archiver_id)
        
        # Log action
        action = LifecycleAction(
            id="",
            knowledge_id=knowledge_id,
            action_type=ChangeType.ARCHIVAL,
            performed_by=archiver_id,
            description=reason or "Knowledge archived",
            previous_state=metadata.state,
            new_state=KnowledgeState.ARCHIVED
        )
        self.lifecycle_actions.append(action)
        
        self.stats['knowledge_archived'] += 1
        
        return True
    
    async def retire_knowledge(self, knowledge_id: str, retirer_id: str,
                             reason: str = "") -> bool:
        """Retire knowledge item"""
        
        if knowledge_id not in self.knowledge_items:
            return False
        
        metadata = self.knowledge_items[knowledge_id]
        
        # Check dependencies
        if metadata.referenced_by:
            self.logger.warning(f"Cannot retire {knowledge_id}: still referenced by {metadata.referenced_by}")
            return False
        
        # Transition to retired state
        await self._transition_state(knowledge_id, KnowledgeState.RETIRED, retirer_id)
        
        # Log action
        action = LifecycleAction(
            id="",
            knowledge_id=knowledge_id,
            action_type=ChangeType.RETIREMENT,
            performed_by=retirer_id,
            description=reason or "Knowledge retired",
            previous_state=metadata.state,
            new_state=KnowledgeState.RETIRED
        )
        self.lifecycle_actions.append(action)
        
        self.stats['knowledge_retired'] += 1
        
        return True
    
    async def perform_maintenance(self) -> Dict[str, Any]:
        """Perform automatic maintenance tasks"""
        
        maintenance_results = {
            'items_reviewed': 0,
            'items_archived': 0,
            'items_flagged_for_update': 0,
            'quality_assessments_performed': 0
        }
        
        now = datetime.now()
        
        for knowledge_id, metadata in self.knowledge_items.items():
            
            # Skip already archived or retired items
            if metadata.state in [KnowledgeState.ARCHIVED, KnowledgeState.RETIRED]:
                continue
            
            maintenance_results['items_reviewed'] += 1
            
            # Check for auto-archival
            if (metadata.access_pattern == AccessPattern.DORMANT and
                metadata.last_accessed and
                now - metadata.last_accessed > self.lifecycle_policies['auto_archive_threshold']):
                
                await self.archive_knowledge(knowledge_id, "automated_system", 
                                           "Auto-archived due to inactivity")
                maintenance_results['items_archived'] += 1
                continue
            
            # Check for mandatory updates
            if (now - metadata.updated_at > self.lifecycle_policies['mandatory_update_threshold']):
                # Flag for update
                metadata.tags.add("needs_update")
                maintenance_results['items_flagged_for_update'] += 1
            
            # Check for quality review
            last_assessment = None
            if knowledge_id in self.quality_assessments:
                assessments = self.quality_assessments[knowledge_id]
                if assessments:
                    last_assessment = max(assessments, key=lambda a: a.assessment_date)
            
            if (not last_assessment or 
                now - last_assessment.assessment_date > self.lifecycle_policies['quality_review_interval']):
                
                await self.validate_knowledge(knowledge_id, "automated_system")
                maintenance_results['quality_assessments_performed'] += 1
        
        self.logger.info(f"Maintenance completed: {maintenance_results}")
        
        return maintenance_results
    
    async def _transition_state(self, knowledge_id: str, new_state: KnowledgeState,
                              performer_id: str) -> None:
        """Transition knowledge to new state"""
        
        metadata = self.knowledge_items[knowledge_id]
        old_state = metadata.state
        metadata.state = new_state
        
        self.logger.debug(f"Transitioned {knowledge_id}: {old_state.value} -> {new_state.value}")
    
    def get_lifecycle_report(self) -> Dict[str, Any]:
        """Generate comprehensive lifecycle report"""
        
        # State distribution
        state_distribution = defaultdict(int)
        quality_distribution = defaultdict(int)
        access_pattern_distribution = defaultdict(int)
        
        for metadata in self.knowledge_items.values():
            state_distribution[metadata.state.value] += 1
            quality_distribution[metadata.quality_level.value] += 1
            access_pattern_distribution[metadata.access_pattern.value] += 1
        
        # Recent activity
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_actions = [
            action for action in self.lifecycle_actions
            if action.performed_at >= recent_cutoff
        ]
        
        action_types = defaultdict(int)
        for action in recent_actions:
            action_types[action.action_type.value] += 1
        
        # Version statistics
        version_stats = self.version_manager.get_statistics()
        
        # Access statistics
        popular_knowledge = self.access_analyzer.identify_popular_knowledge(10)
        dormant_knowledge = self.access_analyzer.identify_dormant_knowledge()
        
        return {
            'overview': {
                'total_knowledge_items': len(self.knowledge_items),
                'state_distribution': dict(state_distribution),
                'quality_distribution': dict(quality_distribution),
                'access_pattern_distribution': dict(access_pattern_distribution)
            },
            'recent_activity': {
                'total_recent_actions': len(recent_actions),
                'action_type_distribution': dict(action_types)
            },
            'version_management': version_stats,
            'access_analysis': {
                'popular_knowledge': popular_knowledge,
                'dormant_knowledge_count': len(dormant_knowledge)
            },
            'system_statistics': self.stats,
            'maintenance_policies': {
                key: str(value) for key, value in self.lifecycle_policies.items()
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_knowledge_creation_and_validation():
    """Demo: Creating and validating knowledge"""
    print("\nDEMO 1: KNOWLEDGE CREATION AND VALIDATION")
    print("=" * 50)
    
    lifecycle_manager = LifecycleManager()
    await lifecycle_manager.initialize()
    
    # Create different types of knowledge
    print("Creating knowledge items:")
    
    knowledge_items = [
        {
            'title': 'Python Best Practices',
            'content': {
                'summary': 'Comprehensive guide to Python programming best practices',
                'guidelines': ['Use meaningful variable names', 'Follow PEP 8', 'Write docstrings'],
                'examples': ['def calculate_total(price, tax):', '    return price * (1 + tax)']
            },
            'creator': 'expert_developer',
            'source': 'programming_standards',
            'tags': ['python', 'programming', 'best-practices'],
            'categories': ['development', 'standards']
        },
        {
            'title': 'Database Security Guidelines',
            'content': {
                'summary': 'Security practices for database management',
                'requirements': ['Use strong passwords', 'Enable encryption', 'Regular backups'],
                'threats': ['SQL injection', 'Unauthorized access', 'Data leakage']
            },
            'creator': 'security_expert',
            'source': 'security_team',
            'tags': ['database', 'security', 'guidelines'],
            'categories': ['security', 'database']
        },
        {
            'title': 'Marketing Campaign Results Q3',
            'content': {
                'summary': 'Results from Q3 marketing campaigns',
                'metrics': {'conversion_rate': 0.15, 'roi': 2.3, 'engagement': 0.08},
                'insights': ['Social media campaigns performed best', 'Email had lowest ROI']
            },
            'creator': 'marketing_analyst',
            'source': 'marketing_department',
            'tags': ['marketing', 'q3', 'results'],
            'categories': ['marketing', 'analytics']
        }
    ]
    
    created_ids = []
    
    for item in knowledge_items:
        knowledge_id = await lifecycle_manager.create_knowledge(
            item['title'], item['content'], item['creator'],
            item['source'], item['tags'], item['categories']
        )
        created_ids.append(knowledge_id)
        
        print(f"  Created: {item['title']} (ID: {knowledge_id[:8]}...)")
    
    # Validate knowledge with different methods
    print(f"\nValidating knowledge with different methods:")
    
    validation_methods = ['automated', 'peer_review', 'expert_review']
    
    for i, knowledge_id in enumerate(created_ids):
        method = validation_methods[i % len(validation_methods)]
        validator = f"{method}_validator"
        
        success = await lifecycle_manager.validate_knowledge(knowledge_id, validator, method)
        
        if success:
            metadata = lifecycle_manager.knowledge_items[knowledge_id]
            print(f"  ✓ Validated {metadata.title}")
            print(f"    Method: {method}")
            print(f"    Quality: {metadata.quality_level.value} (score: {metadata.quality_score:.2f})")
        else:
            print(f"  ✗ Validation failed for {knowledge_id}")
    
    # Show quality assessments
    print(f"\nQuality assessment details:")
    
    for knowledge_id in created_ids:
        assessments = lifecycle_manager.quality_assessments.get(knowledge_id, [])
        
        if assessments:
            latest_assessment = assessments[-1]
            metadata = lifecycle_manager.knowledge_items[knowledge_id]
            
            print(f"\n{metadata.title}:")
            print(f"  Accuracy: {latest_assessment.accuracy_score:.2f}")
            print(f"  Completeness: {latest_assessment.completeness_score:.2f}")
            print(f"  Currency: {latest_assessment.currency_score:.2f}")
            print(f"  Reliability: {latest_assessment.reliability_score:.2f}")
            
            if latest_assessment.recommendations:
                print(f"  Recommendations:")
                for rec in latest_assessment.recommendations:
                    print(f"    - {rec}")

async def demo_version_management():
    """Demo: Managing knowledge versions"""
    print("\nDEMO 2: VERSION MANAGEMENT")
    print("=" * 50)
    
    lifecycle_manager = LifecycleManager()
    await lifecycle_manager.initialize()
    
    # Create initial knowledge
    knowledge_id = await lifecycle_manager.create_knowledge(
        "API Documentation",
        {
            'version': '1.0',
            'endpoints': ['/users', '/posts'],
            'authentication': 'API key required'
        },
        "api_team"
    )
    
    print(f"Created initial knowledge: {knowledge_id[:8]}...")
    
    # Show initial version
    current_version = lifecycle_manager.version_manager.get_current_version(knowledge_id)
    print(f"Initial version: {current_version.version_number}")
    
    # Update knowledge multiple times
    print(f"\nPerforming updates:")
    
    updates = [
        {
            'content': {
                'version': '1.1',
                'endpoints': ['/users', '/posts', '/comments'],
                'authentication': 'API key required',
                'rate_limits': '1000 requests/hour'
            },
            'description': 'Added comments endpoint and rate limiting'
        },
        {
            'content': {
                'version': '1.2',
                'endpoints': ['/users', '/posts', '/comments'],
                'authentication': 'Bearer token required',
                'rate_limits': '1000 requests/hour',
                'pagination': 'Cursor-based pagination'
            },
            'description': 'Updated authentication and added pagination'
        },
        {
            'content': {
                'version': '2.0',
                'endpoints': ['/v2/users', '/v2/posts', '/v2/comments', '/v2/analytics'],
                'authentication': 'OAuth 2.0',
                'rate_limits': '5000 requests/hour',
                'pagination': 'Cursor-based pagination',
                'versioning': 'URL versioning'
            },
            'description': 'Major version update with new API structure'
        }
    ]
    
    for i, update in enumerate(updates):
        success = await lifecycle_manager.update_knowledge(
            knowledge_id, update['content'], 
            f"developer_{i+1}", update['description']
        )
        
        if success:
            current_version = lifecycle_manager.version_manager.get_current_version(knowledge_id)
            print(f"  Update {i+1}: Version {current_version.version_number}")
            print(f"    Description: {update['description']}")
    
    # Show version history
    print(f"\nVersion history:")
    version_history = lifecycle_manager.version_manager.get_version_history(knowledge_id)
    
    for version in version_history:
        print(f"  {version.version_number} ({version.change_type.value})")
        print(f"    Created: {version.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"    By: {version.created_by}")
        print(f"    Size: {version.content_size} bytes")
        print(f"    Description: {version.change_description}")
        print()
    
    # Compare versions
    print(f"Comparing versions 1.0.0 and 2.0.0:")
    comparison = lifecycle_manager.version_manager.compare_versions(
        knowledge_id, "1.0.0", "2.0.0"
    )
    
    if 'error' not in comparison:
        print(f"  Content changed: {comparison['differences']['content_changed']}")
        print(f"  Size change: {comparison['differences']['size_change']} bytes")
        print(f"  Time difference: {comparison['differences']['time_difference']}")
    
    # Show version management statistics
    print(f"\nVersion management statistics:")
    stats = lifecycle_manager.version_manager.get_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {dict(value)}")
        else:
            print(f"  {key}: {value}")

async def demo_access_analysis():
    """Demo: Analyzing knowledge access patterns"""
    print("\nDEMO 3: ACCESS ANALYSIS")
    print("=" * 50)
    
    lifecycle_manager = LifecycleManager()
    await lifecycle_manager.initialize()
    
    # Create knowledge items
    knowledge_ids = []
    knowledge_titles = [
        "User Authentication Guide",
        "Database Schema Documentation", 
        "API Error Codes",
        "Deployment Procedures",
        "Security Policies"
    ]
    
    print("Creating knowledge items:")
    for title in knowledge_titles:
        knowledge_id = await lifecycle_manager.create_knowledge(
            title, f"Content for {title}", "content_creator"
        )
        knowledge_ids.append(knowledge_id)
        print(f"  Created: {title}")
    
    # Simulate different access patterns
    print(f"\nSimulating access patterns:")
    
    import random
    
    # Simulate 90 days of access
    base_time = datetime.now() - timedelta(days=90)
    
    access_patterns = {
        knowledge_ids[0]: 'frequent',    # Daily access
        knowledge_ids[1]: 'regular',     # Weekly access
        knowledge_ids[2]: 'occasional',  # Monthly access
        knowledge_ids[3]: 'rare',        # Quarterly access
        knowledge_ids[4]: 'dormant'      # No recent access
    }
    
    users = ['developer_1', 'developer_2', 'admin_1', 'manager_1', 'analyst_1']
    
    for i in range(90):  # 90 days
        current_day = base_time + timedelta(days=i)
        
        for knowledge_id, pattern in access_patterns.items():
            # Simulate access based on pattern
            access_probability = {
                'frequent': 0.8,   # 80% chance per day
                'regular': 0.15,   # 15% chance per day
                'occasional': 0.05, # 5% chance per day
                'rare': 0.01,      # 1% chance per day
                'dormant': 0.0     # No access
            }
            
            if random.random() < access_probability[pattern]:
                user = random.choice(users)
                
                # Simulate access (this updates access statistics)
                content = await lifecycle_manager.access_knowledge(knowledge_id, user)
                
                # Manually set timestamp for simulation
                lifecycle_manager.access_analyzer.access_log[-1]['timestamp'] = current_day
    
    # Analyze access patterns
    print(f"\nAccess pattern analysis:")
    
    for i, knowledge_id in enumerate(knowledge_ids):
        title = knowledge_titles[i]
        metadata = lifecycle_manager.knowledge_items[knowledge_id]
        
        # Update access pattern based on analysis
        metadata.access_pattern = lifecycle_manager.access_analyzer.analyze_access_pattern(knowledge_id)
        
        stats = lifecycle_manager.access_analyzer.get_access_statistics(knowledge_id)
        
        print(f"\n{title}:")
        print(f"  Access pattern: {metadata.access_pattern.value}")
        print(f"  Total accesses: {stats['total_accesses']}")
        print(f"  Unique users: {stats['unique_users']}")
        print(f"  Days since last access: {stats['days_since_last_access']}")
    
    # Identify popular and dormant knowledge
    print(f"\nPopular knowledge (top 3):")
    popular = lifecycle_manager.access_analyzer.identify_popular_knowledge(3)
    
    for item in popular:
        knowledge_title = next(
            title for title, kid in zip(knowledge_titles, knowledge_ids) 
            if kid == item['knowledge_id']
        )
        print(f"  {knowledge_title}: {item['access_count']} accesses ({item['access_pattern']})")
    
    print(f"\nDormant knowledge:")
    dormant = lifecycle_manager.access_analyzer.identify_dormant_knowledge(timedelta(days=60))
    
    for knowledge_id in dormant:
        knowledge_title = next(
            title for title, kid in zip(knowledge_titles, knowledge_ids) 
            if kid == knowledge_id
        )
        print(f"  {knowledge_title}")

async def demo_lifecycle_maintenance():
    """Demo: Automated lifecycle maintenance"""
    print("\nDEMO 4: LIFECYCLE MAINTENANCE")
    print("=" * 50)
    
    lifecycle_manager = LifecycleManager()
    await lifecycle_manager.initialize()
    
    # Adjust policies for demo
    lifecycle_manager.lifecycle_policies = {
        'auto_archive_threshold': timedelta(days=30),    # Archive after 30 days of inactivity
        'auto_retire_threshold': timedelta(days=90),     # Retire after 90 days
        'quality_review_interval': timedelta(days=45),   # Review quality every 45 days
        'mandatory_update_threshold': timedelta(days=60) # Flag for update after 60 days
    }
    
    print("Creating knowledge items with different ages:")
    
    # Create knowledge items with different timestamps
    knowledge_items = [
        {
            'title': 'Recent Documentation',
            'age_days': 5,
            'last_access_days': 1
        },
        {
            'title': 'Moderately Old Guide',
            'age_days': 50,
            'last_access_days': 10
        },
        {
            'title': 'Old Inactive Content',
            'age_days': 100,
            'last_access_days': 40
        },
        {
            'title': 'Very Old Dormant Item',
            'age_days': 200,
            'last_access_days': 120
        }
    ]
    
    knowledge_ids = []
    
    for item in knowledge_items:
        # Create knowledge
        knowledge_id = await lifecycle_manager.create_knowledge(
            item['title'], f"Content for {item['title']}", "creator"
        )
        knowledge_ids.append(knowledge_id)
        
        metadata = lifecycle_manager.knowledge_items[knowledge_id]
        
        # Simulate age
        created_time = datetime.now() - timedelta(days=item['age_days'])
        metadata.created_at = created_time
        metadata.updated_at = created_time
        
        # Simulate last access
        if item['last_access_days'] > 0:
            last_access = datetime.now() - timedelta(days=item['last_access_days'])
            metadata.last_accessed = last_access
            metadata.access_count = 5  # Some access history
            
            # Determine access pattern based on recency
            if item['last_access_days'] <= 7:
                metadata.access_pattern = AccessPattern.FREQUENT
            elif item['last_access_days'] <= 30:
                metadata.access_pattern = AccessPattern.OCCASIONAL
            else:
                metadata.access_pattern = AccessPattern.DORMANT
        
        print(f"  {item['title']}: Created {item['age_days']} days ago, last accessed {item['last_access_days']} days ago")
    
    # Show initial states
    print(f"\nInitial knowledge states:")
    for knowledge_id in knowledge_ids:
        metadata = lifecycle_manager.knowledge_items[knowledge_id]
        print(f"  {metadata.title}: {metadata.state.value} (access: {metadata.access_pattern.value})")
    
    # Perform maintenance
    print(f"\nPerforming automated maintenance:")
    maintenance_results = await lifecycle_manager.perform_maintenance()
    
    print(f"Maintenance results:")
    for key, value in maintenance_results.items():
        print(f"  {key}: {value}")
    
    # Show updated states
    print(f"\nUpdated knowledge states after maintenance:")
    for knowledge_id in knowledge_ids:
        metadata = lifecycle_manager.knowledge_items[knowledge_id]
        status_tags = list(metadata.tags) if metadata.tags else []
        print(f"  {metadata.title}: {metadata.state.value}")
        
        if status_tags:
            print(f"    Tags: {status_tags}")
    
    # Show recent lifecycle actions
    print(f"\nRecent lifecycle actions:")
    recent_actions = lifecycle_manager.lifecycle_actions[-5:]  # Last 5 actions
    
    for action in recent_actions:
        metadata = lifecycle_manager.knowledge_items.get(action.knowledge_id)
        title = metadata.title if metadata else "Unknown"
        
        print(f"  {action.action_type.value}: {title}")
        print(f"    Performed by: {action.performed_by}")
        print(f"    Description: {action.description}")
        
        if action.previous_state and action.new_state:
            print(f"    State: {action.previous_state.value} → {action.new_state.value}")
        print()

async def demo_comprehensive_lifecycle_report():
    """Demo: Comprehensive lifecycle reporting"""
    print("\nDEMO 5: COMPREHENSIVE LIFECYCLE REPORT")
    print("=" * 50)
    
    lifecycle_manager = LifecycleManager()
    await lifecycle_manager.initialize()
    
    # Create diverse knowledge ecosystem
    print("Building comprehensive knowledge ecosystem:")
    
    # Create knowledge with different characteristics
    knowledge_configs = [
        {'title': 'High Quality Guide', 'quality': 'excellent', 'access': 'frequent'},
        {'title': 'Good Documentation', 'quality': 'good', 'access': 'regular'},
        {'title': 'Adequate Manual', 'quality': 'adequate', 'access': 'occasional'},
        {'title': 'Poor Quality Item', 'quality': 'poor', 'access': 'rare'},
        {'title': 'Draft Content', 'quality': 'unassessed', 'access': 'dormant'},
        {'title': 'Archived Resource', 'quality': 'good', 'access': 'dormant'},
        {'title': 'Deprecated Guide', 'quality': 'adequate', 'access': 'rare'}
    ]
    
    knowledge_ids = []
    
    for config in knowledge_configs:
        # Create knowledge
        knowledge_id = await lifecycle_manager.create_knowledge(
            config['title'], 
            f"Content for {config['title']}", 
            "system_creator",
            tags=['demo', config['quality']],
            categories=['documentation']
        )
        knowledge_ids.append(knowledge_id)
        
        metadata = lifecycle_manager.knowledge_items[knowledge_id]
        
        # Set quality level
        quality_mapping = {
            'excellent': QualityLevel.EXCELLENT,
            'good': QualityLevel.GOOD,
            'adequate': QualityLevel.ADEQUATE,
            'poor': QualityLevel.POOR,
            'unassessed': QualityLevel.UNASSESSED
        }
        metadata.quality_level = quality_mapping[config['quality']]
        
        # Set access pattern
        access_mapping = {
            'frequent': AccessPattern.FREQUENT,
            'regular': AccessPattern.REGULAR,
            'occasional': AccessPattern.OCCASIONAL,
            'rare': AccessPattern.RARE,
            'dormant': AccessPattern.DORMANT
        }
        metadata.access_pattern = access_mapping[config['access']]
        
        # Set appropriate states
        if config['title'] == 'Draft Content':
            metadata.state = KnowledgeState.DRAFT
        elif config['title'] == 'Archived Resource':
            metadata.state = KnowledgeState.ARCHIVED
        elif config['title'] == 'Deprecated Guide':
            metadata.state = KnowledgeState.DEPRECATED
        else:
            metadata.state = KnowledgeState.PUBLISHED
    
    # Perform some lifecycle actions
    print("Performing lifecycle actions:")
    
    # Validate some knowledge
    await lifecycle_manager.validate_knowledge(knowledge_ids[0], "expert_validator", "expert_review")
    await lifecycle_manager.validate_knowledge(knowledge_ids[1], "peer_validator", "peer_review")
    
    # Update some knowledge
    await lifecycle_manager.update_knowledge(
        knowledge_ids[1], "Updated content", "updater", "Regular content update"
    )
    
    # Archive one item
    await lifecycle_manager.archive_knowledge(
        knowledge_ids[5], "archivist", "Low usage, moved to archive"
    )
    
    # Simulate some access
    for i in range(20):
        # Random access to popular items
        popular_items = knowledge_ids[:3]
        random_item = random.choice(popular_items)
        await lifecycle_manager.access_knowledge(random_item, f"user_{i % 5}")
    
    print("Knowledge ecosystem created and exercised")
    
    # Generate comprehensive report
    print(f"\nGenerating comprehensive lifecycle report:")
    report = lifecycle_manager.get_lifecycle_report()
    
    print(f"\n{'='*20} KNOWLEDGE LIFECYCLE REPORT {'='*20}")
    
    # Overview
    print(f"\nOVERVIEW:")
    overview = report['overview']
    print(f"  Total Knowledge Items: {overview['total_knowledge_items']}")
    
    print(f"\n  State Distribution:")
    for state, count in overview['state_distribution'].items():
        print(f"    {state.title()}: {count}")
    
    print(f"\n  Quality Distribution:")
    for quality, count in overview['quality_distribution'].items():
        print(f"    {quality.title()}: {count}")
    
    print(f"\n  Access Pattern Distribution:")
    for pattern, count in overview['access_pattern_distribution'].items():
        print(f"    {pattern.title()}: {count}")
    
    # Recent Activity
    print(f"\nRECENT ACTIVITY (Last 30 days):")
    activity = report['recent_activity']
    print(f"  Total Recent Actions: {activity['total_recent_actions']}")
    
    if activity['action_type_distribution']:
        print(f"  Action Types:")
        for action_type, count in activity['action_type_distribution'].items():
            print(f"    {action_type.title()}: {count}")
    
    # Version Management
    print(f"\nVERSION MANAGEMENT:")
    version_mgmt = report['version_management']
    print(f"  Knowledge Items with Versions: {version_mgmt['total_knowledge_items']}")
    print(f"  Total Versions: {version_mgmt['total_versions']}")
    print(f"  Average Versions per Item: {version_mgmt['average_versions_per_item']:.1f}")
    
    if version_mgmt['change_type_distribution']:
        print(f"  Change Type Distribution:")
        for change_type, count in version_mgmt['change_type_distribution'].items():
            print(f"    {change_type.title()}: {count}")
    
    # Access Analysis
    print(f"\nACCESS ANALYSIS:")
    access_analysis = report['access_analysis']
    
    if access_analysis['popular_knowledge']:
        print(f"  Popular Knowledge:")
        for item in access_analysis['popular_knowledge'][:3]:
            knowledge_title = next(
                config['title'] for config, kid in zip(knowledge_configs, knowledge_ids)
                if kid == item['knowledge_id']
            )
            print(f"    {knowledge_title}: {item['access_count']} accesses")
    
    print(f"  Dormant Knowledge Items: {access_analysis['dormant_knowledge_count']}")
    
    # System Statistics
    print(f"\nSYSTEM STATISTICS:")
    stats = report['system_statistics']
    for key, value in stats.items():
        print(f"  {key.title().replace('_', ' ')}: {value}")
    
    # Maintenance Policies
    print(f"\nMAINTENANCE POLICIES:")
    policies = report['maintenance_policies']
    for key, value in policies.items():
        print(f"  {key.title().replace('_', ' ')}: {value}")

async def main():
    """
    Demonstrate Knowledge Lifecycle Management for complete lifecycle from creation to retirement
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement comprehensive knowledge creation and validation processes
    2. How to manage knowledge versions and track changes over time
    3. How to analyze access patterns and optimize knowledge usage
    4. How to perform automated lifecycle maintenance and quality management
    5. How to build complete lifecycle management systems with reporting
    6. How to balance knowledge quality, accessibility, and system performance
    
    REAL WORLD APPLICATIONS:
    =======================
    - Enterprise knowledge management systems maintaining institutional memory
    - Medical knowledge bases ensuring current and accurate treatment information
    - Software documentation systems managing API and technical specifications
    - Educational platforms maintaining course content and learning materials
    - Legal systems managing case law, regulations, and compliance information
    - Research databases organizing and maintaining scientific publications
    """
    
    print("KNOWLEDGE LIFECYCLE MANAGEMENT DEMONSTRATION")
    print("Complete lifecycle from creation to retirement!")
    
    await demo_knowledge_creation_and_validation()
    await demo_version_management()
    await demo_access_analysis()
    await demo_lifecycle_maintenance()
    await demo_comprehensive_lifecycle_report()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Knowledge creation includes proper metadata and validation processes")
    print("✓ Version management tracks changes and enables rollback capabilities")
    print("✓ Access analysis identifies usage patterns and optimization opportunities")
    print("✓ Automated maintenance ensures knowledge quality and system health")
    print("✓ Comprehensive reporting provides insights for strategic decisions")
    print("✓ Complete lifecycle management balances quality, performance, and usability")
    print("\nTHE POWER OF LIFECYCLE MANAGEMENT:")
    print("- Transforms static repositories into dynamic, self-improving systems")
    print("- Ensures knowledge remains accurate, current, and reliable over time")
    print("- Provides clear audit trails and compliance with regulatory requirements")
    print("- Creates sustainable knowledge ecosystems that improve rather than degrade")
    print("- Enables organizations to maintain institutional memory effectively")

if __name__ == "__main__":
    asyncio.run(main())
