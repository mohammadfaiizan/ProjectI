#!/usr/bin/env python3
"""
Knowledge Security and Privacy: Protecting Sensitive Information in AI Systems
============================================================================

WHAT IS THE PROBLEM?
==================
Knowledge systems handle sensitive information without proper protection:
- AI systems learn from private data but can leak personal information
- Knowledge graphs contain confidential business intelligence and trade secrets
- Training data includes personally identifiable information (PII) that must be protected
- Users query systems with sensitive information that could be exposed
- Knowledge bases aggregate data from multiple sources with different privacy requirements
- Adversarial attacks can extract private information from AI model outputs

Example: Medical AI Privacy Violations
INSECURE KNOWLEDGE SYSTEM (Traditional):
- Medical AI trained on patient records without proper anonymization
- System can memorize and regurgitate specific patient information
- Queries like "What treatment did John Smith receive?" return private details
- Model weights encode patient-specific information that can be extracted
- No access controls prevent unauthorized queries about individuals
- Result: HIPAA violations, privacy breaches, loss of patient trust

REAL WORLD EXAMPLE:
=================
How does Apple implement differential privacy in Siri?

APPLE'S PRIVACY-PRESERVING KNOWLEDGE:
1. LOCAL DIFFERENTIAL PRIVACY: Adds noise to user data before it leaves device
2. K-ANONYMITY: Ensures user queries can't be linked to specific individuals
3. SELECTIVE DISCLOSURE: Only shares aggregated, non-identifiable patterns
4. ON-DEVICE PROCESSING: Keeps sensitive computations local when possible
5. FEDERATED LEARNING: Trains models without centralizing raw user data
6. CRYPTOGRAPHIC PROTECTION: Encrypts knowledge in transit and at rest
7. ACCESS CONTROLS: Implements strict authorization for different data types

BENEFITS OF KNOWLEDGE SECURITY & PRIVACY:
- Protects individual privacy while enabling collective intelligence
- Ensures compliance with GDPR, CCPA, HIPAA, and other regulations
- Maintains user trust through transparent privacy practices
- Prevents adversarial extraction of sensitive information
- Enables secure knowledge sharing across organizational boundaries
- Reduces liability and reputational risks from privacy breaches

THE PRIVACY ADVANTAGE:
====================
INSECURE: Raw data → AI learns everything → Privacy violations
SECURE: Anonymized data → Privacy-preserving AI → Protected insights

KNOWLEDGE SECURITY COMPONENTS:
=============================
1. DIFFERENTIAL PRIVACY: Add statistical noise to prevent individual identification
2. FEDERATED LEARNING: Train models without centralizing sensitive data
3. HOMOMORPHIC ENCRYPTION: Compute on encrypted data without decryption
4. SECURE MULTI-PARTY COMPUTATION: Multiple parties compute without revealing inputs
5. ZERO-KNOWLEDGE PROOFS: Prove knowledge without revealing the knowledge itself
6. ACCESS CONTROL: Role-based permissions for different types of knowledge
7. DATA ANONYMIZATION: Remove or obfuscate personally identifiable information
8. AUDIT TRAILS: Track all access and modifications to sensitive knowledge

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems that are both intelligent and privacy-preserving
- Critical for adoption in healthcare, finance, government, and personal AI
- Prevents adversarial attacks that extract private training data
- Ensures regulatory compliance while maintaining system functionality
- Creates foundation for trustworthy AI that respects individual privacy
- Enables secure collaboration without revealing sensitive information
"""

import asyncio
import time
import json
import uuid
import hashlib
import hmac
import secrets
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class SecurityAction(Enum):
    """Types of security actions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MODIFY = "modify"
    QUERY = "query"
    EXPORT = "export"
    SHARE = "share"

class AnonymizationMethod(Enum):
    """Data anonymization methods"""
    REDACTION = "redaction"
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"
    PERTURBATION = "perturbation"
    PSEUDONYMIZATION = "pseudonymization"
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"

class AttackType(Enum):
    """Types of privacy attacks"""
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    PROPERTY_INFERENCE = "property_inference"
    EXTRACTION = "extraction"
    RECONSTRUCTION = "reconstruction"
    LINKAGE = "linkage"

@dataclass
class SecurityPolicy:
    """Defines security and privacy policies"""
    
    id: str
    name: str
    description: str
    
    # Privacy settings
    privacy_level: PrivacyLevel
    retention_period: timedelta
    
    # Access controls
    allowed_roles: Set[str] = field(default_factory=set)
    allowed_actions: Set[SecurityAction] = field(default_factory=set)
    
    # Anonymization requirements
    anonymization_methods: List[AnonymizationMethod] = field(default_factory=list)
    min_k_anonymity: int = 5
    
    # Encryption requirements
    encryption_required: bool = True
    encryption_algorithm: str = "AES-256"
    
    # Audit requirements
    audit_all_access: bool = True
    
    # Geographic restrictions
    allowed_regions: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class UserContext:
    """Represents user context for access control"""
    
    user_id: str
    roles: Set[str]
    clearance_level: PrivacyLevel
    
    # Location context
    region: str = ""
    ip_address: str = ""
    
    # Session context
    session_id: str = ""
    authenticated_at: datetime = field(default_factory=datetime.now)
    
    # Usage context
    purpose: str = ""
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def has_clearance(self, required_level: PrivacyLevel) -> bool:
        """Check if user has required security clearance"""
        
        clearance_hierarchy = {
            PrivacyLevel.PUBLIC: 0,
            PrivacyLevel.INTERNAL: 1,
            PrivacyLevel.CONFIDENTIAL: 2,
            PrivacyLevel.RESTRICTED: 3,
            PrivacyLevel.TOP_SECRET: 4
        }
        
        user_level = clearance_hierarchy.get(self.clearance_level, 0)
        required_level_value = clearance_hierarchy.get(required_level, 0)
        
        return user_level >= required_level_value

@dataclass
class AuditEvent:
    """Represents a security audit event"""
    
    id: str
    timestamp: datetime
    
    # Event details
    action: SecurityAction
    resource: str
    user_id: str
    
    # Context
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""
    
    # Results
    success: bool = True
    error_message: str = ""
    
    # Privacy impact
    privacy_level: Optional[PrivacyLevel] = None
    data_size: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class PrivateData:
    """Represents private data with protection metadata"""
    
    id: str
    content: Any
    privacy_level: PrivacyLevel
    
    # Ownership
    owner_id: str = ""
    data_source: str = ""
    
    # Privacy protection
    is_anonymized: bool = False
    anonymization_methods: List[AnonymizationMethod] = field(default_factory=list)
    
    # Encryption
    is_encrypted: bool = False
    encryption_key_id: str = ""
    
    # Access tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Expiration
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class DifferentialPrivacy:
    """Implements differential privacy mechanisms"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        
        self.logger = logging.getLogger("DifferentialPrivacy")
    
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy"""
        
        # Laplace mechanism: scale = sensitivity / epsilon
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float) -> float:
        """Add Gaussian noise for differential privacy"""
        
        # Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        
        return value + noise
    
    def privatize_query_result(self, result: Union[int, float], 
                             sensitivity: float, mechanism: str = "laplace") -> float:
        """Add noise to query result for privacy"""
        
        if mechanism == "laplace":
            return self.add_laplace_noise(float(result), sensitivity)
        elif mechanism == "gaussian":
            return self.add_gaussian_noise(float(result), sensitivity)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
    
    def privatize_histogram(self, histogram: Dict[str, int], 
                          sensitivity: float = 1.0) -> Dict[str, float]:
        """Add noise to histogram counts"""
        
        private_histogram = {}
        
        for key, count in histogram.items():
            noisy_count = self.add_laplace_noise(count, sensitivity)
            # Ensure non-negative counts
            private_histogram[key] = max(0, noisy_count)
        
        return private_histogram
    
    def check_privacy_budget(self, requested_epsilon: float) -> bool:
        """Check if privacy budget allows the request"""
        
        # Simple budget tracking (in practice, implement sophisticated composition)
        return requested_epsilon <= self.epsilon
    
    def consume_privacy_budget(self, used_epsilon: float) -> None:
        """Consume privacy budget"""
        
        self.epsilon = max(0, self.epsilon - used_epsilon)
        self.logger.debug(f"Privacy budget consumed: {used_epsilon}, remaining: {self.epsilon}")

class DataAnonymizer:
    """Implements various data anonymization techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataAnonymizer")
        
        # Common PII patterns
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
            'name': re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b')  # Simple name pattern
        }
    
    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Redact personally identifiable information"""
        
        anonymized_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            anonymized_text = pattern.sub(replacement, anonymized_text)
        
        return anonymized_text
    
    def generalize_age(self, age: int, bucket_size: int = 10) -> str:
        """Generalize age into age ranges"""
        
        if age < 0:
            return "unknown"
        
        lower_bound = (age // bucket_size) * bucket_size
        upper_bound = lower_bound + bucket_size - 1
        
        return f"{lower_bound}-{upper_bound}"
    
    def generalize_location(self, address: str, level: str = "city") -> str:
        """Generalize location to less specific level"""
        
        # Simple implementation - in practice, use proper geocoding
        parts = address.split(",")
        
        if level == "country" and len(parts) >= 1:
            return parts[-1].strip()
        elif level == "state" and len(parts) >= 2:
            return ",".join(parts[-2:]).strip()
        elif level == "city" and len(parts) >= 3:
            return ",".join(parts[-3:]).strip()
        else:
            return address
    
    def k_anonymize_table(self, data: List[Dict[str, Any]], 
                         quasi_identifiers: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """Apply k-anonymity to tabular data"""
        
        # Group records by quasi-identifier combinations
        groups = defaultdict(list)
        
        for record in data:
            # Create key from quasi-identifiers
            key = tuple(record.get(qi, "") for qi in quasi_identifiers)
            groups[key].append(record)
        
        # Process groups to ensure k-anonymity
        anonymized_data = []
        
        for group_key, group_records in groups.items():
            if len(group_records) >= k:
                # Group already satisfies k-anonymity
                anonymized_data.extend(group_records)
            else:
                # Need to generalize or suppress
                generalized_records = self._generalize_group(
                    group_records, quasi_identifiers, k
                )
                anonymized_data.extend(generalized_records)
        
        return anonymized_data
    
    def _generalize_group(self, records: List[Dict[str, Any]], 
                         quasi_identifiers: List[str], k: int) -> List[Dict[str, Any]]:
        """Generalize a group to achieve k-anonymity"""
        
        # Simple generalization strategy
        generalized_records = []
        
        for record in records:
            generalized_record = record.copy()
            
            # Generalize quasi-identifiers
            for qi in quasi_identifiers:
                if qi in record:
                    value = record[qi]
                    
                    # Apply appropriate generalization
                    if isinstance(value, int) and qi == 'age':
                        generalized_record[qi] = self.generalize_age(value)
                    elif isinstance(value, str) and 'address' in qi.lower():
                        generalized_record[qi] = self.generalize_location(value)
                    else:
                        # Default: use range or category
                        generalized_record[qi] = f"category_{qi}"
            
            generalized_records.append(generalized_record)
        
        return generalized_records
    
    def pseudonymize_identifiers(self, data: Dict[str, Any], 
                               identifier_fields: List[str]) -> Dict[str, Any]:
        """Replace identifiers with pseudonyms"""
        
        pseudonymized_data = data.copy()
        
        for field in identifier_fields:
            if field in data:
                original_value = str(data[field])
                
                # Generate deterministic pseudonym using hash
                pseudonym = hashlib.sha256(original_value.encode()).hexdigest()[:16]
                pseudonymized_data[field] = f"pseudo_{pseudonym}"
        
        return pseudonymized_data

class EncryptionManager:
    """Manages encryption and decryption of sensitive data"""
    
    def __init__(self):
        self.keys: Dict[str, bytes] = {}
        self.logger = logging.getLogger("EncryptionManager")
    
    def generate_key(self, key_id: str) -> str:
        """Generate a new encryption key"""
        
        key = Fernet.generate_key()
        self.keys[key_id] = key
        
        self.logger.debug(f"Generated encryption key: {key_id}")
        
        return key_id
    
    def encrypt_data(self, data: Any, key_id: str) -> bytes:
        """Encrypt data using specified key"""
        
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        # Serialize data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = json.dumps(data).encode('utf-8')
        
        # Encrypt
        fernet = Fernet(self.keys[key_id])
        encrypted_data = fernet.encrypt(data_bytes)
        
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> Any:
        """Decrypt data using specified key"""
        
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        try:
            # Decrypt
            fernet = Fernet(self.keys[key_id])
            decrypted_bytes = fernet.decrypt(encrypted_data)
            
            # Deserialize
            decrypted_str = decrypted_bytes.decode('utf-8')
            
            try:
                # Try to parse as JSON
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                # Return as string
                return decrypted_str
                
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def rotate_key(self, old_key_id: str, new_key_id: str) -> str:
        """Rotate encryption key"""
        
        if old_key_id not in self.keys:
            raise ValueError(f"Old key not found: {old_key_id}")
        
        # Generate new key
        new_key = Fernet.generate_key()
        self.keys[new_key_id] = new_key
        
        # Note: In practice, re-encrypt all data with new key
        self.logger.info(f"Key rotated: {old_key_id} -> {new_key_id}")
        
        return new_key_id

class AccessController:
    """Implements role-based access control"""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.audit_events: List[AuditEvent] = []
        
        self.logger = logging.getLogger("AccessController")
    
    def add_policy(self, policy: SecurityPolicy) -> bool:
        """Add a security policy"""
        
        try:
            self.policies[policy.id] = policy
            self.logger.debug(f"Added security policy: {policy.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add policy: {e}")
            return False
    
    def check_access(self, user_context: UserContext, resource_id: str,
                    action: SecurityAction, privacy_level: PrivacyLevel) -> bool:
        """Check if user has access to perform action on resource"""
        
        # Find applicable policy
        policy = self._find_applicable_policy(resource_id, privacy_level)
        
        if not policy:
            self.logger.warning(f"No policy found for resource: {resource_id}")
            return False
        
        # Check role-based access
        if not self._check_role_access(user_context, policy):
            self._log_audit_event(user_context, resource_id, action, False, "Insufficient role")
            return False
        
        # Check clearance level
        if not user_context.has_clearance(policy.privacy_level):
            self._log_audit_event(user_context, resource_id, action, False, "Insufficient clearance")
            return False
        
        # Check allowed actions
        if action not in policy.allowed_actions:
            self._log_audit_event(user_context, resource_id, action, False, "Action not allowed")
            return False
        
        # Check geographic restrictions
        if policy.allowed_regions and user_context.region not in policy.allowed_regions:
            self._log_audit_event(user_context, resource_id, action, False, "Geographic restriction")
            return False
        
        # Access granted
        self._log_audit_event(user_context, resource_id, action, True)
        return True
    
    def _find_applicable_policy(self, resource_id: str, 
                               privacy_level: PrivacyLevel) -> Optional[SecurityPolicy]:
        """Find the most specific applicable policy"""
        
        # Simple implementation - in practice, use hierarchical policy matching
        for policy in self.policies.values():
            if policy.privacy_level == privacy_level:
                return policy
        
        # Default to most restrictive policy
        most_restrictive = None
        max_level = -1
        
        for policy in self.policies.values():
            level_value = list(PrivacyLevel).index(policy.privacy_level)
            if level_value > max_level:
                max_level = level_value
                most_restrictive = policy
        
        return most_restrictive
    
    def _check_role_access(self, user_context: UserContext, 
                          policy: SecurityPolicy) -> bool:
        """Check if user roles satisfy policy requirements"""
        
        if not policy.allowed_roles:
            return True  # No role restrictions
        
        # Check if user has any of the allowed roles
        return bool(user_context.roles.intersection(policy.allowed_roles))
    
    def _log_audit_event(self, user_context: UserContext, resource_id: str,
                        action: SecurityAction, success: bool, 
                        error_message: str = "") -> None:
        """Log security audit event"""
        
        event = AuditEvent(
            id="",
            timestamp=datetime.now(),
            action=action,
            resource=resource_id,
            user_id=user_context.user_id,
            session_id=user_context.session_id,
            ip_address=user_context.ip_address,
            success=success,
            error_message=error_message
        )
        
        self.audit_events.append(event)
        
        if not success:
            self.logger.warning(f"Access denied: {user_context.user_id} -> {resource_id} ({error_message})")
    
    def get_audit_events(self, start_time: datetime = None, 
                        end_time: datetime = None,
                        user_id: str = None) -> List[AuditEvent]:
        """Retrieve audit events with optional filtering"""
        
        filtered_events = self.audit_events
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        return filtered_events

class PrivacyAttackDetector:
    """Detects potential privacy attacks"""
    
    def __init__(self):
        self.query_history: List[Dict[str, Any]] = []
        self.attack_patterns: Dict[AttackType, Dict[str, Any]] = {
            AttackType.MEMBERSHIP_INFERENCE: {
                'repeated_queries': 10,
                'time_window': timedelta(minutes=5)
            },
            AttackType.EXTRACTION: {
                'systematic_queries': 20,
                'pattern_threshold': 0.8
            }
        }
        
        self.logger = logging.getLogger("PrivacyAttackDetector")
    
    def analyze_query(self, user_id: str, query: str, 
                     result: Any, timestamp: datetime = None) -> List[AttackType]:
        """Analyze query for potential privacy attacks"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Record query
        query_record = {
            'user_id': user_id,
            'query': query,
            'result': result,
            'timestamp': timestamp,
            'query_hash': hashlib.sha256(query.encode()).hexdigest()
        }
        
        self.query_history.append(query_record)
        
        # Clean old history
        self._clean_old_queries()
        
        # Detect attacks
        detected_attacks = []
        
        # Check for membership inference
        if self._detect_membership_inference(user_id, timestamp):
            detected_attacks.append(AttackType.MEMBERSHIP_INFERENCE)
        
        # Check for systematic extraction
        if self._detect_systematic_extraction(user_id, timestamp):
            detected_attacks.append(AttackType.EXTRACTION)
        
        if detected_attacks:
            self.logger.warning(f"Potential privacy attacks detected for user {user_id}: {detected_attacks}")
        
        return detected_attacks
    
    def _detect_membership_inference(self, user_id: str, timestamp: datetime) -> bool:
        """Detect potential membership inference attacks"""
        
        pattern = self.attack_patterns[AttackType.MEMBERSHIP_INFERENCE]
        time_window = pattern['time_window']
        threshold = pattern['repeated_queries']
        
        # Count recent queries from user
        cutoff_time = timestamp - time_window
        recent_queries = [
            q for q in self.query_history
            if q['user_id'] == user_id and q['timestamp'] >= cutoff_time
        ]
        
        # Check for repeated similar queries
        query_hashes = [q['query_hash'] for q in recent_queries]
        unique_hashes = set(query_hashes)
        
        # If many repeated queries, might be membership inference
        repetition_rate = len(query_hashes) / max(1, len(unique_hashes))
        
        return len(recent_queries) >= threshold and repetition_rate > 2.0
    
    def _detect_systematic_extraction(self, user_id: str, timestamp: datetime) -> bool:
        """Detect systematic data extraction attempts"""
        
        pattern = self.attack_patterns[AttackType.EXTRACTION]
        threshold = pattern['systematic_queries']
        pattern_threshold = pattern['pattern_threshold']
        
        # Get user's query history
        user_queries = [
            q for q in self.query_history
            if q['user_id'] == user_id
        ]
        
        if len(user_queries) < threshold:
            return False
        
        # Analyze query patterns
        query_similarity = self._calculate_query_similarity(user_queries)
        
        return query_similarity > pattern_threshold
    
    def _calculate_query_similarity(self, queries: List[Dict[str, Any]]) -> float:
        """Calculate similarity between queries"""
        
        if len(queries) < 2:
            return 0.0
        
        # Simple similarity based on common terms
        all_terms = []
        query_term_sets = []
        
        for query in queries:
            terms = set(query['query'].lower().split())
            query_term_sets.append(terms)
            all_terms.extend(terms)
        
        # Calculate average pairwise Jaccard similarity
        similarities = []
        
        for i in range(len(query_term_sets)):
            for j in range(i + 1, len(query_term_sets)):
                set1, set2 = query_term_sets[i], query_term_sets[j]
                
                if len(set1) == 0 and len(set2) == 0:
                    similarity = 1.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _clean_old_queries(self, retention_period: timedelta = timedelta(days=7)) -> None:
        """Remove old query records"""
        
        cutoff_time = datetime.now() - retention_period
        
        self.query_history = [
            q for q in self.query_history
            if q['timestamp'] >= cutoff_time
        ]

class SecureKnowledgeSystem:
    """Complete secure knowledge system with privacy protection"""
    
    def __init__(self):
        # Core components
        self.differential_privacy = DifferentialPrivacy(epsilon=1.0)
        self.anonymizer = DataAnonymizer()
        self.encryption_manager = EncryptionManager()
        self.access_controller = AccessController()
        self.attack_detector = PrivacyAttackDetector()
        
        # Data storage
        self.private_data: Dict[str, PrivateData] = {}
        self.policies: Dict[str, SecurityPolicy] = {}
        
        # System configuration
        self.default_privacy_level = PrivacyLevel.CONFIDENTIAL
        self.require_encryption = True
        
        # Statistics
        self.stats = {
            'data_items': 0,
            'queries_processed': 0,
            'access_denied': 0,
            'attacks_detected': 0,
            'anonymizations_performed': 0
        }
        
        self.logger = logging.getLogger("SecureKnowledgeSystem")
    
    async def initialize(self) -> None:
        """Initialize the secure knowledge system"""
        
        # Create default policies
        await self._create_default_policies()
        
        self.logger.info("Secure knowledge system initialized")
    
    async def store_data(self, content: Any, privacy_level: PrivacyLevel,
                        owner_id: str = "", anonymize: bool = True) -> str:
        """Store data with privacy protection"""
        
        try:
            # Create data record
            data_id = str(uuid.uuid4())
            
            processed_content = content
            anonymization_methods = []
            
            # Apply anonymization if requested
            if anonymize:
                if isinstance(content, str):
                    processed_content = self.anonymizer.redact_pii(content)
                    anonymization_methods.append(AnonymizationMethod.REDACTION)
                elif isinstance(content, dict):
                    processed_content = self.anonymizer.pseudonymize_identifiers(
                        content, ['id', 'user_id', 'customer_id']
                    )
                    anonymization_methods.append(AnonymizationMethod.PSEUDONYMIZATION)
                
                self.stats['anonymizations_performed'] += 1
            
            # Encrypt if required
            encryption_key_id = ""
            is_encrypted = False
            
            if self.require_encryption or privacy_level in [PrivacyLevel.RESTRICTED, PrivacyLevel.TOP_SECRET]:
                encryption_key_id = self.encryption_manager.generate_key(f"key_{data_id}")
                processed_content = self.encryption_manager.encrypt_data(processed_content, encryption_key_id)
                is_encrypted = True
            
            # Create private data record
            private_data = PrivateData(
                id=data_id,
                content=processed_content,
                privacy_level=privacy_level,
                owner_id=owner_id,
                is_anonymized=anonymize,
                anonymization_methods=anonymization_methods,
                is_encrypted=is_encrypted,
                encryption_key_id=encryption_key_id
            )
            
            self.private_data[data_id] = private_data
            self.stats['data_items'] += 1
            
            self.logger.debug(f"Stored private data: {data_id}")
            
            return data_id
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
            raise
    
    async def query_data(self, data_id: str, user_context: UserContext) -> Optional[Any]:
        """Query data with access control and privacy protection"""
        
        try:
            self.stats['queries_processed'] += 1
            
            # Check if data exists
            if data_id not in self.private_data:
                return None
            
            private_data = self.private_data[data_id]
            
            # Check access permissions
            has_access = self.access_controller.check_access(
                user_context, data_id, SecurityAction.READ, private_data.privacy_level
            )
            
            if not has_access:
                self.stats['access_denied'] += 1
                return None
            
            # Detect potential attacks
            attacks = self.attack_detector.analyze_query(
                user_context.user_id, f"query_data:{data_id}", "data_returned"
            )
            
            if attacks:
                self.stats['attacks_detected'] += len(attacks)
                # In practice, might block or throttle suspicious users
            
            # Decrypt if needed
            content = private_data.content
            if private_data.is_encrypted:
                content = self.encryption_manager.decrypt_data(
                    content, private_data.encryption_key_id
                )
            
            # Update access tracking
            private_data.access_count += 1
            private_data.last_accessed = datetime.now()
            
            # Apply differential privacy if needed
            if private_data.privacy_level in [PrivacyLevel.RESTRICTED, PrivacyLevel.TOP_SECRET]:
                if isinstance(content, (int, float)):
                    content = self.differential_privacy.privatize_query_result(content, sensitivity=1.0)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return None
    
    async def aggregate_query(self, query_pattern: str, user_context: UserContext,
                            apply_differential_privacy: bool = True) -> Dict[str, Any]:
        """Perform aggregate query with privacy protection"""
        
        try:
            # Simple pattern matching for demo
            matching_data = []
            
            for data_id, private_data in self.private_data.items():
                # Check access
                has_access = self.access_controller.check_access(
                    user_context, data_id, SecurityAction.QUERY, private_data.privacy_level
                )
                
                if has_access:
                    # Simple pattern matching
                    if isinstance(private_data.content, str) and query_pattern.lower() in private_data.content.lower():
                        matching_data.append(private_data)
            
            # Calculate aggregates
            aggregates = {
                'count': len(matching_data),
                'privacy_levels': defaultdict(int),
                'avg_access_count': 0.0
            }
            
            if matching_data:
                # Privacy level distribution
                for data in matching_data:
                    aggregates['privacy_levels'][data.privacy_level.value] += 1
                
                # Average access count
                total_access = sum(data.access_count for data in matching_data)
                aggregates['avg_access_count'] = total_access / len(matching_data)
            
            # Apply differential privacy to aggregates
            if apply_differential_privacy:
                aggregates['count'] = self.differential_privacy.privatize_query_result(
                    aggregates['count'], sensitivity=1.0
                )
                
                # Privatize histogram
                aggregates['privacy_levels'] = self.differential_privacy.privatize_histogram(
                    dict(aggregates['privacy_levels'])
                )
            
            # Detect potential attacks
            self.attack_detector.analyze_query(
                user_context.user_id, f"aggregate:{query_pattern}", aggregates
            )
            
            return aggregates
            
        except Exception as e:
            self.logger.error(f"Aggregate query failed: {e}")
            return {}
    
    async def create_policy(self, name: str, privacy_level: PrivacyLevel,
                          allowed_roles: List[str], allowed_actions: List[SecurityAction]) -> str:
        """Create a new security policy"""
        
        policy = SecurityPolicy(
            id="",
            name=name,
            description=f"Policy for {privacy_level.value} data",
            privacy_level=privacy_level,
            retention_period=timedelta(days=365),
            allowed_roles=set(allowed_roles),
            allowed_actions=set(allowed_actions)
        )
        
        success = self.access_controller.add_policy(policy)
        
        if success:
            self.policies[policy.id] = policy
            return policy.id
        else:
            raise Exception("Failed to create policy")
    
    async def _create_default_policies(self) -> None:
        """Create default security policies"""
        
        # Public data policy
        await self.create_policy(
            "Public Data",
            PrivacyLevel.PUBLIC,
            ["guest", "user", "admin"],
            [SecurityAction.READ, SecurityAction.QUERY]
        )
        
        # Confidential data policy
        await self.create_policy(
            "Confidential Data",
            PrivacyLevel.CONFIDENTIAL,
            ["user", "admin"],
            [SecurityAction.READ, SecurityAction.QUERY]
        )
        
        # Restricted data policy
        await self.create_policy(
            "Restricted Data",
            PrivacyLevel.RESTRICTED,
            ["admin"],
            [SecurityAction.READ]
        )
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy and security report"""
        
        # Analyze stored data
        data_analysis = {
            'total_items': len(self.private_data),
            'privacy_distribution': defaultdict(int),
            'encryption_status': {'encrypted': 0, 'unencrypted': 0},
            'anonymization_status': {'anonymized': 0, 'raw': 0}
        }
        
        for data in self.private_data.values():
            data_analysis['privacy_distribution'][data.privacy_level.value] += 1
            
            if data.is_encrypted:
                data_analysis['encryption_status']['encrypted'] += 1
            else:
                data_analysis['encryption_status']['unencrypted'] += 1
            
            if data.is_anonymized:
                data_analysis['anonymization_status']['anonymized'] += 1
            else:
                data_analysis['anonymization_status']['raw'] += 1
        
        # Analyze access patterns
        recent_audits = self.access_controller.get_audit_events(
            start_time=datetime.now() - timedelta(days=7)
        )
        
        access_analysis = {
            'total_access_attempts': len(recent_audits),
            'successful_access': len([e for e in recent_audits if e.success]),
            'denied_access': len([e for e in recent_audits if not e.success]),
            'unique_users': len(set(e.user_id for e in recent_audits))
        }
        
        return {
            'system_statistics': self.stats,
            'data_analysis': dict(data_analysis),
            'access_analysis': access_analysis,
            'privacy_budget_remaining': self.differential_privacy.epsilon,
            'policies_configured': len(self.policies)
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_data_anonymization():
    """Demo: Data anonymization techniques"""
    print("\nDEMO 1: DATA ANONYMIZATION")
    print("=" * 50)
    
    anonymizer = DataAnonymizer()
    
    # Test PII redaction
    print("Testing PII redaction:")
    sensitive_text = """
    Contact John Smith at john.smith@email.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012.
    Address: 123 Main St, Anytown, CA 90210
    """
    
    print("Original text:")
    print(sensitive_text)
    
    anonymized_text = anonymizer.redact_pii(sensitive_text)
    print("\nAnonymized text:")
    print(anonymized_text)
    
    # Test age generalization
    print(f"\nTesting age generalization:")
    ages = [25, 33, 47, 52, 68, 71]
    
    for age in ages:
        generalized = anonymizer.generalize_age(age)
        print(f"  Age {age} → {generalized}")
    
    # Test k-anonymity
    print(f"\nTesting k-anonymity (k=3):")
    sample_data = [
        {'name': 'Alice', 'age': 25, 'zip': '12345', 'disease': 'flu'},
        {'name': 'Bob', 'age': 27, 'zip': '12345', 'disease': 'cold'},
        {'name': 'Charlie', 'age': 30, 'zip': '54321', 'disease': 'flu'},
        {'name': 'David', 'age': 32, 'zip': '54321', 'disease': 'headache'},
        {'name': 'Eve', 'age': 28, 'zip': '12345', 'disease': 'flu'},
    ]
    
    print("Original data:")
    for record in sample_data:
        print(f"  {record}")
    
    anonymized_data = anonymizer.k_anonymize_table(
        sample_data, ['age', 'zip'], k=3
    )
    
    print("\nK-anonymized data:")
    for record in anonymized_data:
        print(f"  {record}")

async def demo_differential_privacy():
    """Demo: Differential privacy mechanisms"""
    print("\nDEMO 2: DIFFERENTIAL PRIVACY")
    print("=" * 50)
    
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    
    # Test query result privatization
    print("Testing query result privatization:")
    true_count = 1000
    
    print(f"True count: {true_count}")
    print("Privatized results (Laplace mechanism):")
    
    for i in range(5):
        noisy_count = dp.privatize_query_result(true_count, sensitivity=1.0, mechanism="laplace")
        print(f"  Query {i+1}: {noisy_count:.1f}")
    
    print("\nPrivatized results (Gaussian mechanism):")
    for i in range(5):
        noisy_count = dp.privatize_query_result(true_count, sensitivity=1.0, mechanism="gaussian")
        print(f"  Query {i+1}: {noisy_count:.1f}")
    
    # Test histogram privatization
    print(f"\nTesting histogram privatization:")
    true_histogram = {
        'category_A': 150,
        'category_B': 300,
        'category_C': 450,
        'category_D': 100
    }
    
    print("True histogram:")
    for category, count in true_histogram.items():
        print(f"  {category}: {count}")
    
    private_histogram = dp.privatize_histogram(true_histogram)
    
    print("\nPrivatized histogram:")
    for category, count in private_histogram.items():
        print(f"  {category}: {count:.1f}")
    
    # Show privacy budget consumption
    print(f"\nPrivacy budget tracking:")
    print(f"  Initial epsilon: 1.0")
    print(f"  Current epsilon: {dp.epsilon:.3f}")
    print(f"  Queries used budget, remaining privacy protection")

async def demo_access_control():
    """Demo: Role-based access control"""
    print("\nDEMO 3: ACCESS CONTROL")
    print("=" * 50)
    
    controller = AccessController()
    
    # Create security policies
    print("Creating security policies:")
    
    policies = [
        SecurityPolicy(
            id="public_policy",
            name="Public Data Policy",
            description="Policy for public information",
            privacy_level=PrivacyLevel.PUBLIC,
            retention_period=timedelta(days=30),
            allowed_roles={"guest", "user", "admin"},
            allowed_actions={SecurityAction.READ, SecurityAction.QUERY}
        ),
        SecurityPolicy(
            id="confidential_policy",
            name="Confidential Data Policy", 
            description="Policy for confidential information",
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            retention_period=timedelta(days=365),
            allowed_roles={"user", "admin"},
            allowed_actions={SecurityAction.READ, SecurityAction.QUERY}
        ),
        SecurityPolicy(
            id="restricted_policy",
            name="Restricted Data Policy",
            description="Policy for restricted information",
            privacy_level=PrivacyLevel.RESTRICTED,
            retention_period=timedelta(days=1095),
            allowed_roles={"admin"},
            allowed_actions={SecurityAction.READ}
        )
    ]
    
    for policy in policies:
        controller.add_policy(policy)
        print(f"  Created: {policy.name} (Level: {policy.privacy_level.value})")
    
    # Test access control with different users
    print(f"\nTesting access control:")
    
    users = [
        UserContext(
            user_id="guest_001",
            roles={"guest"},
            clearance_level=PrivacyLevel.PUBLIC,
            region="US"
        ),
        UserContext(
            user_id="user_001", 
            roles={"user"},
            clearance_level=PrivacyLevel.CONFIDENTIAL,
            region="US"
        ),
        UserContext(
            user_id="admin_001",
            roles={"admin"},
            clearance_level=PrivacyLevel.RESTRICTED,
            region="US"
        )
    ]
    
    test_cases = [
        ("public_data", SecurityAction.READ, PrivacyLevel.PUBLIC),
        ("confidential_data", SecurityAction.READ, PrivacyLevel.CONFIDENTIAL),
        ("restricted_data", SecurityAction.READ, PrivacyLevel.RESTRICTED),
        ("confidential_data", SecurityAction.WRITE, PrivacyLevel.CONFIDENTIAL)
    ]
    
    for user in users:
        print(f"\nUser: {user.user_id} (roles: {user.roles}, clearance: {user.clearance_level.value})")
        
        for resource, action, privacy_level in test_cases:
            has_access = controller.check_access(user, resource, action, privacy_level)
            status = "✓ GRANTED" if has_access else "✗ DENIED"
            print(f"  {resource} ({action.value}): {status}")
    
    # Show audit events
    print(f"\nAudit events:")
    audit_events = controller.get_audit_events()
    
    for event in audit_events[-5:]:  # Show last 5 events
        status = "SUCCESS" if event.success else "FAILED"
        print(f"  {event.user_id} {event.action.value} {event.resource}: {status}")

async def demo_attack_detection():
    """Demo: Privacy attack detection"""
    print("\nDEMO 4: ATTACK DETECTION")
    print("=" * 50)
    
    detector = PrivacyAttackDetector()
    
    # Simulate normal queries
    print("Simulating normal user queries:")
    normal_queries = [
        "What is the weather today?",
        "Show me recent news",
        "Find restaurants nearby",
        "What time is it?",
        "How do I reset my password?"
    ]
    
    timestamp = datetime.now()
    
    for i, query in enumerate(normal_queries):
        attacks = detector.analyze_query(
            f"normal_user", query, f"result_{i}", 
            timestamp + timedelta(minutes=i)
        )
        print(f"  Query: '{query}' - Attacks detected: {attacks}")
    
    # Simulate membership inference attack
    print(f"\nSimulating membership inference attack:")
    suspicious_queries = [
        "Is user_123 in the database?",
        "Does user_123 exist?", 
        "Show user_123 details",
        "user_123 information",
        "Find user_123",
        "user_123 data",
        "Is user_123 present?",
        "user_123 exists?",
        "Show user_123",
        "user_123 query"
    ]
    
    attack_start = datetime.now()
    
    for i, query in enumerate(suspicious_queries):
        attacks = detector.analyze_query(
            "suspicious_user", query, "no_result",
            attack_start + timedelta(seconds=30*i)  # Rapid queries
        )
        
        if attacks:
            print(f"  🚨 ATTACK DETECTED: {query} - {attacks}")
        else:
            print(f"  Query: '{query}' - No attacks detected")
    
    # Simulate systematic extraction
    print(f"\nSimulating systematic extraction attempt:")
    extraction_queries = [
        "SELECT * FROM users WHERE id=1",
        "SELECT * FROM users WHERE id=2",
        "SELECT * FROM users WHERE id=3",
        "SELECT * FROM users WHERE id=4",
        "SELECT * FROM users WHERE id=5",
        "SELECT * FROM users WHERE id=6"
    ]
    
    extraction_start = datetime.now()
    
    for i, query in enumerate(extraction_queries):
        attacks = detector.analyze_query(
            "extraction_user", query, f"user_data_{i}",
            extraction_start + timedelta(minutes=i)
        )
        
        if attacks:
            print(f"  🚨 ATTACK DETECTED: {query} - {attacks}")
        else:
            print(f"  Query: '{query}' - No attacks detected")

async def demo_secure_knowledge_system():
    """Demo: Complete secure knowledge system"""
    print("\nDEMO 5: SECURE KNOWLEDGE SYSTEM")
    print("=" * 50)
    
    system = SecureKnowledgeSystem()
    await system.initialize()
    
    print("Secure knowledge system initialized")
    
    # Store different types of data
    print(f"\nStoring sensitive data:")
    
    data_items = [
        {
            'content': "Patient John Doe has diabetes and takes insulin daily",
            'privacy_level': PrivacyLevel.RESTRICTED,
            'owner_id': "doctor_001"
        },
        {
            'content': {"user_id": "user_123", "age": 35, "location": "New York", "salary": 75000},
            'privacy_level': PrivacyLevel.CONFIDENTIAL,
            'owner_id': "hr_dept"
        },
        {
            'content': "Public announcement: Company picnic scheduled for next Friday",
            'privacy_level': PrivacyLevel.PUBLIC,
            'owner_id': "admin_001"
        },
        {
            'content': "Trade secret: Our new algorithm improves efficiency by 40%",
            'privacy_level': PrivacyLevel.TOP_SECRET,
            'owner_id': "cto_001"
        }
    ]
    
    stored_ids = []
    
    for item in data_items:
        data_id = await system.store_data(
            item['content'], 
            item['privacy_level'],
            item['owner_id']
        )
        stored_ids.append(data_id)
        
        print(f"  Stored {item['privacy_level'].value} data: {data_id[:8]}...")
    
    # Test queries with different user contexts
    print(f"\nTesting data access with different users:")
    
    test_users = [
        UserContext(
            user_id="guest_user",
            roles={"guest"},
            clearance_level=PrivacyLevel.PUBLIC,
            region="US"
        ),
        UserContext(
            user_id="regular_user",
            roles={"user"},
            clearance_level=PrivacyLevel.CONFIDENTIAL,
            region="US"
        ),
        UserContext(
            user_id="admin_user",
            roles={"admin"},
            clearance_level=PrivacyLevel.RESTRICTED,
            region="US"
        )
    ]
    
    for user in test_users:
        print(f"\nUser: {user.user_id} (clearance: {user.clearance_level.value})")
        
        accessible_count = 0
        
        for data_id in stored_ids:
            result = await system.query_data(data_id, user)
            
            if result is not None:
                accessible_count += 1
                if isinstance(result, str):
                    preview = result[:50] + "..." if len(result) > 50 else result
                else:
                    preview = str(result)[:50] + "..."
                
                print(f"  ✓ Access granted to {data_id[:8]}: {preview}")
            else:
                print(f"  ✗ Access denied to {data_id[:8]}")
        
        print(f"  Total accessible items: {accessible_count}/{len(stored_ids)}")
    
    # Test aggregate queries
    print(f"\nTesting aggregate queries:")
    
    admin_user = test_users[2]  # Use admin for aggregate queries
    
    aggregates = await system.aggregate_query("patient", admin_user)
    print(f"Aggregate query for 'patient':")
    print(f"  Count (with DP noise): {aggregates.get('count', 0):.1f}")
    print(f"  Privacy level distribution: {dict(aggregates.get('privacy_levels', {}))}")
    
    # Generate privacy report
    print(f"\nPrivacy and security report:")
    report = system.get_privacy_report()
    
    print(f"System statistics:")
    for key, value in report['system_statistics'].items():
        print(f"  {key}: {value}")
    
    print(f"Data analysis:")
    for key, value in report['data_analysis'].items():
        if isinstance(value, dict):
            print(f"  {key}: {dict(value)}")
        else:
            print(f"  {key}: {value}")
    
    print(f"Access analysis:")
    for key, value in report['access_analysis'].items():
        print(f"  {key}: {value}")

async def main():
    """
    Demonstrate Knowledge Security and Privacy for protecting sensitive information
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement data anonymization techniques (redaction, generalization, k-anonymity)
    2. How to apply differential privacy to protect individual privacy in aggregate queries
    3. How to implement role-based access control for sensitive knowledge
    4. How to detect potential privacy attacks (membership inference, extraction)
    5. How to build complete secure knowledge systems with encryption and audit trails
    6. How to balance privacy protection with system functionality and usability
    
    REAL WORLD APPLICATIONS:
    =======================
    - Healthcare systems protecting patient privacy while enabling medical research
    - Financial systems securing customer data while providing analytics
    - Educational platforms protecting student privacy in learning analytics
    - Government systems balancing transparency with national security
    - Social media platforms protecting user privacy while enabling targeted services
    - Enterprise systems securing trade secrets while enabling collaboration
    """
    
    print("KNOWLEDGE SECURITY AND PRIVACY DEMONSTRATION")
    print("Protecting sensitive information in AI systems!")
    
    await demo_data_anonymization()
    await demo_differential_privacy()
    await demo_access_control()
    await demo_attack_detection()
    await demo_secure_knowledge_system()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Data anonymization removes personally identifiable information")
    print("✓ Differential privacy adds statistical noise to protect individuals")
    print("✓ Access control enforces role-based permissions for sensitive data")
    print("✓ Attack detection identifies potential privacy violations")
    print("✓ Complete secure systems balance privacy protection with functionality")
    print("✓ Multiple protection layers provide defense in depth")
    print("\nTHE POWER OF PRIVACY-PRESERVING AI:")
    print("- Enables AI systems that are both intelligent and privacy-preserving")
    print("- Ensures regulatory compliance while maintaining system functionality")
    print("- Prevents adversarial attacks that extract private training data")
    print("- Maintains user trust through transparent privacy practices")
    print("- Creates foundation for trustworthy AI in sensitive domains")

if __name__ == "__main__":
    asyncio.run(main())
