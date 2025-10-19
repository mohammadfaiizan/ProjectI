#!/usr/bin/env python3
"""
Dynamic Knowledge Updates: Real-Time Knowledge Evolution and Synchronization
===========================================================================

WHAT IS THE PROBLEM?
==================
Knowledge becomes stale and outdated in static systems:
- Information changes rapidly but systems don't update automatically
- Knowledge bases become inconsistent with real-world facts
- Users receive outdated information leading to poor decisions
- Manual updates are slow, error-prone, and don't scale
- Different systems have different versions of the same information
- Critical updates may be missed, leading to safety issues

Example: Stock Trading Information Disaster
STATIC SYSTEM (Traditional):
- Trading system uses yesterday's market data for today's decisions
- Company merger information takes days to propagate
- Earnings reports aren't reflected in analysis systems
- Currency exchange rates remain fixed while markets fluctuate
- Regulatory changes aren't incorporated into compliance systems
- Result: Massive financial losses, regulatory violations, market disruption

REAL WORLD EXAMPLE:
=================
How does Google Search maintain fresh information?

GOOGLE'S DYNAMIC UPDATE SYSTEM:
1. CONTINUOUS CRAWLING: Web crawlers constantly monitor content changes
2. REAL-TIME INDEXING: New content is indexed within minutes of publication
3. FRESHNESS SIGNALS: Recent content is prioritized for time-sensitive queries
4. CHANGE DETECTION: Algorithms detect when important information is updated
5. CONFLICT RESOLUTION: Systems handle conflicting information from multiple sources
6. TEMPORAL VERSIONING: Track how information changes over time
7. ROLLBACK CAPABILITIES: Ability to revert to previous versions when needed

BENEFITS OF DYNAMIC KNOWLEDGE UPDATES:
- Always current information leading to better decisions
- Automatic propagation of changes across dependent systems
- Reduced manual maintenance overhead and human errors
- Real-time synchronization between multiple knowledge sources
- Ability to track and audit knowledge evolution over time
- Improved user trust through consistently fresh information

THE DYNAMIC ADVANTAGE:
====================
STATIC KNOWLEDGE: Information → Gets Stale → Poor Decisions
DYNAMIC KNOWLEDGE: Information → Continuous Updates → Current Decisions

DYNAMIC UPDATE COMPONENTS:
========================
1. CHANGE DETECTION: Monitor sources for information updates
2. UPDATE PROPAGATION: Distribute changes across knowledge systems
3. CONFLICT RESOLUTION: Handle contradictory information from different sources
4. VERSIONING SYSTEM: Track evolution and maintain historical context
5. VALIDATION PIPELINE: Ensure update quality before propagation
6. ROLLBACK MECHANISM: Revert problematic updates when necessary
7. NOTIFICATION SYSTEM: Alert users and systems to important changes

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems to stay current with rapidly changing world
- Provides foundation for real-time decision support systems
- Critical for time-sensitive applications like finance, healthcare, news
- Enables collaborative knowledge building across organizations
- Powers adaptive AI that learns and evolves with new information
- Creates competitive advantage through superior information currency
"""

import asyncio
import time
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ChangeType(Enum):
    """Types of knowledge changes"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    SPLIT = "split"
    DEPRECATE = "deprecate"
    RESTORE = "restore"

class UpdatePriority(Enum):
    """Priority levels for updates"""
    CRITICAL = "critical"       # Security, safety issues
    HIGH = "high"              # Important corrections
    MEDIUM = "medium"          # Regular updates
    LOW = "low"               # Minor improvements
    BACKGROUND = "background"  # Maintenance updates

class ConflictResolution(Enum):
    """Strategies for resolving conflicts"""
    LATEST_WINS = "latest_wins"           # Most recent update takes precedence
    HIGHEST_AUTHORITY = "highest_authority"  # Most authoritative source wins
    MANUAL_REVIEW = "manual_review"       # Require human intervention
    MERGE_VALUES = "merge_values"         # Combine information from sources
    KEEP_BOTH = "keep_both"              # Maintain multiple versions
    REJECT_UPDATE = "reject_update"       # Reject conflicting update

class UpdateStatus(Enum):
    """Status of knowledge updates"""
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    FAILED = "failed"
    CONFLICT = "conflict"
    ROLLED_BACK = "rolled_back"

@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge"""
    
    id: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Versioning
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Source tracking
    source_id: str = ""
    authority_score: float = 0.5
    
    # Content fingerprint for change detection
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.content_hash:
            self.content_hash = self.calculate_content_hash()
    
    def calculate_content_hash(self) -> str:
        """Calculate content hash for change detection"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def update_content(self, new_content: Any) -> bool:
        """Update content and return True if changed"""
        new_hash = hashlib.sha256(
            json.dumps(new_content, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        if new_hash != self.content_hash:
            self.content = new_content
            self.content_hash = new_hash
            self.updated_at = datetime.now()
            # Increment version
            version_parts = self.version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            self.version = '.'.join(version_parts)
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'source_id': self.source_id,
            'authority_score': self.authority_score,
            'content_hash': self.content_hash
        }

@dataclass
class ChangeEvent:
    """Represents a change to knowledge"""
    
    id: str
    change_type: ChangeType
    item_id: str
    
    # Change details
    old_content: Any = None
    new_content: Any = None
    change_description: str = ""
    
    # Change metadata
    source_id: str = ""
    priority: UpdatePriority = UpdatePriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Processing status
    status: UpdateStatus = UpdateStatus.PENDING
    applied_at: Optional[datetime] = None
    error_message: str = ""
    
    # Conflict information
    conflicts_with: List[str] = field(default_factory=list)
    resolution_strategy: Optional[ConflictResolution] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class UpdateSubscription:
    """Represents a subscription to knowledge updates"""
    
    id: str
    subscriber_id: str
    
    # Subscription filters
    item_filters: Dict[str, Any] = field(default_factory=dict)
    change_type_filters: List[ChangeType] = field(default_factory=list)
    priority_threshold: UpdatePriority = UpdatePriority.LOW
    
    # Callback configuration
    callback: Optional[Callable] = None
    callback_url: str = ""
    
    # Subscription metadata
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class ChangeDetector:
    """Detects changes in knowledge sources"""
    
    def __init__(self):
        self.source_monitors: Dict[str, Dict[str, Any]] = {}
        self.detection_strategies = {
            'polling': self._polling_detection,
            'webhook': self._webhook_detection,
            'filesystem': self._filesystem_detection,
            'database': self._database_detection
        }
        
        self.logger = logging.getLogger("ChangeDetector")
    
    async def register_source(self, source_id: str, source_config: Dict[str, Any]) -> bool:
        """Register a source for change detection"""
        
        try:
            strategy = source_config.get('detection_strategy', 'polling')
            
            if strategy not in self.detection_strategies:
                raise ValueError(f"Unknown detection strategy: {strategy}")
            
            self.source_monitors[source_id] = {
                'config': source_config,
                'strategy': strategy,
                'last_check': datetime.now(),
                'last_content_hash': "",
                'active': True
            }
            
            self.logger.info(f"Registered source {source_id} with {strategy} detection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register source {source_id}: {e}")
            return False
    
    async def detect_changes(self, source_id: str) -> List[ChangeEvent]:
        """Detect changes for a specific source"""
        
        if source_id not in self.source_monitors:
            return []
        
        monitor = self.source_monitors[source_id]
        
        if not monitor['active']:
            return []
        
        try:
            strategy = monitor['strategy']
            detection_func = self.detection_strategies[strategy]
            
            changes = await detection_func(source_id, monitor)
            
            # Update last check time
            monitor['last_check'] = datetime.now()
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Change detection failed for source {source_id}: {e}")
            return []
    
    async def detect_all_changes(self) -> List[ChangeEvent]:
        """Detect changes across all registered sources"""
        
        all_changes = []
        
        for source_id in self.source_monitors:
            changes = await self.detect_changes(source_id)
            all_changes.extend(changes)
        
        return all_changes
    
    async def _polling_detection(self, source_id: str, monitor: Dict[str, Any]) -> List[ChangeEvent]:
        """Detect changes using polling strategy"""
        
        config = monitor['config']
        
        # Simulate fetching content from source
        current_content = await self._fetch_source_content(source_id, config)
        
        if current_content is None:
            return []
        
        # Calculate content hash
        content_str = json.dumps(current_content, sort_keys=True, default=str)
        current_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        changes = []
        
        if monitor['last_content_hash'] and current_hash != monitor['last_content_hash']:
            # Content has changed
            change = ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id=source_id,
                new_content=current_content,
                change_description=f"Content updated via polling detection",
                source_id=source_id,
                priority=config.get('default_priority', UpdatePriority.MEDIUM)
            )
            changes.append(change)
        
        monitor['last_content_hash'] = current_hash
        
        return changes
    
    async def _webhook_detection(self, source_id: str, monitor: Dict[str, Any]) -> List[ChangeEvent]:
        """Detect changes using webhook notifications"""
        
        # In a real implementation, this would process webhook queue
        # For demo, simulate occasional webhook notifications
        
        import random
        
        if random.random() < 0.2:  # 20% chance of webhook notification
            change = ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id=source_id,
                change_description="Content updated via webhook notification",
                source_id=source_id,
                priority=UpdatePriority.HIGH
            )
            return [change]
        
        return []
    
    async def _filesystem_detection(self, source_id: str, monitor: Dict[str, Any]) -> List[ChangeEvent]:
        """Detect changes using filesystem monitoring"""
        
        # Simulate filesystem change detection
        import random
        
        if random.random() < 0.15:  # 15% chance of file change
            change = ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id=source_id,
                change_description="File updated on filesystem",
                source_id=source_id,
                priority=UpdatePriority.MEDIUM
            )
            return [change]
        
        return []
    
    async def _database_detection(self, source_id: str, monitor: Dict[str, Any]) -> List[ChangeEvent]:
        """Detect changes using database triggers/logs"""
        
        # Simulate database change log processing
        import random
        
        changes = []
        
        # Simulate multiple potential changes
        for _ in range(random.randint(0, 3)):
            if random.random() < 0.3:
                change_type = random.choice(list(ChangeType))
                
                change = ChangeEvent(
                    id="",
                    change_type=change_type,
                    item_id=f"{source_id}_item_{random.randint(1, 100)}",
                    change_description=f"Database {change_type.value} detected",
                    source_id=source_id,
                    priority=UpdatePriority.MEDIUM
                )
                changes.append(change)
        
        return changes
    
    async def _fetch_source_content(self, source_id: str, config: Dict[str, Any]) -> Optional[Any]:
        """Fetch current content from source"""
        
        # Simulate content fetching with some variation
        import random
        
        base_content = {
            'data': f"Content from {source_id}",
            'timestamp': datetime.now().isoformat(),
            'value': random.randint(1, 1000),
            'status': random.choice(['active', 'pending', 'completed'])
        }
        
        return base_content

class ConflictResolver:
    """Resolves conflicts between knowledge updates"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictResolution.LATEST_WINS: self._latest_wins_strategy,
            ConflictResolution.HIGHEST_AUTHORITY: self._highest_authority_strategy,
            ConflictResolution.MANUAL_REVIEW: self._manual_review_strategy,
            ConflictResolution.MERGE_VALUES: self._merge_values_strategy,
            ConflictResolution.KEEP_BOTH: self._keep_both_strategy,
            ConflictResolution.REJECT_UPDATE: self._reject_update_strategy
        }
        
        self.logger = logging.getLogger("ConflictResolver")
    
    async def resolve_conflict(self, existing_item: KnowledgeItem,
                             update_event: ChangeEvent,
                             strategy: ConflictResolution) -> Dict[str, Any]:
        """Resolve conflict using specified strategy"""
        
        try:
            if strategy not in self.resolution_strategies:
                raise ValueError(f"Unknown resolution strategy: {strategy}")
            
            resolution_func = self.resolution_strategies[strategy]
            result = await resolution_func(existing_item, update_event)
            
            self.logger.debug(f"Resolved conflict using {strategy.value} strategy")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'reject_update'
            }
    
    async def _latest_wins_strategy(self, existing_item: KnowledgeItem,
                                  update_event: ChangeEvent) -> Dict[str, Any]:
        """Latest update wins strategy"""
        
        if update_event.timestamp > existing_item.updated_at:
            return {
                'success': True,
                'action': 'apply_update',
                'reason': 'Update is more recent'
            }
        else:
            return {
                'success': True,
                'action': 'reject_update',
                'reason': 'Existing item is more recent'
            }
    
    async def _highest_authority_strategy(self, existing_item: KnowledgeItem,
                                        update_event: ChangeEvent) -> Dict[str, Any]:
        """Highest authority source wins strategy"""
        
        # Get authority scores (would come from source registry in real implementation)
        existing_authority = existing_item.authority_score
        update_authority = 0.7  # Simulated authority score
        
        if update_authority > existing_authority:
            return {
                'success': True,
                'action': 'apply_update',
                'reason': f'Update source has higher authority ({update_authority} vs {existing_authority})'
            }
        else:
            return {
                'success': True,
                'action': 'reject_update',
                'reason': f'Existing source has higher authority ({existing_authority} vs {update_authority})'
            }
    
    async def _manual_review_strategy(self, existing_item: KnowledgeItem,
                                    update_event: ChangeEvent) -> Dict[str, Any]:
        """Manual review required strategy"""
        
        return {
            'success': True,
            'action': 'require_manual_review',
            'reason': 'Conflict requires human intervention',
            'review_data': {
                'existing_content': existing_item.content,
                'proposed_content': update_event.new_content,
                'conflict_details': 'Content differs significantly'
            }
        }
    
    async def _merge_values_strategy(self, existing_item: KnowledgeItem,
                                   update_event: ChangeEvent) -> Dict[str, Any]:
        """Merge values strategy"""
        
        try:
            # Attempt to merge content intelligently
            if isinstance(existing_item.content, dict) and isinstance(update_event.new_content, dict):
                merged_content = existing_item.content.copy()
                merged_content.update(update_event.new_content)
                
                return {
                    'success': True,
                    'action': 'apply_merged_update',
                    'merged_content': merged_content,
                    'reason': 'Successfully merged dictionary contents'
                }
            
            elif isinstance(existing_item.content, list) and isinstance(update_event.new_content, list):
                # Merge lists by combining unique elements
                merged_content = list(set(existing_item.content + update_event.new_content))
                
                return {
                    'success': True,
                    'action': 'apply_merged_update',
                    'merged_content': merged_content,
                    'reason': 'Successfully merged list contents'
                }
            
            else:
                # Can't merge different types, use latest wins
                return await self._latest_wins_strategy(existing_item, update_event)
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Merge failed: {e}",
                'action': 'require_manual_review'
            }
    
    async def _keep_both_strategy(self, existing_item: KnowledgeItem,
                                update_event: ChangeEvent) -> Dict[str, Any]:
        """Keep both versions strategy"""
        
        return {
            'success': True,
            'action': 'keep_both_versions',
            'reason': 'Maintaining multiple versions of conflicting information',
            'version_metadata': {
                'existing_version': existing_item.version,
                'new_version_id': str(uuid.uuid4())
            }
        }
    
    async def _reject_update_strategy(self, existing_item: KnowledgeItem,
                                    update_event: ChangeEvent) -> Dict[str, Any]:
        """Reject update strategy"""
        
        return {
            'success': True,
            'action': 'reject_update',
            'reason': 'Update rejected by policy'
        }

class UpdatePropagator:
    """Propagates knowledge updates to subscribers"""
    
    def __init__(self):
        self.subscriptions: Dict[str, UpdateSubscription] = {}
        self.notification_queue = deque()
        
        self.logger = logging.getLogger("UpdatePropagator")
    
    async def subscribe(self, subscription: UpdateSubscription) -> bool:
        """Subscribe to knowledge updates"""
        
        try:
            self.subscriptions[subscription.id] = subscription
            
            self.logger.info(f"Added subscription {subscription.id} for subscriber {subscription.subscriber_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add subscription: {e}")
            return False
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from knowledge updates"""
        
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            self.logger.info(f"Removed subscription {subscription_id}")
            return True
        
        return False
    
    async def propagate_update(self, item: KnowledgeItem, change_event: ChangeEvent) -> int:
        """Propagate update to relevant subscribers"""
        
        notifications_sent = 0
        
        for subscription_id, subscription in self.subscriptions.items():
            if not subscription.active:
                continue
            
            # Check if update matches subscription filters
            if await self._matches_subscription(subscription, item, change_event):
                try:
                    await self._send_notification(subscription, item, change_event)
                    notifications_sent += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to notify subscription {subscription_id}: {e}")
        
        self.logger.debug(f"Propagated update to {notifications_sent} subscribers")
        
        return notifications_sent
    
    async def _matches_subscription(self, subscription: UpdateSubscription,
                                  item: KnowledgeItem, change_event: ChangeEvent) -> bool:
        """Check if update matches subscription filters"""
        
        # Check change type filter
        if subscription.change_type_filters:
            if change_event.change_type not in subscription.change_type_filters:
                return False
        
        # Check priority threshold
        priority_levels = {
            UpdatePriority.BACKGROUND: 0,
            UpdatePriority.LOW: 1,
            UpdatePriority.MEDIUM: 2,
            UpdatePriority.HIGH: 3,
            UpdatePriority.CRITICAL: 4
        }
        
        if priority_levels[change_event.priority] < priority_levels[subscription.priority_threshold]:
            return False
        
        # Check item filters
        if subscription.item_filters:
            for filter_key, filter_value in subscription.item_filters.items():
                if filter_key == 'source_id':
                    if item.source_id != filter_value:
                        return False
                elif filter_key == 'metadata':
                    if not self._matches_metadata_filter(item.metadata, filter_value):
                        return False
        
        return True
    
    def _matches_metadata_filter(self, metadata: Dict[str, Any], filter_spec: Dict[str, Any]) -> bool:
        """Check if metadata matches filter specification"""
        
        for key, expected_value in filter_spec.items():
            if key not in metadata:
                return False
            
            if metadata[key] != expected_value:
                return False
        
        return True
    
    async def _send_notification(self, subscription: UpdateSubscription,
                               item: KnowledgeItem, change_event: ChangeEvent) -> None:
        """Send notification to subscriber"""
        
        notification_data = {
            'subscription_id': subscription.id,
            'subscriber_id': subscription.subscriber_id,
            'item': item.to_dict(),
            'change_event': {
                'id': change_event.id,
                'change_type': change_event.change_type.value,
                'item_id': change_event.item_id,
                'change_description': change_event.change_description,
                'priority': change_event.priority.value,
                'timestamp': change_event.timestamp.isoformat()
            },
            'notification_timestamp': datetime.now().isoformat()
        }
        
        # Use callback if available
        if subscription.callback:
            try:
                await subscription.callback(notification_data)
            except Exception as e:
                self.logger.error(f"Callback failed for subscription {subscription.id}: {e}")
        
        # Queue for HTTP notification if URL provided
        if subscription.callback_url:
            self.notification_queue.append({
                'url': subscription.callback_url,
                'data': notification_data
            })

class KnowledgeVersionManager:
    """Manages knowledge versioning and history"""
    
    def __init__(self):
        self.version_history: Dict[str, List[KnowledgeItem]] = defaultdict(list)
        self.change_log: List[ChangeEvent] = []
        
        self.logger = logging.getLogger("KnowledgeVersionManager")
    
    async def save_version(self, item: KnowledgeItem) -> str:
        """Save a version of a knowledge item"""
        
        # Create a deep copy for versioning
        version_item = KnowledgeItem(
            id=item.id,
            content=item.content,
            metadata=item.metadata.copy(),
            version=item.version,
            created_at=item.created_at,
            updated_at=item.updated_at,
            source_id=item.source_id,
            authority_score=item.authority_score,
            content_hash=item.content_hash
        )
        
        self.version_history[item.id].append(version_item)
        
        # Keep only last 10 versions to manage memory
        if len(self.version_history[item.id]) > 10:
            self.version_history[item.id] = self.version_history[item.id][-10:]
        
        self.logger.debug(f"Saved version {item.version} for item {item.id}")
        
        return item.version
    
    async def get_version(self, item_id: str, version: str) -> Optional[KnowledgeItem]:
        """Get specific version of a knowledge item"""
        
        if item_id not in self.version_history:
            return None
        
        for version_item in self.version_history[item_id]:
            if version_item.version == version:
                return version_item
        
        return None
    
    async def get_version_history(self, item_id: str) -> List[KnowledgeItem]:
        """Get version history for a knowledge item"""
        
        return self.version_history.get(item_id, [])
    
    async def rollback_to_version(self, item_id: str, target_version: str) -> Optional[KnowledgeItem]:
        """Rollback item to a specific version"""
        
        target_item = await self.get_version(item_id, target_version)
        
        if target_item:
            # Create new version with rollback
            rollback_item = KnowledgeItem(
                id=target_item.id,
                content=target_item.content,
                metadata=target_item.metadata.copy(),
                version=f"{target_item.version}_rollback_{int(time.time())}",
                created_at=target_item.created_at,
                updated_at=datetime.now(),
                source_id=target_item.source_id,
                authority_score=target_item.authority_score
            )
            
            await self.save_version(rollback_item)
            
            self.logger.info(f"Rolled back item {item_id} to version {target_version}")
            
            return rollback_item
        
        return None
    
    async def log_change(self, change_event: ChangeEvent) -> None:
        """Log change event"""
        
        self.change_log.append(change_event)
        
        # Keep only last 1000 change events
        if len(self.change_log) > 1000:
            self.change_log = self.change_log[-1000:]
    
    async def get_change_history(self, item_id: str) -> List[ChangeEvent]:
        """Get change history for a specific item"""
        
        return [event for event in self.change_log if event.item_id == item_id]

class DynamicKnowledgeUpdateSystem:
    """Complete dynamic knowledge update system"""
    
    def __init__(self):
        # Core components
        self.change_detector = ChangeDetector()
        self.conflict_resolver = ConflictResolver()
        self.update_propagator = UpdatePropagator()
        self.version_manager = KnowledgeVersionManager()
        
        # Knowledge storage
        self.knowledge_store: Dict[str, KnowledgeItem] = {}
        
        # Update processing
        self.update_queue = deque()
        self.processing_active = False
        
        # System configuration
        self.auto_conflict_resolution = True
        self.default_conflict_strategy = ConflictResolution.LATEST_WINS
        self.update_batch_size = 10
        
        # Statistics
        self.stats = {
            'updates_processed': 0,
            'conflicts_resolved': 0,
            'rollbacks_performed': 0,
            'notifications_sent': 0,
            'processing_time': 0.0
        }
        
        self.logger = logging.getLogger("DynamicKnowledgeUpdateSystem")
    
    async def initialize(self) -> None:
        """Initialize the dynamic update system"""
        
        # Start background processing
        asyncio.create_task(self._background_update_processor())
        
        self.logger.info("Dynamic knowledge update system initialized")
    
    async def register_knowledge_source(self, source_id: str, source_config: Dict[str, Any]) -> bool:
        """Register a knowledge source for monitoring"""
        
        return await self.change_detector.register_source(source_id, source_config)
    
    async def add_knowledge_item(self, item: KnowledgeItem) -> bool:
        """Add new knowledge item"""
        
        try:
            self.knowledge_store[item.id] = item
            await self.version_manager.save_version(item)
            
            self.logger.debug(f"Added knowledge item {item.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add knowledge item: {e}")
            return False
    
    async def update_knowledge_item(self, item_id: str, new_content: Any,
                                  source_id: str = "", priority: UpdatePriority = UpdatePriority.MEDIUM) -> bool:
        """Update existing knowledge item"""
        
        # Create update event
        change_event = ChangeEvent(
            id="",
            change_type=ChangeType.UPDATE,
            item_id=item_id,
            new_content=new_content,
            source_id=source_id,
            priority=priority,
            change_description="Manual update"
        )
        
        # Queue for processing
        self.update_queue.append(change_event)
        
        return True
    
    async def subscribe_to_updates(self, subscription: UpdateSubscription) -> bool:
        """Subscribe to knowledge updates"""
        
        return await self.update_propagator.subscribe(subscription)
    
    async def process_detected_changes(self) -> int:
        """Process changes detected from monitored sources"""
        
        changes = await self.change_detector.detect_all_changes()
        
        for change in changes:
            self.update_queue.append(change)
        
        self.logger.debug(f"Queued {len(changes)} detected changes")
        
        return len(changes)
    
    async def _background_update_processor(self) -> None:
        """Background task to process update queue"""
        
        self.processing_active = True
        
        while self.processing_active:
            try:
                if self.update_queue:
                    # Process batch of updates
                    batch_size = min(self.update_batch_size, len(self.update_queue))
                    batch = [self.update_queue.popleft() for _ in range(batch_size)]
                    
                    await self._process_update_batch(batch)
                
                # Sleep briefly before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Background processor error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _process_update_batch(self, batch: List[ChangeEvent]) -> None:
        """Process a batch of updates"""
        
        start_time = time.time()
        
        for change_event in batch:
            await self._process_single_update(change_event)
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] += processing_time
        
        self.logger.debug(f"Processed batch of {len(batch)} updates in {processing_time:.3f}s")
    
    async def _process_single_update(self, change_event: ChangeEvent) -> None:
        """Process a single update event"""
        
        try:
            change_event.status = UpdateStatus.PROCESSING
            
            # Handle different change types
            if change_event.change_type == ChangeType.CREATE:
                await self._handle_create(change_event)
            elif change_event.change_type == ChangeType.UPDATE:
                await self._handle_update(change_event)
            elif change_event.change_type == ChangeType.DELETE:
                await self._handle_delete(change_event)
            else:
                self.logger.warning(f"Unhandled change type: {change_event.change_type}")
                return
            
            # Log change
            await self.version_manager.log_change(change_event)
            
            self.stats['updates_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process update {change_event.id}: {e}")
            change_event.status = UpdateStatus.FAILED
            change_event.error_message = str(e)
    
    async def _handle_create(self, change_event: ChangeEvent) -> None:
        """Handle create change event"""
        
        if change_event.item_id in self.knowledge_store:
            # Item already exists, treat as update
            change_event.change_type = ChangeType.UPDATE
            await self._handle_update(change_event)
            return
        
        # Create new item
        new_item = KnowledgeItem(
            id=change_event.item_id,
            content=change_event.new_content,
            source_id=change_event.source_id
        )
        
        await self.add_knowledge_item(new_item)
        
        # Propagate to subscribers
        await self.update_propagator.propagate_update(new_item, change_event)
        
        change_event.status = UpdateStatus.APPLIED
        change_event.applied_at = datetime.now()
    
    async def _handle_update(self, change_event: ChangeEvent) -> None:
        """Handle update change event"""
        
        if change_event.item_id not in self.knowledge_store:
            # Item doesn't exist, treat as create
            change_event.change_type = ChangeType.CREATE
            await self._handle_create(change_event)
            return
        
        existing_item = self.knowledge_store[change_event.item_id]
        
        # Save current version before update
        await self.version_manager.save_version(existing_item)
        
        # Check for conflicts
        has_conflict = await self._detect_content_conflict(existing_item, change_event)
        
        if has_conflict and self.auto_conflict_resolution:
            # Resolve conflict automatically
            resolution = await self.conflict_resolver.resolve_conflict(
                existing_item, change_event, self.default_conflict_strategy
            )
            
            if resolution['success']:
                await self._apply_conflict_resolution(existing_item, change_event, resolution)
                self.stats['conflicts_resolved'] += 1
            else:
                change_event.status = UpdateStatus.CONFLICT
                return
        
        elif has_conflict:
            # Manual review required
            change_event.status = UpdateStatus.CONFLICT
            change_event.resolution_strategy = ConflictResolution.MANUAL_REVIEW
            return
        
        # Apply update
        change_event.old_content = existing_item.content
        
        if existing_item.update_content(change_event.new_content):
            # Content actually changed
            change_event.status = UpdateStatus.APPLIED
            change_event.applied_at = datetime.now()
            
            # Propagate to subscribers
            await self.update_propagator.propagate_update(existing_item, change_event)
        else:
            # No actual change
            change_event.status = UpdateStatus.APPLIED
            change_event.change_description += " (no content change)"
    
    async def _handle_delete(self, change_event: ChangeEvent) -> None:
        """Handle delete change event"""
        
        if change_event.item_id not in self.knowledge_store:
            change_event.status = UpdateStatus.FAILED
            change_event.error_message = "Item not found"
            return
        
        item = self.knowledge_store[change_event.item_id]
        
        # Save version before deletion
        await self.version_manager.save_version(item)
        
        # Remove from store
        del self.knowledge_store[change_event.item_id]
        
        # Propagate deletion
        await self.update_propagator.propagate_update(item, change_event)
        
        change_event.status = UpdateStatus.APPLIED
        change_event.applied_at = datetime.now()
    
    async def _detect_content_conflict(self, existing_item: KnowledgeItem,
                                     change_event: ChangeEvent) -> bool:
        """Detect if update conflicts with existing content"""
        
        # Simple conflict detection based on content hash
        if change_event.new_content is None:
            return False
        
        # Calculate hash of new content
        new_content_str = json.dumps(change_event.new_content, sort_keys=True, default=str)
        new_hash = hashlib.sha256(new_content_str.encode()).hexdigest()
        
        # Check if content is different
        return new_hash != existing_item.content_hash
    
    async def _apply_conflict_resolution(self, existing_item: KnowledgeItem,
                                       change_event: ChangeEvent,
                                       resolution: Dict[str, Any]) -> None:
        """Apply conflict resolution result"""
        
        action = resolution.get('action')
        
        if action == 'apply_update':
            # Apply the update as planned
            pass
        
        elif action == 'reject_update':
            change_event.status = UpdateStatus.FAILED
            change_event.error_message = resolution.get('reason', 'Update rejected by conflict resolution')
            return
        
        elif action == 'apply_merged_update':
            # Use merged content
            change_event.new_content = resolution.get('merged_content')
        
        elif action == 'keep_both_versions':
            # Create alternate version
            await self._create_alternate_version(existing_item, change_event, resolution)
            return
        
        elif action == 'require_manual_review':
            change_event.status = UpdateStatus.CONFLICT
            change_event.resolution_strategy = ConflictResolution.MANUAL_REVIEW
            return
    
    async def _create_alternate_version(self, existing_item: KnowledgeItem,
                                      change_event: ChangeEvent,
                                      resolution: Dict[str, Any]) -> None:
        """Create alternate version for keep_both strategy"""
        
        # Create new item ID for alternate version
        alternate_id = f"{existing_item.id}_alt_{int(time.time())}"
        
        alternate_item = KnowledgeItem(
            id=alternate_id,
            content=change_event.new_content,
            metadata=existing_item.metadata.copy(),
            source_id=change_event.source_id
        )
        
        alternate_item.metadata['alternate_of'] = existing_item.id
        alternate_item.metadata['conflict_resolution'] = 'keep_both'
        
        await self.add_knowledge_item(alternate_item)
        
        change_event.status = UpdateStatus.APPLIED
        change_event.applied_at = datetime.now()
        change_event.change_description += f" (created alternate version {alternate_id})"
    
    async def rollback_item(self, item_id: str, target_version: str) -> bool:
        """Rollback item to specific version"""
        
        try:
            rollback_item = await self.version_manager.rollback_to_version(item_id, target_version)
            
            if rollback_item:
                self.knowledge_store[item_id] = rollback_item
                
                # Create rollback change event
                rollback_event = ChangeEvent(
                    id="",
                    change_type=ChangeType.UPDATE,
                    item_id=item_id,
                    new_content=rollback_item.content,
                    change_description=f"Rollback to version {target_version}",
                    priority=UpdatePriority.HIGH
                )
                
                # Propagate rollback
                await self.update_propagator.propagate_update(rollback_item, rollback_event)
                
                self.stats['rollbacks_performed'] += 1
                
                self.logger.info(f"Rolled back item {item_id} to version {target_version}")
                return True
        
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        
        return {
            'update_statistics': self.stats,
            'queue_status': {
                'pending_updates': len(self.update_queue),
                'processing_active': self.processing_active
            },
            'knowledge_store': {
                'total_items': len(self.knowledge_store),
                'sources_monitored': len(self.change_detector.source_monitors)
            },
            'subscriptions': {
                'active_subscriptions': len([s for s in self.update_propagator.subscriptions.values() if s.active])
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_change_detection():
    """Demo: Change detection from various sources"""
    print("\nDEMO 1: CHANGE DETECTION")
    print("=" * 50)
    
    detector = ChangeDetector()
    
    # Register different types of sources
    sources = [
        {
            'id': 'api_source',
            'config': {
                'detection_strategy': 'polling',
                'url': 'https://api.example.com/data',
                'poll_interval': 60,
                'default_priority': UpdatePriority.MEDIUM
            }
        },
        {
            'id': 'webhook_source',
            'config': {
                'detection_strategy': 'webhook',
                'endpoint': '/webhook/updates',
                'default_priority': UpdatePriority.HIGH
            }
        },
        {
            'id': 'file_source',
            'config': {
                'detection_strategy': 'filesystem',
                'path': '/data/knowledge_files',
                'watch_patterns': ['*.json', '*.xml'],
                'default_priority': UpdatePriority.LOW
            }
        },
        {
            'id': 'db_source',
            'config': {
                'detection_strategy': 'database',
                'connection_string': 'postgresql://localhost/knowledge',
                'table': 'knowledge_items',
                'default_priority': UpdatePriority.MEDIUM
            }
        }
    ]
    
    print("Registering knowledge sources:")
    
    for source in sources:
        success = await detector.register_source(source['id'], source['config'])
        strategy = source['config']['detection_strategy']
        print(f"  ✓ {source['id']} ({strategy}): {'Registered' if success else 'Failed'}")
    
    print(f"\nDetecting changes across all sources:")
    
    # Simulate multiple detection cycles
    for cycle in range(3):
        print(f"\n--- Detection Cycle {cycle + 1} ---")
        
        all_changes = await detector.detect_all_changes()
        
        if all_changes:
            print(f"Detected {len(all_changes)} changes:")
            
            for change in all_changes:
                print(f"  - {change.change_type.value} on {change.item_id}")
                print(f"    Source: {change.source_id}")
                print(f"    Priority: {change.priority.value}")
                print(f"    Description: {change.change_description}")
        else:
            print("No changes detected")
        
        # Wait before next cycle
        await asyncio.sleep(1)

async def demo_conflict_resolution():
    """Demo: Conflict resolution strategies"""
    print("\nDEMO 2: CONFLICT RESOLUTION")
    print("=" * 50)
    
    resolver = ConflictResolver()
    
    # Create existing knowledge item
    existing_item = KnowledgeItem(
        id="product_123",
        content={
            'name': 'MacBook Pro',
            'price': 1299,
            'memory': '8GB',
            'storage': '256GB'
        },
        source_id="official_source",
        authority_score=0.9,
        updated_at=datetime.now() - timedelta(hours=2)
    )
    
    # Test different conflict scenarios
    conflict_scenarios = [
        {
            'name': 'Recent Price Update',
            'update_event': ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id="product_123",
                new_content={
                    'name': 'MacBook Pro',
                    'price': 1399,  # Price changed
                    'memory': '8GB',
                    'storage': '256GB'
                },
                source_id="retailer_source",
                timestamp=datetime.now()  # More recent
            ),
            'strategy': ConflictResolution.LATEST_WINS
        },
        {
            'name': 'Lower Authority Source',
            'update_event': ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id="product_123", 
                new_content={
                    'name': 'MacBook Pro',
                    'price': 999,  # Different price
                    'memory': '8GB',
                    'storage': '256GB'
                },
                source_id="unofficial_source",
                timestamp=datetime.now()
            ),
            'strategy': ConflictResolution.HIGHEST_AUTHORITY
        },
        {
            'name': 'Mergeable Content',
            'update_event': ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id="product_123",
                new_content={
                    'color': 'Space Gray',  # New field
                    'weight': '1.4kg'       # Another new field
                },
                source_id="specs_source",
                timestamp=datetime.now()
            ),
            'strategy': ConflictResolution.MERGE_VALUES
        },
        {
            'name': 'Significant Discrepancy',
            'update_event': ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id="product_123",
                new_content={
                    'name': 'Dell XPS',     # Completely different product
                    'price': 800,
                    'memory': '16GB',
                    'storage': '512GB'
                },
                source_id="error_source",
                timestamp=datetime.now()
            ),
            'strategy': ConflictResolution.MANUAL_REVIEW
        }
    ]
    
    print("Testing conflict resolution strategies:")
    
    for scenario in conflict_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Strategy: {scenario['strategy'].value}")
        print(f"Existing content: {existing_item.content}")
        print(f"Update content: {scenario['update_event'].new_content}")
        
        resolution = await resolver.resolve_conflict(
            existing_item,
            scenario['update_event'],
            scenario['strategy']
        )
        
        print(f"\nResolution Result:")
        print(f"  Success: {resolution['success']}")
        print(f"  Action: {resolution.get('action', 'unknown')}")
        print(f"  Reason: {resolution.get('reason', 'no reason provided')}")
        
        if 'merged_content' in resolution:
            print(f"  Merged Content: {resolution['merged_content']}")
        
        if 'error' in resolution:
            print(f"  Error: {resolution['error']}")

async def demo_update_propagation():
    """Demo: Update propagation to subscribers"""
    print("\nDEMO 3: UPDATE PROPAGATION")
    print("=" * 50)
    
    propagator = UpdatePropagator()
    
    # Define callback functions for different subscribers
    notifications_received = []
    
    async def product_team_callback(notification):
        notifications_received.append(('product_team', notification))
        print(f"  📧 Product Team notified: {notification['change_event']['change_type']} on {notification['item']['id']}")
    
    async def price_monitor_callback(notification):
        notifications_received.append(('price_monitor', notification))
        print(f"  💰 Price Monitor notified: Item {notification['item']['id']} updated")
    
    async def critical_alerts_callback(notification):
        notifications_received.append(('critical_alerts', notification))
        print(f"  🚨 Critical Alert: {notification['change_event']['priority']} update!")
    
    # Create subscriptions
    subscriptions = [
        UpdateSubscription(
            id="",
            subscriber_id="product_team",
            change_type_filters=[ChangeType.UPDATE, ChangeType.CREATE],
            priority_threshold=UpdatePriority.LOW,
            callback=product_team_callback,
            item_filters={'source_id': 'official_source'}
        ),
        UpdateSubscription(
            id="",
            subscriber_id="price_monitor",
            change_type_filters=[ChangeType.UPDATE],
            priority_threshold=UpdatePriority.MEDIUM,
            callback=price_monitor_callback
        ),
        UpdateSubscription(
            id="",
            subscriber_id="critical_alerts",
            priority_threshold=UpdatePriority.HIGH,
            callback=critical_alerts_callback
        )
    ]
    
    print("Setting up subscriptions:")
    
    for sub in subscriptions:
        success = await propagator.subscribe(sub)
        print(f"  ✓ {sub.subscriber_id}: {'Subscribed' if success else 'Failed'}")
    
    # Test propagation with different updates
    test_updates = [
        {
            'item': KnowledgeItem(
                id="product_456",
                content={'name': 'iPhone 15', 'price': 799},
                source_id="official_source"
            ),
            'event': ChangeEvent(
                id="",
                change_type=ChangeType.CREATE,
                item_id="product_456",
                priority=UpdatePriority.MEDIUM,
                change_description="New product added"
            )
        },
        {
            'item': KnowledgeItem(
                id="product_456", 
                content={'name': 'iPhone 15', 'price': 899},
                source_id="official_source"
            ),
            'event': ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id="product_456",
                priority=UpdatePriority.HIGH,
                change_description="Price increase detected"
            )
        },
        {
            'item': KnowledgeItem(
                id="product_789",
                content={'name': 'Random Product', 'price': 50},
                source_id="third_party_source"
            ),
            'event': ChangeEvent(
                id="",
                change_type=ChangeType.UPDATE,
                item_id="product_789",
                priority=UpdatePriority.LOW,
                change_description="Minor update"
            )
        },
        {
            'item': KnowledgeItem(
                id="security_alert",
                content={'alert': 'Critical security vulnerability found'},
                source_id="security_source"
            ),
            'event': ChangeEvent(
                id="",
                change_type=ChangeType.CREATE,
                item_id="security_alert",
                priority=UpdatePriority.CRITICAL,
                change_description="Security alert issued"
            )
        }
    ]
    
    print(f"\nTesting update propagation:")
    
    for i, update in enumerate(test_updates, 1):
        print(f"\n--- Update {i}: {update['event'].change_description} ---")
        
        notifications_sent = await propagator.propagate_update(
            update['item'], 
            update['event']
        )
        
        print(f"Notifications sent: {notifications_sent}")
    
    print(f"\nTotal notifications received: {len(notifications_received)}")
    
    # Show notification summary
    subscriber_counts = defaultdict(int)
    for subscriber, _ in notifications_received:
        subscriber_counts[subscriber] += 1
    
    print(f"Notifications by subscriber:")
    for subscriber, count in subscriber_counts.items():
        print(f"  {subscriber}: {count}")

async def demo_version_management():
    """Demo: Knowledge versioning and rollback"""
    print("\nDEMO 4: VERSION MANAGEMENT")
    print("=" * 50)
    
    version_manager = KnowledgeVersionManager()
    
    # Create initial knowledge item
    initial_item = KnowledgeItem(
        id="doc_123",
        content={
            'title': 'API Documentation',
            'version': '1.0',
            'endpoints': ['GET /api/v1/users', 'POST /api/v1/users']
        },
        source_id="docs_team"
    )
    
    print("Managing knowledge item versions:")
    print(f"Initial item: {initial_item.content}")
    
    # Save initial version
    version = await version_manager.save_version(initial_item)
    print(f"\n✓ Saved initial version: {version}")
    
    # Simulate several updates
    updates = [
        {
            'description': 'Added new endpoint',
            'content': {
                'title': 'API Documentation',
                'version': '1.1',
                'endpoints': ['GET /api/v1/users', 'POST /api/v1/users', 'PUT /api/v1/users/{id}']
            }
        },
        {
            'description': 'Updated version and added authentication',
            'content': {
                'title': 'API Documentation',
                'version': '1.2',
                'endpoints': ['GET /api/v1/users', 'POST /api/v1/users', 'PUT /api/v1/users/{id}'],
                'authentication': 'Bearer token required'
            }
        },
        {
            'description': 'Added rate limiting info',
            'content': {
                'title': 'API Documentation', 
                'version': '1.3',
                'endpoints': ['GET /api/v1/users', 'POST /api/v1/users', 'PUT /api/v1/users/{id}'],
                'authentication': 'Bearer token required',
                'rate_limit': '100 requests per minute'
            }
        }
    ]
    
    for i, update in enumerate(updates, 1):
        print(f"\n--- Update {i}: {update['description']} ---")
        
        # Update content
        if initial_item.update_content(update['content']):
            version = await version_manager.save_version(initial_item)
            print(f"✓ Saved version: {version}")
            print(f"  New content: {initial_item.content}")
        
        # Log change
        change_event = ChangeEvent(
            id="",
            change_type=ChangeType.UPDATE,
            item_id=initial_item.id,
            change_description=update['description'],
            old_content=None,  # Would be previous content in real scenario
            new_content=update['content']
        )
        
        await version_manager.log_change(change_event)
    
    # Show version history
    print(f"\nVersion History:")
    history = await version_manager.get_version_history(initial_item.id)
    
    for version_item in history:
        print(f"  Version {version_item.version} ({version_item.updated_at.strftime('%H:%M:%S')})")
        print(f"    Content: {version_item.content}")
    
    # Test rollback
    print(f"\nTesting rollback to version 1.1:")
    
    if len(history) >= 2:
        target_version = history[1].version  # Second version
        
        rollback_item = await version_manager.rollback_to_version(
            initial_item.id, 
            target_version
        )
        
        if rollback_item:
            print(f"✓ Rolled back successfully")
            print(f"  Rollback version: {rollback_item.version}")
            print(f"  Content: {rollback_item.content}")
        else:
            print("✗ Rollback failed")
    
    # Show change history
    print(f"\nChange History:")
    changes = await version_manager.get_change_history(initial_item.id)
    
    for change in changes:
        print(f"  {change.timestamp.strftime('%H:%M:%S')} - {change.change_type.value}")
        print(f"    Description: {change.change_description}")

async def demo_complete_dynamic_system():
    """Demo: Complete dynamic knowledge update system"""
    print("\nDEMO 5: COMPLETE DYNAMIC UPDATE SYSTEM")
    print("=" * 50)
    
    system = DynamicKnowledgeUpdateSystem()
    await system.initialize()
    
    # Set up notification tracking
    system_notifications = []
    
    async def system_notification_callback(notification):
        system_notifications.append(notification)
        print(f"  📢 System notification: {notification['change_event']['change_type']} on {notification['item']['id']}")
    
    # Register knowledge sources
    print("Setting up dynamic knowledge system:")
    
    sources = [
        ('product_catalog', {
            'detection_strategy': 'polling',
            'url': 'https://catalog.example.com',
            'default_priority': UpdatePriority.MEDIUM
        }),
        ('inventory_system', {
            'detection_strategy': 'webhook',
            'endpoint': '/webhook/inventory',
            'default_priority': UpdatePriority.HIGH
        })
    ]
    
    for source_id, config in sources:
        success = await system.register_knowledge_source(source_id, config)
        print(f"  ✓ Registered source: {source_id} ({'Success' if success else 'Failed'})")
    
    # Add initial knowledge items
    initial_items = [
        KnowledgeItem(
            id="product_001",
            content={'name': 'Widget A', 'price': 19.99, 'stock': 100},
            source_id="product_catalog"
        ),
        KnowledgeItem(
            id="product_002", 
            content={'name': 'Widget B', 'price': 29.99, 'stock': 50},
            source_id="product_catalog"
        ),
        KnowledgeItem(
            id="product_003",
            content={'name': 'Widget C', 'price': 39.99, 'stock': 25},
            source_id="inventory_system"
        )
    ]
    
    print(f"\nAdding initial knowledge items:")
    for item in initial_items:
        success = await system.add_knowledge_item(item)
        print(f"  ✓ Added item {item.id}: {item.content['name']}")
    
    # Set up subscriptions
    subscriptions = [
        UpdateSubscription(
            id="",
            subscriber_id="pricing_team",
            change_type_filters=[ChangeType.UPDATE],
            priority_threshold=UpdatePriority.MEDIUM,
            callback=system_notification_callback
        ),
        UpdateSubscription(
            id="",
            subscriber_id="inventory_alerts",
            priority_threshold=UpdatePriority.HIGH,
            callback=system_notification_callback,
            item_filters={'source_id': 'inventory_system'}
        )
    ]
    
    for sub in subscriptions:
        await system.subscribe_to_updates(sub)
        print(f"  ✓ Created subscription for {sub.subscriber_id}")
    
    # Simulate dynamic updates
    print(f"\nSimulating dynamic updates:")
    
    updates = [
        {
            'item_id': 'product_001',
            'new_content': {'name': 'Widget A', 'price': 24.99, 'stock': 100},
            'description': 'Price increase'
        },
        {
            'item_id': 'product_002',
            'new_content': {'name': 'Widget B', 'price': 29.99, 'stock': 5},
            'description': 'Low stock alert',
            'priority': UpdatePriority.HIGH
        },
        {
            'item_id': 'product_003',
            'new_content': {'name': 'Widget C', 'price': 39.99, 'stock': 0},
            'description': 'Out of stock',
            'priority': UpdatePriority.CRITICAL
        }
    ]
    
    for i, update in enumerate(updates, 1):
        print(f"\n--- Dynamic Update {i}: {update['description']} ---")
        
        success = await system.update_knowledge_item(
            update['item_id'],
            update['new_content'],
            priority=update.get('priority', UpdatePriority.MEDIUM)
        )
        
        print(f"  Update queued: {'Success' if success else 'Failed'}")
        
        # Wait for processing
        await asyncio.sleep(2)
    
    # Process detected changes
    print(f"\nProcessing detected changes from sources:")
    changes_detected = await system.process_detected_changes()
    print(f"  Detected {changes_detected} changes from monitored sources")
    
    # Wait for background processing
    await asyncio.sleep(3)
    
    # Show system statistics
    stats = system.get_statistics()
    
    print(f"\nSystem Statistics:")
    update_stats = stats['update_statistics']
    print(f"  Updates processed: {update_stats['updates_processed']}")
    print(f"  Conflicts resolved: {update_stats['conflicts_resolved']}")
    print(f"  Notifications sent: {len(system_notifications)}")
    
    queue_status = stats['queue_status']
    print(f"  Pending updates: {queue_status['pending_updates']}")
    print(f"  Processing active: {queue_status['processing_active']}")
    
    knowledge_store = stats['knowledge_store']
    print(f"  Total items: {knowledge_store['total_items']}")
    print(f"  Sources monitored: {knowledge_store['sources_monitored']}")
    
    print(f"\nNotifications received: {len(system_notifications)}")
    for notification in system_notifications:
        item_name = notification['item']['content'].get('name', 'Unknown')
        change_type = notification['change_event']['change_type']
        print(f"  - {change_type} on {item_name}")

async def main():
    """
    Demonstrate Dynamic Knowledge Updates for real-time knowledge evolution
    
    WHAT YOU'LL LEARN:
    ================
    1. How to detect changes across multiple knowledge sources
    2. How to resolve conflicts between conflicting updates
    3. How to propagate updates to interested subscribers
    4. How to manage knowledge versioning and rollback capabilities
    5. How to build complete dynamic update systems
    6. How to ensure consistency in rapidly changing knowledge bases
    
    REAL WORLD APPLICATIONS:
    =======================
    - Financial trading systems requiring real-time data updates
    - E-commerce platforms tracking inventory and pricing changes
    - News and media systems with rapidly evolving information
    - Medical systems with updated treatment protocols
    - Software documentation with continuous feature updates
    - IoT systems with sensor data and status changes
    """
    
    print("DYNAMIC KNOWLEDGE UPDATES DEMONSTRATION")
    print("Real-time knowledge evolution and synchronization!")
    
    await demo_change_detection()
    await demo_conflict_resolution()
    await demo_update_propagation()
    await demo_version_management()
    await demo_complete_dynamic_system()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Change detection monitors knowledge sources for updates")
    print("✓ Conflict resolution handles contradictory information intelligently")
    print("✓ Update propagation notifies subscribers of relevant changes")
    print("✓ Version management tracks evolution and enables rollbacks")
    print("✓ Complete systems provide real-time knowledge synchronization")
    print("✓ Dynamic updates enable current decision-making")
    print("\nTHE POWER OF DYNAMIC UPDATES:")
    print("- Keeps AI systems current with rapidly changing world")
    print("- Enables real-time decision support and recommendations")
    print("- Provides competitive advantage through superior information currency")
    print("- Critical for time-sensitive applications and domains")

if __name__ == "__main__":
    asyncio.run(main())
