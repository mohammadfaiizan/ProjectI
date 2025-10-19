#!/usr/bin/env python3
"""
Real-Time RAG Systems: Live Information Retrieval and Processing
==============================================================

WHAT IS THE PROBLEM?
==================
Traditional RAG systems work with static information:
- Cannot access live data feeds and real-time information
- Miss breaking news, market changes, and current events
- Cannot adapt to rapidly changing information landscapes
- Lack awareness of time-sensitive content and updates
- Cannot handle streaming data and continuous information flows
- Miss critical timing for decision-making and responses

Example: Financial Trading Complexity
STATIC RAG (Traditional):
- Query: "Should I buy Tesla stock now?"
- Retrieves: Historical analysis from last week
- Misses: Current market movement, breaking news, live sentiment
- Result: Outdated advice based on stale information

REAL WORLD EXAMPLE:
=================
How does Bloomberg Terminal work?

BLOOMBERG'S REAL-TIME SYSTEM:
1. LIVE DATA FEEDS: Real-time market data, news, and analytics
2. STREAMING UPDATES: Continuous information flow from multiple sources
3. EVENT DETECTION: Automatic identification of market-moving events
4. INSTANT ALERTS: Immediate notifications for relevant changes
5. DYNAMIC ANALYSIS: Real-time calculation of metrics and insights
6. TEMPORAL CONTEXT: Understanding of time-sensitive information
7. PREDICTIVE INSIGHTS: Forward-looking analysis based on current trends

BENEFITS OF REAL-TIME RAG:
- Access to current, up-to-the-minute information
- Ability to respond to breaking developments
- Time-sensitive decision support
- Competitive advantage through information freshness
- Dynamic adaptation to changing conditions
- Proactive alerts and notifications

THE REAL-TIME ADVANTAGE:
======================
TRADITIONAL RAG: Static knowledge base → Query → Historical response
REAL-TIME RAG: Live data streams + Dynamic knowledge + Current context → Timely response

REAL-TIME COMPONENTS:
===================
1. STREAMING DATA INGESTION: Continuous data feeds from multiple sources
2. LIVE INDEX UPDATES: Dynamic updating of knowledge base
3. TEMPORAL AWARENESS: Understanding of information freshness and relevance
4. EVENT DETECTION: Identification of significant changes and patterns
5. PRIORITY PROCESSING: Handling urgent vs. routine information
6. CACHE MANAGEMENT: Balancing speed with information accuracy
7. STREAMING RETRIEVAL: Real-time search and filtering

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI systems that work with current reality
- Provides time-sensitive decision support
- Powers applications requiring immediate response
- Critical for financial, news, and emergency response systems
- Enables proactive rather than reactive AI assistance
- Bridges the gap between AI and real-world dynamics
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import re
from datetime import datetime, timedelta
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataStreamType(Enum):
    """Types of real-time data streams"""
    NEWS_FEED = "news_feed"
    MARKET_DATA = "market_data"
    SOCIAL_MEDIA = "social_media"
    SENSOR_DATA = "sensor_data"
    EVENT_LOGS = "event_logs"
    USER_ACTIVITY = "user_activity"
    SYSTEM_METRICS = "system_metrics"
    WEATHER_DATA = "weather_data"

class UpdatePriority(Enum):
    """Priority levels for real-time updates"""
    CRITICAL = "critical"     # Immediate processing required
    HIGH = "high"            # Process within seconds
    MEDIUM = "medium"        # Process within minutes
    LOW = "low"              # Process when convenient

class ProcessingMode(Enum):
    """Modes for processing real-time data"""
    STREAMING = "streaming"   # Continuous processing
    BATCH = "batch"          # Periodic batch processing
    HYBRID = "hybrid"        # Combination of both
    ON_DEMAND = "on_demand"  # Process only when queried

@dataclass
class StreamingDataPoint:
    """Single data point from a real-time stream"""
    data_id: str
    stream_type: DataStreamType
    content: str
    
    # Temporal metadata
    timestamp: datetime
    sequence_number: int = 0
    
    # Content metadata
    source: str = ""
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    priority: UpdatePriority = UpdatePriority.MEDIUM
    
    # Processing state
    processed: bool = False
    indexed: bool = False
    embedding_computed: bool = False
    
    # Quality metrics
    reliability_score: float = 1.0
    freshness_score: float = 1.0
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if not self.data_id:
            self.data_id = str(uuid.uuid4())
        
        # Calculate freshness score based on age
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        self.freshness_score = max(0.0, 1.0 - (age_seconds / 3600))  # 1-hour decay
    
    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if data point is stale"""
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        return age_seconds > max_age_seconds

@dataclass
class EventAlert:
    """Alert for significant events detected in real-time data"""
    alert_id: str
    event_type: str
    description: str
    
    # Timing
    detection_time: datetime
    event_start_time: Optional[datetime] = None
    
    # Severity and impact
    severity: UpdatePriority = UpdatePriority.MEDIUM
    confidence: float = 0.0
    impact_score: float = 0.0
    
    # Related data
    related_data_points: List[str] = field(default_factory=list)
    affected_entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Action recommendations
    recommended_actions: List[str] = field(default_factory=list)
    urgent_response_needed: bool = False
    
    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())

class DataStreamGenerator:
    """Generates simulated real-time data streams"""
    
    def __init__(self, stream_type: DataStreamType):
        self.stream_type = stream_type
        self.sequence_counter = 0
        self.is_active = False
        
        # Stream configuration
        self.update_frequency = self._get_default_frequency()
        self.content_templates = self._get_content_templates()
        
        self.logger = logging.getLogger(f"StreamGenerator-{stream_type.value}")
    
    async def start_stream(self) -> AsyncGenerator[StreamingDataPoint, None]:
        """Start generating real-time data stream"""
        
        self.is_active = True
        self.logger.info(f"Started {self.stream_type.value} stream")
        
        try:
            while self.is_active:
                # Generate data point
                data_point = self._generate_data_point()
                yield data_point
                
                # Wait based on frequency
                await asyncio.sleep(self.update_frequency)
                
        except Exception as e:
            self.logger.error(f"Stream generation error: {e}")
        finally:
            self.is_active = False
    
    def stop_stream(self) -> None:
        """Stop the data stream"""
        self.is_active = False
        self.logger.info(f"Stopped {self.stream_type.value} stream")
    
    def _generate_data_point(self) -> StreamingDataPoint:
        """Generate a single data point"""
        
        self.sequence_counter += 1
        
        # Select content template
        template = self.content_templates[self.sequence_counter % len(self.content_templates)]
        
        # Generate content
        content = self._fill_template(template)
        
        # Determine priority (some data points are more urgent)
        priority = self._determine_priority()
        
        # Generate metadata
        entities = self._extract_entities(content)
        tags = self._generate_tags(content)
        
        data_point = StreamingDataPoint(
            data_id="",
            stream_type=self.stream_type,
            content=content,
            timestamp=datetime.now(),
            sequence_number=self.sequence_counter,
            source=f"{self.stream_type.value}_feed",
            tags=tags,
            entities=entities,
            priority=priority,
            reliability_score=self._calculate_reliability()
        )
        
        return data_point
    
    def _get_default_frequency(self) -> float:
        """Get default update frequency for stream type"""
        frequencies = {
            DataStreamType.NEWS_FEED: 30.0,      # Every 30 seconds
            DataStreamType.MARKET_DATA: 1.0,     # Every second
            DataStreamType.SOCIAL_MEDIA: 10.0,   # Every 10 seconds
            DataStreamType.SENSOR_DATA: 5.0,     # Every 5 seconds
            DataStreamType.EVENT_LOGS: 2.0,      # Every 2 seconds
            DataStreamType.USER_ACTIVITY: 15.0,  # Every 15 seconds
            DataStreamType.SYSTEM_METRICS: 60.0, # Every minute
            DataStreamType.WEATHER_DATA: 300.0   # Every 5 minutes
        }
        
        return frequencies.get(self.stream_type, 10.0)
    
    def _get_content_templates(self) -> List[str]:
        """Get content templates for stream type"""
        
        if self.stream_type == DataStreamType.NEWS_FEED:
            return [
                "Breaking: {company} announces {event_type} with {impact} implications",
                "Market update: {sector} sector shows {trend} movement due to {reason}",
                "Technology news: {tech_topic} advancement could {impact} {industry}",
                "Economic report: {metric} shows {change} for {time_period}",
                "Industry analysis: {company} {action} in response to {market_condition}"
            ]
        
        elif self.stream_type == DataStreamType.MARKET_DATA:
            return [
                "{symbol} price: ${price} ({change}%) Volume: {volume}",
                "Market index {index} at {value} ({movement} {percent}%)",
                "Currency update: {currency_pair} trading at {rate}",
                "Commodity report: {commodity} futures {trend} to ${price}",
                "Crypto update: {crypto} moves {direction} by {percent}%"
            ]
        
        elif self.stream_type == DataStreamType.SOCIAL_MEDIA:
            return [
                "Trending topic: {topic} sentiment {sentiment} with {engagement} engagement",
                "Viral content: {content_type} about {subject} reaches {reach} views",
                "Influencer update: {handle} posts about {topic} generating {response}",
                "Platform news: {platform} introduces {feature} affecting {users}",
                "Community discussion: {forum} debates {issue} with {activity} posts"
            ]
        
        else:
            return [
                f"{self.stream_type.value} update: Event {{{self.stream_type.value}_event}} occurred",
                f"{self.stream_type.value} data: Metric {{{self.stream_type.value}_metric}} changed",
                f"{self.stream_type.value} alert: Status {{{self.stream_type.value}_status}} detected"
            ]
    
    def _fill_template(self, template: str) -> str:
        """Fill template with generated values"""
        import random
        
        # Common replacement values
        replacements = {
            # Companies and entities
            'company': random.choice(['Apple', 'Google', 'Microsoft', 'Tesla', 'Amazon', 'Meta']),
            'symbol': random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']),
            'crypto': random.choice(['Bitcoin', 'Ethereum', 'Solana', 'Cardano']),
            
            # Events and actions
            'event_type': random.choice(['merger', 'acquisition', 'partnership', 'product launch', 'earnings report']),
            'action': random.choice(['expands', 'launches', 'acquires', 'partners with', 'invests in']),
            
            # Trends and movements
            'trend': random.choice(['upward', 'downward', 'sideways', 'volatile']),
            'movement': random.choice(['up', 'down', 'flat']),
            'direction': random.choice(['higher', 'lower']),
            'change': random.choice(['+2.5', '-1.8', '+0.9', '-3.2', '+1.1']),
            
            # Metrics and values
            'price': f"{random.uniform(50, 500):.2f}",
            'percent': f"{random.uniform(0.1, 5.0):.1f}",
            'volume': f"{random.randint(1, 100)}M",
            'value': f"{random.randint(15000, 35000)}",
            
            # Topics and subjects
            'topic': random.choice(['AI development', 'climate change', 'market volatility', 'tech innovation']),
            'subject': random.choice(['technology', 'finance', 'healthcare', 'environment']),
            'tech_topic': random.choice(['artificial intelligence', 'quantum computing', 'blockchain', 'robotics']),
            
            # Impact and sentiment
            'impact': random.choice(['positive', 'negative', 'mixed', 'significant']),
            'sentiment': random.choice(['positive', 'negative', 'neutral', 'mixed']),
            'engagement': random.choice(['high', 'moderate', 'low', 'viral'])
        }
        
        # Replace placeholders
        content = template
        for key, value in replacements.items():
            content = content.replace(f'{{{key}}}', str(value))
        
        return content
    
    def _determine_priority(self) -> UpdatePriority:
        """Determine priority of data point"""
        import random
        
        # Different streams have different priority distributions
        if self.stream_type == DataStreamType.MARKET_DATA:
            return random.choices([
                UpdatePriority.CRITICAL,
                UpdatePriority.HIGH,
                UpdatePriority.MEDIUM,
                UpdatePriority.LOW
            ], weights=[0.1, 0.3, 0.5, 0.1])[0]
        
        elif self.stream_type == DataStreamType.NEWS_FEED:
            return random.choices([
                UpdatePriority.CRITICAL,
                UpdatePriority.HIGH,
                UpdatePriority.MEDIUM,
                UpdatePriority.LOW
            ], weights=[0.05, 0.2, 0.6, 0.15])[0]
        
        else:
            return random.choices([
                UpdatePriority.HIGH,
                UpdatePriority.MEDIUM,
                UpdatePriority.LOW
            ], weights=[0.2, 0.6, 0.2])[0]
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from content"""
        import re
        
        entities = []
        
        # Company symbols
        symbols = re.findall(r'\b[A-Z]{2,5}\b', content)
        entities.extend(symbols)
        
        # Price mentions
        prices = re.findall(r'\$[\d,]+\.?\d*', content)
        entities.extend([p.replace('$', '').replace(',', '') for p in prices])
        
        # Percentage mentions
        percentages = re.findall(r'[\+\-]?\d+\.?\d*%', content)
        entities.extend(percentages)
        
        return entities[:5]  # Limit to 5 entities
    
    def _generate_tags(self, content: str) -> List[str]:
        """Generate tags for content"""
        tags = []
        
        content_lower = content.lower()
        
        # Stream-specific tags
        if self.stream_type == DataStreamType.MARKET_DATA:
            if 'price' in content_lower:
                tags.append('price_update')
            if 'volume' in content_lower:
                tags.append('volume_data')
            if any(word in content_lower for word in ['up', 'down', 'higher', 'lower']):
                tags.append('price_movement')
        
        elif self.stream_type == DataStreamType.NEWS_FEED:
            if 'breaking' in content_lower:
                tags.append('breaking_news')
            if any(word in content_lower for word in ['announces', 'launches', 'acquires']):
                tags.append('corporate_action')
            if 'market' in content_lower:
                tags.append('market_news')
        
        # General tags
        if any(word in content_lower for word in ['positive', 'up', 'gain', 'growth']):
            tags.append('positive')
        elif any(word in content_lower for word in ['negative', 'down', 'loss', 'decline']):
            tags.append('negative')
        
        return tags[:3]  # Limit to 3 tags
    
    def _calculate_reliability(self) -> float:
        """Calculate reliability score for data point"""
        import random
        
        # Different sources have different reliability
        base_reliability = {
            DataStreamType.MARKET_DATA: 0.95,
            DataStreamType.NEWS_FEED: 0.85,
            DataStreamType.SOCIAL_MEDIA: 0.70,
            DataStreamType.SENSOR_DATA: 0.90,
            DataStreamType.EVENT_LOGS: 0.95
        }.get(self.stream_type, 0.80)
        
        # Add some random variation
        variation = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_reliability + variation))

class EventDetectionEngine:
    """Detects significant events in real-time data streams"""
    
    def __init__(self):
        # Event detection patterns
        self.event_patterns = {
            'price_spike': {
                'pattern': r'[\+\-]([3-9]|[1-9]\d)\.?\d*%',
                'severity': UpdatePriority.HIGH,
                'description': 'Significant price movement detected'
            },
            'breaking_news': {
                'pattern': r'\b(?:breaking|urgent|alert)\b',
                'severity': UpdatePriority.CRITICAL,
                'description': 'Breaking news event detected'
            },
            'market_volatility': {
                'pattern': r'\b(?:volatile|volatility|turbulent|unstable)\b',
                'severity': UpdatePriority.HIGH,
                'description': 'Market volatility detected'
            },
            'corporate_action': {
                'pattern': r'\b(?:merger|acquisition|partnership|launch)\b',
                'severity': UpdatePriority.MEDIUM,
                'description': 'Corporate action detected'
            }
        }
        
        # Event tracking
        self.recent_events: deque = deque(maxlen=100)
        self.event_counts = defaultdict(int)
        
        self.logger = logging.getLogger("EventDetectionEngine")
    
    async def analyze_data_point(self, data_point: StreamingDataPoint) -> List[EventAlert]:
        """Analyze data point for significant events"""
        
        alerts = []
        
        try:
            # Check each event pattern
            for event_type, pattern_config in self.event_patterns.items():
                matches = re.findall(pattern_config['pattern'], data_point.content, re.IGNORECASE)
                
                if matches:
                    alert = EventAlert(
                        alert_id="",
                        event_type=event_type,
                        description=f"{pattern_config['description']}: {data_point.content[:100]}...",
                        detection_time=datetime.now(),
                        event_start_time=data_point.timestamp,
                        severity=pattern_config['severity'],
                        confidence=0.8,
                        impact_score=self._calculate_impact_score(event_type, data_point),
                        related_data_points=[data_point.data_id],
                        affected_entities=data_point.entities.copy(),
                        keywords=matches
                    )
                    
                    # Add recommendations
                    alert.recommended_actions = self._generate_recommendations(event_type, data_point)
                    alert.urgent_response_needed = pattern_config['severity'] == UpdatePriority.CRITICAL
                    
                    alerts.append(alert)
                    
                    # Track event
                    self.recent_events.append(alert)
                    self.event_counts[event_type] += 1
                    
                    self.logger.info(f"Event detected: {event_type} - {alert.description[:50]}...")
            
            # Check for compound events (multiple related events)
            compound_alerts = await self._detect_compound_events(data_point, alerts)
            alerts.extend(compound_alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Event analysis failed: {e}")
            return []
    
    def _calculate_impact_score(self, event_type: str, data_point: StreamingDataPoint) -> float:
        """Calculate impact score for event"""
        
        base_scores = {
            'price_spike': 0.8,
            'breaking_news': 0.9,
            'market_volatility': 0.7,
            'corporate_action': 0.6
        }
        
        base_score = base_scores.get(event_type, 0.5)
        
        # Adjust based on data point characteristics
        if data_point.priority == UpdatePriority.CRITICAL:
            base_score += 0.2
        elif data_point.priority == UpdatePriority.HIGH:
            base_score += 0.1
        
        # Adjust based on reliability
        base_score *= data_point.reliability_score
        
        return min(1.0, base_score)
    
    def _generate_recommendations(self, event_type: str, data_point: StreamingDataPoint) -> List[str]:
        """Generate action recommendations for event"""
        
        recommendations = {
            'price_spike': [
                "Monitor price movements closely",
                "Check for related news or announcements", 
                "Consider volatility impact on positions"
            ],
            'breaking_news': [
                "Review news details immediately",
                "Assess impact on relevant assets",
                "Prepare for increased market activity"
            ],
            'market_volatility': [
                "Increase risk monitoring",
                "Review portfolio exposure",
                "Consider hedging strategies"
            ],
            'corporate_action': [
                "Analyze impact on stock price",
                "Review company fundamentals",
                "Monitor market reaction"
            ]
        }
        
        return recommendations.get(event_type, ["Monitor situation closely"])
    
    async def _detect_compound_events(self, data_point: StreamingDataPoint, 
                                    current_alerts: List[EventAlert]) -> List[EventAlert]:
        """Detect compound events from multiple related signals"""
        
        compound_alerts = []
        
        # Check for multiple events in short time window
        recent_window = datetime.now() - timedelta(minutes=5)
        recent_events = [event for event in self.recent_events if event.detection_time >= recent_window]
        
        if len(recent_events) >= 3:
            # Multiple events suggest larger market event
            compound_alert = EventAlert(
                alert_id="",
                event_type="market_event_cluster",
                description=f"Multiple related events detected: {len(recent_events)} events in 5 minutes",
                detection_time=datetime.now(),
                severity=UpdatePriority.CRITICAL,
                confidence=0.85,
                impact_score=0.9,
                related_data_points=[event.alert_id for event in recent_events[-3:]],
                recommended_actions=[
                    "Investigate underlying cause",
                    "Prepare for market volatility",
                    "Monitor all related assets"
                ],
                urgent_response_needed=True
            )
            
            compound_alerts.append(compound_alert)
        
        return compound_alerts
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event detection statistics"""
        
        total_events = sum(self.event_counts.values())
        
        return {
            'total_events_detected': total_events,
            'events_by_type': dict(self.event_counts),
            'recent_events_count': len(self.recent_events),
            'detection_rate': total_events / max(len(self.recent_events), 1)
        }

class RealTimeIndexManager:
    """Manages real-time indexing and updating of document collections"""
    
    def __init__(self, max_documents: int = 10000):
        self.max_documents = max_documents
        
        # Document storage
        self.documents: Dict[str, StreamingDataPoint] = {}
        self.document_queue: deque = deque()
        
        # Indexing structures
        self.temporal_index: Dict[datetime, List[str]] = defaultdict(list)
        self.entity_index: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.priority_index: Dict[UpdatePriority, List[str]] = defaultdict(list)
        
        # Update tracking
        self.pending_updates: deque = deque()
        self.processing_stats = {
            'documents_added': 0,
            'documents_removed': 0,
            'updates_processed': 0,
            'processing_time_total': 0.0
        }
        
        self.logger = logging.getLogger("RealTimeIndexManager")
    
    async def add_document(self, data_point: StreamingDataPoint) -> None:
        """Add new document to real-time index"""
        
        start_time = time.time()
        
        try:
            # Check if we need to remove old documents
            await self._manage_document_capacity()
            
            # Store document
            self.documents[data_point.data_id] = data_point
            self.document_queue.append(data_point.data_id)
            
            # Update indexes
            await self._update_indexes(data_point)
            
            # Mark as indexed
            data_point.indexed = True
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processing_stats['documents_added'] += 1
            self.processing_stats['processing_time_total'] += processing_time
            
            self.logger.debug(f"Added document {data_point.data_id} in {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to add document {data_point.data_id}: {e}")
    
    async def update_document(self, data_point: StreamingDataPoint) -> None:
        """Update existing document in index"""
        
        if data_point.data_id in self.documents:
            # Remove from old indexes
            old_document = self.documents[data_point.data_id]
            await self._remove_from_indexes(old_document)
            
            # Update document
            self.documents[data_point.data_id] = data_point
            
            # Update indexes with new data
            await self._update_indexes(data_point)
            
            self.processing_stats['updates_processed'] += 1
        else:
            # Document doesn't exist, add it
            await self.add_document(data_point)
    
    async def search_realtime(self, query: str, 
                            time_range: Optional[Tuple[datetime, datetime]] = None,
                            priority_filter: Optional[List[UpdatePriority]] = None,
                            entity_filter: Optional[List[str]] = None,
                            top_k: int = 10) -> List[StreamingDataPoint]:
        """Search real-time index with temporal and contextual filters"""
        
        try:
            candidate_docs = set(self.documents.keys())
            
            # Apply time range filter
            if time_range:
                start_time, end_time = time_range
                time_filtered = set()
                
                for doc_time, doc_ids in self.temporal_index.items():
                    if start_time <= doc_time <= end_time:
                        time_filtered.update(doc_ids)
                
                candidate_docs &= time_filtered
            
            # Apply priority filter
            if priority_filter:
                priority_filtered = set()
                for priority in priority_filter:
                    priority_filtered.update(self.priority_index[priority])
                
                candidate_docs &= priority_filtered
            
            # Apply entity filter
            if entity_filter:
                entity_filtered = set()
                for entity in entity_filter:
                    entity_filtered.update(self.entity_index[entity])
                
                candidate_docs &= entity_filtered
            
            # Score and rank candidates
            scored_docs = []
            query_words = set(query.lower().split())
            
            for doc_id in candidate_docs:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    
                    # Calculate relevance score
                    content_words = set(doc.content.lower().split())
                    overlap = len(query_words & content_words)
                    content_score = overlap / max(len(query_words | content_words), 1)
                    
                    # Combine with freshness and reliability
                    total_score = (
                        content_score * 0.5 +
                        doc.freshness_score * 0.3 +
                        doc.reliability_score * 0.2
                    )
                    
                    scored_docs.append((total_score, doc))
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for _, doc in scored_docs[:top_k]]
            
        except Exception as e:
            self.logger.error(f"Real-time search failed: {e}")
            return []
    
    async def get_recent_documents(self, minutes: int = 10, 
                                 priority_filter: Optional[UpdatePriority] = None) -> List[StreamingDataPoint]:
        """Get documents from recent time window"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_docs = []
        for doc in self.documents.values():
            if doc.timestamp >= cutoff_time:
                if priority_filter is None or doc.priority == priority_filter:
                    recent_docs.append(doc)
        
        # Sort by timestamp (newest first)
        recent_docs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return recent_docs
    
    async def _manage_document_capacity(self) -> None:
        """Manage document storage capacity"""
        
        while len(self.documents) >= self.max_documents:
            # Remove oldest document
            oldest_doc_id = self.document_queue.popleft()
            
            if oldest_doc_id in self.documents:
                old_doc = self.documents[oldest_doc_id]
                await self._remove_from_indexes(old_doc)
                del self.documents[oldest_doc_id]
                
                self.processing_stats['documents_removed'] += 1
    
    async def _update_indexes(self, data_point: StreamingDataPoint) -> None:
        """Update all indexes with new document"""
        
        # Temporal index (round to minute for efficiency)
        minute_key = data_point.timestamp.replace(second=0, microsecond=0)
        self.temporal_index[minute_key].append(data_point.data_id)
        
        # Entity index
        for entity in data_point.entities:
            self.entity_index[entity].append(data_point.data_id)
        
        # Tag index
        for tag in data_point.tags:
            self.tag_index[tag].append(data_point.data_id)
        
        # Priority index
        self.priority_index[data_point.priority].append(data_point.data_id)
    
    async def _remove_from_indexes(self, data_point: StreamingDataPoint) -> None:
        """Remove document from all indexes"""
        
        # Remove from temporal index
        minute_key = data_point.timestamp.replace(second=0, microsecond=0)
        if minute_key in self.temporal_index:
            if data_point.data_id in self.temporal_index[minute_key]:
                self.temporal_index[minute_key].remove(data_point.data_id)
            
            # Clean up empty time buckets
            if not self.temporal_index[minute_key]:
                del self.temporal_index[minute_key]
        
        # Remove from entity index
        for entity in data_point.entities:
            if entity in self.entity_index:
                if data_point.data_id in self.entity_index[entity]:
                    self.entity_index[entity].remove(data_point.data_id)
                
                # Clean up empty entity buckets
                if not self.entity_index[entity]:
                    del self.entity_index[entity]
        
        # Remove from tag and priority indexes similarly
        for tag in data_point.tags:
            if tag in self.tag_index and data_point.data_id in self.tag_index[tag]:
                self.tag_index[tag].remove(data_point.data_id)
        
        if data_point.data_id in self.priority_index[data_point.priority]:
            self.priority_index[data_point.priority].remove(data_point.data_id)
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        
        avg_processing_time = (
            self.processing_stats['processing_time_total'] / 
            max(self.processing_stats['documents_added'], 1)
        )
        
        return {
            'total_documents': len(self.documents),
            'processing_stats': self.processing_stats,
            'average_processing_time': avg_processing_time,
            'index_sizes': {
                'temporal_buckets': len(self.temporal_index),
                'unique_entities': len(self.entity_index),
                'unique_tags': len(self.tag_index),
                'priority_levels': len(self.priority_index)
            }
        }

class RealTimeRAGSystem:
    """
    Complete Real-Time RAG System for live information processing
    
    EXAMPLE USAGE:
    =============
    # Create real-time RAG system
    rag = RealTimeRAGSystem()
    await rag.initialize()
    
    # Start data streams
    await rag.start_data_streams([
        DataStreamType.NEWS_FEED,
        DataStreamType.MARKET_DATA,
        DataStreamType.SOCIAL_MEDIA
    ])
    
    # Process real-time query
    result = await rag.realtime_query(
        "What's happening with Tesla stock right now?"
    )
    
    # Get live alerts
    alerts = await rag.get_active_alerts()
    
    print(f"Found {len(result['documents'])} real-time documents")
    print(f"Active alerts: {len(alerts)}")
    """
    
    def __init__(self):
        # Core components
        self.index_manager = RealTimeIndexManager()
        self.event_detector = EventDetectionEngine()
        
        # Stream management
        self.active_streams: Dict[DataStreamType, DataStreamGenerator] = {}
        self.stream_tasks: Dict[DataStreamType, asyncio.Task] = {}
        
        # Alert management
        self.active_alerts: List[EventAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # System state
        self.is_running = False
        self.processing_mode = ProcessingMode.STREAMING
        
        # Performance tracking
        self.system_stats = {
            'total_data_points_processed': 0,
            'total_queries_processed': 0,
            'average_query_time': 0.0,
            'alerts_generated': 0,
            'streams_active': 0
        }
        
        self.logger = logging.getLogger("RealTimeRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize real-time RAG system"""
        self.logger.info("Real-time RAG system initialized")
    
    async def start_data_streams(self, stream_types: List[DataStreamType]) -> None:
        """Start specified data streams"""
        
        self.is_running = True
        
        for stream_type in stream_types:
            if stream_type not in self.active_streams:
                # Create stream generator
                generator = DataStreamGenerator(stream_type)
                self.active_streams[stream_type] = generator
                
                # Start stream processing task
                task = asyncio.create_task(self._process_stream(generator))
                self.stream_tasks[stream_type] = task
                
                self.system_stats['streams_active'] += 1
                
                self.logger.info(f"Started {stream_type.value} stream")
        
        self.logger.info(f"Started {len(stream_types)} data streams")
    
    async def stop_data_streams(self) -> None:
        """Stop all data streams"""
        
        self.is_running = False
        
        # Stop all stream generators
        for generator in self.active_streams.values():
            generator.stop_stream()
        
        # Cancel all stream tasks
        for task in self.stream_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.stream_tasks:
            await asyncio.gather(*self.stream_tasks.values(), return_exceptions=True)
        
        # Clear state
        self.active_streams.clear()
        self.stream_tasks.clear()
        self.system_stats['streams_active'] = 0
        
        self.logger.info("Stopped all data streams")
    
    async def realtime_query(self, query: str, 
                           time_window_minutes: int = 60,
                           priority_filter: Optional[List[UpdatePriority]] = None,
                           include_events: bool = True) -> Dict[str, Any]:
        """Process real-time query with temporal context"""
        
        start_time = time.time()
        self.system_stats['total_queries_processed'] += 1
        
        try:
            # Define time range for search
            end_time = datetime.now()
            start_time_range = end_time - timedelta(minutes=time_window_minutes)
            
            # Search real-time index
            documents = await self.index_manager.search_realtime(
                query=query,
                time_range=(start_time_range, end_time),
                priority_filter=priority_filter,
                top_k=10
            )
            
            # Get recent events if requested
            related_events = []
            if include_events:
                related_events = [
                    alert for alert in self.active_alerts
                    if any(keyword.lower() in query.lower() for keyword in alert.keywords)
                ]
            
            # Calculate recency and relevance scores
            total_score = 0.0
            if documents:
                for doc in documents:
                    # Boost score for very recent documents
                    age_minutes = (datetime.now() - doc.timestamp).total_seconds() / 60
                    recency_boost = max(0.0, 1.0 - (age_minutes / 60))  # 1-hour decay
                    total_score += doc.reliability_score * (1.0 + recency_boost)
                
                total_score /= len(documents)
            
            processing_time = time.time() - start_time
            
            # Update performance statistics
            self._update_query_stats(processing_time)
            
            result = {
                'success': True,
                'query': query,
                'documents_found': len(documents),
                'documents': [self._doc_to_dict(doc) for doc in documents],
                'related_events': [self._alert_to_dict(alert) for alert in related_events],
                'time_window_minutes': time_window_minutes,
                'query_time': processing_time,
                'freshness_score': total_score,
                'search_metadata': {
                    'search_start_time': start_time_range.isoformat(),
                    'search_end_time': end_time.isoformat(),
                    'priority_filter': [p.value for p in priority_filter] if priority_filter else None,
                    'events_included': include_events
                }
            }
            
            self.logger.info(f"Real-time query processed: {len(documents)} docs, {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'query_time': time.time() - start_time
            }
    
    async def get_active_alerts(self, severity_filter: Optional[UpdatePriority] = None) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        
        filtered_alerts = self.active_alerts
        
        if severity_filter:
            filtered_alerts = [alert for alert in self.active_alerts if alert.severity == severity_filter]
        
        return [self._alert_to_dict(alert) for alert in filtered_alerts]
    
    async def get_live_summary(self, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get summary of live information"""
        
        # Get recent documents (last 30 minutes)
        recent_docs = await self.index_manager.get_recent_documents(minutes=30)
        
        # Filter by topics if specified
        if topics:
            topic_filtered = []
            for doc in recent_docs:
                doc_content = doc.content.lower()
                if any(topic.lower() in doc_content for topic in topics):
                    topic_filtered.append(doc)
            recent_docs = topic_filtered
        
        # Group by stream type
        by_stream = defaultdict(list)
        for doc in recent_docs:
            by_stream[doc.stream_type].append(doc)
        
        # Get priority distribution
        priority_counts = defaultdict(int)
        for doc in recent_docs:
            priority_counts[doc.priority] += 1
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.active_alerts
            if (datetime.now() - alert.detection_time).total_seconds() < 1800  # 30 minutes
        ]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'recent_activity': {
                'total_documents': len(recent_docs),
                'by_stream_type': {
                    stream_type.value: len(docs) for stream_type, docs in by_stream.items()
                },
                'by_priority': {
                    priority.value: count for priority, count in priority_counts.items()
                }
            },
            'recent_alerts': {
                'total_alerts': len(recent_alerts),
                'by_severity': {
                    severity.value: len([a for a in recent_alerts if a.severity == severity])
                    for severity in UpdatePriority
                }
            },
            'system_status': {
                'streams_active': len(self.active_streams),
                'processing_mode': self.processing_mode.value,
                'total_documents_indexed': len(self.index_manager.documents)
            }
        }
        
        return summary
    
    def add_alert_callback(self, callback: Callable[[EventAlert], None]) -> None:
        """Add callback function for new alerts"""
        self.alert_callbacks.append(callback)
    
    async def _process_stream(self, generator: DataStreamGenerator) -> None:
        """Process data from a stream generator"""
        
        try:
            async for data_point in generator.start_stream():
                if not self.is_running:
                    break
                
                # Add to index
                await self.index_manager.add_document(data_point)
                
                # Detect events
                alerts = await self.event_detector.analyze_data_point(data_point)
                
                # Process alerts
                for alert in alerts:
                    self.active_alerts.append(alert)
                    self.system_stats['alerts_generated'] += 1
                    
                    # Notify callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            self.logger.warning(f"Alert callback failed: {e}")
                
                # Clean up old alerts (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.active_alerts = [
                    alert for alert in self.active_alerts
                    if alert.detection_time >= cutoff_time
                ]
                
                self.system_stats['total_data_points_processed'] += 1
                
        except Exception as e:
            self.logger.error(f"Stream processing failed: {e}")
    
    def _update_query_stats(self, processing_time: float) -> None:
        """Update query processing statistics"""
        
        query_count = self.system_stats['total_queries_processed']
        current_avg = self.system_stats['average_query_time']
        
        self.system_stats['average_query_time'] = (
            (current_avg * (query_count - 1) + processing_time) / query_count
        )
    
    def _doc_to_dict(self, doc: StreamingDataPoint) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            'data_id': doc.data_id,
            'content': doc.content,
            'timestamp': doc.timestamp.isoformat(),
            'stream_type': doc.stream_type.value,
            'source': doc.source,
            'entities': doc.entities,
            'tags': doc.tags,
            'priority': doc.priority.value,
            'freshness_score': doc.freshness_score,
            'reliability_score': doc.reliability_score
        }
    
    def _alert_to_dict(self, alert: EventAlert) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': alert.alert_id,
            'event_type': alert.event_type,
            'description': alert.description,
            'detection_time': alert.detection_time.isoformat(),
            'severity': alert.severity.value,
            'confidence': alert.confidence,
            'impact_score': alert.impact_score,
            'affected_entities': alert.affected_entities,
            'recommended_actions': alert.recommended_actions,
            'urgent_response_needed': alert.urgent_response_needed
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        index_stats = self.index_manager.get_index_statistics()
        event_stats = self.event_detector.get_event_statistics()
        
        return {
            'system_stats': self.system_stats,
            'index_stats': index_stats,
            'event_stats': event_stats,
            'stream_status': {
                'active_streams': list(stream.value for stream in self.active_streams.keys()),
                'processing_mode': self.processing_mode.value,
                'system_running': self.is_running
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_data_streams():
    """Demo: Real-time data stream generation"""
    print("\nDEMO 1: REAL-TIME DATA STREAMS")
    print("=" * 50)
    
    # Create stream generators
    stream_types = [
        DataStreamType.NEWS_FEED,
        DataStreamType.MARKET_DATA,
        DataStreamType.SOCIAL_MEDIA
    ]
    
    print("Starting real-time data streams:")
    
    for stream_type in stream_types:
        print(f"\n--- {stream_type.value.replace('_', ' ').title()} Stream ---")
        
        generator = DataStreamGenerator(stream_type)
        
        print(f"Update frequency: {generator.update_frequency}s")
        print(f"Content templates: {len(generator.content_templates)}")
        
        # Generate a few sample data points
        print("Sample data points:")
        
        count = 0
        async for data_point in generator.start_stream():
            print(f"  {count + 1}. [{data_point.priority.value}] {data_point.content}")
            print(f"     Entities: {data_point.entities}")
            print(f"     Tags: {data_point.tags}")
            print(f"     Reliability: {data_point.reliability_score:.2f}")
            
            count += 1
            if count >= 3:  # Show first 3 data points
                generator.stop_stream()
                break
            
            await asyncio.sleep(0.1)  # Brief pause for demo

async def demo_event_detection():
    """Demo: Event detection in real-time data"""
    print("\nDEMO 2: EVENT DETECTION")
    print("=" * 50)
    
    detector = EventDetectionEngine()
    
    # Create test data points with different event types
    test_data_points = [
        StreamingDataPoint(
            data_id="test1",
            stream_type=DataStreamType.MARKET_DATA,
            content="TSLA price: $250.00 (+8.5%) Volume: 50M - Significant surge",
            timestamp=datetime.now(),
            entities=["TSLA", "8.5%", "250.00"],
            tags=["price_movement", "positive"]
        ),
        StreamingDataPoint(
            data_id="test2",
            stream_type=DataStreamType.NEWS_FEED,
            content="Breaking: Tesla announces major breakthrough in battery technology",
            timestamp=datetime.now(),
            entities=["Tesla", "battery"],
            tags=["breaking_news", "technology"]
        ),
        StreamingDataPoint(
            data_id="test3",
            stream_type=DataStreamType.MARKET_DATA,
            content="Market showing volatile conditions with multiple sectors affected",
            timestamp=datetime.now(),
            entities=["market", "sectors"],
            tags=["volatility", "market_wide"]
        )
    ]
    
    print("Analyzing data points for events:")
    
    for i, data_point in enumerate(test_data_points, 1):
        print(f"\n--- Data Point {i} ---")
        print(f"Content: {data_point.content}")
        
        alerts = await detector.analyze_data_point(data_point)
        
        if alerts:
            print(f"Events detected: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert.event_type}: {alert.description[:60]}...")
                print(f"    Severity: {alert.severity.value}, Confidence: {alert.confidence:.2f}")
                print(f"    Recommendations: {alert.recommended_actions[0] if alert.recommended_actions else 'None'}")
        else:
            print("No events detected")
    
    # Show detection statistics
    print(f"\nEvent Detection Statistics:")
    stats = detector.get_event_statistics()
    print(f"Total events detected: {stats['total_events_detected']}")
    print(f"Events by type: {stats['events_by_type']}")

async def demo_realtime_indexing():
    """Demo: Real-time indexing and search"""
    print("\nDEMO 3: REAL-TIME INDEXING")
    print("=" * 50)
    
    index_manager = RealTimeIndexManager(max_documents=50)
    
    # Add documents with different characteristics
    test_documents = []
    
    # Recent high-priority market data
    doc1 = StreamingDataPoint(
        data_id="market1",
        stream_type=DataStreamType.MARKET_DATA,
        content="Apple stock surges 5% on strong earnings report",
        timestamp=datetime.now() - timedelta(minutes=2),
        entities=["Apple", "5%", "earnings"],
        priority=UpdatePriority.HIGH,
        tags=["earnings", "positive"]
    )
    test_documents.append(doc1)
    
    # Breaking news
    doc2 = StreamingDataPoint(
        data_id="news1",
        stream_type=DataStreamType.NEWS_FEED,
        content="Federal Reserve announces interest rate decision",
        timestamp=datetime.now() - timedelta(minutes=10),
        entities=["Federal Reserve", "interest rate"],
        priority=UpdatePriority.CRITICAL,
        tags=["breaking_news", "monetary_policy"]
    )
    test_documents.append(doc2)
    
    # Social media sentiment
    doc3 = StreamingDataPoint(
        data_id="social1",
        stream_type=DataStreamType.SOCIAL_MEDIA,
        content="Tech sector sentiment remains positive amid AI developments",
        timestamp=datetime.now() - timedelta(minutes=30),
        entities=["tech sector", "AI"],
        priority=UpdatePriority.MEDIUM,
        tags=["sentiment", "technology"]
    )
    test_documents.append(doc3)
    
    print("Adding documents to real-time index:")
    
    for doc in test_documents:
        await index_manager.add_document(doc)
        print(f"  ✓ Added: {doc.content[:50]}...")
    
    # Test different search scenarios
    search_scenarios = [
        {
            'name': 'Recent Market Activity',
            'query': 'Apple stock earnings',
            'time_range': (datetime.now() - timedelta(minutes=5), datetime.now()),
            'priority_filter': [UpdatePriority.HIGH, UpdatePriority.CRITICAL]
        },
        {
            'name': 'Breaking News',
            'query': 'Federal Reserve interest rate',
            'priority_filter': [UpdatePriority.CRITICAL]
        },
        {
            'name': 'Technology Sentiment',
            'query': 'tech AI developments',
            'entity_filter': ['tech sector', 'AI']
        }
    ]
    
    print(f"\nTesting real-time search scenarios:")
    
    for scenario in search_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Query: {scenario['query']}")
        
        results = await index_manager.search_realtime(
            query=scenario['query'],
            time_range=scenario.get('time_range'),
            priority_filter=scenario.get('priority_filter'),
            entity_filter=scenario.get('entity_filter'),
            top_k=5
        )
        
        print(f"Results found: {len(results)}")
        for i, doc in enumerate(results, 1):
            age_minutes = (datetime.now() - doc.timestamp).total_seconds() / 60
            print(f"  {i}. {doc.content[:40]}... (age: {age_minutes:.1f}m, priority: {doc.priority.value})")
    
    # Show indexing statistics
    print(f"\nIndexing Statistics:")
    stats = index_manager.get_index_statistics()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Average processing time: {stats['average_processing_time']:.4f}s")
    print(f"Index sizes: {stats['index_sizes']}")

async def demo_realtime_rag_system():
    """Demo: Complete real-time RAG system"""
    print("\nDEMO 4: COMPLETE REAL-TIME RAG SYSTEM")
    print("=" * 50)
    
    rag_system = RealTimeRAGSystem()
    await rag_system.initialize()
    
    # Add alert callback for demonstration
    def alert_callback(alert: EventAlert):
        print(f"🚨 ALERT: {alert.event_type} - {alert.description[:50]}...")
    
    rag_system.add_alert_callback(alert_callback)
    
    # Start data streams
    stream_types = [DataStreamType.NEWS_FEED, DataStreamType.MARKET_DATA]
    
    print(f"Starting {len(stream_types)} data streams...")
    await rag_system.start_data_streams(stream_types)
    
    # Let streams run for a short time to collect data
    print("Collecting real-time data...")
    await asyncio.sleep(5)
    
    # Test real-time queries
    test_queries = [
        "What's happening with tech stocks right now?",
        "Any breaking news in the market?",
        "Recent developments in AI and technology"
    ]
    
    print(f"\nProcessing real-time queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {query}")
        
        result = await rag_system.realtime_query(
            query=query,
            time_window_minutes=10,
            include_events=True
        )
        
        if result['success']:
            print(f"Documents found: {result['documents_found']}")
            print(f"Related events: {len(result['related_events'])}")
            print(f"Freshness score: {result['freshness_score']:.2f}")
            print(f"Query time: {result['query_time']:.3f}s")
            
            if result['documents']:
                print("Top result:")
                top_doc = result['documents'][0]
                print(f"  Content: {top_doc['content'][:80]}...")
                print(f"  Source: {top_doc['stream_type']}, Priority: {top_doc['priority']}")
                print(f"  Freshness: {top_doc['freshness_score']:.2f}")
        
        else:
            print(f"Query failed: {result['error']}")
    
    # Get live summary
    print(f"\n--- Live System Summary ---")
    summary = await rag_system.get_live_summary()
    
    print(f"Recent activity (30 min):")
    activity = summary['recent_activity']
    print(f"  Total documents: {activity['total_documents']}")
    print(f"  By stream: {activity['by_stream_type']}")
    print(f"  By priority: {activity['by_priority']}")
    
    print(f"Alert summary:")
    alerts = summary['recent_alerts']
    print(f"  Total alerts: {alerts['total_alerts']}")
    print(f"  By severity: {alerts['by_severity']}")
    
    # Stop streams
    await rag_system.stop_data_streams()
    print("\nStopped all data streams")

async def demo_system_performance():
    """Demo: System performance and analytics"""
    print("\nDEMO 5: SYSTEM PERFORMANCE ANALYTICS")
    print("=" * 50)
    
    rag_system = RealTimeRAGSystem()
    await rag_system.initialize()
    
    # Start multiple streams for comprehensive testing
    all_stream_types = [
        DataStreamType.NEWS_FEED,
        DataStreamType.MARKET_DATA,
        DataStreamType.SOCIAL_MEDIA
    ]
    
    print("Starting comprehensive real-time processing...")
    await rag_system.start_data_streams(all_stream_types)
    
    # Process multiple queries concurrently
    concurrent_queries = [
        "Market volatility and price movements",
        "Breaking news and announcements",
        "Social media sentiment and trends",
        "Technology developments and innovations",
        "Economic indicators and analysis"
    ]
    
    print(f"Processing {len(concurrent_queries)} concurrent queries...")
    
    # Run queries concurrently to test system load
    query_tasks = []
    for query in concurrent_queries:
        task = asyncio.create_task(
            rag_system.realtime_query(query, time_window_minutes=15)
        )
        query_tasks.append(task)
    
    # Let system run and collect data
    await asyncio.sleep(3)
    
    # Complete all queries
    query_results = await asyncio.gather(*query_tasks)
    
    # Analyze performance
    successful_queries = [r for r in query_results if r['success']]
    
    print(f"\nPerformance Analysis:")
    print(f"Query success rate: {len(successful_queries)}/{len(query_results)} ({len(successful_queries)/len(query_results)*100:.1f}%)")
    
    if successful_queries:
        avg_query_time = sum(r['query_time'] for r in successful_queries) / len(successful_queries)
        avg_docs_found = sum(r['documents_found'] for r in successful_queries) / len(successful_queries)
        avg_freshness = sum(r['freshness_score'] for r in successful_queries) / len(successful_queries)
        
        print(f"Average query time: {avg_query_time:.3f}s")
        print(f"Average documents found: {avg_docs_found:.1f}")
        print(f"Average freshness score: {avg_freshness:.2f}")
    
    # Get comprehensive system statistics
    stats = rag_system.get_system_statistics()
    
    print(f"\nSystem Statistics:")
    system_stats = stats['system_stats']
    print(f"Data points processed: {system_stats['total_data_points_processed']}")
    print(f"Queries processed: {system_stats['total_queries_processed']}")
    print(f"Alerts generated: {system_stats['alerts_generated']}")
    print(f"Active streams: {system_stats['streams_active']}")
    
    print(f"\nIndex Performance:")
    index_stats = stats['index_stats']
    print(f"Total documents indexed: {index_stats['total_documents']}")
    print(f"Average indexing time: {index_stats['average_processing_time']:.4f}s")
    print(f"Documents added: {index_stats['processing_stats']['documents_added']}")
    print(f"Updates processed: {index_stats['processing_stats']['updates_processed']}")
    
    print(f"\nEvent Detection:")
    event_stats = stats['event_stats']
    print(f"Total events detected: {event_stats['total_events_detected']}")
    print(f"Events by type: {event_stats['events_by_type']}")
    
    # Stop system
    await rag_system.stop_data_streams()
    print("\nSystem performance analysis complete")

async def main():
    """
    Demonstrate Real-Time RAG Systems for live information processing
    
    WHAT YOU'LL LEARN:
    ================
    1. How to build streaming data ingestion systems
    2. How to implement real-time event detection and alerting
    3. How to create dynamic indexing for live data
    4. How to process time-sensitive queries effectively
    5. How to build production-ready real-time AI systems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Financial trading and market monitoring systems
    - News aggregation and breaking news detection
    - Social media monitoring and sentiment tracking
    - IoT sensor data processing and alerting
    - Emergency response and crisis management
    - Business intelligence with live data feeds
    """
    
    print("REAL-TIME RAG SYSTEMS DEMONSTRATION")
    print("Building live information processing systems with streaming data!")
    
    await demo_data_streams()
    await demo_event_detection()
    await demo_realtime_indexing()
    await demo_realtime_rag_system()
    await demo_system_performance()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Real-time data streams enable live information processing")
    print("✓ Event detection identifies significant changes and patterns")
    print("✓ Dynamic indexing supports immediate search and retrieval")
    print("✓ Temporal awareness provides time-sensitive decision support")
    print("✓ Complete systems handle streaming data at production scale")
    print("✓ Performance monitoring ensures reliable real-time operation")
    print("\nTHE POWER OF REAL-TIME RAG:")
    print("- Enables AI systems that work with current reality")
    print("- Provides immediate response to breaking developments")
    print("- Supports time-critical decision making and alerts")
    print("- Powers next-generation live intelligence systems")

if __name__ == "__main__":
    asyncio.run(main())
