#!/usr/bin/env python3
"""
Episodic Memory System: Storing and Retrieving Personal Experiences
=================================================================

WHAT IS THE PROBLEM?
==================
AI agents lack the ability to remember personal experiences and episodes:
- Cannot recall specific interactions or events that happened in the past
- No memory of when, where, and how things occurred in previous sessions
- Unable to learn from past experiences or build upon previous interactions
- Missing autobiographical memory that provides personal context and continuity
- Cannot distinguish between general knowledge and personal experiences
- Lack of temporal context makes it impossible to reference "last week" or "yesterday"

Example: Personal AI Assistant Without Episodic Memory
WITHOUT EPISODIC MEMORY (Traditional):
- User: "Remember when we discussed that Python project last Tuesday?"
- AI: "I don't have any memory of previous conversations"
- User: "What was the outcome of the debugging session we had?"
- AI: "I cannot recall any debugging sessions"
- User: "Have we talked about machine learning before?"
- AI: "I don't have access to our conversation history"
- Result: No continuity, no personal relationship, frustrating user experience

REAL WORLD EXAMPLE:
=================
How does human episodic memory work in daily life?

HUMAN EPISODIC MEMORY:
1. ENCODING: Personal experiences are stored with rich contextual details
2. SPATIAL CONTEXT: Remember where events happened ("in the office", "at home")
3. TEMPORAL CONTEXT: Remember when events occurred ("last Tuesday", "this morning")
4. EMOTIONAL CONTEXT: Remember how events felt ("exciting", "frustrating")
5. SOCIAL CONTEXT: Remember who was involved ("with Sarah", "during the meeting")
6. CAUSAL CONTEXT: Remember why events happened and their consequences
7. AUTOBIOGRAPHICAL TIMELINE: Personal life story with connected episodes

BENEFITS OF EPISODIC MEMORY:
- Enables personalized conversations with reference to shared history
- Supports learning from past mistakes and building on successes
- Creates sense of continuity and personal relationship
- Allows understanding of personal preferences and patterns
- Enables contextual reasoning about past, present, and future
- Provides foundation for autobiographical identity and growth

THE MEMORY ADVANTAGE:
===================
NO EPISODES: Each interaction isolated → No personal connection
WITH EPISODES: Connected experiences → Rich personal relationship

EPISODIC MEMORY COMPONENTS:
==========================
1. EPISODE ENCODING: Capturing experiences with rich contextual metadata
2. TEMPORAL INDEXING: Organizing episodes by time and sequence
3. SPATIAL CONTEXT: Recording location and environmental information
4. EMOTIONAL TAGGING: Capturing emotional states and responses
5. SOCIAL CONTEXT: Recording participants and social dynamics
6. RETRIEVAL CUES: Multiple pathways to find relevant episodes
7. AUTOBIOGRAPHICAL TIMELINE: Connected narrative of personal experiences

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI agents to form personal relationships and maintain continuity
- Critical for personalized AI assistants, tutors, and companions
- Foundation for AI systems that learn and grow through experience
- Supports contextual understanding and reference to shared history
- Creates more engaging and human-like AI interactions
- Enables AI to understand personal growth and development patterns
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import sqlite3
import numpy as np
from contextlib import contextmanager
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EpisodeType(Enum):
    """Types of episodic memories"""
    CONVERSATION = "conversation"
    TASK_COMPLETION = "task_completion"
    LEARNING_EVENT = "learning_event"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    INTERACTION = "interaction"
    MILESTONE = "milestone"
    ERROR_RECOVERY = "error_recovery"

class EmotionalState(Enum):
    """Emotional states during episodes"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    CONFUSED = "confused"
    CONFIDENT = "confident"

class EpisodeSignificance(Enum):
    """Significance levels for episodes"""
    ROUTINE = 1
    NOTABLE = 2
    IMPORTANT = 3
    SIGNIFICANT = 4
    MILESTONE = 5

@dataclass
class SpatialContext:
    """Spatial context information"""
    location: str = ""
    environment: str = ""
    coordinates: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location': self.location,
            'environment': self.environment,
            'coordinates': self.coordinates
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpatialContext':
        return cls(
            location=data.get('location', ''),
            environment=data.get('environment', ''),
            coordinates=data.get('coordinates')
        )

@dataclass
class TemporalContext:
    """Temporal context information"""
    timestamp: datetime
    duration: Optional[timedelta] = None
    time_of_day: str = ""
    day_of_week: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration.total_seconds() if self.duration else None,
            'time_of_day': self.time_of_day,
            'day_of_week': self.day_of_week
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalContext':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            duration=timedelta(seconds=data['duration']) if data.get('duration') else None,
            time_of_day=data.get('time_of_day', ''),
            day_of_week=data.get('day_of_week', '')
        )

@dataclass
class SocialContext:
    """Social context information"""
    participants: List[str] = field(default_factory=list)
    participant_roles: Dict[str, str] = field(default_factory=dict)
    social_setting: str = ""
    interaction_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'participants': self.participants,
            'participant_roles': self.participant_roles,
            'social_setting': self.social_setting,
            'interaction_type': self.interaction_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialContext':
        return cls(
            participants=data.get('participants', []),
            participant_roles=data.get('participant_roles', {}),
            social_setting=data.get('social_setting', ''),
            interaction_type=data.get('interaction_type', '')
        )

@dataclass
class EmotionalContext:
    """Emotional context information"""
    primary_emotion: EmotionalState = EmotionalState.NEUTRAL
    emotion_intensity: float = 0.5  # 0.0 to 1.0
    emotional_tags: List[str] = field(default_factory=list)
    mood_before: str = ""
    mood_after: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_emotion': self.primary_emotion.value,
            'emotion_intensity': self.emotion_intensity,
            'emotional_tags': self.emotional_tags,
            'mood_before': self.mood_before,
            'mood_after': self.mood_after
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalContext':
        return cls(
            primary_emotion=EmotionalState(data.get('primary_emotion', EmotionalState.NEUTRAL.value)),
            emotion_intensity=data.get('emotion_intensity', 0.5),
            emotional_tags=data.get('emotional_tags', []),
            mood_before=data.get('mood_before', ''),
            mood_after=data.get('mood_after', '')
        )

@dataclass
class Episode:
    """Represents an episodic memory"""
    
    id: str
    episode_type: EpisodeType
    title: str
    description: str
    
    # Content
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    temporal_context: TemporalContext = field(default_factory=lambda: TemporalContext(datetime.now()))
    spatial_context: SpatialContext = field(default_factory=SpatialContext)
    social_context: SocialContext = field(default_factory=SocialContext)
    emotional_context: EmotionalContext = field(default_factory=EmotionalContext)
    
    # Metadata
    significance: EpisodeSignificance = EpisodeSignificance.ROUTINE
    tags: Set[str] = field(default_factory=set)
    
    # Outcomes and consequences
    outcomes: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    follow_up_actions: List[str] = field(default_factory=list)
    
    # Connections
    related_episodes: Set[str] = field(default_factory=set)
    caused_by: Optional[str] = None
    led_to: Optional[str] = None
    
    # Retrieval and recall
    recall_count: int = 0
    last_recalled: Optional[datetime] = None
    
    # Versioning
    version: int = 1
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def recall(self) -> None:
        """Record episode recall"""
        self.recall_count += 1
        self.last_recalled = datetime.now()
    
    def add_outcome(self, outcome: str) -> None:
        """Add an outcome to the episode"""
        if outcome not in self.outcomes:
            self.outcomes.append(outcome)
            self.last_updated = datetime.now()
    
    def add_lesson(self, lesson: str) -> None:
        """Add a lesson learned"""
        if lesson not in self.lessons_learned:
            self.lessons_learned.append(lesson)
            self.last_updated = datetime.now()
    
    def update_content(self, new_content: Dict[str, Any]) -> None:
        """Update episode content"""
        self.content.update(new_content)
        self.version += 1
        self.last_updated = datetime.now()
    
    def calculate_recency_score(self) -> float:
        """Calculate recency score (0.0 to 1.0)"""
        now = datetime.now()
        time_diff = now - self.temporal_context.timestamp
        
        # Decay function: more recent episodes have higher scores
        days_old = time_diff.total_seconds() / (24 * 3600)
        recency_score = max(0.0, 1.0 - (days_old / 365.0))  # Decay over a year
        
        return recency_score
    
    def calculate_relevance_score(self, query_tags: Set[str], 
                                query_participants: List[str],
                                query_time_range: Optional[Tuple[datetime, datetime]] = None) -> float:
        """Calculate relevance to a query"""
        
        scores = []
        
        # Tag overlap
        if query_tags and self.tags:
            tag_overlap = len(query_tags.intersection(self.tags))
            tag_score = tag_overlap / len(query_tags.union(self.tags))
            scores.append(tag_score * 0.4)
        
        # Participant overlap
        if query_participants and self.social_context.participants:
            participant_overlap = len(set(query_participants).intersection(set(self.social_context.participants)))
            participant_score = participant_overlap / max(len(query_participants), len(self.social_context.participants))
            scores.append(participant_score * 0.3)
        
        # Temporal relevance
        if query_time_range:
            start_time, end_time = query_time_range
            if start_time <= self.temporal_context.timestamp <= end_time:
                scores.append(1.0 * 0.2)
            else:
                scores.append(0.0)
        else:
            # Use recency as temporal relevance
            scores.append(self.calculate_recency_score() * 0.2)
        
        # Significance boost
        significance_score = self.significance.value / EpisodeSignificance.MILESTONE.value
        scores.append(significance_score * 0.1)
        
        return sum(scores) if scores else 0.0

class EpisodeStorage:
    """Storage backend for episodic memories"""
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger("EpisodeStorage")
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create episodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    episode_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    temporal_context_json TEXT NOT NULL,
                    spatial_context_json TEXT,
                    social_context_json TEXT,
                    emotional_context_json TEXT,
                    significance INTEGER NOT NULL,
                    tags_json TEXT,
                    outcomes_json TEXT,
                    lessons_learned_json TEXT,
                    follow_up_actions_json TEXT,
                    related_episodes_json TEXT,
                    caused_by TEXT,
                    led_to TEXT,
                    recall_count INTEGER DEFAULT 0,
                    last_recalled TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    last_updated TIMESTAMP NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_type ON episodes(episode_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(json_extract(temporal_context_json, "$.timestamp"))')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_significance ON episodes(significance)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_updated ON episodes(last_updated)')
            
            conn.commit()
    
    async def store_episode(self, episode: Episode) -> bool:
        """Store an episode"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO episodes (
                            id, episode_type, title, description, content_json,
                            temporal_context_json, spatial_context_json, social_context_json,
                            emotional_context_json, significance, tags_json, outcomes_json,
                            lessons_learned_json, follow_up_actions_json, related_episodes_json,
                            caused_by, led_to, recall_count, last_recalled, version, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        episode.id,
                        episode.episode_type.value,
                        episode.title,
                        episode.description,
                        json.dumps(episode.content),
                        json.dumps(episode.temporal_context.to_dict()),
                        json.dumps(episode.spatial_context.to_dict()),
                        json.dumps(episode.social_context.to_dict()),
                        json.dumps(episode.emotional_context.to_dict()),
                        episode.significance.value,
                        json.dumps(list(episode.tags)),
                        json.dumps(episode.outcomes),
                        json.dumps(episode.lessons_learned),
                        json.dumps(episode.follow_up_actions),
                        json.dumps(list(episode.related_episodes)),
                        episode.caused_by,
                        episode.led_to,
                        episode.recall_count,
                        episode.last_recalled.isoformat() if episode.last_recalled else None,
                        episode.version,
                        episode.last_updated.isoformat()
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to store episode {episode.id}: {e}")
            return False
    
    async def retrieve_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT * FROM episodes WHERE id = ?', (episode_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_episode(row)
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to retrieve episode {episode_id}: {e}")
            return None
    
    async def search_episodes(self, query: str = "", episode_types: List[EpisodeType] = None,
                            tags: Set[str] = None, participants: List[str] = None,
                            time_range: Optional[Tuple[datetime, datetime]] = None,
                            significance_min: EpisodeSignificance = None,
                            limit: int = 20) -> List[Episode]:
        """Search for episodes"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Build search query
                    conditions = []
                    params = []
                    
                    if query:
                        conditions.append('(title LIKE ? OR description LIKE ? OR content_json LIKE ?)')
                        params.extend([f'%{query}%', f'%{query}%', f'%{query}%'])
                    
                    if episode_types:
                        type_placeholders = ','.join('?' * len(episode_types))
                        conditions.append(f'episode_type IN ({type_placeholders})')
                        params.extend([et.value for et in episode_types])
                    
                    if significance_min:
                        conditions.append('significance >= ?')
                        params.append(significance_min.value)
                    
                    if time_range:
                        start_time, end_time = time_range
                        conditions.append('''
                            datetime(json_extract(temporal_context_json, "$.timestamp")) 
                            BETWEEN ? AND ?
                        ''')
                        params.extend([start_time.isoformat(), end_time.isoformat()])
                    
                    where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
                    
                    sql = f'''
                        SELECT * FROM episodes
                        {where_clause}
                        ORDER BY json_extract(temporal_context_json, "$.timestamp") DESC
                        LIMIT ?
                    '''
                    params.append(limit)
                    
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    episodes = [self._row_to_episode(row) for row in rows]
                    
                    # Apply additional filters
                    if tags:
                        episodes = [e for e in episodes if tags.intersection(e.tags)]
                    
                    if participants:
                        episodes = [e for e in episodes 
                                   if any(p in e.social_context.participants for p in participants)]
                    
                    return episodes
                    
        except Exception as e:
            self.logger.error(f"Failed to search episodes: {e}")
            return []
    
    def _row_to_episode(self, row: Tuple) -> Episode:
        """Convert database row to Episode object"""
        
        (id, episode_type, title, description, content_json, temporal_context_json,
         spatial_context_json, social_context_json, emotional_context_json,
         significance, tags_json, outcomes_json, lessons_learned_json,
         follow_up_actions_json, related_episodes_json, caused_by, led_to,
         recall_count, last_recalled, version, last_updated) = row
        
        episode = Episode(
            id=id,
            episode_type=EpisodeType(episode_type),
            title=title,
            description=description,
            content=json.loads(content_json),
            temporal_context=TemporalContext.from_dict(json.loads(temporal_context_json)),
            spatial_context=SpatialContext.from_dict(json.loads(spatial_context_json or '{}')),
            social_context=SocialContext.from_dict(json.loads(social_context_json or '{}')),
            emotional_context=EmotionalContext.from_dict(json.loads(emotional_context_json or '{}')),
            significance=EpisodeSignificance(significance),
            tags=set(json.loads(tags_json or '[]')),
            outcomes=json.loads(outcomes_json or '[]'),
            lessons_learned=json.loads(lessons_learned_json or '[]'),
            follow_up_actions=json.loads(follow_up_actions_json or '[]'),
            related_episodes=set(json.loads(related_episodes_json or '[]')),
            caused_by=caused_by,
            led_to=led_to,
            recall_count=recall_count,
            last_recalled=datetime.fromisoformat(last_recalled) if last_recalled else None,
            version=version,
            last_updated=datetime.fromisoformat(last_updated)
        )
        
        return episode
    
    async def get_timeline(self, start_date: datetime, end_date: datetime) -> List[Episode]:
        """Get episodes in chronological order for a time period"""
        
        episodes = await self.search_episodes(
            time_range=(start_date, end_date),
            limit=1000
        )
        
        # Sort by timestamp
        episodes.sort(key=lambda e: e.temporal_context.timestamp)
        
        return episodes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Total episodes
                    cursor.execute('SELECT COUNT(*) FROM episodes')
                    total_episodes = cursor.fetchone()[0]
                    
                    # Episodes by type
                    cursor.execute('''
                        SELECT episode_type, COUNT(*) 
                        FROM episodes 
                        GROUP BY episode_type
                    ''')
                    type_distribution = dict(cursor.fetchall())
                    
                    # Episodes by significance
                    cursor.execute('''
                        SELECT significance, COUNT(*) 
                        FROM episodes 
                        GROUP BY significance
                    ''')
                    significance_distribution = dict(cursor.fetchalls())
                    
                    return {
                        'total_episodes': total_episodes,
                        'type_distribution': type_distribution,
                        'significance_distribution': significance_distribution
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

class AutobiographicalMemory:
    """Manages autobiographical timeline and life narrative"""
    
    def __init__(self, storage: EpisodeStorage):
        self.storage = storage
        
        # Timeline management
        self.timeline_cache: Dict[str, List[Episode]] = {}
        self.life_chapters: Dict[str, Tuple[datetime, datetime]] = {}
        
        self.logger = logging.getLogger("AutobiographicalMemory")
    
    async def get_life_timeline(self, granularity: str = "monthly") -> Dict[str, List[Episode]]:
        """Get complete life timeline organized by time periods"""
        
        # Get all episodes
        all_episodes = await self.storage.search_episodes(limit=10000)
        
        # Organize by time periods
        timeline = defaultdict(list)
        
        for episode in all_episodes:
            timestamp = episode.temporal_context.timestamp
            
            if granularity == "daily":
                period_key = timestamp.strftime("%Y-%m-%d")
            elif granularity == "weekly":
                # Get week start (Monday)
                week_start = timestamp - timedelta(days=timestamp.weekday())
                period_key = week_start.strftime("%Y-W%U")
            elif granularity == "monthly":
                period_key = timestamp.strftime("%Y-%m")
            elif granularity == "yearly":
                period_key = timestamp.strftime("%Y")
            else:
                period_key = "all"
            
            timeline[period_key].append(episode)
        
        # Sort episodes within each period
        for period in timeline:
            timeline[period].sort(key=lambda e: e.temporal_context.timestamp)
        
        return dict(timeline)
    
    async def identify_life_chapters(self) -> Dict[str, Dict[str, Any]]:
        """Identify significant life chapters based on episode patterns"""
        
        all_episodes = await self.storage.search_episodes(limit=10000)
        
        if not all_episodes:
            return {}
        
        # Sort episodes chronologically
        all_episodes.sort(key=lambda e: e.temporal_context.timestamp)
        
        # Identify chapters based on clustering of significant events
        chapters = {}
        current_chapter = None
        chapter_events = []
        
        for episode in all_episodes:
            # Check if this starts a new chapter
            if (episode.significance.value >= EpisodeSignificance.IMPORTANT.value or
                episode.episode_type in [EpisodeType.MILESTONE, EpisodeType.LEARNING_EVENT]):
                
                # Save previous chapter if exists
                if current_chapter and chapter_events:
                    chapters[current_chapter] = self._analyze_chapter(chapter_events)
                
                # Start new chapter
                current_chapter = self._generate_chapter_name(episode)
                chapter_events = [episode]
            else:
                if current_chapter:
                    chapter_events.append(episode)
        
        # Save final chapter
        if current_chapter and chapter_events:
            chapters[current_chapter] = self._analyze_chapter(chapter_events)
        
        return chapters
    
    def _generate_chapter_name(self, pivotal_episode: Episode) -> str:
        """Generate a name for a life chapter"""
        
        timestamp = pivotal_episode.temporal_context.timestamp
        base_name = f"{timestamp.strftime('%Y-%m')} - {pivotal_episode.title}"
        
        return base_name
    
    def _analyze_chapter(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze a chapter to extract key information"""
        
        if not episodes:
            return {}
        
        start_time = min(e.temporal_context.timestamp for e in episodes)
        end_time = max(e.temporal_context.timestamp for e in episodes)
        
        # Extract themes (common tags)
        all_tags = set()
        for episode in episodes:
            all_tags.update(episode.tags)
        
        # Count tag frequency
        tag_counts = defaultdict(int)
        for episode in episodes:
            for tag in episode.tags:
                tag_counts[tag] += 1
        
        # Get most common themes
        common_themes = [tag for tag, count in sorted(tag_counts.items(), 
                                                     key=lambda x: x[1], reverse=True)[:5]]
        
        # Identify key participants
        all_participants = set()
        for episode in episodes:
            all_participants.update(episode.social_context.participants)
        
        # Extract key outcomes and lessons
        all_outcomes = []
        all_lessons = []
        
        for episode in episodes:
            all_outcomes.extend(episode.outcomes)
            all_lessons.extend(episode.lessons_learned)
        
        return {
            'start_date': start_time,
            'end_date': end_time,
            'duration_days': (end_time - start_time).days,
            'episode_count': len(episodes),
            'key_themes': common_themes,
            'key_participants': list(all_participants),
            'major_outcomes': list(set(all_outcomes)),
            'lessons_learned': list(set(all_lessons)),
            'significant_episodes': [e.id for e in episodes 
                                   if e.significance.value >= EpisodeSignificance.IMPORTANT.value]
        }
    
    async def get_personal_narrative(self) -> str:
        """Generate a personal narrative summary"""
        
        chapters = await self.identify_life_chapters()
        
        if not chapters:
            return "No significant life chapters recorded yet."
        
        narrative_parts = ["Personal Life Journey:\n"]
        
        for chapter_name, chapter_info in chapters.items():
            narrative_parts.append(f"\n{chapter_name}:")
            narrative_parts.append(f"  Duration: {chapter_info['duration_days']} days")
            narrative_parts.append(f"  Episodes: {chapter_info['episode_count']}")
            
            if chapter_info['key_themes']:
                narrative_parts.append(f"  Key themes: {', '.join(chapter_info['key_themes'])}")
            
            if chapter_info['major_outcomes']:
                narrative_parts.append(f"  Major outcomes: {'; '.join(chapter_info['major_outcomes'][:3])}")
            
            if chapter_info['lessons_learned']:
                narrative_parts.append(f"  Lessons learned: {'; '.join(chapter_info['lessons_learned'][:2])}")
        
        return "\n".join(narrative_parts)

class EpisodicMemorySystem:
    """Complete episodic memory system"""
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        # Core components
        self.storage = EpisodeStorage(db_path)
        self.autobiographical_memory = AutobiographicalMemory(self.storage)
        
        # Episode management
        self.current_episode: Optional[Episode] = None
        self.episode_stack: List[Episode] = []  # For nested episodes
        
        # Statistics
        self.stats = {
            'episodes_created': 0,
            'episodes_recalled': 0,
            'chapters_identified': 0,
            'connections_made': 0
        }
        
        self.logger = logging.getLogger("EpisodicMemorySystem")
    
    async def initialize(self) -> None:
        """Initialize the episodic memory system"""
        self.logger.info("Episodic memory system initialized")
    
    async def start_episode(self, episode_type: EpisodeType, title: str,
                          description: str = "", participants: List[str] = None,
                          location: str = "", emotion: EmotionalState = EmotionalState.NEUTRAL) -> str:
        """Start recording a new episode"""
        
        episode = Episode(
            id="",
            episode_type=episode_type,
            title=title,
            description=description,
            temporal_context=TemporalContext(
                timestamp=datetime.now(),
                time_of_day=datetime.now().strftime("%H:%M"),
                day_of_week=datetime.now().strftime("%A")
            ),
            spatial_context=SpatialContext(location=location),
            social_context=SocialContext(
                participants=participants or [],
                interaction_type=episode_type.value
            ),
            emotional_context=EmotionalContext(primary_emotion=emotion)
        )
        
        # If there's a current episode, push it to stack
        if self.current_episode:
            self.episode_stack.append(self.current_episode)
        
        self.current_episode = episode
        self.stats['episodes_created'] += 1
        
        self.logger.debug(f"Started episode: {title}")
        
        return episode.id
    
    async def add_to_current_episode(self, content: Dict[str, Any],
                                   tags: Set[str] = None,
                                   outcome: str = None,
                                   lesson: str = None) -> bool:
        """Add content to the current episode"""
        
        if not self.current_episode:
            return False
        
        # Update content
        self.current_episode.content.update(content)
        
        # Add tags
        if tags:
            self.current_episode.tags.update(tags)
        
        # Add outcome
        if outcome:
            self.current_episode.add_outcome(outcome)
        
        # Add lesson
        if lesson:
            self.current_episode.add_lesson(lesson)
        
        return True
    
    async def end_episode(self, final_emotion: EmotionalState = None,
                        significance: EpisodeSignificance = EpisodeSignificance.ROUTINE,
                        summary: str = "") -> bool:
        """End the current episode and store it"""
        
        if not self.current_episode:
            return False
        
        # Update final details
        if final_emotion:
            self.current_episode.emotional_context.mood_after = final_emotion.value
        
        self.current_episode.significance = significance
        
        if summary:
            self.current_episode.content['summary'] = summary
        
        # Calculate duration
        now = datetime.now()
        start_time = self.current_episode.temporal_context.timestamp
        self.current_episode.temporal_context.duration = now - start_time
        
        # Store the episode
        success = await self.storage.store_episode(self.current_episode)
        
        if success:
            self.logger.debug(f"Stored episode: {self.current_episode.title}")
        
        # Restore previous episode from stack
        if self.episode_stack:
            self.current_episode = self.episode_stack.pop()
        else:
            self.current_episode = None
        
        return success
    
    async def recall_episode(self, episode_id: str) -> Optional[Episode]:
        """Recall a specific episode"""
        
        episode = await self.storage.retrieve_episode(episode_id)
        
        if episode:
            episode.recall()
            await self.storage.store_episode(episode)
            self.stats['episodes_recalled'] += 1
        
        return episode
    
    async def search_episodes(self, query: str = "", when: str = "",
                            who: List[str] = None, where: str = "",
                            episode_types: List[EpisodeType] = None,
                            tags: Set[str] = None) -> List[Episode]:
        """Search episodes with natural language parameters"""
        
        # Parse temporal queries
        time_range = self._parse_temporal_query(when)
        
        # Search episodes
        episodes = await self.storage.search_episodes(
            query=query,
            episode_types=episode_types,
            tags=tags,
            participants=who,
            time_range=time_range
        )
        
        # Apply location filter
        if where:
            episodes = [e for e in episodes 
                       if where.lower() in e.spatial_context.location.lower()]
        
        # Update recall statistics
        for episode in episodes:
            episode.recall()
            await self.storage.store_episode(episode)
        
        self.stats['episodes_recalled'] += len(episodes)
        
        return episodes
    
    def _parse_temporal_query(self, when: str) -> Optional[Tuple[datetime, datetime]]:
        """Parse temporal queries like 'last week', 'yesterday', etc."""
        
        if not when:
            return None
        
        now = datetime.now()
        when = when.lower()
        
        if "yesterday" in when:
            start = now - timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            return (start, end)
        
        elif "last week" in when:
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday + 7)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
            return (start, end)
        
        elif "last month" in when:
            if now.month == 1:
                start = now.replace(year=now.year-1, month=12, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                start = now.replace(month=now.month-1, day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Last day of the month
            if start.month == 12:
                end = start.replace(year=start.year+1, month=1, day=1)
            else:
                end = start.replace(month=start.month+1, day=1)
            
            return (start, end)
        
        elif "today" in when:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            return (start, end)
        
        elif "this week" in when:
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
            return (start, end)
        
        return None
    
    async def create_episode_connection(self, episode_id1: str, episode_id2: str,
                                      connection_type: str = "related") -> bool:
        """Create connection between episodes"""
        
        episode1 = await self.storage.retrieve_episode(episode_id1)
        episode2 = await self.storage.retrieve_episode(episode_id2)
        
        if not episode1 or not episode2:
            return False
        
        # Add bidirectional connection
        episode1.related_episodes.add(episode_id2)
        episode2.related_episodes.add(episode_id1)
        
        # Handle causal connections
        if connection_type == "caused":
            episode2.caused_by = episode_id1
            episode1.led_to = episode_id2
        
        # Update storage
        success1 = await self.storage.store_episode(episode1)
        success2 = await self.storage.store_episode(episode2)
        
        if success1 and success2:
            self.stats['connections_made'] += 1
            return True
        
        return False
    
    async def get_personal_timeline(self, granularity: str = "monthly") -> Dict[str, List[Episode]]:
        """Get personal timeline organized by time periods"""
        
        return await self.autobiographical_memory.get_life_timeline(granularity)
    
    async def get_life_story(self) -> str:
        """Get autobiographical life story"""
        
        return await self.autobiographical_memory.get_personal_narrative()
    
    async def find_similar_episodes(self, reference_episode_id: str,
                                  similarity_threshold: float = 0.5) -> List[Episode]:
        """Find episodes similar to a reference episode"""
        
        reference = await self.storage.retrieve_episode(reference_episode_id)
        
        if not reference:
            return []
        
        # Get all episodes
        all_episodes = await self.storage.search_episodes(limit=1000)
        
        # Calculate similarity scores
        similar_episodes = []
        
        for episode in all_episodes:
            if episode.id == reference_episode_id:
                continue
            
            # Calculate similarity based on multiple factors
            similarity = self._calculate_episode_similarity(reference, episode)
            
            if similarity >= similarity_threshold:
                similar_episodes.append((episode, similarity))
        
        # Sort by similarity and return episodes
        similar_episodes.sort(key=lambda x: x[1], reverse=True)
        
        return [episode for episode, similarity in similar_episodes]
    
    def _calculate_episode_similarity(self, episode1: Episode, episode2: Episode) -> float:
        """Calculate similarity between two episodes"""
        
        similarity_factors = []
        
        # Type similarity
        if episode1.episode_type == episode2.episode_type:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Tag similarity
        if episode1.tags and episode2.tags:
            tag_overlap = len(episode1.tags.intersection(episode2.tags))
            tag_union = len(episode1.tags.union(episode2.tags))
            tag_similarity = tag_overlap / tag_union if tag_union > 0 else 0.0
            similarity_factors.append(tag_similarity)
        
        # Participant similarity
        participants1 = set(episode1.social_context.participants)
        participants2 = set(episode2.social_context.participants)
        
        if participants1 and participants2:
            participant_overlap = len(participants1.intersection(participants2))
            participant_union = len(participants1.union(participants2))
            participant_similarity = participant_overlap / participant_union if participant_union > 0 else 0.0
            similarity_factors.append(participant_similarity)
        
        # Location similarity
        if (episode1.spatial_context.location and episode2.spatial_context.location and
            episode1.spatial_context.location == episode2.spatial_context.location):
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Emotional similarity
        if episode1.emotional_context.primary_emotion == episode2.emotional_context.primary_emotion:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Average similarity
        return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        storage_stats = self.storage.get_statistics()
        
        return {
            'system_statistics': self.stats,
            'storage_statistics': storage_stats,
            'current_episode_active': self.current_episode is not None,
            'episode_stack_depth': len(self.episode_stack)
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_episode_recording():
    """Demo: Basic episode recording and retrieval"""
    print("\nDEMO 1: BASIC EPISODE RECORDING")
    print("=" * 50)
    
    memory_system = EpisodicMemorySystem("demo_episodes.db")
    await memory_system.initialize()
    
    print("Recording a learning episode:")
    
    # Start a learning episode
    episode_id = await memory_system.start_episode(
        EpisodeType.LEARNING_EVENT,
        "Learning about Neural Networks",
        "Deep dive into neural network architectures",
        participants=["AI_Tutor"],
        location="Study Room",
        emotion=EmotionalState.EXCITED
    )
    
    print(f"  Started episode: {episode_id[:8]}...")
    
    # Add content during the episode
    await memory_system.add_to_current_episode(
        content={
            "topic": "Neural Networks",
            "subtopics": ["Perceptrons", "Backpropagation", "Deep Learning"],
            "resources_used": ["textbook", "online_course", "practice_problems"],
            "difficulty_level": "intermediate"
        },
        tags={"learning", "AI", "neural_networks", "study"},
        outcome="Understood basic concepts",
        lesson="Need more practice with backpropagation math"
    )
    
    print(f"  Added learning content and outcomes")
    
    # Add more progress
    await memory_system.add_to_current_episode(
        content={
            "practice_problems_solved": 5,
            "concepts_mastered": ["forward_propagation", "activation_functions"],
            "concepts_struggling": ["gradient_descent_details"]
        },
        outcome="Completed 5 practice problems successfully"
    )
    
    print(f"  Added practice results")
    
    # End the episode
    await memory_system.end_episode(
        final_emotion=EmotionalState.SATISFIED,
        significance=EpisodeSignificance.IMPORTANT,
        summary="Productive learning session on neural networks with good progress"
    )
    
    print(f"  Ended episode successfully")
    
    # Recall the episode
    print(f"\nRecalling the recorded episode:")
    
    recalled_episode = await memory_system.recall_episode(episode_id)
    
    if recalled_episode:
        print(f"  Title: {recalled_episode.title}")
        print(f"  Duration: {recalled_episode.temporal_context.duration}")
        print(f"  Participants: {recalled_episode.social_context.participants}")
        print(f"  Outcomes: {recalled_episode.outcomes}")
        print(f"  Lessons: {recalled_episode.lessons_learned}")
        print(f"  Significance: {recalled_episode.significance.name}")
        print(f"  Tags: {recalled_episode.tags}")

async def demo_conversation_episodes():
    """Demo: Recording conversation episodes"""
    print("\nDEMO 2: CONVERSATION EPISODES")
    print("=" * 50)
    
    memory_system = EpisodicMemorySystem("demo_conversations.db")
    await memory_system.initialize()
    
    print("Recording multiple conversation episodes:")
    
    # First conversation
    print(f"\n1. Technical discussion with colleague:")
    
    conv1_id = await memory_system.start_episode(
        EpisodeType.CONVERSATION,
        "Technical Discussion about API Design",
        "Discussing REST API best practices",
        participants=["John", "Sarah"],
        location="Office Conference Room",
        emotion=EmotionalState.NEUTRAL
    )
    
    await memory_system.add_to_current_episode(
        content={
            "topic": "REST API Design",
            "key_points": [
                "Resource naming conventions",
                "HTTP status codes usage",
                "Authentication strategies"
            ],
            "decisions_made": [
                "Use JWT for authentication",
                "Implement rate limiting",
                "Follow OpenAPI specification"
            ],
            "action_items": [
                "John will draft API specification",
                "Sarah will research rate limiting libraries"
            ]
        },
        tags={"work", "API", "design", "team_discussion"},
        outcome="Clear direction for API implementation",
        lesson="Need to document decisions better for future reference"
    )
    
    await memory_system.end_episode(
        final_emotion=EmotionalState.SATISFIED,
        significance=EpisodeSignificance.NOTABLE
    )
    
    print(f"    Recorded technical discussion episode")
    
    # Second conversation - casual
    print(f"\n2. Casual chat with friend:")
    
    conv2_id = await memory_system.start_episode(
        EpisodeType.CONVERSATION,
        "Coffee Chat with Alex",
        "Catching up over weekend plans",
        participants=["Alex"],
        location="Local Coffee Shop",
        emotion=EmotionalState.POSITIVE
    )
    
    await memory_system.add_to_current_episode(
        content={
            "topics": ["weekend_plans", "recent_vacation", "new_hobby"],
            "alex_updates": {
                "vacation": "Trip to mountains was amazing",
                "hobby": "Started learning guitar",
                "work": "Got promotion at work"
            },
            "my_updates": {
                "projects": "Working on AI learning project",
                "plans": "Planning hiking trip next month"
            }
        },
        tags={"personal", "friendship", "casual", "coffee"},
        outcome="Enjoyed catching up and made hiking plans"
    )
    
    await memory_system.end_episode(
        final_emotion=EmotionalState.POSITIVE,
        significance=EpisodeSignificance.ROUTINE
    )
    
    print(f"    Recorded casual conversation episode")
    
    # Search conversations
    print(f"\nSearching for conversation episodes:")
    
    # Search by participant
    conversations_with_sarah = await memory_system.search_episodes(
        who=["Sarah"],
        episode_types=[EpisodeType.CONVERSATION]
    )
    
    print(f"  Conversations with Sarah: {len(conversations_with_sarah)}")
    for conv in conversations_with_sarah:
        print(f"    - {conv.title}")
    
    # Search by location
    office_conversations = await memory_system.search_episodes(
        where="Office",
        episode_types=[EpisodeType.CONVERSATION]
    )
    
    print(f"  Office conversations: {len(office_conversations)}")
    for conv in office_conversations:
        print(f"    - {conv.title}")
    
    # Search by topic
    api_discussions = await memory_system.search_episodes(
        query="API",
        tags={"work"}
    )
    
    print(f"  API-related discussions: {len(api_discussions)}")
    for conv in api_discussions:
        print(f"    - {conv.title}")

async def demo_temporal_episode_search():
    """Demo: Temporal episode search and timeline"""
    print("\nDEMO 3: TEMPORAL EPISODE SEARCH")
    print("=" * 50)
    
    memory_system = EpisodicMemorySystem("demo_temporal.db")
    await memory_system.initialize()
    
    print("Creating episodes with different timestamps:")
    
    # Create episodes with various time stamps
    now = datetime.now()
    
    episodes_to_create = [
        {
            "time_offset": timedelta(hours=2),
            "title": "Morning Code Review",
            "type": EpisodeType.TASK_COMPLETION,
            "tags": {"work", "code_review", "morning"}
        },
        {
            "time_offset": timedelta(days=1),
            "title": "Yesterday's Team Meeting", 
            "type": EpisodeType.CONVERSATION,
            "tags": {"work", "meeting", "team"}
        },
        {
            "time_offset": timedelta(days=7),
            "title": "Last Week's Learning Session",
            "type": EpisodeType.LEARNING_EVENT,
            "tags": {"learning", "AI", "study"}
        },
        {
            "time_offset": timedelta(days=30),
            "title": "Monthly Planning Session",
            "type": EpisodeType.DECISION_MAKING,
            "tags": {"planning", "goals", "monthly"}
        }
    ]
    
    created_episodes = []
    
    for episode_data in episodes_to_create:
        episode_id = await memory_system.start_episode(
            episode_data["type"],
            episode_data["title"],
            f"Episode from {episode_data['time_offset']} ago"
        )
        
        await memory_system.add_to_current_episode(
            content={"simulated": True},
            tags=episode_data["tags"]
        )
        
        await memory_system.end_episode(significance=EpisodeSignificance.NOTABLE)
        
        # Manually adjust timestamp for demo
        episode = await memory_system.recall_episode(episode_id)
        episode.temporal_context.timestamp = now - episode_data["time_offset"]
        await memory_system.storage.store_episode(episode)
        
        created_episodes.append(episode_id)
        print(f"  Created: {episode_data['title']}")
    
    # Test temporal searches
    print(f"\nTesting temporal searches:")
    
    # Search for today's episodes
    today_episodes = await memory_system.search_episodes(when="today")
    print(f"  Today's episodes: {len(today_episodes)}")
    for episode in today_episodes:
        print(f"    - {episode.title}")
    
    # Search for yesterday's episodes
    yesterday_episodes = await memory_system.search_episodes(when="yesterday")
    print(f"  Yesterday's episodes: {len(yesterday_episodes)}")
    for episode in yesterday_episodes:
        print(f"    - {episode.title}")
    
    # Search for last week's episodes
    last_week_episodes = await memory_system.search_episodes(when="last week")
    print(f"  Last week's episodes: {len(last_week_episodes)}")
    for episode in last_week_episodes:
        print(f"    - {episode.title}")
    
    # Search for last month's episodes
    last_month_episodes = await memory_system.search_episodes(when="last month")
    print(f"  Last month's episodes: {len(last_month_episodes)}")
    for episode in last_month_episodes:
        print(f"    - {episode.title}")
    
    # Get timeline view
    print(f"\nPersonal timeline (daily granularity):")
    timeline = await memory_system.get_personal_timeline("daily")
    
    for date, episodes in sorted(timeline.items()):
        print(f"  {date}: {len(episodes)} episodes")
        for episode in episodes:
            print(f"    - {episode.title}")

async def demo_episode_connections():
    """Demo: Creating connections between episodes"""
    print("\nDEMO 4: EPISODE CONNECTIONS")
    print("=" * 50)
    
    memory_system = EpisodicMemorySystem("demo_connections.db")
    await memory_system.initialize()
    
    print("Creating connected episodes (cause and effect):")
    
    # First episode: Planning
    planning_id = await memory_system.start_episode(
        EpisodeType.DECISION_MAKING,
        "Planning New Feature Implementation",
        "Deciding on architecture for user authentication feature"
    )
    
    await memory_system.add_to_current_episode(
        content={
            "feature": "User Authentication",
            "options_considered": ["JWT", "Sessions", "OAuth"],
            "decision": "Implement JWT with refresh tokens",
            "reasoning": "Better scalability and stateless design"
        },
        tags={"planning", "authentication", "architecture"},
        outcome="Decided on JWT implementation approach"
    )
    
    await memory_system.end_episode(significance=EpisodeSignificance.IMPORTANT)
    print(f"  Created planning episode")
    
    # Second episode: Implementation (caused by planning)
    implementation_id = await memory_system.start_episode(
        EpisodeType.TASK_COMPLETION,
        "Implementing JWT Authentication",
        "Building the authentication system decided in planning"
    )
    
    await memory_system.add_to_current_episode(
        content={
            "tasks_completed": [
                "Set up JWT library",
                "Create login endpoint",
                "Implement token validation middleware",
                "Add refresh token logic"
            ],
            "challenges": [
                "Token expiration handling",
                "Secure storage of refresh tokens"
            ],
            "time_spent": "6 hours"
        },
        tags={"implementation", "authentication", "JWT", "coding"},
        outcome="Successfully implemented JWT authentication system",
        lesson="Token expiration needs more thought for UX"
    )
    
    await memory_system.end_episode(significance=EpisodeSignificance.IMPORTANT)
    print(f"  Created implementation episode")
    
    # Third episode: Testing (caused by implementation)
    testing_id = await memory_system.start_episode(
        EpisodeType.TASK_COMPLETION,
        "Testing Authentication System",
        "Comprehensive testing of the new authentication feature"
    )
    
    await memory_system.add_to_current_episode(
        content={
            "test_types": ["unit_tests", "integration_tests", "security_tests"],
            "tests_written": 15,
            "bugs_found": 3,
            "bugs_fixed": 3,
            "test_coverage": "95%"
        },
        tags={"testing", "authentication", "quality_assurance"},
        outcome="All tests passing, system ready for deployment",
        lesson="Security testing revealed edge cases we hadn't considered"
    )
    
    await memory_system.end_episode(significance=EpisodeSignificance.NOTABLE)
    print(f"  Created testing episode")
    
    # Create causal connections
    print(f"\nCreating causal connections between episodes:")
    
    # Planning led to implementation
    await memory_system.create_episode_connection(
        planning_id, implementation_id, "caused"
    )
    print(f"  Connected planning → implementation")
    
    # Implementation led to testing
    await memory_system.create_episode_connection(
        implementation_id, testing_id, "caused"
    )
    print(f"  Connected implementation → testing")
    
    # Verify connections
    print(f"\nVerifying episode connections:")
    
    planning_episode = await memory_system.recall_episode(planning_id)
    print(f"  Planning episode led to: {planning_episode.led_to}")
    print(f"  Related episodes: {planning_episode.related_episodes}")
    
    implementation_episode = await memory_system.recall_episode(implementation_id)
    print(f"  Implementation was caused by: {implementation_episode.caused_by}")
    print(f"  Implementation led to: {implementation_episode.led_to}")
    
    testing_episode = await memory_system.recall_episode(testing_id)
    print(f"  Testing was caused by: {testing_episode.caused_by}")
    
    # Find similar episodes
    print(f"\nFinding similar episodes to implementation:")
    
    similar = await memory_system.find_similar_episodes(implementation_id)
    print(f"  Found {len(similar)} similar episodes")
    for episode in similar:
        print(f"    - {episode.title} (similarity based on type and tags)")

async def demo_autobiographical_memory():
    """Demo: Autobiographical memory and life narrative"""
    print("\nDEMO 5: AUTOBIOGRAPHICAL MEMORY")
    print("=" * 50)
    
    memory_system = EpisodicMemorySystem("demo_autobiography.db")
    await memory_system.initialize()
    
    print("Creating a series of episodes that form life chapters:")
    
    # Create episodes representing a learning journey
    learning_episodes = [
        {
            "title": "Started Learning Programming",
            "type": EpisodeType.MILESTONE,
            "content": {"language": "Python", "motivation": "Career change"},
            "tags": {"learning", "programming", "python", "career"},
            "significance": EpisodeSignificance.MILESTONE,
            "days_ago": 180
        },
        {
            "title": "First Programming Project",
            "type": EpisodeType.TASK_COMPLETION,
            "content": {"project": "Calculator app", "lines_of_code": 200},
            "tags": {"programming", "project", "python", "milestone"},
            "significance": EpisodeSignificance.IMPORTANT,
            "days_ago": 150
        },
        {
            "title": "Learned Object-Oriented Programming",
            "type": EpisodeType.LEARNING_EVENT,
            "content": {"concepts": ["classes", "inheritance", "polymorphism"]},
            "tags": {"learning", "OOP", "programming", "concepts"},
            "significance": EpisodeSignificance.IMPORTANT,
            "days_ago": 120
        },
        {
            "title": "Built Web Application",
            "type": EpisodeType.TASK_COMPLETION,
            "content": {"framework": "Flask", "features": ["user_auth", "database"]},
            "tags": {"web_development", "flask", "project", "achievement"},
            "significance": EpisodeSignificance.SIGNIFICANT,
            "days_ago": 90
        },
        {
            "title": "Got First Programming Job",
            "type": EpisodeType.MILESTONE,
            "content": {"company": "TechCorp", "role": "Junior Developer"},
            "tags": {"career", "job", "milestone", "achievement"},
            "significance": EpisodeSignificance.MILESTONE,
            "days_ago": 60
        },
        {
            "title": "Started Learning AI/ML",
            "type": EpisodeType.LEARNING_EVENT,
            "content": {"focus": "Machine Learning", "goal": "Specialization"},
            "tags": {"learning", "AI", "ML", "specialization"},
            "significance": EpisodeSignificance.IMPORTANT,
            "days_ago": 30
        }
    ]
    
    now = datetime.now()
    
    for episode_data in learning_episodes:
        episode_id = await memory_system.start_episode(
            episode_data["type"],
            episode_data["title"],
            f"Important step in programming journey"
        )
        
        await memory_system.add_to_current_episode(
            content=episode_data["content"],
            tags=episode_data["tags"]
        )
        
        await memory_system.end_episode(significance=episode_data["significance"])
        
        # Adjust timestamp
        episode = await memory_system.recall_episode(episode_id)
        episode.temporal_context.timestamp = now - timedelta(days=episode_data["days_ago"])
        await memory_system.storage.store_episode(episode)
        
        print(f"  Created: {episode_data['title']}")
    
    # Generate life story
    print(f"\nGenerating autobiographical life story:")
    
    life_story = await memory_system.get_life_story()
    print(life_story)
    
    # Get monthly timeline
    print(f"\nMonthly timeline view:")
    
    timeline = await memory_system.get_personal_timeline("monthly")
    
    for month, episodes in sorted(timeline.items(), reverse=True):
        print(f"\n{month}: {len(episodes)} episodes")
        for episode in episodes:
            print(f"  - {episode.title} ({episode.significance.name})")
    
    # Show system statistics
    print(f"\nAutobiographical memory statistics:")
    stats = memory_system.get_system_statistics()
    
    print(f"  Episodes created: {stats['system_statistics']['episodes_created']}")
    print(f"  Episodes recalled: {stats['system_statistics']['episodes_recalled']}")
    print(f"  Connections made: {stats['system_statistics']['connections_made']}")
    print(f"  Total episodes stored: {stats['storage_statistics']['total_episodes']}")

async def main():
    """
    Demonstrate Episodic Memory System for storing and retrieving personal experiences
    
    WHAT YOU'LL LEARN:
    ================
    1. How to record and structure episodic memories with rich context
    2. How to implement temporal, spatial, social, and emotional context tracking
    3. How to search episodes using natural language temporal queries
    4. How to create causal connections between related episodes
    5. How to build autobiographical memory and life narrative systems
    6. How to create complete episodic memory systems for personal AI
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal AI assistants that remember shared experiences
    - Educational systems that track learning journeys and milestones
    - Therapeutic AI that maintains client history and progress
    - Customer service systems that remember interaction history
    - Collaborative AI that recalls team projects and decisions
    - Life coaching AI that tracks personal growth and achievements
    """
    
    print("EPISODIC MEMORY SYSTEM DEMONSTRATION")
    print("Storing and retrieving personal experiences!")
    
    await demo_basic_episode_recording()
    await demo_conversation_episodes()
    await demo_temporal_episode_search()
    await demo_episode_connections()
    await demo_autobiographical_memory()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Episodic memory captures rich contextual information about experiences")
    print("✓ Temporal search enables natural language queries about 'when' events occurred")
    print("✓ Episode connections create causal chains and relationship networks")
    print("✓ Autobiographical memory builds coherent life narratives")
    print("✓ Complete systems support personal AI with human-like memory")
    print("✓ Context tracking enables understanding of circumstances and outcomes")
    print("\nTHE POWER OF EPISODIC MEMORY:")
    print("- Enables AI to form personal relationships with continuity")
    print("- Supports learning from past experiences and building on successes")
    print("- Creates sense of personal identity and growth over time")
    print("- Allows contextual understanding of current situations")
    print("- Forms foundation for truly personalized and adaptive AI systems")

if __name__ == "__main__":
    asyncio.run(main())
