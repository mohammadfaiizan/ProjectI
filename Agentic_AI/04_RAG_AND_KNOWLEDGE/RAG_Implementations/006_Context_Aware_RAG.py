#!/usr/bin/env python3
"""
Context-Aware RAG: Intelligent Contextual Information Retrieval
==============================================================

WHAT IS THE PROBLEM?
==================
Traditional RAG systems ignore context:
- Treat each query in isolation without conversation history
- Cannot maintain coherent multi-turn conversations
- Ignore user preferences and past interactions
- Cannot adapt to domain-specific contexts
- Miss implicit context and unstated assumptions
- Cannot handle follow-up questions and references

Example: Customer Support Complexity
CONTEXT-BLIND RAG (Traditional):
- User: "How do I reset my password?"
- System: Generic password reset instructions
- User: "That didn't work for my account"
- System: Same generic instructions (no context of previous failure)
- User: "I'm using the mobile app"
- System: Generic instructions again (ignores mobile app context)
- Result: Frustrated user, ineffective support

REAL WORLD EXAMPLE:
=================
How does Amazon's Alexa handle context?

ALEXA'S CONTEXT-AWARE SYSTEM:
1. CONVERSATION MEMORY: Remembers what was discussed
2. USER MODELING: Adapts to individual preferences and patterns
3. DEVICE CONTEXT: Knows which device is being used
4. TEMPORAL CONTEXT: Considers time of day, day of week
5. LOCATION CONTEXT: Uses geographic and situational context
6. ACTIVITY CONTEXT: Understands what user is currently doing
7. FOLLOW-UP HANDLING: Resolves pronouns and references

BENEFITS OF CONTEXT AWARENESS:
- Natural conversational flow like human interaction
- Personalized responses based on user history
- Reduced need for repetitive clarifications
- Higher accuracy through contextual disambiguation
- Better user experience with intelligent assistance
- Efficient information retrieval through context filtering

THE CONTEXT ADVANTAGE:
====================
TRADITIONAL RAG: Query → Retrieve → Generate (isolated)
CONTEXT-AWARE RAG: Context + Query → Contextual Retrieve → Context-Enhanced Generate

CONTEXT TYPES:
=============
1. CONVERSATIONAL CONTEXT: Previous messages and responses
2. USER CONTEXT: Preferences, history, expertise level
3. DOMAIN CONTEXT: Technical field, business area, industry
4. TEMPORAL CONTEXT: Time-based relevance and recency
5. SITUATIONAL CONTEXT: Current task, goal, environment
6. SEMANTIC CONTEXT: Concept relationships and implications
7. EMOTIONAL CONTEXT: User sentiment and communication style

WHY THIS IS REVOLUTIONARY:
========================
- Enables natural, human-like conversational AI
- Provides personalized and adaptive user experiences
- Supports complex multi-turn problem-solving
- Powers next-generation intelligent assistants
- Critical for enterprise AI adoption and user satisfaction
- Enables AI systems that truly understand and remember
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import re
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ContextType(Enum):
    """Types of context for RAG systems"""
    CONVERSATIONAL = "conversational"   # Previous conversation turns
    USER_PROFILE = "user_profile"       # User preferences and history
    DOMAIN_SPECIFIC = "domain_specific" # Technical/business domain context
    TEMPORAL = "temporal"               # Time-based context
    SITUATIONAL = "situational"        # Current task/goal context
    SEMANTIC = "semantic"               # Concept and relationship context
    EMOTIONAL = "emotional"             # Sentiment and communication style

class ContextScope(Enum):
    """Scope of context application"""
    IMMEDIATE = "immediate"             # Current query only
    SESSION = "session"                 # Current conversation session
    USER_HISTORY = "user_history"       # User's historical interactions
    GLOBAL = "global"                   # System-wide patterns

class UserExpertiseLevel(Enum):
    """User expertise levels for context adaptation"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    turn_id: str
    user_query: str
    system_response: str
    timestamp: datetime
    
    # Context extracted from this turn
    entities_mentioned: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    sentiment: float = 0.0  # -1 to 1
    
    # Metadata
    response_quality: float = 0.0
    user_satisfaction: Optional[bool] = None
    follow_up_needed: bool = False
    
    def __post_init__(self):
        if not self.turn_id:
            self.turn_id = str(uuid.uuid4())

@dataclass
class UserProfile:
    """User profile for personalization"""
    user_id: str
    
    # Preferences
    expertise_level: UserExpertiseLevel = UserExpertiseLevel.INTERMEDIATE
    preferred_detail_level: str = "medium"  # low, medium, high
    communication_style: str = "formal"    # formal, casual, technical
    
    # Interests and domains
    primary_domains: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    
    # Behavioral patterns
    typical_query_types: List[str] = field(default_factory=list)
    interaction_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Historical data
    total_interactions: int = 0
    successful_interactions: int = 0
    average_session_length: float = 0.0
    
    # Temporal patterns
    active_hours: List[int] = field(default_factory=list)  # Hours of day
    active_days: List[str] = field(default_factory=list)   # Days of week
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
    
    def update_from_interaction(self, turn: ConversationTurn) -> None:
        """Update profile based on interaction"""
        self.total_interactions += 1
        
        if turn.user_satisfaction is True:
            self.successful_interactions += 1
        
        # Update interests
        for topic in turn.topics_discussed:
            if topic not in self.interests:
                self.interests.append(topic)
        
        # Update temporal patterns
        hour = turn.timestamp.hour
        if hour not in self.active_hours:
            self.active_hours.append(hour)
        
        day = turn.timestamp.strftime('%A')
        if day not in self.active_days:
            self.active_days.append(day)

@dataclass
class ContextState:
    """Current context state for RAG system"""
    
    # Conversation context
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    current_session_id: str = ""
    session_start_time: datetime = field(default_factory=datetime.now)
    
    # User context
    user_profile: Optional[UserProfile] = None
    current_user_intent: Optional[str] = None
    user_expertise_context: UserExpertiseLevel = UserExpertiseLevel.INTERMEDIATE
    
    # Domain context
    current_domain: Optional[str] = None
    domain_vocabulary: Dict[str, float] = field(default_factory=dict)  # term -> importance
    domain_concepts: List[str] = field(default_factory=list)
    
    # Temporal context
    current_time: datetime = field(default_factory=datetime.now)
    time_sensitive_topics: List[str] = field(default_factory=list)
    
    # Situational context
    current_task: Optional[str] = None
    task_progress: float = 0.0  # 0.0 to 1.0
    immediate_needs: List[str] = field(default_factory=list)
    
    # Semantic context
    active_entities: Dict[str, float] = field(default_factory=dict)  # entity -> relevance
    concept_relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    # Emotional context
    current_sentiment: float = 0.0  # -1 to 1
    user_frustration_level: float = 0.0  # 0.0 to 1.0
    communication_tone: str = "neutral"
    
    def add_conversation_turn(self, turn: ConversationTurn) -> None:
        """Add new conversation turn and update context"""
        self.conversation_history.append(turn)
        
        # Update active entities
        for entity in turn.entities_mentioned:
            self.active_entities[entity] = self.active_entities.get(entity, 0.0) + 0.3
        
        # Decay older entity relevance
        for entity in list(self.active_entities.keys()):
            self.active_entities[entity] *= 0.9
            if self.active_entities[entity] < 0.1:
                del self.active_entities[entity]
        
        # Update sentiment
        self.current_sentiment = 0.7 * self.current_sentiment + 0.3 * turn.sentiment
        
        # Update user profile if available
        if self.user_profile:
            self.user_profile.update_from_interaction(turn)
    
    def get_recent_context(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation context"""
        return self.conversation_history[-max_turns:]
    
    def get_relevant_entities(self, threshold: float = 0.2) -> List[str]:
        """Get currently relevant entities"""
        return [entity for entity, relevance in self.active_entities.items() 
                if relevance >= threshold]

class ContextExtractor:
    """Extracts various types of context from queries and conversations"""
    
    def __init__(self):
        # Entity patterns
        self.entity_patterns = {
            'person': r'\b(?:Mr\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+',
            'organization': r'\b(?:Inc\.|Corp\.|LLC|Ltd\.)\b|\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Inc\.|Corp\.|LLC|Ltd\.)',
            'technology': r'\b(?:AI|ML|API|SDK|GPU|CPU|RAM|SSD|HTTP|JSON|XML|SQL|NoSQL)\b',
            'product': r'\b(?:iPhone|Android|Windows|Linux|MacOS|Office|Photoshop)\b'
        }
        
        # Intent patterns
        self.intent_patterns = {
            'question': r'\b(?:what|how|why|when|where|which|who)\b',
            'request': r'\b(?:please|can you|could you|would you|help me)\b',
            'instruction': r'\b(?:show me|tell me|explain|describe|guide me)\b',
            'problem': r'\b(?:error|issue|problem|trouble|difficulty|fail)\b',
            'comparison': r'\b(?:compare|versus|vs|difference|better|worse)\b'
        }
        
        # Sentiment indicators
        self.positive_words = {'good', 'great', 'excellent', 'perfect', 'amazing', 'helpful', 'useful', 'clear'}
        self.negative_words = {'bad', 'terrible', 'awful', 'confusing', 'unhelpful', 'unclear', 'frustrated', 'annoying'}
        
        self.logger = logging.getLogger("ContextExtractor")
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = defaultdict(list)
        
        text_lower = text.lower()
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type].extend(matches)
        
        return dict(entities)
    
    async def extract_intent(self, text: str) -> Optional[str]:
        """Extract primary intent from text"""
        text_lower = text.lower()
        
        intent_scores = {}
        for intent, pattern in self.intent_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                intent_scores[intent] = matches
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def extract_sentiment(self, text: str) -> float:
        """Extract sentiment score from text"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, sentiment))
    
    async def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified)"""
        # Simple topic extraction based on keywords
        topic_keywords = {
            'artificial_intelligence': ['ai', 'artificial intelligence', 'machine learning', 'deep learning'],
            'software_development': ['programming', 'coding', 'development', 'software', 'application'],
            'business': ['business', 'strategy', 'market', 'customer', 'revenue', 'growth'],
            'technology': ['technology', 'tech', 'innovation', 'digital', 'computing'],
            'data_science': ['data', 'analytics', 'statistics', 'analysis', 'insights'],
            'healthcare': ['health', 'medical', 'healthcare', 'patient', 'treatment'],
            'finance': ['finance', 'financial', 'money', 'investment', 'banking'],
            'education': ['education', 'learning', 'teaching', 'student', 'course']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    async def extract_conversation_context(self, turn: ConversationTurn) -> None:
        """Extract context from a conversation turn"""
        
        # Extract entities
        entities = await self.extract_entities(turn.user_query)
        turn.entities_mentioned = []
        for entity_list in entities.values():
            turn.entities_mentioned.extend(entity_list)
        
        # Extract intent
        turn.intent = await self.extract_intent(turn.user_query)
        
        # Extract sentiment
        turn.sentiment = await self.extract_sentiment(turn.user_query)
        
        # Extract topics
        turn.topics_discussed = await self.extract_topics(turn.user_query)

class ContextualQueryEnhancer:
    """Enhances queries with contextual information"""
    
    def __init__(self):
        self.logger = logging.getLogger("ContextualQueryEnhancer")
    
    async def enhance_query(self, original_query: str, context_state: ContextState) -> str:
        """Enhance query with contextual information"""
        
        enhanced_parts = [original_query]
        
        # Add conversational context
        conversation_context = self._build_conversation_context(context_state)
        if conversation_context:
            enhanced_parts.append(f"Context from conversation: {conversation_context}")
        
        # Add user context
        user_context = self._build_user_context(context_state)
        if user_context:
            enhanced_parts.append(f"User context: {user_context}")
        
        # Add domain context
        domain_context = self._build_domain_context(context_state)
        if domain_context:
            enhanced_parts.append(f"Domain context: {domain_context}")
        
        # Add temporal context
        temporal_context = self._build_temporal_context(context_state)
        if temporal_context:
            enhanced_parts.append(f"Temporal context: {temporal_context}")
        
        # Add entity context
        entity_context = self._build_entity_context(context_state)
        if entity_context:
            enhanced_parts.append(f"Relevant entities: {entity_context}")
        
        enhanced_query = " | ".join(enhanced_parts)
        
        self.logger.debug(f"Enhanced query: {enhanced_query[:200]}...")
        
        return enhanced_query
    
    def _build_conversation_context(self, context_state: ContextState) -> str:
        """Build conversation context summary"""
        recent_turns = context_state.get_recent_context(max_turns=3)
        
        if not recent_turns:
            return ""
        
        context_parts = []
        
        for turn in recent_turns[-2:]:  # Last 2 turns
            if turn.topics_discussed:
                topics = ", ".join(turn.topics_discussed[:2])
                context_parts.append(f"previously discussed {topics}")
            
            if turn.intent:
                context_parts.append(f"previous intent was {turn.intent}")
        
        return "; ".join(context_parts)
    
    def _build_user_context(self, context_state: ContextState) -> str:
        """Build user context summary"""
        if not context_state.user_profile:
            return ""
        
        profile = context_state.user_profile
        context_parts = []
        
        # Expertise level
        context_parts.append(f"user expertise: {profile.expertise_level.value}")
        
        # Communication style
        if profile.communication_style != "formal":
            context_parts.append(f"communication style: {profile.communication_style}")
        
        # Primary domains
        if profile.primary_domains:
            domains = ", ".join(profile.primary_domains[:2])
            context_parts.append(f"domains of interest: {domains}")
        
        return "; ".join(context_parts)
    
    def _build_domain_context(self, context_state: ContextState) -> str:
        """Build domain-specific context"""
        if not context_state.current_domain:
            return ""
        
        context_parts = [f"current domain: {context_state.current_domain}"]
        
        # Top domain concepts
        if context_state.domain_concepts:
            concepts = ", ".join(context_state.domain_concepts[:3])
            context_parts.append(f"key concepts: {concepts}")
        
        return "; ".join(context_parts)
    
    def _build_temporal_context(self, context_state: ContextState) -> str:
        """Build temporal context"""
        context_parts = []
        
        # Time-sensitive topics
        if context_state.time_sensitive_topics:
            topics = ", ".join(context_state.time_sensitive_topics[:2])
            context_parts.append(f"time-sensitive: {topics}")
        
        # Current time context
        now = context_state.current_time
        if now.hour < 12:
            context_parts.append("morning context")
        elif now.hour < 17:
            context_parts.append("afternoon context")
        else:
            context_parts.append("evening context")
        
        return "; ".join(context_parts)
    
    def _build_entity_context(self, context_state: ContextState) -> str:
        """Build entity context"""
        relevant_entities = context_state.get_relevant_entities()
        
        if not relevant_entities:
            return ""
        
        # Take most relevant entities
        entity_items = [(entity, relevance) for entity, relevance in context_state.active_entities.items()]
        entity_items.sort(key=lambda x: x[1], reverse=True)
        
        top_entities = [entity for entity, _ in entity_items[:3]]
        
        return ", ".join(top_entities)

class ContextualRetriever:
    """Retriever that uses context for enhanced document retrieval"""
    
    def __init__(self):
        # Simulated document corpus with metadata
        self.documents = self._create_contextual_documents()
        self.document_embeddings = self._create_document_embeddings()
        
        self.logger = logging.getLogger("ContextualRetriever")
    
    async def contextual_retrieve(self, enhanced_query: str, context_state: ContextState,
                                top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using contextual information"""
        
        try:
            # Score documents based on multiple contextual factors
            doc_scores = []
            
            for i, doc in enumerate(self.documents):
                # Base relevance score
                relevance_score = self._calculate_base_relevance(enhanced_query, doc)
                
                # Contextual scoring
                user_score = self._score_user_context(doc, context_state)
                domain_score = self._score_domain_context(doc, context_state)
                temporal_score = self._score_temporal_context(doc, context_state)
                entity_score = self._score_entity_context(doc, context_state)
                
                # Combine scores with weights
                total_score = (
                    relevance_score * 0.4 +
                    user_score * 0.2 +
                    domain_score * 0.2 +
                    temporal_score * 0.1 +
                    entity_score * 0.1
                )
                
                doc_scores.append((i, total_score, doc))
            
            # Sort by score and return top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = doc_scores[:top_k]
            
            retrieved_docs = []
            for doc_idx, score, doc in top_results:
                doc_copy = doc.copy()
                doc_copy['retrieval_score'] = score
                doc_copy['context_factors'] = {
                    'user_match': self._score_user_context(doc, context_state),
                    'domain_match': self._score_domain_context(doc, context_state),
                    'temporal_match': self._score_temporal_context(doc, context_state),
                    'entity_match': self._score_entity_context(doc, context_state)
                }
                retrieved_docs.append(doc_copy)
            
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Contextual retrieval failed: {e}")
            return []
    
    def _create_contextual_documents(self) -> List[Dict[str, Any]]:
        """Create documents with rich contextual metadata"""
        documents = []
        
        # Document templates with different characteristics
        doc_templates = [
            {
                'domain': 'artificial_intelligence',
                'expertise_level': 'beginner',
                'content_type': 'tutorial',
                'topics': ['machine learning', 'neural networks', 'AI basics'],
                'entities': ['TensorFlow', 'Python', 'algorithms'],
                'time_sensitivity': 'low'
            },
            {
                'domain': 'artificial_intelligence',
                'expertise_level': 'advanced',
                'content_type': 'research',
                'topics': ['deep learning', 'transformers', 'attention mechanisms'],
                'entities': ['BERT', 'GPT', 'attention'],
                'time_sensitivity': 'high'
            },
            {
                'domain': 'business',
                'expertise_level': 'intermediate',
                'content_type': 'analysis',
                'topics': ['market strategy', 'competitive analysis', 'growth'],
                'entities': ['market share', 'ROI', 'KPIs'],
                'time_sensitivity': 'medium'
            },
            {
                'domain': 'software_development',
                'expertise_level': 'beginner',
                'content_type': 'guide',
                'topics': ['programming basics', 'Python', 'coding'],
                'entities': ['variables', 'functions', 'loops'],
                'time_sensitivity': 'low'
            },
            {
                'domain': 'healthcare',
                'expertise_level': 'expert',
                'content_type': 'research',
                'topics': ['medical AI', 'diagnosis', 'patient care'],
                'entities': ['medical imaging', 'clinical trials', 'FDA'],
                'time_sensitivity': 'high'
            }
        ]
        
        # Generate documents based on templates
        for i in range(100):
            template = doc_templates[i % len(doc_templates)]
            
            doc = {
                'id': f'ctx_doc_{i:03d}',
                'title': f'{template["domain"].replace("_", " ").title()} - {template["content_type"].title()} {i // len(doc_templates) + 1}',
                'content': self._generate_content(template, i),
                'domain': template['domain'],
                'expertise_level': template['expertise_level'],
                'content_type': template['content_type'],
                'topics': template['topics'].copy(),
                'entities': template['entities'].copy(),
                'time_sensitivity': template['time_sensitivity'],
                'publish_date': self._generate_publish_date(i),
                'author_expertise': template['expertise_level'],
                'user_ratings': {'helpful': i % 10, 'clear': (i + 3) % 10},
                'access_count': (i * 7) % 1000,
                'last_updated': self._generate_update_date(i)
            }
            
            documents.append(doc)
        
        return documents
    
    def _generate_content(self, template: Dict[str, Any], doc_index: int) -> str:
        """Generate document content based on template"""
        domain = template['domain'].replace('_', ' ')
        content_type = template['content_type']
        topics = ', '.join(template['topics'][:2])
        
        content = f"This {content_type} covers {domain} with focus on {topics}. "
        
        if template['expertise_level'] == 'beginner':
            content += "Written for beginners with clear explanations and examples. "
        elif template['expertise_level'] == 'advanced':
            content += "Advanced content for experienced practitioners with technical depth. "
        else:
            content += "Intermediate-level content balancing accessibility and detail. "
        
        content += f"Document {doc_index} provides comprehensive coverage of the topic "
        content += "with practical applications and real-world examples."
        
        return content
    
    def _generate_publish_date(self, doc_index: int) -> str:
        """Generate publish date for document"""
        base_date = datetime(2024, 1, 1)
        days_offset = doc_index * 3  # Spread documents over time
        pub_date = base_date + timedelta(days=days_offset)
        return pub_date.isoformat()
    
    def _generate_update_date(self, doc_index: int) -> str:
        """Generate last update date for document"""
        base_date = datetime(2024, 1, 1)
        days_offset = doc_index * 3 + (doc_index % 30)  # Some documents updated more recently
        update_date = base_date + timedelta(days=days_offset)
        return update_date.isoformat()
    
    def _create_document_embeddings(self) -> List[List[float]]:
        """Create embeddings for documents (simulated)"""
        import numpy as np
        
        np.random.seed(42)
        embeddings = []
        
        for doc in self.documents:
            # Create embedding based on document characteristics
            embedding = np.random.normal(0, 1, 384)
            
            # Add domain-specific bias
            if doc['domain'] == 'artificial_intelligence':
                embedding[:50] += 0.5
            elif doc['domain'] == 'business':
                embedding[50:100] += 0.5
            elif doc['domain'] == 'software_development':
                embedding[100:150] += 0.5
            
            # Add expertise level bias
            if doc['expertise_level'] == 'beginner':
                embedding[200:220] += 0.3
            elif doc['expertise_level'] == 'advanced':
                embedding[220:240] += 0.3
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    def _calculate_base_relevance(self, query: str, doc: Dict[str, Any]) -> float:
        """Calculate base relevance score"""
        query_words = set(query.lower().split())
        doc_words = set(doc['content'].lower().split())
        title_words = set(doc['title'].lower().split())
        
        # Word overlap scoring
        content_overlap = len(query_words & doc_words) / max(len(query_words | doc_words), 1)
        title_overlap = len(query_words & title_words) / max(len(query_words | title_words), 1)
        
        # Combined relevance
        relevance = content_overlap * 0.7 + title_overlap * 0.3
        
        return relevance
    
    def _score_user_context(self, doc: Dict[str, Any], context_state: ContextState) -> float:
        """Score document based on user context"""
        if not context_state.user_profile:
            return 0.5  # Neutral score
        
        profile = context_state.user_profile
        score = 0.0
        
        # Expertise level match
        doc_level = doc.get('expertise_level', 'intermediate')
        user_level = profile.expertise_level.value
        
        if doc_level == user_level:
            score += 0.4
        elif abs(self._expertise_level_numeric(doc_level) - self._expertise_level_numeric(user_level)) == 1:
            score += 0.2
        
        # Domain interest match
        doc_domain = doc.get('domain', '')
        if doc_domain in profile.primary_domains:
            score += 0.3
        
        # Communication style match (simplified)
        if profile.communication_style == 'technical' and doc.get('content_type') == 'research':
            score += 0.2
        elif profile.communication_style == 'casual' and doc.get('content_type') == 'guide':
            score += 0.2
        
        # User rating correlation
        if 'user_ratings' in doc and profile.total_interactions > 10:
            avg_rating = sum(doc['user_ratings'].values()) / len(doc['user_ratings'])
            score += (avg_rating / 10.0) * 0.1
        
        return min(score, 1.0)
    
    def _score_domain_context(self, doc: Dict[str, Any], context_state: ContextState) -> float:
        """Score document based on domain context"""
        score = 0.0
        
        # Current domain match
        if context_state.current_domain and doc.get('domain') == context_state.current_domain:
            score += 0.5
        
        # Domain concept match
        doc_topics = set(doc.get('topics', []))
        domain_concepts = set(context_state.domain_concepts)
        
        if doc_topics and domain_concepts:
            overlap = len(doc_topics & domain_concepts)
            score += (overlap / max(len(doc_topics | domain_concepts), 1)) * 0.3
        
        # Domain vocabulary match
        doc_entities = set(doc.get('entities', []))
        domain_vocab = set(context_state.domain_vocabulary.keys())
        
        if doc_entities and domain_vocab:
            overlap = len(doc_entities & domain_vocab)
            score += (overlap / max(len(doc_entities | domain_vocab), 1)) * 0.2
        
        return min(score, 1.0)
    
    def _score_temporal_context(self, doc: Dict[str, Any], context_state: ContextState) -> float:
        """Score document based on temporal context"""
        score = 0.0
        
        # Time sensitivity match
        doc_time_sensitivity = doc.get('time_sensitivity', 'medium')
        
        if context_state.time_sensitive_topics:
            if doc_time_sensitivity == 'high':
                score += 0.4
            elif doc_time_sensitivity == 'medium':
                score += 0.2
        else:
            # Prefer less time-sensitive content for general queries
            if doc_time_sensitivity == 'low':
                score += 0.3
        
        # Recency scoring
        try:
            doc_date = datetime.fromisoformat(doc.get('publish_date', '2024-01-01'))
            days_old = (context_state.current_time - doc_date).days
            
            # Exponential decay for recency (30-day half-life)
            recency_score = 2 ** (-days_old / 30.0)
            score += recency_score * 0.3
            
        except:
            score += 0.1  # Default for unparseable dates
        
        # Update recency
        try:
            update_date = datetime.fromisoformat(doc.get('last_updated', '2024-01-01'))
            days_since_update = (context_state.current_time - update_date).days
            
            update_recency = 2 ** (-days_since_update / 45.0)  # 45-day half-life for updates
            score += update_recency * 0.3
            
        except:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_entity_context(self, doc: Dict[str, Any], context_state: ContextState) -> float:
        """Score document based on entity context"""
        score = 0.0
        
        # Active entity match
        doc_entities = set(doc.get('entities', []))
        active_entities = set(context_state.get_relevant_entities())
        
        if doc_entities and active_entities:
            overlap = len(doc_entities & active_entities)
            score += (overlap / max(len(doc_entities | active_entities), 1)) * 0.6
        
        # Entity relevance weighting
        for entity in doc_entities:
            if entity in context_state.active_entities:
                entity_relevance = context_state.active_entities[entity]
                score += entity_relevance * 0.1
        
        return min(score, 1.0)
    
    def _expertise_level_numeric(self, level: str) -> int:
        """Convert expertise level to numeric for comparison"""
        level_map = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 4
        }
        return level_map.get(level, 2)

class ContextAwareRAGSystem:
    """
    Complete Context-Aware RAG System for intelligent conversational AI
    
    EXAMPLE USAGE:
    =============
    # Create context-aware RAG system
    rag = ContextAwareRAGSystem()
    await rag.initialize()
    
    # Start conversation with user context
    user_profile = UserProfile(
        user_id="user_123",
        expertise_level=UserExpertiseLevel.INTERMEDIATE,
        primary_domains=["artificial_intelligence", "software_development"]
    )
    
    session_id = await rag.start_conversation_session(user_profile)
    
    # Process queries with growing context
    response1 = await rag.contextual_query(session_id, "What is machine learning?")
    response2 = await rag.contextual_query(session_id, "How do I implement it in Python?")
    response3 = await rag.contextual_query(session_id, "What are the best libraries for this?")
    
    # Each response becomes more contextually aware and personalized
    """
    
    def __init__(self):
        # Core components
        self.context_extractor = ContextExtractor()
        self.query_enhancer = ContextualQueryEnhancer()
        self.contextual_retriever = ContextualRetriever()
        
        # Session management
        self.active_sessions: Dict[str, ContextState] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # System statistics
        self.system_stats = {
            'total_sessions': 0,
            'total_queries': 0,
            'average_session_length': 0.0,
            'context_effectiveness': 0.0,
            'user_satisfaction_rate': 0.0
        }
        
        self.logger = logging.getLogger("ContextAwareRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize context-aware RAG system"""
        self.logger.info("Context-aware RAG system initialized")
    
    async def start_conversation_session(self, user_profile: Optional[UserProfile] = None) -> str:
        """Start new conversation session"""
        
        session_id = str(uuid.uuid4())
        
        # Create context state for session
        context_state = ContextState(
            current_session_id=session_id,
            session_start_time=datetime.now(),
            user_profile=user_profile
        )
        
        # Set user expertise context
        if user_profile:
            context_state.user_expertise_context = user_profile.expertise_level
            
            # Set initial domain context
            if user_profile.primary_domains:
                context_state.current_domain = user_profile.primary_domains[0]
        
        self.active_sessions[session_id] = context_state
        
        # Store user profile
        if user_profile:
            self.user_profiles[user_profile.user_id] = user_profile
        
        self.system_stats['total_sessions'] += 1
        
        self.logger.info(f"Started conversation session: {session_id}")
        
        return session_id
    
    async def contextual_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process query with full contextual awareness"""
        
        start_time = time.time()
        self.system_stats['total_queries'] += 1
        
        if session_id not in self.active_sessions:
            return {
                'success': False,
                'error': 'Session not found',
                'session_id': session_id
            }
        
        context_state = self.active_sessions[session_id]
        
        try:
            # Create conversation turn
            turn = ConversationTurn(
                turn_id="",
                user_query=query,
                system_response="",  # Will be filled later
                timestamp=datetime.now()
            )
            
            # Extract context from current query
            await self.context_extractor.extract_conversation_context(turn)
            
            # Update context state
            context_state.add_conversation_turn(turn)
            
            # Enhance query with context
            enhanced_query = await self.query_enhancer.enhance_query(query, context_state)
            
            # Retrieve contextually relevant documents
            retrieved_docs = await self.contextual_retriever.contextual_retrieve(
                enhanced_query, 
                context_state, 
                top_k=5
            )
            
            # Generate contextual response
            response = await self._generate_contextual_response(
                query, 
                retrieved_docs, 
                context_state
            )
            
            # Update turn with response
            turn.system_response = response['content']
            turn.response_quality = response['quality_score']
            
            # Update context state with complete turn
            context_state.conversation_history[-1] = turn
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'session_id': session_id,
                'query': query,
                'enhanced_query': enhanced_query,
                'response': response['content'],
                'context_factors': {
                    'entities_extracted': turn.entities_mentioned,
                    'topics_identified': turn.topics_discussed,
                    'intent_detected': turn.intent,
                    'sentiment_score': turn.sentiment,
                    'active_entities': list(context_state.get_relevant_entities()),
                    'domain_context': context_state.current_domain,
                    'user_expertise': context_state.user_expertise_context.value if context_state.user_expertise_context else None
                },
                'retrieved_documents': len(retrieved_docs),
                'processing_time': processing_time,
                'quality_metrics': {
                    'response_quality': response['quality_score'],
                    'context_relevance': response['context_relevance'],
                    'personalization_score': response['personalization_score']
                }
            }
            
            self.logger.info(f"Contextual query processed: session={session_id}, time={processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Contextual query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'processing_time': time.time() - start_time
            }
    
    async def _generate_contextual_response(self, query: str, 
                                          retrieved_docs: List[Dict[str, Any]], 
                                          context_state: ContextState) -> Dict[str, Any]:
        """Generate response considering context"""
        
        if not retrieved_docs:
            return {
                'content': "I couldn't find relevant information to answer your question.",
                'quality_score': 0.3,
                'context_relevance': 0.0,
                'personalization_score': 0.0
            }
        
        # Build response based on context
        response_parts = []
        
        # Add contextual greeting if first interaction
        if len(context_state.conversation_history) <= 1:
            if context_state.user_profile:
                expertise = context_state.user_profile.expertise_level.value
                response_parts.append(f"Based on your {expertise}-level expertise, here's what I found:")
            else:
                response_parts.append("Here's what I found:")
        
        # Add main content from retrieved documents
        main_content = self._synthesize_document_content(retrieved_docs, context_state)
        response_parts.append(main_content)
        
        # Add contextual follow-up
        follow_up = self._generate_contextual_followup(query, context_state)
        if follow_up:
            response_parts.append(follow_up)
        
        response_content = " ".join(response_parts)
        
        # Calculate quality metrics
        quality_score = self._calculate_response_quality(response_content, retrieved_docs)
        context_relevance = self._calculate_context_relevance(retrieved_docs, context_state)
        personalization_score = self._calculate_personalization_score(retrieved_docs, context_state)
        
        return {
            'content': response_content,
            'quality_score': quality_score,
            'context_relevance': context_relevance,
            'personalization_score': personalization_score
        }
    
    def _synthesize_document_content(self, docs: List[Dict[str, Any]], 
                                   context_state: ContextState) -> str:
        """Synthesize content from retrieved documents"""
        
        if not docs:
            return "No relevant information found."
        
        # Use top documents for synthesis
        top_docs = docs[:3]
        
        # Adapt content based on user expertise
        expertise_level = "intermediate"
        if context_state.user_profile:
            expertise_level = context_state.user_profile.expertise_level.value
        
        content_parts = []
        
        for i, doc in enumerate(top_docs):
            doc_content = doc['content']
            
            # Truncate content based on expertise level
            if expertise_level == "beginner":
                content_parts.append(f"From a beginner-friendly source: {doc_content[:100]}...")
            elif expertise_level == "expert":
                content_parts.append(f"Advanced insight: {doc_content[:150]}...")
            else:
                content_parts.append(f"Key information: {doc_content[:120]}...")
        
        return " ".join(content_parts)
    
    def _generate_contextual_followup(self, query: str, context_state: ContextState) -> str:
        """Generate contextual follow-up suggestions"""
        
        follow_ups = []
        
        # Based on user expertise
        if context_state.user_profile:
            expertise = context_state.user_profile.expertise_level.value
            
            if expertise == "beginner":
                follow_ups.append("Would you like me to explain this in simpler terms?")
            elif expertise == "expert":
                follow_ups.append("Would you like more technical details or implementation specifics?")
        
        # Based on conversation history
        if len(context_state.conversation_history) > 1:
            recent_topics = []
            for turn in context_state.conversation_history[-2:]:
                recent_topics.extend(turn.topics_discussed)
            
            if recent_topics:
                topic = recent_topics[-1].replace('_', ' ')
                follow_ups.append(f"I can also help with related {topic} questions.")
        
        # Based on active entities
        relevant_entities = context_state.get_relevant_entities()
        if relevant_entities:
            entity = relevant_entities[0]
            follow_ups.append(f"Need more information about {entity}?")
        
        return " ".join(follow_ups[:2])  # Max 2 follow-ups
    
    def _calculate_response_quality(self, response: str, docs: List[Dict[str, Any]]) -> float:
        """Calculate response quality score"""
        
        # Basic quality factors
        quality = 0.5  # Base score
        
        # Length appropriateness
        response_length = len(response.split())
        if 20 <= response_length <= 200:
            quality += 0.2
        
        # Document utilization
        if docs:
            avg_doc_score = sum(doc.get('retrieval_score', 0.5) for doc in docs) / len(docs)
            quality += avg_doc_score * 0.3
        
        return min(quality, 1.0)
    
    def _calculate_context_relevance(self, docs: List[Dict[str, Any]], 
                                   context_state: ContextState) -> float:
        """Calculate how well context was utilized"""
        
        if not docs:
            return 0.0
        
        relevance = 0.0
        
        # Check if context factors were used in retrieval
        for doc in docs:
            context_factors = doc.get('context_factors', {})
            
            relevance += context_factors.get('user_match', 0.0) * 0.3
            relevance += context_factors.get('domain_match', 0.0) * 0.3
            relevance += context_factors.get('temporal_match', 0.0) * 0.2
            relevance += context_factors.get('entity_match', 0.0) * 0.2
        
        return relevance / len(docs)
    
    def _calculate_personalization_score(self, docs: List[Dict[str, Any]], 
                                       context_state: ContextState) -> float:
        """Calculate personalization effectiveness"""
        
        if not context_state.user_profile or not docs:
            return 0.0
        
        personalization = 0.0
        
        # Expertise level matching
        user_expertise = context_state.user_profile.expertise_level.value
        matching_docs = sum(1 for doc in docs if doc.get('expertise_level') == user_expertise)
        personalization += (matching_docs / len(docs)) * 0.4
        
        # Domain interest matching
        user_domains = set(context_state.user_profile.primary_domains)
        for doc in docs:
            if doc.get('domain') in user_domains:
                personalization += 0.3 / len(docs)
        
        # Communication style adaptation
        user_style = context_state.user_profile.communication_style
        for doc in docs:
            if ((user_style == 'technical' and doc.get('content_type') == 'research') or
                (user_style == 'casual' and doc.get('content_type') == 'guide')):
                personalization += 0.3 / len(docs)
        
        return min(personalization, 1.0)
    
    def end_conversation_session(self, session_id: str) -> Dict[str, Any]:
        """End conversation session and return analytics"""
        
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        context_state = self.active_sessions[session_id]
        
        # Calculate session metrics
        session_duration = (datetime.now() - context_state.session_start_time).total_seconds()
        total_turns = len(context_state.conversation_history)
        
        # Update system statistics
        current_avg_length = self.system_stats['average_session_length']
        session_count = self.system_stats['total_sessions']
        
        self.system_stats['average_session_length'] = (
            (current_avg_length * (session_count - 1) + session_duration) / session_count
        )
        
        # Clean up session
        del self.active_sessions[session_id]
        
        session_summary = {
            'success': True,
            'session_id': session_id,
            'session_duration': session_duration,
            'total_turns': total_turns,
            'topics_discussed': list(set(
                topic for turn in context_state.conversation_history 
                for topic in turn.topics_discussed
            )),
            'entities_mentioned': list(set(
                entity for turn in context_state.conversation_history 
                for entity in turn.entities_mentioned
            )),
            'average_sentiment': sum(turn.sentiment for turn in context_state.conversation_history) / max(total_turns, 1),
            'user_profile_updated': context_state.user_profile is not None
        }
        
        self.logger.info(f"Ended conversation session: {session_id}")
        
        return session_summary
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            'system_stats': self.system_stats,
            'active_sessions': len(self.active_sessions),
            'registered_users': len(self.user_profiles),
            'capabilities': {
                'contextual_understanding': True,
                'conversation_memory': True,
                'user_personalization': True,
                'domain_adaptation': True,
                'temporal_awareness': True,
                'entity_tracking': True,
                'sentiment_awareness': True
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_context_extraction():
    """Demo: Context extraction from conversations"""
    print("\nDEMO 1: CONTEXT EXTRACTION")
    print("=" * 50)
    
    extractor = ContextExtractor()
    
    test_queries = [
        "What is machine learning and how does it work?",
        "Can you help me with this TensorFlow error I'm getting?",
        "That explanation was confusing. Can you simplify it?",
        "Great! Now how do I implement this in Python?",
        "I'm frustrated with these complex algorithms. Is there an easier approach?"
    ]
    
    print("Extracting context from conversation turns:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Query: {query}")
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=f"turn_{i}",
            user_query=query,
            system_response="",
            timestamp=datetime.now()
        )
        
        # Extract context
        await extractor.extract_conversation_context(turn)
        
        print(f"Entities: {turn.entities_mentioned}")
        print(f"Topics: {turn.topics_discussed}")
        print(f"Intent: {turn.intent}")
        print(f"Sentiment: {turn.sentiment:.2f}")

async def demo_user_profiles():
    """Demo: User profile creation and adaptation"""
    print("\nDEMO 2: USER PROFILES AND PERSONALIZATION")
    print("=" * 50)
    
    # Create different user profiles
    profiles = [
        UserProfile(
            user_id="beginner_user",
            expertise_level=UserExpertiseLevel.BEGINNER,
            communication_style="casual",
            primary_domains=["software_development"],
            interests=["Python", "web development"]
        ),
        UserProfile(
            user_id="expert_user", 
            expertise_level=UserExpertiseLevel.EXPERT,
            communication_style="technical",
            primary_domains=["artificial_intelligence", "machine_learning"],
            interests=["deep learning", "neural networks", "research"]
        ),
        UserProfile(
            user_id="business_user",
            expertise_level=UserExpertiseLevel.INTERMEDIATE,
            communication_style="formal",
            primary_domains=["business", "strategy"],
            interests=["market analysis", "growth", "KPIs"]
        )
    ]
    
    print("User profiles for personalization:")
    
    for profile in profiles:
        print(f"\n--- {profile.user_id} ---")
        print(f"Expertise: {profile.expertise_level.value}")
        print(f"Style: {profile.communication_style}")
        print(f"Domains: {', '.join(profile.primary_domains)}")
        print(f"Interests: {', '.join(profile.interests[:3])}")
        
        # Simulate interaction and profile update
        turn = ConversationTurn(
            turn_id="",
            user_query="How does AI work in business applications?",
            system_response="AI helps automate business processes...",
            timestamp=datetime.now(),
            topics_discussed=["artificial_intelligence", "business"],
            user_satisfaction=True
        )
        
        profile.update_from_interaction(turn)
        
        print(f"After interaction - Total: {profile.total_interactions}, Success: {profile.successful_interactions}")

async def demo_contextual_enhancement():
    """Demo: Query enhancement with context"""
    print("\nDEMO 3: CONTEXTUAL QUERY ENHANCEMENT")
    print("=" * 50)
    
    enhancer = ContextualQueryEnhancer()
    
    # Create context state with conversation history
    context_state = ContextState()
    context_state.current_domain = "artificial_intelligence"
    context_state.domain_concepts = ["machine learning", "neural networks", "algorithms"]
    context_state.active_entities = {"Python": 0.8, "TensorFlow": 0.6, "algorithms": 0.4}
    
    # Add user profile
    context_state.user_profile = UserProfile(
        user_id="test_user",
        expertise_level=UserExpertiseLevel.INTERMEDIATE,
        primary_domains=["artificial_intelligence"]
    )
    
    # Add conversation history
    previous_turns = [
        ConversationTurn(
            turn_id="turn1",
            user_query="What is machine learning?",
            system_response="Machine learning is a subset of AI...",
            timestamp=datetime.now() - timedelta(minutes=5),
            topics_discussed=["machine_learning", "artificial_intelligence"],
            entities_mentioned=["algorithms", "data"]
        ),
        ConversationTurn(
            turn_id="turn2", 
            user_query="How do I start with it?",
            system_response="To start with machine learning...",
            timestamp=datetime.now() - timedelta(minutes=2),
            topics_discussed=["machine_learning", "beginner"],
            entities_mentioned=["Python", "tutorials"]
        )
    ]
    
    for turn in previous_turns:
        context_state.add_conversation_turn(turn)
    
    # Test query enhancement
    test_queries = [
        "What libraries should I use?",
        "Can you explain neural networks?",
        "How do I implement this?",
        "What's the difference between supervised and unsupervised learning?"
    ]
    
    print("Query enhancement with context:")
    
    for query in test_queries:
        print(f"\nOriginal: {query}")
        
        enhanced = await enhancer.enhance_query(query, context_state)
        print(f"Enhanced: {enhanced[:150]}...")

async def demo_contextual_retrieval():
    """Demo: Context-aware document retrieval"""
    print("\nDEMO 4: CONTEXTUAL DOCUMENT RETRIEVAL")
    print("=" * 50)
    
    retriever = ContextualRetriever()
    
    # Create context states for different scenarios
    scenarios = [
        {
            'name': 'Beginner User - AI Query',
            'context': ContextState(
                user_profile=UserProfile(
                    user_id="beginner",
                    expertise_level=UserExpertiseLevel.BEGINNER,
                    primary_domains=["artificial_intelligence"]
                ),
                current_domain="artificial_intelligence"
            ),
            'query': 'artificial intelligence machine learning basics tutorial'
        },
        {
            'name': 'Expert User - Technical Query',
            'context': ContextState(
                user_profile=UserProfile(
                    user_id="expert",
                    expertise_level=UserExpertiseLevel.EXPERT,
                    primary_domains=["artificial_intelligence"]
                ),
                current_domain="artificial_intelligence",
                active_entities={"transformers": 0.9, "attention": 0.7}
            ),
            'query': 'advanced deep learning transformers attention mechanisms research'
        },
        {
            'name': 'Business User - Strategy Query',
            'context': ContextState(
                user_profile=UserProfile(
                    user_id="business",
                    expertise_level=UserExpertiseLevel.INTERMEDIATE,
                    primary_domains=["business"]
                ),
                current_domain="business"
            ),
            'query': 'business strategy market analysis competitive advantage'
        }
    ]
    
    print("Contextual retrieval for different user scenarios:")
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Query: {scenario['query']}")
        
        docs = await retriever.contextual_retrieve(
            scenario['query'], 
            scenario['context'], 
            top_k=3
        )
        
        print(f"Retrieved {len(docs)} documents:")
        
        for i, doc in enumerate(docs, 1):
            score = doc.get('retrieval_score', 0.0)
            expertise = doc.get('expertise_level', 'unknown')
            domain = doc.get('domain', 'unknown')
            
            print(f"  {i}. {doc['title']}")
            print(f"     Score: {score:.3f}, Expertise: {expertise}, Domain: {domain}")
            
            context_factors = doc.get('context_factors', {})
            print(f"     Context matches - User: {context_factors.get('user_match', 0):.2f}, "
                  f"Domain: {context_factors.get('domain_match', 0):.2f}")

async def demo_conversation_flow():
    """Demo: Complete conversational flow with context"""
    print("\nDEMO 5: COMPLETE CONVERSATIONAL FLOW")
    print("=" * 50)
    
    rag_system = ContextAwareRAGSystem()
    await rag_system.initialize()
    
    # Create user profile
    user_profile = UserProfile(
        user_id="demo_user",
        expertise_level=UserExpertiseLevel.INTERMEDIATE,
        communication_style="technical",
        primary_domains=["artificial_intelligence", "software_development"]
    )
    
    # Start conversation session
    session_id = await rag_system.start_conversation_session(user_profile)
    
    print(f"Started conversation session: {session_id}")
    print(f"User profile: {user_profile.expertise_level.value} level, domains: {', '.join(user_profile.primary_domains)}")
    
    # Conversation flow
    conversation = [
        "What is machine learning?",
        "How do I implement it in Python?", 
        "What libraries are best for this?",
        "Can you show me a simple example?",
        "What about deep learning? How is it different?"
    ]
    
    print(f"\nConversation flow with growing context:")
    
    for i, query in enumerate(conversation, 1):
        print(f"\n{'='*60}")
        print(f"TURN {i}")
        print(f"{'='*60}")
        print(f"User: {query}")
        
        result = await rag_system.contextual_query(session_id, query)
        
        if result['success']:
            print(f"\nSystem Response:")
            print(result['response'][:200] + "...")
            
            print(f"\nContext Analysis:")
            factors = result['context_factors']
            print(f"  Entities extracted: {factors['entities_extracted']}")
            print(f"  Topics identified: {factors['topics_identified']}")
            print(f"  Intent detected: {factors['intent_detected']}")
            print(f"  Sentiment: {factors['sentiment_score']:.2f}")
            print(f"  Active entities: {factors['active_entities']}")
            
            print(f"\nQuality Metrics:")
            metrics = result['quality_metrics']
            print(f"  Response quality: {metrics['response_quality']:.2f}")
            print(f"  Context relevance: {metrics['context_relevance']:.2f}")
            print(f"  Personalization: {metrics['personalization_score']:.2f}")
            
        else:
            print(f"Error: {result['error']}")
    
    # End session
    session_summary = rag_system.end_conversation_session(session_id)
    
    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Duration: {session_summary['session_duration']:.1f} seconds")
    print(f"Total turns: {session_summary['total_turns']}")
    print(f"Topics discussed: {', '.join(session_summary['topics_discussed'])}")
    print(f"Average sentiment: {session_summary['average_sentiment']:.2f}")

async def demo_system_analytics():
    """Demo: System analytics and performance"""
    print("\nDEMO 6: SYSTEM ANALYTICS")
    print("=" * 50)
    
    rag_system = ContextAwareRAGSystem()
    await rag_system.initialize()
    
    # Simulate multiple user sessions
    test_scenarios = [
        {
            'profile': UserProfile(
                user_id="user_1",
                expertise_level=UserExpertiseLevel.BEGINNER,
                primary_domains=["software_development"]
            ),
            'queries': ["What is Python?", "How do I start coding?", "What are variables?"]
        },
        {
            'profile': UserProfile(
                user_id="user_2", 
                expertise_level=UserExpertiseLevel.EXPERT,
                primary_domains=["artificial_intelligence"]
            ),
            'queries': ["Latest advances in transformer architectures", "BERT vs GPT comparison", "Attention mechanism optimization"]
        },
        {
            'profile': UserProfile(
                user_id="user_3",
                expertise_level=UserExpertiseLevel.INTERMEDIATE,
                primary_domains=["business"]
            ),
            'queries': ["AI impact on business", "Digital transformation strategy", "ROI of automation"]
        }
    ]
    
    print("Simulating multiple conversation sessions...")
    
    session_results = []
    for scenario in test_scenarios:
        session_id = await rag_system.start_conversation_session(scenario['profile'])
        
        for query in scenario['queries']:
            result = await rag_system.contextual_query(session_id, query)
            session_results.append(result)
        
        session_summary = rag_system.end_conversation_session(session_id)
        print(f"  ✓ Session completed: {scenario['profile'].user_id}")
    
    # Get system statistics
    stats = rag_system.get_system_statistics()
    
    print(f"\nCONTEXT-AWARE RAG SYSTEM ANALYTICS")
    print("=" * 40)
    
    print(f"\nSystem Performance:")
    system_stats = stats['system_stats']
    print(f"  Total sessions: {system_stats['total_sessions']}")
    print(f"  Total queries: {system_stats['total_queries']}")
    print(f"  Average session length: {system_stats['average_session_length']:.1f}s")
    
    print(f"\nSystem State:")
    print(f"  Active sessions: {stats['active_sessions']}")
    print(f"  Registered users: {stats['registered_users']}")
    
    print(f"\nSystem Capabilities:")
    capabilities = stats['capabilities']
    for capability, enabled in capabilities.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {capability.replace('_', ' ').title()}")
    
    print(f"\nQuery Processing Analysis:")
    successful_queries = [r for r in session_results if r['success']]
    if successful_queries:
        avg_processing_time = sum(r['processing_time'] for r in successful_queries) / len(successful_queries)
        avg_docs_retrieved = sum(r['retrieved_documents'] for r in successful_queries) / len(successful_queries)
        
        print(f"  Success rate: {len(successful_queries)}/{len(session_results)} ({len(successful_queries)/len(session_results)*100:.1f}%)")
        print(f"  Average processing time: {avg_processing_time:.3f}s")
        print(f"  Average documents retrieved: {avg_docs_retrieved:.1f}")
        
        # Quality metrics
        quality_metrics = [r['quality_metrics'] for r in successful_queries if 'quality_metrics' in r]
        if quality_metrics:
            avg_response_quality = sum(m['response_quality'] for m in quality_metrics) / len(quality_metrics)
            avg_context_relevance = sum(m['context_relevance'] for m in quality_metrics) / len(quality_metrics)
            avg_personalization = sum(m['personalization_score'] for m in quality_metrics) / len(quality_metrics)
            
            print(f"\nQuality Metrics:")
            print(f"  Average response quality: {avg_response_quality:.2f}")
            print(f"  Average context relevance: {avg_context_relevance:.2f}")
            print(f"  Average personalization: {avg_personalization:.2f}")

async def main():
    """
    Demonstrate Context-Aware RAG for intelligent conversational AI
    
    WHAT YOU'LL LEARN:
    ================
    1. How to extract and track conversational context
    2. How to build user profiles for personalization
    3. How to enhance queries with contextual information
    4. How to implement context-aware document retrieval
    5. How to create natural conversational AI experiences
    
    REAL WORLD APPLICATIONS:
    =======================
    - Intelligent customer service chatbots
    - Personal AI assistants and companions
    - Educational tutoring systems with adaptive learning
    - Technical support systems with expertise adaptation
    - Business intelligence assistants with domain knowledge
    - Healthcare AI with patient history awareness
    """
    
    print("CONTEXT-AWARE RAG DEMONSTRATION")
    print("Building intelligent conversational AI with memory and personalization!")
    
    await demo_context_extraction()
    await demo_user_profiles()
    await demo_contextual_enhancement()
    await demo_contextual_retrieval()
    await demo_conversation_flow()
    await demo_system_analytics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Context extraction enables understanding of conversation flow")
    print("✓ User profiles provide personalization and adaptive responses")
    print("✓ Query enhancement incorporates conversational and user context")
    print("✓ Contextual retrieval finds more relevant and personalized content")
    print("✓ Complete systems enable natural, human-like AI interactions")
    print("✓ Analytics help optimize context utilization and user satisfaction")
    print("\nTHE POWER OF CONTEXT-AWARE RAG:")
    print("- Enables natural, conversational AI that remembers and learns")
    print("- Provides personalized experiences adapted to individual users")
    print("- Supports complex multi-turn problem solving")
    print("- Powers next-generation intelligent assistants and companions")

if __name__ == "__main__":
    asyncio.run(main())
