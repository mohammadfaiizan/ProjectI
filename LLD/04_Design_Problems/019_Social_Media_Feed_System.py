"""
SOCIAL MEDIA FEED SYSTEM - Complete System Design
================================================

Problem Statement:
Design a comprehensive social media feed system that handles:
- User posts creation and management (text, images, videos)
- Feed generation and personalization algorithms
- Real-time updates and notifications
- User interactions (likes, comments, shares, reactions)
- Following/follower relationships
- Content moderation and filtering
- Trending topics and hashtag systems
- Privacy controls and content visibility
- Feed caching and performance optimization
- Analytics and engagement tracking

Requirements:
- Support multiple content types (text, images, videos, links)
- Implement feed ranking algorithms (chronological, algorithmic)
- Handle real-time feed updates efficiently
- Support user interactions and engagement metrics
- Implement follower/following system with privacy controls
- Provide content moderation and spam filtering
- Support hashtags and mentions
- Handle high-scale feed generation and delivery
- Implement feed caching strategies
- Provide comprehensive analytics and insights

Design Patterns Used:
- Strategy: Feed ranking and content filtering strategies
- Observer: Real-time feed updates and notifications
- Factory: Post and interaction creation
- Decorator: Content moderation and filtering
- Command: User actions with undo capability
- Repository: Data persistence abstraction
- Facade: Simplified feed API
- Proxy: Content delivery and caching proxy
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
import hashlib
import json
import re
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import random


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class PostType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"
    POLL = "poll"
    REPOST = "repost"


class InteractionType(Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    LOVE = "love"
    LAUGH = "laugh"
    ANGRY = "angry"
    COMMENT = "comment"
    SHARE = "share"
    SAVE = "save"


class PostStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"
    BLOCKED = "blocked"


class PrivacyLevel(Enum):
    PUBLIC = "public"
    FRIENDS = "friends"
    PRIVATE = "private"
    CUSTOM = "custom"


class RelationshipType(Enum):
    FOLLOWING = "following"
    FOLLOWER = "follower"
    FRIEND = "friend"
    BLOCKED = "blocked"
    MUTED = "muted"


class FeedType(Enum):
    HOME = "home"
    TRENDING = "trending"
    FOLLOWING = "following"
    DISCOVER = "discover"
    HASHTAG = "hashtag"


@dataclass
class User:
    """Social media user."""
    user_id: str
    username: str
    email: str
    display_name: str
    bio: str = ""
    profile_image_url: str = ""
    cover_image_url: str = ""
    
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    # Privacy settings
    is_private: bool = False
    allow_messages: bool = True
    allow_tags: bool = True
    
    # Statistics
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0
    
    # Verification
    is_verified: bool = False
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())


@dataclass
class Post:
    """Social media post."""
    post_id: str
    user_id: str
    content: str
    post_type: PostType = PostType.TEXT
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    # Content metadata
    media_urls: List[str] = field(default_factory=list)
    hashtags: Set[str] = field(default_factory=set)
    mentions: Set[str] = field(default_factory=set)
    
    # Engagement
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    views_count: int = 0
    
    # Settings
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    allow_comments: bool = True
    allow_shares: bool = True
    
    # Status
    status: PostStatus = PostStatus.PUBLISHED
    
    # Location and scheduling
    location: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    
    # Repost information
    original_post_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.post_id:
            self.post_id = str(uuid.uuid4())
        
        # Extract hashtags and mentions from content
        self._extract_hashtags_and_mentions()
    
    def _extract_hashtags_and_mentions(self):
        """Extract hashtags and mentions from content."""
        # Extract hashtags (#tag)
        hashtag_pattern = r'#(\w+)'
        self.hashtags.update(re.findall(hashtag_pattern, self.content, re.IGNORECASE))
        
        # Extract mentions (@username)
        mention_pattern = r'@(\w+)'
        self.mentions.update(re.findall(mention_pattern, self.content, re.IGNORECASE))


@dataclass
class Interaction:
    """User interaction with a post."""
    interaction_id: str
    user_id: str
    post_id: str
    interaction_type: InteractionType
    
    created_at: datetime = field(default_factory=datetime.now)
    
    # Additional data for specific interactions
    comment_text: Optional[str] = None
    reply_to_interaction_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())


@dataclass
class UserRelationship:
    """Relationship between users."""
    from_user_id: str
    to_user_id: str
    relationship_type: RelationshipType
    created_at: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    is_close_friend: bool = False
    notifications_enabled: bool = True


@dataclass
class FeedItem:
    """Item in a user's feed."""
    post: Post
    score: float = 0.0
    reason: str = ""
    shown_at: Optional[datetime] = None
    interacted: bool = False


@dataclass
class TrendingTopic:
    """Trending hashtag or topic."""
    topic: str
    post_count: int
    engagement_score: float
    trend_start: datetime
    category: str = "general"


# ============================================================================
# FEED RANKING STRATEGIES
# ============================================================================

class FeedRankingStrategy(ABC):
    """Abstract feed ranking strategy."""
    
    @abstractmethod
    def rank_posts(self, posts: List[Post], user: User, 
                  user_relationships: Dict[str, UserRelationship],
                  user_interactions: List[Interaction]) -> List[FeedItem]:
        """Rank posts for user feed."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class ChronologicalRanking(FeedRankingStrategy):
    """Chronological feed ranking (newest first)."""
    
    def rank_posts(self, posts: List[Post], user: User,
                  user_relationships: Dict[str, UserRelationship],
                  user_interactions: List[Interaction]) -> List[FeedItem]:
        """Rank posts chronologically."""
        sorted_posts = sorted(posts, key=lambda p: p.created_at, reverse=True)
        
        feed_items = []
        for post in sorted_posts:
            feed_item = FeedItem(
                post=post,
                score=post.created_at.timestamp(),
                reason="chronological"
            )
            feed_items.append(feed_item)
        
        return feed_items
    
    def get_strategy_name(self) -> str:
        return "Chronological"


class EngagementBasedRanking(FeedRankingStrategy):
    """Engagement-based feed ranking."""
    
    def rank_posts(self, posts: List[Post], user: User,
                  user_relationships: Dict[str, UserRelationship],
                  user_interactions: List[Interaction]) -> List[FeedItem]:
        """Rank posts based on engagement."""
        feed_items = []
        
        for post in posts:
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(post)
            
            # Apply time decay
            time_decay = self._calculate_time_decay(post.created_at)
            
            # Apply relationship bonus
            relationship_bonus = self._calculate_relationship_bonus(
                post.user_id, user.user_id, user_relationships
            )
            
            total_score = engagement_score * time_decay * relationship_bonus
            
            feed_item = FeedItem(
                post=post,
                score=total_score,
                reason=f"engagement_score_{engagement_score:.2f}"
            )
            feed_items.append(feed_item)
        
        # Sort by score
        feed_items.sort(key=lambda item: item.score, reverse=True)
        return feed_items
    
    def _calculate_engagement_score(self, post: Post) -> float:
        """Calculate post engagement score."""
        # Weighted engagement score
        likes_weight = 1.0
        comments_weight = 3.0  # Comments are more valuable
        shares_weight = 5.0    # Shares are most valuable
        
        score = (post.likes_count * likes_weight +
                post.comments_count * comments_weight +
                post.shares_count * shares_weight)
        
        return max(1.0, score)  # Minimum score of 1
    
    def _calculate_time_decay(self, post_time: datetime) -> float:
        """Calculate time decay factor."""
        hours_old = (datetime.now() - post_time).total_seconds() / 3600
        
        # Exponential decay: newer posts get higher scores
        decay_factor = 2 ** (-hours_old / 24)  # Half-life of 24 hours
        
        return max(0.1, decay_factor)  # Minimum decay of 0.1
    
    def _calculate_relationship_bonus(self, post_user_id: str, feed_user_id: str,
                                    relationships: Dict[str, UserRelationship]) -> float:
        """Calculate relationship bonus."""
        if post_user_id == feed_user_id:
            return 1.0  # Own posts
        
        rel_key = f"{feed_user_id}:{post_user_id}"
        if rel_key in relationships:
            relationship = relationships[rel_key]
            
            if relationship.relationship_type == RelationshipType.FRIEND:
                return 2.0 if relationship.is_close_friend else 1.5
            elif relationship.relationship_type == RelationshipType.FOLLOWING:
                return 1.3
        
        return 1.0  # Default bonus
    
    def get_strategy_name(self) -> str:
        return "Engagement-Based"


class PersonalizedRanking(FeedRankingStrategy):
    """Personalized feed ranking using user behavior."""
    
    def rank_posts(self, posts: List[Post], user: User,
                  user_relationships: Dict[str, UserRelationship],
                  user_interactions: List[Interaction]) -> List[FeedItem]:
        """Rank posts using personalization."""
        # Build user interest profile
        user_interests = self._build_user_interests(user_interactions)
        
        feed_items = []
        
        for post in posts:
            # Calculate content similarity score
            content_score = self._calculate_content_similarity(post, user_interests)
            
            # Calculate user affinity score
            affinity_score = self._calculate_user_affinity(
                post.user_id, user.user_id, user_interactions
            )
            
            # Apply recency factor
            recency_factor = self._calculate_recency_factor(post.created_at)
            
            # Combine scores
            total_score = content_score * affinity_score * recency_factor
            
            feed_item = FeedItem(
                post=post,
                score=total_score,
                reason=f"personalized_{content_score:.2f}_{affinity_score:.2f}"
            )
            feed_items.append(feed_item)
        
        # Sort by score
        feed_items.sort(key=lambda item: item.score, reverse=True)
        return feed_items
    
    def _build_user_interests(self, interactions: List[Interaction]) -> Dict[str, float]:
        """Build user interest profile from interactions."""
        interests = defaultdict(float)
        
        # Weight different interaction types
        interaction_weights = {
            InteractionType.LIKE: 1.0,
            InteractionType.LOVE: 2.0,
            InteractionType.COMMENT: 3.0,
            InteractionType.SHARE: 5.0,
            InteractionType.SAVE: 4.0
        }
        
        for interaction in interactions:
            weight = interaction_weights.get(interaction.interaction_type, 1.0)
            
            # For simplicity, use post_id as interest indicator
            # In reality, you'd analyze post content, hashtags, etc.
            interests[interaction.post_id] += weight
        
        return dict(interests)
    
    def _calculate_content_similarity(self, post: Post, user_interests: Dict[str, float]) -> float:
        """Calculate content similarity to user interests."""
        if not user_interests:
            return 1.0
        
        # Simple similarity based on hashtags
        score = 1.0
        
        for hashtag in post.hashtags:
            if hashtag in user_interests:
                score += user_interests[hashtag] * 0.1
        
        return score
    
    def _calculate_user_affinity(self, post_user_id: str, feed_user_id: str,
                               interactions: List[Interaction]) -> float:
        """Calculate user affinity score."""
        if post_user_id == feed_user_id:
            return 1.5  # Slightly boost own posts
        
        # Count interactions with this user's posts
        interaction_count = sum(1 for interaction in interactions
                              if interaction.post_id.startswith(post_user_id))
        
        # Logarithmic scaling
        import math
        affinity = 1.0 + math.log(interaction_count + 1) * 0.1
        
        return min(affinity, 3.0)  # Cap at 3.0
    
    def _calculate_recency_factor(self, post_time: datetime) -> float:
        """Calculate recency factor with gentle decay."""
        hours_old = (datetime.now() - post_time).total_seconds() / 3600
        
        # Gentler decay than pure chronological
        recency_factor = 1.0 / (1.0 + hours_old / 48)  # 48-hour half-life
        
        return max(0.2, recency_factor)
    
    def get_strategy_name(self) -> str:
        return "Personalized"


# ============================================================================
# CONTENT MODERATION
# ============================================================================

class ContentModerator:
    """Content moderation system."""
    
    def __init__(self):
        self.spam_keywords = {
            'spam', 'promotion', 'buy now', 'click here', 'free money',
            'get rich quick', 'miracle cure', 'lose weight fast'
        }
        
        self.inappropriate_keywords = {
            'hate', 'violence', 'abuse', 'harassment', 'discrimination'
        }
        
        self.blocked_domains = {
            'malicious-site.com', 'spam-site.net', 'phishing-site.org'
        }
    
    def moderate_post(self, post: Post) -> Tuple[bool, List[str]]:
        """Moderate post content."""
        issues = []
        
        # Check for spam
        if self._is_spam(post):
            issues.append("Potential spam content")
        
        # Check for inappropriate content
        if self._is_inappropriate(post):
            issues.append("Inappropriate content detected")
        
        # Check for malicious links
        if self._has_malicious_links(post):
            issues.append("Malicious links detected")
        
        # Check for excessive hashtags
        if len(post.hashtags) > 10:
            issues.append("Excessive hashtags")
        
        # Check for duplicate content (simplified)
        if self._is_duplicate_content(post):
            issues.append("Duplicate content")
        
        is_approved = len(issues) == 0
        return is_approved, issues
    
    def _is_spam(self, post: Post) -> bool:
        """Check if post is spam."""
        content_lower = post.content.lower()
        
        # Check for spam keywords
        spam_count = sum(1 for keyword in self.spam_keywords 
                        if keyword in content_lower)
        
        # Check for excessive capitals
        if len(post.content) > 10:
            capital_ratio = sum(1 for c in post.content if c.isupper()) / len(post.content)
            if capital_ratio > 0.7:
                spam_count += 1
        
        # Check for excessive repeated characters
        if re.search(r'(.)\1{4,}', post.content):
            spam_count += 1
        
        return spam_count >= 2
    
    def _is_inappropriate(self, post: Post) -> bool:
        """Check if post contains inappropriate content."""
        content_lower = post.content.lower()
        
        return any(keyword in content_lower for keyword in self.inappropriate_keywords)
    
    def _has_malicious_links(self, post: Post) -> bool:
        """Check for malicious links."""
        # Extract URLs from content
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'
        urls = re.findall(url_pattern, post.content)
        
        for url in urls:
            for domain in self.blocked_domains:
                if domain in url:
                    return True
        
        return False
    
    def _is_duplicate_content(self, post: Post) -> bool:
        """Check for duplicate content (simplified)."""
        # In a real system, you'd check against existing posts
        # For demo, just check if content is too repetitive
        words = post.content.split()
        if len(words) > 5:
            unique_words = set(words)
            uniqueness_ratio = len(unique_words) / len(words)
            return uniqueness_ratio < 0.3
        
        return False


# ============================================================================
# TRENDING ANALYSIS
# ============================================================================

class TrendingAnalyzer:
    """Analyze trending topics and hashtags."""
    
    def __init__(self):
        self.hashtag_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.topic_engagement: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def record_hashtag_usage(self, hashtags: Set[str], engagement_score: float = 1.0) -> None:
        """Record hashtag usage."""
        with self._lock:
            now = datetime.now()
            
            for hashtag in hashtags:
                self.hashtag_counts[hashtag].append(now)
                self.topic_engagement[hashtag] += engagement_score
    
    def get_trending_topics(self, time_window: timedelta = timedelta(hours=24),
                          limit: int = 10) -> List[TrendingTopic]:
        """Get trending topics in time window."""
        with self._lock:
            cutoff_time = datetime.now() - time_window
            trending_topics = []
            
            for hashtag, timestamps in self.hashtag_counts.items():
                # Count recent posts
                recent_posts = [t for t in timestamps if t > cutoff_time]
                post_count = len(recent_posts)
                
                if post_count < 2:  # Minimum threshold
                    continue
                
                # Calculate trend score
                trend_score = self._calculate_trend_score(
                    recent_posts, self.topic_engagement[hashtag]
                )
                
                # Find trend start
                trend_start = min(recent_posts) if recent_posts else datetime.now()
                
                trending_topic = TrendingTopic(
                    topic=hashtag,
                    post_count=post_count,
                    engagement_score=trend_score,
                    trend_start=trend_start
                )
                
                trending_topics.append(trending_topic)
            
            # Sort by engagement score and return top topics
            trending_topics.sort(key=lambda t: t.engagement_score, reverse=True)
            return trending_topics[:limit]
    
    def _calculate_trend_score(self, timestamps: List[datetime], 
                             total_engagement: float) -> float:
        """Calculate trend score based on velocity and engagement."""
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate velocity (posts per hour)
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        velocity = len(timestamps) / max(time_span, 0.1)
        
        # Combine velocity and engagement
        trend_score = velocity * total_engagement
        
        return trend_score
    
    def cleanup_old_data(self, retention_period: timedelta = timedelta(days=7)) -> None:
        """Clean up old hashtag data."""
        with self._lock:
            cutoff_time = datetime.now() - retention_period
            
            for hashtag in list(self.hashtag_counts.keys()):
                self.hashtag_counts[hashtag] = [
                    t for t in self.hashtag_counts[hashtag] if t > cutoff_time
                ]
                
                # Remove hashtags with no recent activity
                if not self.hashtag_counts[hashtag]:
                    del self.hashtag_counts[hashtag]
                    self.topic_engagement.pop(hashtag, None)


# ============================================================================
# FEED CACHE
# ============================================================================

class FeedCache:
    """Feed caching system for performance."""
    
    def __init__(self, cache_ttl: timedelta = timedelta(minutes=15)):
        self.cache: Dict[str, Tuple[List[FeedItem], datetime]] = {}
        self.cache_ttl = cache_ttl
        self._lock = threading.Lock()
    
    def get_cached_feed(self, user_id: str, feed_type: FeedType) -> Optional[List[FeedItem]]:
        """Get cached feed for user."""
        with self._lock:
            cache_key = f"{user_id}:{feed_type.value}"
            
            if cache_key in self.cache:
                feed_items, cached_at = self.cache[cache_key]
                
                # Check if cache is still valid
                if datetime.now() - cached_at < self.cache_ttl:
                    return feed_items
                else:
                    # Remove expired cache
                    del self.cache[cache_key]
            
            return None
    
    def cache_feed(self, user_id: str, feed_type: FeedType, feed_items: List[FeedItem]) -> None:
        """Cache feed for user."""
        with self._lock:
            cache_key = f"{user_id}:{feed_type.value}"
            self.cache[cache_key] = (feed_items, datetime.now())
    
    def invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate all cached feeds for user."""
        with self._lock:
            keys_to_remove = [key for key in self.cache.keys() 
                             if key.startswith(f"{user_id}:")]
            
            for key in keys_to_remove:
                del self.cache[key]
    
    def invalidate_feed_type(self, feed_type: FeedType) -> None:
        """Invalidate all cached feeds of specific type."""
        with self._lock:
            keys_to_remove = [key for key in self.cache.keys() 
                             if key.endswith(f":{feed_type.value}")]
            
            for key in keys_to_remove:
                del self.cache[key]
    
    def cleanup_expired_cache(self) -> None:
        """Clean up expired cache entries."""
        with self._lock:
            now = datetime.now()
            expired_keys = []
            
            for key, (_, cached_at) in self.cache.items():
                if now - cached_at >= self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]


# ============================================================================
# MAIN SOCIAL MEDIA SYSTEM
# ============================================================================

class SocialMediaFeedSystem:
    """Main social media feed system."""
    
    def __init__(self):
        # Data storage (in-memory for demo)
        self.users: Dict[str, User] = {}
        self.posts: Dict[str, Post] = {}
        self.interactions: Dict[str, List[Interaction]] = defaultdict(list)
        self.relationships: Dict[str, UserRelationship] = {}
        
        # System components
        self.moderator = ContentModerator()
        self.trending_analyzer = TrendingAnalyzer()
        self.feed_cache = FeedCache()
        
        # Feed ranking strategies
        self.ranking_strategies = {
            'chronological': ChronologicalRanking(),
            'engagement': EngagementBasedRanking(),
            'personalized': PersonalizedRanking()
        }
        
        # Threading
        self._lock = threading.RLock()
        
        # Analytics
        self.analytics = {
            'posts_created': 0,
            'interactions_created': 0,
            'feeds_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print("📱 Social Media Feed System initialized")
    
    def create_user(self, username: str, email: str, display_name: str) -> User:
        """Create a new user."""
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            display_name=display_name
        )
        
        with self._lock:
            self.users[user.user_id] = user
        
        return user
    
    def create_post(self, user_id: str, content: str, post_type: PostType = PostType.TEXT,
                   media_urls: List[str] = None, privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC,
                   **kwargs) -> Post:
        """Create a new post."""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        post = Post(
            post_id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            post_type=post_type,
            media_urls=media_urls or [],
            privacy_level=privacy_level,
            **kwargs
        )
        
        # Moderate content
        is_approved, issues = self.moderator.moderate_post(post)
        if not is_approved:
            post.status = PostStatus.BLOCKED
            print(f"Post blocked: {', '.join(issues)}")
        
        with self._lock:
            self.posts[post.post_id] = post
            self.users[user_id].posts_count += 1
            self.analytics['posts_created'] += 1
        
        # Record hashtag usage for trending
        if post.hashtags and post.status == PostStatus.PUBLISHED:
            self.trending_analyzer.record_hashtag_usage(post.hashtags)
        
        # Invalidate relevant caches
        self._invalidate_feeds_for_new_post(user_id)
        
        return post
    
    def create_interaction(self, user_id: str, post_id: str, 
                         interaction_type: InteractionType, **kwargs) -> Interaction:
        """Create a user interaction with a post."""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        if post_id not in self.posts:
            raise ValueError("Post not found")
        
        post = self.posts[post_id]
        
        # Check if interaction is allowed
        if not self._is_interaction_allowed(user_id, post, interaction_type):
            raise ValueError("Interaction not allowed")
        
        interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            post_id=post_id,
            interaction_type=interaction_type,
            **kwargs
        )
        
        with self._lock:
            self.interactions[post_id].append(interaction)
            self.analytics['interactions_created'] += 1
            
            # Update post engagement counters
            if interaction_type in [InteractionType.LIKE, InteractionType.LOVE, 
                                  InteractionType.LAUGH, InteractionType.ANGRY]:
                post.likes_count += 1
            elif interaction_type == InteractionType.COMMENT:
                post.comments_count += 1
            elif interaction_type == InteractionType.SHARE:
                post.shares_count += 1
        
        # Update trending analysis
        engagement_score = self._calculate_interaction_engagement_score(interaction_type)
        self.trending_analyzer.record_hashtag_usage(post.hashtags, engagement_score)
        
        return interaction
    
    def follow_user(self, follower_id: str, followee_id: str) -> bool:
        """Create follow relationship."""
        if follower_id not in self.users or followee_id not in self.users:
            return False
        
        if follower_id == followee_id:
            return False
        
        followee = self.users[followee_id]
        
        # Check if followee account is private
        if followee.is_private:
            # In a real system, this would create a follow request
            print(f"Follow request sent to private account {followee.username}")
            return True
        
        # Create relationship
        relationship = UserRelationship(
            from_user_id=follower_id,
            to_user_id=followee_id,
            relationship_type=RelationshipType.FOLLOWING
        )
        
        with self._lock:
            rel_key = f"{follower_id}:{followee_id}"
            self.relationships[rel_key] = relationship
            
            # Update counters
            self.users[follower_id].following_count += 1
            self.users[followee_id].followers_count += 1
        
        # Invalidate caches
        self.feed_cache.invalidate_user_cache(follower_id)
        
        return True
    
    def generate_feed(self, user_id: str, feed_type: FeedType = FeedType.HOME,
                     ranking_strategy: str = 'personalized', limit: int = 20,
                     use_cache: bool = True) -> List[FeedItem]:
        """Generate feed for user."""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        # Check cache first
        if use_cache:
            cached_feed = self.feed_cache.get_cached_feed(user_id, feed_type)
            if cached_feed:
                self.analytics['cache_hits'] += 1
                return cached_feed[:limit]
            else:
                self.analytics['cache_misses'] += 1
        
        user = self.users[user_id]
        
        # Get relevant posts based on feed type
        relevant_posts = self._get_relevant_posts(user_id, feed_type)
        
        # Get user relationships and interactions
        user_relationships = self._get_user_relationships(user_id)
        user_interactions = self._get_user_interactions(user_id)
        
        # Rank posts using selected strategy
        if ranking_strategy not in self.ranking_strategies:
            ranking_strategy = 'personalized'
        
        ranker = self.ranking_strategies[ranking_strategy]
        feed_items = ranker.rank_posts(relevant_posts, user, user_relationships, user_interactions)
        
        # Cache the results
        if use_cache:
            self.feed_cache.cache_feed(user_id, feed_type, feed_items)
        
        self.analytics['feeds_generated'] += 1
        
        return feed_items[:limit]
    
    def get_trending_topics(self, limit: int = 10) -> List[TrendingTopic]:
        """Get current trending topics."""
        return self.trending_analyzer.get_trending_topics(limit=limit)
    
    def search_posts(self, query: str, user_id: str = None, limit: int = 20) -> List[Post]:
        """Search posts by content or hashtags."""
        query_lower = query.lower()
        matching_posts = []
        
        for post in self.posts.values():
            if post.status != PostStatus.PUBLISHED:
                continue
            
            # Check if user can see this post
            if user_id and not self._can_user_see_post(user_id, post):
                continue
            
            # Check content match
            if (query_lower in post.content.lower() or
                query_lower in post.hashtags or
                any(query_lower in mention.lower() for mention in post.mentions)):
                matching_posts.append(post)
        
        # Sort by relevance (simplified: by creation time)
        matching_posts.sort(key=lambda p: p.created_at, reverse=True)
        
        return matching_posts[:limit]
    
    def get_user_posts(self, user_id: str, viewer_id: str = None, limit: int = 20) -> List[Post]:
        """Get posts by a specific user."""
        if user_id not in self.users:
            return []
        
        user_posts = [post for post in self.posts.values() 
                     if (post.user_id == user_id and 
                         post.status == PostStatus.PUBLISHED)]
        
        # Filter by privacy if viewer is different user
        if viewer_id and viewer_id != user_id:
            user_posts = [post for post in user_posts 
                         if self._can_user_see_post(viewer_id, post)]
        
        # Sort by creation time
        user_posts.sort(key=lambda p: p.created_at, reverse=True)
        
        return user_posts[:limit]
    
    def get_post_interactions(self, post_id: str, interaction_type: InteractionType = None) -> List[Interaction]:
        """Get interactions for a post."""
        if post_id not in self.posts:
            return []
        
        interactions = self.interactions.get(post_id, [])
        
        if interaction_type:
            interactions = [i for i in interactions if i.interaction_type == interaction_type]
        
        return interactions
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get system analytics summary."""
        with self._lock:
            total_users = len(self.users)
            total_posts = len(self.posts)
            total_interactions = sum(len(interactions) for interactions in self.interactions.values())
            
            # Calculate engagement rate
            engagement_rate = (total_interactions / max(1, total_posts)) * 100
            
            # Cache performance
            total_cache_requests = self.analytics['cache_hits'] + self.analytics['cache_misses']
            cache_hit_rate = (self.analytics['cache_hits'] / max(1, total_cache_requests)) * 100
            
            return {
                'users': total_users,
                'posts': total_posts,
                'interactions': total_interactions,
                'engagement_rate': engagement_rate,
                'cache_hit_rate': cache_hit_rate,
                'feeds_generated': self.analytics['feeds_generated'],
                'trending_topics_count': len(self.trending_analyzer.hashtag_counts),
                **self.analytics
            }
    
    def _get_relevant_posts(self, user_id: str, feed_type: FeedType) -> List[Post]:
        """Get relevant posts for feed generation."""
        if feed_type == FeedType.HOME:
            # Get posts from followed users
            following_user_ids = self._get_following_user_ids(user_id)
            following_user_ids.add(user_id)  # Include own posts
            
            relevant_posts = [post for post in self.posts.values()
                            if (post.user_id in following_user_ids and
                                post.status == PostStatus.PUBLISHED and
                                self._can_user_see_post(user_id, post))]
        
        elif feed_type == FeedType.TRENDING:
            # Get posts with high engagement
            relevant_posts = [post for post in self.posts.values()
                            if (post.status == PostStatus.PUBLISHED and
                                self._can_user_see_post(user_id, post) and
                                (post.likes_count + post.comments_count + post.shares_count) > 5)]
        
        elif feed_type == FeedType.DISCOVER:
            # Get posts from users not followed
            following_user_ids = self._get_following_user_ids(user_id)
            
            relevant_posts = [post for post in self.posts.values()
                            if (post.user_id not in following_user_ids and
                                post.user_id != user_id and
                                post.status == PostStatus.PUBLISHED and
                                post.privacy_level == PrivacyLevel.PUBLIC)]
        
        else:
            # Default: all visible posts
            relevant_posts = [post for post in self.posts.values()
                            if (post.status == PostStatus.PUBLISHED and
                                self._can_user_see_post(user_id, post))]
        
        return relevant_posts
    
    def _get_user_relationships(self, user_id: str) -> Dict[str, UserRelationship]:
        """Get user relationships."""
        user_relationships = {}
        
        for rel_key, relationship in self.relationships.items():
            if relationship.from_user_id == user_id:
                user_relationships[rel_key] = relationship
        
        return user_relationships
    
    def _get_user_interactions(self, user_id: str) -> List[Interaction]:
        """Get user interactions."""
        user_interactions = []
        
        for interactions in self.interactions.values():
            user_interactions.extend([i for i in interactions if i.user_id == user_id])
        
        return user_interactions
    
    def _get_following_user_ids(self, user_id: str) -> Set[str]:
        """Get IDs of users that this user follows."""
        following_ids = set()
        
        for relationship in self.relationships.values():
            if (relationship.from_user_id == user_id and
                relationship.relationship_type == RelationshipType.FOLLOWING):
                following_ids.add(relationship.to_user_id)
        
        return following_ids
    
    def _can_user_see_post(self, user_id: str, post: Post) -> bool:
        """Check if user can see a post based on privacy settings."""
        if post.user_id == user_id:
            return True  # Own posts
        
        if post.privacy_level == PrivacyLevel.PUBLIC:
            return True
        
        if post.privacy_level == PrivacyLevel.PRIVATE:
            return False
        
        if post.privacy_level == PrivacyLevel.FRIENDS:
            # Check if users are friends/following
            rel_key = f"{user_id}:{post.user_id}"
            return rel_key in self.relationships
        
        return False
    
    def _is_interaction_allowed(self, user_id: str, post: Post, interaction_type: InteractionType) -> bool:
        """Check if interaction is allowed."""
        # Check if user can see the post
        if not self._can_user_see_post(user_id, post):
            return False
        
        # Check post-specific settings
        if interaction_type == InteractionType.COMMENT and not post.allow_comments:
            return False
        
        if interaction_type == InteractionType.SHARE and not post.allow_shares:
            return False
        
        return True
    
    def _calculate_interaction_engagement_score(self, interaction_type: InteractionType) -> float:
        """Calculate engagement score for interaction type."""
        scores = {
            InteractionType.LIKE: 1.0,
            InteractionType.LOVE: 1.5,
            InteractionType.LAUGH: 1.2,
            InteractionType.ANGRY: 1.2,
            InteractionType.COMMENT: 3.0,
            InteractionType.SHARE: 5.0,
            InteractionType.SAVE: 2.0
        }
        
        return scores.get(interaction_type, 1.0)
    
    def _invalidate_feeds_for_new_post(self, user_id: str) -> None:
        """Invalidate feeds when new post is created."""
        # Invalidate home feeds for followers
        for relationship in self.relationships.values():
            if (relationship.to_user_id == user_id and
                relationship.relationship_type == RelationshipType.FOLLOWING):
                self.feed_cache.invalidate_user_cache(relationship.from_user_id)
        
        # Invalidate trending and discover feeds
        self.feed_cache.invalidate_feed_type(FeedType.TRENDING)
        self.feed_cache.invalidate_feed_type(FeedType.DISCOVER)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_social_media_feed():
    """Demonstrate the social media feed system."""
    print("=== SOCIAL MEDIA FEED SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    print("1. SYSTEM INITIALIZATION:")
    
    system = SocialMediaFeedSystem()
    print("   ✓ Social media feed system initialized")
    print()
    
    # Create users
    print("2. USER CREATION:")
    
    users = {}
    user_data = [
        ("alice_doe", "alice@example.com", "Alice Doe"),
        ("bob_smith", "bob@example.com", "Bob Smith"),
        ("charlie_brown", "charlie@example.com", "Charlie Brown"),
        ("diana_prince", "diana@example.com", "Diana Prince"),
        ("eve_wilson", "eve@example.com", "Eve Wilson")
    ]
    
    for username, email, display_name in user_data:
        user = system.create_user(username, email, display_name)
        users[username] = user
        print(f"   ✓ Created user: {display_name} (@{username})")
    
    print()
    
    # Create follow relationships
    print("3. FOLLOW RELATIONSHIPS:")
    
    follow_relationships = [
        ("alice_doe", "bob_smith"),
        ("alice_doe", "charlie_brown"),
        ("alice_doe", "diana_prince"),
        ("bob_smith", "alice_doe"),
        ("bob_smith", "charlie_brown"),
        ("charlie_brown", "alice_doe"),
        ("charlie_brown", "diana_prince"),
        ("diana_prince", "alice_doe"),
        ("diana_prince", "eve_wilson"),
        ("eve_wilson", "alice_doe"),
        ("eve_wilson", "bob_smith")
    ]
    
    for follower, followee in follow_relationships:
        success = system.follow_user(users[follower].user_id, users[followee].user_id)
        print(f"   {'✓' if success else '✗'} {follower} follows {followee}")
    
    print()
    
    # Create posts
    print("4. POST CREATION:")
    
    posts_data = [
        ("alice_doe", "Just had an amazing coffee at the new cafe! ☕ #coffee #morning", PostType.TEXT),
        ("bob_smith", "Working on my new #Python project. Excited to share soon! #coding #tech", PostType.TEXT),
        ("charlie_brown", "Beautiful sunset today 🌅 #nature #photography #sunset", PostType.IMAGE),
        ("diana_prince", "Great workout session at the gym 💪 #fitness #health #motivation", PostType.TEXT),
        ("eve_wilson", "Sharing my favorite #recipe for chocolate cake 🍰 #baking #food", PostType.TEXT),
        ("alice_doe", "Check out this amazing #AI demo I found! The future is here. #technology #innovation", PostType.LINK),
        ("bob_smith", "Debugging code at 2 AM... again 😅 #programming #developer #life", PostType.TEXT),
        ("charlie_brown", "Amazing #concert last night! The band was incredible 🎵 #music #live", PostType.IMAGE),
        ("diana_prince", "Finished reading 'The Art of War' - highly recommend! 📚 #books #reading", PostType.TEXT),
        ("eve_wilson", "Homemade pizza night! 🍕 Who wants the recipe? #cooking #pizza #homemade", PostType.IMAGE)
    ]
    
    created_posts = []
    for username, content, post_type in posts_data:
        try:
            post = system.create_post(
                user_id=users[username].user_id,
                content=content,
                post_type=post_type
            )
            created_posts.append(post)
            print(f"   ✓ {username}: {content[:50]}...")
        except Exception as e:
            print(f"   ✗ Failed to create post for {username}: {e}")
    
    print()
    
    # Create interactions
    print("5. USER INTERACTIONS:")
    
    # Simulate various interactions
    interaction_scenarios = [
        ("bob_smith", 0, InteractionType.LIKE),
        ("charlie_brown", 0, InteractionType.LOVE),
        ("diana_prince", 1, InteractionType.LIKE),
        ("alice_doe", 1, InteractionType.COMMENT, {"comment_text": "Looks interesting! Can't wait to see it."}),
        ("eve_wilson", 2, InteractionType.LOVE),
        ("alice_doe", 2, InteractionType.COMMENT, {"comment_text": "Gorgeous photo! Where was this taken?"}),
        ("bob_smith", 3, InteractionType.LIKE),
        ("charlie_brown", 4, InteractionType.SAVE),
        ("diana_prince", 4, InteractionType.COMMENT, {"comment_text": "Would love the recipe!"}),
        ("alice_doe", 5, InteractionType.SHARE),
        ("eve_wilson", 6, InteractionType.LAUGH),
        ("charlie_brown", 7, InteractionType.LOVE),
        ("bob_smith", 8, InteractionType.LIKE),
        ("alice_doe", 9, InteractionType.COMMENT, {"comment_text": "Yes please! Share the recipe!"})
    ]
    
    for username, post_index, interaction_type, *extra_data in interaction_scenarios:
        if post_index < len(created_posts):
            try:
                kwargs = extra_data[0] if extra_data else {}
                interaction = system.create_interaction(
                    user_id=users[username].user_id,
                    post_id=created_posts[post_index].post_id,
                    interaction_type=interaction_type,
                    **kwargs
                )
                print(f"   ✓ {username} {interaction_type.value}d post by {created_posts[post_index].user_id}")
            except Exception as e:
                print(f"   ✗ Interaction failed: {e}")
    
    print()
    
    # Test different feed generation strategies
    print("6. FEED GENERATION TEST:")
    
    alice_id = users["alice_doe"].user_id
    
    strategies = ['chronological', 'engagement', 'personalized']
    
    for strategy in strategies:
        feed = system.generate_feed(
            user_id=alice_id,
            feed_type=FeedType.HOME,
            ranking_strategy=strategy,
            limit=5
        )
        
        print(f"   {strategy.capitalize()} Feed for Alice:")
        for i, feed_item in enumerate(feed):
            post_user = next(user for user in users.values() 
                           if user.user_id == feed_item.post.user_id)
            print(f"     {i+1}. @{post_user.username}: {feed_item.post.content[:40]}... "
                  f"(score: {feed_item.score:.2f})")
        print()
    
    # Test different feed types
    print("7. DIFFERENT FEED TYPES:")
    
    feed_types = [FeedType.HOME, FeedType.TRENDING, FeedType.DISCOVER]
    
    for feed_type in feed_types:
        feed = system.generate_feed(
            user_id=alice_id,
            feed_type=feed_type,
            limit=3
        )
        
        print(f"   {feed_type.value.capitalize()} Feed:")
        for i, feed_item in enumerate(feed):
            post_user = next(user for user in users.values() 
                           if user.user_id == feed_item.post.user_id)
            print(f"     {i+1}. @{post_user.username}: {feed_item.post.content[:40]}...")
        print()
    
    # Test trending topics
    print("8. TRENDING TOPICS:")
    
    trending = system.get_trending_topics(limit=5)
    
    if trending:
        print("   Current trending topics:")
        for i, topic in enumerate(trending):
            print(f"     {i+1}. #{topic.topic}: {topic.post_count} posts "
                  f"(score: {topic.engagement_score:.2f})")
    else:
        print("   No trending topics found")
    
    print()
    
    # Test search functionality
    print("9. SEARCH FUNCTIONALITY:")
    
    search_queries = ["coffee", "python", "recipe", "#photography"]
    
    for query in search_queries:
        results = system.search_posts(query, alice_id, limit=3)
        
        print(f"   Search for '{query}': {len(results)} results")
        for result in results:
            post_user = next(user for user in users.values() 
                           if user.user_id == result.user_id)
            print(f"     @{post_user.username}: {result.content[:50]}...")
        print()
    
    # Test content moderation
    print("10. CONTENT MODERATION TEST:")
    
    moderation_test_posts = [
        "BUY NOW! GET RICH QUICK! CLICK HERE FOR FREE MONEY!!!",
        "This contains hate speech and violence",
        "Check out this malicious link: http://malicious-site.com/virus",
        "Normal post with reasonable content"
    ]
    
    for content in moderation_test_posts:
        try:
            post = system.create_post(
                user_id=users["alice_doe"].user_id,
                content=content
            )
            status = "✓ Published" if post.status == PostStatus.PUBLISHED else "✗ Blocked"
            print(f"   {status}: {content[:50]}...")
        except Exception as e:
            print(f"   ✗ Error: {content[:30]}... - {e}")
    
    print()
    
    # Test cache performance
    print("11. CACHE PERFORMANCE TEST:")
    
    # Generate feeds multiple times to test caching
    start_time = time.time()
    
    for _ in range(5):
        system.generate_feed(alice_id, FeedType.HOME, use_cache=True)
    
    cached_time = time.time() - start_time
    
    # Clear cache and test without caching
    system.feed_cache.cache.clear()
    start_time = time.time()
    
    for _ in range(5):
        system.generate_feed(alice_id, FeedType.HOME, use_cache=False)
    
    uncached_time = time.time() - start_time
    
    print(f"   5 feed generations with cache: {cached_time:.3f}s")
    print(f"   5 feed generations without cache: {uncached_time:.3f}s")
    print(f"   Cache speedup: {uncached_time/cached_time:.1f}x")
    
    print()
    
    # Test user profile and posts
    print("12. USER PROFILE TEST:")
    
    alice_posts = system.get_user_posts(users["alice_doe"].user_id, limit=5)
    
    print(f"   Alice's posts ({len(alice_posts)}):")
    for post in alice_posts:
        interactions_count = len(system.get_post_interactions(post.post_id))
        print(f"     {post.content[:50]}... ({interactions_count} interactions)")
    
    print()
    
    # Show comprehensive analytics
    print("13. SYSTEM ANALYTICS:")
    
    analytics = system.get_analytics_summary()
    
    print(f"   Users: {analytics['users']}")
    print(f"   Posts: {analytics['posts']}")
    print(f"   Interactions: {analytics['interactions']}")
    print(f"   Engagement Rate: {analytics['engagement_rate']:.1f}%")
    print(f"   Cache Hit Rate: {analytics['cache_hit_rate']:.1f}%")
    print(f"   Feeds Generated: {analytics['feeds_generated']}")
    print(f"   Trending Topics: {analytics['trending_topics_count']}")
    
    print()
    
    # Test feed personalization over time
    print("14. PERSONALIZATION EVOLUTION:")
    
    # Simulate Alice interacting more with certain topics
    tech_posts = [post for post in created_posts if '#tech' in post.content or '#programming' in post.content]
    
    for post in tech_posts:
        system.create_interaction(
            user_id=users["alice_doe"].user_id,
            post_id=post.post_id,
            interaction_type=InteractionType.LIKE
        )
    
    # Generate personalized feed
    personalized_feed = system.generate_feed(
        user_id=alice_id,
        ranking_strategy='personalized',
        limit=5
    )
    
    print("   Alice's personalized feed after tech interactions:")
    for i, feed_item in enumerate(personalized_feed):
        post_user = next(user for user in users.values() 
                       if user.user_id == feed_item.post.user_id)
        print(f"     {i+1}. @{post_user.username}: {feed_item.post.content[:40]}... "
              f"(score: {feed_item.score:.2f})")
    
    print()
    
    # Show final system state
    print("15. FINAL SYSTEM STATE:")
    
    final_analytics = system.get_analytics_summary()
    
    print(f"   Total Users: {final_analytics['users']}")
    print(f"   Total Posts: {final_analytics['posts']}")
    print(f"   Total Interactions: {final_analytics['interactions']}")
    print(f"   Overall Engagement Rate: {final_analytics['engagement_rate']:.1f}%")
    
    # Show user statistics
    print("\n   User Statistics:")
    for username, user in users.items():
        print(f"     @{username}: {user.posts_count} posts, "
              f"{user.followers_count} followers, {user.following_count} following")
    
    # Show most engaged posts
    most_engaged = sorted(created_posts, 
                         key=lambda p: p.likes_count + p.comments_count + p.shares_count, 
                         reverse=True)[:3]
    
    print("\n   Most Engaged Posts:")
    for i, post in enumerate(most_engaged):
        post_user = next(user for user in users.values() if user.user_id == post.user_id)
        total_engagement = post.likes_count + post.comments_count + post.shares_count
        print(f"     {i+1}. @{post_user.username}: {post.content[:40]}... "
              f"({total_engagement} interactions)")
    
    print()
    print("=== SOCIAL MEDIA FEED SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_social_media_feed()
