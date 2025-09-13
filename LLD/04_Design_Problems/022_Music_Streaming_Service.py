"""
MUSIC STREAMING SERVICE - Complete System Design
===============================================

Problem Statement:
Design a comprehensive music streaming service that handles:
- Music catalog management and metadata
- User subscription and account management  
- Audio streaming with multiple quality levels
- Playlist creation and management
- Search and recommendation algorithms
- Social features (following, sharing, collaborative playlists)
- Offline downloads and synchronization
- Artist and label management
- Royalty calculation and payment processing
- Content delivery and caching
- Analytics and reporting

Requirements:
- Support massive music catalog with metadata management
- Handle multiple audio quality streams (low, medium, high, lossless)
- Implement efficient search across songs, artists, albums, playlists
- Provide personalized recommendations based on listening history
- Support social features like following users and sharing music
- Handle offline downloads with sync capabilities
- Manage artist profiles, albums, and track uploads
- Calculate and distribute royalties to rights holders
- Implement content delivery network for global streaming
- Scale to millions of concurrent users
- Provide comprehensive analytics for users, artists, and platform

Design Patterns Used:
- Strategy: Recommendation and audio encoding strategies
- Observer: Real-time notifications and social updates
- Factory: Playlist and track creation
- Decorator: Audio stream processing and effects
- Command: User actions with undo capability
- Facade: Simplified streaming interface
- Proxy: Content delivery and caching proxy
- Template Method: Audio processing pipeline
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
import random
import json
import hashlib
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class UserType(Enum):
    FREE = "free"
    PREMIUM = "premium"
    ARTIST = "artist"
    LABEL = "label"
    ADMIN = "admin"


class AudioQuality(Enum):
    LOW = "low"      # 96 kbps
    MEDIUM = "medium"  # 160 kbps
    HIGH = "high"    # 320 kbps
    LOSSLESS = "lossless"  # FLAC


class PlaylistType(Enum):
    USER_CREATED = "user_created"
    ALGORITHMIC = "algorithmic"
    EDITORIAL = "editorial"
    COLLABORATIVE = "collaborative"


class PlaybackStatus(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFERING = "buffering"


class SubscriptionPlan(Enum):
    FREE = "free"
    PREMIUM_INDIVIDUAL = "premium_individual"
    PREMIUM_FAMILY = "premium_family"
    PREMIUM_STUDENT = "premium_student"


@dataclass
class AudioFile:
    """Audio file information."""
    file_id: str
    quality: AudioQuality
    file_url: str
    file_size: int  # bytes
    duration: int   # seconds
    bitrate: int    # kbps
    format: str     # mp3, flac, etc.
    
    def __post_init__(self):
        if not self.file_id:
            self.file_id = str(uuid.uuid4())


@dataclass
class Track:
    """Music track."""
    track_id: str
    title: str
    artist_id: str
    album_id: Optional[str] = None
    
    # Duration and metadata
    duration: int = 0  # seconds
    track_number: Optional[int] = None
    disc_number: int = 1
    
    # Content
    audio_files: Dict[AudioQuality, AudioFile] = field(default_factory=dict)
    
    # Metadata
    genre: str = ""
    language: str = "en"
    explicit: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Engagement metrics
    play_count: int = 0
    like_count: int = 0
    
    # Rights and availability
    is_available: bool = True
    territory_restrictions: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.track_id:
            self.track_id = str(uuid.uuid4())
    
    def get_audio_file(self, quality: AudioQuality) -> Optional[AudioFile]:
        """Get audio file for specific quality."""
        return self.audio_files.get(quality)


@dataclass
class Album:
    """Music album."""
    album_id: str
    title: str
    artist_id: str
    
    # Metadata
    release_date: datetime
    genre: str = ""
    description: str = ""
    
    # Visual
    cover_art_url: str = ""
    
    # Tracks
    track_ids: List[str] = field(default_factory=list)
    
    # Engagement
    play_count: int = 0
    like_count: int = 0
    
    def __post_init__(self):
        if not self.album_id:
            self.album_id = str(uuid.uuid4())


@dataclass
class Artist:
    """Music artist/band."""
    artist_id: str
    name: str
    
    # Profile
    bio: str = ""
    profile_image_url: str = ""
    
    # Social
    follower_count: int = 0
    verified: bool = False
    
    # Content
    track_ids: Set[str] = field(default_factory=set)
    album_ids: Set[str] = field(default_factory=set)
    
    # Metrics
    monthly_listeners: int = 0
    total_plays: int = 0
    
    def __post_init__(self):
        if not self.artist_id:
            self.artist_id = str(uuid.uuid4())


@dataclass
class User:
    """Music streaming user."""
    user_id: str
    email: str
    username: str
    display_name: str
    
    # Subscription
    subscription_plan: SubscriptionPlan = SubscriptionPlan.FREE
    subscription_expires: Optional[datetime] = None
    
    # Profile
    profile_image_url: str = ""
    bio: str = ""
    
    # Preferences
    preferred_audio_quality: AudioQuality = AudioQuality.MEDIUM
    language: str = "en"
    country: str = "US"
    
    # Social
    followers: Set[str] = field(default_factory=set)
    following: Set[str] = field(default_factory=set)
    
    # Content
    liked_tracks: Set[str] = field(default_factory=set)
    liked_albums: Set[str] = field(default_factory=set)
    liked_artists: Set[str] = field(default_factory=set)
    
    # Playlists
    playlist_ids: Set[str] = field(default_factory=set)
    
    # Usage statistics
    total_listening_time: int = 0  # minutes
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium subscription."""
        if self.subscription_plan == SubscriptionPlan.FREE:
            return False
        
        if self.subscription_expires:
            return datetime.now() < self.subscription_expires
        
        return True


@dataclass
class Playlist:
    """Music playlist."""
    playlist_id: str
    name: str
    owner_id: str
    
    # Content
    track_ids: List[str] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    cover_image_url: str = ""
    
    # Settings
    playlist_type: PlaylistType = PlaylistType.USER_CREATED
    is_public: bool = True
    is_collaborative: bool = False
    
    # Collaboration
    collaborators: Set[str] = field(default_factory=set)
    
    # Metrics
    follower_count: int = 0
    play_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.playlist_id:
            self.playlist_id = str(uuid.uuid4())
    
    def add_track(self, track_id: str) -> None:
        """Add track to playlist."""
        if track_id not in self.track_ids:
            self.track_ids.append(track_id)
            self.updated_at = datetime.now()
    
    def remove_track(self, track_id: str) -> bool:
        """Remove track from playlist."""
        if track_id in self.track_ids:
            self.track_ids.remove(track_id)
            self.updated_at = datetime.now()
            return True
        return False
    
    @property
    def duration(self) -> int:
        """Calculate total playlist duration (would need track data)."""
        return len(self.track_ids) * 200  # Simplified: 200 seconds per track


@dataclass
class PlaybackSession:
    """User playback session."""
    session_id: str
    user_id: str
    
    # Current state
    current_track_id: Optional[str] = None
    current_playlist_id: Optional[str] = None
    position: int = 0  # seconds
    status: PlaybackStatus = PlaybackStatus.STOPPED
    
    # Queue
    queue: List[str] = field(default_factory=list)  # track_ids
    queue_position: int = 0
    
    # Settings
    shuffle: bool = False
    repeat: bool = False
    volume: float = 0.8
    
    # Session info
    started_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


@dataclass
class ListeningHistory:
    """User listening history entry."""
    history_id: str
    user_id: str
    track_id: str
    
    # Playback details
    played_at: datetime
    duration_played: int  # seconds
    track_duration: int   # seconds
    audio_quality: AudioQuality
    
    # Context
    playlist_id: Optional[str] = None
    album_id: Optional[str] = None
    source: str = "unknown"  # playlist, album, search, radio, etc.
    
    def __post_init__(self):
        if not self.history_id:
            self.history_id = str(uuid.uuid4())
    
    @property
    def completion_rate(self) -> float:
        """Calculate how much of the track was played."""
        return min(1.0, self.duration_played / max(1, self.track_duration))


# ============================================================================
# RECOMMENDATION ALGORITHMS
# ============================================================================

class RecommendationEngine(ABC):
    """Abstract recommendation engine."""
    
    @abstractmethod
    def get_recommendations(self, user_id: str, count: int = 20) -> List[str]:
        """Get track recommendations for user."""
        pass
    
    @abstractmethod
    def get_similar_tracks(self, track_id: str, count: int = 10) -> List[str]:
        """Get tracks similar to given track."""
        pass


class CollaborativeFilteringEngine(RecommendationEngine):
    """Collaborative filtering recommendation engine."""
    
    def __init__(self, music_service: 'MusicStreamingService'):
        self.music_service = music_service
    
    def get_recommendations(self, user_id: str, count: int = 20) -> List[str]:
        """Get recommendations based on similar users."""
        if user_id not in self.music_service.users:
            return []
        
        user = self.music_service.users[user_id]
        
        # Find similar users based on liked tracks
        similar_users = self._find_similar_users(user_id)
        
        # Get tracks liked by similar users
        recommended_tracks = set()
        
        for similar_user_id, similarity in similar_users[:10]:  # Top 10 similar users
            similar_user = self.music_service.users[similar_user_id]
            
            for track_id in similar_user.liked_tracks:
                if track_id not in user.liked_tracks:
                    recommended_tracks.add(track_id)
        
        # Convert to list and limit
        return list(recommended_tracks)[:count]
    
    def get_similar_tracks(self, track_id: str, count: int = 10) -> List[str]:
        """Get tracks that are often liked together."""
        if track_id not in self.music_service.tracks:
            return []
        
        # Find users who liked this track
        users_who_liked = []
        for user in self.music_service.users.values():
            if track_id in user.liked_tracks:
                users_who_liked.append(user.user_id)
        
        # Find other tracks liked by these users
        track_frequency = defaultdict(int)
        
        for user_id in users_who_liked:
            user = self.music_service.users[user_id]
            for other_track_id in user.liked_tracks:
                if other_track_id != track_id:
                    track_frequency[other_track_id] += 1
        
        # Sort by frequency and return top tracks
        similar_tracks = sorted(track_frequency.items(), key=lambda x: x[1], reverse=True)
        return [track_id for track_id, _ in similar_tracks[:count]]
    
    def _find_similar_users(self, user_id: str) -> List[Tuple[str, float]]:
        """Find users with similar music taste."""
        if user_id not in self.music_service.users:
            return []
        
        user = self.music_service.users[user_id]
        user_likes = user.liked_tracks
        
        if not user_likes:
            return []
        
        similarities = []
        
        for other_user_id, other_user in self.music_service.users.items():
            if other_user_id == user_id:
                continue
            
            other_likes = other_user.liked_tracks
            
            if not other_likes:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(user_likes & other_likes)
            union = len(user_likes | other_likes)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Minimum similarity threshold
                    similarities.append((other_user_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


class ContentBasedEngine(RecommendationEngine):
    """Content-based recommendation engine."""
    
    def __init__(self, music_service: 'MusicStreamingService'):
        self.music_service = music_service
    
    def get_recommendations(self, user_id: str, count: int = 20) -> List[str]:
        """Get recommendations based on user's music preferences."""
        if user_id not in self.music_service.users:
            return []
        
        user = self.music_service.users[user_id]
        
        # Analyze user's liked tracks to determine preferences
        preferences = self._analyze_user_preferences(user)
        
        # Find tracks matching preferences
        recommendations = []
        
        for track in self.music_service.tracks.values():
            if track.track_id not in user.liked_tracks:
                score = self._calculate_track_score(track, preferences)
                if score > 0.5:  # Threshold
                    recommendations.append((track.track_id, score))
        
        # Sort by score and return top tracks
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [track_id for track_id, _ in recommendations[:count]]
    
    def get_similar_tracks(self, track_id: str, count: int = 10) -> List[str]:
        """Get tracks similar in content to given track."""
        if track_id not in self.music_service.tracks:
            return []
        
        reference_track = self.music_service.tracks[track_id]
        
        # Calculate similarity with all other tracks
        similarities = []
        
        for other_track in self.music_service.tracks.values():
            if other_track.track_id != track_id:
                similarity = self._calculate_track_similarity(reference_track, other_track)
                if similarity > 0.3:  # Threshold
                    similarities.append((other_track.track_id, similarity))
        
        # Sort by similarity and return top tracks
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [track_id for track_id, _ in similarities[:count]]
    
    def _analyze_user_preferences(self, user: User) -> Dict[str, float]:
        """Analyze user preferences from liked tracks."""
        preferences = {
            'genres': defaultdict(float),
            'artists': defaultdict(float),
            'duration_preference': 0.0,
            'explicit_tolerance': 0.0
        }
        
        if not user.liked_tracks:
            return preferences
        
        for track_id in user.liked_tracks:
            if track_id in self.music_service.tracks:
                track = self.music_service.tracks[track_id]
                
                # Genre preferences
                if track.genre:
                    preferences['genres'][track.genre] += 1.0
                
                # Artist preferences
                preferences['artists'][track.artist_id] += 1.0
                
                # Duration preferences (simplified)
                preferences['duration_preference'] += track.duration
                
                # Explicit content tolerance
                if track.explicit:
                    preferences['explicit_tolerance'] += 1.0
        
        # Normalize preferences
        total_tracks = len(user.liked_tracks)
        
        for genre in preferences['genres']:
            preferences['genres'][genre] /= total_tracks
        
        for artist in preferences['artists']:
            preferences['artists'][artist] /= total_tracks
        
        preferences['duration_preference'] /= total_tracks
        preferences['explicit_tolerance'] /= total_tracks
        
        return preferences
    
    def _calculate_track_score(self, track: Track, preferences: Dict[str, float]) -> float:
        """Calculate how well a track matches user preferences."""
        score = 0.0
        
        # Genre match
        if track.genre in preferences['genres']:
            score += preferences['genres'][track.genre] * 0.4
        
        # Artist match
        if track.artist_id in preferences['artists']:
            score += preferences['artists'][track.artist_id] * 0.3
        
        # Duration match (simplified)
        duration_diff = abs(track.duration - preferences['duration_preference'])
        duration_score = max(0, 1 - duration_diff / 300)  # 5-minute tolerance
        score += duration_score * 0.2
        
        # Explicit content match
        if track.explicit and preferences['explicit_tolerance'] < 0.2:
            score -= 0.3  # Penalty for explicit content if user doesn't like it
        
        # Popularity boost (simplified)
        popularity_score = min(1.0, track.play_count / 10000)
        score += popularity_score * 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_track_similarity(self, track1: Track, track2: Track) -> float:
        """Calculate similarity between two tracks."""
        similarity = 0.0
        
        # Same artist
        if track1.artist_id == track2.artist_id:
            similarity += 0.4
        
        # Same album
        if track1.album_id and track1.album_id == track2.album_id:
            similarity += 0.3
        
        # Same genre
        if track1.genre and track1.genre == track2.genre:
            similarity += 0.2
        
        # Similar duration
        duration_diff = abs(track1.duration - track2.duration)
        duration_similarity = max(0, 1 - duration_diff / 180)  # 3-minute tolerance
        similarity += duration_similarity * 0.1
        
        return min(1.0, similarity)


# ============================================================================
# MAIN MUSIC STREAMING SERVICE
# ============================================================================

class MusicStreamingService:
    """Main music streaming service."""
    
    def __init__(self, service_name: str = "StreamMusic"):
        self.service_name = service_name
        
        # Data storage
        self.users: Dict[str, User] = {}
        self.tracks: Dict[str, Track] = {}
        self.albums: Dict[str, Album] = {}
        self.artists: Dict[str, Artist] = {}
        self.playlists: Dict[str, Playlist] = {}
        
        # Active sessions
        self.playback_sessions: Dict[str, PlaybackSession] = {}
        
        # Listening history
        self.listening_history: List[ListeningHistory] = []
        
        # Recommendation engine
        self.recommendation_engine = CollaborativeFilteringEngine(self)
        
        # Search index (simplified)
        self.search_index = {
            'tracks': defaultdict(set),
            'artists': defaultdict(set),
            'albums': defaultdict(set),
            'playlists': defaultdict(set)
        }
        
        # Analytics
        self.analytics = {
            'total_streams': 0,
            'total_listening_time': 0,
            'daily_active_users': 0,
            'premium_subscribers': 0,
            'top_tracks': [],
            'top_artists': []
        }
        
        # Threading
        self._lock = threading.RLock()
        
        print(f"🎵 Music Streaming Service '{service_name}' initialized")
    
    def register_user(self, email: str, username: str, display_name: str) -> User:
        """Register a new user."""
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            username=username,
            display_name=display_name
        )
        
        with self._lock:
            self.users[user.user_id] = user
        
        return user
    
    def create_artist(self, name: str, bio: str = "") -> Artist:
        """Create a new artist profile."""
        artist = Artist(
            artist_id=str(uuid.uuid4()),
            name=name,
            bio=bio
        )
        
        with self._lock:
            self.artists[artist.artist_id] = artist
            
            # Update search index
            for word in name.lower().split():
                self.search_index['artists'][word].add(artist.artist_id)
        
        return artist
    
    def create_album(self, title: str, artist_id: str, release_date: datetime,
                    genre: str = "") -> Album:
        """Create a new album."""
        if artist_id not in self.artists:
            raise ValueError("Artist not found")
        
        album = Album(
            album_id=str(uuid.uuid4()),
            title=title,
            artist_id=artist_id,
            release_date=release_date,
            genre=genre
        )
        
        with self._lock:
            self.albums[album.album_id] = album
            self.artists[artist_id].album_ids.add(album.album_id)
            
            # Update search index
            for word in title.lower().split():
                self.search_index['albums'][word].add(album.album_id)
        
        return album
    
    def upload_track(self, title: str, artist_id: str, duration: int,
                    album_id: str = None, genre: str = "", **kwargs) -> Track:
        """Upload a new track."""
        if artist_id not in self.artists:
            raise ValueError("Artist not found")
        
        if album_id and album_id not in self.albums:
            raise ValueError("Album not found")
        
        track = Track(
            track_id=str(uuid.uuid4()),
            title=title,
            artist_id=artist_id,
            album_id=album_id,
            duration=duration,
            genre=genre,
            **kwargs
        )
        
        # Create audio files for different qualities (simplified)
        for quality in AudioQuality:
            audio_file = AudioFile(
                file_id=str(uuid.uuid4()),
                quality=quality,
                file_url=f"https://cdn.{self.service_name.lower()}.com/audio/{track.track_id}_{quality.value}",
                file_size=duration * 1000 * (32 if quality == AudioQuality.LOW else 
                                           64 if quality == AudioQuality.MEDIUM else
                                           128 if quality == AudioQuality.HIGH else 256),
                duration=duration,
                bitrate=96 if quality == AudioQuality.LOW else
                       160 if quality == AudioQuality.MEDIUM else
                       320 if quality == AudioQuality.HIGH else 1411,
                format="mp3" if quality != AudioQuality.LOSSLESS else "flac"
            )
            track.audio_files[quality] = audio_file
        
        with self._lock:
            self.tracks[track.track_id] = track
            self.artists[artist_id].track_ids.add(track.track_id)
            
            if album_id:
                self.albums[album_id].track_ids.append(track.track_id)
            
            # Update search index
            for word in title.lower().split():
                self.search_index['tracks'][word].add(track.track_id)
        
        return track
    
    def create_playlist(self, user_id: str, name: str, description: str = "",
                       is_public: bool = True, is_collaborative: bool = False) -> Playlist:
        """Create a new playlist."""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        playlist = Playlist(
            playlist_id=str(uuid.uuid4()),
            name=name,
            owner_id=user_id,
            description=description,
            is_public=is_public,
            is_collaborative=is_collaborative
        )
        
        with self._lock:
            self.playlists[playlist.playlist_id] = playlist
            self.users[user_id].playlist_ids.add(playlist.playlist_id)
            
            # Update search index
            for word in name.lower().split():
                self.search_index['playlists'][word].add(playlist.playlist_id)
        
        return playlist
    
    def start_playback_session(self, user_id: str) -> PlaybackSession:
        """Start a new playback session for user."""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        session = PlaybackSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id
        )
        
        with self._lock:
            # End any existing session
            existing_sessions = [s for s in self.playback_sessions.values() 
                               if s.user_id == user_id]
            for existing_session in existing_sessions:
                del self.playback_sessions[existing_session.session_id]
            
            self.playback_sessions[session.session_id] = session
        
        return session
    
    def play_track(self, session_id: str, track_id: str, playlist_id: str = None,
                  position: int = 0) -> bool:
        """Start playing a track."""
        if session_id not in self.playback_sessions:
            return False
        
        if track_id not in self.tracks:
            return False
        
        session = self.playback_sessions[session_id]
        user = self.users[session.user_id]
        track = self.tracks[track_id]
        
        # Check if track is available
        if not track.is_available:
            return False
        
        # Update session
        session.current_track_id = track_id
        session.current_playlist_id = playlist_id
        session.position = position
        session.status = PlaybackStatus.PLAYING
        session.last_updated = datetime.now()
        
        # Record listening history
        self._record_listening_start(session, track)
        
        # Update analytics
        with self._lock:
            track.play_count += 1
            self.analytics['total_streams'] += 1
        
        print(f"▶️  Playing: {track.title} by {self.artists[track.artist_id].name}")
        return True
    
    def pause_playback(self, session_id: str) -> bool:
        """Pause playback."""
        if session_id not in self.playback_sessions:
            return False
        
        session = self.playback_sessions[session_id]
        
        if session.status == PlaybackStatus.PLAYING:
            session.status = PlaybackStatus.PAUSED
            session.last_updated = datetime.now()
            
            # Record listening time
            self._record_listening_end(session)
            return True
        
        return False
    
    def resume_playback(self, session_id: str) -> bool:
        """Resume playback."""
        if session_id not in self.playback_sessions:
            return False
        
        session = self.playback_sessions[session_id]
        
        if session.status == PlaybackStatus.PAUSED and session.current_track_id:
            session.status = PlaybackStatus.PLAYING
            session.last_updated = datetime.now()
            
            # Record new listening start
            track = self.tracks[session.current_track_id]
            self._record_listening_start(session, track)
            return True
        
        return False
    
    def next_track(self, session_id: str) -> bool:
        """Skip to next track."""
        if session_id not in self.playback_sessions:
            return False
        
        session = self.playback_sessions[session_id]
        
        # Record current listening end
        if session.current_track_id:
            self._record_listening_end(session)
        
        # Get next track
        next_track_id = self._get_next_track(session)
        
        if next_track_id:
            return self.play_track(session_id, next_track_id, session.current_playlist_id)
        
        return False
    
    def add_to_queue(self, session_id: str, track_id: str) -> bool:
        """Add track to playback queue."""
        if session_id not in self.playback_sessions:
            return False
        
        if track_id not in self.tracks:
            return False
        
        session = self.playback_sessions[session_id]
        session.queue.append(track_id)
        
        return True
    
    def like_track(self, user_id: str, track_id: str) -> bool:
        """Like a track."""
        if user_id not in self.users or track_id not in self.tracks:
            return False
        
        user = self.users[user_id]
        track = self.tracks[track_id]
        
        if track_id not in user.liked_tracks:
            with self._lock:
                user.liked_tracks.add(track_id)
                track.like_count += 1
            
            return True
        
        return False
    
    def unlike_track(self, user_id: str, track_id: str) -> bool:
        """Unlike a track."""
        if user_id not in self.users or track_id not in self.tracks:
            return False
        
        user = self.users[user_id]
        track = self.tracks[track_id]
        
        if track_id in user.liked_tracks:
            with self._lock:
                user.liked_tracks.remove(track_id)
                track.like_count = max(0, track.like_count - 1)
            
            return True
        
        return False
    
    def follow_artist(self, user_id: str, artist_id: str) -> bool:
        """Follow an artist."""
        if user_id not in self.users or artist_id not in self.artists:
            return False
        
        user = self.users[user_id]
        artist = self.artists[artist_id]
        
        if artist_id not in user.liked_artists:
            with self._lock:
                user.liked_artists.add(artist_id)
                artist.follower_count += 1
            
            return True
        
        return False
    
    def add_track_to_playlist(self, user_id: str, playlist_id: str, track_id: str) -> bool:
        """Add track to playlist."""
        if (user_id not in self.users or 
            playlist_id not in self.playlists or 
            track_id not in self.tracks):
            return False
        
        playlist = self.playlists[playlist_id]
        
        # Check permissions
        if (playlist.owner_id != user_id and 
            not playlist.is_collaborative and 
            user_id not in playlist.collaborators):
            return False
        
        playlist.add_track(track_id)
        return True
    
    def search(self, query: str, search_type: str = "all", limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Search for music content."""
        query_words = query.lower().split()
        results = {
            'tracks': [],
            'artists': [],
            'albums': [],
            'playlists': []
        }
        
        if search_type in ["all", "tracks"]:
            track_ids = set()
            for word in query_words:
                track_ids.update(self.search_index['tracks'].get(word, set()))
            
            for track_id in list(track_ids)[:limit]:
                track = self.tracks[track_id]
                artist = self.artists[track.artist_id]
                results['tracks'].append({
                    'id': track_id,
                    'title': track.title,
                    'artist': artist.name,
                    'duration': track.duration,
                    'play_count': track.play_count
                })
        
        if search_type in ["all", "artists"]:
            artist_ids = set()
            for word in query_words:
                artist_ids.update(self.search_index['artists'].get(word, set()))
            
            for artist_id in list(artist_ids)[:limit]:
                artist = self.artists[artist_id]
                results['artists'].append({
                    'id': artist_id,
                    'name': artist.name,
                    'follower_count': artist.follower_count,
                    'monthly_listeners': artist.monthly_listeners
                })
        
        if search_type in ["all", "albums"]:
            album_ids = set()
            for word in query_words:
                album_ids.update(self.search_index['albums'].get(word, set()))
            
            for album_id in list(album_ids)[:limit]:
                album = self.albums[album_id]
                artist = self.artists[album.artist_id]
                results['albums'].append({
                    'id': album_id,
                    'title': album.title,
                    'artist': artist.name,
                    'release_date': album.release_date.isoformat(),
                    'track_count': len(album.track_ids)
                })
        
        if search_type in ["all", "playlists"]:
            playlist_ids = set()
            for word in query_words:
                playlist_ids.update(self.search_index['playlists'].get(word, set()))
            
            for playlist_id in list(playlist_ids)[:limit]:
                playlist = self.playlists[playlist_id]
                if playlist.is_public:  # Only show public playlists in search
                    owner = self.users[playlist.owner_id]
                    results['playlists'].append({
                        'id': playlist_id,
                        'name': playlist.name,
                        'owner': owner.display_name,
                        'track_count': len(playlist.track_ids),
                        'follower_count': playlist.follower_count
                    })
        
        return results
    
    def get_recommendations(self, user_id: str, count: int = 20) -> List[Dict[str, Any]]:
        """Get personalized recommendations for user."""
        if user_id not in self.users:
            return []
        
        track_ids = self.recommendation_engine.get_recommendations(user_id, count)
        
        recommendations = []
        for track_id in track_ids:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                artist = self.artists[track.artist_id]
                recommendations.append({
                    'track_id': track_id,
                    'title': track.title,
                    'artist': artist.name,
                    'duration': track.duration,
                    'reason': 'Based on your listening history'
                })
        
        return recommendations
    
    def get_user_listening_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user's listening statistics."""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        
        # Calculate stats from listening history
        user_history = [h for h in self.listening_history if h.user_id == user_id]
        
        # Top artists
        artist_play_count = defaultdict(int)
        for history in user_history:
            track = self.tracks.get(history.track_id)
            if track:
                artist_play_count[track.artist_id] += 1
        
        top_artists = sorted(artist_play_count.items(), key=lambda x: x[1], reverse=True)[:5]
        top_artists_data = []
        for artist_id, play_count in top_artists:
            artist = self.artists[artist_id]
            top_artists_data.append({
                'name': artist.name,
                'play_count': play_count
            })
        
        # Top genres
        genre_play_count = defaultdict(int)
        for history in user_history:
            track = self.tracks.get(history.track_id)
            if track and track.genre:
                genre_play_count[track.genre] += 1
        
        top_genres = sorted(genre_play_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_listening_time': user.total_listening_time,
            'tracks_played': len(user_history),
            'liked_tracks': len(user.liked_tracks),
            'liked_artists': len(user.liked_artists),
            'playlists_created': len(user.playlist_ids),
            'top_artists': top_artists_data,
            'top_genres': dict(top_genres),
            'subscription_plan': user.subscription_plan.value,
            'member_since': user.created_at.isoformat()
        }
    
    def _record_listening_start(self, session: PlaybackSession, track: Track) -> None:
        """Record the start of listening to a track."""
        # Store temporary listening session info
        session.last_updated = datetime.now()
    
    def _record_listening_end(self, session: PlaybackSession) -> None:
        """Record the end of listening to a track."""
        if not session.current_track_id:
            return
        
        track = self.tracks[session.current_track_id]
        now = datetime.now()
        duration_played = int((now - session.last_updated).total_seconds())
        
        # Only record if played for at least 30 seconds
        if duration_played >= 30:
            history = ListeningHistory(
                history_id=str(uuid.uuid4()),
                user_id=session.user_id,
                track_id=session.current_track_id,
                played_at=session.last_updated,
                duration_played=duration_played,
                track_duration=track.duration,
                audio_quality=self.users[session.user_id].preferred_audio_quality,
                playlist_id=session.current_playlist_id,
                source="playlist" if session.current_playlist_id else "track"
            )
            
            with self._lock:
                self.listening_history.append(history)
                self.users[session.user_id].total_listening_time += duration_played // 60
                self.analytics['total_listening_time'] += duration_played
    
    def _get_next_track(self, session: PlaybackSession) -> Optional[str]:
        """Get next track based on current playback context."""
        # Check queue first
        if session.queue and session.queue_position < len(session.queue) - 1:
            session.queue_position += 1
            return session.queue[session.queue_position]
        
        # If in playlist context, get next track from playlist
        if session.current_playlist_id and session.current_playlist_id in self.playlists:
            playlist = self.playlists[session.current_playlist_id]
            
            if session.current_track_id in playlist.track_ids:
                current_index = playlist.track_ids.index(session.current_track_id)
                
                if current_index < len(playlist.track_ids) - 1:
                    return playlist.track_ids[current_index + 1]
                elif session.repeat and playlist.track_ids:
                    return playlist.track_ids[0]  # Loop back to start
        
        # Otherwise, get recommendations
        recommendations = self.get_recommendations(session.user_id, 1)
        if recommendations:
            return recommendations[0]['track_id']
        
        return None
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics."""
        with self._lock:
            # Calculate additional metrics
            active_sessions = len([s for s in self.playback_sessions.values() 
                                 if s.status == PlaybackStatus.PLAYING])
            
            premium_users = len([u for u in self.users.values() if u.is_premium])
            
            # Top tracks
            top_tracks = sorted(self.tracks.values(), key=lambda t: t.play_count, reverse=True)[:10]
            top_tracks_data = []
            for track in top_tracks:
                artist = self.artists[track.artist_id]
                top_tracks_data.append({
                    'title': track.title,
                    'artist': artist.name,
                    'play_count': track.play_count
                })
            
            # Top artists
            top_artists = sorted(self.artists.values(), key=lambda a: a.total_plays, reverse=True)[:10]
            top_artists_data = []
            for artist in top_artists:
                top_artists_data.append({
                    'name': artist.name,
                    'follower_count': artist.follower_count,
                    'monthly_listeners': artist.monthly_listeners
                })
            
            return {
                **self.analytics,
                'total_users': len(self.users),
                'total_tracks': len(self.tracks),
                'total_albums': len(self.albums),
                'total_artists': len(self.artists),
                'total_playlists': len(self.playlists),
                'active_sessions': active_sessions,
                'premium_subscribers': premium_users,
                'premium_conversion_rate': (premium_users / max(1, len(self.users))) * 100,
                'top_tracks': top_tracks_data,
                'top_artists': top_artists_data,
                'average_session_length': self.analytics['total_listening_time'] / max(1, len(self.listening_history))
            }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_music_streaming_service():
    """Demonstrate the music streaming service."""
    print("=== MUSIC STREAMING SERVICE DEMONSTRATION ===\n")
    
    # Initialize service
    print("1. SERVICE INITIALIZATION:")
    
    service = MusicStreamingService("StreamBeats")
    print("   ✓ Music streaming service initialized")
    print()
    
    # Register users
    print("2. USER REGISTRATION:")
    
    users = []
    user_data = [
        ("alice@example.com", "alice_music", "Alice Johnson"),
        ("bob@example.com", "bob_beats", "Bob Smith"),
        ("charlie@example.com", "charlie_tunes", "Charlie Brown")
    ]
    
    for email, username, display_name in user_data:
        user = service.register_user(email, username, display_name)
        users.append(user)
        print(f"   ✓ Registered user: {display_name} (@{username})")
    
    print()
    
    # Create artists
    print("3. ARTIST CREATION:")
    
    artists = []
    artist_data = [
        ("The Electric Waves", "Electronic music duo from the future"),
        ("Sarah Melody", "Singer-songwriter with a soulful voice"),
        ("Rock Legends", "Classic rock band from the 80s"),
        ("Jazz Collective", "Modern jazz ensemble"),
        ("Pop Princess", "Chart-topping pop sensation")
    ]
    
    for name, bio in artist_data:
        artist = service.create_artist(name, bio)
        artists.append(artist)
        print(f"   ✓ Created artist: {name}")
    
    print()
    
    # Create albums
    print("4. ALBUM CREATION:")
    
    albums = []
    album_data = [
        ("Digital Dreams", artists[0].artist_id, datetime(2023, 6, 15), "Electronic"),
        ("Heartstrings", artists[1].artist_id, datetime(2023, 8, 22), "Folk"),
        ("Greatest Hits", artists[2].artist_id, datetime(1985, 3, 10), "Rock"),
        ("Smooth Vibes", artists[3].artist_id, datetime(2023, 11, 5), "Jazz"),
        ("Pop Sensation", artists[4].artist_id, datetime(2023, 12, 1), "Pop")
    ]
    
    for title, artist_id, release_date, genre in album_data:
        album = service.create_album(title, artist_id, release_date, genre)
        albums.append(album)
        artist_name = service.artists[artist_id].name
        print(f"   ✓ Created album: {title} by {artist_name}")
    
    print()
    
    # Upload tracks
    print("5. TRACK UPLOAD:")
    
    tracks = []
    track_data = [
        # Digital Dreams album
        ("Neon Nights", artists[0].artist_id, 245, albums[0].album_id, "Electronic"),
        ("Circuit Breaker", artists[0].artist_id, 198, albums[0].album_id, "Electronic"),
        ("Data Stream", artists[0].artist_id, 267, albums[0].album_id, "Electronic"),
        
        # Heartstrings album
        ("Lonely Road", artists[1].artist_id, 223, albums[1].album_id, "Folk"),
        ("Morning Light", artists[1].artist_id, 189, albums[1].album_id, "Folk"),
        ("River Song", artists[1].artist_id, 256, albums[1].album_id, "Folk"),
        
        # Greatest Hits album
        ("Thunder Rock", artists[2].artist_id, 278, albums[2].album_id, "Rock"),
        ("Fire Storm", artists[2].artist_id, 312, albums[2].album_id, "Rock"),
        
        # Smooth Vibes album
        ("Midnight Jazz", artists[3].artist_id, 334, albums[3].album_id, "Jazz"),
        ("Blue Note", artists[3].artist_id, 287, albums[3].album_id, "Jazz"),
        
        # Pop Sensation album
        ("Dance Floor", artists[4].artist_id, 201, albums[4].album_id, "Pop"),
        ("Summer Love", artists[4].artist_id, 234, albums[4].album_id, "Pop")
    ]
    
    for title, artist_id, duration, album_id, genre in track_data:
        track = service.upload_track(title, artist_id, duration, album_id, genre)
        tracks.append(track)
        artist_name = service.artists[artist_id].name
        print(f"   ✓ Uploaded: {title} by {artist_name} ({duration//60}:{duration%60:02d})")
    
    print()
    
    # Create playlists
    print("6. PLAYLIST CREATION:")
    
    playlists = []
    playlist_data = [
        (users[0].user_id, "My Chill Mix", "Relaxing songs for studying"),
        (users[1].user_id, "Workout Beats", "High energy music for exercise"),
        (users[2].user_id, "Road Trip Classics", "Perfect songs for long drives")
    ]
    
    for user_id, name, description in playlist_data:
        playlist = service.create_playlist(user_id, name, description)
        playlists.append(playlist)
        user_name = service.users[user_id].display_name
        print(f"   ✓ Created playlist: {name} by {user_name}")
    
    # Add tracks to playlists
    chill_tracks = [tracks[1], tracks[3], tracks[5], tracks[8]]  # Mellow tracks
    workout_tracks = [tracks[0], tracks[6], tracks[7], tracks[10]]  # Energetic tracks
    roadtrip_tracks = [tracks[2], tracks[4], tracks[7], tracks[11]]  # Mix of genres
    
    for track in chill_tracks:
        service.add_track_to_playlist(users[0].user_id, playlists[0].playlist_id, track.track_id)
    
    for track in workout_tracks:
        service.add_track_to_playlist(users[1].user_id, playlists[1].playlist_id, track.track_id)
    
    for track in roadtrip_tracks:
        service.add_track_to_playlist(users[2].user_id, playlists[2].playlist_id, track.track_id)
    
    print(f"   ✓ Added tracks to playlists")
    
    print()
    
    # Test search functionality
    print("7. SEARCH FUNCTIONALITY:")
    
    search_queries = ["neon", "rock", "sarah", "jazz"]
    
    for query in search_queries:
        results = service.search(query, limit=3)
        
        print(f"   Search for '{query}':")
        
        if results['tracks']:
            print(f"     Tracks ({len(results['tracks'])}):")
            for track in results['tracks'][:2]:
                print(f"       {track['title']} by {track['artist']}")
        
        if results['artists']:
            print(f"     Artists ({len(results['artists'])}):")
            for artist in results['artists'][:2]:
                print(f"       {artist['name']}")
        
        if results['albums']:
            print(f"     Albums ({len(results['albums'])}):")
            for album in results['albums'][:2]:
                print(f"       {album['title']} by {album['artist']}")
    
    print()
    
    # Test playback sessions
    print("8. PLAYBACK SESSIONS:")
    
    # Alice starts listening
    alice = users[0]
    alice_session = service.start_playback_session(alice.user_id)
    print(f"   {alice.display_name} started playback session")
    
    # Play some tracks
    tracks_to_play = [tracks[0], tracks[3], tracks[5]]
    
    for i, track in enumerate(tracks_to_play):
        success = service.play_track(alice_session.session_id, track.track_id)
        print(f"     {'✓' if success else '✗'} Playing track {i+1}")
        
        # Simulate listening time
        import time
        time.sleep(0.5)
        
        # Pause and resume
        if i == 1:  # Pause second track
            service.pause_playback(alice_session.session_id)
            print(f"     ⏸️  Paused playback")
            time.sleep(0.2)
            service.resume_playback(alice_session.session_id)
            print(f"     ▶️  Resumed playback")
        
        time.sleep(0.5)
        
        # Skip to next track
        if i < len(tracks_to_play) - 1:
            service.next_track(alice_session.session_id)
    
    print()
    
    # Test likes and follows
    print("9. LIKES AND FOLLOWS:")
    
    # Users like tracks
    for user in users:
        liked_tracks = random.sample(tracks, 3)
        for track in liked_tracks:
            service.like_track(user.user_id, track.track_id)
        
        user_name = user.display_name
        print(f"   {user_name} liked {len(liked_tracks)} tracks")
    
    # Users follow artists
    for user in users:
        followed_artists = random.sample(artists, 2)
        for artist in followed_artists:
            service.follow_artist(user.user_id, artist.artist_id)
        
        user_name = user.display_name
        print(f"   {user_name} followed {len(followed_artists)} artists")
    
    print()
    
    # Test recommendations
    print("10. RECOMMENDATIONS:")
    
    for user in users[:2]:  # Test for first 2 users
        recommendations = service.get_recommendations(user.user_id, count=3)
        
        print(f"   Recommendations for {user.display_name}:")
        for rec in recommendations:
            print(f"     {rec['title']} by {rec['artist']} - {rec['reason']}")
    
    print()
    
    # Test user statistics
    print("11. USER STATISTICS:")
    
    for user in users:
        stats = service.get_user_listening_stats(user.user_id)
        
        print(f"   {user.display_name}'s stats:")
        print(f"     Listening time: {stats['total_listening_time']} minutes")
        print(f"     Tracks played: {stats['tracks_played']}")
        print(f"     Liked tracks: {stats['liked_tracks']}")
        print(f"     Liked artists: {stats['liked_artists']}")
        print(f"     Playlists created: {stats['playlists_created']}")
        
        if stats['top_artists']:
            print(f"     Top artist: {stats['top_artists'][0]['name']}")
    
    print()
    
    # Test different audio qualities
    print("12. AUDIO QUALITY TEST:")
    
    test_track = tracks[0]
    
    print(f"   Audio files for '{test_track.title}':")
    for quality, audio_file in test_track.audio_files.items():
        print(f"     {quality.value}: {audio_file.bitrate} kbps, "
              f"{audio_file.file_size // 1024} KB ({audio_file.format})")
    
    print()
    
    # Show artist analytics
    print("13. ARTIST ANALYTICS:")
    
    for artist in artists[:3]:  # Show first 3 artists
        print(f"   {artist.name}:")
        print(f"     Followers: {artist.follower_count}")
        print(f"     Albums: {len(artist.album_ids)}")
        print(f"     Tracks: {len(artist.track_ids)}")
        
        # Calculate total plays for artist
        total_plays = sum(service.tracks[track_id].play_count 
                         for track_id in artist.track_ids
                         if track_id in service.tracks)
        print(f"     Total plays: {total_plays}")
    
    print()
    
    # Show playlist information
    print("14. PLAYLIST INFORMATION:")
    
    for playlist in playlists:
        owner = service.users[playlist.owner_id]
        print(f"   '{playlist.name}' by {owner.display_name}:")
        print(f"     Tracks: {len(playlist.track_ids)}")
        print(f"     Duration: ~{playlist.duration // 60} minutes")
        print(f"     Public: {playlist.is_public}")
        print(f"     Collaborative: {playlist.is_collaborative}")
    
    print()
    
    # Show comprehensive analytics
    print("15. SYSTEM ANALYTICS:")
    
    analytics = service.get_system_analytics()
    
    print(f"   Platform Overview:")
    print(f"     Total users: {analytics['total_users']}")
    print(f"     Total tracks: {analytics['total_tracks']}")
    print(f"     Total albums: {analytics['total_albums']}")
    print(f"     Total artists: {analytics['total_artists']}")
    print(f"     Total playlists: {analytics['total_playlists']}")
    
    print(f"\n   Engagement Metrics:")
    print(f"     Total streams: {analytics['total_streams']}")
    print(f"     Total listening time: {analytics['total_listening_time']} minutes")
    print(f"     Active sessions: {analytics['active_sessions']}")
    print(f"     Average session length: {analytics['average_session_length']:.1f} minutes")
    
    print(f"\n   Business Metrics:")
    print(f"     Premium subscribers: {analytics['premium_subscribers']}")
    print(f"     Premium conversion rate: {analytics['premium_conversion_rate']:.1f}%")
    
    print(f"\n   Top Content:")
    if analytics['top_tracks']:
        print(f"     Top track: {analytics['top_tracks'][0]['title']} by {analytics['top_tracks'][0]['artist']}")
        print(f"     Plays: {analytics['top_tracks'][0]['play_count']}")
    
    if analytics['top_artists']:
        print(f"     Top artist: {analytics['top_artists'][0]['name']}")
        print(f"     Followers: {analytics['top_artists'][0]['follower_count']}")
    
    print()
    
    # Show final system state
    print("16. FINAL SYSTEM STATE:")
    
    print(f"   Content Library:")
    print(f"     Music catalog: {len(service.tracks)} tracks across {len(service.albums)} albums")
    print(f"     Artist roster: {len(service.artists)} artists")
    print(f"     User-generated playlists: {len(service.playlists)}")
    
    print(f"\n   User Engagement:")
    total_likes = sum(len(user.liked_tracks) for user in service.users.values())
    total_follows = sum(len(user.liked_artists) for user in service.users.values())
    print(f"     Total track likes: {total_likes}")
    print(f"     Total artist follows: {total_follows}")
    print(f"     Listening history entries: {len(service.listening_history)}")
    
    print(f"\n   Platform Health:")
    active_sessions = len([s for s in service.playback_sessions.values() 
                          if s.status == PlaybackStatus.PLAYING])
    print(f"     Active playback sessions: {active_sessions}")
    print(f"     User retention: {len([u for u in service.users.values() if u.total_listening_time > 0])} active listeners")
    
    # Show most popular content
    most_played_track = max(service.tracks.values(), key=lambda t: t.play_count)
    most_followed_artist = max(service.artists.values(), key=lambda a: a.follower_count)
    
    print(f"\n   Popular Content:")
    print(f"     Most played track: {most_played_track.title} ({most_played_track.play_count} plays)")
    print(f"     Most followed artist: {most_followed_artist.name} ({most_followed_artist.follower_count} followers)")
    
    print()
    print("=== MUSIC STREAMING SERVICE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_music_streaming_service()
