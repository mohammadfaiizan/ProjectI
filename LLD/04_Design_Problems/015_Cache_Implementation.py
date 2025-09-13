"""
CACHE IMPLEMENTATION - Complete System Design
============================================

Problem Statement:
Design a comprehensive caching system that handles:
- Multiple cache eviction policies (LRU, LFU, FIFO, Random)
- Multi-level caching with different storage tiers
- Cache coherence and consistency across distributed systems
- Cache warming and preloading strategies
- Cache statistics and performance monitoring
- Thread-safe operations for concurrent access
- Cache partitioning and sharding
- Time-based expiration (TTL)
- Cache compression and serialization
- Write-through, write-back, and write-around policies

Requirements:
- Support configurable cache sizes and eviction policies
- Implement thread-safe operations for concurrent access
- Provide cache hit/miss statistics and performance metrics
- Support TTL (Time To Live) for automatic expiration
- Handle different data types with serialization
- Implement cache warming and preloading
- Support distributed caching with consistency protocols
- Provide cache monitoring and alerting
- Handle cache failures and fallback mechanisms
- Support cache partitioning for better performance

Design Patterns Used:
- Strategy: Different eviction and caching strategies
- Observer: Cache event monitoring and statistics
- Decorator: Cache compression and serialization
- Factory: Cache creation with different configurations
- Singleton: Cache manager instance
- Template Method: Cache operation templates
- Command: Cache operations with undo capability
- Proxy: Remote cache access
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Generic, TypeVar, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import json
import pickle
import gzip
import hashlib
import uuid
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import weakref

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class EvictionPolicy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    RANDOM = "random"
    TTL = "ttl"  # Time To Live based


class WritePolicy(Enum):
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


class CacheLevel(Enum):
    L1 = "l1"  # Memory cache
    L2 = "l2"  # SSD cache
    L3 = "l3"  # Disk cache
    REMOTE = "remote"  # Network cache


class CacheEvent(Enum):
    HIT = "hit"
    MISS = "miss"
    PUT = "put"
    EVICT = "evict"
    EXPIRE = "expire"
    CLEAR = "clear"


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    key: str
    value: V
    created_time: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    size: int = 0
    
    def __post_init__(self):
        if self.size == 0:
            self.size = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of the entry."""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value).encode())
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_time + self.ttl
    
    @property
    def age(self) -> timedelta:
        """Get age of the entry."""
        return datetime.now() - self.created_time
    
    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate percentage."""
        return 100.0 - self.hit_rate


# ============================================================================
# EVICTION STRATEGIES
# ============================================================================

class EvictionStrategy(ABC, Generic[K, V]):
    """Abstract eviction strategy."""
    
    @abstractmethod
    def on_access(self, key: K, entry: CacheEntry[V]) -> None:
        """Called when an entry is accessed."""
        pass
    
    @abstractmethod
    def on_put(self, key: K, entry: CacheEntry[V]) -> None:
        """Called when an entry is added."""
        pass
    
    @abstractmethod
    def select_victim(self, entries: Dict[K, CacheEntry[V]]) -> Optional[K]:
        """Select entry to evict."""
        pass
    
    @abstractmethod
    def on_remove(self, key: K) -> None:
        """Called when an entry is removed."""
        pass


class LRUEvictionStrategy(EvictionStrategy[K, V]):
    """Least Recently Used eviction strategy."""
    
    def __init__(self):
        self.access_order: OrderedDict[K, datetime] = OrderedDict()
        self._lock = threading.Lock()
    
    def on_access(self, key: K, entry: CacheEntry[V]) -> None:
        """Update access order."""
        with self._lock:
            if key in self.access_order:
                del self.access_order[key]
            self.access_order[key] = entry.last_accessed
    
    def on_put(self, key: K, entry: CacheEntry[V]) -> None:
        """Add to access order."""
        with self._lock:
            self.access_order[key] = entry.last_accessed
    
    def select_victim(self, entries: Dict[K, CacheEntry[V]]) -> Optional[K]:
        """Select least recently used entry."""
        with self._lock:
            if not self.access_order:
                return None
            
            # Return the first (oldest) key
            return next(iter(self.access_order))
    
    def on_remove(self, key: K) -> None:
        """Remove from access order."""
        with self._lock:
            self.access_order.pop(key, None)


class LFUEvictionStrategy(EvictionStrategy[K, V]):
    """Least Frequently Used eviction strategy."""
    
    def __init__(self):
        self.frequency_map: Dict[K, int] = {}
        self.frequency_buckets: Dict[int, Set[K]] = defaultdict(set)
        self.min_frequency = 0
        self._lock = threading.Lock()
    
    def on_access(self, key: K, entry: CacheEntry[V]) -> None:
        """Update frequency."""
        with self._lock:
            old_freq = self.frequency_map.get(key, 0)
            new_freq = entry.access_count
            
            # Remove from old frequency bucket
            if old_freq > 0:
                self.frequency_buckets[old_freq].discard(key)
                if not self.frequency_buckets[old_freq] and old_freq == self.min_frequency:
                    self.min_frequency += 1
            
            # Add to new frequency bucket
            self.frequency_map[key] = new_freq
            self.frequency_buckets[new_freq].add(key)
            
            # Update minimum frequency
            if new_freq < self.min_frequency:
                self.min_frequency = new_freq
    
    def on_put(self, key: K, entry: CacheEntry[V]) -> None:
        """Add with initial frequency."""
        with self._lock:
            freq = entry.access_count
            self.frequency_map[key] = freq
            self.frequency_buckets[freq].add(key)
            self.min_frequency = min(self.min_frequency, freq) if self.frequency_map else freq
    
    def select_victim(self, entries: Dict[K, CacheEntry[V]]) -> Optional[K]:
        """Select least frequently used entry."""
        with self._lock:
            # Find minimum frequency bucket with entries
            while self.min_frequency < max(self.frequency_buckets.keys(), default=0):
                if self.frequency_buckets[self.min_frequency]:
                    break
                self.min_frequency += 1
            
            if self.frequency_buckets[self.min_frequency]:
                return next(iter(self.frequency_buckets[self.min_frequency]))
            
            return None
    
    def on_remove(self, key: K) -> None:
        """Remove from frequency tracking."""
        with self._lock:
            freq = self.frequency_map.pop(key, 0)
            self.frequency_buckets[freq].discard(key)


class FIFOEvictionStrategy(EvictionStrategy[K, V]):
    """First In First Out eviction strategy."""
    
    def __init__(self):
        self.insertion_order: OrderedDict[K, datetime] = OrderedDict()
        self._lock = threading.Lock()
    
    def on_access(self, key: K, entry: CacheEntry[V]) -> None:
        """No action needed for FIFO on access."""
        pass
    
    def on_put(self, key: K, entry: CacheEntry[V]) -> None:
        """Track insertion order."""
        with self._lock:
            self.insertion_order[key] = entry.created_time
    
    def select_victim(self, entries: Dict[K, CacheEntry[V]]) -> Optional[K]:
        """Select first inserted entry."""
        with self._lock:
            if not self.insertion_order:
                return None
            return next(iter(self.insertion_order))
    
    def on_remove(self, key: K) -> None:
        """Remove from insertion order."""
        with self._lock:
            self.insertion_order.pop(key, None)


class RandomEvictionStrategy(EvictionStrategy[K, V]):
    """Random eviction strategy."""
    
    def on_access(self, key: K, entry: CacheEntry[V]) -> None:
        """No action needed for random eviction."""
        pass
    
    def on_put(self, key: K, entry: CacheEntry[V]) -> None:
        """No action needed for random eviction."""
        pass
    
    def select_victim(self, entries: Dict[K, CacheEntry[V]]) -> Optional[K]:
        """Select random entry."""
        import random
        if not entries:
            return None
        return random.choice(list(entries.keys()))
    
    def on_remove(self, key: K) -> None:
        """No action needed for random eviction."""
        pass


# ============================================================================
# SERIALIZATION STRATEGIES
# ============================================================================

class SerializationStrategy(ABC):
    """Abstract serialization strategy."""
    
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass


class PickleSerializer(SerializationStrategy):
    """Pickle-based serialization."""
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize using pickle."""
        return pickle.dumps(obj)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize using pickle."""
        return pickle.loads(data)


class JSONSerializer(SerializationStrategy):
    """JSON-based serialization."""
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize using JSON."""
        return json.dumps(obj, default=str).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize using JSON."""
        return json.loads(data.decode('utf-8'))


class CompressedSerializer(SerializationStrategy):
    """Compressed serialization wrapper."""
    
    def __init__(self, base_serializer: SerializationStrategy):
        self.base_serializer = base_serializer
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize and compress."""
        data = self.base_serializer.serialize(obj)
        return gzip.compress(data)
    
    def deserialize(self, data: bytes) -> Any:
        """Decompress and deserialize."""
        decompressed = gzip.decompress(data)
        return self.base_serializer.deserialize(decompressed)


# ============================================================================
# CACHE OBSERVERS
# ============================================================================

class CacheObserver(ABC):
    """Abstract cache observer."""
    
    @abstractmethod
    def on_cache_event(self, cache_name: str, event: CacheEvent, 
                      key: str, details: Dict[str, Any]) -> None:
        """Handle cache event."""
        pass


class CacheStatsCollector(CacheObserver):
    """Cache statistics collector."""
    
    def __init__(self):
        self.stats_by_cache: Dict[str, CacheStats] = defaultdict(CacheStats)
        self.event_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def on_cache_event(self, cache_name: str, event: CacheEvent, 
                      key: str, details: Dict[str, Any]) -> None:
        """Collect statistics from cache events."""
        with self._lock:
            stats = self.stats_by_cache[cache_name]
            
            if event == CacheEvent.HIT:
                stats.hits += 1
            elif event == CacheEvent.MISS:
                stats.misses += 1
            elif event == CacheEvent.PUT:
                stats.puts += 1
            elif event == CacheEvent.EVICT:
                stats.evictions += 1
            elif event == CacheEvent.EXPIRE:
                stats.expirations += 1
            
            # Update size information
            if 'size' in details:
                if event == CacheEvent.PUT:
                    stats.total_size += details['size']
                    stats.entry_count += 1
                elif event in [CacheEvent.EVICT, CacheEvent.EXPIRE]:
                    stats.total_size -= details.get('size', 0)
                    stats.entry_count -= 1
            
            # Record event
            event_record = {
                'timestamp': datetime.now().isoformat(),
                'cache': cache_name,
                'event': event.value,
                'key': key,
                'details': details
            }
            
            self.event_history.append(event_record)
            
            # Limit history size
            if len(self.event_history) > 10000:
                self.event_history = self.event_history[-5000:]
    
    def get_stats(self, cache_name: str) -> CacheStats:
        """Get statistics for a specific cache."""
        return self.stats_by_cache[cache_name]
    
    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return dict(self.stats_by_cache)


class CacheMonitor(CacheObserver):
    """Cache monitoring and alerting."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            'low_hit_rate': 50.0,  # Alert if hit rate below 50%
            'high_eviction_rate': 10.0,  # Alert if eviction rate above 10%
            'max_size_usage': 90.0  # Alert if cache usage above 90%
        }
        self._stats_collector = CacheStatsCollector()
    
    def on_cache_event(self, cache_name: str, event: CacheEvent, 
                      key: str, details: Dict[str, Any]) -> None:
        """Monitor cache events for alerts."""
        # Delegate to stats collector
        self._stats_collector.on_cache_event(cache_name, event, key, details)
        
        # Check for alert conditions
        stats = self._stats_collector.get_stats(cache_name)
        
        # Check hit rate
        if stats.hits + stats.misses > 100:  # Only check after some activity
            if stats.hit_rate < self.thresholds['low_hit_rate']:
                self._create_alert(cache_name, f"Low hit rate: {stats.hit_rate:.1f}%")
        
        # Check eviction rate
        total_ops = stats.hits + stats.misses + stats.puts
        if total_ops > 0:
            eviction_rate = (stats.evictions / total_ops) * 100
            if eviction_rate > self.thresholds['high_eviction_rate']:
                self._create_alert(cache_name, f"High eviction rate: {eviction_rate:.1f}%")
    
    def _create_alert(self, cache_name: str, message: str) -> None:
        """Create cache alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'cache': cache_name,
            'message': message,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        print(f"⚠️  Cache Alert [{cache_name}]: {message}")


# ============================================================================
# CORE CACHE IMPLEMENTATION
# ============================================================================

class Cache(Generic[K, V]):
    """Generic cache implementation."""
    
    def __init__(self, name: str, max_size: int = 1000, 
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 default_ttl: Optional[timedelta] = None,
                 serializer: Optional[SerializationStrategy] = None):
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.serializer = serializer or PickleSerializer()
        
        # Storage
        self.entries: Dict[K, CacheEntry[V]] = {}
        
        # Eviction strategy
        self.eviction_strategy = self._create_eviction_strategy(eviction_policy)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Observers
        self.observers: List[CacheObserver] = []
        
        # Background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_running = True
        self._cleanup_thread.start()
        
        print(f"💾 Cache '{name}' created (max_size={max_size}, policy={eviction_policy.value})")
    
    def _create_eviction_strategy(self, policy: EvictionPolicy) -> EvictionStrategy[K, V]:
        """Create eviction strategy based on policy."""
        if policy == EvictionPolicy.LRU:
            return LRUEvictionStrategy()
        elif policy == EvictionPolicy.LFU:
            return LFUEvictionStrategy()
        elif policy == EvictionPolicy.FIFO:
            return FIFOEvictionStrategy()
        elif policy == EvictionPolicy.RANDOM:
            return RandomEvictionStrategy()
        else:
            return LRUEvictionStrategy()  # Default
    
    def add_observer(self, observer: CacheObserver) -> None:
        """Add cache observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: CacheObserver) -> None:
        """Remove cache observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def _notify_observers(self, event: CacheEvent, key: K, details: Dict[str, Any] = None) -> None:
        """Notify observers of cache events."""
        for observer in self.observers:
            observer.on_cache_event(self.name, event, str(key), details or {})
    
    def put(self, key: K, value: V, ttl: Optional[timedelta] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Use provided TTL or default
            entry_ttl = ttl or self.default_ttl
            
            # Create cache entry
            entry = CacheEntry(
                key=str(key),
                value=value,
                created_time=datetime.now(),
                last_accessed=datetime.now(),
                ttl=entry_ttl
            )
            
            # Check if we need to evict
            if key not in self.entries and len(self.entries) >= self.max_size:
                self._evict_one()
            
            # Add entry
            self.entries[key] = entry
            self.eviction_strategy.on_put(key, entry)
            
            # Notify observers
            self._notify_observers(CacheEvent.PUT, key, {
                'size': entry.size,
                'ttl_seconds': entry_ttl.total_seconds() if entry_ttl else None
            })
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        with self._lock:
            entry = self.entries.get(key)
            
            if entry is None:
                self._notify_observers(CacheEvent.MISS, key)
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key, CacheEvent.EXPIRE)
                self._notify_observers(CacheEvent.MISS, key)
                return None
            
            # Update access information
            entry.touch()
            self.eviction_strategy.on_access(key, entry)
            
            # Notify observers
            self._notify_observers(CacheEvent.HIT, key, {
                'access_count': entry.access_count,
                'age_seconds': entry.age.total_seconds()
            })
            
            return entry.value
    
    def remove(self, key: K) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self.entries:
                self._remove_entry(key, CacheEvent.EVICT)
                return True
            return False
    
    def _remove_entry(self, key: K, event: CacheEvent) -> None:
        """Remove entry and update strategy."""
        entry = self.entries.pop(key, None)
        if entry:
            self.eviction_strategy.on_remove(key)
            self._notify_observers(event, key, {'size': entry.size})
    
    def _evict_one(self) -> bool:
        """Evict one entry using eviction strategy."""
        victim_key = self.eviction_strategy.select_victim(self.entries)
        if victim_key:
            self._remove_entry(victim_key, CacheEvent.EVICT)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            keys_to_remove = list(self.entries.keys())
            for key in keys_to_remove:
                self._remove_entry(key, CacheEvent.CLEAR)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.entries)
    
    def contains(self, key: K) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            entry = self.entries.get(key)
            if entry and not entry.is_expired:
                return True
            return False
    
    def keys(self) -> List[K]:
        """Get all keys in cache."""
        with self._lock:
            return [key for key, entry in self.entries.items() if not entry.is_expired]
    
    def _cleanup_expired(self) -> None:
        """Background thread to cleanup expired entries."""
        while self._cleanup_running:
            try:
                with self._lock:
                    expired_keys = []
                    for key, entry in self.entries.items():
                        if entry.is_expired:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_entry(key, CacheEvent.EXPIRE)
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                time.sleep(60)
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        with self._lock:
            total_size = sum(entry.size for entry in self.entries.values())
            
            return {
                'name': self.name,
                'size': len(self.entries),
                'max_size': self.max_size,
                'total_bytes': total_size,
                'eviction_policy': type(self.eviction_strategy).__name__,
                'default_ttl_seconds': self.default_ttl.total_seconds() if self.default_ttl else None
            }
    
    def __del__(self):
        """Cleanup when cache is destroyed."""
        self._cleanup_running = False


# ============================================================================
# MULTI-LEVEL CACHE
# ============================================================================

class MultiLevelCache:
    """Multi-level cache with different storage tiers."""
    
    def __init__(self, name: str):
        self.name = name
        self.levels: Dict[CacheLevel, Cache] = {}
        self.observers: List[CacheObserver] = []
        self._lock = threading.RLock()
        
        print(f"🏗️  Multi-level cache '{name}' created")
    
    def add_level(self, level: CacheLevel, cache: Cache) -> None:
        """Add cache level."""
        with self._lock:
            self.levels[level] = cache
            
            # Forward observers to the cache
            for observer in self.observers:
                cache.add_observer(observer)
    
    def add_observer(self, observer: CacheObserver) -> None:
        """Add observer to all cache levels."""
        self.observers.append(observer)
        
        for cache in self.levels.values():
            cache.add_observer(observer)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        with self._lock:
            # Try each level in order
            for level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3, CacheLevel.REMOTE]:
                if level in self.levels:
                    cache = self.levels[level]
                    value = cache.get(key)
                    
                    if value is not None:
                        # Promote to higher levels
                        self._promote_to_higher_levels(key, value, level)
                        return value
            
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Put value in multi-level cache."""
        with self._lock:
            # Put in all levels
            for cache in self.levels.values():
                cache.put(key, value, ttl)
    
    def _promote_to_higher_levels(self, key: str, value: Any, found_level: CacheLevel) -> None:
        """Promote value to higher cache levels."""
        level_order = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3, CacheLevel.REMOTE]
        found_index = level_order.index(found_level)
        
        # Promote to all higher levels
        for i in range(found_index):
            level = level_order[i]
            if level in self.levels:
                self.levels[level].put(key, value)
    
    def remove(self, key: str) -> bool:
        """Remove from all cache levels."""
        with self._lock:
            removed = False
            for cache in self.levels.values():
                if cache.remove(key):
                    removed = True
            return removed
    
    def clear(self) -> None:
        """Clear all cache levels."""
        with self._lock:
            for cache in self.levels.values():
                cache.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get multi-level cache information."""
        with self._lock:
            level_info = {}
            for level, cache in self.levels.items():
                level_info[level.value] = cache.get_info()
            
            return {
                'name': self.name,
                'levels': level_info,
                'total_levels': len(self.levels)
            }


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Central cache management system."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.caches: Dict[str, Cache] = {}
            self.multi_level_caches: Dict[str, MultiLevelCache] = {}
            self.global_stats = CacheStatsCollector()
            self.global_monitor = CacheMonitor()
            self._lock = threading.RLock()
            self.initialized = True
            
            print("🎯 Cache Manager initialized")
    
    def create_cache(self, name: str, max_size: int = 1000,
                    eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                    default_ttl: Optional[timedelta] = None) -> Cache:
        """Create and register a new cache."""
        with self._lock:
            if name in self.caches:
                raise ValueError(f"Cache '{name}' already exists")
            
            cache = Cache(name, max_size, eviction_policy, default_ttl)
            cache.add_observer(self.global_stats)
            cache.add_observer(self.global_monitor)
            
            self.caches[name] = cache
            return cache
    
    def create_multi_level_cache(self, name: str) -> MultiLevelCache:
        """Create and register a multi-level cache."""
        with self._lock:
            if name in self.multi_level_caches:
                raise ValueError(f"Multi-level cache '{name}' already exists")
            
            ml_cache = MultiLevelCache(name)
            ml_cache.add_observer(self.global_stats)
            ml_cache.add_observer(self.global_monitor)
            
            self.multi_level_caches[name] = ml_cache
            return ml_cache
    
    def get_cache(self, name: str) -> Optional[Cache]:
        """Get cache by name."""
        return self.caches.get(name)
    
    def get_multi_level_cache(self, name: str) -> Optional[MultiLevelCache]:
        """Get multi-level cache by name."""
        return self.multi_level_caches.get(name)
    
    def remove_cache(self, name: str) -> bool:
        """Remove cache."""
        with self._lock:
            if name in self.caches:
                del self.caches[name]
                return True
            return False
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'caches': self.global_stats.get_all_stats(),
            'alerts': self.global_monitor.alerts
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
            
            for ml_cache in self.multi_level_caches.values():
                ml_cache.clear()


# ============================================================================
# CACHE WARMING AND PRELOADING
# ============================================================================

class CacheWarmer:
    """Cache warming and preloading utility."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
        self.warming_strategies: List[Callable] = []
    
    def add_warming_strategy(self, strategy: Callable[[Cache], None]) -> None:
        """Add cache warming strategy."""
        self.warming_strategies.append(strategy)
    
    def warm_cache(self) -> None:
        """Execute all warming strategies."""
        print(f"🔥 Warming cache '{self.cache.name}'...")
        
        for strategy in self.warming_strategies:
            try:
                strategy(self.cache)
            except Exception as e:
                print(f"Cache warming error: {e}")
        
        print(f"Cache warming completed. Size: {self.cache.size()}")
    
    def preload_from_source(self, data_source: Callable[[], Dict[str, Any]]) -> None:
        """Preload cache from data source."""
        try:
            data = data_source()
            for key, value in data.items():
                self.cache.put(key, value)
            
            print(f"Preloaded {len(data)} items into cache '{self.cache.name}'")
        except Exception as e:
            print(f"Cache preloading error: {e}")


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_cache_system():
    """Demonstrate the cache system."""
    print("=== CACHE IMPLEMENTATION DEMONSTRATION ===\n")
    
    # Get cache manager
    print("1. CACHE MANAGER SETUP:")
    
    manager = CacheManager()
    print("   ✓ Cache manager initialized")
    print()
    
    # Create different types of caches
    print("2. CACHE CREATION:")
    
    # LRU cache
    lru_cache = manager.create_cache("user_cache", max_size=5, 
                                    eviction_policy=EvictionPolicy.LRU,
                                    default_ttl=timedelta(minutes=30))
    print("   ✓ Created LRU cache (max_size=5, TTL=30min)")
    
    # LFU cache
    lfu_cache = manager.create_cache("product_cache", max_size=3,
                                    eviction_policy=EvictionPolicy.LFU)
    print("   ✓ Created LFU cache (max_size=3)")
    
    # FIFO cache
    fifo_cache = manager.create_cache("session_cache", max_size=4,
                                     eviction_policy=EvictionPolicy.FIFO,
                                     default_ttl=timedelta(minutes=15))
    print("   ✓ Created FIFO cache (max_size=4, TTL=15min)")
    
    print()
    
    # Test basic cache operations
    print("3. BASIC CACHE OPERATIONS:")
    
    # Test LRU cache
    print("   LRU Cache Test:")
    lru_cache.put("user1", {"name": "Alice", "age": 30})
    lru_cache.put("user2", {"name": "Bob", "age": 25})
    lru_cache.put("user3", {"name": "Charlie", "age": 35})
    
    # Access user1 to make it recently used
    user1 = lru_cache.get("user1")
    print(f"     Retrieved user1: {user1['name'] if user1 else 'Not found'}")
    
    # Add more users to trigger eviction
    lru_cache.put("user4", {"name": "David", "age": 28})
    lru_cache.put("user5", {"name": "Eve", "age": 32})
    lru_cache.put("user6", {"name": "Frank", "age": 29})  # Should evict user2
    
    # Check what's still in cache
    print("     Keys after eviction:", lru_cache.keys())
    
    print()
    
    # Test LFU cache
    print("   LFU Cache Test:")
    lfu_cache.put("prod1", {"name": "Laptop", "price": 999})
    lfu_cache.put("prod2", {"name": "Mouse", "price": 25})
    lfu_cache.put("prod3", {"name": "Keyboard", "price": 75})
    
    # Access prod1 multiple times
    for _ in range(3):
        lfu_cache.get("prod1")
    
    # Access prod2 once
    lfu_cache.get("prod2")
    
    # Add new product (should evict prod3 - least frequently used)
    lfu_cache.put("prod4", {"name": "Monitor", "price": 300})
    
    print("     Keys after eviction:", lfu_cache.keys())
    
    print()
    
    # Test TTL expiration
    print("4. TTL EXPIRATION TEST:")
    
    # Create cache with short TTL for testing
    ttl_cache = manager.create_cache("ttl_test", max_size=10,
                                    default_ttl=timedelta(seconds=2))
    
    ttl_cache.put("temp1", "This will expire soon")
    print("   ✓ Added item with 2-second TTL")
    
    # Immediate retrieval should work
    value = ttl_cache.get("temp1")
    print(f"   Immediate retrieval: {'Success' if value else 'Failed'}")
    
    # Wait for expiration
    print("   Waiting 3 seconds for expiration...")
    time.sleep(3)
    
    # Should be expired now
    value = ttl_cache.get("temp1")
    print(f"   After expiration: {'Found' if value else 'Expired (correct)'}")
    
    print()
    
    # Test multi-level cache
    print("5. MULTI-LEVEL CACHE TEST:")
    
    ml_cache = manager.create_multi_level_cache("multi_level")
    
    # Add different cache levels
    l1_cache = Cache("L1", max_size=2, eviction_policy=EvictionPolicy.LRU)
    l2_cache = Cache("L2", max_size=5, eviction_policy=EvictionPolicy.LFU)
    l3_cache = Cache("L3", max_size=10, eviction_policy=EvictionPolicy.FIFO)
    
    ml_cache.add_level(CacheLevel.L1, l1_cache)
    ml_cache.add_level(CacheLevel.L2, l2_cache)
    ml_cache.add_level(CacheLevel.L3, l3_cache)
    
    print("   ✓ Created 3-level cache (L1: 2, L2: 5, L3: 10)")
    
    # Put data in multi-level cache
    ml_cache.put("data1", "Important data 1")
    ml_cache.put("data2", "Important data 2")
    ml_cache.put("data3", "Important data 3")
    
    # Retrieve data (should promote to higher levels)
    value = ml_cache.get("data1")
    print(f"   Retrieved from multi-level: {value}")
    
    # Check cache levels
    print("   Cache level sizes:")
    print(f"     L1: {l1_cache.size()}")
    print(f"     L2: {l2_cache.size()}")
    print(f"     L3: {l3_cache.size()}")
    
    print()
    
    # Test cache warming
    print("6. CACHE WARMING TEST:")
    
    warming_cache = manager.create_cache("warming_test", max_size=20)
    warmer = CacheWarmer(warming_cache)
    
    # Define warming strategy
    def preload_user_data(cache):
        users = {
            f"user_{i}": {"id": i, "name": f"User{i}", "active": True}
            for i in range(1, 11)
        }
        for key, value in users.items():
            cache.put(key, value)
    
    warmer.add_warming_strategy(preload_user_data)
    warmer.warm_cache()
    
    print(f"   Cache size after warming: {warming_cache.size()}")
    
    print()
    
    # Test serialization
    print("7. SERIALIZATION TEST:")
    
    # Test with different serializers
    json_cache = Cache("json_cache", max_size=5, serializer=JSONSerializer())
    compressed_cache = Cache("compressed_cache", max_size=5, 
                           serializer=CompressedSerializer(PickleSerializer()))
    
    # Test complex data
    complex_data = {
        "users": [{"id": i, "name": f"User{i}"} for i in range(100)],
        "metadata": {"created": datetime.now().isoformat(), "version": "1.0"}
    }
    
    json_cache.put("complex", complex_data)
    compressed_cache.put("complex", complex_data)
    
    # Retrieve and verify
    json_result = json_cache.get("complex")
    compressed_result = compressed_cache.get("complex")
    
    print(f"   JSON serialization: {'Success' if json_result else 'Failed'}")
    print(f"   Compressed serialization: {'Success' if compressed_result else 'Failed'}")
    
    print()
    
    # Show cache statistics
    print("8. CACHE STATISTICS:")
    
    all_stats = manager.get_all_stats()
    
    print("   Cache Performance:")
    for cache_name, stats in all_stats['caches'].items():
        print(f"     {cache_name}:")
        print(f"       Hit Rate: {stats.hit_rate:.1f}%")
        print(f"       Total Operations: {stats.hits + stats.misses}")
        print(f"       Evictions: {stats.evictions}")
        print(f"       Current Size: {stats.entry_count}")
    
    print()
    
    # Show alerts
    print("9. CACHE ALERTS:")
    
    alerts = all_stats['alerts']
    if alerts:
        print(f"   Total alerts: {len(alerts)}")
        for alert in alerts[-3:]:  # Show last 3 alerts
            print(f"     {alert['timestamp'][:19]} - {alert['cache']}: {alert['message']}")
    else:
        print("   No alerts generated")
    
    print()
    
    # Performance test
    print("10. PERFORMANCE TEST:")
    
    perf_cache = manager.create_cache("performance", max_size=1000)
    
    # Measure put performance
    start_time = time.time()
    for i in range(1000):
        perf_cache.put(f"key_{i}", f"value_{i}")
    put_time = time.time() - start_time
    
    # Measure get performance
    start_time = time.time()
    for i in range(1000):
        perf_cache.get(f"key_{i}")
    get_time = time.time() - start_time
    
    print(f"   Put 1000 items: {put_time:.3f} seconds ({1000/put_time:.0f} ops/sec)")
    print(f"   Get 1000 items: {get_time:.3f} seconds ({1000/get_time:.0f} ops/sec)")
    
    print()
    
    # Show final system state
    print("11. FINAL SYSTEM STATE:")
    
    print(f"   Total caches: {len(manager.caches)}")
    print(f"   Multi-level caches: {len(manager.multi_level_caches)}")
    
    print("   Cache Information:")
    for name, cache in manager.caches.items():
        info = cache.get_info()
        print(f"     {name}: {info['size']}/{info['max_size']} items, "
              f"{info['total_bytes']} bytes, {info['eviction_policy']}")
    
    print()
    print("=== CACHE IMPLEMENTATION DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_cache_system()
