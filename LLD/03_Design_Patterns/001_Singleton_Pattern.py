"""
SINGLETON PATTERN - Creational Design Pattern
=============================================

Problem Statement:
Implement the Singleton pattern to ensure a class has only one instance
and provide global access to that instance:
- Thread-safe singleton implementations
- Lazy initialization patterns
- Singleton with parameters
- Singleton destruction and cleanup
- Testing strategies for singletons

Learning Objectives:
- Understand when and why to use Singleton pattern
- Implement thread-safe singleton in Python
- Handle singleton lifecycle management
- Avoid common singleton anti-patterns
- Test singleton classes effectively
"""

import threading
import time
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod
import weakref
import atexit


# ============================================================================
# BASIC SINGLETON IMPLEMENTATIONS
# ============================================================================

class BasicSingleton:
    """
    Basic singleton implementation using __new__ method.
    Not thread-safe - for educational purposes only.
    """
    
    _instance: Optional['BasicSingleton'] = None
    
    def __new__(cls) -> 'BasicSingleton':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.value = 0
        print(f"BasicSingleton instance created: {id(self)}")
    
    def increment(self) -> int:
        """Increment and return the value."""
        self.value += 1
        return self.value
    
    def get_value(self) -> int:
        """Get current value."""
        return self.value


class ThreadSafeSingleton:
    """
    Thread-safe singleton implementation using double-checked locking.
    """
    
    _instance: Optional['ThreadSafeSingleton'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ThreadSafeSingleton':
        # First check without locking for performance
        if cls._instance is None:
            with cls._lock:
                # Double-check inside the lock
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.value = 0
        self.creation_time = time.time()
        print(f"ThreadSafeSingleton instance created: {id(self)}")
    
    def increment(self) -> int:
        """Thread-safe increment operation."""
        with self._lock:
            self.value += 1
            return self.value
    
    def get_value(self) -> int:
        """Get current value."""
        return self.value
    
    def get_creation_time(self) -> float:
        """Get instance creation time."""
        return self.creation_time


# ============================================================================
# METACLASS-BASED SINGLETON
# ============================================================================

class SingletonMeta(type):
    """
    Metaclass that creates singleton instances.
    Thread-safe and supports inheritance.
    """
    
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """
    Database connection singleton using metaclass.
    """
    
    def __init__(self, connection_string: str = "default_db"):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.connection_string = connection_string
        self.is_connected = False
        self.query_count = 0
        print(f"DatabaseConnection created: {connection_string}")
    
    def connect(self) -> bool:
        """Establish database connection."""
        if not self.is_connected:
            print(f"Connecting to database: {self.connection_string}")
            self.is_connected = True
        return self.is_connected
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.is_connected:
            print(f"Disconnecting from database: {self.connection_string}")
            self.is_connected = False
    
    def execute_query(self, query: str) -> str:
        """Execute a database query."""
        if not self.is_connected:
            self.connect()
        
        self.query_count += 1
        result = f"Query executed: {query} (#{self.query_count})"
        print(result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connection_string': self.connection_string,
            'is_connected': self.is_connected,
            'query_count': self.query_count
        }


class Logger(metaclass=SingletonMeta):
    """
    Logger singleton for application-wide logging.
    """
    
    def __init__(self, log_file: str = "app.log"):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.log_file = log_file
        self.logs = []
        self._lock = threading.Lock()
        print(f"Logger initialized: {log_file}")
    
    def log(self, level: str, message: str) -> None:
        """Log a message with timestamp."""
        with self._lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {level}: {message}"
            self.logs.append(log_entry)
            print(log_entry)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.log("INFO", message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.log("WARNING", message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.log("ERROR", message)
    
    def get_logs(self) -> list:
        """Get all log entries."""
        return self.logs.copy()
    
    def clear_logs(self) -> None:
        """Clear all log entries."""
        with self._lock:
            self.logs.clear()


# ============================================================================
# DECORATOR-BASED SINGLETON
# ============================================================================

def singleton(cls):
    """
    Decorator that converts a class to singleton.
    """
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


@singleton
class ConfigurationManager:
    """
    Configuration manager singleton using decorator.
    """
    
    def __init__(self):
        self.config = {}
        self.config_file = "config.json"
        self.load_default_config()
        print("ConfigurationManager initialized")
    
    def load_default_config(self) -> None:
        """Load default configuration."""
        self.config = {
            'app_name': 'MyApplication',
            'version': '1.0.0',
            'debug': False,
            'max_connections': 100,
            'timeout': 30
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        print(f"Config updated: {key} = {value}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self.config.copy()
    
    def load_from_file(self, filename: str) -> bool:
        """Load configuration from file (simulated)."""
        print(f"Loading configuration from {filename}")
        # In real implementation, would read from actual file
        return True
    
    def save_to_file(self, filename: str = None) -> bool:
        """Save configuration to file (simulated)."""
        file_to_save = filename or self.config_file
        print(f"Saving configuration to {file_to_save}")
        # In real implementation, would write to actual file
        return True


# ============================================================================
# LAZY INITIALIZATION SINGLETON
# ============================================================================

class LazySingleton:
    """
    Lazy initialization singleton - instance created only when first accessed.
    """
    
    _instance: Optional['LazySingleton'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        raise RuntimeError("Use get_instance() to create singleton instance")
    
    @classmethod
    def get_instance(cls) -> 'LazySingleton':
        """Get singleton instance with lazy initialization."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Bypass __init__ restriction
                    cls._instance = cls.__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the singleton instance."""
        self.data = {}
        self.created_at = time.time()
        self.access_count = 0
        print(f"LazySingleton initialized at {self.created_at}")
    
    def add_data(self, key: str, value: Any) -> None:
        """Add data to singleton."""
        self.data[key] = value
        self.access_count += 1
    
    def get_data(self, key: str) -> Any:
        """Get data from singleton."""
        self.access_count += 1
        return self.data.get(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get singleton statistics."""
        return {
            'created_at': self.created_at,
            'access_count': self.access_count,
            'data_count': len(self.data)
        }


# ============================================================================
# SINGLETON WITH PARAMETERS
# ============================================================================

class ParameterizedSingleton:
    """
    Singleton that accepts parameters but maintains single instance.
    First call parameters are used for initialization.
    """
    
    _instance: Optional['ParameterizedSingleton'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, name: str = "default", max_size: int = 100):
        if self._initialized:
            print(f"Singleton already initialized. Ignoring parameters: name={name}, max_size={max_size}")
            return
        
        self.name = name
        self.max_size = max_size
        self.items = []
        self._initialized = True
        print(f"ParameterizedSingleton initialized: name={name}, max_size={max_size}")
    
    def add_item(self, item: Any) -> bool:
        """Add item to singleton."""
        if len(self.items) >= self.max_size:
            print(f"Cannot add item: maximum size ({self.max_size}) reached")
            return False
        
        self.items.append(item)
        print(f"Item added: {item} (total: {len(self.items)})")
        return True
    
    def get_items(self) -> list:
        """Get all items."""
        return self.items.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """Get singleton information."""
        return {
            'name': self.name,
            'max_size': self.max_size,
            'current_size': len(self.items),
            'initialized': self._initialized
        }


# ============================================================================
# SINGLETON WITH CLEANUP
# ============================================================================

class ManagedSingleton:
    """
    Singleton with proper cleanup and resource management.
    """
    
    _instance: Optional['ManagedSingleton'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Register cleanup function
                    atexit.register(cls._instance.cleanup)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.resources = []
        self.is_active = True
        print("ManagedSingleton initialized with cleanup registration")
    
    def acquire_resource(self, resource_name: str) -> str:
        """Acquire a resource."""
        if not self.is_active:
            raise RuntimeError("Singleton has been cleaned up")
        
        resource_id = f"{resource_name}_{len(self.resources)}"
        self.resources.append(resource_id)
        print(f"Resource acquired: {resource_id}")
        return resource_id
    
    def release_resource(self, resource_id: str) -> bool:
        """Release a specific resource."""
        if resource_id in self.resources:
            self.resources.remove(resource_id)
            print(f"Resource released: {resource_id}")
            return True
        return False
    
    def get_resource_count(self) -> int:
        """Get number of active resources."""
        return len(self.resources)
    
    def cleanup(self) -> None:
        """Cleanup all resources."""
        if not self.is_active:
            return
        
        print("Cleaning up ManagedSingleton...")
        for resource in self.resources.copy():
            self.release_resource(resource)
        
        self.is_active = False
        print("ManagedSingleton cleanup completed")
    
    def __del__(self):
        """Destructor - ensure cleanup is called."""
        self.cleanup()


# ============================================================================
# SINGLETON REGISTRY PATTERN
# ============================================================================

class SingletonRegistry:
    """
    Registry to manage multiple singleton instances by name.
    """
    
    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, name: str, instance_class: type, *args, **kwargs) -> Any:
        """Get or create named singleton instance."""
        if name not in cls._instances:
            with cls._lock:
                if name not in cls._instances:
                    cls._instances[name] = instance_class(*args, **kwargs)
                    print(f"Created singleton instance '{name}' of type {instance_class.__name__}")
        return cls._instances[name]
    
    @classmethod
    def remove_instance(cls, name: str) -> bool:
        """Remove named singleton instance."""
        if name in cls._instances:
            with cls._lock:
                if name in cls._instances:
                    instance = cls._instances.pop(name)
                    # Call cleanup if available
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                    print(f"Removed singleton instance '{name}'")
                    return True
        return False
    
    @classmethod
    def list_instances(cls) -> list:
        """List all registered singleton names."""
        return list(cls._instances.keys())
    
    @classmethod
    def clear_all(cls) -> None:
        """Clear all singleton instances."""
        with cls._lock:
            for name, instance in cls._instances.items():
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
            cls._instances.clear()
            print("All singleton instances cleared")


class CacheManager:
    """
    Cache manager that can be used with SingletonRegistry.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
        print(f"CacheManager initialized with max_size={max_size}")
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])
            self.cache.pop(least_accessed[0])
            self.access_count.pop(least_accessed[0])
        
        self.cache[key] = value
        self.access_count[key] = 0
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def cleanup(self) -> None:
        """Cleanup cache resources."""
        self.clear()
        print("CacheManager cleaned up")


# ============================================================================
# TESTING UTILITIES FOR SINGLETONS
# ============================================================================

class SingletonTestHelper:
    """
    Helper class for testing singleton implementations.
    """
    
    @staticmethod
    def reset_singleton(singleton_class: type) -> None:
        """Reset singleton instance for testing."""
        if hasattr(singleton_class, '_instance'):
            singleton_class._instance = None
        if hasattr(singleton_class, '_instances'):
            singleton_class._instances.clear()
    
    @staticmethod
    def test_singleton_behavior(singleton_class: type, *args, **kwargs) -> Dict[str, Any]:
        """Test basic singleton behavior."""
        # Create two instances
        instance1 = singleton_class(*args, **kwargs)
        instance2 = singleton_class(*args, **kwargs)
        
        # Test identity
        same_instance = instance1 is instance2
        same_id = id(instance1) == id(instance2)
        
        return {
            'same_instance': same_instance,
            'same_id': same_id,
            'instance1_id': id(instance1),
            'instance2_id': id(instance2),
            'test_passed': same_instance and same_id
        }
    
    @staticmethod
    def test_thread_safety(singleton_class: type, num_threads: int = 10) -> Dict[str, Any]:
        """Test thread safety of singleton."""
        instances = []
        threads = []
        
        def create_instance():
            instance = singleton_class()
            instances.append(instance)
        
        # Create multiple threads
        for _ in range(num_threads):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check if all instances are the same
        unique_ids = set(id(instance) for instance in instances)
        
        return {
            'total_instances': len(instances),
            'unique_instances': len(unique_ids),
            'all_same': len(unique_ids) == 1,
            'thread_safe': len(unique_ids) == 1
        }


def demonstrate_singleton_patterns():
    """
    Demonstrate various singleton pattern implementations.
    """
    print("=== SINGLETON PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Singleton
    print("1. BASIC SINGLETON:")
    basic1 = BasicSingleton()
    basic2 = BasicSingleton()
    print(f"   Same instance: {basic1 is basic2}")
    print(f"   Instance IDs: {id(basic1)} == {id(basic2)}")
    print(f"   Value after increment: {basic1.increment()}")
    print(f"   Value from second reference: {basic2.get_value()}")
    print()
    
    # 2. Thread-Safe Singleton
    print("2. THREAD-SAFE SINGLETON:")
    
    def create_thread_safe_singleton():
        return ThreadSafeSingleton()
    
    # Test thread safety
    threads = []
    instances = []
    
    def thread_worker():
        instance = create_thread_safe_singleton()
        instances.append(instance)
        instance.increment()
    
    for _ in range(5):
        thread = threading.Thread(target=thread_worker)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print(f"   Created {len(instances)} instances from {len(threads)} threads")
    print(f"   All same instance: {all(inst is instances[0] for inst in instances)}")
    print(f"   Final value: {instances[0].get_value()}")
    print()
    
    # 3. Metaclass Singleton
    print("3. METACLASS SINGLETON:")
    db1 = DatabaseConnection("postgresql://localhost/db1")
    db2 = DatabaseConnection("mysql://localhost/db2")  # Parameters ignored
    
    print(f"   Same instance: {db1 is db2}")
    db1.execute_query("SELECT * FROM users")
    db2.execute_query("SELECT * FROM products")
    print(f"   Stats: {db1.get_stats()}")
    print()
    
    # 4. Logger Singleton
    print("4. LOGGER SINGLETON:")
    logger1 = Logger("app.log")
    logger2 = Logger("different.log")  # Parameters ignored
    
    logger1.info("Application started")
    logger2.warning("This is a warning")
    logger1.error("An error occurred")
    
    print(f"   Same logger instance: {logger1 is logger2}")
    print(f"   Total logs: {len(logger1.get_logs())}")
    print()
    
    # 5. Decorator Singleton
    print("5. DECORATOR SINGLETON:")
    config1 = ConfigurationManager()
    config2 = ConfigurationManager()
    
    config1.set("debug", True)
    config1.set("max_connections", 200)
    
    print(f"   Same instance: {config1 is config2}")
    print(f"   Debug setting from config2: {config2.get('debug')}")
    print(f"   All config: {config2.get_all_config()}")
    print()
    
    # 6. Lazy Singleton
    print("6. LAZY SINGLETON:")
    print("   Before getting instance...")
    lazy1 = LazySingleton.get_instance()
    print("   After getting first instance")
    lazy2 = LazySingleton.get_instance()
    
    lazy1.add_data("key1", "value1")
    lazy2.add_data("key2", "value2")
    
    print(f"   Same instance: {lazy1 is lazy2}")
    print(f"   Data from lazy2: {lazy2.get_data('key1')}")
    print(f"   Stats: {lazy1.get_stats()}")
    print()
    
    # 7. Parameterized Singleton
    print("7. PARAMETERIZED SINGLETON:")
    param1 = ParameterizedSingleton("cache", 50)
    param2 = ParameterizedSingleton("different", 100)  # Parameters ignored
    
    param1.add_item("item1")
    param2.add_item("item2")
    
    print(f"   Same instance: {param1 is param2}")
    print(f"   Info: {param1.get_info()}")
    print(f"   Items: {param1.get_items()}")
    print()
    
    # 8. Managed Singleton
    print("8. MANAGED SINGLETON:")
    managed = ManagedSingleton()
    
    resource1 = managed.acquire_resource("database")
    resource2 = managed.acquire_resource("file_handle")
    
    print(f"   Active resources: {managed.get_resource_count()}")
    managed.release_resource(resource1)
    print(f"   After releasing one: {managed.get_resource_count()}")
    print()
    
    # 9. Singleton Registry
    print("9. SINGLETON REGISTRY:")
    
    # Create different cache instances
    cache1 = SingletonRegistry.get_instance("user_cache", CacheManager, 100)
    cache2 = SingletonRegistry.get_instance("product_cache", CacheManager, 200)
    cache3 = SingletonRegistry.get_instance("user_cache", CacheManager, 300)  # Same as cache1
    
    cache1.put("user1", {"name": "John", "age": 30})
    cache2.put("product1", {"name": "Laptop", "price": 999})
    
    print(f"   cache1 is cache3: {cache1 is cache3}")
    print(f"   cache1 is cache2: {cache1 is cache2}")
    print(f"   Registered instances: {SingletonRegistry.list_instances()}")
    print(f"   User cache size: {cache1.size()}")
    print(f"   Product cache size: {cache2.size()}")
    print()
    
    # 10. Testing Utilities
    print("10. SINGLETON TESTING:")
    
    # Reset and test basic singleton
    SingletonTestHelper.reset_singleton(BasicSingleton)
    test_result = SingletonTestHelper.test_singleton_behavior(BasicSingleton)
    print(f"   Basic singleton test: {test_result}")
    
    # Test thread safety
    SingletonTestHelper.reset_singleton(ThreadSafeSingleton)
    thread_test = SingletonTestHelper.test_thread_safety(ThreadSafeSingleton, 5)
    print(f"   Thread safety test: {thread_test}")
    print()
    
    # 11. Cleanup
    print("11. CLEANUP:")
    managed.cleanup()
    SingletonRegistry.clear_all()
    print("   All singletons cleaned up")
    print()
    
    print("=== SINGLETON PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_singleton_patterns()
