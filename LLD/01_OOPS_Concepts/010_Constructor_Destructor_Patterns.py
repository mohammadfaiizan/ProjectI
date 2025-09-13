"""
CONSTRUCTOR DESTRUCTOR PATTERNS - Object Lifecycle Management
=============================================================

Problem Statement:
Demonstrate constructor and destructor patterns:
- Different types of constructors (__init__, __new__)
- Constructor overloading and factory methods
- Destructor (__del__) and cleanup patterns
- Context managers for resource management
- Object lifecycle best practices

Learning Objectives:
- Master constructor patterns and initialization
- Understand destructor behavior and limitations
- Implement proper resource cleanup
- Use context managers effectively
- Design robust object lifecycles
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os
import tempfile
import threading
import weakref
import atexit


class DatabaseConnection:
    """
    Database connection class demonstrating constructor/destructor patterns.
    """
    
    # Class-level connection tracking
    _active_connections: List['DatabaseConnection'] = []
    _connection_count = 0
    
    def __init__(self, host: str, database: str, username: str, password: str, port: int = 5432):
        """
        Standard constructor for database connection.
        """
        self.host = host
        self.database = database
        self.username = username
        self.port = port
        self.connection_id = DatabaseConnection._connection_count
        DatabaseConnection._connection_count += 1
        
        # Private attributes
        self._password = password
        self._is_connected = False
        self._connection_time = None
        self._transaction_count = 0
        
        # Add to active connections
        DatabaseConnection._active_connections.append(self)
        
        print(f"DatabaseConnection {self.connection_id} initialized")
    
    def connect(self) -> bool:
        """Connect to database."""
        if not self._is_connected:
            print(f"Connecting to {self.host}:{self.port}/{self.database}")
            self._is_connected = True
            self._connection_time = datetime.now()
            return True
        return False
    
    def disconnect(self) -> bool:
        """Disconnect from database."""
        if self._is_connected:
            print(f"Disconnecting from {self.host}:{self.port}/{self.database}")
            self._is_connected = False
            return True
        return False
    
    def execute_query(self, query: str) -> str:
        """Execute database query."""
        if self._is_connected:
            self._transaction_count += 1
            return f"Executed: {query} (Transaction #{self._transaction_count})"
        return "Error: Not connected"
    
    def __del__(self):
        """
        Destructor - called when object is garbage collected.
        Note: __del__ is not guaranteed to be called immediately.
        """
        print(f"DatabaseConnection {self.connection_id} destructor called")
        
        # Cleanup: disconnect if still connected
        if self._is_connected:
            print(f"Warning: Connection {self.connection_id} was not properly closed")
            self.disconnect()
        
        # Remove from active connections
        try:
            DatabaseConnection._active_connections.remove(self)
        except ValueError:
            pass  # Already removed
    
    @classmethod
    def create_local_connection(cls, database: str) -> 'DatabaseConnection':
        """Factory method for local database connection."""
        return cls("localhost", database, "local_user", "local_pass", 5432)
    
    @classmethod
    def create_remote_connection(cls, host: str, database: str, credentials: Dict[str, str]) -> 'DatabaseConnection':
        """Factory method for remote database connection."""
        return cls(host, database, credentials['username'], credentials['password'], 
                  credentials.get('port', 5432))
    
    @classmethod
    def get_active_connections(cls) -> List['DatabaseConnection']:
        """Get list of active connections."""
        return cls._active_connections.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.disconnect()
        return False  # Don't suppress exceptions
    
    def __str__(self) -> str:
        status = "connected" if self._is_connected else "disconnected"
        return f"DatabaseConnection({self.connection_id}, {self.host}:{self.port}, {status})"


class FileManager:
    """
    File manager demonstrating resource management patterns.
    """
    
    def __init__(self, filename: str, mode: str = 'r', auto_cleanup: bool = True):
        """
        Initialize file manager.
        
        Args:
            filename: Name of file to manage
            mode: File open mode
            auto_cleanup: Whether to auto-cleanup on destruction
        """
        self.filename = filename
        self.mode = mode
        self.auto_cleanup = auto_cleanup
        self._file_handle = None
        self._is_open = False
        self._created_temp_file = False
        
        print(f"FileManager initialized for {filename}")
    
    def open(self) -> bool:
        """Open the file."""
        try:
            # Create temp file if filename doesn't exist and mode is write
            if not os.path.exists(self.filename) and 'w' in self.mode:
                # Create temporary file
                temp_dir = tempfile.gettempdir()
                self.filename = os.path.join(temp_dir, os.path.basename(self.filename))
                self._created_temp_file = True
            
            self._file_handle = open(self.filename, self.mode)
            self._is_open = True
            print(f"File {self.filename} opened in mode {self.mode}")
            return True
        except IOError as e:
            print(f"Error opening file {self.filename}: {e}")
            return False
    
    def write(self, content: str) -> bool:
        """Write content to file."""
        if self._is_open and self._file_handle:
            try:
                self._file_handle.write(content)
                self._file_handle.flush()
                return True
            except IOError as e:
                print(f"Error writing to file: {e}")
                return False
        return False
    
    def read(self) -> Optional[str]:
        """Read content from file."""
        if self._is_open and self._file_handle:
            try:
                return self._file_handle.read()
            except IOError as e:
                print(f"Error reading from file: {e}")
                return None
        return None
    
    def close(self) -> bool:
        """Close the file."""
        if self._is_open and self._file_handle:
            self._file_handle.close()
            self._is_open = False
            print(f"File {self.filename} closed")
            return True
        return False
    
    def cleanup(self) -> None:
        """Cleanup resources and temporary files."""
        self.close()
        
        if self._created_temp_file and os.path.exists(self.filename):
            try:
                os.remove(self.filename)
                print(f"Temporary file {self.filename} removed")
            except OSError as e:
                print(f"Error removing temporary file: {e}")
    
    def __del__(self):
        """Destructor with optional auto-cleanup."""
        print(f"FileManager destructor called for {self.filename}")
        
        if self.auto_cleanup:
            self.cleanup()
        elif self._is_open:
            print(f"Warning: File {self.filename} was not properly closed")
            self.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


class Singleton:
    """
    Singleton pattern using __new__ constructor.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to implement singleton pattern.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("Creating new Singleton instance")
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, name: str = "default"):
        """
        Initialize singleton (called every time, but should only initialize once).
        """
        if not self._initialized:
            self.name = name
            self.created_at = datetime.now()
            self._initialized = True
            print(f"Singleton initialized with name: {name}")
        else:
            print(f"Singleton already initialized (current name: {self.name})")
    
    def get_info(self) -> Dict[str, Any]:
        """Get singleton information."""
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'instance_id': id(self)
        }
    
    @classmethod
    def reset(cls):
        """Reset singleton for testing purposes."""
        cls._instance = None


class ResourcePool:
    """
    Resource pool demonstrating advanced constructor patterns.
    """
    
    def __init__(self, pool_size: int = 5, resource_type: str = "generic"):
        """
        Initialize resource pool.
        """
        self.pool_size = pool_size
        self.resource_type = resource_type
        self._available_resources = []
        self._used_resources = set()
        self._created_count = 0
        
        # Pre-create resources
        self._initialize_pool()
        
        # Register cleanup at program exit
        atexit.register(self._cleanup_all_resources)
        
        print(f"ResourcePool initialized with {pool_size} {resource_type} resources")
    
    def _initialize_pool(self) -> None:
        """Initialize the resource pool."""
        for i in range(self.pool_size):
            resource = self._create_resource(i)
            self._available_resources.append(resource)
    
    def _create_resource(self, resource_id: int) -> Dict[str, Any]:
        """Create a new resource."""
        self._created_count += 1
        return {
            'id': resource_id,
            'type': self.resource_type,
            'created_at': datetime.now(),
            'in_use': False
        }
    
    def acquire_resource(self) -> Optional[Dict[str, Any]]:
        """Acquire a resource from the pool."""
        if self._available_resources:
            resource = self._available_resources.pop()
            resource['in_use'] = True
            resource['acquired_at'] = datetime.now()
            self._used_resources.add(resource['id'])
            print(f"Acquired resource {resource['id']}")
            return resource
        else:
            print("No resources available in pool")
            return None
    
    def release_resource(self, resource: Dict[str, Any]) -> bool:
        """Release a resource back to the pool."""
        if resource['id'] in self._used_resources:
            resource['in_use'] = False
            resource['released_at'] = datetime.now()
            self._used_resources.remove(resource['id'])
            self._available_resources.append(resource)
            print(f"Released resource {resource['id']}")
            return True
        else:
            print(f"Resource {resource['id']} was not acquired from this pool")
            return False
    
    def _cleanup_all_resources(self) -> None:
        """Cleanup all resources."""
        print(f"Cleaning up ResourcePool with {self._created_count} resources")
        total_resources = len(self._available_resources) + len(self._used_resources)
        print(f"Available: {len(self._available_resources)}, In use: {len(self._used_resources)}")
        
        # Force release all used resources
        for resource_id in list(self._used_resources):
            print(f"Force releasing resource {resource_id}")
        
        self._available_resources.clear()
        self._used_resources.clear()
    
    def __del__(self):
        """Destructor."""
        print("ResourcePool destructor called")
        self._cleanup_all_resources()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status."""
        return {
            'pool_size': self.pool_size,
            'resource_type': self.resource_type,
            'available': len(self._available_resources),
            'in_use': len(self._used_resources),
            'total_created': self._created_count
        }


class WeakReferenceManager:
    """
    Demonstrates weak references to avoid circular references.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._observers = weakref.WeakSet()
        self._callbacks = []
        
        print(f"WeakReferenceManager {name} created")
    
    def add_observer(self, observer) -> None:
        """Add observer using weak reference."""
        self._observers.add(observer)
        print(f"Observer added to {self.name}")
    
    def add_callback(self, callback) -> None:
        """Add callback with weak reference."""
        weak_callback = weakref.WeakMethod(callback) if hasattr(callback, '__self__') else weakref.ref(callback)
        self._callbacks.append(weak_callback)
        print(f"Callback added to {self.name}")
    
    def notify_observers(self, message: str) -> None:
        """Notify all observers."""
        print(f"Notifying observers of {self.name}: {message}")
        
        # Clean up dead weak references
        alive_observers = []
        for observer in self._observers:
            if observer is not None:
                alive_observers.append(observer)
                if hasattr(observer, 'notify'):
                    observer.notify(f"{self.name}: {message}")
        
        print(f"Notified {len(alive_observers)} observers")
    
    def execute_callbacks(self, data: Any) -> None:
        """Execute all callbacks."""
        alive_callbacks = []
        for weak_callback in self._callbacks:
            callback = weak_callback()
            if callback is not None:
                alive_callbacks.append(callback)
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error executing callback: {e}")
        
        # Update callbacks list to remove dead references
        self._callbacks = [wc for wc in self._callbacks if wc() is not None]
        print(f"Executed {len(alive_callbacks)} callbacks")
    
    def __del__(self):
        """Destructor."""
        print(f"WeakReferenceManager {self.name} destructor called")


class Observer:
    """Observer class for weak reference demonstration."""
    
    def __init__(self, name: str):
        self.name = name
    
    def notify(self, message: str) -> None:
        """Receive notification."""
        print(f"Observer {self.name} received: {message}")
    
    def callback_method(self, data: Any) -> None:
        """Callback method."""
        print(f"Observer {self.name} callback received: {data}")
    
    def __del__(self):
        """Destructor."""
        print(f"Observer {self.name} destructor called")


def demonstrate_constructor_destructor_patterns():
    """
    Demonstrate constructor and destructor patterns.
    """
    print("=== CONSTRUCTOR DESTRUCTOR PATTERNS DEMONSTRATION ===\n")
    
    # 1. Basic Constructor/Destructor
    print("1. Basic Constructor/Destructor:")
    
    # Create database connection
    db_conn = DatabaseConnection("localhost", "testdb", "user", "pass")
    db_conn.connect()
    db_conn.execute_query("SELECT * FROM users")
    
    # Show active connections
    active_conns = DatabaseConnection.get_active_connections()
    print(f"Active connections: {len(active_conns)}")
    
    # Delete connection (destructor will be called)
    del db_conn
    print("Database connection deleted\n")
    
    # 2. Factory Methods (Alternative Constructors)
    print("2. Factory Methods (Alternative Constructors):")
    
    # Create connections using factory methods
    local_db = DatabaseConnection.create_local_connection("local_db")
    remote_db = DatabaseConnection.create_remote_connection(
        "remote.server.com", 
        "remote_db", 
        {'username': 'remote_user', 'password': 'remote_pass', 'port': 3306}
    )
    
    print(f"Local DB: {local_db}")
    print(f"Remote DB: {remote_db}")
    
    # Cleanup
    del local_db, remote_db
    print()
    
    # 3. Context Managers (Guaranteed Cleanup)
    print("3. Context Managers (Guaranteed Cleanup):")
    
    # Using context manager ensures proper cleanup
    with DatabaseConnection("context.server.com", "context_db", "user", "pass") as db:
        db.execute_query("SELECT * FROM products")
        print("Database operations completed")
    # Connection automatically closed here
    
    print("Context manager demonstration completed\n")
    
    # 4. File Manager with Resource Cleanup
    print("4. File Manager with Resource Cleanup:")
    
    # Create file manager with auto-cleanup
    file_mgr = FileManager("test_file.txt", "w", auto_cleanup=True)
    file_mgr.open()
    file_mgr.write("Hello, World!\n")
    file_mgr.write("This is a test file.\n")
    
    # File will be cleaned up automatically when deleted
    del file_mgr
    
    # Using context manager for guaranteed cleanup
    with FileManager("context_file.txt", "w") as fm:
        fm.write("Context manager file content\n")
    # File automatically cleaned up here
    
    print()
    
    # 5. Singleton Pattern with __new__
    print("5. Singleton Pattern with __new__:")
    
    # Create multiple singleton instances
    singleton1 = Singleton("first")
    singleton2 = Singleton("second")  # Should reuse existing instance
    singleton3 = Singleton("third")   # Should reuse existing instance
    
    print(f"singleton1 is singleton2: {singleton1 is singleton2}")
    print(f"singleton2 is singleton3: {singleton2 is singleton3}")
    print(f"Singleton info: {singleton1.get_info()}")
    
    # Reset singleton
    Singleton.reset()
    singleton4 = Singleton("new_instance")
    print(f"After reset - singleton1 is singleton4: {singleton1 is singleton4}")
    print()
    
    # 6. Resource Pool Management
    print("6. Resource Pool Management:")
    
    pool = ResourcePool(3, "database_connection")
    print(f"Pool status: {pool.get_pool_status()}")
    
    # Acquire resources
    resource1 = pool.acquire_resource()
    resource2 = pool.acquire_resource()
    resource3 = pool.acquire_resource()
    resource4 = pool.acquire_resource()  # Should fail - pool exhausted
    
    print(f"Pool status after acquisitions: {pool.get_pool_status()}")
    
    # Release resources
    if resource1:
        pool.release_resource(resource1)
    if resource2:
        pool.release_resource(resource2)
    
    print(f"Pool status after releases: {pool.get_pool_status()}")
    
    # Pool will be cleaned up automatically
    del pool
    print()
    
    # 7. Weak References to Avoid Circular References
    print("7. Weak References to Avoid Circular References:")
    
    manager = WeakReferenceManager("EventManager")
    
    # Create observers
    observer1 = Observer("Observer1")
    observer2 = Observer("Observer2")
    
    # Add observers using weak references
    manager.add_observer(observer1)
    manager.add_observer(observer2)
    
    # Add callbacks using weak references
    manager.add_callback(observer1.callback_method)
    manager.add_callback(observer2.callback_method)
    
    # Notify observers
    manager.notify_observers("System startup")
    manager.execute_callbacks("Callback data")
    
    # Delete one observer
    del observer1
    print("Observer1 deleted")
    
    # Notify again - weak references automatically cleaned up
    manager.notify_observers("After observer deletion")
    manager.execute_callbacks("More callback data")
    
    # Cleanup
    del observer2, manager
    print()
    
    # 8. Constructor/Destructor Best Practices
    print("8. Constructor/Destructor Best Practices:")
    print("✓ Initialize all attributes in __init__")
    print("✓ Use factory methods for alternative constructors")
    print("✓ Implement __del__ for cleanup, but don't rely on it")
    print("✓ Use context managers for guaranteed resource cleanup")
    print("✓ Register cleanup functions with atexit for critical resources")
    print("✓ Use weak references to avoid circular references")
    print("✓ Handle exceptions in constructors gracefully")
    print("✓ Document resource management expectations")
    print()
    
    print("=== CONSTRUCTOR DESTRUCTOR PATTERNS DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_constructor_destructor_patterns()
