"""
PROXY PATTERN - Structural Design Pattern
=========================================

Problem Statement:
Implement the Proxy pattern to provide a placeholder or surrogate for another
object to control access to it:
- Virtual proxy for expensive object creation
- Protection proxy for access control
- Remote proxy for distributed objects
- Caching proxy for performance optimization
- Smart reference proxy for additional functionality

Learning Objectives:
- Understand different types of proxy patterns
- Implement lazy loading and virtual proxies
- Design access control and security proxies
- Create caching and performance proxies
- Handle remote object access through proxies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
import time
import threading
import weakref
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib


# ============================================================================
# SUBJECT INTERFACE
# ============================================================================

class ImageService(ABC):
    """Abstract interface for image operations."""
    
    @abstractmethod
    def load_image(self, image_path: str) -> Dict[str, Any]:
        """Load image from path."""
        pass
    
    @abstractmethod
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image metadata."""
        pass
    
    @abstractmethod
    def resize_image(self, image_path: str, width: int, height: int) -> bool:
        """Resize image to specified dimensions."""
        pass
    
    @abstractmethod
    def save_image(self, image_data: Dict[str, Any], output_path: str) -> bool:
        """Save image to path."""
        pass


class DatabaseService(ABC):
    """Abstract interface for database operations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from database."""
        pass
    
    @abstractmethod
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query and return results."""
        pass
    
    @abstractmethod
    def execute(self, sql: str) -> int:
        """Execute update/insert/delete and return affected rows."""
        pass
    
    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        pass


class FileService(ABC):
    """Abstract interface for file operations."""
    
    @abstractmethod
    def read_file(self, file_path: str) -> str:
        """Read file content."""
        pass
    
    @abstractmethod
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file."""
        pass
    
    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """Delete file."""
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        pass


# ============================================================================
# REAL SUBJECTS (EXPENSIVE/SENSITIVE OPERATIONS)
# ============================================================================

class RealImageService(ImageService):
    """Real image service with expensive operations."""
    
    def __init__(self):
        self.loaded_images: Dict[str, Dict[str, Any]] = {}
        print("RealImageService: Initialized (expensive operation)")
    
    def load_image(self, image_path: str) -> Dict[str, Any]:
        """Load image from disk (expensive operation)."""
        print(f"RealImageService: Loading image from {image_path} (expensive I/O)")
        
        # Simulate expensive image loading
        time.sleep(0.5)  # Simulate disk I/O
        
        # Create mock image data
        image_data = {
            'path': image_path,
            'width': 1920,
            'height': 1080,
            'format': 'JPEG',
            'size_bytes': 2048000,
            'data': f"BINARY_IMAGE_DATA_FOR_{image_path}",
            'loaded_at': datetime.now().isoformat()
        }
        
        self.loaded_images[image_path] = image_data
        print(f"RealImageService: Image loaded successfully")
        return image_data
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image metadata (less expensive)."""
        print(f"RealImageService: Getting info for {image_path}")
        
        if image_path in self.loaded_images:
            return self.loaded_images[image_path]
        
        # Simulate reading just metadata (faster than full load)
        time.sleep(0.1)
        
        return {
            'path': image_path,
            'width': 1920,
            'height': 1080,
            'format': 'JPEG',
            'size_bytes': 2048000,
            'loaded': False
        }
    
    def resize_image(self, image_path: str, width: int, height: int) -> bool:
        """Resize image (expensive operation)."""
        print(f"RealImageService: Resizing {image_path} to {width}x{height}")
        
        # Ensure image is loaded
        if image_path not in self.loaded_images:
            self.load_image(image_path)
        
        # Simulate expensive resize operation
        time.sleep(0.3)
        
        # Update image data
        self.loaded_images[image_path]['width'] = width
        self.loaded_images[image_path]['height'] = height
        self.loaded_images[image_path]['modified_at'] = datetime.now().isoformat()
        
        print(f"RealImageService: Image resized successfully")
        return True
    
    def save_image(self, image_data: Dict[str, Any], output_path: str) -> bool:
        """Save image to disk (expensive operation)."""
        print(f"RealImageService: Saving image to {output_path}")
        
        # Simulate expensive save operation
        time.sleep(0.4)
        
        print(f"RealImageService: Image saved successfully")
        return True


class RealDatabaseService(DatabaseService):
    """Real database service with actual database operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False
        self.connection_count = 0
        self.query_log = []
        print(f"RealDatabaseService: Created for {connection_string}")
    
    def connect(self) -> bool:
        """Connect to database (expensive operation)."""
        if self.is_connected:
            return True
        
        print(f"RealDatabaseService: Connecting to database...")
        
        # Simulate expensive connection establishment
        time.sleep(1.0)
        
        self.is_connected = True
        self.connection_count += 1
        print(f"RealDatabaseService: Connected successfully")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        if self.is_connected:
            print(f"RealDatabaseService: Disconnecting from database")
            self.is_connected = False
    
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query (potentially expensive)."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        print(f"RealDatabaseService: Executing query: {sql[:50]}...")
        
        # Log query
        self.query_log.append({
            'sql': sql,
            'timestamp': datetime.now().isoformat(),
            'type': 'SELECT'
        })
        
        # Simulate query execution time
        time.sleep(0.2)
        
        # Return mock results
        if 'users' in sql.lower():
            return [
                {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
                {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com'}
            ]
        elif 'products' in sql.lower():
            return [
                {'id': 1, 'name': 'Laptop', 'price': 999.99},
                {'id': 2, 'name': 'Mouse', 'price': 29.99}
            ]
        else:
            return [{'result': 'mock_data'}]
    
    def execute(self, sql: str) -> int:
        """Execute update/insert/delete."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        print(f"RealDatabaseService: Executing update: {sql[:50]}...")
        
        # Log query
        self.query_log.append({
            'sql': sql,
            'timestamp': datetime.now().isoformat(),
            'type': 'UPDATE'
        })
        
        # Simulate execution time
        time.sleep(0.1)
        
        return 1  # Mock affected rows
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            'connection_string': self.connection_string,
            'is_connected': self.is_connected,
            'connection_count': self.connection_count,
            'queries_executed': len(self.query_log)
        }


class RealFileService(FileService):
    """Real file service with actual file operations."""
    
    def __init__(self):
        self.file_cache: Dict[str, str] = {}
        print("RealFileService: Initialized")
    
    def read_file(self, file_path: str) -> str:
        """Read file from disk (I/O operation)."""
        print(f"RealFileService: Reading file {file_path}")
        
        # Simulate file I/O
        time.sleep(0.1)
        
        # Mock file content
        content = f"File content from {file_path}\nTimestamp: {datetime.now()}\nLine 3\nLine 4"
        self.file_cache[file_path] = content
        
        print(f"RealFileService: File read successfully ({len(content)} bytes)")
        return content
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file (I/O operation)."""
        print(f"RealFileService: Writing to file {file_path} ({len(content)} bytes)")
        
        # Simulate file I/O
        time.sleep(0.05)
        
        self.file_cache[file_path] = content
        print(f"RealFileService: File written successfully")
        return True
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file (I/O operation)."""
        print(f"RealFileService: Deleting file {file_path}")
        
        # Simulate file I/O
        time.sleep(0.02)
        
        if file_path in self.file_cache:
            del self.file_cache[file_path]
        
        print(f"RealFileService: File deleted successfully")
        return True
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        print(f"RealFileService: Getting info for {file_path}")
        
        content = self.file_cache.get(file_path, "")
        
        return {
            'path': file_path,
            'size_bytes': len(content),
            'exists': file_path in self.file_cache,
            'last_modified': datetime.now().isoformat(),
            'readable': True,
            'writable': True
        }


# ============================================================================
# VIRTUAL PROXY (LAZY LOADING)
# ============================================================================

class VirtualImageProxy(ImageService):
    """Virtual proxy for lazy loading of images."""
    
    def __init__(self):
        self._real_service: Optional[RealImageService] = None
        self._image_cache: Dict[str, Dict[str, Any]] = {}
        print("VirtualImageProxy: Created (lightweight)")
    
    def _get_real_service(self) -> RealImageService:
        """Get real service instance (lazy initialization)."""
        if self._real_service is None:
            print("VirtualImageProxy: Creating real service (lazy loading)")
            self._real_service = RealImageService()
        return self._real_service
    
    def load_image(self, image_path: str) -> Dict[str, Any]:
        """Load image through proxy with caching."""
        # Check cache first
        if image_path in self._image_cache:
            print(f"VirtualImageProxy: Returning cached image for {image_path}")
            return self._image_cache[image_path]
        
        # Load through real service
        real_service = self._get_real_service()
        image_data = real_service.load_image(image_path)
        
        # Cache the result
        self._image_cache[image_path] = image_data
        
        return image_data
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image info (can be served without loading full image)."""
        # Check if we have cached full image data
        if image_path in self._image_cache:
            print(f"VirtualImageProxy: Returning cached info for {image_path}")
            return self._image_cache[image_path]
        
        # Get info from real service (doesn't require full load)
        real_service = self._get_real_service()
        return real_service.get_image_info(image_path)
    
    def resize_image(self, image_path: str, width: int, height: int) -> bool:
        """Resize image through proxy."""
        real_service = self._get_real_service()
        result = real_service.resize_image(image_path, width, height)
        
        # Update cache if image was cached
        if image_path in self._image_cache:
            self._image_cache[image_path]['width'] = width
            self._image_cache[image_path]['height'] = height
            self._image_cache[image_path]['modified_at'] = datetime.now().isoformat()
        
        return result
    
    def save_image(self, image_data: Dict[str, Any], output_path: str) -> bool:
        """Save image through proxy."""
        real_service = self._get_real_service()
        return real_service.save_image(image_data, output_path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get proxy cache statistics."""
        return {
            'cached_images': len(self._image_cache),
            'real_service_created': self._real_service is not None,
            'cache_keys': list(self._image_cache.keys())
        }


# ============================================================================
# PROTECTION PROXY (ACCESS CONTROL)
# ============================================================================

class UserRole(Enum):
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class User:
    """User class for authentication."""
    
    def __init__(self, username: str, role: UserRole, permissions: List[str] = None):
        self.username = username
        self.role = role
        self.permissions = permissions or []
        self.login_time = datetime.now()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]


class ProtectionDatabaseProxy(DatabaseService):
    """Protection proxy for database access control."""
    
    def __init__(self, real_service: RealDatabaseService):
        self._real_service = real_service
        self._current_user: Optional[User] = None
        self._access_log: List[Dict[str, Any]] = []
        self._failed_attempts: Dict[str, int] = {}
        
        # Define permission requirements
        self._permissions = {
            'connect': ['db_connect'],
            'query_users': ['read_users'],
            'query_products': ['read_products'],
            'update_users': ['write_users'],
            'update_products': ['write_products'],
            'admin_queries': ['admin_access']
        }
        
        print("ProtectionDatabaseProxy: Created with access control")
    
    def authenticate(self, user: User) -> bool:
        """Authenticate user for database access."""
        self._current_user = user
        self._log_access('authenticate', True, f"User {user.username} authenticated")
        print(f"ProtectionDatabaseProxy: User {user.username} authenticated")
        return True
    
    def logout(self) -> None:
        """Logout current user."""
        if self._current_user:
            self._log_access('logout', True, f"User {self._current_user.username} logged out")
            print(f"ProtectionDatabaseProxy: User {self._current_user.username} logged out")
            self._current_user = None
    
    def _check_permission(self, operation: str) -> bool:
        """Check if current user has permission for operation."""
        if not self._current_user:
            self._log_access(operation, False, "No authenticated user")
            return False
        
        # Super admin can do everything
        if self._current_user.role == UserRole.SUPER_ADMIN:
            return True
        
        # Check specific permissions
        required_perms = self._permissions.get(operation, [])
        for perm in required_perms:
            if not self._current_user.has_permission(perm):
                self._log_access(operation, False, f"Missing permission: {perm}")
                return False
        
        return True
    
    def _log_access(self, operation: str, success: bool, details: str) -> None:
        """Log access attempt."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': self._current_user.username if self._current_user else 'anonymous',
            'operation': operation,
            'success': success,
            'details': details
        }
        self._access_log.append(log_entry)
    
    def connect(self) -> bool:
        """Connect with permission check."""
        if not self._check_permission('connect'):
            print("ProtectionDatabaseProxy: Connection denied - insufficient permissions")
            return False
        
        result = self._real_service.connect()
        self._log_access('connect', result, "Database connection")
        return result
    
    def disconnect(self) -> None:
        """Disconnect (always allowed)."""
        self._real_service.disconnect()
        self._log_access('disconnect', True, "Database disconnection")
    
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query with permission check."""
        # Determine required permission based on query
        sql_lower = sql.lower()
        if 'users' in sql_lower:
            permission = 'query_users'
        elif 'products' in sql_lower:
            permission = 'query_products'
        elif any(keyword in sql_lower for keyword in ['drop', 'create', 'alter']):
            permission = 'admin_queries'
        else:
            permission = 'query_users'  # Default permission
        
        if not self._check_permission(permission):
            print(f"ProtectionDatabaseProxy: Query denied - insufficient permissions")
            raise PermissionError(f"Insufficient permissions for query: {sql[:50]}...")
        
        result = self._real_service.query(sql)
        self._log_access('query', True, f"Query executed: {sql[:50]}...")
        return result
    
    def execute(self, sql: str) -> int:
        """Execute update with permission check."""
        # Determine required permission based on query
        sql_lower = sql.lower()
        if 'users' in sql_lower:
            permission = 'update_users'
        elif 'products' in sql_lower:
            permission = 'update_products'
        else:
            permission = 'update_users'  # Default permission
        
        if not self._check_permission(permission):
            print(f"ProtectionDatabaseProxy: Update denied - insufficient permissions")
            raise PermissionError(f"Insufficient permissions for update: {sql[:50]}...")
        
        result = self._real_service.execute(sql)
        self._log_access('execute', True, f"Update executed: {sql[:50]}...")
        return result
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection info (admin only)."""
        if not self._check_permission('admin_queries'):
            print("ProtectionDatabaseProxy: Connection info denied - admin access required")
            return {'error': 'Insufficient permissions'}
        
        info = self._real_service.get_connection_info()
        self._log_access('get_connection_info', True, "Connection info retrieved")
        return info
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log (admin only)."""
        if not self._current_user or not self._current_user.is_admin():
            return []
        
        return self._access_log.copy()


# ============================================================================
# CACHING PROXY (PERFORMANCE OPTIMIZATION)
# ============================================================================

class CachingFileProxy(FileService):
    """Caching proxy for file operations."""
    
    def __init__(self, real_service: RealFileService, cache_ttl: int = 300):
        self._real_service = real_service
        self._cache_ttl = cache_ttl  # Time to live in seconds
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        print(f"CachingFileProxy: Created with {cache_ttl}s TTL")
    
    def _is_cache_valid(self, file_path: str) -> bool:
        """Check if cached entry is still valid."""
        if file_path not in self._file_cache:
            return False
        
        cache_entry = self._file_cache[file_path]
        age = time.time() - cache_entry['cached_at']
        return age < self._cache_ttl
    
    def _cache_file(self, file_path: str, content: str) -> None:
        """Cache file content."""
        self._file_cache[file_path] = {
            'content': content,
            'cached_at': time.time(),
            'access_count': 1
        }
    
    def _evict_expired_entries(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for file_path, cache_entry in self._file_cache.items():
            if current_time - cache_entry['cached_at'] >= self._cache_ttl:
                expired_keys.append(file_path)
        
        for key in expired_keys:
            del self._file_cache[key]
            self._cache_stats['evictions'] += 1
    
    def read_file(self, file_path: str) -> str:
        """Read file with caching."""
        # Clean up expired entries
        self._evict_expired_entries()
        
        # Check cache first
        if self._is_cache_valid(file_path):
            print(f"CachingFileProxy: Cache hit for {file_path}")
            self._cache_stats['hits'] += 1
            cache_entry = self._file_cache[file_path]
            cache_entry['access_count'] += 1
            return cache_entry['content']
        
        # Cache miss - read from real service
        print(f"CachingFileProxy: Cache miss for {file_path}")
        self._cache_stats['misses'] += 1
        
        content = self._real_service.read_file(file_path)
        self._cache_file(file_path, content)
        
        return content
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write file and update cache."""
        result = self._real_service.write_file(file_path, content)
        
        if result:
            # Update cache with new content
            self._cache_file(file_path, content)
            print(f"CachingFileProxy: Cache updated for {file_path}")
        
        return result
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file and remove from cache."""
        result = self._real_service.delete_file(file_path)
        
        if result and file_path in self._file_cache:
            del self._file_cache[file_path]
            print(f"CachingFileProxy: Cache entry removed for {file_path}")
        
        return result
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file info (not cached - metadata changes frequently)."""
        return self._real_service.get_file_info(file_path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_ratio = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._file_cache),
            'hit_ratio': round(hit_ratio, 2),
            'stats': self._cache_stats.copy(),
            'cached_files': list(self._file_cache.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear all cached entries."""
        cleared_count = len(self._file_cache)
        self._file_cache.clear()
        print(f"CachingFileProxy: Cleared {cleared_count} cache entries")


# ============================================================================
# REMOTE PROXY (DISTRIBUTED OBJECTS)
# ============================================================================

class RemoteServiceProxy:
    """Proxy for remote service calls."""
    
    def __init__(self, service_url: str, timeout: int = 30):
        self.service_url = service_url
        self.timeout = timeout
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        print(f"RemoteServiceProxy: Created for {service_url}")
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate remote HTTP request."""
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        print(f"RemoteServiceProxy: {method} {self.service_url}{endpoint}")
        
        # Simulate network latency
        time.sleep(0.1)
        
        # Simulate occasional network errors
        import random
        if random.random() < 0.05:  # 5% error rate
            self.error_count += 1
            raise ConnectionError(f"Network error connecting to {self.service_url}")
        
        # Simulate successful response
        return {
            'status': 'success',
            'data': f"Response from {endpoint}",
            'timestamp': datetime.now().isoformat(),
            'request_id': f"req_{self.request_count}"
        }
    
    def get_user(self, user_id: int) -> Dict[str, Any]:
        """Get user from remote service."""
        try:
            response = self._make_request(f"/users/{user_id}")
            return {
                'id': user_id,
                'name': f'User {user_id}',
                'email': f'user{user_id}@example.com',
                'remote_response': response
            }
        except ConnectionError as e:
            print(f"RemoteServiceProxy: Error getting user {user_id}: {e}")
            return {'error': str(e)}
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user on remote service."""
        try:
            response = self._make_request("/users", "POST", user_data)
            return {
                'created': True,
                'user_id': self.request_count,  # Mock ID
                'remote_response': response
            }
        except ConnectionError as e:
            print(f"RemoteServiceProxy: Error creating user: {e}")
            return {'error': str(e)}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get proxy statistics."""
        return {
            'service_url': self.service_url,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            'last_request': self.last_request_time.isoformat() if self.last_request_time else None
        }


# ============================================================================
# SMART REFERENCE PROXY
# ============================================================================

class SmartReferenceProxy:
    """Smart reference proxy that adds additional functionality."""
    
    def __init__(self, target_object: Any):
        self._target = target_object
        self._reference_count = 0
        self._access_log: List[Dict[str, Any]] = []
        self._created_at = datetime.now()
        print(f"SmartReferenceProxy: Created for {type(target_object).__name__}")
    
    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access."""
        self._reference_count += 1
        self._log_access(name, 'attribute_access')
        
        # Get attribute from target object
        attr = getattr(self._target, name)
        
        # If it's a method, wrap it
        if callable(attr):
            def wrapped_method(*args, **kwargs):
                self._log_access(name, 'method_call', {'args': len(args), 'kwargs': len(kwargs)})
                start_time = time.time()
                
                try:
                    result = attr(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self._log_access(name, 'method_success', {'execution_time': execution_time})
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._log_access(name, 'method_error', {'error': str(e), 'execution_time': execution_time})
                    raise
            
            return wrapped_method
        
        return attr
    
    def _log_access(self, attribute: str, access_type: str, details: Dict[str, Any] = None) -> None:
        """Log access to target object."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'attribute': attribute,
            'access_type': access_type,
            'details': details or {}
        }
        self._access_log.append(log_entry)
    
    def get_reference_stats(self) -> Dict[str, Any]:
        """Get reference statistics."""
        method_calls = [log for log in self._access_log if log['access_type'] == 'method_call']
        successful_calls = [log for log in self._access_log if log['access_type'] == 'method_success']
        error_calls = [log for log in self._access_log if log['access_type'] == 'method_error']
        
        return {
            'target_type': type(self._target).__name__,
            'reference_count': self._reference_count,
            'total_accesses': len(self._access_log),
            'method_calls': len(method_calls),
            'successful_calls': len(successful_calls),
            'error_calls': len(error_calls),
            'created_at': self._created_at.isoformat(),
            'uptime_seconds': (datetime.now() - self._created_at).total_seconds()
        }
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log."""
        return self._access_log.copy()


# ============================================================================
# PROXY FACTORY AND MANAGER
# ============================================================================

class ProxyFactory:
    """Factory for creating different types of proxies."""
    
    def __init__(self):
        self._created_proxies: List[Any] = []
    
    def create_virtual_proxy(self, target_class: type, *args, **kwargs) -> Any:
        """Create a virtual proxy for lazy initialization."""
        if target_class == RealImageService:
            proxy = VirtualImageProxy()
        else:
            # Generic virtual proxy implementation
            proxy = self._create_generic_virtual_proxy(target_class, *args, **kwargs)
        
        self._created_proxies.append(proxy)
        return proxy
    
    def create_protection_proxy(self, target_object: Any, permissions: Dict[str, List[str]]) -> Any:
        """Create a protection proxy with access control."""
        if isinstance(target_object, RealDatabaseService):
            proxy = ProtectionDatabaseProxy(target_object)
        else:
            # Generic protection proxy would go here
            proxy = target_object  # Simplified for demo
        
        self._created_proxies.append(proxy)
        return proxy
    
    def create_caching_proxy(self, target_object: Any, cache_ttl: int = 300) -> Any:
        """Create a caching proxy."""
        if isinstance(target_object, RealFileService):
            proxy = CachingFileProxy(target_object, cache_ttl)
        else:
            # Generic caching proxy would go here
            proxy = target_object  # Simplified for demo
        
        self._created_proxies.append(proxy)
        return proxy
    
    def create_remote_proxy(self, service_url: str, timeout: int = 30) -> RemoteServiceProxy:
        """Create a remote proxy."""
        proxy = RemoteServiceProxy(service_url, timeout)
        self._created_proxies.append(proxy)
        return proxy
    
    def create_smart_reference(self, target_object: Any) -> SmartReferenceProxy:
        """Create a smart reference proxy."""
        proxy = SmartReferenceProxy(target_object)
        self._created_proxies.append(proxy)
        return proxy
    
    def _create_generic_virtual_proxy(self, target_class: type, *args, **kwargs) -> Any:
        """Create a generic virtual proxy."""
        class GenericVirtualProxy:
            def __init__(self):
                self._target = None
                self._target_class = target_class
                self._args = args
                self._kwargs = kwargs
            
            def _get_target(self):
                if self._target is None:
                    print(f"GenericVirtualProxy: Creating {target_class.__name__} (lazy loading)")
                    self._target = target_class(*self._args, **self._kwargs)
                return self._target
            
            def __getattr__(self, name):
                return getattr(self._get_target(), name)
        
        return GenericVirtualProxy()
    
    def get_proxy_count(self) -> int:
        """Get number of created proxies."""
        return len(self._created_proxies)
    
    def get_proxy_types(self) -> List[str]:
        """Get types of created proxies."""
        return [type(proxy).__name__ for proxy in self._created_proxies]


def demonstrate_proxy_pattern():
    """
    Demonstrate Proxy pattern implementations.
    """
    print("=== PROXY PATTERN DEMONSTRATION ===\n")
    
    # 1. Virtual Proxy (Lazy Loading)
    print("1. VIRTUAL PROXY (LAZY LOADING):")
    
    # Create virtual proxy - real service not created yet
    image_proxy = VirtualImageProxy()
    print("   Virtual proxy created - real service not instantiated yet")
    
    # Get image info - doesn't require full loading
    print("\n   Getting image info (lightweight operation):")
    info = image_proxy.get_image_info("photo1.jpg")
    print(f"   Image info: {info['width']}x{info['height']} {info['format']}")
    
    # Load image - now real service is created
    print("\n   Loading image (triggers real service creation):")
    image_data = image_proxy.load_image("photo1.jpg")
    print(f"   Image loaded: {image_data['path']} ({image_data['size_bytes']} bytes)")
    
    # Load another image - uses existing real service
    print("\n   Loading another image (reuses real service):")
    image_data2 = image_proxy.load_image("photo2.jpg")
    print(f"   Second image loaded: {image_data2['path']}")
    
    # Load first image again - served from cache
    print("\n   Loading first image again (served from cache):")
    cached_image = image_proxy.load_image("photo1.jpg")
    print(f"   Cached image: {cached_image['path']}")
    
    # Show cache statistics
    cache_stats = image_proxy.get_cache_stats()
    print(f"\n   Cache statistics: {cache_stats}")
    
    print()
    
    # 2. Protection Proxy (Access Control)
    print("2. PROTECTION PROXY (ACCESS CONTROL):")
    
    # Create real database service
    real_db = RealDatabaseService("postgresql://localhost:5432/mydb")
    
    # Create protection proxy
    db_proxy = ProtectionDatabaseProxy(real_db)
    
    # Create users with different roles
    guest_user = User("guest", UserRole.GUEST, [])
    regular_user = User("john", UserRole.USER, ["db_connect", "read_users", "read_products"])
    admin_user = User("admin", UserRole.ADMIN, ["db_connect", "read_users", "read_products", "write_users", "admin_access"])
    
    # Test access with different users
    print("   Testing access with different user roles:")
    
    # Guest user (no permissions)
    print("\n   Guest user attempting access:")
    db_proxy.authenticate(guest_user)
    try:
        db_proxy.connect()
        print("   Guest connected (unexpected)")
    except:
        print("   Guest connection denied (expected)")
    
    # Regular user
    print("\n   Regular user attempting access:")
    db_proxy.authenticate(regular_user)
    if db_proxy.connect():
        print("   Regular user connected successfully")
        
        # Try to query users (allowed)
        try:
            users = db_proxy.query("SELECT * FROM users")
            print(f"   Query successful: {len(users)} users found")
        except PermissionError as e:
            print(f"   Query denied: {e}")
        
        # Try to update users (not allowed)
        try:
            db_proxy.execute("UPDATE users SET name='New Name' WHERE id=1")
            print("   Update successful (unexpected)")
        except PermissionError as e:
            print(f"   Update denied: {str(e)[:50]}...")
    
    # Admin user
    print("\n   Admin user attempting access:")
    db_proxy.authenticate(admin_user)
    if db_proxy.connect():
        print("   Admin connected successfully")
        
        # Try admin operations
        try:
            info = db_proxy.get_connection_info()
            print(f"   Connection info retrieved: {info['connection_count']} connections")
        except:
            print("   Connection info denied")
        
        # Get access log
        access_log = db_proxy.get_access_log()
        print(f"   Access log entries: {len(access_log)}")
    
    db_proxy.disconnect()
    print()
    
    # 3. Caching Proxy (Performance Optimization)
    print("3. CACHING PROXY (PERFORMANCE OPTIMIZATION):")
    
    # Create real file service and caching proxy
    real_file_service = RealFileService()
    file_proxy = CachingFileProxy(real_file_service, cache_ttl=60)
    
    print("   Testing file operations with caching:")
    
    # First read - cache miss
    print("\n   First read (cache miss):")
    start_time = time.time()
    content1 = file_proxy.read_file("document1.txt")
    read_time1 = time.time() - start_time
    print(f"   Read completed in {read_time1:.3f}s")
    print(f"   Content length: {len(content1)} characters")
    
    # Second read - cache hit
    print("\n   Second read (cache hit):")
    start_time = time.time()
    content2 = file_proxy.read_file("document1.txt")
    read_time2 = time.time() - start_time
    print(f"   Read completed in {read_time2:.3f}s")
    print(f"   Speed improvement: {read_time1/read_time2:.1f}x faster")
    
    # Write file - updates cache
    print("\n   Writing file (updates cache):")
    new_content = "Updated content with timestamp: " + datetime.now().isoformat()
    file_proxy.write_file("document1.txt", new_content)
    
    # Read updated content - served from cache
    print("\n   Reading updated content (from cache):")
    updated_content = file_proxy.read_file("document1.txt")
    print(f"   Content updated: {updated_content[:50]}...")
    
    # Show cache statistics
    cache_stats = file_proxy.get_cache_stats()
    print(f"\n   Cache statistics:")
    print(f"     Hit ratio: {cache_stats['hit_ratio']}%")
    print(f"     Cache size: {cache_stats['cache_size']} files")
    print(f"     Hits: {cache_stats['stats']['hits']}")
    print(f"     Misses: {cache_stats['stats']['misses']}")
    
    print()
    
    # 4. Remote Proxy (Distributed Objects)
    print("4. REMOTE PROXY (DISTRIBUTED OBJECTS):")
    
    # Create remote service proxy
    remote_proxy = RemoteServiceProxy("https://api.example.com", timeout=30)
    
    print("   Testing remote service operations:")
    
    # Get user from remote service
    print("\n   Getting user from remote service:")
    user_data = remote_proxy.get_user(123)
    if 'error' not in user_data:
        print(f"   User retrieved: {user_data['name']} ({user_data['email']})")
    else:
        print(f"   Error: {user_data['error']}")
    
    # Create user on remote service
    print("\n   Creating user on remote service:")
    new_user_data = {
        'name': 'Alice Johnson',
        'email': 'alice@example.com',
        'role': 'user'
    }
    
    create_result = remote_proxy.create_user(new_user_data)
    if 'error' not in create_result:
        print(f"   User created with ID: {create_result['user_id']}")
    else:
        print(f"   Error: {create_result['error']}")
    
    # Show service statistics
    service_stats = remote_proxy.get_service_stats()
    print(f"\n   Remote service statistics:")
    print(f"     Requests made: {service_stats['request_count']}")
    print(f"     Error rate: {service_stats['error_rate']:.1f}%")
    print(f"     Service URL: {service_stats['service_url']}")
    
    print()
    
    # 5. Smart Reference Proxy
    print("5. SMART REFERENCE PROXY:")
    
    # Create a target object and smart reference
    target_service = RealFileService()
    smart_ref = SmartReferenceProxy(target_service)
    
    print("   Testing smart reference functionality:")
    
    # Use the service through smart reference
    print("\n   Using service through smart reference:")
    content = smart_ref.read_file("test.txt")
    print(f"   File read through smart reference: {len(content)} characters")
    
    # Write file
    smart_ref.write_file("test.txt", "Modified content")
    print("   File written through smart reference")
    
    # Get file info
    info = smart_ref.get_file_info("test.txt")
    print(f"   File info: {info['size_bytes']} bytes")
    
    # Show reference statistics
    ref_stats = smart_ref.get_reference_stats()
    print(f"\n   Smart reference statistics:")
    print(f"     Target type: {ref_stats['target_type']}")
    print(f"     Reference count: {ref_stats['reference_count']}")
    print(f"     Method calls: {ref_stats['method_calls']}")
    print(f"     Successful calls: {ref_stats['successful_calls']}")
    print(f"     Uptime: {ref_stats['uptime_seconds']:.1f} seconds")
    
    print()
    
    # 6. Proxy Factory
    print("6. PROXY FACTORY:")
    
    factory = ProxyFactory()
    
    print("   Creating different types of proxies through factory:")
    
    # Create virtual proxy
    virtual_proxy = factory.create_virtual_proxy(RealImageService)
    print(f"   Virtual proxy created: {type(virtual_proxy).__name__}")
    
    # Create protection proxy
    protection_proxy = factory.create_protection_proxy(real_db, {})
    print(f"   Protection proxy created: {type(protection_proxy).__name__}")
    
    # Create caching proxy
    caching_proxy = factory.create_caching_proxy(real_file_service, 120)
    print(f"   Caching proxy created: {type(caching_proxy).__name__}")
    
    # Create remote proxy
    remote_proxy2 = factory.create_remote_proxy("https://api2.example.com")
    print(f"   Remote proxy created: {type(remote_proxy2).__name__}")
    
    # Create smart reference
    smart_ref2 = factory.create_smart_reference(target_service)
    print(f"   Smart reference created: {type(smart_ref2).__name__}")
    
    # Show factory statistics
    print(f"\n   Factory statistics:")
    print(f"     Total proxies created: {factory.get_proxy_count()}")
    print(f"     Proxy types: {factory.get_proxy_types()}")
    
    print()
    
    # 7. Proxy Pattern Benefits
    print("7. PROXY PATTERN BENEFITS:")
    print("   ✓ Lazy Loading: Expensive objects created only when needed")
    print("   ✓ Access Control: Fine-grained permission management")
    print("   ✓ Caching: Improved performance through result caching")
    print("   ✓ Remote Access: Transparent access to distributed objects")
    print("   ✓ Smart References: Additional functionality without changing interface")
    print("   ✓ Resource Management: Control over resource allocation and cleanup")
    print("   ✓ Logging/Monitoring: Transparent operation tracking")
    print("   ✓ Interface Preservation: Same interface as real object")
    print()
    
    print("=== PROXY PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_proxy_pattern()
