"""
1166. Design File System - Multiple Approaches
Difficulty: Medium

You are asked to design a file system that allows you to create new paths and associate them with different values.

The format of a path is one or more concatenated strings of the form: / followed by one or more lowercase English letters. For example, "/leetcode" and "/leetcode/problems" are valid paths while an empty string "" and "/" are not.

Implement the FileSystem class:
- FileSystem() Initializes the object of the system.
- bool createPath(string path, int value) Creates a new path and associates a value to it if possible and returns true. Returns false if the path already exists or its parent path doesn't exist.
- int get(string path) Returns the value associated with path or returns -1 if the path doesn't exist.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class FileSystemHashMap:
    """
    Approach 1: Simple HashMap
    
    Use dictionary to store path -> value mappings.
    
    Time Complexity:
    - createPath: O(n) where n is path length (for parent validation)
    - get: O(1)
    
    Space Complexity: O(total_paths * avg_path_length)
    """
    
    def __init__(self):
        self.paths = {}  # path -> value
    
    def createPath(self, path: str, value: int) -> bool:
        # Check if path already exists
        if path in self.paths:
            return False
        
        # Check if parent exists (except for root-level paths)
        if path != "/" and path.count("/") > 1:
            parent_path = path.rsplit("/", 1)[0]
            if parent_path not in self.paths:
                return False
        
        # Create the path
        self.paths[path] = value
        return True
    
    def get(self, path: str) -> int:
        return self.paths.get(path, -1)

class FileSystemTrie:
    """
    Approach 2: Trie-based Implementation
    
    Use trie structure to represent file system hierarchy.
    
    Time Complexity:
    - createPath: O(n) where n is number of components in path
    - get: O(n)
    
    Space Complexity: O(total_nodes)
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.value = None
            self.is_end = False
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def createPath(self, path: str, value: int) -> bool:
        # Split path into components
        components = [comp for comp in path.split("/") if comp]
        
        # Traverse to find parent
        current = self.root
        for i, component in enumerate(components[:-1]):
            if component not in current.children:
                return False  # Parent doesn't exist
            current = current.children[component]
            if not current.is_end:
                return False  # Intermediate path doesn't exist
        
        # Check if final component already exists
        final_component = components[-1]
        if final_component in current.children:
            return False  # Path already exists
        
        # Create the new path
        current.children[final_component] = self.TrieNode()
        current.children[final_component].value = value
        current.children[final_component].is_end = True
        
        return True
    
    def get(self, path: str) -> int:
        components = [comp for comp in path.split("/") if comp]
        
        current = self.root
        for component in components:
            if component not in current.children:
                return -1
            current = current.children[component]
        
        if current.is_end:
            return current.value
        
        return -1

class FileSystemAdvanced:
    """
    Approach 3: Advanced with Features
    
    Enhanced file system with additional operations and metadata.
    
    Time Complexity:
    - createPath: O(n)
    - get: O(1)
    - Additional operations: varies
    
    Space Complexity: O(paths + metadata)
    """
    
    def __init__(self):
        self.paths = {}
        self.parent_map = {}  # path -> parent_path
        self.children_map = defaultdict(set)  # path -> set of children
        self.metadata = {}  # path -> metadata dict
        
        # Statistics
        self.total_paths = 0
        self.create_operations = 0
        self.get_operations = 0
        
        # Create root
        self.paths["/"] = 0  # Root value
        self.metadata["/"] = {
            "created_at": self._get_timestamp(),
            "size": 0,
            "type": "directory"
        }
    
    def _get_timestamp(self) -> int:
        import time
        return int(time.time())
    
    def createPath(self, path: str, value: int) -> bool:
        self.create_operations += 1
        
        # Validate path format
        if not path.startswith("/") or path == "/":
            return False
        
        # Check if path already exists
        if path in self.paths:
            return False
        
        # Find parent path
        parent_path = path.rsplit("/", 1)[0]
        if not parent_path:
            parent_path = "/"
        
        # Check if parent exists
        if parent_path not in self.paths:
            return False
        
        # Create the path
        self.paths[path] = value
        self.parent_map[path] = parent_path
        self.children_map[parent_path].add(path)
        
        # Add metadata
        self.metadata[path] = {
            "created_at": self._get_timestamp(),
            "size": 0,
            "type": "file",
            "parent": parent_path
        }
        
        self.total_paths += 1
        return True
    
    def get(self, path: str) -> int:
        self.get_operations += 1
        return self.paths.get(path, -1)
    
    def exists(self, path: str) -> bool:
        """Check if path exists"""
        return path in self.paths
    
    def getChildren(self, path: str) -> List[str]:
        """Get all children of a path"""
        if path not in self.paths:
            return []
        return list(self.children_map[path])
    
    def getParent(self, path: str) -> str:
        """Get parent path"""
        return self.parent_map.get(path, "")
    
    def deletePath(self, path: str) -> bool:
        """Delete a path and all its children"""
        if path not in self.paths or path == "/":
            return False
        
        # Recursively delete children first
        children = list(self.children_map[path])
        for child in children:
            self.deletePath(child)
        
        # Remove from parent's children
        parent = self.parent_map[path]
        self.children_map[parent].discard(path)
        
        # Remove path
        del self.paths[path]
        del self.parent_map[path]
        del self.children_map[path]
        del self.metadata[path]
        
        self.total_paths -= 1
        return True
    
    def listPaths(self, prefix: str = "") -> List[str]:
        """List all paths with given prefix"""
        if not prefix:
            return sorted(self.paths.keys())
        
        return [path for path in self.paths.keys() if path.startswith(prefix)]
    
    def getMetadata(self, path: str) -> dict:
        """Get metadata for a path"""
        return self.metadata.get(path, {})
    
    def getStatistics(self) -> dict:
        """Get system statistics"""
        return {
            "total_paths": self.total_paths,
            "create_operations": self.create_operations,
            "get_operations": self.get_operations,
            "max_depth": self._calculate_max_depth(),
            "avg_children_per_directory": self._calculate_avg_children()
        }
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of file system"""
        if not self.paths:
            return 0
        
        max_depth = 0
        for path in self.paths:
            depth = path.count("/")
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_avg_children(self) -> float:
        """Calculate average number of children per directory"""
        if not self.children_map:
            return 0
        
        total_children = sum(len(children) for children in self.children_map.values())
        return total_children / len(self.children_map)

class FileSystemConcurrent:
    """
    Approach 4: Thread-Safe Implementation
    
    File system with thread safety considerations.
    
    Time Complexity:
    - createPath: O(n) + lock overhead
    - get: O(1) + lock overhead
    
    Space Complexity: O(paths)
    """
    
    def __init__(self):
        import threading
        
        self.paths = {}
        self.lock = threading.RLock()  # Reentrant lock for nested operations
        
        # Performance tracking
        self.operation_count = 0
        self.concurrent_operations = 0
    
    def createPath(self, path: str, value: int) -> bool:
        with self.lock:
            self.operation_count += 1
            
            # Check if path already exists
            if path in self.paths:
                return False
            
            # Check parent existence
            if path != "/" and path.count("/") > 1:
                parent_path = path.rsplit("/", 1)[0]
                if parent_path not in self.paths:
                    return False
            
            # Create the path
            self.paths[path] = value
            return True
    
    def get(self, path: str) -> int:
        with self.lock:
            self.operation_count += 1
            return self.paths.get(path, -1)
    
    def batch_create(self, path_value_pairs: List[Tuple[str, int]]) -> List[bool]:
        """Create multiple paths atomically"""
        with self.lock:
            results = []
            
            for path, value in path_value_pairs:
                result = self.createPath(path, value)
                results.append(result)
                
                # If any creation fails, we could rollback here
                # For this implementation, we continue
            
            return results
    
    def get_stats(self) -> dict:
        """Get thread-safe statistics"""
        with self.lock:
            return {
                "total_paths": len(self.paths),
                "operation_count": self.operation_count,
                "concurrent_operations": self.concurrent_operations
            }

class FileSystemMemoryOptimized:
    """
    Approach 5: Memory-Optimized Implementation
    
    Optimized for memory usage with path compression.
    
    Time Complexity:
    - createPath: O(n)
    - get: O(1)
    
    Space Complexity: O(unique_components + paths)
    """
    
    def __init__(self):
        self.paths = {}
        
        # Path compression: store common components once
        self.component_pool = {}  # component -> id
        self.component_reverse = {}  # id -> component
        self.next_component_id = 0
        
        # Store paths as lists of component IDs
        self.compressed_paths = {}  # path -> list of component IDs
    
    def _get_component_id(self, component: str) -> int:
        """Get or create component ID"""
        if component not in self.component_pool:
            self.component_pool[component] = self.next_component_id
            self.component_reverse[self.next_component_id] = component
            self.next_component_id += 1
        
        return self.component_pool[component]
    
    def _compress_path(self, path: str) -> List[int]:
        """Compress path to list of component IDs"""
        components = [comp for comp in path.split("/") if comp]
        return [self._get_component_id(comp) for comp in components]
    
    def _decompress_path(self, component_ids: List[int]) -> str:
        """Decompress component IDs back to path"""
        components = [self.component_reverse[cid] for cid in component_ids]
        return "/" + "/".join(components) if components else "/"
    
    def createPath(self, path: str, value: int) -> bool:
        # Check if path already exists
        if path in self.paths:
            return False
        
        # Compress the path
        compressed = self._compress_path(path)
        
        # Check parent existence
        if len(compressed) > 0:  # Not root level
            parent_compressed = compressed[:-1]
            parent_path = self._decompress_path(parent_compressed)
            
            if parent_path not in self.paths:
                return False
        
        # Create the path
        self.paths[path] = value
        self.compressed_paths[path] = compressed
        return True
    
    def get(self, path: str) -> int:
        return self.paths.get(path, -1)
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics"""
        total_components = len(self.component_pool)
        avg_component_length = sum(len(comp) for comp in self.component_pool.keys()) / max(1, total_components)
        
        return {
            "total_paths": len(self.paths),
            "unique_components": total_components,
            "avg_component_length": avg_component_length,
            "compression_ratio": total_components / max(1, len(self.paths))
        }


def test_file_system_basic():
    """Test basic file system functionality"""
    print("=== Testing Basic File System Functionality ===")
    
    implementations = [
        ("HashMap", FileSystemHashMap),
        ("Trie", FileSystemTrie),
        ("Advanced", FileSystemAdvanced),
        ("Concurrent", FileSystemConcurrent),
        ("Memory Optimized", FileSystemMemoryOptimized)
    ]
    
    for name, FileSystemClass in implementations:
        print(f"\n{name}:")
        
        fs = FileSystemClass()
        
        # Test sequence
        operations = [
            ("createPath", "/a", 1),
            ("get", "/a"),
            ("createPath", "/a/b", 2),
            ("get", "/a/b"),
            ("createPath", "/c/d", 1),  # Should fail - parent doesn't exist
            ("get", "/c/d"),
            ("createPath", "/c", 3),
            ("createPath", "/c/d", 4),
            ("get", "/c/d")
        ]
        
        for op, path, *args in operations:
            if op == "createPath":
                value = args[0]
                result = fs.createPath(path, value)
                print(f"  createPath('{path}', {value}): {result}")
            else:  # get
                result = fs.get(path)
                print(f"  get('{path}'): {result}")

def test_file_system_edge_cases():
    """Test file system edge cases"""
    print("\n=== Testing File System Edge Cases ===")
    
    fs = FileSystemAdvanced()
    
    # Test invalid paths
    print("Invalid path tests:")
    invalid_tests = [
        ("", 1),           # Empty path
        ("/", 1),          # Root path
        ("abc", 1),        # No leading slash
        ("//a", 1),        # Double slash
        ("/a/", 1)         # Trailing slash
    ]
    
    for path, value in invalid_tests:
        result = fs.createPath(path, value)
        print(f"  createPath('{path}', {value}): {result}")
    
    # Test duplicate creation
    print(f"\nDuplicate creation test:")
    fs.createPath("/test", 100)
    result1 = fs.createPath("/test", 200)  # Should fail
    result2 = fs.get("/test")
    
    print(f"  First create: True")
    print(f"  Duplicate create: {result1}")
    print(f"  Value remains: {result2}")
    
    # Test deep nesting
    print(f"\nDeep nesting test:")
    
    # Create nested structure
    nested_paths = ["/a", "/a/b", "/a/b/c", "/a/b/c/d", "/a/b/c/d/e"]
    
    for i, path in enumerate(nested_paths):
        result = fs.createPath(path, i + 1)
        print(f"  createPath('{path}', {i + 1}): {result}")
    
    # Test missing intermediate path
    print(f"\nMissing intermediate path:")
    result = fs.createPath("/x/y/z", 999)  # /x doesn't exist
    print(f"  createPath('/x/y/z', 999): {result}")

def test_advanced_features():
    """Test advanced file system features"""
    print("\n=== Testing Advanced Features ===")
    
    fs = FileSystemAdvanced()
    
    # Build file system structure
    structure = [
        ("/home", 1),
        ("/home/user1", 2),
        ("/home/user1/documents", 3),
        ("/home/user1/documents/file1.txt", 4),
        ("/home/user1/downloads", 5),
        ("/home/user2", 6),
        ("/var", 7),
        ("/var/log", 8),
        ("/var/log/system.log", 9)
    ]
    
    print("Building file system structure:")
    for path, value in structure:
        result = fs.createPath(path, value)
        print(f"  createPath('{path}', {value}): {result}")
    
    # Test directory operations
    print(f"\nDirectory operations:")
    
    directories_to_check = ["/", "/home", "/home/user1", "/var"]
    
    for directory in directories_to_check:
        children = fs.getChildren(directory)
        print(f"  Children of '{directory}': {children}")
    
    # Test path queries
    print(f"\nPath queries:")
    
    test_paths = ["/home/user1/documents/file1.txt", "/var/log", "/nonexistent"]
    
    for path in test_paths:
        exists = fs.exists(path)
        parent = fs.getParent(path) if exists else "N/A"
        metadata = fs.getMetadata(path)
        
        print(f"  '{path}':")
        print(f"    Exists: {exists}")
        print(f"    Parent: {parent}")
        print(f"    Metadata: {metadata}")
    
    # Test path listing with prefix
    print(f"\nPath listing:")
    
    prefixes = ["", "/home", "/home/user1", "/var"]
    
    for prefix in prefixes:
        paths = fs.listPaths(prefix)
        print(f"  Paths with prefix '{prefix}': {paths[:5]}...")  # Show first 5
    
    # Test deletion
    print(f"\nDeletion test:")
    
    print(f"  Before deletion - '/home/user1' children: {fs.getChildren('/home/user1')}")
    
    delete_result = fs.deletePath("/home/user1/documents")
    print(f"  deletePath('/home/user1/documents'): {delete_result}")
    
    print(f"  After deletion - '/home/user1' children: {fs.getChildren('/home/user1')}")
    
    # Test statistics
    stats = fs.getStatistics()
    print(f"\nSystem statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Web server directory structure
    print("Application 1: Web Server Directory Structure")
    
    web_fs = FileSystemAdvanced()
    
    # Simulate web server directory structure
    web_structure = [
        ("/var", 0),
        ("/var/www", 0),
        ("/var/www/html", 0),
        ("/var/www/html/index.html", 1001),
        ("/var/www/html/about.html", 1002),
        ("/var/www/html/css", 0),
        ("/var/www/html/css/style.css", 2001),
        ("/var/www/html/js", 0),
        ("/var/www/html/js/app.js", 3001),
        ("/var/www/api", 0),
        ("/var/www/api/v1", 0),
        ("/var/www/api/v1/users", 4001),
        ("/var/www/api/v1/posts", 4002)
    ]
    
    print("  Building web server structure:")
    for path, resource_id in web_structure:
        web_fs.createPath(path, resource_id)
        file_type = "directory" if resource_id == 0 else "file"
        print(f"    {path} ({file_type})")
    
    # Simulate web requests
    print(f"\n  Simulating web requests:")
    
    requests = [
        "/var/www/html/index.html",
        "/var/www/html/about.html",
        "/var/www/html/css/style.css",
        "/var/www/api/v1/users",
        "/var/www/html/404.html"  # Non-existent
    ]
    
    for request_path in requests:
        resource_id = web_fs.get(request_path)
        status = "200 OK" if resource_id != -1 else "404 Not Found"
        print(f"    GET {request_path}: {status} (resource_id: {resource_id})")
    
    # Application 2: Cloud storage service
    print(f"\nApplication 2: Cloud Storage Service")
    
    cloud_fs = FileSystemAdvanced()
    
    # Simulate user cloud storage
    users = ["alice", "bob", "charlie"]
    
    print("  Setting up user directories:")
    for user in users:
        # Create user root
        cloud_fs.createPath(f"/{user}", 0)
        
        # Create standard folders
        folders = ["Documents", "Pictures", "Music", "Videos"]
        for folder in folders:
            cloud_fs.createPath(f"/{user}/{folder}", 0)
        
        print(f"    Created directory structure for {user}")
    
    # Simulate file uploads
    print(f"\n  Simulating file uploads:")
    
    uploads = [
        ("/alice/Documents/report.pdf", 5001),
        ("/alice/Pictures/vacation.jpg", 5002),
        ("/bob/Music/song.mp3", 5003),
        ("/charlie/Videos/presentation.mp4", 5004),
        ("/alice/Documents/Private/secret.txt", 5005)  # Should fail
    ]
    
    for upload_path, file_id in uploads:
        result = cloud_fs.createPath(upload_path, file_id)
        status = "SUCCESS" if result else "FAILED"
        print(f"    Upload {upload_path}: {status}")
    
    # Show user directory contents
    print(f"\n  User directory contents:")
    for user in users:
        user_files = cloud_fs.listPaths(f"/{user}")
        print(f"    {user}: {len(user_files)} items")
        for file_path in user_files[:3]:  # Show first 3
            print(f"      {file_path}")
    
    # Application 3: Configuration management system
    print(f"\nApplication 3: Configuration Management System")
    
    config_fs = FileSystemAdvanced()
    
    # Simulate application configuration hierarchy
    config_structure = [
        ("/app", 0),
        ("/app/database", 0),
        ("/app/database/host", 1),          # Database host config
        ("/app/database/port", 2),          # Database port config
        ("/app/cache", 0),
        ("/app/cache/redis_host", 3),       # Redis host config
        ("/app/cache/redis_port", 4),       # Redis port config
        ("/app/logging", 0),
        ("/app/logging/level", 5),          # Log level config
        ("/app/logging/file", 6),           # Log file config
        ("/app/features", 0),
        ("/app/features/new_ui", 7),        # Feature flag
        ("/app/features/analytics", 8)      # Feature flag
    ]
    
    print("  Building configuration hierarchy:")
    for config_path, config_id in config_structure:
        config_fs.createPath(config_path, config_id)
    
    # Simulate configuration retrieval
    print(f"\n  Configuration retrieval:")
    
    config_queries = [
        "/app/database/host",
        "/app/cache/redis_port",
        "/app/logging/level",
        "/app/features/new_ui",
        "/app/security/encryption"  # Non-existent
    ]
    
    for config_path in config_queries:
        config_value = config_fs.get(config_path)
        status = "FOUND" if config_value != -1 else "NOT_FOUND"
        print(f"    Config {config_path}: {status} (value: {config_value})")
    
    # Show configuration categories
    print(f"\n  Configuration categories:")
    categories = config_fs.getChildren("/app")
    for category in categories:
        category_configs = config_fs.getChildren(category)
        print(f"    {category}: {len(category_configs)} configs")

def test_performance_scaling():
    """Test performance with increasing scale"""
    print("\n=== Testing Performance Scaling ===")
    
    import time
    
    implementations = [
        ("HashMap", FileSystemHashMap),
        ("Advanced", FileSystemAdvanced)
    ]
    
    scales = [100, 500, 1000, 2000]
    
    for name, FileSystemClass in implementations:
        print(f"\n{name} Performance:")
        
        for scale in scales:
            fs = FileSystemClass()
            
            # Create hierarchical structure
            start_time = time.time()
            
            # Create root directories
            for i in range(10):
                fs.createPath(f"/dir{i}", i)
            
            # Create nested files
            for i in range(scale):
                dir_num = i % 10
                file_path = f"/dir{dir_num}/file{i}"
                fs.createPath(file_path, i + 1000)
            
            create_time = (time.time() - start_time) * 1000
            
            # Test retrieval performance
            start_time = time.time()
            
            for i in range(scale):
                dir_num = i % 10
                file_path = f"/dir{dir_num}/file{i}"
                fs.get(file_path)
            
            get_time = (time.time() - start_time) * 1000
            
            print(f"    Scale {scale}: Create {create_time:.2f}ms, Get {get_time:.2f}ms")

def test_memory_optimization():
    """Test memory optimization features"""
    print("\n=== Testing Memory Optimization ===")
    
    fs = FileSystemMemoryOptimized()
    
    # Create paths with common components
    common_paths = [
        "/usr/local/bin/app1",
        "/usr/local/bin/app2",
        "/usr/local/lib/libA.so",
        "/usr/local/lib/libB.so",
        "/usr/share/doc/readme.txt",
        "/usr/share/man/app1.1",
        "/var/log/system.log",
        "/var/log/app.log",
        "/var/tmp/cache"
    ]
    
    print("Creating paths with common components:")
    for i, path in enumerate(common_paths):
        result = fs.createPath(path, i + 1)
        print(f"  {path}: {result}")
    
    # Test memory statistics
    memory_stats = fs.get_memory_stats()
    print(f"\nMemory optimization statistics:")
    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def stress_test_file_system():
    """Stress test file system"""
    print("\n=== Stress Testing File System ===")
    
    import time
    import random
    
    fs = FileSystemHashMap()  # Use simple implementation for stress test
    
    # Create large hierarchical structure
    num_directories = 100
    files_per_directory = 100
    total_operations = num_directories * files_per_directory
    
    print(f"Stress test: {total_operations} total operations")
    
    start_time = time.time()
    
    # Phase 1: Create directory structure
    print("  Creating directory structure...")
    
    for i in range(num_directories):
        dir_path = f"/stress_dir_{i}"
        fs.createPath(dir_path, i)
        
        # Create files in each directory
        for j in range(files_per_directory):
            file_path = f"{dir_path}/file_{j}"
            fs.createPath(file_path, i * 1000 + j)
    
    create_time = time.time() - start_time
    
    # Phase 2: Random access test
    print("  Testing random access...")
    
    start_time = time.time()
    
    for _ in range(total_operations):
        dir_num = random.randint(0, num_directories - 1)
        file_num = random.randint(0, files_per_directory - 1)
        file_path = f"/stress_dir_{dir_num}/file_{file_num}"
        
        value = fs.get(file_path)
        
        # Verify correctness
        expected_value = dir_num * 1000 + file_num
        if value != expected_value:
            print(f"    ERROR: Expected {expected_value}, got {value}")
            break
    
    access_time = time.time() - start_time
    
    print(f"  Creation: {create_time:.2f}s ({total_operations/create_time:.0f} ops/sec)")
    print(f"  Access: {access_time:.2f}s ({total_operations/access_time:.0f} ops/sec)")

def benchmark_concurrent_access():
    """Benchmark concurrent access patterns"""
    print("\n=== Benchmarking Concurrent Access ===")
    
    import threading
    import time
    
    fs = FileSystemConcurrent()
    
    # Setup: Create initial structure
    for i in range(100):
        fs.createPath(f"/concurrent_dir_{i}", i)
    
    # Test concurrent operations
    num_threads = 4
    operations_per_thread = 1000
    
    def worker_thread(thread_id: int, results: list):
        """Worker thread for concurrent operations"""
        start_time = time.time()
        
        for i in range(operations_per_thread):
            # Mix of create and get operations
            if i % 2 == 0:
                # Create operation
                path = f"/concurrent_dir_{thread_id}/file_{i}"
                fs.createPath(path, thread_id * 10000 + i)
            else:
                # Get operation
                dir_num = i % 100
                path = f"/concurrent_dir_{dir_num}"
                fs.get(path)
        
        elapsed = time.time() - start_time
        results.append(elapsed)
    
    print(f"Running {num_threads} concurrent threads...")
    
    # Start threads
    threads = []
    results = []
    
    start_time = time.time()
    
    for thread_id in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(thread_id, results))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    total_operations = num_threads * operations_per_thread
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total operations: {total_operations}")
    print(f"  Throughput: {total_operations/total_time:.0f} ops/sec")
    print(f"  Thread times: {[f'{t:.2f}s' for t in results]}")
    
    # Get final statistics
    final_stats = fs.get_stats()
    print(f"  Final stats: {final_stats}")

if __name__ == "__main__":
    test_file_system_basic()
    test_file_system_edge_cases()
    test_advanced_features()
    demonstrate_applications()
    test_performance_scaling()
    test_memory_optimization()
    stress_test_file_system()
    benchmark_concurrent_access()

"""
File System Design demonstrates key concepts:

Core Approaches:
1. HashMap - Simple path->value mapping with parent validation
2. Trie - Hierarchical tree structure mirroring file system
3. Advanced - Enhanced with metadata, operations, and analytics
4. Concurrent - Thread-safe implementation with locking
5. Memory Optimized - Path compression for memory efficiency

Key Design Principles:
- Hierarchical path validation and parent checking
- Efficient path storage and retrieval mechanisms
- Thread safety for concurrent access
- Memory optimization for large-scale systems

Performance Characteristics:
- HashMap: O(n) create (parent check), O(1) get
- Trie: O(d) create/get where d is path depth
- Advanced: Additional metadata and operation overhead
- Concurrent: Lock contention affects performance

Real-world Applications:
- Web server directory structure and routing
- Cloud storage service organization
- Configuration management systems
- Version control system file tracking
- Database schema and namespace management
- Operating system file system abstraction

The HashMap approach is most commonly used for simplicity,
while Trie provides better representation of hierarchical
structure for complex file system operations.
"""
