"""
588. Design In-Memory File System - Multiple Approaches
Difficulty: Hard

Design a data structure that simulates an in-memory file system.

Implement the FileSystem class:
- FileSystem() Initializes the object of the system.
- List<String> ls(String path) If path is a file path, returns a list that only contains this file's name. 
  If path is a directory path, returns the list of file and directory names in this directory.
- void mkdir(String path) Makes a new directory according to the given path. The given directory path 
  does not exist. If the middle directories in the path do not exist, you should create them as well.
- void addContentToFile(String path, String content) If filePath does not exist, creates that file 
  containing given content. If filePath already exists, appends the given content to original content.
- String readContentFromFile(String path) Returns the content in the file at filePath.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import os

class FileSystemBasic:
    """
    Approach 1: Basic Trie-based File System
    
    Use trie structure with nodes representing directories and files.
    
    Time Complexity: 
    - ls: O(path_length + result_size * log(result_size))
    - mkdir: O(path_length)
    - addContentToFile: O(path_length + content_length)
    - readContentFromFile: O(path_length)
    
    Space Complexity: O(total_files * average_path_length)
    """
    
    class Node:
        def __init__(self, name: str = ""):
            self.name = name
            self.is_file = False
            self.content = ""
            self.children = {}  # name -> Node
    
    def __init__(self):
        self.root = self.Node()
    
    def ls(self, path: str) -> List[str]:
        node = self._navigate_to_path(path)
        if not node:
            return []
        
        if node.is_file:
            # Return just the file name
            return [node.name]
        else:
            # Return sorted list of children
            return sorted(node.children.keys())
    
    def mkdir(self, path: str) -> None:
        self._navigate_to_path(path, create_if_missing=True)
    
    def addContentToFile(self, filePath: str, content: str) -> None:
        node = self._navigate_to_path(filePath, create_if_missing=True)
        node.is_file = True
        node.content += content
    
    def readContentFromFile(self, filePath: str) -> str:
        node = self._navigate_to_path(filePath)
        return node.content if node and node.is_file else ""
    
    def _navigate_to_path(self, path: str, create_if_missing: bool = False) -> Optional['FileSystemBasic.Node']:
        if path == "/":
            return self.root
        
        parts = [p for p in path.split("/") if p]  # Remove empty parts
        current = self.root
        
        for i, part in enumerate(parts):
            if part not in current.children:
                if create_if_missing:
                    current.children[part] = self.Node(part)
                else:
                    return None
            
            current = current.children[part]
            
            # Set name for the final node
            if i == len(parts) - 1:
                current.name = part
        
        return current

class FileSystemWithMetadata:
    """
    Approach 2: Enhanced with File Metadata
    
    Track creation time, modification time, and file size.
    
    Time Complexity: Same as basic implementation
    Space Complexity: O(total_files * (average_path_length + metadata_size))
    """
    
    class Node:
        def __init__(self, name: str = ""):
            self.name = name
            self.is_file = False
            self.content = ""
            self.children = {}
            
            # Metadata
            import time
            self.created_time = time.time()
            self.modified_time = time.time()
            self.size = 0
    
    def __init__(self):
        self.root = self.Node("/")
    
    def ls(self, path: str) -> List[str]:
        node = self._navigate_to_path(path)
        if not node:
            return []
        
        if node.is_file:
            return [node.name]
        else:
            return sorted(node.children.keys())
    
    def mkdir(self, path: str) -> None:
        self._navigate_to_path(path, create_if_missing=True)
    
    def addContentToFile(self, filePath: str, content: str) -> None:
        import time
        node = self._navigate_to_path(filePath, create_if_missing=True)
        node.is_file = True
        node.content += content
        node.modified_time = time.time()
        node.size = len(node.content)
    
    def readContentFromFile(self, filePath: str) -> str:
        node = self._navigate_to_path(filePath)
        return node.content if node and node.is_file else ""
    
    def getFileInfo(self, filePath: str) -> Dict[str, any]:
        """Get file metadata"""
        node = self._navigate_to_path(filePath)
        if not node:
            return {}
        
        return {
            'name': node.name,
            'is_file': node.is_file,
            'size': node.size,
            'created_time': node.created_time,
            'modified_time': node.modified_time,
            'content_length': len(node.content)
        }
    
    def _navigate_to_path(self, path: str, create_if_missing: bool = False) -> Optional['FileSystemWithMetadata.Node']:
        if path == "/":
            return self.root
        
        parts = [p for p in path.split("/") if p]
        current = self.root
        
        for i, part in enumerate(parts):
            if part not in current.children:
                if create_if_missing:
                    current.children[part] = self.Node(part)
                else:
                    return None
            current = current.children[part]
        
        return current

class FileSystemWithPermissions:
    """
    Approach 3: File System with Permissions
    
    Add basic permission system (read/write/execute).
    
    Time Complexity: Same as basic with permission check overhead
    Space Complexity: O(total_files * (path_length + permissions))
    """
    
    class Node:
        def __init__(self, name: str = ""):
            self.name = name
            self.is_file = False
            self.content = ""
            self.children = {}
            self.permissions = {'read': True, 'write': True, 'execute': True}
            self.owner = 'default'
    
    def __init__(self, user: str = 'default'):
        self.root = self.Node("/")
        self.current_user = user
    
    def ls(self, path: str) -> List[str]:
        node = self._navigate_to_path(path)
        if not node or not self._has_permission(node, 'read'):
            return []
        
        if node.is_file:
            return [node.name]
        else:
            # Filter by read permissions
            result = []
            for child_name, child_node in node.children.items():
                if self._has_permission(child_node, 'read'):
                    result.append(child_name)
            return sorted(result)
    
    def mkdir(self, path: str) -> None:
        # Check parent directory permissions
        parent_path = '/'.join(path.split('/')[:-1]) or '/'
        parent = self._navigate_to_path(parent_path)
        
        if parent and self._has_permission(parent, 'write'):
            node = self._navigate_to_path(path, create_if_missing=True)
            node.owner = self.current_user
    
    def addContentToFile(self, filePath: str, content: str) -> None:
        node = self._navigate_to_path(filePath)
        
        if node and node.is_file:
            # Existing file - check write permission
            if self._has_permission(node, 'write'):
                node.content += content
        else:
            # New file - check parent directory permission
            parent_path = '/'.join(filePath.split('/')[:-1]) or '/'
            parent = self._navigate_to_path(parent_path)
            
            if parent and self._has_permission(parent, 'write'):
                node = self._navigate_to_path(filePath, create_if_missing=True)
                node.is_file = True
                node.content = content
                node.owner = self.current_user
    
    def readContentFromFile(self, filePath: str) -> str:
        node = self._navigate_to_path(filePath)
        if node and node.is_file and self._has_permission(node, 'read'):
            return node.content
        return ""
    
    def chmod(self, path: str, permissions: Dict[str, bool]) -> bool:
        """Change file permissions"""
        node = self._navigate_to_path(path)
        if node and (node.owner == self.current_user or self.current_user == 'root'):
            node.permissions.update(permissions)
            return True
        return False
    
    def _has_permission(self, node: 'FileSystemWithPermissions.Node', permission: str) -> bool:
        """Check if current user has permission"""
        if self.current_user == 'root' or node.owner == self.current_user:
            return True
        return node.permissions.get(permission, False)
    
    def _navigate_to_path(self, path: str, create_if_missing: bool = False) -> Optional['FileSystemWithPermissions.Node']:
        if path == "/":
            return self.root
        
        parts = [p for p in path.split("/") if p]
        current = self.root
        
        for part in parts:
            if part not in current.children:
                if create_if_missing:
                    current.children[part] = self.Node(part)
                else:
                    return None
            current = current.children[part]
        
        return current

class FileSystemHashMap:
    """
    Approach 4: HashMap-based Implementation
    
    Use flat HashMap to store all paths for simpler implementation.
    
    Time Complexity: 
    - ls: O(total_paths) to filter + O(result * log(result)) to sort
    - mkdir: O(path_length)
    - addContentToFile: O(path_length + content_length)
    - readContentFromFile: O(path_length)
    
    Space Complexity: O(total_paths * average_path_length)
    """
    
    def __init__(self):
        self.files = {}  # path -> content
        self.directories = set()  # set of directory paths
        self.directories.add("/")
    
    def ls(self, path: str) -> List[str]:
        if path in self.files:
            # It's a file, return just the filename
            return [path.split("/")[-1]]
        
        if path not in self.directories:
            return []
        
        # It's a directory, find all immediate children
        children = set()
        prefix = path if path.endswith("/") else path + "/"
        if path == "/":
            prefix = "/"
        
        # Check files
        for file_path in self.files:
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if "/" not in relative:  # Direct child
                    children.add(relative)
        
        # Check directories
        for dir_path in self.directories:
            if dir_path != path and dir_path.startswith(prefix):
                relative = dir_path[len(prefix):]
                if "/" in relative:
                    # Get first directory component
                    children.add(relative.split("/")[0])
                elif relative:  # Direct child directory
                    children.add(relative)
        
        return sorted(list(children))
    
    def mkdir(self, path: str) -> None:
        # Create all parent directories
        parts = [p for p in path.split("/") if p]
        current_path = ""
        
        for part in parts:
            current_path += "/" + part
            self.directories.add(current_path)
    
    def addContentToFile(self, filePath: str, content: str) -> None:
        # Create parent directories
        parent = "/".join(filePath.split("/")[:-1])
        if parent and parent != "/":
            self.mkdir(parent)
        
        # Add or append content
        if filePath in self.files:
            self.files[filePath] += content
        else:
            self.files[filePath] = content
    
    def readContentFromFile(self, filePath: str) -> str:
        return self.files.get(filePath, "")

class FileSystemAdvanced:
    """
    Approach 5: Advanced File System with Additional Features
    
    Include features like file copying, moving, and searching.
    
    Time Complexity: Varies by operation
    Space Complexity: O(total_files * average_file_size)
    """
    
    class Node:
        def __init__(self, name: str = ""):
            self.name = name
            self.is_file = False
            self.content = ""
            self.children = {}
            import time
            self.created_time = time.time()
            self.modified_time = time.time()
    
    def __init__(self):
        self.root = self.Node("/")
    
    def ls(self, path: str) -> List[str]:
        node = self._navigate_to_path(path)
        if not node:
            return []
        
        if node.is_file:
            return [node.name]
        else:
            return sorted(node.children.keys())
    
    def mkdir(self, path: str) -> None:
        self._navigate_to_path(path, create_if_missing=True)
    
    def addContentToFile(self, filePath: str, content: str) -> None:
        import time
        node = self._navigate_to_path(filePath, create_if_missing=True)
        node.is_file = True
        node.content += content
        node.modified_time = time.time()
    
    def readContentFromFile(self, filePath: str) -> str:
        node = self._navigate_to_path(filePath)
        return node.content if node and node.is_file else ""
    
    def copyFile(self, source: str, destination: str) -> bool:
        """Copy file from source to destination"""
        source_node = self._navigate_to_path(source)
        if not source_node or not source_node.is_file:
            return False
        
        dest_node = self._navigate_to_path(destination, create_if_missing=True)
        dest_node.is_file = True
        dest_node.content = source_node.content
        import time
        dest_node.created_time = time.time()
        dest_node.modified_time = time.time()
        
        return True
    
    def moveFile(self, source: str, destination: str) -> bool:
        """Move file from source to destination"""
        if self.copyFile(source, destination):
            return self.deleteFile(source)
        return False
    
    def deleteFile(self, filePath: str) -> bool:
        """Delete a file"""
        parts = [p for p in filePath.split("/") if p]
        if not parts:
            return False
        
        # Navigate to parent
        current = self.root
        for part in parts[:-1]:
            if part not in current.children:
                return False
            current = current.children[part]
        
        # Delete the file/directory
        filename = parts[-1]
        if filename in current.children:
            del current.children[filename]
            return True
        
        return False
    
    def findFiles(self, pattern: str, root_path: str = "/") -> List[str]:
        """Find files matching pattern"""
        import re
        regex = re.compile(pattern)
        results = []
        
        def dfs(node: 'FileSystemAdvanced.Node', current_path: str) -> None:
            if node.is_file and regex.search(node.name):
                results.append(current_path)
            
            for child_name, child_node in node.children.items():
                child_path = current_path + "/" + child_name if current_path != "/" else "/" + child_name
                dfs(child_node, child_path)
        
        start_node = self._navigate_to_path(root_path)
        if start_node:
            dfs(start_node, root_path)
        
        return results
    
    def getDirectorySize(self, path: str) -> int:
        """Get total size of directory"""
        node = self._navigate_to_path(path)
        if not node:
            return 0
        
        def calculate_size(node: 'FileSystemAdvanced.Node') -> int:
            size = len(node.content) if node.is_file else 0
            for child in node.children.values():
                size += calculate_size(child)
            return size
        
        return calculate_size(node)
    
    def _navigate_to_path(self, path: str, create_if_missing: bool = False) -> Optional['FileSystemAdvanced.Node']:
        if path == "/":
            return self.root
        
        parts = [p for p in path.split("/") if p]
        current = self.root
        
        for part in parts:
            if part not in current.children:
                if create_if_missing:
                    current.children[part] = self.Node(part)
                else:
                    return None
            current = current.children[part]
        
        return current


def test_basic_file_operations():
    """Test basic file system operations"""
    print("=== Testing Basic File Operations ===")
    
    implementations = [
        ("Basic Trie", FileSystemBasic),
        ("With Metadata", FileSystemWithMetadata),
        ("HashMap-based", FileSystemHashMap)
    ]
    
    for name, FSClass in implementations:
        print(f"\n{name}:")
        
        fs = FSClass()
        
        # Test mkdir
        fs.mkdir("/home/user")
        print(f"  mkdir /home/user")
        
        # Test ls on directory
        result = fs.ls("/home")
        print(f"  ls /home: {result}")
        
        # Test file creation
        fs.addContentToFile("/home/user/file1.txt", "Hello World")
        print(f"  Created file1.txt with content")
        
        # Test file reading
        content = fs.readContentFromFile("/home/user/file1.txt")
        print(f"  Read file1.txt: '{content}'")
        
        # Test ls on directory with files
        result = fs.ls("/home/user")
        print(f"  ls /home/user: {result}")
        
        # Test ls on file
        result = fs.ls("/home/user/file1.txt")
        print(f"  ls /home/user/file1.txt: {result}")

def test_file_system_edge_cases():
    """Test file system edge cases"""
    print("\n=== Testing File System Edge Cases ===")
    
    fs = FileSystemBasic()
    
    # Test root directory
    print("Root directory operations:")
    result = fs.ls("/")
    print(f"  ls /: {result}")
    
    # Test non-existent path
    result = fs.ls("/non/existent/path")
    print(f"  ls /non/existent/path: {result}")
    
    # Test reading non-existent file
    content = fs.readContentFromFile("/non/existent/file.txt")
    print(f"  Read non-existent file: '{content}'")
    
    # Test creating nested directories
    fs.mkdir("/a/b/c/d")
    result = fs.ls("/a/b/c")
    print(f"  After mkdir /a/b/c/d, ls /a/b/c: {result}")
    
    # Test appending to file
    fs.addContentToFile("/test.txt", "First line\n")
    fs.addContentToFile("/test.txt", "Second line\n")
    content = fs.readContentFromFile("/test.txt")
    print(f"  Appended content: '{content.strip()}'")

def test_file_metadata():
    """Test file metadata features"""
    print("\n=== Testing File Metadata ===")
    
    fs = FileSystemWithMetadata()
    
    # Create file and check metadata
    fs.addContentToFile("/document.txt", "Sample content")
    
    info = fs.getFileInfo("/document.txt")
    print(f"File metadata:")
    print(f"  Name: {info['name']}")
    print(f"  Is file: {info['is_file']}")
    print(f"  Size: {info['size']} bytes")
    print(f"  Content length: {info['content_length']}")
    
    # Test directory metadata
    fs.mkdir("/projects")
    dir_info = fs.getFileInfo("/projects")
    print(f"\nDirectory metadata:")
    print(f"  Name: {dir_info['name']}")
    print(f"  Is file: {dir_info['is_file']}")

def test_permissions_system():
    """Test permissions system"""
    print("\n=== Testing Permissions System ===")
    
    # Create file system as regular user
    fs = FileSystemWithPermissions("user1")
    
    # Create file
    fs.addContentToFile("/user1_file.txt", "User1's content")
    print("User1 created file")
    
    # Change user
    fs.current_user = "user2"
    
    # Try to read (should work - default permissions)
    content = fs.readContentFromFile("/user1_file.txt")
    print(f"User2 read file: '{content}'")
    
    # Try to write (should work - default permissions)
    fs.addContentToFile("/user1_file.txt", "\nUser2's addition")
    content = fs.readContentFromFile("/user1_file.txt")
    print(f"After User2 write: '{content}'")
    
    # Change back to user1 and restrict permissions
    fs.current_user = "user1"
    fs.chmod("/user1_file.txt", {"read": True, "write": False})
    
    # Switch to user2 and try to write
    fs.current_user = "user2"
    fs.addContentToFile("/user1_file.txt", "\nShould not work")
    
    content = fs.readContentFromFile("/user1_file.txt")
    print(f"After restricted write attempt: '{content}'")

def test_advanced_features():
    """Test advanced file system features"""
    print("\n=== Testing Advanced Features ===")
    
    fs = FileSystemAdvanced()
    
    # Create some files
    fs.addContentToFile("/documents/report.txt", "Annual report content")
    fs.addContentToFile("/documents/notes.txt", "Meeting notes")
    fs.addContentToFile("/projects/code.py", "print('Hello World')")
    
    # Test file copying
    success = fs.copyFile("/documents/report.txt", "/backup/report_copy.txt")
    print(f"Copy file: {'Success' if success else 'Failed'}")
    
    # Verify copy
    original = fs.readContentFromFile("/documents/report.txt")
    copy_content = fs.readContentFromFile("/backup/report_copy.txt")
    print(f"Copy verification: {'✓' if original == copy_content else '✗'}")
    
    # Test file search
    txt_files = fs.findFiles(r".*\.txt$")
    print(f"Found .txt files: {txt_files}")
    
    # Test directory size
    docs_size = fs.getDirectorySize("/documents")
    print(f"Documents directory size: {docs_size} bytes")
    
    # Test file deletion
    success = fs.deleteFile("/documents/notes.txt")
    print(f"Delete file: {'Success' if success else 'Failed'}")
    
    remaining = fs.ls("/documents")
    print(f"Remaining files in documents: {remaining}")

def test_file_system_performance():
    """Test file system performance"""
    print("\n=== Testing File System Performance ===")
    
    import time
    
    implementations = [
        ("Basic Trie", FileSystemBasic),
        ("HashMap-based", FileSystemHashMap)
    ]
    
    num_files = 1000
    
    for name, FSClass in implementations:
        fs = FSClass()
        
        # Test file creation performance
        start_time = time.time()
        for i in range(num_files):
            fs.addContentToFile(f"/files/file_{i:04d}.txt", f"Content for file {i}")
        create_time = (time.time() - start_time) * 1000
        
        # Test file reading performance
        start_time = time.time()
        for i in range(0, num_files, 10):  # Read every 10th file
            fs.readContentFromFile(f"/files/file_{i:04d}.txt")
        read_time = (time.time() - start_time) * 1000
        
        # Test directory listing performance
        start_time = time.time()
        result = fs.ls("/files")
        ls_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Create {num_files} files: {create_time:.2f}ms")
        print(f"    Read {num_files//10} files: {read_time:.2f}ms")
        print(f"    List directory ({len(result)} items): {ls_time:.2f}ms")

def demonstrate_file_system_usage():
    """Demonstrate practical file system usage"""
    print("\n=== Demonstrating Practical Usage ===")
    
    fs = FileSystemAdvanced()
    
    # Simulate a project structure
    print("Creating project structure:")
    
    # Project directories
    fs.mkdir("/project/src")
    fs.mkdir("/project/tests")
    fs.mkdir("/project/docs")
    
    # Source files
    fs.addContentToFile("/project/src/main.py", "def main():\n    print('Hello World')\n")
    fs.addContentToFile("/project/src/utils.py", "def helper():\n    return 42\n")
    
    # Test files
    fs.addContentToFile("/project/tests/test_main.py", "import unittest\n\nclass TestMain(unittest.TestCase):\n    pass\n")
    
    # Documentation
    fs.addContentToFile("/project/docs/README.md", "# Project Documentation\n\nThis is a sample project.\n")
    
    # Show project structure
    print("Project structure:")
    def show_tree(path, prefix=""):
        items = fs.ls(path)
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            item_path = path + "/" + item if path != "/" else "/" + item
            
            # Check if it's a file or directory
            is_file = len(fs.readContentFromFile(item_path)) > 0
            icon = "📄" if is_file else "📁"
            
            print(f"{prefix}{'└── ' if is_last else '├── '}{icon} {item}")
            
            if not is_file:  # It's a directory
                extension = "    " if is_last else "│   "
                show_tree(item_path, prefix + extension)
    
    show_tree("/project")
    
    # Show file sizes
    print(f"\nProject statistics:")
    total_size = fs.getDirectorySize("/project")
    print(f"  Total size: {total_size} bytes")
    
    # Find Python files
    python_files = fs.findFiles(r".*\.py$", "/project")
    print(f"  Python files: {len(python_files)}")
    
    # Show content of main file
    main_content = fs.readContentFromFile("/project/src/main.py")
    print(f"  Main file content:\n{main_content}")

def benchmark_different_operations():
    """Benchmark different file system operations"""
    print("\n=== Benchmarking Operations ===")
    
    import time
    
    fs = FileSystemBasic()
    
    # Benchmark directory creation
    start_time = time.time()
    for i in range(100):
        fs.mkdir(f"/dirs/level1_{i}/level2_{i}")
    mkdir_time = (time.time() - start_time) * 1000
    
    # Benchmark file creation in different directories
    start_time = time.time()
    for i in range(100):
        fs.addContentToFile(f"/dirs/level1_{i}/file_{i}.txt", f"Content {i}")
    file_create_time = (time.time() - start_time) * 1000
    
    # Benchmark ls operations
    start_time = time.time()
    for i in range(100):
        fs.ls(f"/dirs/level1_{i}")
    ls_time = (time.time() - start_time) * 1000
    
    print(f"Operation benchmarks:")
    print(f"  100 mkdir operations: {mkdir_time:.2f}ms")
    print(f"  100 file creations: {file_create_time:.2f}ms")
    print(f"  100 ls operations: {ls_time:.2f}ms")

if __name__ == "__main__":
    test_basic_file_operations()
    test_file_system_edge_cases()
    test_file_metadata()
    test_permissions_system()
    test_advanced_features()
    test_file_system_performance()
    demonstrate_file_system_usage()
    benchmark_different_operations()

"""
In-Memory File System Design demonstrates key concepts:

Core Approaches:
1. Basic Trie - Tree structure mirroring file system hierarchy
2. With Metadata - Enhanced with creation/modification times and sizes
3. With Permissions - Basic user permission system
4. HashMap-based - Flat structure using path strings as keys
5. Advanced Features - File operations like copy, move, search

Key Design Principles:
- Hierarchical organization using trie data structure
- Efficient path navigation and resolution
- Memory management for file content storage
- Metadata tracking for file system semantics

Advanced Features:
- File permissions and access control
- File copying and moving operations
- Pattern-based file searching
- Directory size calculation
- File metadata tracking

Real-world Applications:
- IDE virtual file systems
- Container file systems (Docker)
- In-memory caching layers
- Version control systems
- Build system temporary storage
- Testing frameworks with mock file systems

The trie-based approach provides the most natural
representation of hierarchical file system structure.
"""
