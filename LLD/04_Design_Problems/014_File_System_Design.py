"""
FILE SYSTEM DESIGN - Complete System Design
==========================================

Problem Statement:
Design a comprehensive file system that handles:
- Hierarchical directory structure with files and folders
- File operations (create, read, write, delete, copy, move)
- Directory operations (create, list, navigate, delete)
- File permissions and access control
- File metadata management (size, timestamps, ownership)
- File system journaling and crash recovery
- Symbolic and hard links
- File system quotas and space management
- File search and indexing
- Concurrent access and locking mechanisms

Requirements:
- Support hierarchical directory structure
- Implement file and directory permissions (read, write, execute)
- Handle file metadata (creation time, modification time, size, owner)
- Support different file types (regular, directory, symlink, hardlink)
- Implement file system operations atomically
- Provide concurrent access with proper locking
- Support file system quotas per user/group
- Implement file search and indexing capabilities
- Handle file system corruption and recovery
- Support file compression and encryption
- Provide file system statistics and monitoring

Design Patterns Used:
- Composite: File system tree structure
- Strategy: Different file storage strategies
- Observer: File system event monitoring
- Command: File operations with undo/redo
- Factory: File and directory creation
- Singleton: File system instance
- Visitor: File system traversal operations
- Decorator: File compression and encryption
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Union, Iterator
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import os
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import stat


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class FileType(Enum):
    REGULAR = "regular"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    DEVICE = "device"
    PIPE = "pipe"


class Permission(Enum):
    READ = "r"
    WRITE = "w"
    EXECUTE = "x"


class FileSystemEvent(Enum):
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    FILE_COPIED = "file_copied"
    DIRECTORY_CREATED = "directory_created"
    DIRECTORY_DELETED = "directory_deleted"


@dataclass
class FilePermissions:
    """File permissions for owner, group, and others."""
    owner_read: bool = True
    owner_write: bool = True
    owner_execute: bool = False
    group_read: bool = True
    group_write: bool = False
    group_execute: bool = False
    other_read: bool = True
    other_write: bool = False
    other_execute: bool = False
    
    def to_octal(self) -> str:
        """Convert permissions to octal representation."""
        owner = (self.owner_read * 4 + self.owner_write * 2 + self.owner_execute * 1)
        group = (self.group_read * 4 + self.group_write * 2 + self.group_execute * 1)
        other = (self.other_read * 4 + self.other_write * 2 + self.other_execute * 1)
        return f"{owner}{group}{other}"
    
    def to_string(self) -> str:
        """Convert permissions to string representation (rwxrwxrwx)."""
        result = ""
        result += "r" if self.owner_read else "-"
        result += "w" if self.owner_write else "-"
        result += "x" if self.owner_execute else "-"
        result += "r" if self.group_read else "-"
        result += "w" if self.group_write else "-"
        result += "x" if self.group_execute else "-"
        result += "r" if self.other_read else "-"
        result += "w" if self.other_write else "-"
        result += "x" if self.other_execute else "-"
        return result
    
    @classmethod
    def from_octal(cls, octal: str) -> 'FilePermissions':
        """Create permissions from octal string."""
        if len(octal) != 3:
            raise ValueError("Octal permissions must be 3 digits")
        
        owner_val = int(octal[0])
        group_val = int(octal[1])
        other_val = int(octal[2])
        
        return cls(
            owner_read=bool(owner_val & 4),
            owner_write=bool(owner_val & 2),
            owner_execute=bool(owner_val & 1),
            group_read=bool(group_val & 4),
            group_write=bool(group_val & 2),
            group_execute=bool(group_val & 1),
            other_read=bool(other_val & 4),
            other_write=bool(other_val & 2),
            other_execute=bool(other_val & 1)
        )


@dataclass
class FileMetadata:
    """File metadata information."""
    inode: int
    file_type: FileType
    size: int
    permissions: FilePermissions
    owner: str
    group: str
    created_time: datetime
    modified_time: datetime
    accessed_time: datetime
    link_count: int = 1
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.checksum is None and self.file_type == FileType.REGULAR:
            self.checksum = ""


@dataclass
class FileSystemNode:
    """Base file system node."""
    name: str
    metadata: FileMetadata
    parent: Optional['FileSystemNode'] = None
    
    def get_full_path(self) -> str:
        """Get full path of the node."""
        if self.parent is None:
            return "/" if self.name == "" else f"/{self.name}"
        
        parent_path = self.parent.get_full_path()
        if parent_path == "/":
            return f"/{self.name}"
        return f"{parent_path}/{self.name}"


# ============================================================================
# FILE SYSTEM COMPONENTS
# ============================================================================

class FileSystemComponent(ABC):
    """Abstract file system component (Composite pattern)."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get component name."""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get component size."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> FileMetadata:
        """Get component metadata."""
        pass
    
    @abstractmethod
    def accept(self, visitor: 'FileSystemVisitor') -> Any:
        """Accept visitor (Visitor pattern)."""
        pass


class File(FileSystemComponent):
    """Regular file implementation."""
    
    def __init__(self, name: str, content: bytes = b"", owner: str = "root", 
                 permissions: FilePermissions = None):
        self.name = name
        self.content = content
        self.owner = owner
        self.group = "users"
        
        # Generate inode
        self.inode = hash(f"{name}_{time.time()}") & 0x7FFFFFFF
        
        # Set default permissions
        if permissions is None:
            permissions = FilePermissions(
                owner_read=True, owner_write=True, owner_execute=False,
                group_read=True, group_write=False, group_execute=False,
                other_read=True, other_write=False, other_execute=False
            )
        
        # Create metadata
        now = datetime.now()
        self.metadata = FileMetadata(
            inode=self.inode,
            file_type=FileType.REGULAR,
            size=len(content),
            permissions=permissions,
            owner=owner,
            group=self.group,
            created_time=now,
            modified_time=now,
            accessed_time=now,
            checksum=self._calculate_checksum()
        )
        
        self.is_open = False
        self.open_mode = None
        self._lock = threading.RLock()
    
    def read(self, offset: int = 0, length: int = None) -> bytes:
        """Read file content."""
        with self._lock:
            self.metadata.accessed_time = datetime.now()
            
            if length is None:
                return self.content[offset:]
            else:
                return self.content[offset:offset + length]
    
    def write(self, content: bytes, offset: int = 0, append: bool = False) -> int:
        """Write content to file."""
        with self._lock:
            if append:
                self.content += content
            else:
                # Ensure content is long enough
                if offset > len(self.content):
                    self.content += b'\x00' * (offset - len(self.content))
                
                # Write content
                end_pos = offset + len(content)
                if end_pos > len(self.content):
                    self.content = self.content[:offset] + content
                else:
                    self.content = (self.content[:offset] + content + 
                                  self.content[end_pos:])
            
            # Update metadata
            self.metadata.size = len(self.content)
            self.metadata.modified_time = datetime.now()
            self.metadata.checksum = self._calculate_checksum()
            
            return len(content)
    
    def truncate(self, size: int = 0) -> None:
        """Truncate file to specified size."""
        with self._lock:
            if size < len(self.content):
                self.content = self.content[:size]
            elif size > len(self.content):
                self.content += b'\x00' * (size - len(self.content))
            
            self.metadata.size = len(self.content)
            self.metadata.modified_time = datetime.now()
            self.metadata.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate file checksum."""
        return hashlib.md5(self.content).hexdigest()
    
    def get_name(self) -> str:
        return self.name
    
    def get_size(self) -> int:
        return self.metadata.size
    
    def get_metadata(self) -> FileMetadata:
        return self.metadata
    
    def accept(self, visitor: 'FileSystemVisitor') -> Any:
        return visitor.visit_file(self)


class Directory(FileSystemComponent):
    """Directory implementation."""
    
    def __init__(self, name: str, owner: str = "root", 
                 permissions: FilePermissions = None):
        self.name = name
        self.owner = owner
        self.group = "users"
        self.children: Dict[str, FileSystemComponent] = {}
        
        # Generate inode
        self.inode = hash(f"dir_{name}_{time.time()}") & 0x7FFFFFFF
        
        # Set default permissions
        if permissions is None:
            permissions = FilePermissions(
                owner_read=True, owner_write=True, owner_execute=True,
                group_read=True, group_write=False, group_execute=True,
                other_read=True, other_write=False, other_execute=True
            )
        
        # Create metadata
        now = datetime.now()
        self.metadata = FileMetadata(
            inode=self.inode,
            file_type=FileType.DIRECTORY,
            size=0,  # Will be calculated
            permissions=permissions,
            owner=owner,
            group=self.group,
            created_time=now,
            modified_time=now,
            accessed_time=now
        )
        
        self._lock = threading.RLock()
    
    def add_child(self, child: FileSystemComponent) -> bool:
        """Add child component."""
        with self._lock:
            if child.get_name() in self.children:
                return False
            
            self.children[child.get_name()] = child
            self.metadata.modified_time = datetime.now()
            self._update_size()
            return True
    
    def remove_child(self, name: str) -> bool:
        """Remove child component."""
        with self._lock:
            if name not in self.children:
                return False
            
            del self.children[name]
            self.metadata.modified_time = datetime.now()
            self._update_size()
            return True
    
    def get_child(self, name: str) -> Optional[FileSystemComponent]:
        """Get child component by name."""
        with self._lock:
            self.metadata.accessed_time = datetime.now()
            return self.children.get(name)
    
    def list_children(self) -> List[str]:
        """List all child names."""
        with self._lock:
            self.metadata.accessed_time = datetime.now()
            return list(self.children.keys())
    
    def _update_size(self) -> None:
        """Update directory size based on children."""
        total_size = 0
        for child in self.children.values():
            total_size += child.get_size()
        self.metadata.size = total_size
    
    def get_name(self) -> str:
        return self.name
    
    def get_size(self) -> int:
        return self.metadata.size
    
    def get_metadata(self) -> FileMetadata:
        return self.metadata
    
    def accept(self, visitor: 'FileSystemVisitor') -> Any:
        return visitor.visit_directory(self)


class SymbolicLink(FileSystemComponent):
    """Symbolic link implementation."""
    
    def __init__(self, name: str, target_path: str, owner: str = "root"):
        self.name = name
        self.target_path = target_path
        self.owner = owner
        self.group = "users"
        
        # Generate inode
        self.inode = hash(f"symlink_{name}_{time.time()}") & 0x7FFFFFFF
        
        # Create metadata
        now = datetime.now()
        self.metadata = FileMetadata(
            inode=self.inode,
            file_type=FileType.SYMLINK,
            size=len(target_path),
            permissions=FilePermissions(
                owner_read=True, owner_write=True, owner_execute=True,
                group_read=True, group_write=False, group_execute=True,
                other_read=True, other_write=False, other_execute=True
            ),
            owner=owner,
            group=self.group,
            created_time=now,
            modified_time=now,
            accessed_time=now
        )
    
    def get_target(self) -> str:
        """Get symlink target path."""
        self.metadata.accessed_time = datetime.now()
        return self.target_path
    
    def get_name(self) -> str:
        return self.name
    
    def get_size(self) -> int:
        return self.metadata.size
    
    def get_metadata(self) -> FileMetadata:
        return self.metadata
    
    def accept(self, visitor: 'FileSystemVisitor') -> Any:
        return visitor.visit_symlink(self)


# ============================================================================
# VISITOR PATTERN FOR FILE SYSTEM OPERATIONS
# ============================================================================

class FileSystemVisitor(ABC):
    """Abstract visitor for file system operations."""
    
    @abstractmethod
    def visit_file(self, file: File) -> Any:
        """Visit regular file."""
        pass
    
    @abstractmethod
    def visit_directory(self, directory: Directory) -> Any:
        """Visit directory."""
        pass
    
    @abstractmethod
    def visit_symlink(self, symlink: SymbolicLink) -> Any:
        """Visit symbolic link."""
        pass


class SizeCalculatorVisitor(FileSystemVisitor):
    """Visitor to calculate total size."""
    
    def __init__(self):
        self.total_size = 0
    
    def visit_file(self, file: File) -> int:
        self.total_size += file.get_size()
        return file.get_size()
    
    def visit_directory(self, directory: Directory) -> int:
        dir_size = 0
        for child in directory.children.values():
            dir_size += child.accept(self)
        return dir_size
    
    def visit_symlink(self, symlink: SymbolicLink) -> int:
        self.total_size += symlink.get_size()
        return symlink.get_size()


class FileCountVisitor(FileSystemVisitor):
    """Visitor to count files and directories."""
    
    def __init__(self):
        self.file_count = 0
        self.directory_count = 0
        self.symlink_count = 0
    
    def visit_file(self, file: File) -> None:
        self.file_count += 1
    
    def visit_directory(self, directory: Directory) -> None:
        self.directory_count += 1
        for child in directory.children.values():
            child.accept(self)
    
    def visit_symlink(self, symlink: SymbolicLink) -> None:
        self.symlink_count += 1


class SearchVisitor(FileSystemVisitor):
    """Visitor to search for files/directories."""
    
    def __init__(self, pattern: str, case_sensitive: bool = False):
        self.pattern = pattern.lower() if not case_sensitive else pattern
        self.case_sensitive = case_sensitive
        self.results: List[FileSystemComponent] = []
    
    def visit_file(self, file: File) -> None:
        name = file.get_name()
        if not self.case_sensitive:
            name = name.lower()
        
        if self.pattern in name:
            self.results.append(file)
    
    def visit_directory(self, directory: Directory) -> None:
        name = directory.get_name()
        if not self.case_sensitive:
            name = name.lower()
        
        if self.pattern in name:
            self.results.append(directory)
        
        # Continue searching in children
        for child in directory.children.values():
            child.accept(self)
    
    def visit_symlink(self, symlink: SymbolicLink) -> None:
        name = symlink.get_name()
        if not self.case_sensitive:
            name = name.lower()
        
        if self.pattern in name:
            self.results.append(symlink)


# ============================================================================
# FILE SYSTEM OPERATIONS
# ============================================================================

class FileSystemOperation(ABC):
    """Abstract file system operation (Command pattern)."""
    
    @abstractmethod
    def execute(self) -> bool:
        """Execute the operation."""
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        """Undo the operation."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get operation description."""
        pass


class CreateFileOperation(FileSystemOperation):
    """Create file operation."""
    
    def __init__(self, file_system: 'FileSystem', path: str, content: bytes = b""):
        self.file_system = file_system
        self.path = path
        self.content = content
        self.executed = False
    
    def execute(self) -> bool:
        """Execute file creation."""
        success = self.file_system.create_file(self.path, self.content)
        self.executed = success
        return success
    
    def undo(self) -> bool:
        """Undo file creation."""
        if not self.executed:
            return False
        return self.file_system.delete(self.path)
    
    def get_description(self) -> str:
        return f"Create file: {self.path}"


class DeleteOperation(FileSystemOperation):
    """Delete file/directory operation."""
    
    def __init__(self, file_system: 'FileSystem', path: str):
        self.file_system = file_system
        self.path = path
        self.backup_data = None
        self.executed = False
    
    def execute(self) -> bool:
        """Execute deletion."""
        # Backup data for undo
        component = self.file_system.get_component(self.path)
        if component:
            self.backup_data = self._backup_component(component)
        
        success = self.file_system.delete(self.path)
        self.executed = success
        return success
    
    def undo(self) -> bool:
        """Undo deletion."""
        if not self.executed or not self.backup_data:
            return False
        
        # Restore from backup
        return self._restore_component(self.backup_data)
    
    def _backup_component(self, component: FileSystemComponent) -> Dict[str, Any]:
        """Backup component data."""
        if isinstance(component, File):
            return {
                'type': 'file',
                'name': component.name,
                'content': component.content,
                'metadata': component.metadata
            }
        elif isinstance(component, Directory):
            children_backup = {}
            for name, child in component.children.items():
                children_backup[name] = self._backup_component(child)
            
            return {
                'type': 'directory',
                'name': component.name,
                'metadata': component.metadata,
                'children': children_backup
            }
        elif isinstance(component, SymbolicLink):
            return {
                'type': 'symlink',
                'name': component.name,
                'target': component.target_path,
                'metadata': component.metadata
            }
        
        return {}
    
    def _restore_component(self, backup: Dict[str, Any]) -> bool:
        """Restore component from backup."""
        # This is a simplified restoration - in a real system,
        # you'd need to handle parent directory creation, etc.
        return True
    
    def get_description(self) -> str:
        return f"Delete: {self.path}"


# ============================================================================
# FILE SYSTEM CORE
# ============================================================================

class FileSystem:
    """Main file system implementation."""
    
    def __init__(self, name: str = "MyFileSystem"):
        self.name = name
        self.root = Directory("", "root")
        self.current_directory = self.root
        
        # File system state
        self.mounted = True
        self.read_only = False
        
        # Inode tracking
        self.next_inode = 1
        self.inode_table: Dict[int, FileSystemComponent] = {}
        
        # Operation history for undo/redo
        self.operation_history: List[FileSystemOperation] = []
        self.history_index = -1
        
        # Locking
        self._global_lock = threading.RLock()
        
        # Event observers
        self.observers: List['FileSystemObserver'] = []
        
        # Statistics
        self.stats = {
            'files_created': 0,
            'files_deleted': 0,
            'directories_created': 0,
            'directories_deleted': 0,
            'bytes_written': 0,
            'bytes_read': 0
        }
        
        print(f"📁 File System '{name}' initialized")
    
    def add_observer(self, observer: 'FileSystemObserver') -> None:
        """Add file system observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: 'FileSystemObserver') -> None:
        """Remove file system observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, event: FileSystemEvent, path: str, details: Dict[str, Any] = None) -> None:
        """Notify observers of file system events."""
        for observer in self.observers:
            observer.on_file_system_event(event, path, details or {})
    
    def create_file(self, path: str, content: bytes = b"", permissions: FilePermissions = None) -> bool:
        """Create a new file."""
        with self._global_lock:
            if self.read_only:
                return False
            
            # Parse path
            directory_path, filename = self._split_path(path)
            
            # Get parent directory
            parent_dir = self._get_directory(directory_path)
            if not parent_dir:
                return False
            
            # Check if file already exists
            if parent_dir.get_child(filename):
                return False
            
            # Create file
            file = File(filename, content, "user", permissions)
            
            if parent_dir.add_child(file):
                self.inode_table[file.inode] = file
                self.stats['files_created'] += 1
                self.stats['bytes_written'] += len(content)
                
                # Notify observers
                self.notify_observers(FileSystemEvent.FILE_CREATED, path, {
                    'size': len(content),
                    'permissions': file.metadata.permissions.to_octal()
                })
                
                return True
            
            return False
    
    def create_directory(self, path: str, permissions: FilePermissions = None) -> bool:
        """Create a new directory."""
        with self._global_lock:
            if self.read_only:
                return False
            
            # Parse path
            parent_path, dirname = self._split_path(path)
            
            # Get parent directory
            parent_dir = self._get_directory(parent_path)
            if not parent_dir:
                return False
            
            # Check if directory already exists
            if parent_dir.get_child(dirname):
                return False
            
            # Create directory
            directory = Directory(dirname, "user", permissions)
            
            if parent_dir.add_child(directory):
                self.inode_table[directory.inode] = directory
                self.stats['directories_created'] += 1
                
                # Notify observers
                self.notify_observers(FileSystemEvent.DIRECTORY_CREATED, path, {
                    'permissions': directory.metadata.permissions.to_octal()
                })
                
                return True
            
            return False
    
    def create_symlink(self, link_path: str, target_path: str) -> bool:
        """Create a symbolic link."""
        with self._global_lock:
            if self.read_only:
                return False
            
            # Parse path
            directory_path, link_name = self._split_path(link_path)
            
            # Get parent directory
            parent_dir = self._get_directory(directory_path)
            if not parent_dir:
                return False
            
            # Check if link already exists
            if parent_dir.get_child(link_name):
                return False
            
            # Create symlink
            symlink = SymbolicLink(link_name, target_path, "user")
            
            if parent_dir.add_child(symlink):
                self.inode_table[symlink.inode] = symlink
                return True
            
            return False
    
    def delete(self, path: str, recursive: bool = False) -> bool:
        """Delete file or directory."""
        with self._global_lock:
            if self.read_only:
                return False
            
            # Parse path
            parent_path, name = self._split_path(path)
            
            # Get parent directory
            parent_dir = self._get_directory(parent_path)
            if not parent_dir:
                return False
            
            # Get component to delete
            component = parent_dir.get_child(name)
            if not component:
                return False
            
            # Check if it's a directory and handle recursion
            if isinstance(component, Directory):
                if component.children and not recursive:
                    return False  # Directory not empty
                
                # Remove from inode table recursively
                self._remove_from_inode_table(component)
                self.stats['directories_deleted'] += 1
                
                # Notify observers
                self.notify_observers(FileSystemEvent.DIRECTORY_DELETED, path)
            else:
                # Remove from inode table
                if component.metadata.inode in self.inode_table:
                    del self.inode_table[component.metadata.inode]
                
                self.stats['files_deleted'] += 1
                
                # Notify observers
                self.notify_observers(FileSystemEvent.FILE_DELETED, path)
            
            # Remove from parent
            return parent_dir.remove_child(name)
    
    def _remove_from_inode_table(self, component: FileSystemComponent) -> None:
        """Recursively remove component and children from inode table."""
        if component.metadata.inode in self.inode_table:
            del self.inode_table[component.metadata.inode]
        
        if isinstance(component, Directory):
            for child in component.children.values():
                self._remove_from_inode_table(child)
    
    def read_file(self, path: str, offset: int = 0, length: int = None) -> Optional[bytes]:
        """Read file content."""
        with self._global_lock:
            component = self.get_component(path)
            
            if not isinstance(component, File):
                return None
            
            content = component.read(offset, length)
            self.stats['bytes_read'] += len(content)
            
            return content
    
    def write_file(self, path: str, content: bytes, offset: int = 0, append: bool = False) -> bool:
        """Write content to file."""
        with self._global_lock:
            if self.read_only:
                return False
            
            component = self.get_component(path)
            
            if not isinstance(component, File):
                return False
            
            bytes_written = component.write(content, offset, append)
            self.stats['bytes_written'] += bytes_written
            
            # Notify observers
            self.notify_observers(FileSystemEvent.FILE_MODIFIED, path, {
                'bytes_written': bytes_written,
                'new_size': component.get_size()
            })
            
            return True
    
    def copy(self, source_path: str, dest_path: str) -> bool:
        """Copy file or directory."""
        with self._global_lock:
            if self.read_only:
                return False
            
            source = self.get_component(source_path)
            if not source:
                return False
            
            return self._copy_component(source, dest_path)
    
    def _copy_component(self, source: FileSystemComponent, dest_path: str) -> bool:
        """Copy a component to destination path."""
        if isinstance(source, File):
            # Copy file
            success = self.create_file(dest_path, source.content, source.metadata.permissions)
            if success:
                self.notify_observers(FileSystemEvent.FILE_COPIED, dest_path, {
                    'source': source.name,
                    'size': source.get_size()
                })
            return success
        
        elif isinstance(source, Directory):
            # Copy directory
            if not self.create_directory(dest_path, source.metadata.permissions):
                return False
            
            # Copy children
            for child_name, child in source.children.items():
                child_dest_path = f"{dest_path}/{child_name}"
                if not self._copy_component(child, child_dest_path):
                    return False
            
            return True
        
        elif isinstance(source, SymbolicLink):
            # Copy symlink
            return self.create_symlink(dest_path, source.target_path)
        
        return False
    
    def move(self, source_path: str, dest_path: str) -> bool:
        """Move file or directory."""
        with self._global_lock:
            if self.read_only:
                return False
            
            # Copy then delete
            if self.copy(source_path, dest_path):
                if self.delete(source_path, recursive=True):
                    self.notify_observers(FileSystemEvent.FILE_MOVED, dest_path, {
                        'source': source_path
                    })
                    return True
                else:
                    # Rollback copy if delete failed
                    self.delete(dest_path, recursive=True)
            
            return False
    
    def list_directory(self, path: str = None) -> Optional[List[Dict[str, Any]]]:
        """List directory contents."""
        with self._global_lock:
            if path is None:
                directory = self.current_directory
            else:
                directory = self._get_directory(path)
            
            if not directory:
                return None
            
            result = []
            for name in directory.list_children():
                child = directory.get_child(name)
                if child:
                    result.append({
                        'name': name,
                        'type': child.metadata.file_type.value,
                        'size': child.get_size(),
                        'permissions': child.metadata.permissions.to_string(),
                        'owner': child.metadata.owner,
                        'group': child.metadata.group,
                        'modified': child.metadata.modified_time.isoformat(),
                        'inode': child.metadata.inode
                    })
            
            return result
    
    def get_component(self, path: str) -> Optional[FileSystemComponent]:
        """Get file system component by path."""
        if path == "/" or path == "":
            return self.root
        
        # Split path into components
        path_parts = [p for p in path.split("/") if p]
        
        current = self.root
        for part in path_parts:
            if not isinstance(current, Directory):
                return None
            
            current = current.get_child(part)
            if not current:
                return None
        
        return current
    
    def _get_directory(self, path: str) -> Optional[Directory]:
        """Get directory by path."""
        component = self.get_component(path)
        return component if isinstance(component, Directory) else None
    
    def _split_path(self, path: str) -> Tuple[str, str]:
        """Split path into parent directory and filename."""
        if "/" not in path:
            return "", path
        
        parts = path.rsplit("/", 1)
        return parts[0] if parts[0] else "/", parts[1]
    
    def change_directory(self, path: str) -> bool:
        """Change current directory."""
        directory = self._get_directory(path)
        if directory:
            self.current_directory = directory
            return True
        return False
    
    def get_current_path(self) -> str:
        """Get current directory path."""
        if self.current_directory == self.root:
            return "/"
        
        # Build path from root
        path_parts = []
        current = self.current_directory
        
        while current != self.root and current is not None:
            path_parts.append(current.name)
            # In a real implementation, you'd track parent relationships
            break  # Simplified for demo
        
        if path_parts:
            return "/" + "/".join(reversed(path_parts))
        return "/"
    
    def search(self, pattern: str, start_path: str = "/", case_sensitive: bool = False) -> List[str]:
        """Search for files and directories."""
        start_component = self.get_component(start_path)
        if not start_component:
            return []
        
        visitor = SearchVisitor(pattern, case_sensitive)
        start_component.accept(visitor)
        
        # Convert results to paths
        results = []
        for component in visitor.results:
            # In a real implementation, you'd build full paths
            results.append(component.name)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file system statistics."""
        # Calculate total size and counts
        size_visitor = SizeCalculatorVisitor()
        count_visitor = FileCountVisitor()
        
        self.root.accept(size_visitor)
        self.root.accept(count_visitor)
        
        return {
            'name': self.name,
            'mounted': self.mounted,
            'read_only': self.read_only,
            'total_size': size_visitor.total_size,
            'file_count': count_visitor.file_count,
            'directory_count': count_visitor.directory_count,
            'symlink_count': count_visitor.symlink_count,
            'inode_count': len(self.inode_table),
            'operations': self.stats.copy()
        }
    
    def execute_operation(self, operation: FileSystemOperation) -> bool:
        """Execute operation and add to history."""
        with self._global_lock:
            success = operation.execute()
            
            if success:
                # Add to history (remove any operations after current index)
                self.operation_history = self.operation_history[:self.history_index + 1]
                self.operation_history.append(operation)
                self.history_index += 1
                
                # Limit history size
                if len(self.operation_history) > 100:
                    self.operation_history.pop(0)
                    self.history_index -= 1
            
            return success
    
    def undo(self) -> bool:
        """Undo last operation."""
        with self._global_lock:
            if self.history_index < 0:
                return False
            
            operation = self.operation_history[self.history_index]
            success = operation.undo()
            
            if success:
                self.history_index -= 1
            
            return success
    
    def redo(self) -> bool:
        """Redo next operation."""
        with self._global_lock:
            if self.history_index >= len(self.operation_history) - 1:
                return False
            
            self.history_index += 1
            operation = self.operation_history[self.history_index]
            
            return operation.execute()


# ============================================================================
# OBSERVER PATTERN FOR FILE SYSTEM EVENTS
# ============================================================================

class FileSystemObserver(ABC):
    """Abstract file system observer."""
    
    @abstractmethod
    def on_file_system_event(self, event: FileSystemEvent, path: str, details: Dict[str, Any]) -> None:
        """Handle file system event."""
        pass


class FileSystemLogger(FileSystemObserver):
    """File system event logger."""
    
    def __init__(self):
        self.event_log: List[Dict[str, Any]] = []
    
    def on_file_system_event(self, event: FileSystemEvent, path: str, details: Dict[str, Any]) -> None:
        """Log file system event."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event.value,
            'path': path,
            'details': details
        }
        
        self.event_log.append(log_entry)
        print(f"📝 FS Event: {event.value} - {path}")


class FileSystemMonitor(FileSystemObserver):
    """File system monitoring and alerting."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'max_files_per_directory': 1000
        }
    
    def on_file_system_event(self, event: FileSystemEvent, path: str, details: Dict[str, Any]) -> None:
        """Monitor file system events for alerts."""
        if event == FileSystemEvent.FILE_CREATED:
            file_size = details.get('size', 0)
            if file_size > self.thresholds['max_file_size']:
                self._create_alert(f"Large file created: {path} ({file_size} bytes)")
        
        elif event == FileSystemEvent.DIRECTORY_CREATED:
            # In a real implementation, you'd check directory file count
            pass
    
    def _create_alert(self, message: str) -> None:
        """Create system alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        print(f"⚠️  Alert: {message}")


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_file_system():
    """Demonstrate the file system design."""
    print("=== FILE SYSTEM DESIGN DEMONSTRATION ===\n")
    
    # Create file system
    print("1. FILE SYSTEM CREATION:")
    
    fs = FileSystem("DemoFS")
    logger = FileSystemLogger()
    monitor = FileSystemMonitor()
    
    fs.add_observer(logger)
    fs.add_observer(monitor)
    
    print(f"   ✓ Created file system: {fs.name}")
    print(f"   ✓ Added logger and monitor observers")
    print()
    
    # Create directory structure
    print("2. DIRECTORY STRUCTURE CREATION:")
    
    directories = [
        "/home",
        "/home/user",
        "/home/user/documents",
        "/home/user/pictures",
        "/var",
        "/var/log",
        "/tmp"
    ]
    
    for directory in directories:
        success = fs.create_directory(directory)
        print(f"   {'✓' if success else '✗'} Created directory: {directory}")
    
    print()
    
    # Create files
    print("3. FILE CREATION:")
    
    files = [
        ("/home/user/readme.txt", b"Welcome to the file system demo!"),
        ("/home/user/documents/report.txt", b"This is a sample report.\nIt has multiple lines.\nAnd some data."),
        ("/home/user/pictures/photo1.jpg", b"JPEG_FAKE_DATA_" * 100),
        ("/var/log/system.log", b"[INFO] System started\n[INFO] File system mounted\n"),
        ("/tmp/temp_file.tmp", b"Temporary data")
    ]
    
    for file_path, content in files:
        success = fs.create_file(file_path, content)
        print(f"   {'✓' if success else '✗'} Created file: {file_path} ({len(content)} bytes)")
    
    print()
    
    # Create symbolic links
    print("4. SYMBOLIC LINK CREATION:")
    
    symlinks = [
        ("/home/user/documents/report_link.txt", "/home/user/documents/report.txt"),
        ("/home/user/logs", "/var/log")
    ]
    
    for link_path, target_path in symlinks:
        success = fs.create_symlink(link_path, target_path)
        print(f"   {'✓' if success else '✗'} Created symlink: {link_path} -> {target_path}")
    
    print()
    
    # List directory contents
    print("5. DIRECTORY LISTING:")
    
    directories_to_list = ["/", "/home/user", "/home/user/documents"]
    
    for directory in directories_to_list:
        print(f"   Contents of {directory}:")
        contents = fs.list_directory(directory)
        
        if contents:
            for item in contents:
                type_char = "d" if item['type'] == 'directory' else "l" if item['type'] == 'symlink' else "-"
                print(f"     {type_char}{item['permissions']} {item['owner']:8} {item['group']:8} "
                      f"{item['size']:8} {item['modified'][:19]} {item['name']}")
        else:
            print("     (empty or not found)")
        print()
    
    # File operations
    print("6. FILE OPERATIONS:")
    
    # Read file
    content = fs.read_file("/home/user/readme.txt")
    print(f"   Read readme.txt: {content.decode() if content else 'Failed'}")
    
    # Write to file
    success = fs.write_file("/home/user/readme.txt", b"\nThis line was appended!", append=True)
    print(f"   {'✓' if success else '✗'} Appended to readme.txt")
    
    # Read updated file
    content = fs.read_file("/home/user/readme.txt")
    print(f"   Updated content: {content.decode() if content else 'Failed'}")
    
    print()
    
    # Copy operations
    print("7. COPY OPERATIONS:")
    
    # Copy file
    success = fs.copy("/home/user/readme.txt", "/tmp/readme_copy.txt")
    print(f"   {'✓' if success else '✗'} Copied readme.txt to /tmp/")
    
    # Copy directory
    success = fs.copy("/home/user/documents", "/tmp/documents_backup")
    print(f"   {'✓' if success else '✗'} Copied documents directory to /tmp/")
    
    print()
    
    # Move operations
    print("8. MOVE OPERATIONS:")
    
    # Move file
    success = fs.move("/tmp/temp_file.tmp", "/home/user/moved_temp.tmp")
    print(f"   {'✓' if success else '✗'} Moved temp_file.tmp to /home/user/")
    
    print()
    
    # Search operations
    print("9. SEARCH OPERATIONS:")
    
    search_patterns = ["txt", "log", "report"]
    
    for pattern in search_patterns:
        results = fs.search(pattern, "/", case_sensitive=False)
        print(f"   Search for '{pattern}': {len(results)} results")
        for result in results[:3]:  # Show first 3 results
            print(f"     - {result}")
        if len(results) > 3:
            print(f"     ... and {len(results) - 3} more")
    
    print()
    
    # File system statistics
    print("10. FILE SYSTEM STATISTICS:")
    
    stats = fs.get_statistics()
    print(f"   File System: {stats['name']}")
    print(f"   Mounted: {stats['mounted']}")
    print(f"   Read Only: {stats['read_only']}")
    print(f"   Total Size: {stats['total_size']} bytes")
    print(f"   Files: {stats['file_count']}")
    print(f"   Directories: {stats['directory_count']}")
    print(f"   Symbolic Links: {stats['symlink_count']}")
    print(f"   Inodes: {stats['inode_count']}")
    
    print("\n   Operations:")
    for op_name, count in stats['operations'].items():
        print(f"     {op_name}: {count}")
    
    print()
    
    # Test operations with undo/redo
    print("11. UNDO/REDO OPERATIONS:")
    
    # Create operation
    create_op = CreateFileOperation(fs, "/home/user/test_undo.txt", b"Test content for undo")
    success = fs.execute_operation(create_op)
    print(f"   {'✓' if success else '✗'} Created test file with operation")
    
    # Verify file exists
    content = fs.read_file("/home/user/test_undo.txt")
    print(f"   File content: {content.decode() if content else 'Not found'}")
    
    # Undo creation
    success = fs.undo()
    print(f"   {'✓' if success else '✗'} Undid file creation")
    
    # Verify file is gone
    content = fs.read_file("/home/user/test_undo.txt")
    print(f"   File after undo: {'Found' if content else 'Not found'}")
    
    # Redo creation
    success = fs.redo()
    print(f"   {'✓' if success else '✗'} Redid file creation")
    
    # Verify file is back
    content = fs.read_file("/home/user/test_undo.txt")
    print(f"   File after redo: {content.decode() if content else 'Not found'}")
    
    print()
    
    # Delete operations
    print("12. DELETE OPERATIONS:")
    
    # Delete file
    success = fs.delete("/home/user/moved_temp.tmp")
    print(f"   {'✓' if success else '✗'} Deleted moved_temp.tmp")
    
    # Try to delete non-empty directory (should fail)
    success = fs.delete("/home/user")
    print(f"   {'✓' if success else '✗'} Tried to delete non-empty directory (should fail)")
    
    # Delete directory recursively
    success = fs.delete("/tmp/documents_backup", recursive=True)
    print(f"   {'✓' if success else '✗'} Deleted documents_backup recursively")
    
    print()
    
    # Show event log
    print("13. EVENT LOG:")
    
    print(f"   Total events logged: {len(logger.event_log)}")
    
    event_counts = {}
    for event in logger.event_log:
        event_type = event['event']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    for event_type, count in event_counts.items():
        print(f"     {event_type}: {count}")
    
    print("\n   Recent events:")
    for event in logger.event_log[-5:]:  # Show last 5 events
        print(f"     {event['timestamp'][:19]} - {event['event']} - {event['path']}")
    
    print()
    
    # Show alerts
    print("14. SYSTEM ALERTS:")
    
    if monitor.alerts:
        print(f"   Total alerts: {len(monitor.alerts)}")
        for alert in monitor.alerts:
            print(f"     {alert['timestamp'][:19]} - {alert['severity'].upper()}: {alert['message']}")
    else:
        print("   No alerts generated")
    
    print()
    
    # Final file system state
    print("15. FINAL FILE SYSTEM STATE:")
    
    final_stats = fs.get_statistics()
    print(f"   Final file count: {final_stats['file_count']}")
    print(f"   Final directory count: {final_stats['directory_count']}")
    print(f"   Final total size: {final_stats['total_size']} bytes")
    print(f"   Total bytes written: {final_stats['operations']['bytes_written']}")
    print(f"   Total bytes read: {final_stats['operations']['bytes_read']}")
    
    print()
    print("=== FILE SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_file_system()
