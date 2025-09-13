"""
ITERATOR PATTERN - Behavioral Design Pattern
============================================

Problem Statement:
Implement the Iterator pattern to provide a way to access elements of a
collection sequentially without exposing the underlying representation:
- Sequential access to collection elements
- Multiple traversal algorithms for same collection
- Uniform interface for different collection types
- Support for concurrent iteration
- Lazy evaluation and memory-efficient traversal

Learning Objectives:
- Understand Iterator vs Visitor pattern differences
- Implement custom iterators for complex data structures
- Design iterator hierarchies and composition
- Handle concurrent iteration and thread safety
- Create memory-efficient and lazy evaluation systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Generic, TypeVar, Iterator as TypingIterator
import threading
import time
from datetime import datetime
from enum import Enum
import random


# ============================================================================
# ITERATOR INTERFACE
# ============================================================================

T = TypeVar('T')

class Iterator(ABC, Generic[T]):
    """Abstract iterator interface."""
    
    @abstractmethod
    def has_next(self) -> bool:
        """Check if there are more elements."""
        pass
    
    @abstractmethod
    def next(self) -> T:
        """Get next element."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset iterator to beginning."""
        pass
    
    def get_iterator_info(self) -> Dict[str, Any]:
        """Get iterator information."""
        return {
            'type': self.__class__.__name__,
            'has_next': self.has_next(),
            'supports_reset': hasattr(self, 'reset')
        }


class Iterable(ABC, Generic[T]):
    """Abstract iterable interface."""
    
    @abstractmethod
    def create_iterator(self) -> Iterator[T]:
        """Create iterator for this collection."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get collection size."""
        pass
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        return {
            'type': self.__class__.__name__,
            'size': self.size(),
            'supports_multiple_iterators': True
        }


# ============================================================================
# CUSTOM COLLECTION WITH MULTIPLE ITERATORS
# ============================================================================

class Book:
    """Book data class."""
    
    def __init__(self, title: str, author: str, year: int, genre: str, pages: int):
        self.title = title
        self.author = author
        self.year = year
        self.genre = genre
        self.pages = pages
    
    def __str__(self) -> str:
        return f"'{self.title}' by {self.author} ({self.year})"
    
    def __repr__(self) -> str:
        return f"Book('{self.title}', '{self.author}', {self.year}, '{self.genre}', {self.pages})"
    
    def get_info(self) -> Dict[str, Any]:
        """Get book information."""
        return {
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'genre': self.genre,
            'pages': self.pages
        }


class BookCollection(Iterable[Book]):
    """Collection of books with multiple iteration strategies."""
    
    def __init__(self):
        self._books: List[Book] = []
        self._index_by_title: Dict[str, int] = {}
        self._index_by_author: Dict[str, List[int]] = {}
        self._index_by_genre: Dict[str, List[int]] = {}
    
    def add_book(self, book: Book) -> None:
        """Add book to collection."""
        index = len(self._books)
        self._books.append(book)
        
        # Update indices
        self._index_by_title[book.title.lower()] = index
        
        if book.author not in self._index_by_author:
            self._index_by_author[book.author] = []
        self._index_by_author[book.author].append(index)
        
        if book.genre not in self._index_by_genre:
            self._index_by_genre[book.genre] = []
        self._index_by_genre[book.genre].append(index)
        
        print(f"Added book: {book}")
    
    def remove_book(self, title: str) -> bool:
        """Remove book by title."""
        title_lower = title.lower()
        if title_lower in self._index_by_title:
            index = self._index_by_title[title_lower]
            book = self._books[index]
            
            # Mark as None instead of removing to maintain indices
            self._books[index] = None
            del self._index_by_title[title_lower]
            
            # Update author index
            if book.author in self._index_by_author:
                self._index_by_author[book.author].remove(index)
                if not self._index_by_author[book.author]:
                    del self._index_by_author[book.author]
            
            # Update genre index
            if book.genre in self._index_by_genre:
                self._index_by_genre[book.genre].remove(index)
                if not self._index_by_genre[book.genre]:
                    del self._index_by_genre[book.genre]
            
            print(f"Removed book: {book}")
            return True
        
        return False
    
    def create_iterator(self) -> Iterator[Book]:
        """Create default iterator (sequential)."""
        return SequentialBookIterator(self)
    
    def create_author_iterator(self, author: str) -> Iterator[Book]:
        """Create iterator for books by specific author."""
        return AuthorBookIterator(self, author)
    
    def create_genre_iterator(self, genre: str) -> Iterator[Book]:
        """Create iterator for books of specific genre."""
        return GenreBookIterator(self, genre)
    
    def create_year_range_iterator(self, start_year: int, end_year: int) -> Iterator[Book]:
        """Create iterator for books within year range."""
        return YearRangeBookIterator(self, start_year, end_year)
    
    def create_random_iterator(self, seed: int = None) -> Iterator[Book]:
        """Create iterator that returns books in random order."""
        return RandomBookIterator(self, seed)
    
    def create_sorted_iterator(self, key_func: callable, reverse: bool = False) -> Iterator[Book]:
        """Create iterator that returns books in sorted order."""
        return SortedBookIterator(self, key_func, reverse)
    
    def size(self) -> int:
        """Get number of books (excluding removed ones)."""
        return sum(1 for book in self._books if book is not None)
    
    def get_book_at(self, index: int) -> Optional[Book]:
        """Get book at specific index."""
        if 0 <= index < len(self._books):
            return self._books[index]
        return None
    
    def get_all_books(self) -> List[Book]:
        """Get all non-null books."""
        return [book for book in self._books if book is not None]
    
    def get_authors(self) -> List[str]:
        """Get all authors."""
        return list(self._index_by_author.keys())
    
    def get_genres(self) -> List[str]:
        """Get all genres."""
        return list(self._index_by_genre.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        books = self.get_all_books()
        
        if not books:
            return {'total_books': 0}
        
        return {
            'total_books': len(books),
            'authors': len(self._index_by_author),
            'genres': len(self._index_by_genre),
            'year_range': (min(book.year for book in books), max(book.year for book in books)),
            'average_pages': sum(book.pages for book in books) / len(books),
            'total_pages': sum(book.pages for book in books)
        }


# ============================================================================
# CONCRETE ITERATORS
# ============================================================================

class SequentialBookIterator(Iterator[Book]):
    """Sequential iterator for book collection."""
    
    def __init__(self, collection: BookCollection):
        self._collection = collection
        self._current_index = 0
        self._start_time = time.time()
        self._items_returned = 0
    
    def has_next(self) -> bool:
        """Check if there are more books."""
        while (self._current_index < len(self._collection._books) and 
               self._collection._books[self._current_index] is None):
            self._current_index += 1
        
        return self._current_index < len(self._collection._books)
    
    def next(self) -> Book:
        """Get next book."""
        if not self.has_next():
            raise StopIteration("No more books")
        
        book = self._collection._books[self._current_index]
        self._current_index += 1
        self._items_returned += 1
        
        return book
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_index = 0
        self._items_returned = 0
        print("Sequential iterator reset")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get iteration progress."""
        return {
            'current_index': self._current_index,
            'items_returned': self._items_returned,
            'elapsed_time': time.time() - self._start_time,
            'completion_percentage': (self._items_returned / self._collection.size()) * 100 if self._collection.size() > 0 else 0
        }


class AuthorBookIterator(Iterator[Book]):
    """Iterator for books by specific author."""
    
    def __init__(self, collection: BookCollection, author: str):
        self._collection = collection
        self._author = author
        self._book_indices = collection._index_by_author.get(author, []).copy()
        self._current_position = 0
    
    def has_next(self) -> bool:
        """Check if there are more books by this author."""
        return self._current_position < len(self._book_indices)
    
    def next(self) -> Book:
        """Get next book by this author."""
        if not self.has_next():
            raise StopIteration(f"No more books by {self._author}")
        
        book_index = self._book_indices[self._current_position]
        book = self._collection._books[book_index]
        self._current_position += 1
        
        if book is None:
            # Book was removed, skip to next
            return self.next()
        
        return book
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_position = 0
        print(f"Author iterator for '{self._author}' reset")
    
    def get_author(self) -> str:
        """Get author being iterated."""
        return self._author


class GenreBookIterator(Iterator[Book]):
    """Iterator for books of specific genre."""
    
    def __init__(self, collection: BookCollection, genre: str):
        self._collection = collection
        self._genre = genre
        self._book_indices = collection._index_by_genre.get(genre, []).copy()
        self._current_position = 0
    
    def has_next(self) -> bool:
        """Check if there are more books of this genre."""
        return self._current_position < len(self._book_indices)
    
    def next(self) -> Book:
        """Get next book of this genre."""
        if not self.has_next():
            raise StopIteration(f"No more books in genre '{self._genre}'")
        
        book_index = self._book_indices[self._current_position]
        book = self._collection._books[book_index]
        self._current_position += 1
        
        if book is None:
            # Book was removed, skip to next
            return self.next()
        
        return book
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_position = 0
        print(f"Genre iterator for '{self._genre}' reset")
    
    def get_genre(self) -> str:
        """Get genre being iterated."""
        return self._genre


class YearRangeBookIterator(Iterator[Book]):
    """Iterator for books within a year range."""
    
    def __init__(self, collection: BookCollection, start_year: int, end_year: int):
        self._collection = collection
        self._start_year = start_year
        self._end_year = end_year
        self._current_index = 0
        self._filtered_books = self._filter_books()
        self._current_position = 0
    
    def _filter_books(self) -> List[Book]:
        """Filter books by year range."""
        filtered = []
        for book in self._collection.get_all_books():
            if self._start_year <= book.year <= self._end_year:
                filtered.append(book)
        return filtered
    
    def has_next(self) -> bool:
        """Check if there are more books in year range."""
        return self._current_position < len(self._filtered_books)
    
    def next(self) -> Book:
        """Get next book in year range."""
        if not self.has_next():
            raise StopIteration(f"No more books in year range {self._start_year}-{self._end_year}")
        
        book = self._filtered_books[self._current_position]
        self._current_position += 1
        return book
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_position = 0
        self._filtered_books = self._filter_books()  # Refresh filter
        print(f"Year range iterator ({self._start_year}-{self._end_year}) reset")
    
    def get_year_range(self) -> tuple:
        """Get year range being iterated."""
        return (self._start_year, self._end_year)


class RandomBookIterator(Iterator[Book]):
    """Iterator that returns books in random order."""
    
    def __init__(self, collection: BookCollection, seed: int = None):
        self._collection = collection
        self._seed = seed
        self._books = collection.get_all_books().copy()
        self._current_position = 0
        
        # Shuffle books
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._books)
    
    def has_next(self) -> bool:
        """Check if there are more books."""
        return self._current_position < len(self._books)
    
    def next(self) -> Book:
        """Get next random book."""
        if not self.has_next():
            raise StopIteration("No more books in random order")
        
        book = self._books[self._current_position]
        self._current_position += 1
        return book
    
    def reset(self) -> None:
        """Reset and re-shuffle."""
        self._current_position = 0
        self._books = self._collection.get_all_books().copy()
        
        if self._seed is not None:
            random.seed(self._seed)
        random.shuffle(self._books)
        
        print("Random iterator reset and re-shuffled")


class SortedBookIterator(Iterator[Book]):
    """Iterator that returns books in sorted order."""
    
    def __init__(self, collection: BookCollection, key_func: callable, reverse: bool = False):
        self._collection = collection
        self._key_func = key_func
        self._reverse = reverse
        self._books = sorted(collection.get_all_books(), key=key_func, reverse=reverse)
        self._current_position = 0
    
    def has_next(self) -> bool:
        """Check if there are more books."""
        return self._current_position < len(self._books)
    
    def next(self) -> Book:
        """Get next sorted book."""
        if not self.has_next():
            raise StopIteration("No more books in sorted order")
        
        book = self._books[self._current_position]
        self._current_position += 1
        return book
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_position = 0
        self._books = sorted(self._collection.get_all_books(), key=self._key_func, reverse=self._reverse)
        print("Sorted iterator reset")


# ============================================================================
# TREE STRUCTURE WITH ITERATOR
# ============================================================================

class TreeNode:
    """Tree node for hierarchical data."""
    
    def __init__(self, value: Any, node_id: str = None):
        self.value = value
        self.node_id = node_id or str(id(self))
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
    
    def add_child(self, child: 'TreeNode') -> None:
        """Add child node."""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'TreeNode') -> bool:
        """Remove child node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False
    
    def is_leaf(self) -> bool:
        """Check if node is leaf."""
        return len(self.children) == 0
    
    def get_depth(self) -> int:
        """Get node depth from root."""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    def __str__(self) -> str:
        return f"TreeNode({self.value})"


class Tree(Iterable[TreeNode]):
    """Tree data structure with multiple traversal iterators."""
    
    def __init__(self, root_value: Any = None):
        self.root = TreeNode(root_value) if root_value is not None else None
        self._node_count = 1 if self.root else 0
    
    def add_node(self, parent_node: TreeNode, value: Any) -> TreeNode:
        """Add node as child of parent."""
        new_node = TreeNode(value)
        parent_node.add_child(new_node)
        self._node_count += 1
        return new_node
    
    def create_iterator(self) -> Iterator[TreeNode]:
        """Create default iterator (depth-first)."""
        return DepthFirstTreeIterator(self)
    
    def create_breadth_first_iterator(self) -> Iterator[TreeNode]:
        """Create breadth-first iterator."""
        return BreadthFirstTreeIterator(self)
    
    def create_leaf_iterator(self) -> Iterator[TreeNode]:
        """Create iterator for leaf nodes only."""
        return LeafNodeIterator(self)
    
    def create_level_iterator(self, level: int) -> Iterator[TreeNode]:
        """Create iterator for nodes at specific level."""
        return LevelTreeIterator(self, level)
    
    def size(self) -> int:
        """Get number of nodes in tree."""
        return self._node_count
    
    def get_height(self) -> int:
        """Get tree height."""
        if not self.root:
            return 0
        
        def calculate_height(node: TreeNode) -> int:
            if not node.children:
                return 1
            return 1 + max(calculate_height(child) for child in node.children)
        
        return calculate_height(self.root)


class DepthFirstTreeIterator(Iterator[TreeNode]):
    """Depth-first tree iterator."""
    
    def __init__(self, tree: Tree):
        self._tree = tree
        self._stack: List[TreeNode] = []
        self._visited: set = set()
        
        if tree.root:
            self._stack.append(tree.root)
    
    def has_next(self) -> bool:
        """Check if there are more nodes."""
        return len(self._stack) > 0
    
    def next(self) -> TreeNode:
        """Get next node in depth-first order."""
        if not self.has_next():
            raise StopIteration("No more nodes")
        
        node = self._stack.pop()
        self._visited.add(node.node_id)
        
        # Add children to stack (in reverse order for correct traversal)
        for child in reversed(node.children):
            if child.node_id not in self._visited:
                self._stack.append(child)
        
        return node
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._stack = []
        self._visited = set()
        
        if self._tree.root:
            self._stack.append(self._tree.root)
        
        print("Depth-first iterator reset")


class BreadthFirstTreeIterator(Iterator[TreeNode]):
    """Breadth-first tree iterator."""
    
    def __init__(self, tree: Tree):
        self._tree = tree
        self._queue: List[TreeNode] = []
        self._visited: set = set()
        
        if tree.root:
            self._queue.append(tree.root)
    
    def has_next(self) -> bool:
        """Check if there are more nodes."""
        return len(self._queue) > 0
    
    def next(self) -> TreeNode:
        """Get next node in breadth-first order."""
        if not self.has_next():
            raise StopIteration("No more nodes")
        
        node = self._queue.pop(0)  # Remove from front
        self._visited.add(node.node_id)
        
        # Add children to queue
        for child in node.children:
            if child.node_id not in self._visited:
                self._queue.append(child)
        
        return node
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._queue = []
        self._visited = set()
        
        if self._tree.root:
            self._queue.append(self._tree.root)
        
        print("Breadth-first iterator reset")


class LeafNodeIterator(Iterator[TreeNode]):
    """Iterator for leaf nodes only."""
    
    def __init__(self, tree: Tree):
        self._tree = tree
        self._leaf_nodes = self._collect_leaf_nodes()
        self._current_position = 0
    
    def _collect_leaf_nodes(self) -> List[TreeNode]:
        """Collect all leaf nodes."""
        leaves = []
        
        def collect_leaves(node: TreeNode):
            if node.is_leaf():
                leaves.append(node)
            else:
                for child in node.children:
                    collect_leaves(child)
        
        if self._tree.root:
            collect_leaves(self._tree.root)
        
        return leaves
    
    def has_next(self) -> bool:
        """Check if there are more leaf nodes."""
        return self._current_position < len(self._leaf_nodes)
    
    def next(self) -> TreeNode:
        """Get next leaf node."""
        if not self.has_next():
            raise StopIteration("No more leaf nodes")
        
        node = self._leaf_nodes[self._current_position]
        self._current_position += 1
        return node
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_position = 0
        self._leaf_nodes = self._collect_leaf_nodes()  # Refresh leaf nodes
        print("Leaf node iterator reset")


class LevelTreeIterator(Iterator[TreeNode]):
    """Iterator for nodes at specific level."""
    
    def __init__(self, tree: Tree, level: int):
        self._tree = tree
        self._level = level
        self._level_nodes = self._collect_level_nodes()
        self._current_position = 0
    
    def _collect_level_nodes(self) -> List[TreeNode]:
        """Collect nodes at specific level."""
        level_nodes = []
        
        def collect_at_level(node: TreeNode, current_level: int):
            if current_level == self._level:
                level_nodes.append(node)
            elif current_level < self._level:
                for child in node.children:
                    collect_at_level(child, current_level + 1)
        
        if self._tree.root:
            collect_at_level(self._tree.root, 0)
        
        return level_nodes
    
    def has_next(self) -> bool:
        """Check if there are more nodes at this level."""
        return self._current_position < len(self._level_nodes)
    
    def next(self) -> TreeNode:
        """Get next node at this level."""
        if not self.has_next():
            raise StopIteration(f"No more nodes at level {self._level}")
        
        node = self._level_nodes[self._current_position]
        self._current_position += 1
        return node
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_position = 0
        self._level_nodes = self._collect_level_nodes()  # Refresh level nodes
        print(f"Level {self._level} iterator reset")


# ============================================================================
# LAZY EVALUATION ITERATOR
# ============================================================================

class LazyRange(Iterable[int]):
    """Lazy range implementation that generates numbers on demand."""
    
    def __init__(self, start: int, end: int, step: int = 1):
        self.start = start
        self.end = end
        self.step = step
    
    def create_iterator(self) -> Iterator[int]:
        """Create lazy range iterator."""
        return LazyRangeIterator(self.start, self.end, self.step)
    
    def size(self) -> int:
        """Calculate size without generating all numbers."""
        if self.step > 0:
            return max(0, (self.end - self.start + self.step - 1) // self.step)
        else:
            return max(0, (self.start - self.end - self.step - 1) // (-self.step))


class LazyRangeIterator(Iterator[int]):
    """Lazy iterator that generates numbers on demand."""
    
    def __init__(self, start: int, end: int, step: int = 1):
        self.start = start
        self.end = end
        self.step = step
        self.current = start
        self.generated_count = 0
    
    def has_next(self) -> bool:
        """Check if there are more numbers."""
        if self.step > 0:
            return self.current < self.end
        else:
            return self.current > self.end
    
    def next(self) -> int:
        """Generate next number."""
        if not self.has_next():
            raise StopIteration("No more numbers in range")
        
        value = self.current
        self.current += self.step
        self.generated_count += 1
        
        return value
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self.current = self.start
        self.generated_count = 0
        print(f"Lazy range iterator reset")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get generation progress."""
        return {
            'generated_count': self.generated_count,
            'current_value': self.current,
            'has_next': self.has_next()
        }


# ============================================================================
# THREAD-SAFE ITERATOR
# ============================================================================

class ThreadSafeCollection(Iterable[T]):
    """Thread-safe collection with synchronized iterators."""
    
    def __init__(self):
        self._items: List[T] = []
        self._lock = threading.RLock()
        self._version = 0  # Version counter for concurrent modification detection
    
    def add_item(self, item: T) -> None:
        """Add item to collection."""
        with self._lock:
            self._items.append(item)
            self._version += 1
            print(f"Added item: {item}")
    
    def remove_item(self, item: T) -> bool:
        """Remove item from collection."""
        with self._lock:
            if item in self._items:
                self._items.remove(item)
                self._version += 1
                print(f"Removed item: {item}")
                return True
            return False
    
    def create_iterator(self) -> Iterator[T]:
        """Create thread-safe iterator."""
        return ThreadSafeIterator(self)
    
    def size(self) -> int:
        """Get collection size."""
        with self._lock:
            return len(self._items)
    
    def get_snapshot(self) -> List[T]:
        """Get snapshot of current items."""
        with self._lock:
            return self._items.copy()
    
    def get_version(self) -> int:
        """Get current version."""
        with self._lock:
            return self._version


class ThreadSafeIterator(Iterator[T]):
    """Thread-safe iterator with concurrent modification detection."""
    
    def __init__(self, collection: ThreadSafeCollection[T]):
        self._collection = collection
        with collection._lock:
            self._snapshot = collection._items.copy()
            self._expected_version = collection._version
        self._current_index = 0
    
    def has_next(self) -> bool:
        """Check if there are more items."""
        self._check_concurrent_modification()
        return self._current_index < len(self._snapshot)
    
    def next(self) -> T:
        """Get next item."""
        self._check_concurrent_modification()
        
        if not self.has_next():
            raise StopIteration("No more items")
        
        item = self._snapshot[self._current_index]
        self._current_index += 1
        return item
    
    def reset(self) -> None:
        """Reset iterator with fresh snapshot."""
        with self._collection._lock:
            self._snapshot = self._collection._items.copy()
            self._expected_version = self._collection._version
        self._current_index = 0
        print("Thread-safe iterator reset with fresh snapshot")
    
    def _check_concurrent_modification(self) -> None:
        """Check for concurrent modifications."""
        current_version = self._collection.get_version()
        if current_version != self._expected_version:
            print(f"Warning: Collection was modified during iteration "
                  f"(expected version {self._expected_version}, current {current_version})")
            # Note: In a real implementation, you might want to raise an exception
            # or automatically refresh the snapshot


def demonstrate_iterator_pattern():
    """
    Demonstrate Iterator pattern implementations.
    """
    print("=== ITERATOR PATTERN DEMONSTRATION ===\n")
    
    # 1. Book Collection with Multiple Iterators
    print("1. BOOK COLLECTION WITH MULTIPLE ITERATORS:")
    
    # Create book collection
    library = BookCollection()
    
    # Add books
    books_data = [
        ("The Great Gatsby", "F. Scott Fitzgerald", 1925, "Fiction", 180),
        ("To Kill a Mockingbird", "Harper Lee", 1960, "Fiction", 281),
        ("1984", "George Orwell", 1949, "Dystopian", 328),
        ("Animal Farm", "George Orwell", 1945, "Dystopian", 112),
        ("Pride and Prejudice", "Jane Austen", 1813, "Romance", 432),
        ("The Catcher in the Rye", "J.D. Salinger", 1951, "Fiction", 277),
        ("Brave New World", "Aldous Huxley", 1932, "Dystopian", 268),
        ("Jane Eyre", "Charlotte Brontë", 1847, "Romance", 500)
    ]
    
    for title, author, year, genre, pages in books_data:
        library.add_book(Book(title, author, year, genre, pages))
    
    print(f"\n   Library statistics: {library.get_statistics()}")
    
    # Test different iterators
    print(f"\n   Sequential iteration:")
    sequential_iter = library.create_iterator()
    count = 0
    while sequential_iter.has_next() and count < 3:  # Show first 3
        book = sequential_iter.next()
        print(f"     {book}")
        count += 1
    print(f"     ... (showing first 3 of {library.size()} books)")
    
    # Author-specific iteration
    print(f"\n   Books by George Orwell:")
    orwell_iter = library.create_author_iterator("George Orwell")
    while orwell_iter.has_next():
        book = orwell_iter.next()
        print(f"     {book}")
    
    # Genre-specific iteration
    print(f"\n   Dystopian books:")
    dystopian_iter = library.create_genre_iterator("Dystopian")
    while dystopian_iter.has_next():
        book = dystopian_iter.next()
        print(f"     {book}")
    
    # Year range iteration
    print(f"\n   Books from 1940-1960:")
    year_iter = library.create_year_range_iterator(1940, 1960)
    while year_iter.has_next():
        book = year_iter.next()
        print(f"     {book}")
    
    # Sorted iteration
    print(f"\n   Books sorted by year:")
    sorted_iter = library.create_sorted_iterator(lambda book: book.year)
    count = 0
    while sorted_iter.has_next() and count < 4:  # Show first 4
        book = sorted_iter.next()
        print(f"     {book}")
        count += 1
    
    # Random iteration
    print(f"\n   Random book order (seed=42):")
    random_iter = library.create_random_iterator(42)
    count = 0
    while random_iter.has_next() and count < 3:  # Show first 3
        book = random_iter.next()
        print(f"     {book}")
        count += 1
    
    print()
    
    # 2. Tree Structure with Different Traversals
    print("2. TREE STRUCTURE WITH DIFFERENT TRAVERSALS:")
    
    # Build tree structure
    tree = Tree("Root")
    
    # Level 1
    child1 = tree.add_node(tree.root, "Child1")
    child2 = tree.add_node(tree.root, "Child2")
    child3 = tree.add_node(tree.root, "Child3")
    
    # Level 2
    grandchild1 = tree.add_node(child1, "GrandChild1")
    grandchild2 = tree.add_node(child1, "GrandChild2")
    grandchild3 = tree.add_node(child2, "GrandChild3")
    
    # Level 3
    tree.add_node(grandchild1, "GreatGrandChild1")
    tree.add_node(grandchild1, "GreatGrandChild2")
    
    print(f"\n   Tree structure:")
    print(f"     Nodes: {tree.size()}")
    print(f"     Height: {tree.get_height()}")
    
    # Depth-first traversal
    print(f"\n   Depth-first traversal:")
    df_iter = tree.create_iterator()
    while df_iter.has_next():
        node = df_iter.next()
        indent = "  " * node.get_depth()
        print(f"     {indent}{node.value}")
    
    # Breadth-first traversal
    print(f"\n   Breadth-first traversal:")
    bf_iter = tree.create_breadth_first_iterator()
    while bf_iter.has_next():
        node = bf_iter.next()
        print(f"     Level {node.get_depth()}: {node.value}")
    
    # Leaf nodes only
    print(f"\n   Leaf nodes:")
    leaf_iter = tree.create_leaf_iterator()
    while leaf_iter.has_next():
        node = leaf_iter.next()
        print(f"     {node.value}")
    
    # Nodes at specific level
    print(f"\n   Nodes at level 2:")
    level_iter = tree.create_level_iterator(2)
    while level_iter.has_next():
        node = level_iter.next()
        print(f"     {node.value}")
    
    print()
    
    # 3. Lazy Evaluation Iterator
    print("3. LAZY EVALUATION ITERATOR:")
    
    # Create lazy range
    lazy_range = LazyRange(0, 1000000, 7)  # Large range with step 7
    
    print(f"\n   Lazy range: 0 to 1,000,000 step 7")
    print(f"   Calculated size: {lazy_range.size()}")
    
    # Use lazy iterator
    lazy_iter = lazy_range.create_iterator()
    
    print(f"   First 10 values:")
    count = 0
    while lazy_iter.has_next() and count < 10:
        value = lazy_iter.next()
        print(f"     {value}")
        count += 1
    
    # Show progress
    progress = lazy_iter.get_progress()
    print(f"   Progress: {progress}")
    
    # Reset and show different values
    lazy_iter.reset()
    print(f"   After reset, next 5 values:")
    count = 0
    while lazy_iter.has_next() and count < 5:
        value = lazy_iter.next()
        print(f"     {value}")
        count += 1
    
    print()
    
    # 4. Thread-Safe Iterator
    print("4. THREAD-SAFE ITERATOR:")
    
    # Create thread-safe collection
    safe_collection = ThreadSafeCollection[str]()
    
    # Add items
    items = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
    for item in items:
        safe_collection.add_item(item)
    
    print(f"\n   Thread-safe collection size: {safe_collection.size()}")
    
    # Create iterator
    safe_iter = safe_collection.create_iterator()
    
    print(f"   Iterating through collection:")
    while safe_iter.has_next():
        item = safe_iter.next()
        print(f"     {item}")
    
    # Simulate concurrent modification
    print(f"\n   Simulating concurrent modification:")
    safe_iter.reset()
    
    # Start iteration
    print(f"   First item: {safe_iter.next()}")
    
    # Modify collection during iteration
    safe_collection.add_item("Fig")
    safe_collection.remove_item("Banana")
    
    # Continue iteration (should detect modification)
    print(f"   Continuing iteration after modification:")
    while safe_iter.has_next():
        item = safe_iter.next()
        print(f"     {item}")
    
    print()
    
    # 5. Iterator Composition and Chaining
    print("5. ITERATOR COMPOSITION AND CHAINING:")
    
    class FilterIterator(Iterator[T]):
        """Iterator that filters elements based on predicate."""
        
        def __init__(self, source_iterator: Iterator[T], predicate: callable):
            self.source_iterator = source_iterator
            self.predicate = predicate
            self._next_item = None
            self._has_next_cached = None
        
        def has_next(self) -> bool:
            """Check if there are more filtered items."""
            if self._has_next_cached is not None:
                return self._has_next_cached
            
            while self.source_iterator.has_next():
                item = self.source_iterator.next()
                if self.predicate(item):
                    self._next_item = item
                    self._has_next_cached = True
                    return True
            
            self._has_next_cached = False
            return False
        
        def next(self) -> T:
            """Get next filtered item."""
            if not self.has_next():
                raise StopIteration("No more filtered items")
            
            item = self._next_item
            self._next_item = None
            self._has_next_cached = None
            return item
        
        def reset(self) -> None:
            """Reset filter iterator."""
            self.source_iterator.reset()
            self._next_item = None
            self._has_next_cached = None
    
    class MapIterator(Iterator[T]):
        """Iterator that transforms elements."""
        
        def __init__(self, source_iterator: Iterator, transform_func: callable):
            self.source_iterator = source_iterator
            self.transform_func = transform_func
        
        def has_next(self) -> bool:
            """Check if there are more items to transform."""
            return self.source_iterator.has_next()
        
        def next(self) -> T:
            """Get next transformed item."""
            if not self.has_next():
                raise StopIteration("No more items to transform")
            
            item = self.source_iterator.next()
            return self.transform_func(item)
        
        def reset(self) -> None:
            """Reset map iterator."""
            self.source_iterator.reset()
    
    # Chain iterators: filter fiction books, then map to titles
    print(f"   Chained iterators: Fiction books -> Titles:")
    
    # Start with all books
    base_iter = library.create_iterator()
    
    # Filter for fiction books
    fiction_filter = FilterIterator(base_iter, lambda book: book.genre == "Fiction")
    
    # Map to titles
    title_mapper = MapIterator(fiction_filter, lambda book: book.title.upper())
    
    while title_mapper.has_next():
        title = title_mapper.next()
        print(f"     {title}")
    
    print()
    
    # 6. Performance Comparison
    print("6. PERFORMANCE COMPARISON:")
    
    # Create large collection for performance testing
    large_library = BookCollection()
    
    # Add many books
    genres = ["Fiction", "Science", "History", "Biography", "Mystery"]
    authors = ["Author A", "Author B", "Author C", "Author D", "Author E"]
    
    print(f"\n   Creating large collection...")
    start_time = time.time()
    
    for i in range(1000):  # Reduced for demo
        title = f"Book {i+1}"
        author = authors[i % len(authors)]
        year = 1900 + (i % 120)  # Years 1900-2019
        genre = genres[i % len(genres)]
        pages = 200 + (i % 300)  # 200-499 pages
        
        large_library.add_book(Book(title, author, year, genre, pages))
    
    creation_time = time.time() - start_time
    print(f"   Created {large_library.size()} books in {creation_time:.3f} seconds")
    
    # Test different iterator performance
    iterators_to_test = [
        ("Sequential", large_library.create_iterator()),
        ("Author A", large_library.create_author_iterator("Author A")),
        ("Fiction Genre", large_library.create_genre_iterator("Fiction")),
        ("Year Range 1950-1970", large_library.create_year_range_iterator(1950, 1970)),
        ("Sorted by Year", large_library.create_sorted_iterator(lambda book: book.year))
    ]
    
    print(f"\n   Iterator performance comparison:")
    for name, iterator in iterators_to_test:
        start_time = time.time()
        count = 0
        
        while iterator.has_next():
            book = iterator.next()
            count += 1
        
        iteration_time = time.time() - start_time
        print(f"     {name}: {count} items in {iteration_time:.4f} seconds")
    
    print()
    
    # 7. Iterator Pattern Benefits
    print("7. ITERATOR PATTERN BENEFITS:")
    print("   ✓ Uniform Interface: Same interface for different collection types")
    print("   ✓ Encapsulation: Internal collection structure is hidden")
    print("   ✓ Multiple Traversals: Different iteration strategies for same collection")
    print("   ✓ Lazy Evaluation: Elements generated on-demand for memory efficiency")
    print("   ✓ Concurrent Iteration: Multiple iterators can traverse same collection")
    print("   ✓ Composability: Iterators can be chained and composed")
    print("   ✓ Flexibility: Easy to add new iteration strategies")
    print("   ✓ Memory Efficiency: No need to load entire collection into memory")
    print("   ✓ Thread Safety: Can implement thread-safe iteration patterns")
    print("   ✓ Separation of Concerns: Iteration logic separated from collection logic")
    print()
    
    print("=== ITERATOR PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_iterator_pattern()
