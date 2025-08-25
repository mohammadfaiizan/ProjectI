"""
1472. Design Browser History - Multiple Approaches
Difficulty: Medium

You have a browser of one tab where you start on the homepage and you can visit another url, get back to the previous page in the history and move forward to the next page.

Implement the BrowserHistory class:
- BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
- void visit(string url) Visits url from the current page. It clears up all the forward history.
- string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. Return the current url after moving back in history at most steps.
- string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after moving forward in history at most steps.
"""

from typing import List, Optional

class BrowserHistoryArray:
    """
    Approach 1: Dynamic Array Implementation
    
    Use array to store history with current position pointer.
    
    Time Complexity:
    - visit: O(1)
    - back: O(1)
    - forward: O(1)
    
    Space Complexity: O(n) where n is number of pages visited
    """
    
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0  # Current position in history
    
    def visit(self, url: str) -> None:
        # Clear forward history and add new page
        self.current += 1
        self.history = self.history[:self.current]
        self.history.append(url)
    
    def back(self, steps: int) -> str:
        self.current = max(0, self.current - steps)
        return self.history[self.current]
    
    def forward(self, steps: int) -> str:
        self.current = min(len(self.history) - 1, self.current + steps)
        return self.history[self.current]

class BrowserHistoryLinkedList:
    """
    Approach 2: Doubly Linked List Implementation
    
    Use doubly linked list for history navigation.
    
    Time Complexity:
    - visit: O(1)
    - back: O(min(steps, position))
    - forward: O(min(steps, forward_count))
    
    Space Complexity: O(n)
    """
    
    class Node:
        def __init__(self, url: str):
            self.url = url
            self.prev = None
            self.next = None
    
    def __init__(self, homepage: str):
        self.current = self.Node(homepage)
    
    def visit(self, url: str) -> None:
        new_node = self.Node(url)
        new_node.prev = self.current
        self.current.next = new_node
        self.current = new_node
    
    def back(self, steps: int) -> str:
        while steps > 0 and self.current.prev:
            self.current = self.current.prev
            steps -= 1
        return self.current.url
    
    def forward(self, steps: int) -> str:
        while steps > 0 and self.current.next:
            self.current = self.current.next
            steps -= 1
        return self.current.url

class BrowserHistoryStack:
    """
    Approach 3: Two Stacks Implementation
    
    Use two stacks for back and forward history.
    
    Time Complexity:
    - visit: O(1)
    - back: O(min(steps, back_stack_size))
    - forward: O(min(steps, forward_stack_size))
    
    Space Complexity: O(n)
    """
    
    def __init__(self, homepage: str):
        self.current_page = homepage
        self.back_stack = []
        self.forward_stack = []
    
    def visit(self, url: str) -> None:
        # Push current page to back stack
        self.back_stack.append(self.current_page)
        self.current_page = url
        # Clear forward history
        self.forward_stack.clear()
    
    def back(self, steps: int) -> str:
        while steps > 0 and self.back_stack:
            # Move current to forward stack
            self.forward_stack.append(self.current_page)
            # Pop from back stack to current
            self.current_page = self.back_stack.pop()
            steps -= 1
        return self.current_page
    
    def forward(self, steps: int) -> str:
        while steps > 0 and self.forward_stack:
            # Move current to back stack
            self.back_stack.append(self.current_page)
            # Pop from forward stack to current
            self.current_page = self.forward_stack.pop()
            steps -= 1
        return self.current_page

class BrowserHistoryAdvanced:
    """
    Approach 4: Advanced with Features
    
    Enhanced browser history with additional features and analytics.
    
    Time Complexity:
    - visit: O(1)
    - back: O(1)
    - forward: O(1)
    
    Space Complexity: O(n + features)
    """
    
    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0
        
        # Enhanced features
        self.visit_count = {homepage: 1}
        self.total_visits = 1
        self.back_operations = 0
        self.forward_operations = 0
        self.session_start = self._get_current_time()
        
        # Bookmarks and favorites
        self.bookmarks = set()
        self.favorites = []
        
        # Page metadata
        self.page_titles = {homepage: "Homepage"}
        self.visit_timestamps = {homepage: [self._get_current_time()]}
    
    def _get_current_time(self) -> int:
        """Get current timestamp (simplified)"""
        import time
        return int(time.time())
    
    def visit(self, url: str) -> None:
        # Update analytics
        self.total_visits += 1
        self.visit_count[url] = self.visit_count.get(url, 0) + 1
        
        # Update timestamps
        current_time = self._get_current_time()
        if url not in self.visit_timestamps:
            self.visit_timestamps[url] = []
        self.visit_timestamps[url].append(current_time)
        
        # Update history
        self.current += 1
        self.history = self.history[:self.current]
        self.history.append(url)
    
    def back(self, steps: int) -> str:
        self.back_operations += 1
        actual_steps = min(steps, self.current)
        self.current -= actual_steps
        return self.history[self.current]
    
    def forward(self, steps: int) -> str:
        self.forward_operations += 1
        max_forward = len(self.history) - 1 - self.current
        actual_steps = min(steps, max_forward)
        self.current += actual_steps
        return self.history[self.current]
    
    def bookmark(self, url: str = None) -> None:
        """Bookmark current or specified URL"""
        if url is None:
            url = self.history[self.current]
        self.bookmarks.add(url)
    
    def get_bookmarks(self) -> List[str]:
        """Get all bookmarks"""
        return list(self.bookmarks)
    
    def add_to_favorites(self, url: str = None) -> None:
        """Add to favorites"""
        if url is None:
            url = self.history[self.current]
        if url not in self.favorites:
            self.favorites.append(url)
    
    def get_most_visited(self, k: int = 5) -> List[tuple]:
        """Get top K most visited pages"""
        sorted_pages = sorted(self.visit_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:k]
    
    def get_current_url(self) -> str:
        """Get current URL"""
        return self.history[self.current]
    
    def get_history_length(self) -> int:
        """Get total history length"""
        return len(self.history)
    
    def can_go_back(self) -> bool:
        """Check if can go back"""
        return self.current > 0
    
    def can_go_forward(self) -> bool:
        """Check if can go forward"""
        return self.current < len(self.history) - 1
    
    def get_analytics(self) -> dict:
        """Get browsing analytics"""
        session_time = self._get_current_time() - self.session_start
        
        return {
            'total_visits': self.total_visits,
            'unique_pages': len(self.visit_count),
            'current_position': self.current,
            'history_length': len(self.history),
            'back_operations': self.back_operations,
            'forward_operations': self.forward_operations,
            'bookmarks_count': len(self.bookmarks),
            'favorites_count': len(self.favorites),
            'session_duration': session_time,
            'pages_per_minute': (self.total_visits / max(1, session_time / 60))
        }

class BrowserHistoryMemoryOptimized:
    """
    Approach 5: Memory-Optimized Implementation
    
    Optimize memory usage with circular buffer and compression.
    
    Time Complexity:
    - visit: O(1)
    - back: O(1)
    - forward: O(1)
    
    Space Complexity: O(min(n, max_history_size))
    """
    
    def __init__(self, homepage: str, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.history = [None] * max_history_size
        self.history[0] = homepage
        
        self.start = 0  # Start of valid history
        self.current = 0  # Current position
        self.end = 0     # End of valid history
        self.size = 1    # Current size
        
        # URL compression (simple mapping)
        self.url_to_id = {homepage: 0}
        self.id_to_url = {0: homepage}
        self.next_id = 1
    
    def _compress_url(self, url: str) -> int:
        """Compress URL to ID for memory efficiency"""
        if url not in self.url_to_id:
            self.url_to_id[url] = self.next_id
            self.id_to_url[self.next_id] = url
            self.next_id += 1
        return self.url_to_id[url]
    
    def _decompress_url(self, url_id: int) -> str:
        """Decompress ID back to URL"""
        return self.id_to_url[url_id]
    
    def visit(self, url: str) -> None:
        url_id = self._compress_url(url)
        
        # Move current forward and add new page
        self.current = (self.current + 1) % self.max_history_size
        self.history[self.current] = url_id
        
        # Update end position
        self.end = self.current
        
        # Update size and start if necessary
        if self.size < self.max_history_size:
            self.size += 1
        else:
            # Circular buffer is full, move start
            self.start = (self.start + 1) % self.max_history_size
    
    def back(self, steps: int) -> str:
        # Calculate how far back we can go
        back_distance = (self.current - self.start + self.max_history_size) % self.max_history_size
        actual_steps = min(steps, back_distance)
        
        self.current = (self.current - actual_steps + self.max_history_size) % self.max_history_size
        return self._decompress_url(self.history[self.current])
    
    def forward(self, steps: int) -> str:
        # Calculate how far forward we can go
        forward_distance = (self.end - self.current + self.max_history_size) % self.max_history_size
        actual_steps = min(steps, forward_distance)
        
        self.current = (self.current + actual_steps) % self.max_history_size
        return self._decompress_url(self.history[self.current])
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics"""
        return {
            'max_history_size': self.max_history_size,
            'current_size': self.size,
            'unique_urls': len(self.url_to_id),
            'memory_utilization': (self.size / self.max_history_size) * 100,
            'compression_ratio': len(self.url_to_id) / max(1, self.size)
        }


def test_browser_history_basic():
    """Test basic browser history functionality"""
    print("=== Testing Basic Browser History Functionality ===")
    
    implementations = [
        ("Array-based", BrowserHistoryArray),
        ("Linked List", BrowserHistoryLinkedList),
        ("Two Stacks", BrowserHistoryStack),
        ("Advanced", BrowserHistoryAdvanced),
        ("Memory Optimized", lambda homepage: BrowserHistoryMemoryOptimized(homepage, 100))
    ]
    
    for name, BrowserClass in implementations:
        print(f"\n{name}:")
        
        browser = BrowserClass("homepage.com")
        
        # Test sequence from problem
        operations = [
            ("visit", "google.com"), ("visit", "facebook.com"), ("visit", "youtube.com"),
            ("back", 1), ("back", 1), ("forward", 1),
            ("visit", "linkedin.com"), ("forward", 2), ("back", 2), ("back", 7)
        ]
        
        for op, arg in operations:
            if op == "visit":
                browser.visit(arg)
                print(f"  visit('{arg}')")
            elif op == "back":
                result = browser.back(arg)
                print(f"  back({arg}): '{result}'")
            elif op == "forward":
                result = browser.forward(arg)
                print(f"  forward({arg}): '{result}'")

def test_browser_history_edge_cases():
    """Test browser history edge cases"""
    print("\n=== Testing Browser History Edge Cases ===")
    
    browser = BrowserHistoryAdvanced("start.com")
    
    # Test back when at beginning
    print("Back at beginning:")
    result = browser.back(5)
    print(f"  back(5) from start: '{result}'")
    
    # Test forward when at end
    print(f"\nForward at end:")
    result = browser.forward(3)
    print(f"  forward(3) from start: '{result}'")
    
    # Test visit clearing forward history
    print(f"\nVisit clearing forward history:")
    browser.visit("page1.com")
    browser.visit("page2.com")
    browser.back(1)
    print(f"  After back(1): '{browser.get_current_url()}'")
    
    browser.visit("page3.com")
    print(f"  After visit('page3.com'): '{browser.get_current_url()}'")
    
    # Try to go forward (should stay at page3.com)
    result = browser.forward(1)
    print(f"  forward(1) after visit: '{result}'")
    
    # Test large steps
    print(f"\nLarge steps:")
    browser.visit("page4.com")
    browser.visit("page5.com")
    
    result = browser.back(100)
    print(f"  back(100): '{result}'")
    
    result = browser.forward(100)
    print(f"  forward(100): '{result}'")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Array-based", BrowserHistoryArray),
        ("Linked List", BrowserHistoryLinkedList),
        ("Two Stacks", BrowserHistoryStack),
        ("Advanced", BrowserHistoryAdvanced)
    ]
    
    num_operations = 10000
    
    for name, BrowserClass in implementations:
        browser = BrowserClass("homepage.com")
        
        start_time = time.time()
        
        # Mix of operations
        import random
        for i in range(num_operations):
            op_type = random.choice(['visit', 'back', 'forward'])
            
            if op_type == 'visit':
                browser.visit(f"page{i}.com")
            elif op_type == 'back':
                steps = random.randint(1, 5)
                browser.back(steps)
            else:  # forward
                steps = random.randint(1, 5)
                browser.forward(steps)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {num_operations} operations")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    browser = BrowserHistoryAdvanced("homepage.com")
    
    # Build browsing history
    pages = ["google.com", "facebook.com", "youtube.com", "google.com", "stackoverflow.com"]
    
    print("Building browsing history:")
    for page in pages:
        browser.visit(page)
        print(f"  visit('{page}')")
    
    # Test bookmarking
    browser.bookmark("google.com")
    browser.bookmark("stackoverflow.com")
    
    bookmarks = browser.get_bookmarks()
    print(f"\nBookmarks: {bookmarks}")
    
    # Test favorites
    browser.add_to_favorites("youtube.com")
    browser.add_to_favorites("google.com")
    
    print(f"Favorites: {browser.favorites}")
    
    # Test analytics
    analytics = browser.get_analytics()
    print(f"\nBrowsing analytics:")
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test most visited pages
    most_visited = browser.get_most_visited(3)
    print(f"\nMost visited pages:")
    for page, count in most_visited:
        print(f"  {page}: {count} visits")
    
    # Test navigation state
    print(f"\nNavigation state:")
    print(f"  Current URL: {browser.get_current_url()}")
    print(f"  Can go back: {browser.can_go_back()}")
    print(f"  Can go forward: {browser.can_go_forward()}")
    print(f"  History length: {browser.get_history_length()}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Web browser simulation
    print("Application 1: Web Browser Navigation")
    
    browser = BrowserHistoryAdvanced("www.home.com")
    
    # Simulate user browsing session
    browsing_session = [
        ("visit", "www.google.com"),
        ("visit", "www.stackoverflow.com"),
        ("visit", "www.github.com"),
        ("back", 1),    # Back to stackoverflow
        ("visit", "www.python.org"),  # Branch from stackoverflow
        ("back", 2),    # Back to google
        ("forward", 1), # Forward to stackoverflow
        ("visit", "www.reddit.com")   # New branch
    ]
    
    print("  Simulating browsing session:")
    for operation, arg in browsing_session:
        if operation == "visit":
            browser.visit(arg)
            print(f"    Navigate to: {arg}")
        elif operation == "back":
            result = browser.back(arg)
            print(f"    Back {arg} step(s) to: {result}")
        elif operation == "forward":
            result = browser.forward(arg)
            print(f"    Forward {arg} step(s) to: {result}")
    
    # Show session summary
    analytics = browser.get_analytics()
    print(f"  Session summary: {analytics['total_visits']} visits, "
          f"{analytics['back_operations']} backs, {analytics['forward_operations']} forwards")
    
    # Application 2: Mobile app navigation
    print(f"\nApplication 2: Mobile App Screen Navigation")
    
    app_navigator = BrowserHistoryStack("HomeScreen")
    
    # Simulate app navigation
    app_flow = [
        ("visit", "ProfileScreen"),
        ("visit", "SettingsScreen"),
        ("visit", "NotificationSettings"),
        ("back", 1),  # Back to Settings
        ("visit", "PrivacySettings"),
        ("back", 2),  # Back to Profile
        ("visit", "EditProfile")
    ]
    
    print("  App navigation flow:")
    for operation, screen in app_flow:
        if operation == "visit":
            app_navigator.visit(screen)
            print(f"    Navigate to: {screen}")
        elif operation == "back":
            result = app_navigator.back(screen)
            print(f"    Back to: {result}")
    
    # Application 3: Document editing history
    print(f"\nApplication 3: Document Version History")
    
    doc_history = BrowserHistoryArray("doc_v1.0")
    
    # Simulate document versions
    versions = [
        "doc_v1.1_added_intro",
        "doc_v1.2_fixed_typos", 
        "doc_v1.3_added_conclusion",
        "doc_v1.4_peer_review"
    ]
    
    print("  Document version history:")
    for version in versions:
        doc_history.visit(version)
        print(f"    Created: {version}")
    
    # Simulate going back to previous version
    print(f"  Reverting to previous versions:")
    for steps in [1, 2]:
        result = doc_history.back(steps)
        print(f"    Back {steps} version(s) to: {result}")
    
    # Continue from reverted version
    doc_history.visit("doc_v1.2_alternative_ending")
    print(f"    Branched to: doc_v1.2_alternative_ending")

def test_memory_optimization():
    """Test memory optimization features"""
    print("\n=== Testing Memory Optimization ===")
    
    # Test with limited history size
    limited_browser = BrowserHistoryMemoryOptimized("start.com", max_history_size=5)
    
    print("Testing limited history size (max 5 pages):")
    
    # Add more pages than limit
    pages = [f"page{i}.com" for i in range(10)]
    
    for i, page in enumerate(pages):
        limited_browser.visit(page)
        stats = limited_browser.get_memory_stats()
        print(f"  Visit {i+1}: {page}, size: {stats['current_size']}, "
              f"utilization: {stats['memory_utilization']:.1f}%")
    
    # Test navigation with limited history
    print(f"\nNavigation with limited history:")
    
    result = limited_browser.back(3)
    print(f"  Back 3 steps: {result}")
    
    result = limited_browser.forward(2)
    print(f"  Forward 2 steps: {result}")
    
    # Show final memory stats
    final_stats = limited_browser.get_memory_stats()
    print(f"\nFinal memory statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def stress_test_browser_history():
    """Stress test browser history"""
    print("\n=== Stress Testing Browser History ===")
    
    import time
    import random
    
    browser = BrowserHistoryArray("homepage.com")
    
    # Large scale test
    num_operations = 100000
    
    print(f"Stress test: {num_operations} operations")
    
    start_time = time.time()
    
    for i in range(num_operations):
        op_type = random.choice(['visit', 'back', 'forward'])
        
        if op_type == 'visit':
            # Visit new page
            browser.visit(f"page{i % 1000}.com")  # Cycle through 1000 different pages
        
        elif op_type == 'back':
            steps = random.randint(1, 10)
            browser.back(steps)
        
        else:  # forward
            steps = random.randint(1, 10)
            browser.forward(steps)
    
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  Completed in {elapsed:.2f}ms")
    print(f"  Average: {elapsed/num_operations:.4f}ms per operation")
    
    # Final state
    print(f"  Final history length: {len(browser.history)}")

def benchmark_navigation_patterns():
    """Benchmark different navigation patterns"""
    print("\n=== Benchmarking Navigation Patterns ===")
    
    import time
    
    patterns = [
        ("Sequential browsing", "visit"),
        ("Heavy back navigation", "back"),
        ("Heavy forward navigation", "forward"),
        ("Mixed navigation", "mixed")
    ]
    
    for pattern_name, pattern_type in patterns:
        browser = BrowserHistoryArray("start.com")
        
        # Setup for pattern
        if pattern_type in ["back", "forward", "mixed"]:
            # Pre-populate with pages
            for i in range(100):
                browser.visit(f"setup{i}.com")
        
        start_time = time.time()
        
        if pattern_type == "visit":
            # Sequential visits
            for i in range(1000):
                browser.visit(f"page{i}.com")
        
        elif pattern_type == "back":
            # Heavy back navigation
            for _ in range(1000):
                browser.back(random.randint(1, 5))
        
        elif pattern_type == "forward":
            # Go back first, then heavy forward
            browser.back(50)
            for _ in range(1000):
                browser.forward(random.randint(1, 3))
        
        else:  # mixed
            # Mixed pattern
            for _ in range(1000):
                op = random.choice(['visit', 'back', 'forward'])
                if op == 'visit':
                    browser.visit(f"mixed{random.randint(0, 100)}.com")
                elif op == 'back':
                    browser.back(random.randint(1, 3))
                else:
                    browser.forward(random.randint(1, 3))
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {pattern_name}: {elapsed:.2f}ms for 1000 operations")

def test_circular_buffer_efficiency():
    """Test circular buffer efficiency"""
    print("\n=== Testing Circular Buffer Efficiency ===")
    
    buffer_sizes = [10, 50, 100, 500]
    
    for size in buffer_sizes:
        browser = BrowserHistoryMemoryOptimized("start.com", max_history_size=size)
        
        # Fill beyond capacity
        num_pages = size * 2
        
        for i in range(num_pages):
            browser.visit(f"page{i}.com")
        
        # Test navigation
        back_result = browser.back(size // 2)
        forward_result = browser.forward(size // 4)
        
        stats = browser.get_memory_stats()
        
        print(f"  Buffer size {size}:")
        print(f"    Added {num_pages} pages, utilization: {stats['memory_utilization']:.1f}%")
        print(f"    Back navigation: {back_result}")
        print(f"    Forward navigation: {forward_result}")

if __name__ == "__main__":
    test_browser_history_basic()
    test_browser_history_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_optimization()
    stress_test_browser_history()
    benchmark_navigation_patterns()
    test_circular_buffer_efficiency()

"""
Browser History Design demonstrates key concepts:

Core Approaches:
1. Array-based - Dynamic array with current position pointer
2. Linked List - Doubly linked list for flexible navigation
3. Two Stacks - Separate stacks for back and forward history
4. Advanced - Enhanced with bookmarks, analytics, and features
5. Memory Optimized - Circular buffer with URL compression

Key Design Principles:
- History navigation and state management
- Memory vs functionality trade-offs
- Forward history clearing on new visits
- Efficient bidirectional traversal

Performance Characteristics:
- Array: O(1) all operations, O(n) space
- Linked List: O(1) visit, O(k) navigation where k is steps
- Two Stacks: O(1) all operations, optimal for step-by-step navigation
- Memory Optimized: Bounded space with circular buffer

Real-world Applications:
- Web browser back/forward functionality
- Mobile app screen navigation stacks
- Document version control and history
- IDE navigation and breadcrumbs
- Game state save/load systems
- Undo/redo systems with branching

The array-based approach is most commonly used
due to its simple implementation and optimal
O(1) performance for all operations.
"""
