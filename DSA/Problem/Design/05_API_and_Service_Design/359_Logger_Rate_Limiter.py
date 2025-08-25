"""
359. Logger Rate Limiter - Multiple Approaches
Difficulty: Easy

Design a logger system that receives a stream of messages along with their timestamps. Each unique message should only be printed at most every 10 seconds (i.e. a message printed at timestamp t will prevent other identical messages from being printed until timestamp t + 10).

All messages will come in chronological order. Several messages may arrive at the same timestamp.

Implement the Logger class:
- Logger() Initializes the logger object.
- bool shouldPrintMessage(timestamp, message) Returns true if the message should be printed in the given timestamp, otherwise returns false.
"""

from typing import Dict, List
from collections import deque, defaultdict
import time

class LoggerSimple:
    """
    Approach 1: Simple HashMap
    
    Store last print time for each message.
    
    Time Complexity:
    - shouldPrintMessage: O(1)
    
    Space Complexity: O(n) where n is unique messages
    """
    
    def __init__(self):
        self.last_printed = {}  # message -> last_timestamp
    
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.last_printed:
            self.last_printed[message] = timestamp
            return True
        
        if timestamp - self.last_printed[message] >= 10:
            self.last_printed[message] = timestamp
            return True
        
        return False

class LoggerWithCleanup:
    """
    Approach 2: HashMap with Periodic Cleanup
    
    Clean up old entries to prevent memory bloat.
    
    Time Complexity:
    - shouldPrintMessage: O(1) amortized
    
    Space Complexity: O(active_messages)
    """
    
    def __init__(self):
        self.last_printed = {}
        self.cleanup_interval = 100  # Cleanup every 100 operations
        self.operation_count = 0
    
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        self.operation_count += 1
        
        # Periodic cleanup
        if self.operation_count % self.cleanup_interval == 0:
            self._cleanup(timestamp)
        
        if message not in self.last_printed:
            self.last_printed[message] = timestamp
            return True
        
        if timestamp - self.last_printed[message] >= 10:
            self.last_printed[message] = timestamp
            return True
        
        return False
    
    def _cleanup(self, current_timestamp: int) -> None:
        """Remove entries older than 10 seconds"""
        to_remove = []
        for message, last_time in self.last_printed.items():
            if current_timestamp - last_time >= 10:
                to_remove.append(message)
        
        for message in to_remove:
            del self.last_printed[message]

class LoggerSlidingWindow:
    """
    Approach 3: Sliding Window with Deque
    
    Use sliding window approach to track recent messages.
    
    Time Complexity:
    - shouldPrintMessage: O(k) where k is messages in window
    
    Space Complexity: O(k)
    """
    
    def __init__(self):
        self.message_queue = deque()  # (timestamp, message)
    
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        # Remove messages older than 10 seconds
        while self.message_queue and timestamp - self.message_queue[0][0] >= 10:
            self.message_queue.popleft()
        
        # Check if message already exists in window
        for ts, msg in self.message_queue:
            if msg == message:
                return False
        
        # Add message to window and allow printing
        self.message_queue.append((timestamp, message))
        return True

class LoggerAdvanced:
    """
    Approach 4: Advanced with Features and Analytics
    
    Enhanced logger with statistics and additional functionality.
    
    Time Complexity:
    - shouldPrintMessage: O(1) amortized
    
    Space Complexity: O(unique_messages + analytics)
    """
    
    def __init__(self, rate_limit_seconds: int = 10):
        self.rate_limit = rate_limit_seconds
        self.last_printed = {}
        
        # Analytics
        self.message_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_prints': 0,
            'first_seen': None,
            'last_attempt': None
        })
        
        self.global_stats = {
            'total_messages': 0,
            'unique_messages': 0,
            'printed_messages': 0,
            'rate_limited': 0
        }
        
        # Features
        self.message_history = deque(maxlen=1000)  # Recent message history
        self.current_timestamp = 0
    
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        self.current_timestamp = timestamp
        self.global_stats['total_messages'] += 1
        
        # Update message statistics
        if message not in self.message_stats:
            self.global_stats['unique_messages'] += 1
            self.message_stats[message]['first_seen'] = timestamp
        
        self.message_stats[message]['total_attempts'] += 1
        self.message_stats[message]['last_attempt'] = timestamp
        
        # Check rate limiting
        should_print = False
        
        if message not in self.last_printed:
            should_print = True
        elif timestamp - self.last_printed[message] >= self.rate_limit:
            should_print = True
        
        if should_print:
            self.last_printed[message] = timestamp
            self.message_stats[message]['successful_prints'] += 1
            self.global_stats['printed_messages'] += 1
            
            # Add to history
            self.message_history.append((timestamp, message, 'PRINTED'))
        else:
            self.global_stats['rate_limited'] += 1
            self.message_history.append((timestamp, message, 'RATE_LIMITED'))
        
        return should_print
    
    def getMessageStats(self, message: str) -> dict:
        """Get statistics for a specific message"""
        if message not in self.message_stats:
            return {}
        
        stats = self.message_stats[message].copy()
        if stats['total_attempts'] > 0:
            stats['print_rate'] = stats['successful_prints'] / stats['total_attempts']
        
        return stats
    
    def getGlobalStats(self) -> dict:
        """Get global logger statistics"""
        stats = self.global_stats.copy()
        if stats['total_messages'] > 0:
            stats['overall_print_rate'] = stats['printed_messages'] / stats['total_messages']
        
        return stats
    
    def getRecentActivity(self, limit: int = 10) -> List[tuple]:
        """Get recent message activity"""
        return list(self.message_history)[-limit:]
    
    def getTopMessages(self, limit: int = 5) -> List[tuple]:
        """Get most frequent messages"""
        message_counts = [(msg, stats['total_attempts']) 
                         for msg, stats in self.message_stats.items()]
        message_counts.sort(key=lambda x: x[1], reverse=True)
        return message_counts[:limit]
    
    def cleanup(self) -> int:
        """Clean up old entries and return count removed"""
        removed_count = 0
        to_remove = []
        
        for message, last_time in self.last_printed.items():
            if self.current_timestamp - last_time >= self.rate_limit:
                to_remove.append(message)
        
        for message in to_remove:
            del self.last_printed[message]
            removed_count += 1
        
        return removed_count

class LoggerMemoryOptimized:
    """
    Approach 5: Memory-Optimized with LRU Eviction
    
    Use LRU cache to limit memory usage.
    
    Time Complexity:
    - shouldPrintMessage: O(1)
    
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.last_printed = {}
        self.access_order = deque()  # For LRU tracking
        self.message_positions = {}  # message -> position in deque
    
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        # Check if we should print
        should_print = False
        
        if message not in self.last_printed:
            should_print = True
        elif timestamp - self.last_printed[message] >= 10:
            should_print = True
        
        if should_print:
            # Update last printed time
            self.last_printed[message] = timestamp
            
            # Update LRU order
            self._update_access(message)
            
            # Evict if over capacity
            if len(self.last_printed) > self.capacity:
                self._evict_lru()
        
        return should_print
    
    def _update_access(self, message: str) -> None:
        """Update access order for LRU"""
        # Remove from current position if exists
        if message in self.message_positions:
            # For simplicity, we'll just append (not optimal but works)
            pass
        
        # Add to end (most recent)
        self.access_order.append(message)
        self.message_positions[message] = len(self.access_order) - 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used message"""
        if self.access_order:
            # Find oldest message that's still in last_printed
            while self.access_order:
                old_message = self.access_order.popleft()
                if old_message in self.last_printed:
                    del self.last_printed[old_message]
                    del self.message_positions[old_message]
                    break


def test_logger_basic():
    """Test basic logger functionality"""
    print("=== Testing Basic Logger Functionality ===")
    
    implementations = [
        ("Simple", LoggerSimple),
        ("With Cleanup", LoggerWithCleanup),
        ("Sliding Window", LoggerSlidingWindow),
        ("Advanced", LoggerAdvanced),
        ("Memory Optimized", LoggerMemoryOptimized)
    ]
    
    test_cases = [
        (1, "foo", True),
        (2, "bar", True),
        (3, "foo", False),  # Within 10 seconds
        (8, "bar", False),  # Within 10 seconds
        (10, "foo", False), # Exactly 10 seconds (< 10)
        (11, "foo", True),  # More than 10 seconds
        (12, "bar", True),  # More than 10 seconds
    ]
    
    for name, LoggerClass in implementations:
        print(f"\n{name}:")
        
        logger = LoggerClass()
        
        for timestamp, message, expected in test_cases:
            result = logger.shouldPrintMessage(timestamp, message)
            status = "✓" if result == expected else "✗"
            print(f"  shouldPrintMessage({timestamp}, '{message}'): {result} {status}")

def test_logger_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Logger Edge Cases ===")
    
    logger = LoggerAdvanced()
    
    # Test same timestamp, different messages
    print("Same timestamp, different messages:")
    result1 = logger.shouldPrintMessage(1, "msg1")
    result2 = logger.shouldPrintMessage(1, "msg2")
    print(f"  msg1 at t=1: {result1}")
    print(f"  msg2 at t=1: {result2}")
    
    # Test same timestamp, same message
    print(f"\nSame timestamp, same message:")
    result3 = logger.shouldPrintMessage(1, "msg1")
    print(f"  msg1 again at t=1: {result3}")
    
    # Test large timestamp gap
    print(f"\nLarge timestamp gap:")
    result4 = logger.shouldPrintMessage(1000, "msg1")
    print(f"  msg1 at t=1000: {result4}")
    
    # Test boundary conditions
    print(f"\nBoundary conditions:")
    logger2 = LoggerSimple()
    
    logger2.shouldPrintMessage(0, "test")
    
    boundary_tests = [9, 10, 11]
    for ts in boundary_tests:
        result = logger2.shouldPrintMessage(ts, "test")
        print(f"  test at t={ts}: {result}")

def test_advanced_features():
    """Test advanced logger features"""
    print("\n=== Testing Advanced Features ===")
    
    logger = LoggerAdvanced(rate_limit_seconds=5)  # 5-second rate limit
    
    # Generate test activity
    test_messages = [
        (1, "ERROR: Database connection failed"),
        (2, "INFO: User login successful"),
        (3, "ERROR: Database connection failed"),
        (4, "WARNING: High memory usage"),
        (6, "ERROR: Database connection failed"),
        (7, "INFO: User login successful"),
        (8, "ERROR: Network timeout"),
        (10, "ERROR: Database connection failed"),
        (12, "INFO: User login successful")
    ]
    
    print("Processing test messages:")
    for timestamp, message in test_messages:
        should_print = logger.shouldPrintMessage(timestamp, message)
        print(f"  t={timestamp}: {message[:30]}... -> {should_print}")
    
    # Get statistics
    global_stats = logger.getGlobalStats()
    print(f"\nGlobal statistics:")
    for key, value in global_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Get top messages
    top_messages = logger.getTopMessages(3)
    print(f"\nTop messages:")
    for i, (message, count) in enumerate(top_messages):
        print(f"  {i+1}. {message[:40]}... ({count} attempts)")
    
    # Get message-specific stats
    error_msg = "ERROR: Database connection failed"
    error_stats = logger.getMessageStats(error_msg)
    print(f"\nStats for '{error_msg}':")
    for key, value in error_stats.items():
        print(f"  {key}: {value}")
    
    # Get recent activity
    recent = logger.getRecentActivity(5)
    print(f"\nRecent activity (last 5):")
    for timestamp, message, status in recent:
        print(f"  t={timestamp}: {message[:30]}... [{status}]")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: System monitoring and alerting
    print("Application 1: System Monitoring")
    
    monitor_logger = LoggerAdvanced(rate_limit_seconds=30)  # 30-second rate limit
    
    # Simulate system events
    system_events = [
        (100, "CRITICAL: CPU usage above 90%"),
        (105, "WARNING: Memory usage at 80%"),
        (110, "CRITICAL: CPU usage above 90%"),  # Duplicate within 30s
        (115, "ERROR: Disk space low"),
        (120, "INFO: Backup completed successfully"),
        (125, "WARNING: Memory usage at 80%"),   # Duplicate within 30s
        (135, "CRITICAL: CPU usage above 90%"),  # After 30s, should print
        (140, "ERROR: Service timeout")
    ]
    
    print("  System monitoring events:")
    alerts_sent = 0
    
    for timestamp, event in system_events:
        should_alert = monitor_logger.shouldPrintMessage(timestamp, event)
        
        if should_alert:
            alerts_sent += 1
            print(f"    🚨 ALERT SENT t={timestamp}: {event}")
        else:
            print(f"    ⏸️  SUPPRESSED t={timestamp}: {event}")
    
    print(f"  Total alerts sent: {alerts_sent} out of {len(system_events)} events")
    
    # Application 2: Application error logging
    print(f"\nApplication 2: Application Error Logging")
    
    error_logger = LoggerWithCleanup()
    
    # Simulate application errors
    app_errors = [
        (200, "NullPointerException in UserService.getUserById()"),
        (201, "ConnectionTimeout connecting to payment service"),
        (203, "NullPointerException in UserService.getUserById()"),
        (205, "ValidationError: Invalid email format"),
        (207, "ConnectionTimeout connecting to payment service"),
        (212, "NullPointerException in UserService.getUserById()"),  # After 10s
        (215, "OutOfMemoryError in data processing module"),
        (220, "ValidationError: Invalid email format")
    ]
    
    print("  Application error logging:")
    logged_errors = 0
    
    for timestamp, error in app_errors:
        should_log = error_logger.shouldPrintMessage(timestamp, error)
        
        if should_log:
            logged_errors += 1
            print(f"    📝 LOGGED t={timestamp}: {error[:50]}...")
        else:
            print(f"    🔇 FILTERED t={timestamp}: {error[:50]}...")
    
    print(f"  Errors logged: {logged_errors} out of {len(app_errors)} total errors")
    
    # Application 3: Chat spam prevention
    print(f"\nApplication 3: Chat Spam Prevention")
    
    chat_filter = LoggerMemoryOptimized(capacity=100)
    
    # Simulate chat messages
    chat_messages = [
        (300, "Hello everyone!"),
        (301, "Check out this amazing deal: www.spam.com"),
        (302, "How is everyone doing?"),
        (303, "Check out this amazing deal: www.spam.com"),  # Spam repeat
        (305, "Hello everyone!"),  # Repeat within 10s
        (310, "What's the weather like?"),
        (312, "Check out this amazing deal: www.spam.com"),  # After 10s
        (315, "Hello everyone!"),  # After 10s
        (320, "Anyone want to play a game?")
    ]
    
    print("  Chat spam filtering:")
    messages_allowed = 0
    
    for timestamp, message in chat_messages:
        allowed = chat_filter.shouldPrintMessage(timestamp, message)
        
        if allowed:
            messages_allowed += 1
            print(f"    💬 ALLOWED t={timestamp}: {message}")
        else:
            print(f"    🚫 BLOCKED t={timestamp}: {message}")
    
    print(f"  Messages allowed: {messages_allowed} out of {len(chat_messages)} messages")

def test_performance():
    """Test performance with different message patterns"""
    print("\n=== Testing Performance ===")
    
    import time
    
    implementations = [
        ("Simple", LoggerSimple),
        ("With Cleanup", LoggerWithCleanup),
        ("Advanced", LoggerAdvanced)
    ]
    
    # Test different patterns
    patterns = [
        ("Unique messages", lambda i: f"message_{i}"),
        ("Repeated messages", lambda i: f"message_{i % 10}"),
        ("High duplication", lambda i: f"message_{i % 3}")
    ]
    
    for pattern_name, message_gen in patterns:
        print(f"\n{pattern_name}:")
        
        for impl_name, LoggerClass in implementations:
            logger = LoggerClass()
            
            start_time = time.time()
            
            # Process 10000 messages
            for i in range(10000):
                timestamp = i
                message = message_gen(i)
                logger.shouldPrintMessage(timestamp, message)
            
            elapsed = (time.time() - start_time) * 1000
            
            print(f"  {impl_name}: {elapsed:.2f}ms for 10k messages")

def stress_test_logger():
    """Stress test logger with high load"""
    print("\n=== Stress Testing Logger ===")
    
    import time
    import random
    
    logger = LoggerAdvanced()
    
    # High load test
    num_messages = 100000
    message_templates = [
        "ERROR: Database connection failed",
        "WARNING: High memory usage detected",
        "INFO: User authentication successful",
        "CRITICAL: Service unavailable",
        "DEBUG: Processing user request",
        "WARN: Slow query detected",
        "ERROR: Network timeout occurred",
        "INFO: Cache refresh completed"
    ]
    
    print(f"Stress test: {num_messages} messages")
    
    start_time = time.time()
    
    printed_count = 0
    
    for i in range(num_messages):
        timestamp = i // 100  # Slow down timestamp progression
        message = random.choice(message_templates)
        
        if logger.shouldPrintMessage(timestamp, message):
            printed_count += 1
        
        # Progress update
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"    {i + 1} messages processed, {rate:.0f} msgs/sec")
    
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Rate: {num_messages / total_time:.0f} messages/sec")
    print(f"  Messages printed: {printed_count} out of {num_messages}")
    
    # Get final statistics
    final_stats = logger.getGlobalStats()
    print(f"  Print rate: {final_stats.get('overall_print_rate', 0):.3f}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Simple", LoggerSimple),
        ("Memory Optimized", LoggerMemoryOptimized)
    ]
    
    # Test with many unique messages
    for impl_name, LoggerClass in implementations:
        if LoggerClass == LoggerMemoryOptimized:
            logger = LoggerClass(capacity=100)  # Limited capacity
        else:
            logger = LoggerClass()
        
        # Send many unique messages
        num_unique = 1000
        timestamp = 0
        
        for i in range(num_unique):
            message = f"unique_message_{i}"
            logger.shouldPrintMessage(timestamp, message)
            timestamp += 1  # Each message at different time
        
        # Estimate memory usage
        if hasattr(logger, 'last_printed'):
            memory_usage = len(logger.last_printed)
        else:
            memory_usage = num_unique  # Estimate
        
        print(f"  {impl_name}: {memory_usage} entries for {num_unique} unique messages")

def test_cleanup_behavior():
    """Test cleanup behavior"""
    print("\n=== Testing Cleanup Behavior ===")
    
    logger = LoggerWithCleanup()
    
    # Add messages over time
    messages = []
    for i in range(20):
        message = f"message_{i}"
        timestamp = i * 5  # Every 5 seconds
        
        logger.shouldPrintMessage(timestamp, message)
        messages.append((timestamp, message))
    
    print(f"Added {len(messages)} messages over time")
    print(f"Last timestamp: {messages[-1][0]}")
    
    # Check how many are still in memory before cleanup
    before_cleanup = len(logger.last_printed)
    
    # Trigger cleanup by sending message at much later time
    future_timestamp = 200  # Way in the future
    logger.shouldPrintMessage(future_timestamp, "future_message")
    
    after_cleanup = len(logger.last_printed)
    
    print(f"Before cleanup: {before_cleanup} entries")
    print(f"After cleanup: {after_cleanup} entries")
    print(f"Cleaned up: {before_cleanup - after_cleanup} entries")

if __name__ == "__main__":
    test_logger_basic()
    test_logger_edge_cases()
    test_advanced_features()
    demonstrate_applications()
    test_performance()
    stress_test_logger()
    test_memory_efficiency()
    test_cleanup_behavior()

"""
Logger Rate Limiter Design demonstrates key concepts:

Core Approaches:
1. Simple - Basic HashMap storing last print time per message
2. With Cleanup - Periodic cleanup to prevent memory bloat
3. Sliding Window - Deque-based window for tracking recent messages
4. Advanced - Enhanced with analytics, statistics, and features
5. Memory Optimized - LRU eviction to limit memory usage

Key Design Principles:
- Rate limiting for duplicate message suppression
- Memory management for long-running systems
- Efficient timestamp-based filtering
- Analytics and monitoring capabilities

Performance Characteristics:
- Simple: O(1) per operation, unbounded memory growth
- With Cleanup: O(1) amortized, periodic cleanup overhead
- Sliding Window: O(k) per operation where k is window size
- Advanced: O(1) with analytics overhead

Real-world Applications:
- System monitoring and alerting (prevent alert spam)
- Application error logging (reduce log noise)
- Chat and messaging spam prevention
- API rate limiting and abuse detection
- Security event deduplication
- Performance monitoring dashboards

The simple HashMap approach is most commonly used
for basic rate limiting, while advanced implementations
provide comprehensive monitoring and analytics capabilities
needed for production logging systems.
"""
