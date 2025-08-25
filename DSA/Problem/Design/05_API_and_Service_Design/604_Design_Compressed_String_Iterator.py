"""
604. Design Compressed String Iterator - Multiple Approaches
Difficulty: Easy

Design and implement a data structure for a compressed string iterator. The given compressed string will be in the form of each letter followed by a positive integer representing the count of the letter.

Implement the StringIterator class:
- StringIterator(string compressedString) Initializes the object with the given compressed string compressedString.
- char next() Returns the next character in the original uncompressed string.
- boolean hasNext() Returns true if there are still unused characters in the compressed string, otherwise returns false.
"""

from typing import List, Tuple
import re

class StringIteratorSimple:
    """
    Approach 1: Parse and Store All Characters
    
    Parse compressed string and store all characters upfront.
    
    Time Complexity:
    - __init__: O(n) where n is total uncompressed length
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, compressedString: str):
        self.chars = []
        self.index = 0
        
        # Parse compressed string
        i = 0
        while i < len(compressedString):
            char = compressedString[i]
            i += 1
            
            # Parse number
            num_str = ""
            while i < len(compressedString) and compressedString[i].isdigit():
                num_str += compressedString[i]
                i += 1
            
            count = int(num_str)
            
            # Add characters
            self.chars.extend([char] * count)
    
    def next(self) -> str:
        if self.hasNext():
            char = self.chars[self.index]
            self.index += 1
            return char
        return ""
    
    def hasNext(self) -> bool:
        return self.index < len(self.chars)

class StringIteratorLazy:
    """
    Approach 2: Lazy Evaluation with State Tracking
    
    Parse on-demand without storing all characters.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1) amortized
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, compressedString: str):
        self.compressed = compressedString
        self.pos = 0  # Position in compressed string
        self.current_char = ""
        self.current_count = 0
        
        # Initialize first character
        self._load_next_char()
    
    def _load_next_char(self) -> None:
        """Load next character and count from compressed string"""
        if self.pos >= len(self.compressed):
            self.current_char = ""
            self.current_count = 0
            return
        
        # Get character
        self.current_char = self.compressed[self.pos]
        self.pos += 1
        
        # Parse count
        count_str = ""
        while self.pos < len(self.compressed) and self.compressed[self.pos].isdigit():
            count_str += self.compressed[self.pos]
            self.pos += 1
        
        self.current_count = int(count_str) if count_str else 0
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        char = self.current_char
        self.current_count -= 1
        
        # Load next character if current is exhausted
        if self.current_count == 0:
            self._load_next_char()
        
        return char
    
    def hasNext(self) -> bool:
        return self.current_count > 0

class StringIteratorRegex:
    """
    Approach 3: Regex-based Parsing
    
    Use regular expressions for cleaner parsing.
    
    Time Complexity:
    - __init__: O(k) where k is number of char-count pairs
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(k)
    """
    
    def __init__(self, compressedString: str):
        # Parse using regex
        pattern = r'([a-zA-Z])(\d+)'
        matches = re.findall(pattern, compressedString)
        
        self.segments = [(char, int(count)) for char, count in matches]
        self.segment_index = 0
        self.char_index = 0
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        char, count = self.segments[self.segment_index]
        self.char_index += 1
        
        # Move to next segment if current is exhausted
        if self.char_index >= count:
            self.segment_index += 1
            self.char_index = 0
        
        return char
    
    def hasNext(self) -> bool:
        return self.segment_index < len(self.segments)

class StringIteratorAdvanced:
    """
    Approach 4: Advanced with Features and Analytics
    
    Enhanced iterator with statistics and additional functionality.
    
    Time Complexity:
    - __init__: O(k)
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(k + analytics)
    """
    
    def __init__(self, compressedString: str):
        self.original_compressed = compressedString
        self.segments = self._parse_compressed(compressedString)
        
        # State
        self.segment_index = 0
        self.char_index = 0
        
        # Analytics
        self.total_chars = sum(count for _, count in self.segments)
        self.chars_consumed = 0
        self.next_calls = 0
        self.has_next_calls = 0
        
        # Features
        self.char_history = []
        self.segment_stats = {}
        
        # Calculate segment statistics
        for char, count in self.segments:
            if char not in self.segment_stats:
                self.segment_stats[char] = {'total_count': 0, 'segments': 0}
            self.segment_stats[char]['total_count'] += count
            self.segment_stats[char]['segments'] += 1
    
    def _parse_compressed(self, compressed: str) -> List[Tuple[str, int]]:
        """Parse compressed string into segments"""
        segments = []
        i = 0
        
        while i < len(compressed):
            if not compressed[i].isalpha():
                i += 1
                continue
            
            char = compressed[i]
            i += 1
            
            # Parse number
            num_str = ""
            while i < len(compressed) and compressed[i].isdigit():
                num_str += compressed[i]
                i += 1
            
            if num_str:
                count = int(num_str)
                segments.append((char, count))
        
        return segments
    
    def next(self) -> str:
        self.next_calls += 1
        
        if not self.hasNext():
            return ""
        
        char, count = self.segments[self.segment_index]
        self.char_index += 1
        self.chars_consumed += 1
        
        # Add to history
        self.char_history.append(char)
        
        # Move to next segment if needed
        if self.char_index >= count:
            self.segment_index += 1
            self.char_index = 0
        
        return char
    
    def hasNext(self) -> bool:
        self.has_next_calls += 1
        return self.segment_index < len(self.segments)
    
    def get_progress(self) -> dict:
        """Get iteration progress"""
        progress_percent = (self.chars_consumed / max(1, self.total_chars)) * 100
        
        return {
            'total_chars': self.total_chars,
            'chars_consumed': self.chars_consumed,
            'progress_percent': progress_percent,
            'current_segment': self.segment_index,
            'total_segments': len(self.segments)
        }
    
    def get_statistics(self) -> dict:
        """Get iterator statistics"""
        return {
            'next_calls': self.next_calls,
            'has_next_calls': self.has_next_calls,
            'chars_consumed': self.chars_consumed,
            'segment_stats': self.segment_stats.copy()
        }
    
    def peek(self) -> str:
        """Peek at next character without consuming"""
        if not self.hasNext():
            return ""
        
        char, _ = self.segments[self.segment_index]
        return char
    
    def get_remaining_in_segment(self) -> int:
        """Get remaining characters in current segment"""
        if not self.hasNext():
            return 0
        
        _, count = self.segments[self.segment_index]
        return count - self.char_index
    
    def skip(self, n: int) -> int:
        """Skip next n characters, return actual skipped count"""
        skipped = 0
        
        while skipped < n and self.hasNext():
            self.next()
            skipped += 1
        
        return skipped
    
    def reset(self) -> None:
        """Reset iterator to beginning"""
        self.segment_index = 0
        self.char_index = 0
        self.chars_consumed = 0
        self.next_calls = 0
        self.has_next_calls = 0
        self.char_history.clear()

class StringIteratorMemoryOptimized:
    """
    Approach 5: Memory-Optimized for Large Strings
    
    Minimal memory footprint for very large compressed strings.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, compressedString: str):
        self.compressed = compressedString
        self.pos = 0
        self.current_char = None
        self.remaining_count = 0
        
        # Preload first segment
        self._advance_segment()
    
    def _advance_segment(self) -> None:
        """Advance to next segment in compressed string"""
        if self.pos >= len(self.compressed):
            self.current_char = None
            self.remaining_count = 0
            return
        
        # Skip non-alphabetic characters
        while self.pos < len(self.compressed) and not self.compressed[self.pos].isalpha():
            self.pos += 1
        
        if self.pos >= len(self.compressed):
            self.current_char = None
            self.remaining_count = 0
            return
        
        # Read character
        self.current_char = self.compressed[self.pos]
        self.pos += 1
        
        # Read count
        count_str = ""
        while self.pos < len(self.compressed) and self.compressed[self.pos].isdigit():
            count_str += self.compressed[self.pos]
            self.pos += 1
        
        self.remaining_count = int(count_str) if count_str else 0
    
    def next(self) -> str:
        if not self.hasNext():
            return ""
        
        char = self.current_char
        self.remaining_count -= 1
        
        # Advance to next segment if current is exhausted
        if self.remaining_count == 0:
            self._advance_segment()
        
        return char
    
    def hasNext(self) -> bool:
        return self.remaining_count > 0 and self.current_char is not None


def test_string_iterator_basic():
    """Test basic string iterator functionality"""
    print("=== Testing Basic String Iterator Functionality ===")
    
    implementations = [
        ("Simple", StringIteratorSimple),
        ("Lazy", StringIteratorLazy),
        ("Regex", StringIteratorRegex),
        ("Advanced", StringIteratorAdvanced),
        ("Memory Optimized", StringIteratorMemoryOptimized)
    ]
    
    test_cases = [
        "L1e2t1C1o1d1e1",
        "x2y3z1",
        "a10",
        "b1c2"
    ]
    
    for compressed in test_cases:
        print(f"\nTest case: '{compressed}'")
        
        # Generate expected output
        expected = []
        i = 0
        while i < len(compressed):
            if compressed[i].isalpha():
                char = compressed[i]
                i += 1
                
                num_str = ""
                while i < len(compressed) and compressed[i].isdigit():
                    num_str += compressed[i]
                    i += 1
                
                count = int(num_str) if num_str else 0
                expected.extend([char] * count)
            else:
                i += 1
        
        for name, IteratorClass in implementations:
            try:
                iterator = IteratorClass(compressed)
                result = []
                
                while iterator.hasNext():
                    result.append(iterator.next())
                
                correct = result == expected
                print(f"  {name}: {''.join(result)} - {'✓' if correct else '✗'}")
                
            except Exception as e:
                print(f"  {name}: Error - {e}")

def test_string_iterator_edge_cases():
    """Test edge cases"""
    print("\n=== Testing String Iterator Edge Cases ===")
    
    # Test empty/invalid inputs
    edge_cases = [
        ("", "Empty string"),
        ("a0", "Zero count"),
        ("123", "Numbers only"),
        ("abc", "Letters only"),
        ("a1b2c3d4e5", "Multiple single chars")
    ]
    
    for compressed, description in edge_cases:
        print(f"\n{description}: '{compressed}'")
        
        try:
            iterator = StringIteratorLazy(compressed)
            
            result = []
            count = 0
            while iterator.hasNext() and count < 20:  # Limit to prevent infinite loop
                result.append(iterator.next())
                count += 1
            
            print(f"  Result: {''.join(result)}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test calling next() when hasNext() is false
    print(f"\nCalling next() when exhausted:")
    iterator = StringIteratorAdvanced("a3")
    
    # Consume all characters
    while iterator.hasNext():
        iterator.next()
    
    # Try to get one more
    result = iterator.next()
    print(f"  next() after exhaustion: '{result}'")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    iterator = StringIteratorAdvanced("a5b3c2d1")
    
    # Test progress tracking
    print("Progress tracking:")
    
    for i in range(6):
        if iterator.hasNext():
            char = iterator.next()
            progress = iterator.get_progress()
            
            print(f"  Step {i+1}: '{char}' - Progress: {progress['progress_percent']:.1f}%")
    
    # Test peek functionality
    print(f"\nPeek functionality:")
    
    if iterator.hasNext():
        peeked = iterator.peek()
        next_char = iterator.next()
        
        print(f"  Peeked: '{peeked}'")
        print(f"  Next: '{next_char}'")
        print(f"  Match: {peeked == next_char}")
    
    # Test remaining in segment
    remaining = iterator.get_remaining_in_segment()
    print(f"  Remaining in current segment: {remaining}")
    
    # Test skip functionality
    print(f"\nSkip functionality:")
    
    skipped = iterator.skip(3)
    print(f"  Requested to skip 3, actually skipped: {skipped}")
    
    # Test statistics
    stats = iterator.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Test reset
    print(f"\nReset functionality:")
    
    before_reset = iterator.get_progress()
    iterator.reset()
    after_reset = iterator.get_progress()
    
    print(f"  Before reset: {before_reset['chars_consumed']} consumed")
    print(f"  After reset: {after_reset['chars_consumed']} consumed")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Data decompression
    print("Application 1: Data Decompression")
    
    compressed_data = "H3e2l2o1W1o1r1l1d1"  # "HHHeeelloWorld"
    
    decompressor = StringIteratorAdvanced(compressed_data)
    
    print(f"  Compressed: '{compressed_data}'")
    print(f"  Decompressing...")
    
    decompressed = []
    while decompressor.hasNext():
        decompressed.append(decompressor.next())
    
    print(f"  Decompressed: '{''.join(decompressed)}'")
    
    stats = decompressor.get_statistics()
    print(f"  Compression ratio: {len(compressed_data)} -> {len(decompressed)} chars")
    
    # Application 2: Pattern generation
    print(f"\nApplication 2: Pattern Generation")
    
    pattern_string = "A3B2C1A3B2C1"  # Repeating pattern
    pattern_gen = StringIteratorLazy(pattern_string)
    
    print(f"  Pattern: '{pattern_string}'")
    print(f"  Generated sequence:")
    
    sequence = []
    count = 0
    while pattern_gen.hasNext() and count < 20:
        sequence.append(pattern_gen.next())
        count += 1
    
    print(f"    {''.join(sequence)}")
    
    # Application 3: Run-length encoded text processing
    print(f"\nApplication 3: Run-Length Encoded Text")
    
    # Simulate text with repeated characters
    encoded_text = "T1h1e1 5q1u1i1c1k1 5b1r1o1w1n1 5f1o1x1"  # "The     quick     brown     fox"
    
    text_processor = StringIteratorMemoryOptimized(encoded_text)
    
    print(f"  Encoded: '{encoded_text}'")
    
    processed_text = []
    while text_processor.hasNext():
        processed_text.append(text_processor.next())
    
    print(f"  Decoded: '{''.join(processed_text)}'")
    
    # Application 4: Animation frame generation
    print(f"\nApplication 4: Animation Frame Generation")
    
    # Each character represents a frame, number is duration
    animation_sequence = "A2B3C2D1E4"  # A(2 frames), B(3 frames), etc.
    
    animator = StringIteratorAdvanced(animation_sequence)
    
    print(f"  Animation sequence: '{animation_sequence}'")
    print(f"  Frame timeline:")
    
    frame_number = 0
    while animator.hasNext() and frame_number < 15:
        frame = animator.next()
        frame_number += 1
        print(f"    Frame {frame_number}: {frame}")
    
    # Show animation statistics
    anim_stats = animator.get_statistics()
    print(f"  Animation stats: {anim_stats['segment_stats']}")

def test_performance():
    """Test performance with different string sizes"""
    print("\n=== Testing Performance ===")
    
    import time
    
    implementations = [
        ("Simple", StringIteratorSimple),
        ("Lazy", StringIteratorLazy),
        ("Memory Optimized", StringIteratorMemoryOptimized)
    ]
    
    # Generate test strings of different sizes
    test_strings = [
        ("Small", "a10b10c10"),
        ("Medium", "a100b100c100d100e100"),
        ("Large", "x1000y1000z1000"),
        ("Very Large", "a5000b5000")
    ]
    
    for size_name, compressed in test_strings:
        print(f"\n{size_name}: '{compressed}'")
        
        for impl_name, IteratorClass in implementations:
            # Time initialization
            start_time = time.time()
            iterator = IteratorClass(compressed)
            init_time = (time.time() - start_time) * 1000
            
            # Time iteration
            start_time = time.time()
            count = 0
            while iterator.hasNext():
                iterator.next()
                count += 1
            iteration_time = (time.time() - start_time) * 1000
            
            print(f"  {impl_name}:")
            print(f"    Init: {init_time:.3f}ms")
            print(f"    Iteration: {iteration_time:.2f}ms")
            print(f"    Characters: {count}")

def test_memory_usage():
    """Test memory usage patterns"""
    print("\n=== Testing Memory Usage ===")
    
    # Compare memory usage for large strings
    large_compressed = "a10000b10000c10000d10000e10000"
    
    implementations = [
        ("Simple (precomputed)", StringIteratorSimple),
        ("Lazy (on-demand)", StringIteratorLazy),
        ("Memory Optimized", StringIteratorMemoryOptimized)
    ]
    
    print(f"Large string: '{large_compressed}' (50,000 total chars)")
    
    for name, IteratorClass in implementations:
        iterator = IteratorClass(large_compressed)
        
        # Estimate memory usage based on approach
        if name.startswith("Simple"):
            # Stores all characters
            estimated_memory = 50000  # All characters stored
            approach = "All chars stored"
        elif name.startswith("Lazy"):
            # Stores parsed segments
            estimated_memory = 10  # Just segment info
            approach = "Lazy parsing"
        else:
            # Minimal state
            estimated_memory = 3  # Current char + count + position
            approach = "Minimal state"
        
        print(f"  {name}: ~{estimated_memory} units ({approach})")
        
        # Test first few characters to verify correctness
        sample = []
        for _ in range(5):
            if iterator.hasNext():
                sample.append(iterator.next())
        
        print(f"    First 5 chars: {''.join(sample)}")

def stress_test_string_iterator():
    """Stress test string iterator"""
    print("\n=== Stress Testing String Iterator ===")
    
    import time
    
    # Create very large compressed string
    segments = []
    for i in range(1000):  # 1000 segments
        char = chr(ord('a') + (i % 26))
        count = 100
        segments.append(f"{char}{count}")
    
    large_compressed = "".join(segments)
    total_chars = 1000 * 100  # 100,000 characters
    
    print(f"Stress test: {len(large_compressed)} compressed -> {total_chars} chars")
    
    # Test lazy iterator (most memory efficient)
    iterator = StringIteratorLazy(large_compressed)
    
    start_time = time.time()
    
    count = 0
    checksum = 0
    
    while iterator.hasNext():
        char = iterator.next()
        checksum += ord(char)
        count += 1
        
        # Progress update every 10k characters
        if count % 10000 == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"    {count} chars processed, {rate:.0f} chars/sec")
    
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Characters processed: {count}")
    print(f"  Rate: {count / total_time:.0f} chars/sec")
    print(f"  Checksum: {checksum}")

def benchmark_parsing_methods():
    """Benchmark different parsing methods"""
    print("\n=== Benchmarking Parsing Methods ===")
    
    import time
    
    test_string = "a123b456c789d1000e2000f3000g4000h5000"
    
    parsing_methods = [
        ("Manual parsing", StringIteratorLazy),
        ("Regex parsing", StringIteratorRegex)
    ]
    
    num_iterations = 1000
    
    for method_name, IteratorClass in parsing_methods:
        start_time = time.time()
        
        for _ in range(num_iterations):
            iterator = IteratorClass(test_string)
            
            # Consume a few characters to test parsing
            for _ in range(10):
                if iterator.hasNext():
                    iterator.next()
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {method_name}: {elapsed:.2f}ms for {num_iterations} iterations")
        print(f"    Average: {elapsed/num_iterations:.3f}ms per iteration")

def test_compression_ratios():
    """Test with different compression ratios"""
    print("\n=== Testing Compression Ratios ===")
    
    compression_tests = [
        ("High compression", "a1000", "Single char repeated"),
        ("Low compression", "a1b1c1d1e1f1g1h1", "Many different chars"),
        ("Mixed compression", "a100b1c1d100e1f1", "Mixed patterns"),
        ("No compression", "abcdefghijk", "All unique chars")
    ]
    
    for test_name, compressed, description in compression_tests:
        iterator = StringIteratorAdvanced(compressed)
        
        # Count total uncompressed length
        total_chars = 0
        while iterator.hasNext():
            iterator.next()
            total_chars += 1
        
        compression_ratio = len(compressed) / max(1, total_chars)
        
        print(f"  {test_name}: '{compressed}'")
        print(f"    {description}")
        print(f"    Ratio: {len(compressed)} -> {total_chars} chars ({compression_ratio:.3f})")

if __name__ == "__main__":
    test_string_iterator_basic()
    test_string_iterator_edge_cases()
    test_advanced_features()
    demonstrate_applications()
    test_performance()
    test_memory_usage()
    stress_test_string_iterator()
    benchmark_parsing_methods()
    test_compression_ratios()

"""
Compressed String Iterator Design demonstrates key concepts:

Core Approaches:
1. Simple - Parse and store all characters upfront for O(1) access
2. Lazy - Parse on-demand with minimal state tracking
3. Regex - Use regular expressions for cleaner parsing logic
4. Advanced - Enhanced with analytics, progress tracking, and utilities
5. Memory Optimized - Minimal memory footprint for large strings

Key Design Principles:
- Lazy evaluation vs eager preprocessing trade-offs
- Iterator pattern with hasNext/next interface
- String parsing and state management
- Memory efficiency for large datasets

Performance Characteristics:
- Simple: O(n) initialization, O(1) operations, O(n) space
- Lazy: O(1) initialization, O(1) amortized operations, O(1) space
- Regex: O(k) initialization (k=segments), O(1) operations
- Memory Optimized: Minimal space overhead

Real-world Applications:
- Data decompression and file format processing
- Pattern generation for graphics and animations
- Run-length encoded text and media processing
- Network protocol message parsing
- Game development for sprite animations
- Compression algorithm implementations

The lazy evaluation approach provides the best balance
of memory efficiency and performance for most use cases,
especially when dealing with large compressed strings
where only portions may be consumed.
"""
