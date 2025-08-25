"""
1032. Stream of Characters - Multiple Approaches
Difficulty: Medium

Design an algorithm that accepts a stream of characters and checks if a suffix of 
these characters matches any string in a given list of words.

Implement the StreamChecker class:
- StreamChecker(String[] words) Initializes the object with the strings array words.
- boolean query(char letter) Accepts a new character from the stream and returns true 
  if any non-empty suffix of the characters received so far matches any of the strings 
  in words.

LeetCode Problem: https://leetcode.com/problems/stream-of-characters/

Example:
StreamChecker streamChecker = new StreamChecker(["cd", "f", "kl"]);
streamChecker.query("a"); // return False
streamChecker.query("b"); // return False  
streamChecker.query("c"); // return False
streamChecker.query("d"); // return True, because 'cd' is in the wordlist
streamChecker.query("e"); // return False
streamChecker.query("f"); // return True, because 'f' is in the wordlist
streamChecker.query("g"); // return False
streamChecker.query("h"); // return False
streamChecker.query("i"); // return False
streamChecker.query("j"); // return False
streamChecker.query("k"); // return False
streamChecker.query("l"); // return True, because 'kl' is in the wordlist
"""

from typing import List, Set, Dict, Optional, Deque
from collections import deque, defaultdict
import time

class TrieNode:
    """Trie node for stream checking"""
    def __init__(self):
        self.children = {}
        self.is_word = False

class StreamChecker1:
    """
    Approach 1: Reverse Trie with Stream Buffer
    
    Build reverse trie and check suffixes efficiently.
    
    Time: O(k) per query where k is max word length
    Space: O(total word chars + stream length)
    """
    
    def __init__(self, words: List[str]):
        """
        Initialize with reverse trie.
        
        Time: O(sum of word lengths)
        Space: O(sum of word lengths)
        """
        self.root = TrieNode()
        self.stream = deque()
        self.max_len = 0
        
        # Build reverse trie
        for word in words:
            self.max_len = max(self.max_len, len(word))
            node = self.root
            
            # Insert word in reverse order
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
    
    def query(self, letter: str) -> bool:
        """
        Process new character and check for word matches.
        
        Time: O(max_word_length)
        Space: O(1) additional
        """
        self.stream.appendleft(letter)
        
        # Keep stream length bounded
        if len(self.stream) > self.max_len:
            self.stream.pop()
        
        # Check if any suffix matches a word
        node = self.root
        for char in self.stream:
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                return True
        
        return False


class StreamChecker2:
    """
    Approach 2: Multiple Trie Traversals
    
    Maintain multiple active trie traversals for different starting positions.
    
    Time: O(number of active traversals) per query
    Space: O(stream length * alphabet size)
    """
    
    def __init__(self, words: List[str]):
        """Initialize with forward trie"""
        self.root = TrieNode()
        self.active_nodes = set()
        
        # Build trie
        for word in words:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
    
    def query(self, letter: str) -> bool:
        """Process character with multiple active traversals"""
        new_active = set()
        found_word = False
        
        # Always start a new traversal from root
        new_active.add(self.root)
        
        # Continue existing traversals
        for node in self.active_nodes:
            if letter in node.children:
                next_node = node.children[letter]
                new_active.add(next_node)
                if next_node.is_word:
                    found_word = True
        
        # Check if we can start from root
        if letter in self.root.children:
            next_node = self.root.children[letter]
            new_active.add(next_node)
            if next_node.is_word:
                found_word = True
        
        self.active_nodes = new_active
        return found_word


class StreamChecker3:
    """
    Approach 3: Rolling Hash with Set
    
    Use rolling hash to efficiently check all suffixes.
    
    Time: O(max_word_length) per query
    Space: O(number of words + stream length)
    """
    
    def __init__(self, words: List[str]):
        """Initialize with word hashes"""
        self.word_hashes = set()
        self.max_len = 0
        self.base = 31
        self.mod = 10**9 + 7
        
        # Precompute hashes of all words
        for word in words:
            self.max_len = max(self.max_len, len(word))
            word_hash = 0
            power = 1
            
            for char in reversed(word):
                word_hash = (word_hash + (ord(char) - ord('a') + 1) * power) % self.mod
                power = (power * self.base) % self.mod
            
            self.word_hashes.add(word_hash)
        
        self.stream = deque()
        self.powers = [1]  # Precompute powers of base
        for i in range(1, self.max_len):
            self.powers.append((self.powers[-1] * self.base) % self.mod)
    
    def query(self, letter: str) -> bool:
        """Check using rolling hash"""
        self.stream.appendleft(letter)
        
        if len(self.stream) > self.max_len:
            self.stream.pop()
        
        # Check all suffixes using rolling hash
        current_hash = 0
        
        for i, char in enumerate(self.stream):
            char_val = ord(char) - ord('a') + 1
            current_hash = (current_hash + char_val * self.powers[i]) % self.mod
            
            if current_hash in self.word_hashes:
                return True
        
        return False


class StreamChecker4:
    """
    Approach 4: Optimized Reverse Trie with Early Termination
    
    Enhanced reverse trie with optimizations.
    
    Time: O(average matching depth) per query
    Space: O(total word chars)
    """
    
    def __init__(self, words: List[str]):
        """Initialize optimized reverse trie"""
        self.root = TrieNode()
        self.stream = []
        self.max_len = 0
        
        # Build reverse trie with word count tracking
        for word in words:
            self.max_len = max(self.max_len, len(word))
            node = self.root
            
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
    
    def query(self, letter: str) -> bool:
        """Optimized query with early termination"""
        self.stream.append(letter)
        
        # Keep stream bounded
        if len(self.stream) > self.max_len:
            self.stream.pop(0)
        
        # Check suffixes from shortest to longest
        node = self.root
        
        for i in range(len(self.stream)):
            char = self.stream[-(i+1)]  # Go backwards through stream
            
            if char not in node.children:
                break
                
            node = node.children[char]
            
            if node.is_word:
                return True
        
        return False


class StreamChecker5:
    """
    Approach 5: Aho-Corasick Algorithm for Multiple Pattern Matching
    
    Use Aho-Corasick for efficient multiple pattern matching in stream.
    
    Time: O(1) amortized per character
    Space: O(total pattern chars + alphabet size * states)
    """
    
    def __init__(self, words: List[str]):
        """Build Aho-Corasick automaton"""
        self.root = TrieNode()
        self.failure = {}
        self.output = defaultdict(list)
        self.current_state = self.root
        
        # Build trie
        for word in words:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            self.output[id(node)].append(word)
        
        # Build failure function
        self._build_failure_function()
    
    def _build_failure_function(self):
        """Build failure function for Aho-Corasick"""
        queue = deque()
        
        # Initialize first level
        for child in self.root.children.values():
            self.failure[id(child)] = self.root
            queue.append(child)
        
        # Build failure links
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                failure_state = self.failure[id(current)]
                
                while (failure_state != self.root and 
                       char not in failure_state.children):
                    failure_state = self.failure[id(failure_state)]
                
                if char in failure_state.children and failure_state.children[char] != child:
                    self.failure[id(child)] = failure_state.children[char]
                else:
                    self.failure[id(child)] = self.root
                
                # Add output from failure state
                failure_node = self.failure[id(child)]
                self.output[id(child)].extend(self.output[id(failure_node)])
    
    def query(self, letter: str) -> bool:
        """Process character with Aho-Corasick"""
        # Transition based on current state and input character
        while (self.current_state != self.root and 
               letter not in self.current_state.children):
            self.current_state = self.failure[id(self.current_state)]
        
        if letter in self.current_state.children:
            self.current_state = self.current_state.children[letter]
        
        # Check if we found any patterns
        return len(self.output[id(self.current_state)]) > 0


class StreamChecker6:
    """
    Approach 6: Suffix Array with Binary Search
    
    Use suffix array for pattern matching (conceptual implementation).
    
    Time: O(log n + max_word_length) per query
    Space: O(total characters)
    """
    
    def __init__(self, words: List[str]):
        """Initialize with suffix array concept"""
        self.words = set(words)
        self.max_len = max(len(word) for word in words) if words else 0
        self.stream = deque()
    
    def query(self, letter: str) -> bool:
        """Check using suffix matching"""
        self.stream.append(letter)
        
        if len(self.stream) > self.max_len:
            self.stream.popleft()
        
        # Check all suffixes
        stream_str = ''.join(self.stream)
        
        for i in range(len(stream_str)):
            suffix = stream_str[i:]
            if suffix in self.words:
                return True
        
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    test_cases = [
        # LeetCode example
        {
            "words": ["cd", "f", "kl"],
            "queries": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
            "expected": [False, False, False, True, False, True, False, False, False, False, False, True]
        },
        
        # Simple cases
        {
            "words": ["a", "ab"],
            "queries": ["a", "b"],
            "expected": [True, True]
        },
        
        # Overlapping patterns
        {
            "words": ["abc", "bc", "c"],
            "queries": ["a", "b", "c"],
            "expected": [False, False, True]
        },
        
        # Single character
        {
            "words": ["x"],
            "queries": ["y", "x", "z"],
            "expected": [False, True, False]
        }
    ]
    
    implementations = [
        ("Reverse Trie", StreamChecker1),
        ("Multiple Traversals", StreamChecker2),
        ("Rolling Hash", StreamChecker3),
        ("Optimized Trie", StreamChecker4),
        ("Aho-Corasick", StreamChecker5),
        ("Suffix Array", StreamChecker6),
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Words: {test_case['words']}")
        print(f"Queries: {test_case['queries']}")
        print(f"Expected: {test_case['expected']}")
        
        for name, StreamCheckerClass in implementations:
            try:
                checker = StreamCheckerClass(test_case['words'])
                results = []
                
                for char in test_case['queries']:
                    result = checker.query(char)
                    results.append(result)
                
                status = "✓" if results == test_case['expected'] else "✗"
                print(f"  {name:18}: {results} {status}")
                
            except Exception as e:
                print(f"  {name:18}: Error - {e}")


def demonstrate_reverse_trie_approach():
    """Demonstrate reverse trie approach step by step"""
    print("\n=== Reverse Trie Approach Demo ===")
    
    words = ["cd", "f", "kl"]
    print(f"Words: {words}")
    
    # Build reverse trie
    root = TrieNode()
    
    print(f"\nBuilding reverse trie:")
    for word in words:
        print(f"Inserting '{word}' in reverse:")
        node = root
        
        for char in reversed(word):
            print(f"  Character: '{char}'")
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"    Created new node")
            node = node.children[char]
            print(f"    Moved to node for '{char}'")
        
        node.is_word = True
        print(f"  Marked as word end")
    
    # Simulate query process
    print(f"\nQuery simulation:")
    
    stream = deque()
    queries = ["a", "b", "c", "d", "e", "f"]
    
    for char in queries:
        stream.appendleft(char)
        print(f"\nQuery '{char}', stream: {list(stream)}")
        
        # Check suffixes
        node = root
        found = False
        
        for i, stream_char in enumerate(stream):
            print(f"  Checking '{stream_char}' in trie")
            
            if stream_char not in node.children:
                print(f"    No path for '{stream_char}', stopping")
                break
            
            node = node.children[stream_char]
            
            if node.is_word:
                suffix = ''.join(list(stream)[:i+1])
                print(f"    ✓ Found word: '{suffix}'")
                found = True
                break
            else:
                print(f"    Continue in trie...")
        
        print(f"  Result: {found}")


def demonstrate_multiple_traversals():
    """Demonstrate multiple traversals approach"""
    print("\n=== Multiple Traversals Demo ===")
    
    words = ["abc", "bc", "c"]
    print(f"Words: {words}")
    
    # Build forward trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    print(f"\nQuery simulation with multiple active traversals:")
    
    active_nodes = set()
    queries = ["a", "b", "c"]
    
    for char in queries:
        new_active = set()
        found_word = False
        
        print(f"\nQuery '{char}':")
        print(f"  Active nodes before: {len(active_nodes)}")
        
        # Always start new traversal from root
        new_active.add(root)
        print(f"  Added root to new active set")
        
        # Continue existing traversals
        for node in active_nodes:
            if char in node.children:
                next_node = node.children[char]
                new_active.add(next_node)
                print(f"  Continued traversal, added node")
                
                if next_node.is_word:
                    found_word = True
                    print(f"    ✓ Found word ending at this node")
        
        # Check new traversal from root
        if char in root.children:
            next_node = root.children[char]
            new_active.add(next_node)
            print(f"  Started new traversal from root")
            
            if next_node.is_word:
                found_word = True
                print(f"    ✓ Found word ending at this node")
        
        active_nodes = new_active
        print(f"  Active nodes after: {len(active_nodes)}")
        print(f"  Result: {found_word}")


def demonstrate_rolling_hash():
    """Demonstrate rolling hash approach"""
    print("\n=== Rolling Hash Demo ===")
    
    words = ["abc", "bc", "c"]
    print(f"Words: {words}")
    
    # Compute word hashes
    base = 31
    mod = 10**9 + 7
    word_hashes = set()
    
    print(f"\nPrecomputing word hashes:")
    
    for word in words:
        word_hash = 0
        power = 1
        
        print(f"Word '{word}':")
        for char in reversed(word):
            char_val = ord(char) - ord('a') + 1
            word_hash = (word_hash + char_val * power) % mod
            power = (power * base) % mod
            print(f"  Char '{char}' (val={char_val}), hash becomes {word_hash}")
        
        word_hashes.add(word_hash)
        print(f"  Final hash: {word_hash}")
    
    print(f"\nWord hashes: {word_hashes}")
    
    # Simulate queries
    print(f"\nQuery simulation:")
    
    stream = deque()
    queries = ["a", "b", "c"]
    powers = [1, base % mod, (base * base) % mod]
    
    for char in queries:
        stream.appendleft(char)
        print(f"\nQuery '{char}', stream: {list(stream)}")
        
        # Check all suffixes
        current_hash = 0
        found = False
        
        for i, stream_char in enumerate(stream):
            char_val = ord(stream_char) - ord('a') + 1
            current_hash = (current_hash + char_val * powers[i]) % mod
            
            suffix = ''.join(list(stream)[:i+1])
            print(f"  Suffix '{suffix}': hash = {current_hash}")
            
            if current_hash in word_hashes:
                print(f"    ✓ Hash matches a word!")
                found = True
        
        print(f"  Result: {found}")


def benchmark_implementations():
    """Benchmark different implementations"""
    print("\n=== Benchmarking Implementations ===")
    
    import random
    import string
    
    # Generate test data
    def generate_test_case(num_words: int, avg_word_length: int, stream_length: int):
        # Generate words
        words = set()
        while len(words) < num_words:
            length = max(1, avg_word_length + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase[:6], k=length))
            words.add(word)
        
        # Generate query stream
        queries = [random.choice(string.ascii_lowercase[:6]) for _ in range(stream_length)]
        
        return list(words), queries
    
    test_scenarios = [
        ("Small", 10, 4, 50),
        ("Medium", 50, 6, 200),
        ("Large", 100, 8, 500),
    ]
    
    implementations = [
        ("Reverse Trie", StreamChecker1),
        ("Multiple Traversals", StreamChecker2),
        ("Rolling Hash", StreamChecker3),
        ("Optimized Trie", StreamChecker4),
    ]
    
    for scenario_name, num_words, avg_len, stream_len in test_scenarios:
        words, queries = generate_test_case(num_words, avg_len, stream_len)
        
        print(f"\n--- {scenario_name} Test ---")
        print(f"Words: {len(words)}, Avg length: {avg_len}, Stream: {stream_len}")
        
        for impl_name, StreamCheckerClass in implementations:
            start_time = time.time()
            
            try:
                # Measure initialization time
                init_start = time.time()
                checker = StreamCheckerClass(words)
                init_time = time.time() - init_start
                
                # Measure query time
                query_start = time.time()
                match_count = 0
                
                for char in queries:
                    if checker.query(char):
                        match_count += 1
                
                query_time = time.time() - query_start
                total_time = time.time() - start_time
                
                print(f"  {impl_name:18}: {match_count:3} matches, "
                      f"init: {init_time*1000:5.1f}ms, "
                      f"query: {query_time*1000:5.1f}ms, "
                      f"total: {total_time*1000:5.1f}ms")
            
            except Exception as e:
                print(f"  {impl_name:18}: Error - {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Real-time content filtering
    print("1. Content Filtering (Profanity Detection):")
    
    banned_words = ["bad", "evil", "spam"]
    checker = StreamChecker1(banned_words)
    
    text_stream = "This is a good message but spam is bad"
    
    print(f"   Banned words: {banned_words}")
    print(f"   Text stream: '{text_stream}'")
    print(f"   Character-by-character analysis:")
    
    violations = []
    for i, char in enumerate(text_stream):
        if checker.query(char):
            violations.append(i)
            print(f"     Position {i}: '{char}' -> VIOLATION DETECTED!")
        else:
            print(f"     Position {i}: '{char}' -> OK")
    
    print(f"   Violations at positions: {violations}")
    
    # Application 2: DNS domain filtering
    print(f"\n2. DNS Domain Filtering:")
    
    blocked_domains = [".ads", ".spam", ".malware"]
    domain_checker = StreamChecker1(blocked_domains)
    
    domain = "example.ads.com"
    
    print(f"   Blocked patterns: {blocked_domains}")
    print(f"   Checking domain: '{domain}'")
    
    blocked = False
    for char in domain:
        if domain_checker.query(char):
            blocked = True
            print(f"   ✗ Domain blocked (contains blocked pattern)")
            break
    
    if not blocked:
        print(f"   ✓ Domain allowed")
    
    # Application 3: Log monitoring
    print(f"\n3. Real-time Log Monitoring:")
    
    error_patterns = ["ERROR", "FATAL", "WARN"]
    log_checker = StreamChecker1(error_patterns)
    
    log_entry = "INFO: Process started WARN: Low memory ERROR: Connection failed"
    
    print(f"   Error patterns: {error_patterns}")
    print(f"   Log entry: '{log_entry}'")
    
    alerts = []
    current_pos = 0
    
    for char in log_entry:
        if log_checker.query(char):
            alerts.append(current_pos)
        current_pos += 1
    
    print(f"   Alert positions: {alerts}")
    print(f"   Number of alerts: {len(alerts)}")
    
    # Application 4: Chat message moderation
    print(f"\n4. Real-time Chat Moderation:")
    
    inappropriate_words = ["hate", "toxic", "abuse"]
    chat_checker = StreamChecker1(inappropriate_words)
    
    messages = [
        "Hello everyone!",
        "I hate this game",
        "You are toxic",
        "Great job team!"
    ]
    
    print(f"   Filtered words: {inappropriate_words}")
    print(f"   Chat messages:")
    
    for i, message in enumerate(messages):
        # Reset checker for each message
        chat_checker = StreamChecker1(inappropriate_words)
        
        flagged = False
        for char in message:
            if chat_checker.query(char):
                flagged = True
                break
        
        status = "🚫 BLOCKED" if flagged else "✅ ALLOWED"
        print(f"     Message {i+1}: '{message}' -> {status}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        # Empty inputs
        {
            "words": [],
            "queries": ["a", "b"],
            "description": "Empty word list"
        },
        
        # Single character words
        {
            "words": ["a", "b", "c"],
            "queries": ["a", "b", "c", "d"],
            "description": "Single character words"
        },
        
        # Overlapping words
        {
            "words": ["a", "aa", "aaa"],
            "queries": ["a", "a", "a"],
            "description": "Overlapping words"
        },
        
        # Long words
        {
            "words": ["abcdefghijklmnop"],
            "queries": list("abcdefghijklmnopqr"),
            "description": "Very long word"
        },
        
        # All same character
        {
            "words": ["aa", "aaa"],
            "queries": ["a", "a", "a", "a"],
            "description": "Repeated characters"
        },
        
        # No matches
        {
            "words": ["xyz"],
            "queries": ["a", "b", "c"],
            "description": "No possible matches"
        },
    ]
    
    for case in edge_cases:
        print(f"\n{case['description']}:")
        print(f"  Words: {case['words']}")
        print(f"  Queries: {case['queries']}")
        
        try:
            checker = StreamChecker1(case['words'])
            results = []
            
            for char in case['queries']:
                result = checker.query(char)
                results.append(result)
            
            print(f"  Results: {results}")
            
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Reverse Trie",
         "Init: O(∑word_lengths), Query: O(max_word_length)",
         "Space: O(∑word_lengths + max_word_length)"),
        
        ("Multiple Traversals",
         "Init: O(∑word_lengths), Query: O(active_nodes)",
         "Space: O(∑word_lengths + active_nodes)"),
        
        ("Rolling Hash",
         "Init: O(∑word_lengths), Query: O(max_word_length)",
         "Space: O(number_of_words + max_word_length)"),
        
        ("Optimized Trie",
         "Init: O(∑word_lengths), Query: O(average_match_depth)",
         "Space: O(∑word_lengths + max_word_length)"),
        
        ("Aho-Corasick",
         "Init: O(∑word_lengths + alphabet²), Query: O(1) amortized",
         "Space: O(∑word_lengths + alphabet × states)"),
        
        ("Suffix Array",
         "Init: O(∑word_lengths × log(words)), Query: O(log(words) + max_word_length)",
         "Space: O(∑word_lengths + max_word_length)"),
    ]
    
    print("Implementation Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Insights:")
    print(f"  • Reverse trie is most intuitive and efficient for this problem")
    print(f"  • Multiple traversals can have exponential active nodes worst case")
    print(f"  • Rolling hash provides good practical performance")
    print(f"  • Aho-Corasick gives optimal asymptotic performance")
    
    print(f"\nPractical Considerations:")
    print(f"  • Stream length affects memory usage for buffering")
    print(f"  • Word length distribution impacts performance")
    print(f"  • Alphabet size affects trie memory usage")
    print(f"  • Pattern overlap affects active traversal count")
    
    print(f"\nRecommendations:")
    print(f"  • Use Reverse Trie for most practical applications")
    print(f"  • Use Aho-Corasick for high-throughput systems")
    print(f"  • Use Rolling Hash for memory-constrained environments")
    print(f"  • Consider Multiple Traversals for sparse pattern sets")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_reverse_trie_approach()
    demonstrate_multiple_traversals()
    demonstrate_rolling_hash()
    benchmark_implementations()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
1032. Stream of Characters demonstrates comprehensive stream processing approaches:

1. Reverse Trie with Stream Buffer - Build reverse trie and check suffixes efficiently
2. Multiple Trie Traversals - Maintain active traversals for different starting positions
3. Rolling Hash with Set - Use polynomial rolling hash for efficient suffix checking
4. Optimized Reverse Trie - Enhanced reverse trie with early termination optimizations
5. Aho-Corasick Algorithm - Multiple pattern matching with optimal asymptotic performance
6. Suffix Array with Binary Search - Conceptual suffix-based approach for pattern matching

Key concepts:
- Stream processing with bounded memory
- Reverse trie construction for suffix matching
- Multiple active state maintenance
- Rolling hash for efficient string comparison
- Aho-Corasick automaton for multiple pattern matching

Real-world applications:
- Real-time content filtering and profanity detection
- DNS domain filtering and security
- Log monitoring and alerting systems
- Chat message moderation and safety

Each approach offers different trade-offs between preprocessing time,
query performance, and memory usage for streaming pattern detection.
"""
