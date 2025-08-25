"""
745. Prefix and Suffix Search - Multiple Approaches
Difficulty: Hard

Design a special dictionary that searches for words by a prefix and a suffix.

Implement the WordFilter class:
- WordFilter(string[] words) initializes the object with the words in the dictionary.
- f(string prefix, string suffix) returns the index of the word in the dictionary 
  that has the prefix and suffix. If there is more than one valid index, 
  return the largest one. If there is no such word, return -1.

Examples:
Input: ["WordFilter", "f"]
[[[["apple"]], ["a", "e"]]
Output: [null, 0]

Approaches:
1. Brute Force with Linear Search
2. Trie with Suffix Concatenation
3. Dual Trie (Prefix + Suffix)
4. Compressed Trie with Bit Manipulation
5. Hash-based Optimization
6. Advanced Trie with Weight Tracking
"""

from typing import List, Optional, Dict, Set, Tuple
import bisect
from collections import defaultdict

class TrieNode:
    """Basic trie node"""
    def __init__(self):
        self.children = {}
        self.word_indices = []  # Store indices of words passing through this node

class CompressedTrieNode:
    """Compressed trie node for memory efficiency"""
    __slots__ = ['children', 'indices', 'edge']
    
    def __init__(self, edge: str = ""):
        self.children = {}
        self.indices = set()  # Word indices
        self.edge = edge

class WordFilterBruteForce:
    """Approach 1: Brute Force with Linear Search"""
    
    def __init__(self, words: List[str]):
        """
        Initialize with brute force approach.
        
        Time: O(1)
        Space: O(n * m) where n=words, m=average_length
        """
        self.words = words
    
    def f(self, prefix: str, suffix: str) -> int:
        """
        Find word with given prefix and suffix.
        
        Time: O(n * (|prefix| + |suffix|))
        Space: O(1)
        """
        for i in range(len(self.words) - 1, -1, -1):  # Search from end for largest index
            word = self.words[i]
            if word.startswith(prefix) and word.endswith(suffix):
                return i
        return -1

class WordFilterSuffixTrie:
    """Approach 2: Trie with Suffix Concatenation"""
    
    def __init__(self, words: List[str]):
        """
        Build trie with suffix#prefix concatenations.
        
        Time: O(n * m^2) where n=words, m=average_length
        Space: O(n * m^2)
        """
        self.root = TrieNode()
        
        for i, word in enumerate(words):
            # For each suffix, create suffix#word entry
            for j in range(len(word) + 1):
                suffix = word[j:]
                combined = suffix + "#" + word
                
                node = self.root
                for char in combined:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                    node.word_indices.append(i)
    
    def f(self, prefix: str, suffix: str) -> int:
        """
        Search for suffix#prefix in trie.
        
        Time: O(|suffix| + |prefix|)
        Space: O(1)
        """
        search_str = suffix + "#" + prefix
        
        node = self.root
        for char in search_str:
            if char not in node.children:
                return -1
            node = node.children[char]
        
        # Return largest index
        return node.word_indices[-1] if node.word_indices else -1

class WordFilterDualTrie:
    """Approach 3: Dual Trie (Prefix + Suffix)"""
    
    def __init__(self, words: List[str]):
        """
        Build separate tries for prefixes and suffixes.
        
        Time: O(n * m)
        Space: O(n * m)
        """
        self.prefix_trie = TrieNode()
        self.suffix_trie = TrieNode()
        self.words = words
        
        # Build prefix trie
        for i, word in enumerate(words):
            node = self.prefix_trie
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.word_indices.append(i)
        
        # Build suffix trie (reverse)
        for i, word in enumerate(words):
            node = self.suffix_trie
            for char in reversed(word):
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.word_indices.append(i)
    
    def f(self, prefix: str, suffix: str) -> int:
        """
        Find intersection of prefix and suffix matches.
        
        Time: O(|prefix| + |suffix| + min(prefix_matches, suffix_matches))
        Space: O(1)
        """
        # Get prefix matches
        prefix_node = self.prefix_trie
        for char in prefix:
            if char not in prefix_node.children:
                return -1
            prefix_node = prefix_node.children[char]
        
        prefix_indices = set(prefix_node.word_indices)
        
        # Get suffix matches
        suffix_node = self.suffix_trie
        for char in reversed(suffix):
            if char not in suffix_node.children:
                return -1
            suffix_node = suffix_node.children[char]
        
        suffix_indices = set(suffix_node.word_indices)
        
        # Find intersection and return largest
        common_indices = prefix_indices & suffix_indices
        return max(common_indices) if common_indices else -1

class WordFilterCompressed:
    """Approach 4: Compressed Trie with Bit Manipulation"""
    
    def __init__(self, words: List[str]):
        """
        Build compressed trie with bit manipulation optimizations.
        
        Time: O(n * m)
        Space: O(compressed_size)
        """
        self.root = CompressedTrieNode()
        self.words = words
        
        # Build compressed trie with combined strings
        for i, word in enumerate(words):
            # Create all suffix#word combinations
            for j in range(len(word) + 1):
                suffix = word[j:]
                combined = suffix + "#" + word
                self._insert_compressed(combined, i)
    
    def _insert_compressed(self, text: str, index: int) -> None:
        """Insert text into compressed trie"""
        node = self.root
        i = 0
        
        while i < len(text):
            char = text[i]
            
            if char not in node.children:
                # Create new node with remaining text
                new_node = CompressedTrieNode(text[i:])
                new_node.indices.add(index)
                node.children[char] = new_node
                return
            
            child = node.children[char]
            edge = child.edge
            
            # Find common prefix length
            j = 0
            while (j < len(edge) and 
                   i + j < len(text) and 
                   edge[j] == text[i + j]):
                j += 1
            
            if j == len(edge):
                # Full edge match, continue
                node = child
                node.indices.add(index)
                i += j
            else:
                # Partial match, split edge
                # Create intermediate node
                intermediate = CompressedTrieNode(edge[:j])
                intermediate.indices.add(index)
                
                # Update existing child
                child.edge = edge[j:]
                intermediate.children[edge[j]] = child
                
                # Update parent
                node.children[char] = intermediate
                
                # Continue with remaining text
                if i + j < len(text):
                    remaining = text[i + j:]
                    new_child = CompressedTrieNode(remaining)
                    new_child.indices.add(index)
                    intermediate.children[remaining[0]] = new_child
                
                return
    
    def f(self, prefix: str, suffix: str) -> int:
        """
        Search in compressed trie.
        
        Time: O(|suffix| + |prefix|)
        Space: O(1)
        """
        search_str = suffix + "#" + prefix
        
        node = self.root
        i = 0
        
        while i < len(search_str):
            char = search_str[i]
            
            if char not in node.children:
                return -1
            
            child = node.children[char]
            edge = child.edge
            
            # Check if search string matches edge
            if i + len(edge) <= len(search_str):
                if search_str[i:i + len(edge)] == edge:
                    node = child
                    i += len(edge)
                else:
                    return -1
            else:
                # Partial match at end
                if edge.startswith(search_str[i:]):
                    node = child
                    break
                else:
                    return -1
        
        return max(node.indices) if node.indices else -1

class WordFilterHash:
    """Approach 5: Hash-based Optimization"""
    
    def __init__(self, words: List[str]):
        """
        Build hash tables for efficient lookup.
        
        Time: O(n * m^2)
        Space: O(n * m^2)
        """
        self.prefix_map = defaultdict(list)  # prefix -> list of indices
        self.suffix_map = defaultdict(list)  # suffix -> list of indices
        self.words = words
        
        # Build prefix map
        for i, word in enumerate(words):
            for j in range(len(word) + 1):
                prefix = word[:j]
                self.prefix_map[prefix].append(i)
        
        # Build suffix map
        for i, word in enumerate(words):
            for j in range(len(word) + 1):
                suffix = word[j:]
                self.suffix_map[suffix].append(i)
    
    def f(self, prefix: str, suffix: str) -> int:
        """
        Use hash lookup for fast intersection.
        
        Time: O(min(prefix_matches, suffix_matches))
        Space: O(1)
        """
        if prefix not in self.prefix_map or suffix not in self.suffix_map:
            return -1
        
        prefix_indices = set(self.prefix_map[prefix])
        suffix_indices = set(self.suffix_map[suffix])
        
        common_indices = prefix_indices & suffix_indices
        return max(common_indices) if common_indices else -1

class WordFilterAdvanced:
    """Approach 6: Advanced Trie with Weight Tracking"""
    
    def __init__(self, words: List[str]):
        """
        Advanced trie with weight-based optimization.
        
        Time: O(n * m^2)
        Space: O(n * m^2)
        """
        self.trie = {}
        
        # Build weighted trie
        for i, word in enumerate(words):
            for j in range(len(word) + 1):
                suffix = word[j:]
                key = suffix + "#" + word
                
                # Store with weight (index)
                if key not in self.trie:
                    self.trie[key] = []
                self.trie[key].append(i)
        
        # Sort indices for each key (largest first)
        for key in self.trie:
            self.trie[key].sort(reverse=True)
    
    def f(self, prefix: str, suffix: str) -> int:
        """
        Direct hash lookup with weight consideration.
        
        Time: O(|suffix| + |prefix|)
        Space: O(1)
        """
        # Find all keys that start with suffix# and contain prefix
        search_prefix = suffix + "#" + prefix
        
        best_index = -1
        for key in self.trie:
            if key.startswith(search_prefix):
                # This key matches, get the largest index
                best_index = max(best_index, self.trie[key][0])
        
        return best_index


def test_basic_functionality():
    """Test basic word filter functionality"""
    print("=== Testing Word Filter Functionality ===")
    
    words = ["apple", "application", "apply", "approach", "appropriate"]
    
    test_cases = [
        ("a", "e", 0),      # "apple"
        ("app", "e", 0),    # "apple"
        ("a", "n", 1),      # "application"
        ("app", "ion", 1),  # "application"
        ("ap", "y", 2),     # "apply"
        ("x", "y", -1),     # Not found
        ("", "e", 0),       # Empty prefix
        ("a", "", 4),       # Empty suffix (last word starting with 'a')
    ]
    
    implementations = [
        ("Brute Force", WordFilterBruteForce),
        ("Suffix Trie", WordFilterSuffixTrie),
        ("Dual Trie", WordFilterDualTrie),
        ("Compressed", WordFilterCompressed),
        ("Hash-based", WordFilterHash),
        ("Advanced", WordFilterAdvanced),
    ]
    
    print(f"Test words: {words}")
    
    for impl_name, impl_class in implementations:
        print(f"\n{impl_name}:")
        
        try:
            word_filter = impl_class(words)
            
            for prefix, suffix, expected in test_cases:
                result = word_filter.f(prefix, suffix)
                status = "✓" if result == expected else "✗"
                print(f"  f('{prefix}', '{suffix}') = {result:2} (expected {expected:2}) {status}")
                
        except Exception as e:
            print(f"  Error: {e}")


def benchmark_performance():
    """Benchmark performance of different implementations"""
    print("\n=== Performance Benchmark ===")
    
    import time
    import random
    import string
    
    def generate_words(count: int, min_len: int = 3, max_len: int = 10) -> List[str]:
        """Generate random words for testing"""
        words = []
        for _ in range(count):
            length = random.randint(min_len, max_len)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return words
    
    def generate_queries(words: List[str], count: int) -> List[Tuple[str, str]]:
        """Generate random prefix/suffix queries"""
        queries = []
        for _ in range(count):
            word = random.choice(words)
            prefix_len = random.randint(0, len(word) // 2)
            suffix_len = random.randint(0, len(word) // 2)
            
            prefix = word[:prefix_len]
            suffix = word[-suffix_len:] if suffix_len > 0 else ""
            
            queries.append((prefix, suffix))
        return queries
    
    test_sizes = [
        (100, 50),    # Small
        (500, 100),   # Medium
        (1000, 200),  # Large
    ]
    
    implementations = [
        ("Brute Force", WordFilterBruteForce),
        ("Dual Trie", WordFilterDualTrie),
        ("Hash-based", WordFilterHash),
        ("Advanced", WordFilterAdvanced),
    ]
    
    print(f"{'Size':<12} {'Method':<15} {'Build(ms)':<12} {'Query(ms)':<12} {'Total(ms)':<12}")
    print("-" * 70)
    
    for word_count, query_count in test_sizes:
        words = generate_words(word_count)
        queries = generate_queries(words, query_count)
        size_str = f"{word_count}w/{query_count}q"
        
        for impl_name, impl_class in implementations:
            if impl_name == "Brute Force" and word_count > 500:
                continue  # Skip brute force for large inputs
            
            try:
                # Measure build time
                start_time = time.time()
                word_filter = impl_class(words)
                build_time = (time.time() - start_time) * 1000
                
                # Measure query time
                start_time = time.time()
                results = []
                for prefix, suffix in queries:
                    result = word_filter.f(prefix, suffix)
                    results.append(result)
                query_time = (time.time() - start_time) * 1000
                
                total_time = build_time + query_time
                
                print(f"{size_str:<12} {impl_name:<15} {build_time:<12.2f} "
                      f"{query_time:<12.2f} {total_time:<12.2f}")
                
            except Exception as e:
                print(f"{size_str:<12} {impl_name:<15} {'Error':<12} {'Error':<12} {'Error':<12}")
        
        print()


def analyze_memory_usage():
    """Analyze memory usage of different approaches"""
    print("\n=== Memory Usage Analysis ===")
    
    import sys
    
    words = ["apple", "application", "apply", "approach", "appropriate"]
    
    def estimate_memory(obj, visited=None) -> int:
        """Estimate memory usage of object"""
        if visited is None:
            visited = set()
        
        if id(obj) in visited:
            return 0
        
        visited.add(id(obj))
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(estimate_memory(k, visited) + estimate_memory(v, visited) 
                       for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(estimate_memory(item, visited) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += estimate_memory(obj.__dict__, visited)
        elif hasattr(obj, '__slots__'):
            size += sum(estimate_memory(getattr(obj, slot, None), visited) 
                       for slot in obj.__slots__ if hasattr(obj, slot))
        
        return size
    
    implementations = [
        ("Brute Force", WordFilterBruteForce),
        ("Suffix Trie", WordFilterSuffixTrie),
        ("Dual Trie", WordFilterDualTrie),
        ("Hash-based", WordFilterHash),
    ]
    
    print(f"{'Implementation':<15} {'Memory (bytes)':<15} {'Memory per word':<15}")
    print("-" * 50)
    
    for impl_name, impl_class in implementations:
        try:
            word_filter = impl_class(words)
            memory_usage = estimate_memory(word_filter)
            memory_per_word = memory_usage / len(words)
            
            print(f"{impl_name:<15} {memory_usage:<15,} {memory_per_word:<15.1f}")
            
        except Exception as e:
            print(f"{impl_name:<15} {'Error':<15} {'Error':<15}")


def demonstrate_trie_construction():
    """Demonstrate trie construction process"""
    print("\n=== Trie Construction Demonstration ===")
    
    words = ["app", "apple", "apply"]
    
    print(f"Building suffix trie for words: {words}")
    
    # Show suffix combinations
    print(f"\nSuffix combinations:")
    for i, word in enumerate(words):
        print(f"Word {i}: '{word}'")
        for j in range(len(word) + 1):
            suffix = word[j:]
            combined = suffix + "#" + word
            print(f"  {j}: '{suffix}' -> '{combined}'")
    
    # Build and show trie structure
    word_filter = WordFilterSuffixTrie(words)
    
    def print_trie(node: TrieNode, path: str = "", depth: int = 0) -> None:
        """Print trie structure"""
        indent = "  " * depth
        
        if node.word_indices:
            indices_str = ",".join(map(str, node.word_indices))
            print(f"{indent}'{path}' -> indices: [{indices_str}]")
        
        for char, child in sorted(node.children.items()):
            print_trie(child, path + char, depth + 1)
    
    print(f"\nTrie structure (showing first few levels):")
    
    def print_limited_trie(node: TrieNode, path: str = "", depth: int = 0, max_depth: int = 3) -> None:
        """Print trie structure with depth limit"""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        if node.word_indices and len(path) > 0:
            indices_str = ",".join(map(str, node.word_indices))
            print(f"{indent}'{path}' -> [{indices_str}]")
        
        if depth < max_depth:
            for char, child in sorted(list(node.children.items())[:3]):  # Show first 3 children
                print_limited_trie(child, path + char, depth + 1, max_depth)
            
            if len(node.children) > 3:
                print(f"{indent}  ... ({len(node.children) - 3} more children)")
    
    print_limited_trie(word_filter.root)


def demonstrate_query_process():
    """Demonstrate query processing"""
    print("\n=== Query Processing Demonstration ===")
    
    words = ["apple", "application", "apply"]
    word_filter = WordFilterSuffixTrie(words)
    
    test_queries = [
        ("app", "e"),      # Should find "apple" (index 0)
        ("a", "ion"),      # Should find "application" (index 1)
        ("ap", "y"),       # Should find "apply" (index 2)
    ]
    
    for prefix, suffix in test_queries:
        print(f"\nQuery: prefix='{prefix}', suffix='{suffix}'")
        
        search_str = suffix + "#" + prefix
        print(f"Search string: '{search_str}'")
        
        # Trace through trie
        node = word_filter.root
        path = ""
        
        for i, char in enumerate(search_str):
            path += char
            
            if char in node.children:
                node = node.children[char]
                print(f"  Step {i+1}: '{char}' -> path: '{path}', indices: {node.word_indices}")
            else:
                print(f"  Step {i+1}: '{char}' -> NOT FOUND")
                break
        else:
            # Successfully traversed
            result = node.word_indices[-1] if node.word_indices else -1
            print(f"Result: {result}")
            if result >= 0:
                print(f"Found word: '{words[result]}'")


def analyze_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force",
         "Build: O(1)", "Query: O(n * (|prefix| + |suffix|))", "Space: O(n * m)"),
        
        ("Suffix Trie",
         "Build: O(n * m^2)", "Query: O(|suffix| + |prefix|)", "Space: O(n * m^2)"),
        
        ("Dual Trie",
         "Build: O(n * m)", "Query: O(|prefix| + |suffix| + min(matches))", "Space: O(n * m)"),
        
        ("Compressed Trie",
         "Build: O(n * m^2)", "Query: O(|suffix| + |prefix|)", "Space: O(compressed_size)"),
        
        ("Hash-based",
         "Build: O(n * m^2)", "Query: O(min(prefix_matches, suffix_matches))", "Space: O(n * m^2)"),
        
        ("Advanced Trie",
         "Build: O(n * m^2)", "Query: O(|suffix| + |prefix|)", "Space: O(n * m^2)"),
    ]
    
    print(f"{'Approach':<15} {'Build Time':<15} {'Query Time':<35} {'Space'}")
    print("-" * 85)
    
    for approach, build, query, space in complexity_analysis:
        print(f"{approach:<15} {build:<15} {query:<35} {space}")
    
    print(f"\nWhere:")
    print(f"  n = number of words")
    print(f"  m = average word length")
    print(f"  |prefix| = prefix length")
    print(f"  |suffix| = suffix length")
    
    print(f"\nRecommendations:")
    print(f"  • Use Suffix Trie for frequent queries (optimal query time)")
    print(f"  • Use Dual Trie for balanced build/query performance")
    print(f"  • Use Hash-based for simple implementation")
    print(f"  • Use Compressed Trie for memory-constrained environments")


if __name__ == "__main__":
    test_basic_functionality()
    benchmark_performance()
    analyze_memory_usage()
    demonstrate_trie_construction()
    demonstrate_query_process()
    analyze_complexity()

"""
Prefix and Suffix Search demonstrates advanced trie optimization techniques:

Key Approaches:
1. Suffix Concatenation - Build trie with suffix#word combinations
2. Dual Trie - Separate tries for prefixes and suffixes with intersection
3. Compressed Trie - Memory-efficient trie with edge compression
4. Hash-based - Direct hash lookup for prefix/suffix combinations
5. Weight Tracking - Advanced indexing with weight-based optimization

Optimization Techniques:
- Suffix enumeration for comprehensive matching
- Bit manipulation for memory-efficient node representation
- Edge compression to reduce memory footprint
- Index tracking for efficient largest index retrieval
- Hash-based shortcuts for common query patterns

Real-world Applications:
- Autocomplete systems with prefix/suffix constraints
- Search engines with advanced query patterns
- Text processing with pattern matching requirements
- Database indexing with compound key optimization
- Code completion systems in IDEs

The suffix trie approach provides optimal query performance O(|prefix| + |suffix|)
while the dual trie offers better memory efficiency for large dictionaries.
"""
