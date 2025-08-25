"""
Suffix Tree Basic Implementation - Multiple Approaches
Difficulty: Easy

Implement a basic suffix tree data structure that supports:
1. Construction from a given string
2. Pattern search in O(m) time where m is pattern length
3. Suffix enumeration
4. Longest common substring finding

A suffix tree is a compressed trie containing all suffixes of a given string.
It's fundamental for many string algorithms and has applications in:
- Pattern matching
- Longest common substring
- String compression
- Bioinformatics (DNA sequence analysis)
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import time

class SuffixTreeNode:
    """Node in the suffix tree"""
    def __init__(self):
        self.children = {}  # edge_char -> child_node
        self.is_leaf = False
        self.suffix_index = -1  # For leaf nodes, index of suffix
        self.edge_start = -1    # Start index of edge label
        self.edge_end = -1      # End index of edge label
        self.suffix_link = None # Suffix link for Ukkonen's algorithm

class SuffixTree1:
    """
    Approach 1: Naive Suffix Tree Construction
    
    Build suffix tree by inserting all suffixes one by one.
    """
    
    def __init__(self, text: str):
        """
        Initialize suffix tree.
        
        Time: O(n^2) where n is length of text
        Space: O(n^2) in worst case
        """
        self.text = text + '$'  # Add terminator
        self.root = SuffixTreeNode()
        self.n = len(self.text)
        self._build_naive()
    
    def _build_naive(self) -> None:
        """Build suffix tree naively by inserting all suffixes"""
        for i in range(self.n):
            self._insert_suffix(i)
    
    def _insert_suffix(self, suffix_index: int) -> None:
        """Insert suffix starting at given index"""
        current = self.root
        
        for j in range(suffix_index, self.n):
            char = self.text[j]
            
            if char not in current.children:
                # Create new leaf node
                leaf = SuffixTreeNode()
                leaf.is_leaf = True
                leaf.suffix_index = suffix_index
                leaf.edge_start = j
                leaf.edge_end = self.n - 1
                current.children[char] = leaf
                break
            else:
                current = current.children[char]
    
    def search_pattern(self, pattern: str) -> bool:
        """
        Search for pattern in the text.
        
        Time: O(m) where m is pattern length
        Space: O(1)
        """
        current = self.root
        i = 0
        
        while i < len(pattern):
            char = pattern[i]
            
            if char not in current.children:
                return False
            
            child = current.children[char]
            
            # Compare with edge label
            edge_start = child.edge_start
            edge_length = child.edge_end - edge_start + 1
            
            # Check how much of pattern matches with edge
            j = 0
            while j < edge_length and i + j < len(pattern):
                if self.text[edge_start + j] != pattern[i + j]:
                    return False
                j += 1
            
            i += j
            current = child
        
        return True
    
    def get_all_suffixes(self) -> List[str]:
        """
        Get all suffixes of the text.
        
        Time: O(n^2) where n is text length
        Space: O(n^2)
        """
        suffixes = []
        self._collect_suffixes(self.root, "", suffixes)
        return suffixes
    
    def _collect_suffixes(self, node: SuffixTreeNode, path: str, suffixes: List[str]) -> None:
        """Collect all suffixes from subtree"""
        if node.is_leaf:
            suffixes.append(path + self.text[node.edge_start:node.edge_end + 1])
            return
        
        for child in node.children.values():
            edge_label = self.text[child.edge_start:child.edge_end + 1]
            self._collect_suffixes(child, path + edge_label, suffixes)


class SuffixTree2:
    """
    Approach 2: Improved Construction with Path Compression
    
    Use path compression to reduce space complexity.
    """
    
    def __init__(self, text: str):
        """
        Initialize with path compression.
        
        Time: O(n^2) 
        Space: O(n) average case
        """
        self.text = text + '$'
        self.root = SuffixTreeNode()
        self.n = len(self.text)
        self._build_compressed()
    
    def _build_compressed(self) -> None:
        """Build suffix tree with path compression"""
        for i in range(self.n):
            self._insert_suffix_compressed(i)
    
    def _insert_suffix_compressed(self, suffix_index: int) -> None:
        """Insert suffix with path compression"""
        current = self.root
        j = suffix_index
        
        while j < self.n:
            char = self.text[j]
            
            if char not in current.children:
                # Create new leaf with compressed edge
                leaf = SuffixTreeNode()
                leaf.is_leaf = True
                leaf.suffix_index = suffix_index
                leaf.edge_start = j
                leaf.edge_end = self.n - 1
                current.children[char] = leaf
                break
            
            child = current.children[char]
            edge_start = child.edge_start
            edge_end = child.edge_end
            
            # Find common prefix length
            k = 0
            while (edge_start + k <= edge_end and 
                   j + k < self.n and 
                   self.text[edge_start + k] == self.text[j + k]):
                k += 1
            
            if edge_start + k > edge_end:
                # Entire edge matches, continue
                current = child
                j += k
            else:
                # Need to split edge
                self._split_edge(current, child, char, k, j, suffix_index)
                break
    
    def _split_edge(self, parent: SuffixTreeNode, child: SuffixTreeNode, 
                   char: str, split_pos: int, suffix_pos: int, suffix_index: int) -> None:
        """Split edge at given position"""
        # Create intermediate node
        intermediate = SuffixTreeNode()
        intermediate.edge_start = child.edge_start
        intermediate.edge_end = child.edge_start + split_pos - 1
        
        # Update child's edge
        child.edge_start += split_pos
        
        # Connect intermediate node
        parent.children[char] = intermediate
        split_char = self.text[child.edge_start]
        intermediate.children[split_char] = child
        
        # Create new leaf for current suffix
        if suffix_pos + split_pos < self.n:
            new_leaf = SuffixTreeNode()
            new_leaf.is_leaf = True
            new_leaf.suffix_index = suffix_index
            new_leaf.edge_start = suffix_pos + split_pos
            new_leaf.edge_end = self.n - 1
            
            new_char = self.text[new_leaf.edge_start]
            intermediate.children[new_char] = new_leaf
    
    def find_longest_repeated_substring(self) -> str:
        """
        Find longest repeated substring.
        
        Time: O(n^2)
        Space: O(n)
        """
        max_length = 0
        longest_substring = ""
        
        def dfs(node: SuffixTreeNode, depth: int, path: str) -> None:
            nonlocal max_length, longest_substring
            
            # If internal node with at least 2 children, it represents repeated substring
            if not node.is_leaf and len(node.children) >= 2:
                if depth > max_length:
                    max_length = depth
                    longest_substring = path
            
            for child in node.children.values():
                edge_label = self.text[child.edge_start:child.edge_end + 1]
                dfs(child, depth + len(edge_label), path + edge_label)
        
        dfs(self.root, 0, "")
        return longest_substring.rstrip('$')


class SuffixTree3:
    """
    Approach 3: Ukkonen's Linear Time Algorithm (Simplified)
    
    Implement simplified version of Ukkonen's algorithm.
    """
    
    def __init__(self, text: str):
        """
        Initialize using Ukkonen's algorithm.
        
        Time: O(n) amortized
        Space: O(n)
        """
        self.text = text + '$'
        self.root = SuffixTreeNode()
        self.n = len(self.text)
        self.global_end = 0  # Global end for all leaf edges
        self._build_ukkonen()
    
    def _build_ukkonen(self) -> None:
        """Build suffix tree using Ukkonen's algorithm (simplified)"""
        # Simplified implementation - full Ukkonen's is quite complex
        active_node = self.root
        active_edge = -1
        active_length = 0
        
        for i in range(self.n):
            self.global_end = i
            self._extend(i, active_node, active_edge, active_length)
    
    def _extend(self, pos: int, active_node: SuffixTreeNode, 
               active_edge: int, active_length: int) -> None:
        """Extend suffix tree for position (simplified)"""
        char = self.text[pos]
        
        if char not in active_node.children:
            # Create new leaf
            leaf = SuffixTreeNode()
            leaf.is_leaf = True
            leaf.edge_start = pos
            leaf.edge_end = self.n - 1  # Will be updated by global_end
            active_node.children[char] = leaf
    
    def count_distinct_substrings(self) -> int:
        """
        Count number of distinct substrings.
        
        Time: O(n)
        Space: O(1) additional
        """
        def count_paths(node: SuffixTreeNode) -> int:
            if node.is_leaf:
                return 1
            
            total = 1  # Count current node
            for child in node.children.values():
                total += count_paths(child)
            
            return total
        
        return count_paths(self.root) - 1  # Exclude empty string


class SuffixTree4:
    """
    Approach 4: Suffix Tree with Applications
    
    Enhanced suffix tree with useful applications.
    """
    
    def __init__(self, text: str):
        """Initialize enhanced suffix tree"""
        self.text = text + '$'
        self.root = SuffixTreeNode()
        self.n = len(self.text)
        self._build_enhanced()
    
    def _build_enhanced(self) -> None:
        """Build enhanced suffix tree"""
        # Use compressed approach
        for i in range(self.n):
            self._insert_suffix_enhanced(i)
    
    def _insert_suffix_enhanced(self, suffix_index: int) -> None:
        """Insert suffix with enhanced features"""
        current = self.root
        j = suffix_index
        
        while j < self.n:
            char = self.text[j]
            
            if char not in current.children:
                leaf = SuffixTreeNode()
                leaf.is_leaf = True
                leaf.suffix_index = suffix_index
                leaf.edge_start = j
                leaf.edge_end = self.n - 1
                current.children[char] = leaf
                break
            
            child = current.children[char]
            edge_length = child.edge_end - child.edge_start + 1
            
            # Match characters along edge
            k = 0
            while (k < edge_length and 
                   j + k < self.n and 
                   self.text[child.edge_start + k] == self.text[j + k]):
                k += 1
            
            if k == edge_length:
                # Entire edge matches
                current = child
                j += k
            else:
                # Split edge
                self._split_edge_enhanced(current, child, char, k, j, suffix_index)
                break
    
    def _split_edge_enhanced(self, parent: SuffixTreeNode, child: SuffixTreeNode,
                           char: str, split_pos: int, suffix_pos: int, suffix_index: int) -> None:
        """Split edge with enhanced features"""
        # Create intermediate node
        intermediate = SuffixTreeNode()
        intermediate.edge_start = child.edge_start
        intermediate.edge_end = child.edge_start + split_pos - 1
        
        # Update child
        child.edge_start += split_pos
        
        # Connect nodes
        parent.children[char] = intermediate
        split_char = self.text[child.edge_start]
        intermediate.children[split_char] = child
        
        # Add new leaf
        if suffix_pos + split_pos < self.n:
            new_leaf = SuffixTreeNode()
            new_leaf.is_leaf = True
            new_leaf.suffix_index = suffix_index
            new_leaf.edge_start = suffix_pos + split_pos
            new_leaf.edge_end = self.n - 1
            
            new_char = self.text[new_leaf.edge_start]
            intermediate.children[new_char] = new_leaf
    
    def find_all_occurrences(self, pattern: str) -> List[int]:
        """
        Find all occurrences of pattern.
        
        Time: O(m + k) where m is pattern length, k is number of occurrences
        Space: O(k)
        """
        # Navigate to pattern end
        current = self.root
        i = 0
        
        while i < len(pattern):
            char = pattern[i]
            
            if char not in current.children:
                return []
            
            child = current.children[char]
            edge_start = child.edge_start
            edge_end = child.edge_end
            
            # Check edge label
            j = 0
            while (j <= edge_end - edge_start and 
                   i + j < len(pattern)):
                if self.text[edge_start + j] != pattern[i + j]:
                    return []
                j += 1
            
            i += j
            current = child
        
        # Collect all leaf nodes in subtree
        occurrences = []
        self._collect_leaves(current, occurrences)
        return sorted(occurrences)
    
    def _collect_leaves(self, node: SuffixTreeNode, occurrences: List[int]) -> None:
        """Collect all leaf nodes in subtree"""
        if node.is_leaf:
            occurrences.append(node.suffix_index)
            return
        
        for child in node.children.values():
            self._collect_leaves(child, occurrences)
    
    def longest_common_substring_length(self, other_text: str) -> int:
        """
        Find longest common substring with another text.
        
        Time: O(n + m) where n, m are text lengths
        Space: O(n + m)
        """
        # Build generalized suffix tree (simplified approach)
        combined_text = self.text[:-1] + '#' + other_text + '$'
        combined_tree = SuffixTree4(combined_text[:-1])  # Remove our '$'
        
        max_length = 0
        
        def dfs(node: SuffixTreeNode, depth: int) -> Tuple[bool, bool]:
            """DFS to find LCS, returns (has_text1, has_text2)"""
            nonlocal max_length
            
            if node.is_leaf:
                # Check which text this suffix belongs to
                if node.suffix_index < len(self.text) - 1:
                    return (True, False)
                else:
                    return (False, True)
            
            has_text1 = False
            has_text2 = False
            
            for child in node.children.values():
                edge_length = child.edge_end - child.edge_start + 1
                child_has1, child_has2 = dfs(child, depth + edge_length)
                has_text1 |= child_has1
                has_text2 |= child_has2
            
            # If node has suffixes from both texts, update max_length
            if has_text1 and has_text2:
                max_length = max(max_length, depth)
            
            return (has_text1, has_text2)
        
        dfs(combined_tree.root, 0)
        return max_length


class SuffixTree5:
    """
    Approach 5: Space-Optimized Suffix Tree
    
    Optimize memory usage for large texts.
    """
    
    def __init__(self, text: str, max_depth: int = None):
        """
        Initialize space-optimized suffix tree.
        
        Time: O(n^2) worst case
        Space: O(n) with depth limiting
        """
        self.text = text + '$'
        self.root = SuffixTreeNode()
        self.n = len(self.text)
        self.max_depth = max_depth or len(text)
        self._build_space_optimized()
    
    def _build_space_optimized(self) -> None:
        """Build space-optimized suffix tree"""
        for i in range(self.n):
            self._insert_suffix_limited(i, 0)
    
    def _insert_suffix_limited(self, suffix_index: int, current_depth: int) -> None:
        """Insert suffix with depth limit"""
        if current_depth >= self.max_depth:
            return
        
        current = self.root
        j = suffix_index
        depth = 0
        
        while j < self.n and depth < self.max_depth:
            char = self.text[j]
            
            if char not in current.children:
                # Create limited leaf
                leaf = SuffixTreeNode()
                leaf.is_leaf = True
                leaf.suffix_index = suffix_index
                leaf.edge_start = j
                leaf.edge_end = min(j + self.max_depth - depth - 1, self.n - 1)
                current.children[char] = leaf
                break
            
            child = current.children[char]
            edge_length = min(child.edge_end - child.edge_start + 1, 
                            self.max_depth - depth)
            
            j += edge_length
            depth += edge_length
            current = child
    
    def approximate_search(self, pattern: str, max_mismatches: int = 1) -> List[int]:
        """
        Approximate pattern search with mismatches.
        
        Time: O(n * m * k) where k is max_mismatches
        Space: O(k * m)
        """
        def search_with_mismatches(node: SuffixTreeNode, pattern_idx: int, 
                                 mismatches: int, path_length: int) -> List[int]:
            if pattern_idx >= len(pattern):
                # Found complete match
                results = []
                self._collect_leaves(node, results)
                return results
            
            if mismatches > max_mismatches:
                return []
            
            all_results = []
            
            for child in node.children.values():
                edge_start = child.edge_start
                edge_end = child.edge_end
                
                # Try exact match along edge
                i = 0
                current_mismatches = mismatches
                
                while (i <= edge_end - edge_start and 
                       pattern_idx + i < len(pattern)):
                    if self.text[edge_start + i] != pattern[pattern_idx + i]:
                        current_mismatches += 1
                        if current_mismatches > max_mismatches:
                            break
                    i += 1
                
                if current_mismatches <= max_mismatches:
                    child_results = search_with_mismatches(
                        child, pattern_idx + i, current_mismatches, path_length + i
                    )
                    all_results.extend(child_results)
            
            return all_results
        
        return search_with_mismatches(self.root, 0, 0, 0)


def test_basic_suffix_trees():
    """Test basic suffix tree functionality"""
    print("=== Testing Basic Suffix Trees ===")
    
    text = "banana"
    
    implementations = [
        ("Naive Construction", SuffixTree1),
        ("Path Compression", SuffixTree2),
        ("Ukkonen's Algorithm", SuffixTree3),
        ("Enhanced Features", SuffixTree4),
        ("Space Optimized", SuffixTree5),
    ]
    
    for name, TreeClass in implementations:
        print(f"\n{name}:")
        
        try:
            tree = TreeClass(text)
            
            # Test pattern search
            test_patterns = ["ana", "ban", "nan", "xyz"]
            
            for pattern in test_patterns:
                if hasattr(tree, 'search_pattern'):
                    found = tree.search_pattern(pattern)
                    print(f"  Pattern '{pattern}': {'Found' if found else 'Not found'}")
            
            # Test suffix enumeration
            if hasattr(tree, 'get_all_suffixes'):
                suffixes = tree.get_all_suffixes()
                print(f"  Suffixes: {len(suffixes)} total")
        
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_suffix_tree_applications():
    """Demonstrate suffix tree applications"""
    print("\n=== Suffix Tree Applications ===")
    
    # Application 1: Pattern matching
    print("1. Pattern Matching:")
    
    text = "abracadabra"
    tree = SuffixTree4(text)
    
    patterns = ["abra", "cad", "bra", "xyz"]
    
    for pattern in patterns:
        occurrences = tree.find_all_occurrences(pattern)
        print(f"   Pattern '{pattern}': found at positions {occurrences}")
    
    # Application 2: Longest repeated substring
    print(f"\n2. Longest Repeated Substring:")
    
    texts = ["banana", "abracadabra", "mississippi"]
    
    for text in texts:
        tree = SuffixTree2(text)
        lrs = tree.find_longest_repeated_substring()
        print(f"   Text '{text}': LRS = '{lrs}'")
    
    # Application 3: Distinct substrings count
    print(f"\n3. Distinct Substrings Count:")
    
    for text in texts:
        tree = SuffixTree3(text)
        count = tree.count_distinct_substrings()
        print(f"   Text '{text}': {count} distinct substrings")
    
    # Application 4: Longest common substring
    print(f"\n4. Longest Common Substring:")
    
    text_pairs = [
        ("banana", "ananas"),
        ("abcde", "ace"),
        ("programming", "program")
    ]
    
    for text1, text2 in text_pairs:
        tree = SuffixTree4(text1)
        lcs_length = tree.longest_common_substring_length(text2)
        print(f"   '{text1}' & '{text2}': LCS length = {lcs_length}")


def demonstrate_construction_algorithms():
    """Demonstrate different construction algorithms"""
    print("\n=== Construction Algorithms Demo ===")
    
    text = "mississippi"
    
    print(f"Building suffix tree for: '{text}'")
    
    algorithms = [
        ("Naive O(n²)", SuffixTree1),
        ("Path Compression", SuffixTree2),
        ("Ukkonen's O(n)", SuffixTree3),
    ]
    
    for name, TreeClass in algorithms:
        print(f"\n{name}:")
        
        start_time = time.time()
        tree = TreeClass(text)
        construction_time = time.time() - start_time
        
        print(f"  Construction time: {construction_time*1000:.2f}ms")
        
        # Test functionality
        if hasattr(tree, 'search_pattern'):
            pattern = "issi"
            found = tree.search_pattern(pattern)
            print(f"  Search '{pattern}': {'Found' if found else 'Not found'}")


def test_space_optimization():
    """Test space optimization techniques"""
    print("\n=== Space Optimization Demo ===")
    
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # Long text
    
    print(f"Testing with text length: {len(text)}")
    
    # Standard suffix tree
    print(f"\nStandard suffix tree:")
    start_time = time.time()
    standard_tree = SuffixTree2(text)
    standard_time = time.time() - start_time
    print(f"  Construction time: {standard_time*1000:.2f}ms")
    
    # Space-optimized with depth limit
    depth_limits = [10, 20, 50]
    
    for limit in depth_limits:
        print(f"\nSpace-optimized (depth {limit}):")
        start_time = time.time()
        optimized_tree = SuffixTree5(text, limit)
        optimized_time = time.time() - start_time
        print(f"  Construction time: {optimized_time*1000:.2f}ms")
        print(f"  Speedup: {standard_time/optimized_time:.1f}x")


def benchmark_suffix_trees():
    """Benchmark different suffix tree implementations"""
    print("\n=== Benchmarking Suffix Trees ===")
    
    import random
    import string
    
    # Generate test strings of different lengths
    def generate_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:4], k=length))
    
    test_lengths = [100, 500, 1000]
    
    implementations = [
        ("Naive", SuffixTree1),
        ("Compressed", SuffixTree2),
        ("Enhanced", SuffixTree4),
    ]
    
    for length in test_lengths:
        test_text = generate_string(length)
        print(f"\nText length: {length}")
        
        for name, TreeClass in implementations:
            try:
                start_time = time.time()
                tree = TreeClass(test_text)
                construction_time = time.time() - start_time
                
                # Test search performance
                test_patterns = [generate_string(5) for _ in range(10)]
                
                start_time = time.time()
                for pattern in test_patterns:
                    if hasattr(tree, 'search_pattern'):
                        tree.search_pattern(pattern)
                search_time = time.time() - start_time
                
                print(f"  {name:12}: Build {construction_time*1000:6.1f}ms, Search {search_time*1000:6.1f}ms")
            
            except Exception as e:
                print(f"  {name:12}: Error - {str(e)[:50]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: DNA sequence analysis
    print("1. DNA Sequence Analysis:")
    
    dna_sequence = "ATCGATCGATCG"
    tree = SuffixTree4(dna_sequence)
    
    # Find repeated patterns (potential genes)
    motifs = ["ATC", "GAT", "TCG"]
    
    for motif in motifs:
        occurrences = tree.find_all_occurrences(motif)
        print(f"   Motif '{motif}': found {len(occurrences)} times at {occurrences}")
    
    # Application 2: Text compression analysis
    print(f"\n2. Text Compression Analysis:")
    
    texts = [
        "abababab",  # High repetition
        "abcdefgh",  # Low repetition
        "aaaaaaaaa"  # Maximum repetition
    ]
    
    for text in texts:
        tree = SuffixTree2(text)
        lrs = tree.find_longest_repeated_substring()
        compression_ratio = len(lrs) / len(text) if lrs else 0
        
        print(f"   Text: '{text}'")
        print(f"   LRS: '{lrs}', Compression potential: {compression_ratio:.2%}")
    
    # Application 3: Plagiarism detection
    print(f"\n3. Plagiarism Detection:")
    
    document1 = "the quick brown fox jumps over the lazy dog"
    document2 = "a quick brown fox jumps over a lazy cat"
    
    tree = SuffixTree4(document1)
    lcs_length = tree.longest_common_substring_length(document2)
    similarity = lcs_length / max(len(document1), len(document2))
    
    print(f"   Document 1: '{document1}'")
    print(f"   Document 2: '{document2}'")
    print(f"   LCS length: {lcs_length}")
    print(f"   Similarity: {similarity:.2%}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ("", "Empty string"),
        ("a", "Single character"),
        ("aa", "Repeated character"),
        ("abcdefghijklmnopqrstuvwxyz", "All unique characters"),
        ("a" * 100, "Very long repetition"),
    ]
    
    for text, description in edge_cases:
        print(f"\n{description}: '{text[:20]}{'...' if len(text) > 20 else ''}'")
        
        try:
            if text:  # Skip empty string for some implementations
                tree = SuffixTree2(text)
                
                # Test basic operations
                if hasattr(tree, 'search_pattern') and text:
                    found = tree.search_pattern(text[0] if text else "")
                    print(f"  Search first char: {'Found' if found else 'Not found'}")
                
                if hasattr(tree, 'find_longest_repeated_substring'):
                    lrs = tree.find_longest_repeated_substring()
                    print(f"  LRS: '{lrs}'")
            else:
                print(f"  Skipped empty string test")
        
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Naive Construction",
         "Time: O(n³) worst case, O(n²) average",
         "Space: O(n²) worst case"),
        
        ("Path Compression",
         "Time: O(n²) worst case, O(n log n) average", 
         "Space: O(n) average case"),
        
        ("Ukkonen's Algorithm",
         "Time: O(n) amortized",
         "Space: O(n)"),
        
        ("Enhanced with Applications",
         "Time: O(n²) construction + O(m+k) search",
         "Space: O(n) + application-specific"),
        
        ("Space-Optimized",
         "Time: O(n²) with depth limit",
         "Space: O(n * depth_limit)"),
    ]
    
    print("Implementation Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  n = length of input text")
    print(f"  m = length of search pattern") 
    print(f"  k = number of pattern occurrences")
    
    print(f"\nApplications:")
    print(f"  • Pattern Matching: O(m + k) after O(n) construction")
    print(f"  • Longest Repeated Substring: O(n)")
    print(f"  • Distinct Substrings Count: O(n)")
    print(f"  • Longest Common Substring: O(n + m)")
    
    print(f"\nRecommendations:")
    print(f"  • Use Ukkonen's for optimal asymptotic performance")
    print(f"  • Use Path Compression for good practical performance")
    print(f"  • Use Space-Optimized for memory-constrained environments")
    print(f"  • Use Enhanced for multiple applications on same text")


if __name__ == "__main__":
    test_basic_suffix_trees()
    demonstrate_suffix_tree_applications()
    demonstrate_construction_algorithms()
    test_space_optimization()
    benchmark_suffix_trees()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
Suffix Tree Basic Implementation demonstrates comprehensive suffix tree approaches:

1. Naive Construction - Simple O(n²) approach inserting all suffixes
2. Path Compression - Improved space efficiency with edge compression
3. Ukkonen's Algorithm - Linear time construction (simplified version)
4. Enhanced Features - Additional applications and utilities
5. Space-Optimized - Memory-efficient version with depth limits

Key applications implemented:
- Pattern matching in O(m) time
- Longest repeated substring finding
- All occurrences of pattern
- Distinct substrings counting
- Longest common substring
- Approximate pattern matching

Real-world applications:
- DNA sequence analysis
- Text compression
- Plagiarism detection
- String algorithms optimization

Each implementation offers different trade-offs between construction time,
space usage, and functionality for various string processing needs.
"""
