"""
Advanced String Reconstruction - Multiple Approaches
Difficulty: Hard

Advanced string reconstruction problems using trie-based dynamic programming.
This combines multiple complex string reconstruction scenarios:

1. Reconstruct string from encoded patterns
2. String reconstruction with wildcards and constraints
3. Multi-source string reconstruction
4. Probabilistic string reconstruction
5. String reconstruction with error correction
6. Real-time streaming reconstruction

Applications:
- Data compression and decompression
- Error correction in communications
- DNA sequence reconstruction
- Natural language processing
- Distributed system message reconstruction
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import heapq
import random
import time

class TrieNode:
    """Enhanced trie node for advanced reconstruction"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""
        self.frequency = 0
        self.sources = set()  # Track sources that contributed this pattern
        self.confidence = 1.0  # Confidence score for probabilistic reconstruction

class AdvancedStringReconstructor:
    
    def __init__(self):
        """Initialize advanced string reconstructor"""
        self.root = TrieNode()
        self.word_patterns = {}  # pattern -> set of words
        self.encoding_rules = {}  # char -> encoded_form
        self.decoding_rules = {}  # encoded_form -> char
    
    def reconstruct_from_encoded_patterns(self, encoded_patterns: List[str], 
                                        encoding_map: Dict[str, str]) -> List[str]:
        """
        Approach 1: Reconstruct from Encoded Patterns
        
        Given encoded patterns and encoding rules, reconstruct original strings.
        
        Time: O(n * m * k) where n=patterns, m=pattern length, k=alphabet size
        Space: O(n * m)
        """
        # Build decoding map
        self.decoding_rules = {v: k for k, v in encoding_map.items()}
        
        # Build trie of encoded patterns
        encoded_root = TrieNode()
        
        for pattern in encoded_patterns:
            node = encoded_root
            for encoded_char in pattern.split():  # Assume space-separated encoding
                if encoded_char not in node.children:
                    node.children[encoded_char] = TrieNode()
                node = node.children[encoded_char]
            node.is_word = True
            node.word = pattern
        
        # Reconstruct using DP
        def reconstruct_pattern(encoded_pattern: str) -> List[str]:
            """Reconstruct all possible original strings from encoded pattern"""
            tokens = encoded_pattern.split()
            n = len(tokens)
            
            # dp[i] = list of possible reconstructions for tokens[i:]
            dp = [[] for _ in range(n + 1)]
            dp[n] = [""]  # Empty string
            
            for i in range(n - 1, -1, -1):
                encoded_char = tokens[i]
                
                if encoded_char in self.decoding_rules:
                    original_char = self.decoding_rules[encoded_char]
                    
                    for suffix in dp[i + 1]:
                        dp[i].append(original_char + suffix)
                else:
                    # Try all possible decodings
                    for original_char, encoding in encoding_map.items():
                        if encoding == encoded_char:
                            for suffix in dp[i + 1]:
                                dp[i].append(original_char + suffix)
            
            return dp[0]
        
        results = []
        for pattern in encoded_patterns:
            reconstructed = reconstruct_pattern(pattern)
            results.extend(reconstructed)
        
        return list(set(results))  # Remove duplicates
    
    def reconstruct_with_wildcards(self, pattern: str, dictionary: List[str], 
                                 wildcard: str = '*') -> List[str]:
        """
        Approach 2: Reconstruct with Wildcards
        
        Reconstruct strings matching pattern with wildcards.
        
        Time: O(d * p) where d=dictionary size, p=pattern length
        Space: O(d * p)
        """
        # Build trie from dictionary
        dict_root = TrieNode()
        
        for word in dictionary:
            node = dict_root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word = word
        
        # DP matching with wildcards
        def matches_pattern(word: str, pattern: str) -> bool:
            """Check if word matches pattern with wildcards"""
            m, n = len(word), len(pattern)
            
            # dp[i][j] = True if word[:i] matches pattern[:j]
            dp = [[False] * (n + 1) for _ in range(m + 1)]
            dp[0][0] = True
            
            # Handle leading wildcards
            for j in range(1, n + 1):
                if pattern[j - 1] == wildcard:
                    dp[0][j] = dp[0][j - 1]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pattern[j - 1] == wildcard:
                        # Wildcard can match empty or any character
                        dp[i][j] = dp[i][j - 1] or dp[i - 1][j] or dp[i - 1][j - 1]
                    elif word[i - 1] == pattern[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
            
            return dp[m][n]
        
        results = []
        for word in dictionary:
            if matches_pattern(word, pattern):
                results.append(word)
        
        return results
    
    def multi_source_reconstruction(self, sources: Dict[str, List[str]], 
                                  target_length: int) -> List[str]:
        """
        Approach 3: Multi-source String Reconstruction
        
        Reconstruct strings from multiple sources with different patterns.
        
        Time: O(s * n * m) where s=sources, n=strings per source, m=string length
        Space: O(total_strings * m)
        """
        # Build multi-source trie
        multi_root = TrieNode()
        
        for source_id, strings in sources.items():
            for string in strings:
                node = multi_root
                for char in string:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                    node.sources.add(source_id)
                
                node.is_word = True
                node.word = string
                node.sources.add(source_id)
        
        # Find strings that appear in multiple sources
        def find_consensus_strings(min_sources: int = 2) -> List[Tuple[str, Set[str]]]:
            """Find strings supported by minimum number of sources"""
            consensus = []
            
            def dfs(node: TrieNode, path: str) -> None:
                if node.is_word and len(node.sources) >= min_sources:
                    if len(path) == target_length:
                        consensus.append((path, node.sources.copy()))
                
                if len(path) < target_length:
                    for char, child in node.children.items():
                        dfs(child, path + char)
            
            dfs(multi_root, "")
            return consensus
        
        consensus_strings = find_consensus_strings()
        
        # Rank by number of supporting sources
        consensus_strings.sort(key=lambda x: len(x[1]), reverse=True)
        
        return [string for string, sources in consensus_strings]
    
    def probabilistic_reconstruction(self, fragments: List[Tuple[str, float]], 
                                   target_length: int) -> List[Tuple[str, float]]:
        """
        Approach 4: Probabilistic String Reconstruction
        
        Reconstruct strings with probability scores.
        
        Time: O(f * l * 2^l) where f=fragments, l=target_length
        Space: O(2^l * l)
        """
        # Build probabilistic trie
        prob_root = TrieNode()
        
        for fragment, probability in fragments:
            node = prob_root
            for char in fragment:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Update confidence using probability
                node.confidence = max(node.confidence * probability, 
                                    node.confidence + probability - node.confidence * probability)
            
            node.is_word = True
            node.word = fragment
            node.confidence = probability
        
        # Generate all possible reconstructions with scores
        def generate_with_probability(max_results: int = 10) -> List[Tuple[str, float]]:
            """Generate top reconstructions by probability"""
            # Priority queue: (-probability, string)
            pq = [(-1.0, "")]  # Start with empty string, probability 1.0
            results = []
            visited = set()
            
            while pq and len(results) < max_results:
                neg_prob, current_string = heapq.heappop(pq)
                prob = -neg_prob
                
                if len(current_string) == target_length:
                    if current_string not in visited:
                        results.append((current_string, prob))
                        visited.add(current_string)
                    continue
                
                if len(current_string) >= target_length:
                    continue
                
                # Try extending with each possible character
                node = prob_root
                valid_path = True
                
                # Navigate to current position in trie
                for char in current_string:
                    if char in node.children:
                        node = node.children[char]
                    else:
                        valid_path = False
                        break
                
                if valid_path:
                    for char, child in node.children.items():
                        new_string = current_string + char
                        new_prob = prob * child.confidence
                        
                        if new_prob > 0.01:  # Threshold to avoid very low probability paths
                            heapq.heappush(pq, (-new_prob, new_string))
            
            return results
        
        return generate_with_probability()
    
    def error_correction_reconstruction(self, corrupted_strings: List[str], 
                                      dictionary: List[str], max_errors: int = 2) -> Dict[str, List[str]]:
        """
        Approach 5: Error Correction Reconstruction
        
        Reconstruct original strings from corrupted versions.
        
        Time: O(c * d * m^e) where c=corrupted, d=dictionary, m=string length, e=max_errors
        Space: O(d * m)
        """
        # Build dictionary trie
        dict_root = TrieNode()
        
        for word in dictionary:
            node = dict_root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word = word
        
        def find_corrections(corrupted: str) -> List[str]:
            """Find possible corrections for corrupted string"""
            corrections = []
            
            def dfs(node: TrieNode, pos: int, errors: int, path: str) -> None:
                if errors > max_errors:
                    return
                
                if node.is_word and abs(len(path) - len(corrupted)) <= max_errors:
                    # Calculate total edit distance
                    edit_dist = self._edit_distance(corrupted, path)
                    if edit_dist <= max_errors:
                        corrections.append(path)
                
                if pos < len(corrupted):
                    char = corrupted[pos]
                    
                    # Exact match
                    if char in node.children:
                        dfs(node.children[char], pos + 1, errors, path + char)
                    
                    # Substitution
                    for c, child in node.children.items():
                        if c != char:
                            dfs(child, pos + 1, errors + 1, path + c)
                    
                    # Deletion (skip character in corrupted string)
                    dfs(node, pos + 1, errors + 1, path)
                
                # Insertion (add character from trie)
                for c, child in node.children.items():
                    dfs(child, pos, errors + 1, path + c)
            
            dfs(dict_root, 0, 0, "")
            return corrections
        
        results = {}
        for corrupted in corrupted_strings:
            results[corrupted] = find_corrections(corrupted)
        
        return results
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        
        return dp[m][n]
    
    def streaming_reconstruction(self, stream_generator, window_size: int = 100) -> List[str]:
        """
        Approach 6: Real-time Streaming Reconstruction
        
        Reconstruct strings from streaming data.
        
        Time: O(w * p) per window where w=window_size, p=pattern_complexity
        Space: O(w)
        """
        window = deque(maxlen=window_size)
        reconstructed = []
        pattern_buffer = []
        
        # Build streaming trie
        stream_root = TrieNode()
        
        for chunk in stream_generator:
            window.extend(chunk)
            
            # Try to identify complete patterns in window
            window_str = ''.join(window)
            
            # Look for known patterns or word boundaries
            patterns = self._identify_patterns(window_str)
            
            for pattern in patterns:
                # Add to trie
                node = stream_root
                for char in pattern:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                    node.frequency += 1
                
                node.is_word = True
                node.word = pattern
                
                if len(pattern) >= 3:  # Minimum meaningful pattern length
                    reconstructed.append(pattern)
        
        # Final reconstruction from accumulated patterns
        final_patterns = self._extract_meaningful_patterns(stream_root)
        
        return list(set(reconstructed + final_patterns))
    
    def _identify_patterns(self, text: str) -> List[str]:
        """Identify patterns in text using heuristics"""
        patterns = []
        
        # Simple word boundary detection
        words = []
        current_word = ""
        
        for char in text:
            if char.isalpha():
                current_word += char
            else:
                if current_word and len(current_word) >= 2:
                    words.append(current_word)
                current_word = ""
        
        if current_word and len(current_word) >= 2:
            words.append(current_word)
        
        # Look for repeated patterns
        for word in words:
            if len(word) >= 3:
                patterns.append(word)
        
        return patterns
    
    def _extract_meaningful_patterns(self, root: TrieNode) -> List[str]:
        """Extract meaningful patterns from trie based on frequency"""
        patterns = []
        
        def dfs(node: TrieNode, path: str) -> None:
            if node.is_word and node.frequency >= 2:  # Appeared at least twice
                patterns.append(path)
            
            for char, child in node.children.items():
                dfs(child, path + char)
        
        dfs(root, "")
        return patterns
    
    def reconstruct_with_constraints(self, partial_strings: List[str], 
                                   constraints: Dict[str, Any]) -> List[str]:
        """
        Approach 7: Reconstruction with Complex Constraints
        
        Reconstruct strings satisfying multiple constraints.
        
        Time: O(exponential) with constraint pruning
        Space: O(result_size * max_string_length)
        """
        results = []
        
        # Extract constraints
        min_length = constraints.get('min_length', 0)
        max_length = constraints.get('max_length', float('inf'))
        required_chars = set(constraints.get('required_chars', []))
        forbidden_chars = set(constraints.get('forbidden_chars', []))
        pattern_regex = constraints.get('pattern_regex', None)
        char_frequency = constraints.get('char_frequency', {})
        
        def satisfies_constraints(s: str) -> bool:
            """Check if string satisfies all constraints"""
            if not (min_length <= len(s) <= max_length):
                return False
            
            s_chars = set(s)
            if not required_chars.issubset(s_chars):
                return False
            
            if s_chars & forbidden_chars:
                return False
            
            # Check character frequency constraints
            for char, (min_freq, max_freq) in char_frequency.items():
                freq = s.count(char)
                if not (min_freq <= freq <= max_freq):
                    return False
            
            # Pattern regex check would go here if needed
            
            return True
        
        def generate_combinations(partial: str, remaining_partials: List[str]) -> None:
            """Generate all valid combinations"""
            if not remaining_partials:
                if satisfies_constraints(partial):
                    results.append(partial)
                return
            
            # Early pruning
            if len(partial) > max_length:
                return
            
            current_chars = set(partial)
            if current_chars & forbidden_chars:
                return
            
            # Try adding next partial string
            for i, next_partial in enumerate(remaining_partials):
                new_partial = partial + next_partial
                new_remaining = remaining_partials[:i] + remaining_partials[i+1:]
                generate_combinations(new_partial, new_remaining)
                
                # Also try with separator
                if len(partial) > 0:
                    for sep in [' ', '-', '_']:
                        if sep not in forbidden_chars:
                            sep_partial = partial + sep + next_partial
                            generate_combinations(sep_partial, new_remaining)
        
        # Generate all possible combinations
        generate_combinations("", partial_strings)
        
        return list(set(results))


def test_basic_functionality():
    """Test basic reconstruction functionality"""
    print("=== Testing Basic Reconstruction Functionality ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    # Test 1: Encoded pattern reconstruction
    print("1. Encoded Pattern Reconstruction:")
    
    encoding_map = {'a': '01', 'b': '10', 'c': '11'}
    encoded_patterns = ["01 10", "11 01", "10 11"]
    
    print(f"   Encoding map: {encoding_map}")
    print(f"   Encoded patterns: {encoded_patterns}")
    
    decoded = reconstructor.reconstruct_from_encoded_patterns(encoded_patterns, encoding_map)
    print(f"   Decoded strings: {decoded}")
    
    # Test 2: Wildcard reconstruction
    print(f"\n2. Wildcard Reconstruction:")
    
    pattern = "c*t"
    dictionary = ["cat", "cut", "cart", "coat", "chat", "dog"]
    
    print(f"   Pattern: '{pattern}'")
    print(f"   Dictionary: {dictionary}")
    
    matches = reconstructor.reconstruct_with_wildcards(pattern, dictionary)
    print(f"   Matches: {matches}")
    
    # Test 3: Multi-source reconstruction
    print(f"\n3. Multi-source Reconstruction:")
    
    sources = {
        'source1': ['hello', 'world', 'test'],
        'source2': ['hello', 'test', 'data'],
        'source3': ['world', 'test', 'info']
    }
    
    print(f"   Sources: {sources}")
    
    consensus = reconstructor.multi_source_reconstruction(sources, target_length=5)
    print(f"   Consensus strings: {consensus}")
    
    # Test 4: Error correction
    print(f"\n4. Error Correction:")
    
    dictionary = ["hello", "world", "test", "data", "info"]
    corrupted = ["helo", "worl", "tset"]
    
    print(f"   Dictionary: {dictionary}")
    print(f"   Corrupted: {corrupted}")
    
    corrections = reconstructor.error_correction_reconstruction(corrupted, dictionary)
    
    for corrupt, candidates in corrections.items():
        print(f"   '{corrupt}' -> {candidates}")


def demonstrate_probabilistic_reconstruction():
    """Demonstrate probabilistic reconstruction"""
    print("\n=== Probabilistic Reconstruction Demo ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    # Fragments with probabilities
    fragments = [
        ("hel", 0.9),
        ("ell", 0.8),
        ("llo", 0.85),
        ("wor", 0.7),
        ("orl", 0.75),
        ("rld", 0.8),
        ("hel", 0.6),  # Lower confidence duplicate
    ]
    
    target_length = 5
    
    print(f"Fragments with probabilities: {fragments}")
    print(f"Target length: {target_length}")
    
    reconstructions = reconstructor.probabilistic_reconstruction(fragments, target_length)
    
    print(f"\nTop reconstructions:")
    for i, (string, prob) in enumerate(reconstructions):
        print(f"  {i+1}: '{string}' (probability: {prob:.3f})")


def demonstrate_streaming_reconstruction():
    """Demonstrate streaming reconstruction"""
    print("\n=== Streaming Reconstruction Demo ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    # Simulate streaming data
    def stream_generator():
        """Generator for streaming text data"""
        stream_data = [
            "hel", "lo ", "wor", "ld ", "tes", "t d", "ata", " he",
            "llo", " ag", "ain", " wo", "rld", " mo", "re ", "tes", "ts"
        ]
        
        for chunk in stream_data:
            yield chunk
    
    print("Simulating streaming reconstruction...")
    print("Stream chunks: ['hel', 'lo ', 'wor', 'ld ', 'tes', 't d', 'ata', ...]")
    
    reconstructed = reconstructor.streaming_reconstruction(stream_generator(), window_size=20)
    
    print(f"\nReconstructed patterns: {reconstructed}")


def demonstrate_constraint_reconstruction():
    """Demonstrate reconstruction with constraints"""
    print("\n=== Constraint-based Reconstruction Demo ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    partial_strings = ["hel", "lo", "wor", "ld"]
    
    constraints = {
        'min_length': 8,
        'max_length': 12,
        'required_chars': {'h', 'l', 'o'},
        'forbidden_chars': {'x', 'z'},
        'char_frequency': {'l': (2, 4)}  # 'l' should appear 2-4 times
    }
    
    print(f"Partial strings: {partial_strings}")
    print(f"Constraints: {constraints}")
    
    valid_reconstructions = reconstructor.reconstruct_with_constraints(partial_strings, constraints)
    
    print(f"\nValid reconstructions:")
    for i, reconstruction in enumerate(valid_reconstructions):
        print(f"  {i+1}: '{reconstruction}'")


def benchmark_reconstruction_approaches():
    """Benchmark different reconstruction approaches"""
    print("\n=== Benchmarking Reconstruction Approaches ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    # Generate test data
    def generate_encoded_data(size: int):
        encoding_map = {chr(ord('a') + i): f"{i:02d}" for i in range(10)}
        patterns = []
        
        for _ in range(size):
            length = random.randint(3, 8)
            chars = random.choices(list(encoding_map.keys()), k=length)
            encoded = " ".join(encoding_map[c] for c in chars)
            patterns.append(encoded)
        
        return patterns, encoding_map
    
    def generate_wildcard_data(size: int):
        dictionary = ["cat", "bat", "hat", "cut", "but", "hut", "cart", "bart", "hart"]
        patterns = ["c*t", "b*t", "h*t", "*at", "*ut"]
        
        return random.choices(patterns, k=size), dictionary
    
    test_scenarios = [
        ("Small", 20),
        ("Medium", 50),
        ("Large", 100),
    ]
    
    for scenario_name, size in test_scenarios:
        print(f"\n--- {scenario_name} Dataset (size: {size}) ---")
        
        # Test encoded pattern reconstruction
        patterns, encoding_map = generate_encoded_data(size)
        
        start_time = time.time()
        decoded = reconstructor.reconstruct_from_encoded_patterns(patterns, encoding_map)
        encoded_time = (time.time() - start_time) * 1000
        
        print(f"  Encoded reconstruction: {len(decoded)} results in {encoded_time:.2f}ms")
        
        # Test wildcard reconstruction
        wildcard_patterns, dictionary = generate_wildcard_data(min(size, 10))
        
        start_time = time.time()
        total_matches = 0
        
        for pattern in wildcard_patterns:
            matches = reconstructor.reconstruct_with_wildcards(pattern, dictionary)
            total_matches += len(matches)
        
        wildcard_time = (time.time() - start_time) * 1000
        
        print(f"  Wildcard reconstruction: {total_matches} matches in {wildcard_time:.2f}ms")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    # Application 1: DNA sequence reconstruction
    print("1. DNA Sequence Reconstruction:")
    
    fragments = [
        ("ATG", 0.95),  # Start codon
        ("TGC", 0.88),
        ("GCA", 0.92),
        ("CAT", 0.85),
        ("ATT", 0.90),
        ("TAG", 0.87),  # Stop codon
    ]
    
    print(f"   DNA fragments: {fragments}")
    
    dna_reconstructions = reconstructor.probabilistic_reconstruction(fragments, target_length=6)
    
    print(f"   Possible sequences:")
    for seq, prob in dna_reconstructions[:3]:
        print(f"     {seq} (confidence: {prob:.3f})")
    
    # Application 2: Message reconstruction from corrupted transmission
    print(f"\n2. Corrupted Message Reconstruction:")
    
    dictionary = ["hello", "world", "message", "received", "status", "error", "success"]
    corrupted_messages = ["helo", "worl", "mesage", "recived", "sucess"]
    
    print(f"   Known words: {dictionary}")
    print(f"   Corrupted: {corrupted_messages}")
    
    corrections = reconstructor.error_correction_reconstruction(corrupted_messages, dictionary)
    
    print(f"   Corrections:")
    for corrupt, candidates in corrections.items():
        if candidates:
            print(f"     '{corrupt}' -> {candidates[0]} (most likely)")
    
    # Application 3: Log file reconstruction
    print(f"\n3. Log File Pattern Reconstruction:")
    
    def log_stream():
        """Simulate log stream"""
        log_entries = [
            "[INF", "O] ", "Ser", "ver", " st", "art", "ed\n",
            "[ERR", "OR]", " Co", "nne", "cti", "on ", "fai", "led", "\n",
            "[INF", "O] ", "Req", "ues", "t p", "roc", "ess", "ed\n"
        ]
        
        for entry in log_entries:
            yield entry
    
    print(f"   Reconstructing from fragmented log stream...")
    
    log_patterns = reconstructor.streaming_reconstruction(log_stream(), window_size=30)
    
    print(f"   Identified patterns:")
    for pattern in log_patterns[:5]:
        print(f"     '{pattern}'")
    
    # Application 4: Code reconstruction from obfuscated source
    print(f"\n4. Code Reconstruction:")
    
    obfuscated_fragments = ["func", "tion", "main", "retu", "rn"]
    code_constraints = {
        'min_length': 8,
        'max_length': 20,
        'required_chars': {'f', 'n'},
        'forbidden_chars': {' '},
    }
    
    print(f"   Code fragments: {obfuscated_fragments}")
    print(f"   Constraints: {code_constraints}")
    
    code_reconstructions = reconstructor.reconstruct_with_constraints(
        obfuscated_fragments, code_constraints
    )
    
    print(f"   Possible reconstructions:")
    for reconstruction in code_reconstructions[:3]:
        print(f"     '{reconstruction}'")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    reconstructor = AdvancedStringReconstructor()
    
    # Edge case 1: Empty inputs
    print("1. Empty Inputs:")
    
    empty_encoded = reconstructor.reconstruct_from_encoded_patterns([], {})
    print(f"   Empty encoded patterns: {empty_encoded}")
    
    empty_wildcard = reconstructor.reconstruct_with_wildcards("*", [])
    print(f"   Empty dictionary with wildcard: {empty_wildcard}")
    
    # Edge case 2: Single character patterns
    print(f"\n2. Single Character Patterns:")
    
    single_char_dict = ["a", "b", "c"]
    single_matches = reconstructor.reconstruct_with_wildcards("*", single_char_dict)
    print(f"   Single char matches: {single_matches}")
    
    # Edge case 3: No valid reconstructions
    print(f"\n3. No Valid Reconstructions:")
    
    impossible_constraints = {
        'min_length': 10,
        'max_length': 5,  # Impossible constraint
        'required_chars': {'a', 'b', 'c'},
    }
    
    no_solutions = reconstructor.reconstruct_with_constraints(
        ["ab", "cd"], impossible_constraints
    )
    print(f"   Impossible constraints result: {no_solutions}")
    
    # Edge case 4: Very high error tolerance
    print(f"\n4. High Error Tolerance:")
    
    high_error_corrections = reconstructor.error_correction_reconstruction(
        ["xyz"], ["abc"], max_errors=5
    )
    print(f"   High error tolerance: {high_error_corrections}")
    
    # Edge case 5: Probabilistic with zero probabilities
    print(f"\n5. Zero Probability Fragments:")
    
    zero_prob_fragments = [("abc", 0.0), ("def", 0.1)]
    zero_prob_results = reconstructor.probabilistic_reconstruction(
        zero_prob_fragments, target_length=3
    )
    print(f"   Zero probability results: {zero_prob_results}")


def analyze_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Encoded Pattern Reconstruction",
         "Time: O(n * m * k) where n=patterns, m=length, k=alphabet",
         "Space: O(n * m) for storing decoded strings"),
        
        ("Wildcard Reconstruction",
         "Time: O(d * p^2) where d=dictionary size, p=pattern length",
         "Space: O(d * p) for DP table per word"),
        
        ("Multi-source Reconstruction",
         "Time: O(s * n * m) where s=sources, n=strings/source, m=length",
         "Space: O(total_strings * m) for multi-source trie"),
        
        ("Probabilistic Reconstruction",
         "Time: O(f * l * 2^l) where f=fragments, l=target_length",
         "Space: O(2^l * l) for probability combinations"),
        
        ("Error Correction Reconstruction",
         "Time: O(c * d * m^e) where c=corrupted, d=dictionary, e=max_errors",
         "Space: O(d * m) for dictionary trie"),
        
        ("Streaming Reconstruction",
         "Time: O(w * p) per window where w=window_size, p=pattern_complexity",
         "Space: O(w) for sliding window"),
        
        ("Constraint-based Reconstruction",
         "Time: O(exponential) with constraint pruning",
         "Space: O(result_size * max_string_length)"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Trie structures for efficient pattern matching")
    print(f"  • Dynamic programming for optimal substructure problems")
    print(f"  • Probability-based pruning for exponential search spaces")
    print(f"  • Constraint propagation for early termination")
    print(f"  • Streaming algorithms for memory-bounded processing")
    
    print(f"\nPractical Considerations:")
    print(f"  • Memory usage scales with pattern complexity")
    print(f"  • Error tolerance affects reconstruction accuracy")
    print(f"  • Real-time constraints limit algorithm choice")
    print(f"  • Domain-specific knowledge improves results")
    
    print(f"\nRecommendations:")
    print(f"  • Use probabilistic approaches for noisy data")
    print(f"  • Use constraint-based for structured reconstruction")
    print(f"  • Use streaming for real-time applications")
    print(f"  • Use multi-source for consensus building")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_probabilistic_reconstruction()
    demonstrate_streaming_reconstruction()
    demonstrate_constraint_reconstruction()
    benchmark_reconstruction_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()
    analyze_complexity()

"""
Advanced String Reconstruction demonstrates comprehensive reconstruction approaches:

1. Encoded Pattern Reconstruction - Decode strings from encoded representations
2. Wildcard Reconstruction - Reconstruct strings matching wildcard patterns
3. Multi-source Reconstruction - Combine information from multiple sources
4. Probabilistic Reconstruction - Use probability scores for optimal reconstruction
5. Error Correction Reconstruction - Recover original strings from corrupted versions
6. Streaming Reconstruction - Real-time reconstruction from streaming data
7. Constraint-based Reconstruction - Satisfy complex constraints during reconstruction

Key concepts:
- Trie-based pattern matching and storage
- Dynamic programming for optimal reconstruction
- Probabilistic inference and scoring
- Error correction and edit distance algorithms
- Streaming data processing with bounded memory
- Constraint satisfaction and pruning techniques

Real-world applications:
- DNA sequence reconstruction and analysis
- Corrupted message recovery in communications
- Log file pattern reconstruction and analysis
- Code reconstruction from obfuscated sources
- Data compression and decompression systems

Each approach demonstrates different strategies for handling incomplete,
corrupted, or fragmented string data reconstruction problems efficiently.
"""
