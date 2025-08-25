"""
1178. Number of Valid Words for Each Puzzle - Multiple Approaches
Difficulty: Hard

With respect to a given puzzle string, a word is valid if both the following conditions are satisfied:
- word contains the first letter of puzzle.
- For each letter in word, that letter is in puzzle.

Given an array of words and an array of puzzles, return an array answer where 
answer[i] is the number of words in the given word list that is valid with respect to puzzle[i].

Examples:
Input: words = ["aaaa","asas","able","ability","actt","actor","access"], 
       puzzles = ["aboveyz","abrodyz","abslute","absoryz","actresz","gaswxyz"]
Output: [1,1,3,2,4,0]

Approaches:
1. Brute Force with Set Operations
2. Bit Manipulation with Masks
3. Trie with Bitmask Optimization
4. Frequency-based Filtering
5. Subset Enumeration with Caching
6. Advanced Bit Manipulation with Trie
"""

from typing import List, Dict, Set
from collections import defaultdict, Counter
import time

class TrieNode:
    """Trie node with bitmask support"""
    def __init__(self):
        self.children = {}
        self.word_count = 0
        self.bitmask = 0

class PuzzleWordValidator:
    
    def __init__(self):
        """Initialize puzzle word validator"""
        pass
    
    def find_num_words_brute_force(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 1: Brute Force with Set Operations
        
        For each puzzle, check each word using set operations.
        
        Time: O(w * p * (len(word) + len(puzzle)))
        Space: O(len(puzzle)) for set operations
        """
        result = []
        
        for puzzle in puzzles:
            puzzle_set = set(puzzle)
            first_letter = puzzle[0]
            count = 0
            
            for word in words:
                # Check if word contains first letter of puzzle
                if first_letter not in word:
                    continue
                
                # Check if all letters in word are in puzzle
                word_set = set(word)
                if word_set.issubset(puzzle_set):
                    count += 1
            
            result.append(count)
        
        return result
    
    def find_num_words_bitmask(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 2: Bit Manipulation with Masks
        
        Convert words and puzzles to bitmasks for efficient operations.
        
        Time: O(w + p * 2^7) where w=words, p=puzzles
        Space: O(w) for word masks
        """
        def char_to_bit(c: str) -> int:
            """Convert character to bit position"""
            return ord(c) - ord('a')
        
        def string_to_mask(s: str) -> int:
            """Convert string to bitmask"""
            mask = 0
            for c in s:
                mask |= (1 << char_to_bit(c))
            return mask
        
        # Convert words to bitmasks and group by mask
        word_masks = defaultdict(int)
        for word in words:
            mask = string_to_mask(word)
            word_masks[mask] += 1
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = string_to_mask(puzzle)
            first_bit = 1 << char_to_bit(puzzle[0])
            count = 0
            
            # Iterate through all subsets of puzzle
            submask = puzzle_mask
            while submask > 0:
                # Check if submask contains first letter
                if submask & first_bit:
                    count += word_masks[submask]
                
                # Move to next submask
                submask = (submask - 1) & puzzle_mask
            
            result.append(count)
        
        return result
    
    def find_num_words_trie_bitmask(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 3: Trie with Bitmask Optimization
        
        Build trie based on bitmasks with count tracking.
        
        Time: O(w * unique_chars + p * 2^7)
        Space: O(trie_size)
        """
        def string_to_mask(s: str) -> int:
            """Convert string to bitmask"""
            mask = 0
            for c in s:
                mask |= (1 << (ord(c) - ord('a')))
            return mask
        
        # Build trie of word bitmasks
        root = TrieNode()
        
        for word in words:
            mask = string_to_mask(word)
            node = root
            
            # Traverse trie based on set bits
            for i in range(26):
                if mask & (1 << i):
                    if i not in node.children:
                        node.children[i] = TrieNode()
                    node = node.children[i]
                    node.bitmask |= (1 << i)
            
            node.word_count += 1
        
        def count_valid_words(node: TrieNode, puzzle_mask: int, 
                            first_bit: int, has_first: bool) -> int:
            """Count valid words in trie subtree"""
            count = 0
            
            # If current node represents complete words and has first letter
            if has_first:
                count += node.word_count
            
            # Explore children
            for bit_pos, child in node.children.items():
                bit = 1 << bit_pos
                
                # Only explore if bit is in puzzle
                if puzzle_mask & bit:
                    new_has_first = has_first or (bit & first_bit)
                    count += count_valid_words(child, puzzle_mask, first_bit, new_has_first)
            
            return count
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = string_to_mask(puzzle)
            first_bit = 1 << (ord(puzzle[0]) - ord('a'))
            
            count = count_valid_words(root, puzzle_mask, first_bit, False)
            result.append(count)
        
        return result
    
    def find_num_words_frequency_filter(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 4: Frequency-based Filtering
        
        Pre-filter words based on character frequency analysis.
        
        Time: O(w * len(word) + p * filtered_words)
        Space: O(w + unique_chars)
        """
        # Analyze character frequencies in puzzles
        puzzle_chars = set()
        for puzzle in puzzles:
            puzzle_chars.update(puzzle)
        
        # Filter words that only contain puzzle characters
        filtered_words = []
        for word in words:
            word_chars = set(word)
            if word_chars.issubset(puzzle_chars):
                filtered_words.append(word)
        
        print(f"Filtered {len(words)} words to {len(filtered_words)} words")
        
        # Use bitmask approach on filtered words
        def string_to_mask(s: str) -> int:
            mask = 0
            for c in s:
                mask |= (1 << (ord(c) - ord('a')))
            return mask
        
        word_masks = defaultdict(int)
        for word in filtered_words:
            mask = string_to_mask(word)
            word_masks[mask] += 1
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = string_to_mask(puzzle)
            first_bit = 1 << (ord(puzzle[0]) - ord('a'))
            count = 0
            
            # Check each filtered word mask
            for word_mask in word_masks:
                # Check if word is subset of puzzle and contains first letter
                if (word_mask & puzzle_mask) == word_mask and (word_mask & first_bit):
                    count += word_masks[word_mask]
            
            result.append(count)
        
        return result
    
    def find_num_words_subset_enumeration(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 5: Subset Enumeration with Caching
        
        Enumerate subsets efficiently with memoization.
        
        Time: O(w + p * 2^7)
        Space: O(w + 2^7) for caching
        """
        def string_to_mask(s: str) -> int:
            mask = 0
            for c in s:
                mask |= (1 << (ord(c) - ord('a')))
            return mask
        
        # Build word mask frequency map
        word_masks = Counter()
        for word in words:
            mask = string_to_mask(word)
            word_masks[mask] += 1
        
        # Cache for subset enumeration
        subset_cache = {}
        
        def enumerate_subsets_with_first(puzzle_mask: int, first_bit: int) -> int:
            """Enumerate all subsets containing first bit"""
            if (puzzle_mask, first_bit) in subset_cache:
                return subset_cache[(puzzle_mask, first_bit)]
            
            count = 0
            
            # Use bit manipulation to enumerate subsets
            submask = puzzle_mask
            while submask > 0:
                if submask & first_bit:  # Contains first letter
                    count += word_masks[submask]
                submask = (submask - 1) & puzzle_mask
            
            subset_cache[(puzzle_mask, first_bit)] = count
            return count
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = string_to_mask(puzzle)
            first_bit = 1 << (ord(puzzle[0]) - ord('a'))
            
            count = enumerate_subsets_with_first(puzzle_mask, first_bit)
            result.append(count)
        
        return result
    
    def find_num_words_advanced_trie(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 6: Advanced Bit Manipulation with Trie
        
        Optimized trie with bit manipulation and pruning.
        
        Time: O(w * 26 + p * trie_nodes)
        Space: O(trie_size)
        """
        class AdvancedTrieNode:
            __slots__ = ['children', 'count', 'has_words']
            
            def __init__(self):
                self.children = {}  # bit_position -> child
                self.count = 0      # words ending here
                self.has_words = False
        
        def string_to_mask(s: str) -> int:
            mask = 0
            for c in s:
                mask |= (1 << (ord(c) - ord('a')))
            return mask
        
        def mask_to_sorted_bits(mask: int) -> List[int]:
            """Convert mask to sorted list of bit positions"""
            bits = []
            for i in range(26):
                if mask & (1 << i):
                    bits.append(i)
            return bits
        
        # Build advanced trie
        root = AdvancedTrieNode()
        
        for word in words:
            mask = string_to_mask(word)
            bits = mask_to_sorted_bits(mask)
            
            node = root
            for bit in bits:
                if bit not in node.children:
                    node.children[bit] = AdvancedTrieNode()
                node = node.children[bit]
                node.has_words = True
            
            node.count += 1
        
        def dfs_count(node: AdvancedTrieNode, puzzle_bits: Set[int], 
                     first_bit: int, path_bits: Set[int]) -> int:
            """DFS to count valid words"""
            count = 0
            
            # If we have first letter and this is a word endpoint
            if first_bit in path_bits and node.count > 0:
                count += node.count
            
            # Explore children
            for bit, child in node.children.items():
                if bit in puzzle_bits:  # Only if bit is in puzzle
                    new_path_bits = path_bits | {bit}
                    count += dfs_count(child, puzzle_bits, first_bit, new_path_bits)
            
            return count
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = string_to_mask(puzzle)
            puzzle_bits = set(mask_to_sorted_bits(puzzle_mask))
            first_bit = ord(puzzle[0]) - ord('a')
            
            count = dfs_count(root, puzzle_bits, first_bit, set())
            result.append(count)
        
        return result


def test_basic_functionality():
    """Test basic puzzle word validation functionality"""
    print("=== Testing Puzzle Word Validation ===")
    
    validator = PuzzleWordValidator()
    
    test_cases = [
        {
            "words": ["aaaa", "asas", "able", "ability", "actt", "actor", "access"],
            "puzzles": ["aboveyz", "abrodyz", "abslute", "absoryz", "actresz", "gaswxyz"],
            "expected": [1, 1, 3, 2, 4, 0]
        },
        {
            "words": ["apple", "pleas", "please"],
            "puzzles": ["aelwxyz", "aelpxyz", "aelpsxy", "saelpxy", "xaelpsy"],
            "expected": [0, 1, 3, 2, 0]
        },
    ]
    
    approaches = [
        ("Brute Force", validator.find_num_words_brute_force),
        ("Bitmask", validator.find_num_words_bitmask),
        ("Trie Bitmask", validator.find_num_words_trie_bitmask),
        ("Frequency Filter", validator.find_num_words_frequency_filter),
        ("Subset Enumeration", validator.find_num_words_subset_enumeration),
        ("Advanced Trie", validator.find_num_words_advanced_trie),
    ]
    
    for i, test_case in enumerate(test_cases):
        words = test_case["words"]
        puzzles = test_case["puzzles"]
        expected = test_case["expected"]
        
        print(f"\nTest Case {i + 1}:")
        print(f"Words: {words}")
        print(f"Puzzles: {puzzles}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(words.copy(), puzzles.copy())
                status = "✓" if result == expected else "✗"
                print(f"  {approach_name:20}: {result} {status}")
            except Exception as e:
                print(f"  {approach_name:20}: Error - {e}")


def analyze_bit_patterns():
    """Analyze bit patterns in puzzle validation"""
    print("\n=== Bit Pattern Analysis ===")
    
    words = ["aaaa", "asas", "able", "ability", "actt", "actor", "access"]
    puzzles = ["aboveyz", "abrodyz", "abslute", "absoryz", "actresz", "gaswxyz"]
    
    def string_to_mask(s: str) -> int:
        mask = 0
        for c in s:
            mask |= (1 << (ord(c) - ord('a')))
        return mask
    
    def mask_to_string(mask: int) -> str:
        chars = []
        for i in range(26):
            if mask & (1 << i):
                chars.append(chr(ord('a') + i))
        return ''.join(chars)
    
    print("Word bitmasks:")
    word_masks = {}
    for word in words:
        mask = string_to_mask(word)
        word_masks[word] = mask
        print(f"  {word:8}: {mask:8} ({bin(mask)[2:]:>26}) -> {mask_to_string(mask)}")
    
    print(f"\nPuzzle analysis:")
    for puzzle in puzzles:
        puzzle_mask = string_to_mask(puzzle)
        first_bit = 1 << (ord(puzzle[0]) - ord('a'))
        
        print(f"\nPuzzle: {puzzle}")
        print(f"  Mask: {puzzle_mask:8} ({bin(puzzle_mask)[2:]:>26})")
        print(f"  First letter: {puzzle[0]} (bit {ord(puzzle[0]) - ord('a')})")
        print(f"  Valid words:")
        
        count = 0
        for word in words:
            word_mask = word_masks[word]
            is_subset = (word_mask & puzzle_mask) == word_mask
            has_first = bool(word_mask & first_bit)
            is_valid = is_subset and has_first
            
            if is_valid:
                count += 1
                print(f"    {word:8}: ✓ (subset: {is_subset}, has_first: {has_first})")
        
        print(f"  Total valid: {count}")


def demonstrate_subset_enumeration():
    """Demonstrate subset enumeration technique"""
    print("\n=== Subset Enumeration Demonstration ===")
    
    puzzle = "aboveyz"
    
    def string_to_mask(s: str) -> int:
        mask = 0
        for c in s:
            mask |= (1 << (ord(c) - ord('a')))
        return mask
    
    def mask_to_string(mask: int) -> str:
        chars = []
        for i in range(26):
            if mask & (1 << i):
                chars.append(chr(ord('a') + i))
        return ''.join(chars)
    
    puzzle_mask = string_to_mask(puzzle)
    first_bit = 1 << (ord(puzzle[0]) - ord('a'))
    
    print(f"Puzzle: {puzzle}")
    print(f"Puzzle mask: {puzzle_mask} -> {mask_to_string(puzzle_mask)}")
    print(f"First letter: {puzzle[0]} (bit {ord(puzzle[0]) - ord('a')})")
    
    print(f"\nAll subsets containing first letter:")
    
    subsets_with_first = []
    submask = puzzle_mask
    
    while submask > 0:
        if submask & first_bit:
            subset_str = mask_to_string(submask)
            subsets_with_first.append((submask, subset_str))
        submask = (submask - 1) & puzzle_mask
    
    # Sort by mask value for cleaner output
    subsets_with_first.sort()
    
    for i, (mask, subset_str) in enumerate(subsets_with_first):
        print(f"  {i+1:2}: {mask:3} -> {subset_str}")
    
    print(f"\nTotal subsets with first letter: {len(subsets_with_first)}")


def benchmark_performance():
    """Benchmark performance of different approaches"""
    print("\n=== Performance Benchmark ===")
    
    validator = PuzzleWordValidator()
    
    import random
    import string
    
    def generate_test_data(num_words: int, num_puzzles: int):
        """Generate test data"""
        words = []
        for _ in range(num_words):
            length = random.randint(1, 10)
            word = ''.join(random.choices(string.ascii_lowercase[:10], k=length))
            words.append(word)
        
        puzzles = []
        for _ in range(num_puzzles):
            length = 7  # Standard puzzle length
            puzzle = ''.join(random.choices(string.ascii_lowercase[:10], k=length))
            puzzles.append(puzzle)
        
        return words, puzzles
    
    test_sizes = [
        (100, 10),    # Small
        (500, 20),    # Medium
        (1000, 50),   # Large
    ]
    
    approaches = [
        ("Brute Force", validator.find_num_words_brute_force),
        ("Bitmask", validator.find_num_words_bitmask),
        ("Frequency Filter", validator.find_num_words_frequency_filter),
        ("Subset Enumeration", validator.find_num_words_subset_enumeration),
    ]
    
    print(f"{'Size':<15} {'Method':<20} {'Time(ms)':<12} {'Result Sum':<12}")
    print("-" * 65)
    
    for num_words, num_puzzles in test_sizes:
        words, puzzles = generate_test_data(num_words, num_puzzles)
        size_str = f"{num_words}w/{num_puzzles}p"
        
        for approach_name, approach_func in approaches:
            try:
                start_time = time.time()
                result = approach_func(words.copy(), puzzles.copy())
                end_time = time.time()
                
                elapsed_ms = (end_time - start_time) * 1000
                result_sum = sum(result)
                
                print(f"{size_str:<15} {approach_name:<20} {elapsed_ms:<12.2f} {result_sum:<12}")
                
            except Exception as e:
                print(f"{size_str:<15} {approach_name:<20} {'Error':<12} {str(e)[:12]:<12}")
        
        print()


def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques"""
    print("\n=== Optimization Techniques ===")
    
    # Technique 1: Character frequency analysis
    print("1. Character Frequency Analysis:")
    
    words = ["aaaa", "asas", "able", "ability", "actt", "actor", "access"]
    puzzles = ["aboveyz", "abrodyz", "abslute", "absoryz", "actresz", "gaswxyz"]
    
    # Analyze character frequencies
    word_chars = set()
    puzzle_chars = set()
    
    for word in words:
        word_chars.update(word)
    
    for puzzle in puzzles:
        puzzle_chars.update(puzzle)
    
    common_chars = word_chars & puzzle_chars
    word_only = word_chars - puzzle_chars
    puzzle_only = puzzle_chars - word_chars
    
    print(f"   Word characters: {sorted(word_chars)}")
    print(f"   Puzzle characters: {sorted(puzzle_chars)}")
    print(f"   Common characters: {sorted(common_chars)}")
    print(f"   Word-only characters: {sorted(word_only)}")
    print(f"   Puzzle-only characters: {sorted(puzzle_only)}")
    
    # Calculate potential savings
    filterable_words = sum(1 for word in words if not (set(word) - puzzle_chars))
    print(f"   Filterable words: {filterable_words}/{len(words)} ({filterable_words/len(words)*100:.1f}%)")
    
    # Technique 2: Bit manipulation optimizations
    print(f"\n2. Bit Manipulation Optimizations:")
    
    def count_bits(mask: int) -> int:
        """Count set bits in mask"""
        count = 0
        while mask:
            count += mask & 1
            mask >>= 1
        return count
    
    def string_to_mask(s: str) -> int:
        mask = 0
        for c in s:
            mask |= (1 << (ord(c) - ord('a')))
        return mask
    
    # Analyze bit density
    for word in words[:3]:  # Show first 3 words
        mask = string_to_mask(word)
        bit_count = count_bits(mask)
        print(f"   Word '{word}': {bit_count} unique characters (mask: {mask})")
    
    print(f"\n   Puzzle bit densities:")
    for puzzle in puzzles[:3]:  # Show first 3 puzzles
        mask = string_to_mask(puzzle)
        bit_count = count_bits(mask)
        total_subsets = 2 ** bit_count
        print(f"   Puzzle '{puzzle}': {bit_count} chars, {total_subsets} total subsets")
    
    # Technique 3: Early termination strategies
    print(f"\n3. Early Termination Strategies:")
    
    def count_with_early_termination(words: List[str], puzzle: str) -> Tuple[int, int]:
        """Count with early termination tracking"""
        puzzle_set = set(puzzle)
        first_letter = puzzle[0]
        count = 0
        checks = 0
        
        for word in words:
            checks += 1
            
            # Early check: if word is longer than puzzle, skip
            if len(set(word)) > len(puzzle_set):
                continue
            
            # Early check: first letter
            if first_letter not in word:
                continue
            
            # Full check
            if set(word).issubset(puzzle_set):
                count += 1
        
        return count, checks
    
    puzzle = puzzles[0]
    count, checks = count_with_early_termination(words, puzzle)
    print(f"   Puzzle '{puzzle}': {count} valid words, {checks} checks needed")


def analyze_complexity():
    """Analyze time and space complexity of different approaches"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force", 
         "Time: O(w * p * (|word| + |puzzle|))", 
         "Space: O(|puzzle|) for set operations"),
        
        ("Bitmask", 
         "Time: O(w + p * 2^|puzzle|)", 
         "Space: O(w) for word mask storage"),
        
        ("Trie + Bitmask", 
         "Time: O(w * 26 + p * trie_nodes)", 
         "Space: O(trie_size) depends on word diversity"),
        
        ("Frequency Filter", 
         "Time: O(w * |word| + filtered_w * p)", 
         "Space: O(w + unique_chars)"),
        
        ("Subset Enumeration", 
         "Time: O(w + p * 2^7)", 
         "Space: O(w + 2^7) with caching"),
        
        ("Advanced Trie", 
         "Time: O(w * unique_chars + p * trie_depth)", 
         "Space: O(trie_size) with optimized nodes"),
    ]
    
    print("Approach Complexity Analysis:")
    print(f"{'Method':<20} {'Time Complexity':<35} {'Space Complexity'}")
    print("-" * 80)
    
    for method, time_comp, space_comp in complexity_analysis:
        print(f"{method:<20} {time_comp:<35} {space_comp}")
    
    print(f"\nWhere:")
    print(f"  w = number of words")
    print(f"  p = number of puzzles") 
    print(f"  |word| = average word length")
    print(f"  |puzzle| = puzzle length (typically 7)")
    print(f"  2^7 = 128 (maximum subsets for 7-character puzzle)")
    
    print(f"\nRecommendations:")
    print(f"  • Use Bitmask approach for general cases (optimal for most inputs)")
    print(f"  • Use Frequency Filter when word set is much larger than puzzle character set")
    print(f"  • Use Advanced Trie when words have high character overlap")
    print(f"  • Avoid Brute Force for large inputs (quadratic complexity)")


if __name__ == "__main__":
    test_basic_functionality()
    analyze_bit_patterns()
    demonstrate_subset_enumeration()
    benchmark_performance()
    demonstrate_optimization_techniques()
    analyze_complexity()

"""
Number of Valid Words for Each Puzzle demonstrates advanced bit manipulation with tries:

Key Techniques:
1. Bitmask Representation - Convert strings to integer bitmasks for fast operations
2. Subset Enumeration - Efficiently iterate through all subsets using bit manipulation
3. Trie Optimization - Build tries based on character bitmasks for structured search
4. Frequency Filtering - Pre-filter data based on character frequency analysis
5. Early Termination - Skip unnecessary computations using heuristics

Bit Manipulation Optimizations:
- String to bitmask conversion for O(1) subset checking
- Subset enumeration using (submask - 1) & mask technique
- Bit counting and analysis for pruning search space
- First letter checking using bitwise AND operations

Real-world Applications:
- Word game validation and scoring systems
- Pattern matching in bioinformatics (DNA/protein sequences)
- Feature matching in machine learning with binary features
- Cryptographic applications with bit-level operations
- Database query optimization with bitmap indexes

The bitmask approach provides optimal performance for this problem,
reducing the complexity from O(w*p*length) to O(w + p*2^7).
"""
