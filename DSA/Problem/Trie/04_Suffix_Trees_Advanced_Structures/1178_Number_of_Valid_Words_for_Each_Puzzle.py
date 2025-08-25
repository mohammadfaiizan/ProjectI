"""
1178. Number of Valid Words for Each Puzzle - Multiple Approaches
Difficulty: Medium

With respect to a given puzzle string, a word is valid if both the following conditions are satisfied:
- word contains the first letter of puzzle.
- For each letter in word, that letter is also in puzzle.

Given an array of words and an array of puzzles, return an array answer, where answer[i] is the number of words in the given word list that are valid with respect to the puzzle puzzles[i].

LeetCode Problem: https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/

Example:
Input: words = ["aaaa","asas","able","ability","actt","actor","access"], puzzles = ["aboveyz","abrodyz","abslute","absoryz","actresz","gaswxyz"]
Output: [1,1,3,2,4,0]
"""

from typing import List, Dict, Set
from collections import defaultdict, Counter
import time

class Solution:
    
    def findNumOfValidWords1(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 1: Brute Force with Set Operations
        
        For each puzzle, check each word to see if it's valid.
        
        Time: O(W * P * max_word_length) where W=words, P=puzzles
        Space: O(max_word_length)
        """
        result = []
        
        for puzzle in puzzles:
            puzzle_set = set(puzzle)
            first_char = puzzle[0]
            count = 0
            
            for word in words:
                word_set = set(word)
                
                # Check if word contains first letter of puzzle
                if first_char in word_set:
                    # Check if all letters in word are in puzzle
                    if word_set.issubset(puzzle_set):
                        count += 1
            
            result.append(count)
        
        return result
    
    def findNumOfValidWords2(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 2: Bitmask with Precomputation
        
        Convert words and puzzles to bitmasks for efficient operations.
        
        Time: O(W + P * 2^7) where W=words, P=puzzles
        Space: O(W)
        """
        def char_to_bit(c: str) -> int:
            """Convert character to bit position"""
            return ord(c) - ord('a')
        
        def word_to_mask(word: str) -> int:
            """Convert word to bitmask"""
            mask = 0
            for char in word:
                mask |= 1 << char_to_bit(char)
            return mask
        
        # Precompute word masks and count occurrences
        word_mask_count = Counter()
        for word in words:
            mask = word_to_mask(word)
            word_mask_count[mask] += 1
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = word_to_mask(puzzle)
            first_bit = 1 << char_to_bit(puzzle[0])
            count = 0
            
            # Generate all submasks of puzzle
            submask = puzzle_mask
            while submask > 0:
                # Check if submask contains first character
                if submask & first_bit:
                    count += word_mask_count[submask]
                
                # Generate next submask
                submask = (submask - 1) & puzzle_mask
            
            result.append(count)
        
        return result
    
    def findNumOfValidWords3(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 3: Trie-based Solution
        
        Build trie of word bitmasks and traverse for each puzzle.
        
        Time: O(W * log W + P * 2^7)
        Space: O(W * log W)
        """
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.count = 0
        
        def char_to_bit(c: str) -> int:
            return ord(c) - ord('a')
        
        def word_to_mask(word: str) -> int:
            mask = 0
            for char in word:
                mask |= 1 << char_to_bit(char)
            return mask
        
        # Build trie of word bitmasks
        root = TrieNode()
        
        for word in words:
            mask = word_to_mask(word)
            
            # Insert mask into trie (binary representation)
            node = root
            for i in range(26):  # 26 bits for 26 letters
                bit = (mask >> i) & 1
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                node = node.children[bit]
            node.count += 1
        
        def count_valid_submasks(node: TrieNode, puzzle_mask: int, 
                               first_bit: int, pos: int, current_mask: int) -> int:
            """Count valid submasks using trie traversal"""
            if pos == 26:
                # Check if current mask contains first character
                if current_mask & first_bit:
                    return node.count
                return 0
            
            total = 0
            puzzle_bit = (puzzle_mask >> pos) & 1
            
            # Always try not including current bit
            if 0 in node.children:
                total += count_valid_submasks(node.children[0], puzzle_mask, 
                                            first_bit, pos + 1, current_mask)
            
            # Try including current bit if it's in puzzle
            if puzzle_bit and 1 in node.children:
                total += count_valid_submasks(node.children[1], puzzle_mask,
                                            first_bit, pos + 1, current_mask | (1 << pos))
            
            return total
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = word_to_mask(puzzle)
            first_bit = 1 << char_to_bit(puzzle[0])
            
            count = count_valid_submasks(root, puzzle_mask, first_bit, 0, 0)
            result.append(count)
        
        return result
    
    def findNumOfValidWords4(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 4: Optimized Bitmask with Prefiltering
        
        Filter words by first character and use optimized bitmask operations.
        
        Time: O(W + P * min(2^7, filtered_words))
        Space: O(W)
        """
        def char_to_bit(c: str) -> int:
            return ord(c) - ord('a')
        
        def word_to_mask(word: str) -> int:
            mask = 0
            for char in word:
                mask |= 1 << char_to_bit(char)
            return mask
        
        # Group words by their first character
        words_by_first_char = defaultdict(list)
        
        for word in words:
            if word:  # Ensure word is not empty
                mask = word_to_mask(word)
                first_char = word[0]
                words_by_first_char[first_char].append(mask)
        
        # Count occurrences of each mask for each first character
        mask_count_by_first_char = {}
        for first_char, masks in words_by_first_char.items():
            mask_count_by_first_char[first_char] = Counter(masks)
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = word_to_mask(puzzle)
            first_char = puzzle[0]
            count = 0
            
            # Only consider words that start with puzzle's first character
            if first_char in mask_count_by_first_char:
                mask_counts = mask_count_by_first_char[first_char]
                
                # Check all submasks of puzzle
                submask = puzzle_mask
                while submask > 0:
                    if submask in mask_counts:
                        count += mask_counts[submask]
                    submask = (submask - 1) & puzzle_mask
            
            result.append(count)
        
        return result
    
    def findNumOfValidWords5(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 5: Advanced Trie with Pruning
        
        Build character-frequency trie with pruning optimizations.
        
        Time: O(W * avg_word_length + P * 2^puzzle_unique_chars)
        Space: O(unique_character_combinations)
        """
        class AdvancedTrieNode:
            def __init__(self):
                self.children = {}
                self.word_count = 0
                self.total_words = 0  # Total words in subtree
        
        def build_signature(word: str) -> tuple:
            """Build sorted signature of unique characters"""
            return tuple(sorted(set(word)))
        
        # Build trie of character signatures
        root = AdvancedTrieNode()
        
        for word in words:
            signature = build_signature(word)
            node = root
            node.total_words += 1
            
            for char in signature:
                if char not in node.children:
                    node.children[char] = AdvancedTrieNode()
                node = node.children[char]
                node.total_words += 1
            
            node.word_count += 1
        
        def count_valid_words(node: AdvancedTrieNode, puzzle_chars: Set[str],
                             first_char: str, used_first: bool) -> int:
            """Count valid words using advanced trie traversal"""
            if not puzzle_chars:
                # No more characters to use
                return node.word_count if used_first else 0
            
            total = 0
            
            # Option 1: Include words that end here (if used first char)
            if used_first:
                total += node.word_count
            
            # Option 2: Continue with remaining characters
            for char in list(puzzle_chars):
                if char in node.children:
                    child = node.children[char]
                    new_puzzle_chars = puzzle_chars - {char}
                    new_used_first = used_first or (char == first_char)
                    
                    # Pruning: skip if impossible to use first char
                    if not new_used_first and first_char not in new_puzzle_chars:
                        continue
                    
                    total += count_valid_words(child, new_puzzle_chars, 
                                             first_char, new_used_first)
            
            return total
        
        result = []
        
        for puzzle in puzzles:
            puzzle_chars = set(puzzle)
            first_char = puzzle[0]
            
            count = count_valid_words(root, puzzle_chars, first_char, False)
            result.append(count)
        
        return result
    
    def findNumOfValidWords6(self, words: List[str], puzzles: List[str]) -> List[int]:
        """
        Approach 6: Dynamic Programming with Memoization
        
        Use DP to avoid recomputing submask counts.
        
        Time: O(W + P * 2^unique_chars)
        Space: O(2^max_unique_chars)
        """
        def char_to_bit(c: str) -> int:
            return ord(c) - ord('a')
        
        def word_to_mask(word: str) -> int:
            mask = 0
            for char in word:
                mask |= 1 << char_to_bit(char)
            return mask
        
        # Precompute word mask counts
        word_mask_count = Counter()
        for word in words:
            mask = word_to_mask(word)
            word_mask_count[mask] += 1
        
        # Memoization for submask generation
        submask_cache = {}
        
        def get_submasks(mask: int) -> List[int]:
            """Get all submasks of given mask with memoization"""
            if mask in submask_cache:
                return submask_cache[mask]
            
            submasks = []
            submask = mask
            while submask > 0:
                submasks.append(submask)
                submask = (submask - 1) & mask
            
            submask_cache[mask] = submasks
            return submasks
        
        result = []
        
        for puzzle in puzzles:
            puzzle_mask = word_to_mask(puzzle)
            first_bit = 1 << char_to_bit(puzzle[0])
            count = 0
            
            # Get all submasks
            submasks = get_submasks(puzzle_mask)
            
            for submask in submasks:
                if submask & first_bit:  # Contains first character
                    count += word_mask_count[submask]
            
            result.append(count)
        
        return result


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode example
        (["aaaa","asas","able","ability","actt","actor","access"],
         ["aboveyz","abrodyz","abslute","absoryz","actresz","gaswxyz"],
         [1,1,3,2,4,0]),
        
        # Simple cases
        (["apple", "pleas", "please"], ["aelp"], [2]),
        (["word", "good", "wood", "od"], ["oodw"], [2]),
        
        # Edge cases
        ([], ["abc"], [0]),
        (["abc"], [], []),
        (["a"], ["a"], [1]),
        (["ab"], ["ba"], [1]),
    ]
    
    approaches = [
        ("Brute Force", solution.findNumOfValidWords1),
        ("Bitmask", solution.findNumOfValidWords2),
        ("Trie-based", solution.findNumOfValidWords3),
        ("Optimized Bitmask", solution.findNumOfValidWords4),
        ("Advanced Trie", solution.findNumOfValidWords5),
        ("DP Memoization", solution.findNumOfValidWords6),
    ]
    
    for i, (words, puzzles, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Words: {words}")
        print(f"Puzzles: {puzzles}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(words[:], puzzles[:])
                status = "✓" if result == expected else "✗"
                print(f"  {name:18}: {result} {status}")
            except Exception as e:
                print(f"  {name:18}: Error - {e}")


def demonstrate_bitmask_operations():
    """Demonstrate bitmask operations"""
    print("\n=== Bitmask Operations Demo ===")
    
    def char_to_bit(c: str) -> int:
        return ord(c) - ord('a')
    
    def word_to_mask(word: str) -> int:
        mask = 0
        for char in word:
            mask |= 1 << char_to_bit(char)
        return mask
    
    def mask_to_chars(mask: int) -> str:
        chars = []
        for i in range(26):
            if mask & (1 << i):
                chars.append(chr(ord('a') + i))
        return ''.join(chars)
    
    # Example words and puzzles
    words = ["able", "actor", "access"]
    puzzle = "aboveyz"
    
    print(f"Puzzle: '{puzzle}'")
    puzzle_mask = word_to_mask(puzzle)
    print(f"Puzzle mask: {bin(puzzle_mask)} -> characters: {mask_to_chars(puzzle_mask)}")
    
    first_char = puzzle[0]
    first_bit = 1 << char_to_bit(first_char)
    print(f"First character '{first_char}' bit: {bin(first_bit)}")
    
    print(f"\nWord analysis:")
    for word in words:
        word_mask = word_to_mask(word)
        word_chars = mask_to_chars(word_mask)
        
        # Check conditions
        contains_first = bool(word_mask & first_bit)
        is_subset = (word_mask & puzzle_mask) == word_mask
        is_valid = contains_first and is_subset
        
        print(f"  '{word}':")
        print(f"    Mask: {bin(word_mask)} -> characters: {word_chars}")
        print(f"    Contains first char: {contains_first}")
        print(f"    Is subset of puzzle: {is_subset}")
        print(f"    Valid: {is_valid}")
    
    # Demonstrate submask generation
    print(f"\nSubmask generation for puzzle '{puzzle}':")
    submasks = []
    submask = puzzle_mask
    count = 0
    
    while submask > 0 and count < 10:  # Limit output
        chars = mask_to_chars(submask)
        contains_first = bool(submask & first_bit)
        print(f"  Submask {bin(submask):>10} -> '{chars}' (has first: {contains_first})")
        submasks.append(submask)
        submask = (submask - 1) & puzzle_mask
        count += 1
    
    if count == 10:
        print(f"  ... (showing first 10 submasks)")


def demonstrate_trie_approach():
    """Demonstrate trie-based approach"""
    print("\n=== Trie Approach Demo ===")
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.count = 0
    
    def char_to_bit(c: str) -> int:
        return ord(c) - ord('a')
    
    def word_to_mask(word: str) -> int:
        mask = 0
        for char in word:
            mask |= 1 << char_to_bit(char)
        return mask
    
    # Build trie
    words = ["able", "actor", "access", "cat"]
    root = TrieNode()
    
    print("Building trie from word masks:")
    
    for word in words:
        mask = word_to_mask(word)
        print(f"  '{word}' -> mask: {bin(mask)}")
        
        # Insert mask into trie
        node = root
        for i in range(26):
            bit = (mask >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
        node.count += 1
    
    # Demonstrate trie traversal
    puzzle = "actor"
    puzzle_mask = word_to_mask(puzzle)
    first_bit = 1 << char_to_bit(puzzle[0])
    
    print(f"\nTraversing trie for puzzle '{puzzle}':")
    print(f"Puzzle mask: {bin(puzzle_mask)}")
    print(f"First bit: {bin(first_bit)}")
    
    def traverse_trie(node: TrieNode, pos: int, current_mask: int, indent: str = ""):
        """Demonstrate trie traversal"""
        if pos == 26:
            if node.count > 0:
                contains_first = bool(current_mask & first_bit)
                print(f"{indent}Leaf: mask={bin(current_mask)}, count={node.count}, valid={contains_first}")
            return
        
        puzzle_bit = (puzzle_mask >> pos) & 1
        
        # Try bit 0
        if 0 in node.children:
            print(f"{indent}Pos {pos}: choosing 0")
            traverse_trie(node.children[0], pos + 1, current_mask, indent + "  ")
        
        # Try bit 1 if allowed by puzzle
        if puzzle_bit and 1 in node.children:
            print(f"{indent}Pos {pos}: choosing 1 (allowed by puzzle)")
            traverse_trie(node.children[1], pos + 1, current_mask | (1 << pos), indent + "  ")
    
    # Show first few levels of traversal
    print("Trie traversal (first 3 levels):")
    traverse_trie(root, 0, 0)


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_words(count: int, avg_length: int) -> List[str]:
        words = []
        for _ in range(count):
            length = max(1, avg_length + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase[:10], k=length))
            words.append(word)
        return words
    
    def generate_puzzles(count: int, length: int = 7) -> List[str]:
        puzzles = []
        for _ in range(count):
            puzzle = ''.join(random.sample(string.ascii_lowercase[:15], length))
            puzzles.append(puzzle)
        return puzzles
    
    test_scenarios = [
        ("Small", generate_words(100, 5), generate_puzzles(10)),
        ("Medium", generate_words(500, 6), generate_puzzles(20)),
        ("Large", generate_words(1000, 7), generate_puzzles(50)),
    ]
    
    approaches = [
        ("Brute Force", solution.findNumOfValidWords1),
        ("Bitmask", solution.findNumOfValidWords2),
        ("Optimized Bitmask", solution.findNumOfValidWords4),
        ("DP Memoization", solution.findNumOfValidWords6),
    ]
    
    for scenario_name, words, puzzles in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Words: {len(words)}, Puzzles: {len(puzzles)}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            try:
                result = method(words, puzzles)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                print(f"  {approach_name:18}: {execution_time:6.2f}ms")
            
            except Exception as e:
                print(f"  {approach_name:18}: Error - {str(e)[:30]}")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    print("1. Character Frequency Analysis:")
    
    words = ["able", "actor", "access", "cat", "application", "apple"]
    
    # Analyze character frequencies
    char_freq = {}
    for word in words:
        for char in set(word):
            char_freq[char] = char_freq.get(char, 0) + 1
    
    print("   Character frequencies in words:")
    for char, freq in sorted(char_freq.items()):
        print(f"     '{char}': {freq} words")
    
    print("\n2. Bitmask Optimization:")
    
    # Show bitmask compression
    def word_to_mask(word: str) -> int:
        mask = 0
        for char in word:
            mask |= 1 << (ord(char) - ord('a'))
        return mask
    
    print("   Word to bitmask conversion:")
    for word in words:
        mask = word_to_mask(word)
        unique_chars = bin(mask).count('1')
        print(f"     '{word:12}' -> {mask:6} ({unique_chars} unique chars)")
    
    print("\n3. Pruning Strategies:")
    
    puzzle = "abcdefg"
    puzzle_mask = word_to_mask(puzzle)
    first_char = puzzle[0]
    first_bit = 1 << (ord(first_char) - ord('a'))
    
    print(f"   Puzzle: '{puzzle}' (first char: '{first_char}')")
    
    valid_words = []
    pruned_words = []
    
    for word in words:
        word_mask = word_to_mask(word)
        
        # Check pruning conditions
        contains_first = bool(word_mask & first_bit)
        is_subset = (word_mask & puzzle_mask) == word_mask
        
        if contains_first and is_subset:
            valid_words.append(word)
        else:
            pruned_words.append((word, not contains_first, not is_subset))
    
    print(f"   Valid words: {valid_words}")
    print(f"   Pruned words:")
    for word, missing_first, not_subset in pruned_words:
        reasons = []
        if missing_first:
            reasons.append("missing first char")
        if not_subset:
            reasons.append("not subset")
        print(f"     '{word}': {', '.join(reasons)}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty inputs
        ([], ["abc"], "Empty words"),
        (["abc"], [], "Empty puzzles"),
        ([], [], "Both empty"),
        
        # Single character
        (["a"], ["a"], "Single char match"),
        (["a"], ["b"], "Single char no match"),
        
        # Repeated characters
        (["aaa"], ["abc"], "Repeated chars in word"),
        (["abc"], ["aaa"], "Repeated chars in puzzle"),
        
        # All same characters
        (["aaaa", "aa", "aaa"], ["abcd"], "All a's in words"),
        
        # Maximum length
        (["a" * 50], ["abcdefg"], "Very long word"),
        
        # All unique characters
        (["abcdefghijklmnopqrstuvwxyz"], ["abcdefg"], "All alphabet"),
    ]
    
    for words, puzzles, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Words: {words}")
        print(f"  Puzzles: {puzzles}")
        
        try:
            result = solution.findNumOfValidWords2(words, puzzles)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Brute Force Set Operations",
         "Time: O(W * P * max_word_length)",
         "Space: O(max_word_length)"),
        
        ("Bitmask with Submask Generation",
         "Time: O(W + P * 2^7) where 7 is max unique chars",
         "Space: O(W) for mask counting"),
        
        ("Trie-based Solution",
         "Time: O(W * 26 + P * 2^7 * 26)",
         "Space: O(W * 26) for trie storage"),
        
        ("Optimized Bitmask with Prefiltering",
         "Time: O(W + P * min(2^7, filtered_words))",
         "Space: O(W) grouped by first character"),
        
        ("Advanced Trie with Pruning",
         "Time: O(W * unique_chars + P * 2^puzzle_chars)",
         "Space: O(unique_character_combinations)"),
        
        ("DP with Memoization",
         "Time: O(W + P * 2^max_unique_chars)",
         "Space: O(2^max_unique_chars) for memoization"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nKey Observations:")
    print(f"  • Puzzle length is fixed at 7, so 2^7 = 128 submasks maximum")
    print(f"  • Bitmask approaches are most efficient due to small alphabet")
    print(f"  • Prefiltering by first character significantly reduces search space")
    print(f"  • Memoization helps when puzzles share many characters")
    
    print(f"\nOptimization Strategies:")
    print(f"  • Use bitmasks for O(1) subset checking")
    print(f"  • Precompute word masks to avoid recomputation")
    print(f"  • Group words by first character for faster filtering")
    print(f"  • Cache submask generation for repeated puzzles")
    print(f"  • Use bit manipulation tricks for efficient submask iteration")
    
    print(f"\nRecommendations:")
    print(f"  • Use Optimized Bitmask for best practical performance")
    print(f"  • Use DP Memoization when puzzles have overlapping characters")
    print(f"  • Use Trie approach for educational understanding")
    print(f"  • Avoid Brute Force for large inputs")


if __name__ == "__main__":
    test_basic_functionality()
    demonstrate_bitmask_operations()
    demonstrate_trie_approach()
    benchmark_approaches()
    demonstrate_optimization_techniques()
    test_edge_cases()
    analyze_complexity()

"""
1178. Number of Valid Words for Each Puzzle demonstrates advanced optimization techniques:

1. Brute Force - Direct set operations for validation checking
2. Bitmask with Submask Generation - Efficient bit manipulation for subset checking
3. Trie-based Solution - Build trie of word bitmasks for structured search
4. Optimized Bitmask - Prefilter by first character and use optimized operations
5. Advanced Trie with Pruning - Character-frequency trie with intelligent pruning
6. DP with Memoization - Cache submask computations for repeated patterns

Key optimization techniques:
- Bitmask representation for O(1) subset operations
- Submask generation using bit manipulation tricks
- Prefiltering by first character constraint
- Trie structures for organized search
- Memoization for avoiding repeated computations

The problem showcases how bit manipulation and advanced data structures
can dramatically improve performance for constraint satisfaction problems.
"""
