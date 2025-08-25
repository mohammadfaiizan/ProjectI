"""
Combinatorics String Counting - Multiple Approaches
Difficulty: Hard

Combinatorial problems involving string counting and trie structures
for competitive programming.

Problems:
1. Distinct Subsequence Counting
2. String Permutation Analysis
3. Palindromic Substring Counting
4. Lexicographic Ordering
5. String Generation with Constraints
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import math

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0
        self.depth = 0

class CombinatoricsStringCounting:
    
    def __init__(self):
        self.MOD = 10**9 + 7
        self.root = TrieNode()
    
    def count_distinct_subsequences(self, s: str) -> int:
        """
        Count distinct subsequences using trie
        Time: O(n * 2^n) worst case, optimized with trie
        Space: O(unique_subsequences)
        """
        n = len(s)
        trie_root = TrieNode()
        
        # Generate all subsequences and store in trie
        def generate_subsequences(index: int, current: str) -> None:
            if index == n:
                if current:  # Non-empty subsequence
                    node = trie_root
                    for char in current:
                        if char not in node.children:
                            node.children[char] = TrieNode()
                        node = node.children[char]
                    node.is_end = True
                return
            
            # Include current character
            generate_subsequences(index + 1, current + s[index])
            # Exclude current character
            generate_subsequences(index + 1, current)
        
        def count_trie_words(node: TrieNode) -> int:
            count = 1 if node.is_end else 0
            for child in node.children.values():
                count += count_trie_words(child)
            return count
        
        generate_subsequences(0, "")
        return count_trie_words(trie_root)
    
    def analyze_string_permutations(self, s: str) -> Dict[str, int]:
        """
        Analyze permutations and count patterns
        Time: O(n! * n) for generating all permutations
        Space: O(n! * n)
        """
        from itertools import permutations
        
        # Generate all unique permutations
        unique_perms = set([''.join(p) for p in permutations(s)])
        
        # Build trie of permutations
        perm_trie = TrieNode()
        
        for perm in unique_perms:
            node = perm_trie
            for char in perm:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1
            node.is_end = True
        
        # Analyze patterns
        analysis = {
            'total_permutations': len(unique_perms),
            'trie_nodes': self._count_trie_nodes(perm_trie),
            'avg_depth': self._calculate_avg_depth(perm_trie),
            'branching_factor': self._calculate_branching_factor(perm_trie)
        }
        
        return analysis
    
    def count_palindromic_substrings(self, s: str) -> Dict[str, int]:
        """
        Count palindromic substrings using trie
        Time: O(n^2)
        Space: O(palindromes)
        """
        n = len(s)
        palindrome_trie = TrieNode()
        palindrome_counts = defaultdict(int)
        
        # Check all substrings for palindromes
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                
                if self._is_palindrome(substring):
                    # Add to trie
                    node = palindrome_trie
                    for char in substring:
                        if char not in node.children:
                            node.children[char] = TrieNode()
                        node = node.children[char]
                    node.is_end = True
                    node.count += 1
                    
                    palindrome_counts[substring] += 1
        
        return {
            'palindrome_count': len(palindrome_counts),
            'total_occurrences': sum(palindrome_counts.values()),
            'unique_palindromes': list(palindrome_counts.keys()),
            'trie_size': self._count_trie_nodes(palindrome_trie)
        }
    
    def lexicographic_order_analysis(self, strings: List[str]) -> Dict[str, any]:
        """
        Analyze lexicographic ordering using trie
        Time: O(sum(lengths))
        Space: O(trie_size)
        """
        # Build trie
        lex_trie = TrieNode()
        
        for string in strings:
            node = lex_trie
            for char in string:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        # Get lexicographic order
        lex_order = []
        
        def dfs_lexicographic(node: TrieNode, current: str) -> None:
            if node.is_end:
                lex_order.append(current)
            
            # Visit children in sorted order
            for char in sorted(node.children.keys()):
                dfs_lexicographic(node.children[char], current + char)
        
        dfs_lexicographic(lex_trie, "")
        
        # Calculate statistics
        analysis = {
            'lexicographic_order': lex_order,
            'original_order': strings,
            'is_already_sorted': lex_order == strings,
            'inversions': self._count_inversions(strings, lex_order),
            'common_prefixes': self._analyze_common_prefixes(lex_trie)
        }
        
        return analysis
    
    def generate_strings_with_constraints(self, length: int, alphabet: str, 
                                        forbidden_patterns: List[str]) -> List[str]:
        """
        Generate strings with given constraints
        Time: O(|alphabet|^length * pattern_checks)
        Space: O(result_size)
        """
        # Build forbidden pattern trie
        forbidden_trie = TrieNode()
        
        for pattern in forbidden_patterns:
            node = forbidden_trie
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
        
        valid_strings = []
        
        def generate(current: str, remaining: int) -> None:
            if remaining == 0:
                valid_strings.append(current)
                return
            
            for char in alphabet:
                new_string = current + char
                
                # Check if new string contains any forbidden pattern
                if not self._contains_forbidden_pattern(new_string, forbidden_trie):
                    generate(new_string, remaining - 1)
        
        generate("", length)
        return valid_strings
    
    def count_strings_with_property(self, max_length: int, alphabet: str, 
                                   property_func) -> Dict[int, int]:
        """
        Count strings of each length with given property
        Time: O(|alphabet|^max_length)
        Space: O(|alphabet|^max_length)
        """
        counts = defaultdict(int)
        
        def generate_and_count(current: str, remaining: int) -> None:
            if property_func(current):
                counts[len(current)] += 1
            
            if remaining > 0:
                for char in alphabet:
                    generate_and_count(current + char, remaining - 1)
        
        generate_and_count("", max_length)
        return dict(counts)
    
    def _is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome"""
        return s == s[::-1]
    
    def _count_trie_nodes(self, node: TrieNode) -> int:
        """Count total nodes in trie"""
        count = 1
        for child in node.children.values():
            count += self._count_trie_nodes(child)
        return count
    
    def _calculate_avg_depth(self, node: TrieNode, depth: int = 0) -> float:
        """Calculate average depth of words in trie"""
        total_depth = 0
        word_count = 0
        
        def dfs(n: TrieNode, d: int) -> Tuple[int, int]:
            nonlocal total_depth, word_count
            if n.is_end:
                total_depth += d
                word_count += 1
            
            for child in n.children.values():
                dfs(child, d + 1)
        
        dfs(node, depth)
        return total_depth / word_count if word_count > 0 else 0
    
    def _calculate_branching_factor(self, node: TrieNode) -> float:
        """Calculate average branching factor"""
        total_children = 0
        node_count = 0
        
        def dfs(n: TrieNode) -> None:
            nonlocal total_children, node_count
            node_count += 1
            total_children += len(n.children)
            
            for child in n.children.values():
                dfs(child)
        
        dfs(node)
        return total_children / node_count if node_count > 0 else 0
    
    def _count_inversions(self, original: List[str], sorted_list: List[str]) -> int:
        """Count inversions between two orderings"""
        position_map = {string: i for i, string in enumerate(sorted_list)}
        inversions = 0
        
        for i in range(len(original)):
            for j in range(i + 1, len(original)):
                if position_map[original[i]] > position_map[original[j]]:
                    inversions += 1
        
        return inversions
    
    def _analyze_common_prefixes(self, node: TrieNode) -> Dict[str, int]:
        """Analyze common prefixes in trie"""
        prefixes = defaultdict(int)
        
        def dfs(n: TrieNode, prefix: str) -> None:
            if len(n.children) > 1:  # Node with multiple children = common prefix
                prefixes[prefix] = len(n.children)
            
            for char, child in n.children.items():
                dfs(child, prefix + char)
        
        dfs(node, "")
        return dict(prefixes)
    
    def _contains_forbidden_pattern(self, string: str, forbidden_trie: TrieNode) -> bool:
        """Check if string contains any forbidden pattern"""
        for i in range(len(string)):
            node = forbidden_trie
            for j in range(i, len(string)):
                char = string[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end:
                    return True
        return False


def test_subsequence_counting():
    """Test distinct subsequence counting"""
    print("=== Testing Distinct Subsequence Counting ===")
    
    counter = CombinatoricsStringCounting()
    
    test_strings = ["abc", "aab", "abcd"]
    
    for s in test_strings:
        count = counter.count_distinct_subsequences(s)
        print(f"String '{s}': {count} distinct subsequences")

def test_permutation_analysis():
    """Test string permutation analysis"""
    print("\n=== Testing Permutation Analysis ===")
    
    counter = CombinatoricsStringCounting()
    
    test_string = "abc"
    print(f"Analyzing permutations of '{test_string}'")
    
    analysis = counter.analyze_string_permutations(test_string)
    
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

def test_palindrome_counting():
    """Test palindromic substring counting"""
    print("\n=== Testing Palindromic Substring Counting ===")
    
    counter = CombinatoricsStringCounting()
    
    test_strings = ["abcba", "aabaa", "abcd"]
    
    for s in test_strings:
        print(f"\nString '{s}':")
        result = counter.count_palindromic_substrings(s)
        
        print(f"  Palindrome count: {result['palindrome_count']}")
        print(f"  Total occurrences: {result['total_occurrences']}")
        print(f"  Unique palindromes: {result['unique_palindromes']}")

def test_lexicographic_analysis():
    """Test lexicographic order analysis"""
    print("\n=== Testing Lexicographic Analysis ===")
    
    counter = CombinatoricsStringCounting()
    
    strings = ["cat", "dog", "apple", "banana"]
    print(f"Original strings: {strings}")
    
    analysis = counter.lexicographic_order_analysis(strings)
    
    print(f"Lexicographic order: {analysis['lexicographic_order']}")
    print(f"Already sorted: {analysis['is_already_sorted']}")
    print(f"Inversions: {analysis['inversions']}")
    print(f"Common prefixes: {analysis['common_prefixes']}")

def test_constraint_generation():
    """Test string generation with constraints"""
    print("\n=== Testing String Generation with Constraints ===")
    
    counter = CombinatoricsStringCounting()
    
    length = 3
    alphabet = "ab"
    forbidden_patterns = ["aa", "bb"]
    
    print(f"Length: {length}")
    print(f"Alphabet: '{alphabet}'")
    print(f"Forbidden patterns: {forbidden_patterns}")
    
    valid_strings = counter.generate_strings_with_constraints(length, alphabet, forbidden_patterns)
    
    print(f"Valid strings: {valid_strings}")
    print(f"Count: {len(valid_strings)}")

if __name__ == "__main__":
    test_subsequence_counting()
    test_permutation_analysis()
    test_palindrome_counting()
    test_lexicographic_analysis()
    test_constraint_generation()
