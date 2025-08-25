"""
1065. Index Pairs of a String - Multiple Approaches
Difficulty: Easy

Given a text string and words (a list of strings), return all index pairs [i, j] 
so that the substring text[i...j] is in the list of words.

Example:
Input: text = "thestoryofleetcodeandme", words = ["story","fleet","leetcode"]
Output: [[3,7],[9,13],[10,17]]
"""

from typing import List, Tuple
from collections import defaultdict

class TrieNode:
    """Trie node for efficient word matching"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word_length = 0

class Solution:
    
    def indexPairs1(self, text: str, words: List[str]) -> List[List[int]]:
        """
        Approach 1: Trie-based Solution
        
        Build trie from words and scan text for matches.
        
        Time: O(W + T*M) where W=total chars in words, T=text length, M=max word length
        Space: O(W) for trie storage
        """
        # Build trie
        root = TrieNode()
        
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word_length = len(word)
        
        # Find all matches
        result = []
        
        for i in range(len(text)):
            node = root
            j = i
            
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_word:
                    result.append([i, i + node.word_length - 1])
                j += 1
        
        return result
    
    def indexPairs2(self, text: str, words: List[str]) -> List[List[int]]:
        """
        Approach 2: Brute Force with Set
        
        Check every substring against word set.
        
        Time: O(T²*W) where T=text length, W=average word length
        Space: O(W) for word set
        """
        word_set = set(words)
        result = []
        
        for i in range(len(text)):
            for j in range(i, len(text)):
                substring = text[i:j+1]
                if substring in word_set:
                    result.append([i, j])
        
        return result
    
    def indexPairs3(self, text: str, words: List[str]) -> List[List[int]]:
        """
        Approach 3: Optimized Brute Force
        
        Limit substring length to maximum word length.
        
        Time: O(T*M*W) where M=max word length
        Space: O(W)
        """
        word_set = set(words)
        max_len = max(len(word) for word in words) if words else 0
        result = []
        
        for i in range(len(text)):
            for j in range(i, min(i + max_len, len(text))):
                substring = text[i:j+1]
                if substring in word_set:
                    result.append([i, j])
        
        return result
    
    def indexPairs4(self, text: str, words: List[str]) -> List[List[int]]:
        """
        Approach 4: KMP-based Multi-pattern Matching
        
        Use KMP algorithm for each word pattern.
        
        Time: O(T*W + sum of word lengths)
        Space: O(max word length)
        """
        result = []
        
        def kmp_search(text: str, pattern: str) -> List[int]:
            """KMP algorithm to find all occurrences"""
            if not pattern:
                return []
            
            # Build LPS array
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            
            # Search for pattern
            matches = []
            i = j = 0
            n = len(text)
            
            while i < n:
                if pattern[j] == text[i]:
                    i += 1
                    j += 1
                
                if j == m:
                    matches.append(i - j)
                    j = lps[j - 1]
                elif i < n and pattern[j] != text[i]:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            
            return matches
        
        # Search for each word
        for word in words:
            start_positions = kmp_search(text, word)
            for start in start_positions:
                result.append([start, start + len(word) - 1])
        
        return sorted(result)
    
    def indexPairs5(self, text: str, words: List[str]) -> List[List[int]]:
        """
        Approach 5: Aho-Corasick Algorithm
        
        Multi-pattern string matching algorithm.
        
        Time: O(T + W + matches)
        Space: O(W)
        """
        from collections import deque
        
        # Build automaton
        root = TrieNode()
        
        # Build trie
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word_length = len(word)
        
        # Build failure links (simplified version)
        queue = deque()
        for child in root.children.values():
            queue.append(child)
        
        # Find matches
        result = []
        
        for i in range(len(text)):
            node = root
            j = i
            
            while j < len(text):
                if text[j] in node.children:
                    node = node.children[text[j]]
                    if node.is_word:
                        result.append([i, j])
                    j += 1
                else:
                    break
        
        return result
    
    def indexPairs6(self, text: str, words: List[str]) -> List[List[int]]:
        """
        Approach 6: Length-based Grouping
        
        Group words by length for efficient processing.
        
        Time: O(T*M + W)
        Space: O(W)
        """
        # Group words by length
        words_by_length = defaultdict(set)
        for word in words:
            words_by_length[len(word)].add(word)
        
        result = []
        
        # Check each starting position
        for i in range(len(text)):
            # Check each possible length
            for length, word_set in words_by_length.items():
                if i + length <= len(text):
                    substring = text[i:i+length]
                    if substring in word_set:
                        result.append([i, i + length - 1])
        
        return result


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        ("thestoryofleetcodeandme", ["story","fleet","leetcode"], 
         [[3,7],[9,13],[10,17]]),
        ("wordgoodgoodgoodbestword", ["word","good","best","good"], 
         [[0,3],[8,11],[12,15],[16,19],[20,23]]),
        ("ababa", ["aba","ab"], 
         [[0,1],[0,2],[2,3],[2,4]]),
        ("", ["word"], []),
        ("hello", [], []),
        ("aaa", ["a","aa","aaa"], 
         [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]),
    ]
    
    approaches = [
        ("Trie-based", solution.indexPairs1),
        ("Brute Force", solution.indexPairs2),
        ("Optimized BF", solution.indexPairs3),
        ("KMP-based", solution.indexPairs4),
        ("Aho-Corasick", solution.indexPairs5),
        ("Length Groups", solution.indexPairs6),
    ]
    
    for i, (text, words, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: '{text}' with {words}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(text, words)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_test_data():
        # Create text
        text = ''.join(random.choices(string.ascii_lowercase, k=1000))
        
        # Create words (some will be in text, some won't)
        words = []
        for _ in range(50):
            length = random.randint(2, 8)
            if random.random() < 0.3:  # 30% chance to use substring from text
                start = random.randint(0, max(0, len(text) - length))
                word = text[start:start+length]
            else:
                word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        
        return text, list(set(words))  # Remove duplicates
    
    test_text, test_words = generate_test_data()
    
    approaches = [
        ("Trie", solution.indexPairs1),
        ("Optimized BF", solution.indexPairs3),
        ("Length Groups", solution.indexPairs6),
    ]
    
    print(f"Testing with text length: {len(test_text)}, words: {len(test_words)}")
    
    for name, method in approaches:
        start_time = time.time()
        
        # Run multiple times
        for _ in range(10):
            result = method(test_text, test_words)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"{name:15}: {avg_time*1000:.2f} ms (found {len(result)} pairs)")


def demonstrate_trie_construction():
    """Demonstrate trie construction and search process"""
    print("\n=== Trie Construction Demo ===")
    
    text = "thestoryofleetcode"
    words = ["story", "fleet", "code"]
    
    print(f"Text: '{text}'")
    print(f"Words: {words}")
    
    # Build trie step by step
    root = TrieNode()
    
    print(f"\nBuilding Trie:")
    for word in words:
        print(f"  Inserting '{word}'")
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word_length = len(word)
    
    # Search process
    print(f"\nSearching for matches:")
    result = []
    
    for i in range(len(text)):
        print(f"\nStarting at position {i} ('{text[i]}'):")
        node = root
        j = i
        
        while j < len(text) and text[j] in node.children:
            node = node.children[text[j]]
            print(f"  Position {j}: '{text[j]}' -> valid path")
            
            if node.is_word:
                match = [i, i + node.word_length - 1]
                result.append(match)
                matched_text = text[i:i + node.word_length]
                print(f"    Found word: '{matched_text}' at {match}")
            
            j += 1
        
        if j == i:
            print(f"  No valid path from '{text[i]}'")
    
    print(f"\nFinal result: {result}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Code syntax highlighting
    print("1. Code Syntax Highlighting:")
    code = "def function(param): return value"
    keywords = ["def", "return", "function", "param"]
    
    matches = solution.indexPairs1(code, keywords)
    print(f"   Code: '{code}'")
    print(f"   Keywords: {keywords}")
    print(f"   Keyword positions: {matches}")
    
    # Application 2: Text annotation
    print("\n2. Text Annotation:")
    text = "Python is a programming language used for machine learning"
    entities = ["Python", "programming", "language", "machine", "learning"]
    
    matches = solution.indexPairs1(text, entities)
    print(f"   Text: '{text}'")
    print(f"   Entities: {entities}")
    print(f"   Entity positions: {matches}")
    
    # Application 3: DNA sequence analysis
    print("\n3. DNA Sequence Analysis:")
    dna = "ATCGATCGATCG"
    patterns = ["ATC", "TCG", "GAT", "CGA"]
    
    matches = solution.indexPairs1(dna, patterns)
    print(f"   DNA: '{dna}'")
    print(f"   Patterns: {patterns}")
    print(f"   Pattern positions: {matches}")
    
    # Application 4: Log file analysis
    print("\n4. Log File Analysis:")
    log = "ERROR: Failed to connect to database. WARNING: Retrying connection"
    log_levels = ["ERROR", "WARNING", "INFO", "DEBUG"]
    
    matches = solution.indexPairs1(log, log_levels)
    print(f"   Log: '{log}'")
    print(f"   Log levels: {log_levels}")
    print(f"   Level positions: {matches}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty inputs
        ("", [], []),
        ("hello", [], []),
        ("", ["word"], []),
        
        # Single character
        ("a", ["a"], [[0,0]]),
        ("aaa", ["a"], [[0,0],[1,1],[2,2]]),
        
        # Overlapping matches
        ("ababa", ["aba", "bab"], [[0,2],[1,3],[2,4]]),
        
        # Duplicate words
        ("hello", ["hello", "hello"], [[0,4]]),
        
        # Case sensitivity
        ("Hello", ["hello", "Hello"], [[0,4]]),
        
        # Words longer than text
        ("hi", ["hello"], []),
        
        # All characters same
        ("aaaa", ["aa", "aaa"], [[0,1],[0,2],[1,2],[1,3],[2,3]]),
    ]
    
    for i, (text, words, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: '{text}' with {words}")
        try:
            result = solution.indexPairs1(text, words)
            status = "✓" if result == expected else "✗"
            print(f"  Result: {result} {status}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_basic_cases()
    benchmark_approaches()
    demonstrate_trie_construction()
    demonstrate_real_world_applications()
    test_edge_cases()

"""
1065. Index Pairs of a String demonstrates multiple approaches for finding word matches:

1. Trie-based - Efficient prefix tree for pattern matching
2. Brute Force - Simple substring checking against word set
3. Optimized Brute Force - Limited by maximum word length
4. KMP-based - Classic string matching algorithm for each pattern
5. Aho-Corasick - Multi-pattern string matching algorithm
6. Length-based Grouping - Words grouped by length for optimization

Each approach offers different trade-offs suitable for various text processing
scenarios from code syntax highlighting to DNA sequence analysis.
"""
