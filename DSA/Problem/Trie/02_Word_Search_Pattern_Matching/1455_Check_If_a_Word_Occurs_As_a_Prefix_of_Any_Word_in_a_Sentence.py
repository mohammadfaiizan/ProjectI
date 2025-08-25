"""
1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence - Multiple Approaches
Difficulty: Easy

Given a sentence that consists of some words separated by a single space, 
and a searchWord, check if searchWord is a prefix of any word in sentence.

Return the index of the word in sentence (1-indexed) where searchWord is a prefix 
of this word. If searchWord is a prefix of more than one word, return the index 
of the first such word. If there is no such word return -1.

LeetCode Problem: https://leetcode.com/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/

Example:
Input: sentence = "i love eating burger", searchWord = "burg"
Output: 4
Explanation: "burg" is prefix of "burger" which is the 4th word in the sentence.
"""

from typing import List

class TrieNode:
    """Trie node for prefix matching"""
    def __init__(self):
        self.children = {}
        self.word_indices = []  # Store indices of words ending at this node

class Solution:
    
    def isPrefixOfWord1(self, sentence: str, searchWord: str) -> int:
        """
        Approach 1: Simple String Matching
        
        Split sentence and check each word's prefix.
        
        Time: O(n*m) where n is number of words, m is searchWord length
        Space: O(n) for splitting sentence
        """
        words = sentence.split()
        
        for i, word in enumerate(words):
            if word.startswith(searchWord):
                return i + 1  # 1-indexed
        
        return -1
    
    def isPrefixOfWord2(self, sentence: str, searchWord: str) -> int:
        """
        Approach 2: Manual Prefix Check
        
        Check prefix without using built-in startswith method.
        
        Time: O(n*m)
        Space: O(1) if we don't split the sentence
        """
        words = sentence.split()
        
        for i, word in enumerate(words):
            if len(word) >= len(searchWord):
                # Check if searchWord matches prefix
                match = True
                for j in range(len(searchWord)):
                    if word[j] != searchWord[j]:
                        match = False
                        break
                
                if match:
                    return i + 1
        
        return -1
    
    def isPrefixOfWord3(self, sentence: str, searchWord: str) -> int:
        """
        Approach 3: Single Pass Without Split
        
        Process sentence character by character without splitting.
        
        Time: O(n) where n is sentence length
        Space: O(1)
        """
        word_index = 1
        i = 0
        
        while i < len(sentence):
            # Skip leading spaces
            while i < len(sentence) and sentence[i] == ' ':
                i += 1
                word_index += 1
            
            if i >= len(sentence):
                break
            
            # Check if current word starts with searchWord
            j = 0
            start_i = i
            
            # Compare with searchWord
            while (i < len(sentence) and sentence[i] != ' ' and 
                   j < len(searchWord) and sentence[i] == searchWord[j]):
                i += 1
                j += 1
            
            # Found complete prefix match
            if j == len(searchWord):
                return word_index
            
            # Skip rest of current word
            while i < len(sentence) and sentence[i] != ' ':
                i += 1
            
            word_index += 1
        
        return -1
    
    def isPrefixOfWord4(self, sentence: str, searchWord: str) -> int:
        """
        Approach 4: Trie-based Solution
        
        Build trie from sentence words and search for prefix.
        
        Time: O(total_chars + m) for building trie and searching
        Space: O(total_chars) for trie
        """
        # Build trie
        root = TrieNode()
        words = sentence.split()
        
        for word_idx, word in enumerate(words):
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Store word index at each node along the path
                if not node.word_indices or node.word_indices[-1] != word_idx + 1:
                    node.word_indices.append(word_idx + 1)
        
        # Search for prefix
        node = root
        for char in searchWord:
            if char not in node.children:
                return -1
            node = node.children[char]
        
        # Return the first word index that has this prefix
        return node.word_indices[0] if node.word_indices else -1
    
    def isPrefixOfWord5(self, sentence: str, searchWord: str) -> int:
        """
        Approach 5: Optimized with Early Termination
        
        Stop search as soon as first match is found.
        
        Time: O(best case: m, worst case: n*m)
        Space: O(n) for words list
        """
        words = sentence.split()
        search_len = len(searchWord)
        
        for i, word in enumerate(words):
            # Early termination if word is too short
            if len(word) < search_len:
                continue
            
            # Check prefix match
            if word[:search_len] == searchWord:
                return i + 1
        
        return -1
    
    def isPrefixOfWord6(self, sentence: str, searchWord: str) -> int:
        """
        Approach 6: Regular Expression
        
        Use regex for pattern matching.
        
        Time: O(n) with regex engine optimization
        Space: O(1)
        """
        import re
        
        # Create pattern: word boundary + searchWord + any word characters
        pattern = r'\b' + re.escape(searchWord) + r'\w*'
        
        words = sentence.split()
        
        for i, word in enumerate(words):
            if re.match(pattern, word):
                return i + 1
        
        return -1
    
    def isPrefixOfWord7(self, sentence: str, searchWord: str) -> int:
        """
        Approach 7: KMP-inspired Prefix Search
        
        Use KMP algorithm concept for efficient prefix matching.
        
        Time: O(n + m)
        Space: O(m) for failure function
        """
        def build_failure_function(pattern: str) -> List[int]:
            """Build KMP failure function"""
            m = len(pattern)
            failure = [0] * m
            j = 0
            
            for i in range(1, m):
                while j > 0 and pattern[i] != pattern[j]:
                    j = failure[j - 1]
                
                if pattern[i] == pattern[j]:
                    j += 1
                
                failure[i] = j
            
            return failure
        
        failure = build_failure_function(searchWord)
        words = sentence.split()
        
        for word_idx, word in enumerate(words):
            # Use KMP-style matching for prefix
            i = j = 0
            
            while i < len(word) and j < len(searchWord):
                if word[i] == searchWord[j]:
                    i += 1
                    j += 1
                else:
                    if j > 0:
                        j = failure[j - 1]
                    else:
                        i += 1
                
                # Found complete prefix
                if j == len(searchWord):
                    return word_idx + 1
        
        return -1


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        ("i love eating burger", "burg", 4),
        ("this problem is an easy problem", "pro", 2),
        ("i am tired", "you", -1),
        ("i use triple pillow", "pill", 4),
        ("hello from the other side", "they", -1),
        ("hello world", "world", -1),  # "world" is not a prefix
        ("hello world", "wor", 2),     # "wor" is prefix of "world"
        ("a", "a", 1),
        ("a aa aaa", "aa", 2),
        ("", "test", -1),
    ]
    
    approaches = [
        ("String Matching", solution.isPrefixOfWord1),
        ("Manual Check", solution.isPrefixOfWord2),
        ("Single Pass", solution.isPrefixOfWord3),
        ("Trie-based", solution.isPrefixOfWord4),
        ("Optimized", solution.isPrefixOfWord5),
        ("Regex", solution.isPrefixOfWord6),
        ("KMP-inspired", solution.isPrefixOfWord7),
    ]
    
    for i, (sentence, searchWord, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: '{sentence}' | searchWord: '{searchWord}'")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(sentence, searchWord)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_step_by_step():
    """Demonstrate step-by-step process"""
    print("\n=== Step-by-Step Demo ===")
    
    sentence = "i love eating burger"
    searchWord = "burg"
    
    print(f"Sentence: '{sentence}'")
    print(f"Search word: '{searchWord}'")
    
    words = sentence.split()
    print(f"\nWords: {words}")
    
    for i, word in enumerate(words):
        print(f"\nWord {i+1}: '{word}'")
        
        if len(word) >= len(searchWord):
            prefix = word[:len(searchWord)]
            print(f"  Prefix of length {len(searchWord)}: '{prefix}'")
            
            if prefix == searchWord:
                print(f"  ✓ Match found! '{searchWord}' is prefix of '{word}'")
                print(f"  Return index: {i+1}")
                break
            else:
                print(f"  ✗ No match: '{prefix}' != '{searchWord}'")
        else:
            print(f"  ✗ Word too short (length {len(word)} < {len(searchWord)})")


def demonstrate_trie_approach():
    """Demonstrate trie-based approach"""
    print("\n=== Trie Approach Demo ===")
    
    sentence = "hello world wonderful"
    searchWord = "wor"
    
    print(f"Sentence: '{sentence}'")
    print(f"Search word: '{searchWord}'")
    
    # Build trie
    root = TrieNode()
    words = sentence.split()
    
    print(f"\nBuilding trie:")
    for word_idx, word in enumerate(words):
        print(f"  Inserting word {word_idx+1}: '{word}'")
        node = root
        
        for char_idx, char in enumerate(word):
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"    Created node for '{char}' at depth {char_idx+1}")
            
            node = node.children[char]
            if not node.word_indices or node.word_indices[-1] != word_idx + 1:
                node.word_indices.append(word_idx + 1)
                print(f"    Added word index {word_idx+1} to node '{word[:char_idx+1]}'")
    
    # Search for prefix
    print(f"\nSearching for prefix '{searchWord}':")
    node = root
    
    for i, char in enumerate(searchWord):
        print(f"  Step {i+1}: Looking for '{char}'")
        
        if char not in node.children:
            print(f"    Character '{char}' not found")
            return
        
        node = node.children[char]
        print(f"    Found '{char}', current prefix: '{searchWord[:i+1]}'")
        print(f"    Words with this prefix: {node.word_indices}")
    
    result = node.word_indices[0] if node.word_indices else -1
    print(f"\nResult: {result}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_sentence(num_words: int, avg_word_length: int) -> str:
        words = []
        for _ in range(num_words):
            length = max(1, avg_word_length + random.randint(-2, 2))
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return ' '.join(words)
    
    test_scenarios = [
        ("Short", generate_sentence(10, 5), "test"),
        ("Medium", generate_sentence(50, 6), "pref"),
        ("Long", generate_sentence(200, 7), "word"),
    ]
    
    approaches = [
        ("String Match", solution.isPrefixOfWord1),
        ("Single Pass", solution.isPrefixOfWord3),
        ("Optimized", solution.isPrefixOfWord5),
        ("Trie-based", solution.isPrefixOfWord4),
    ]
    
    for scenario_name, sentence, searchWord in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Sentence length: {len(sentence)} chars, Search: '{searchWord}'")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(100):
                result = method(sentence, searchWord)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            
            print(f"  {approach_name:12}: {avg_time*1000:.3f}ms per call")


def demonstrate_edge_cases():
    """Demonstrate edge case handling"""
    print("\n=== Edge Cases Demo ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty cases
        ("", "word", "Empty sentence"),
        ("hello world", "", "Empty search word"),
        
        # Single character
        ("a b c", "a", "Single character match"),
        ("x y z", "a", "Single character no match"),
        
        # Search word longer than any word
        ("hi bye", "hello", "Search longer than words"),
        
        # Exact word match
        ("hello world", "hello", "Exact word as prefix"),
        
        # Case sensitivity
        ("Hello world", "hello", "Case sensitivity"),
        
        # Multiple matches
        ("test testing tester", "test", "Multiple prefix matches"),
        
        # Word boundaries
        ("testing test", "test", "Word boundary importance"),
    ]
    
    for sentence, searchWord, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Sentence: '{sentence}'")
        print(f"  Search: '{searchWord}'")
        
        try:
            result = solution.isPrefixOfWord1(sentence, searchWord)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Search autocomplete
    print("1. Search Autocomplete System:")
    
    search_history = [
        "python programming tutorial",
        "java script frameworks",
        "machine learning algorithms",
        "data structures and algorithms",
        "web development basics"
    ]
    
    user_query = "prog"
    
    print(f"   Search history: {search_history}")
    print(f"   User types: '{user_query}'")
    print(f"   Matching suggestions:")
    
    for i, query in enumerate(search_history):
        match_pos = solution.isPrefixOfWord1(query, user_query)
        if match_pos != -1:
            words = query.split()
            matched_word = words[match_pos - 1]
            print(f"     '{query}' -> word {match_pos}: '{matched_word}'")
    
    # Application 2: Command line completion
    print(f"\n2. Command Line Completion:")
    
    commands = [
        "git commit -m message",
        "docker run container",
        "npm install package",
        "python script.py",
        "ls -la directory"
    ]
    
    partial_command = "doc"
    
    print(f"   Available commands: {commands}")
    print(f"   User types: '{partial_command}'")
    
    for command in commands:
        match_pos = solution.isPrefixOfWord1(command, partial_command)
        if match_pos != -1:
            print(f"     Match: '{command}' (word {match_pos})")
    
    # Application 3: Document search
    print(f"\n3. Document Keyword Search:")
    
    documents = [
        "artificial intelligence research paper",
        "natural language processing techniques",
        "computer vision applications",
        "deep learning neural networks"
    ]
    
    keyword = "nat"
    
    print(f"   Documents: {documents}")
    print(f"   Keyword search: '{keyword}'")
    
    for i, doc in enumerate(documents):
        match_pos = solution.isPrefixOfWord1(doc, keyword)
        if match_pos != -1:
            print(f"     Document {i+1}: Match at word position {match_pos}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("String Matching",
         "Time: O(n*m) - check each word's prefix",
         "Space: O(n) - store words array"),
        
        ("Single Pass",
         "Time: O(s) - single pass through sentence",
         "Space: O(1) - constant extra space"),
        
        ("Trie-based",
         "Time: O(total_chars + m) - build trie + search",
         "Space: O(total_chars) - trie storage"),
        
        ("Optimized",
         "Time: O(best: m, worst: n*m) - early termination",
         "Space: O(n) - words array"),
        
        ("KMP-inspired",
         "Time: O(n + m) - linear time complexity",
         "Space: O(m) - failure function"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  n = number of words in sentence")
    print(f"  m = length of search word")
    print(f"  s = length of sentence")
    print(f"  total_chars = sum of all character lengths")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_step_by_step()
    demonstrate_trie_approach()
    benchmark_approaches()
    demonstrate_edge_cases()
    demonstrate_real_world_applications()
    analyze_complexity()

"""
1455. Check If a Word Occurs As a Prefix demonstrates multiple prefix matching approaches:

1. String Matching - Simple built-in startswith() method
2. Manual Check - Character-by-character comparison without built-ins
3. Single Pass - Process sentence without splitting into words
4. Trie-based - Structured approach for multiple queries
5. Optimized - Early termination and slice comparison
6. Regular Expression - Pattern matching with regex engine
7. KMP-inspired - Linear time algorithm for pattern matching

Each approach shows different trade-offs between simplicity, efficiency,
and suitability for various use cases from simple searches to autocomplete systems.
"""
