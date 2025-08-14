"""
Trie Interview Problems - LeetCode and Company Questions
========================================================

Topics: Classic interview problems with detailed solutions and analysis
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: O(m) to O(n*m) depending on problem
Space Complexity: O(ALPHABET_SIZE * N * M) for trie storage
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import heapq

class TrieInterviewProblems:
    
    def __init__(self):
        """Initialize with interview problem tracking"""
        self.problem_count = 0
        self.solution_stats = {}
    
    # ==========================================
    # 1. BASIC TRIE PROBLEMS
    # ==========================================
    
    def implement_trie_problem(self) -> None:
        """
        LeetCode 208: Implement Trie (Prefix Tree)
        
        Company: Google, Amazon, Microsoft, Facebook
        Difficulty: Medium
        Time: O(m) for all operations, Space: O(ALPHABET_SIZE * N * M)
        
        Implement a trie with insert, search, and startsWith operations
        """
        print("=== LEETCODE 208: IMPLEMENT TRIE ===")
        print("Problem: Implement Trie with insert, search, startsWith")
        print("Approach: Standard trie implementation with boolean end markers")
        print()
        
        trie = ImplementTrieLeetCode()
        
        # Test operations as in LeetCode example
        operations = [
            ("insert", "apple"),
            ("search", "apple"),      # True
            ("search", "app"),        # False
            ("startsWith", "app"),    # True
            ("insert", "app"),
            ("search", "app")         # True
        ]
        
        print("Test sequence:")
        for i, (operation, word) in enumerate(operations):
            print(f"Step {i+1}: {operation}('{word}')")
            
            if operation == "insert":
                trie.insert(word)
                print(f"  Inserted '{word}'")
            elif operation == "search":
                result = trie.search(word)
                print(f"  Search result: {result}")
            elif operation == "startsWith":
                result = trie.startsWith(word)
                print(f"  StartsWith result: {result}")
            print()
    
    def design_add_search_words(self) -> None:
        """
        LeetCode 211: Design Add and Search Words Data Structure
        
        Company: Facebook, Amazon, Google
        Difficulty: Medium
        Time: O(m) for add, O(n*26^m) for search with wildcards
        Space: O(ALPHABET_SIZE * N * M)
        
        Support '.' as wildcard character in search
        """
        print("=== LEETCODE 211: ADD AND SEARCH WORDS ===")
        print("Problem: Support wildcard '.' in search operations")
        print("Approach: DFS search with backtracking for '.' characters")
        print()
        
        word_dict = WordDictionary()
        
        # Test operations
        operations = [
            ("addWord", "bad"),
            ("addWord", "dad"),
            ("addWord", "mad"),
            ("search", "pad"),        # False
            ("search", "bad"),        # True
            ("search", ".ad"),        # True
            ("search", "b.."),        # True
            ("search", "..."),        # True
            ("search", "...."),       # False
        ]
        
        print("Test sequence:")
        for i, (operation, word) in enumerate(operations):
            print(f"Step {i+1}: {operation}('{word}')")
            
            if operation == "addWord":
                word_dict.addWord(word)
                print(f"  Added '{word}'")
            elif operation == "search":
                result = word_dict.search(word)
                print(f"  Search result: {result}")
            print()
    
    def word_search_ii(self) -> None:
        """
        LeetCode 212: Word Search II
        
        Company: Google, Amazon, Microsoft, Airbnb
        Difficulty: Hard
        Time: O(M*N*4^L) where M,N are board dimensions, L is max word length
        Space: O(K*L) where K is number of words
        
        Find all words from dictionary that can be formed on board
        """
        print("=== LEETCODE 212: WORD SEARCH II ===")
        print("Problem: Find all dictionary words that can be formed on 2D board")
        print("Approach: Build trie + DFS with backtracking + pruning")
        print()
        
        board = [
            ['o','a','a','n'],
            ['e','t','a','e'],
            ['i','h','k','r'],
            ['i','f','l','v']
        ]
        
        words = ["oath","pea","eat","rain","hklf","hf"]
        
        print("Board:")
        for row in board:
            print(f"  {' '.join(row)}")
        print()
        print(f"Dictionary words: {words}")
        print()
        
        solver = WordSearchII()
        result = solver.findWords(board, words)
        
        print(f"Words found: {result}")


class ImplementTrieLeetCode:
    """
    LeetCode 208: Implement Trie solution
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def insert(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = self.TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class WordDictionary:
    """
    LeetCode 211: Design Add and Search Words Data Structure
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def addWord(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = self.TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        def dfs(node, index):
            if index == len(word):
                return node.is_end_of_word
            
            char = word[index]
            if char == '.':
                # Wildcard - try all possible children
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # Regular character
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)
        
        return dfs(self.root, 0)


class WordSearchII:
    """
    LeetCode 212: Word Search II solution with optimization
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None  # Store complete word at end nodes
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # Build trie from words
        root = self.TrieNode()
        
        print("Building trie from dictionary words:")
        for word in words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = self.TrieNode()
                current = current.children[char]
            current.word = word
            print(f"  Added '{word}' to trie")
        
        print("\nSearching board using DFS + trie:")
        
        result = []
        rows, cols = len(board), len(board[0])
        
        def dfs(r, c, node):
            char = board[r][c]
            
            # Check if character exists in trie
            if char not in node.children:
                return
            
            # Move to next trie node
            next_node = node.children[char]
            
            # Check if we found a complete word
            if next_node.word:
                print(f"  Found word: '{next_node.word}' ending at ({r}, {c})")
                result.append(next_node.word)
                next_node.word = None  # Avoid duplicates
            
            # Mark current cell as visited
            board[r][c] = '#'
            
            # Explore all 4 directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                    dfs(nr, nc, next_node)
            
            # Restore original character
            board[r][c] = char
        
        # Try starting DFS from each cell
        for r in range(rows):
            for c in range(cols):
                char = board[r][c]
                if char in root.children:
                    dfs(r, c, root)
        
        return result


# ==========================================
# 2. ADVANCED TRIE PROBLEMS
# ==========================================

class AdvancedTrieProblems:
    """Advanced trie interview problems"""
    
    def longest_word_in_dictionary(self) -> None:
        """
        LeetCode 720: Longest Word in Dictionary
        
        Company: Google, Amazon
        Difficulty: Medium
        Time: O(sum of word lengths), Space: O(sum of word lengths)
        
        Find longest word that can be built one character at a time
        """
        print("=== LEETCODE 720: LONGEST WORD IN DICTIONARY ===")
        print("Problem: Find longest word built one character at a time")
        print("Approach: Trie + DFS to find valid building paths")
        print()
        
        words = ["w","wo","wor","worl","world"]
        
        print(f"Input words: {words}")
        
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_word = False
                self.word = ""
        
        # Build trie
        root = TrieNode()
        
        for word in words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.is_end_of_word = True
            current.word = word
        
        print("Built trie from words")
        
        # DFS to find longest word that can be built step by step
        def dfs(node):
            result = node.word
            
            for child in node.children.values():
                if child.is_end_of_word:  # Can only continue if each step is a valid word
                    candidate = dfs(child)
                    if len(candidate) > len(result) or (len(candidate) == len(result) and candidate < result):
                        result = candidate
            
            return result
        
        longest = dfs(root)
        print(f"Longest word that can be built: '{longest}'")
        print()
    
    def replace_words(self) -> None:
        """
        LeetCode 648: Replace Words
        
        Company: Google, Amazon
        Difficulty: Medium
        Time: O(sum of dictionary lengths + sentence length)
        Space: O(sum of dictionary lengths)
        
        Replace words with their shortest root from dictionary
        """
        print("=== LEETCODE 648: REPLACE WORDS ===")
        print("Problem: Replace words with their shortest dictionary root")
        print("Approach: Trie for roots + prefix matching")
        print()
        
        dictionary = ["cat","bat","rat"]
        sentence = "the cattle was rattled by the battery"
        
        print(f"Dictionary: {dictionary}")
        print(f"Sentence: '{sentence}'")
        
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_word = False
        
        # Build trie from dictionary
        root = TrieNode()
        
        print("\nBuilding trie from dictionary:")
        for word in dictionary:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.is_end_of_word = True
            print(f"  Added root: '{word}'")
        
        # Function to find shortest root
        def find_root(word):
            current = root
            
            for i, char in enumerate(word):
                if char not in current.children:
                    return word  # No root found, return original word
                
                current = current.children[char]
                
                if current.is_end_of_word:
                    return word[:i+1]  # Found root, return prefix
            
            return word  # No root found
        
        # Process sentence
        words = sentence.split()
        result_words = []
        
        print("\nReplacing words:")
        for word in words:
            root_word = find_root(word)
            result_words.append(root_word)
            if root_word != word:
                print(f"  '{word}' -> '{root_word}'")
            else:
                print(f"  '{word}' (no replacement)")
        
        result = " ".join(result_words)
        print(f"\nResult: '{result}'")
        print()
    
    def palindrome_pairs(self) -> None:
        """
        LeetCode 336: Palindrome Pairs
        
        Company: Google, Amazon, Airbnb
        Difficulty: Hard
        Time: O(n * m^2) where n is number of words, m is average word length
        Space: O(n * m)
        
        Find all pairs of words that form palindrome when concatenated
        """
        print("=== LEETCODE 336: PALINDROME PAIRS ===")
        print("Problem: Find pairs that form palindromes when concatenated")
        print("Approach: Trie + palindrome checking + edge case handling")
        print()
        
        words = ["abcd","dcba","lls","s","sssll"]
        
        print(f"Input words: {words}")
        
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.word_index = -1
                self.palindrome_suffixes = []  # Indices of words with palindromic suffixes
        
        def is_palindrome(s):
            return s == s[::-1]
        
        # Build trie with reversed words
        root = TrieNode()
        
        print("\nBuilding trie with reversed words:")
        for i, word in enumerate(words):
            current = root
            reversed_word = word[::-1]
            
            for j, char in enumerate(reversed_word):
                # Check if remaining part forms palindrome
                remaining = word[:len(word)-j]
                if is_palindrome(remaining):
                    current.palindrome_suffixes.append(i)
                
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            
            current.word_index = i
            print(f"  Added reversed '{word}' -> '{reversed_word}' at index {i}")
        
        result = []
        
        print("\nFinding palindrome pairs:")
        for i, word in enumerate(words):
            current = root
            
            # Search for word in trie of reversed words
            for j, char in enumerate(word):
                # Case 1: Found complete word and remaining forms palindrome
                if current.word_index != -1 and current.word_index != i:
                    remaining = word[j:]
                    if is_palindrome(remaining):
                        pair = [i, current.word_index]
                        result.append(pair)
                        print(f"  Found pair: {words[i]} + {words[current.word_index]} (case 1)")
                
                if char not in current.children:
                    break
                current = current.children[char]
            else:
                # Case 2: Consumed entire word, check palindromic suffixes
                if current.word_index != -1 and current.word_index != i:
                    pair = [i, current.word_index]
                    result.append(pair)
                    print(f"  Found pair: {words[i]} + {words[current.word_index]} (case 2)")
                
                # Case 3: Check words with palindromic suffixes
                for suffix_idx in current.palindrome_suffixes:
                    if suffix_idx != i:
                        pair = [i, suffix_idx]
                        result.append(pair)
                        print(f"  Found pair: {words[i]} + {words[suffix_idx]} (case 3)")
        
        print(f"\nAll palindrome pairs: {result}")
        
        # Verify results
        print("\nVerification:")
        for i, j in result:
            concatenated = words[i] + words[j]
            is_pal = is_palindrome(concatenated)
            print(f"  '{words[i]}' + '{words[j]}' = '{concatenated}' -> {'✓' if is_pal else '✗'}")
        
        print()


# ==========================================
# 3. TRIE OPTIMIZATION PROBLEMS
# ==========================================

class TrieOptimizationProblems:
    """Problems focusing on trie optimization techniques"""
    
    def maximum_xor_queries(self) -> None:
        """
        LeetCode 1707: Maximum XOR With an Element From Array
        
        Company: Google, Facebook
        Difficulty: Hard
        Time: O(n log n + q log max_val), Space: O(n * log max_val)
        
        Answer queries for maximum XOR with constraint
        """
        print("=== LEETCODE 1707: MAXIMUM XOR QUERIES ===")
        print("Problem: Find maximum XOR with elements ≤ constraint")
        print("Approach: Binary trie + offline query processing")
        print()
        
        nums = [0,1,2,3,4]
        queries = [[3,1],[1,3],[5,6]]
        
        print(f"Array: {nums}")
        print(f"Queries: {queries} (format: [x, m] - find max XOR of x with elements ≤ m)")
        
        class BinaryTrieNode:
            def __init__(self):
                self.children = {}
                self.min_value = float('inf')  # Minimum value in subtree
        
        class BinaryTrie:
            def __init__(self):
                self.root = BinaryTrieNode()
            
            def insert(self, num):
                current = self.root
                current.min_value = min(current.min_value, num)
                
                # Process from most significant bit (31st bit for 32-bit integers)
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    
                    if bit not in current.children:
                        current.children[bit] = BinaryTrieNode()
                    
                    current = current.children[bit]
                    current.min_value = min(current.min_value, num)
            
            def find_max_xor(self, num, max_val):
                if self.root.min_value > max_val:
                    return -1
                
                current = self.root
                result = 0
                
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    desired_bit = 1 - bit  # We want opposite bit for maximum XOR
                    
                    if (desired_bit in current.children and 
                        current.children[desired_bit].min_value <= max_val):
                        result |= (1 << i)
                        current = current.children[desired_bit]
                    elif bit in current.children:
                        current = current.children[bit]
                    else:
                        return -1
                
                return result
        
        # Sort nums for offline processing
        sorted_nums = sorted(nums)
        trie = BinaryTrie()
        
        result = []
        
        print("\nProcessing queries:")
        for query_idx, (x, m) in enumerate(queries):
            print(f"Query {query_idx + 1}: x={x}, m={m}")
            
            # Add all numbers ≤ m to trie
            for num in sorted_nums:
                if num <= m:
                    trie.insert(num)
                    print(f"  Added {num} to trie (binary: {bin(num)[2:].zfill(8)})")
                else:
                    break
            
            # Find maximum XOR
            max_xor = trie.find_max_xor(x, m)
            result.append(max_xor)
            
            print(f"  Maximum XOR of {x} with elements ≤ {m}: {max_xor}")
            if max_xor != -1:
                print(f"  Binary: {bin(x)[2:].zfill(8)} XOR ? = {bin(max_xor)[2:].zfill(8)}")
            print()
        
        print(f"Final results: {result}")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_trie_interview_problems():
    """Demonstrate all trie interview problems"""
    print("=== TRIE INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    interview_problems = TrieInterviewProblems()
    
    # 1. Basic trie problems
    print("=== BASIC TRIE PROBLEMS ===")
    
    interview_problems.implement_trie_problem()
    print("\n" + "-"*50 + "\n")
    
    interview_problems.design_add_search_words()
    print("\n" + "-"*50 + "\n")
    
    interview_problems.word_search_ii()
    print("\n" + "="*60 + "\n")
    
    # 2. Advanced trie problems
    print("=== ADVANCED TRIE PROBLEMS ===")
    
    advanced_problems = AdvancedTrieProblems()
    
    advanced_problems.longest_word_in_dictionary()
    print("-"*50 + "\n")
    
    advanced_problems.replace_words()
    print("-"*50 + "\n")
    
    advanced_problems.palindrome_pairs()
    print("="*60 + "\n")
    
    # 3. Optimization problems
    print("=== TRIE OPTIMIZATION PROBLEMS ===")
    
    optimization_problems = TrieOptimizationProblems()
    optimization_problems.maximum_xor_queries()


if __name__ == "__main__":
    demonstrate_trie_interview_problems()
    
    print("\n=== TRIE INTERVIEW PROBLEMS MASTERY GUIDE ===")
    
    print("\n🎯 PROBLEM PATTERNS:")
    print("• Basic Implementation: Trie class with insert/search/prefix")
    print("• Wildcard Search: DFS with backtracking for '.' character")
    print("• Board Search: Trie + DFS for 2D grid word finding")
    print("• String Processing: Root finding, word building, palindromes")
    print("• Bit Manipulation: Binary trie for XOR problems")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Basic Operations: O(m) where m is word/prefix length")
    print("• Wildcard Search: O(n * 26^k) where k is wildcards")
    print("• Board Search: O(M*N*4^L) for M×N board, L max word length")
    print("• XOR Problems: O(32) per operation for 32-bit integers")
    
    print("\n⚡ OPTIMIZATION TECHNIQUES:")
    print("• Early termination in DFS when no valid paths remain")
    print("• Remove words from trie after finding to avoid duplicates")
    print("• Use iterative approaches to avoid stack overflow")
    print("• Precompute palindromic suffixes for palindrome problems")
    print("• Sort data for offline query processing")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Mark visited cells in board problems and restore after DFS")
    print("• Handle edge cases: empty strings, single characters")
    print("• Use appropriate data structures for different alphabets")
    print("• Consider memory optimization for large datasets")
    print("• Add comprehensive test cases for edge conditions")
    
    print("\n🏆 INTERVIEW SUCCESS STRATEGIES:")
    print("• Start with basic trie implementation and extend")
    print("• Clearly explain the approach before coding")
    print("• Discuss time/space complexity trade-offs")
    print("• Handle edge cases and validate inputs")
    print("• Test with provided examples and additional cases")
    
    print("\n🎓 PROBLEM DIFFICULTY PROGRESSION:")
    print("Easy: Basic trie operations, simple prefix matching")
    print("Medium: Wildcard search, word building, dictionary problems")
    print("Hard: Board search, palindrome pairs, XOR optimization")
    print("Expert: Distributed tries, persistent data structures")
    
    print("\n📚 LEETCODE PROBLEMS TO MASTER:")
    print("• 208. Implement Trie (Prefix Tree) - Foundation")
    print("• 211. Design Add and Search Words - Wildcards")
    print("• 212. Word Search II - Board + Trie")
    print("• 648. Replace Words - String processing")
    print("• 720. Longest Word in Dictionary - Word building")
    print("• 336. Palindrome Pairs - Advanced string algorithms")
    print("• 421. Maximum XOR of Two Numbers - Binary trie")
    print("• 1707. Maximum XOR Queries - Advanced XOR with constraints")
