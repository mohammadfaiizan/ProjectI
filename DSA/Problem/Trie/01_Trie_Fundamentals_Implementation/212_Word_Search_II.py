"""
212. Word Search II - Multiple Approaches
Difficulty: Hard

Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent 
cells are horizontally or vertically neighboring. The same letter cell may not be used 
more than once in a word.

LeetCode Problem: https://leetcode.com/problems/word-search-ii/

Example:
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], 
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
"""

from typing import List, Set
from collections import defaultdict

class TrieNode:
    """Trie node for word search"""
    def __init__(self):
        self.children = {}
        self.word = None  # Store complete word at end nodes

class Solution:
    
    def findWords1(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 1: Trie + DFS with Backtracking
        
        Build trie from words and use DFS to search board.
        
        Time: O(m*n*4^L) where m,n are board dimensions, L is max word length
        Space: O(W*L) where W is number of words
        """
        if not board or not board[0] or not words:
            return []
        
        # Build trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        
        result = []
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, node: TrieNode):
            # Check bounds and if character exists in trie
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] not in node.children):
                return
            
            char = board[i][j]
            node = node.children[char]
            
            # Found a word
            if node.word:
                result.append(node.word)
                node.word = None  # Avoid duplicates
            
            # Mark as visited
            board[i][j] = '#'
            
            # Explore all 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, node)
            
            # Backtrack
            board[i][j] = char
        
        # Start DFS from each cell
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        
        return result
    
    def findWords2(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 2: Optimized Trie with Pruning
        
        Enhanced trie with node pruning to improve performance.
        
        Time: O(m*n*4^L) but with significant pruning
        Space: O(W*L)
        """
        if not board or not board[0] or not words:
            return []
        
        # Build trie with reference counting
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        
        result = []
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, parent: TrieNode, node: TrieNode):
            char = board[i][j]
            
            # Found a word
            if node.word:
                result.append(node.word)
                node.word = None  # Remove to avoid duplicates
            
            # Mark as visited
            board[i][j] = '#'
            
            # Explore 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < m and 0 <= nj < n and 
                    board[ni][nj] in node.children):
                    dfs(ni, nj, node, node.children[board[ni][nj]])
            
            # Backtrack
            board[i][j] = char
            
            # Pruning: remove node if it has no children and no word
            if not node.children and not node.word and parent:
                del parent.children[char]
        
        # Start DFS from each cell
        for i in range(m):
            for j in range(n):
                if board[i][j] in root.children:
                    dfs(i, j, root, root.children[board[i][j]])
        
        return result
    
    def findWords3(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 3: Character Frequency Optimization
        
        Pre-filter words based on character frequency in board.
        
        Time: O(m*n + W*L + filtered_searches)
        Space: O(W*L + alphabet_size)
        """
        if not board or not board[0] or not words:
            return []
        
        # Count character frequency in board
        char_count = defaultdict(int)
        for row in board:
            for char in row:
                char_count[char] += 1
        
        # Filter words that can't possibly be formed
        valid_words = []
        for word in words:
            word_char_count = defaultdict(int)
            for char in word:
                word_char_count[char] += 1
            
            # Check if board has enough characters
            possible = True
            for char, count in word_char_count.items():
                if char_count[char] < count:
                    possible = False
                    break
            
            if possible:
                valid_words.append(word)
        
        # If no valid words, return empty
        if not valid_words:
            return []
        
        # Use standard trie approach with filtered words
        return self.findWords1(board, valid_words)
    
    def findWords4(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 4: Reverse Word Strategy
        
        For each word, reverse it and try both directions.
        
        Time: O(W * m * n * 4^L)
        Space: O(W * L)
        """
        if not board or not board[0] or not words:
            return []
        
        # Build trie with both original and reversed words
        root = TrieNode()
        
        for word in words:
            # Insert original word
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
            
            # Insert reversed word
            reversed_word = word[::-1]
            if reversed_word != word:  # Avoid duplicates for palindromes
                node = root
                for char in reversed_word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                if not node.word:  # Only set if not already set
                    node.word = word  # Store original word
        
        result = set()
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, node: TrieNode):
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] not in node.children):
                return
            
            char = board[i][j]
            node = node.children[char]
            
            if node.word:
                result.add(node.word)
            
            board[i][j] = '#'
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, node)
            
            board[i][j] = char
        
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        
        return list(result)
    
    def findWords5(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 5: Iterative BFS with Queue
        
        Use BFS instead of DFS for word search.
        
        Time: O(m*n*4^L)
        Space: O(m*n*L) for queue
        """
        if not board or not board[0] or not words:
            return []
        
        from collections import deque
        
        # Build trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        
        result = []
        m, n = len(board), len(board[0])
        
        # BFS from each starting position
        for start_i in range(m):
            for start_j in range(n):
                if board[start_i][start_j] not in root.children:
                    continue
                
                # Queue: (i, j, trie_node, visited_set)
                queue = deque([(start_i, start_j, root.children[board[start_i][start_j]], 
                               {(start_i, start_j)})])
                
                while queue:
                    i, j, node, visited = queue.popleft()
                    
                    if node.word:
                        result.append(node.word)
                        node.word = None  # Avoid duplicates
                    
                    # Explore neighbors
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        
                        if (0 <= ni < m and 0 <= nj < n and 
                            (ni, nj) not in visited and
                            board[ni][nj] in node.children):
                            
                            new_visited = visited | {(ni, nj)}
                            queue.append((ni, nj, node.children[board[ni][nj]], new_visited))
        
        return result
    
    def findWords6(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 6: Multi-threaded Search (Conceptual)
        
        Divide board into regions for parallel processing.
        
        Time: O(m*n*4^L / threads)
        Space: O(W*L)
        """
        # This is a conceptual approach - actual implementation would require
        # proper thread management and synchronization
        return self.findWords1(board, words)


def test_basic_cases():
    """Test basic word search functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode example
        ([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], 
         ["oath","pea","eat","rain"], 
         ["eat","oath"]),
        
        # Simple case
        ([["a","b"],["c","d"]], 
         ["ab","cd","ac","bd"], 
         ["ab","cd","ac","bd"]),
        
        # No words found
        ([["a","b"],["c","d"]], 
         ["xyz"], 
         []),
        
        # Single cell
        ([["a"]], 
         ["a"], 
         ["a"]),
        
        # Overlapping paths
        ([["a","a","a"],["a","a","a"],["a","a","a"]], 
         ["aaa","aaaa"], 
         ["aaa","aaaa"]),
    ]
    
    approaches = [
        ("Trie + DFS", solution.findWords1),
        ("Optimized Trie", solution.findWords2),
        ("Char Frequency", solution.findWords3),
        ("Reverse Strategy", solution.findWords4),
        ("BFS Iterative", solution.findWords5),
    ]
    
    for i, (board, words, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Board: {board}")
        print(f"Words: {words}")
        print(f"Expected: {sorted(expected)}")
        
        for name, method in approaches:
            try:
                # Make copy of board since some methods modify it
                board_copy = [row[:] for row in board]
                result = method(board_copy, words)
                result_sorted = sorted(result)
                status = "✓" if result_sorted == sorted(expected) else "✗"
                print(f"  {name:15}: {result_sorted} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_trie_construction():
    """Demonstrate trie construction for word search"""
    print("\n=== Trie Construction Demo ===")
    
    words = ["oath", "pea", "eat", "rain"]
    print(f"Building trie for words: {words}")
    
    # Build trie step by step
    root = TrieNode()
    
    for word in words:
        print(f"\nInserting '{word}':")
        node = root
        path = ""
        
        for char in word:
            path += char
            if char not in node.children:
                node.children[char] = TrieNode()
                print(f"  Created node for '{path}'")
            else:
                print(f"  Using existing node for '{path}'")
            node = node.children[char]
        
        node.word = word
        print(f"  Marked '{word}' as complete word")
    
    # Show trie structure
    def print_trie(node, prefix="", level=0):
        if node.word:
            print(f"{'  ' * level}'{prefix}' -> WORD: {node.word}")
        
        for char, child in node.children.items():
            print(f"{'  ' * level}'{prefix}' -> '{char}'")
            print_trie(child, prefix + char, level + 1)
    
    print(f"\nTrie structure:")
    print_trie(root)


def demonstrate_search_process():
    """Demonstrate the search process on board"""
    print("\n=== Search Process Demo ===")
    
    board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    words = ["eat"]
    
    print(f"Board:")
    for i, row in enumerate(board):
        print(f"  {i}: {row}")
    
    print(f"Searching for: {words}")
    
    # Build trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    m, n = len(board), len(board[0])
    result = []
    
    def dfs_demo(i: int, j: int, node: TrieNode, path: str, visited: Set):
        print(f"    Visiting ({i},{j}) = '{board[i][j]}', path so far: '{path}'")
        
        # Check bounds and if character exists in trie
        if (i < 0 or i >= m or j < 0 or j >= n or 
            (i, j) in visited or board[i][j] not in node.children):
            print(f"    Invalid move or already visited")
            return
        
        char = board[i][j]
        node = node.children[char]
        new_path = path + char
        new_visited = visited | {(i, j)}
        
        # Found a word
        if node.word:
            print(f"    FOUND WORD: {node.word}")
            result.append(node.word)
            return  # Stop here for demo
        
        # Explore all 4 directions
        print(f"    Exploring neighbors from ({i},{j})")
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in new_visited:
                print(f"    Trying direction to ({ni},{nj}) = '{board[ni][nj]}'")
                dfs_demo(ni, nj, node, new_path, new_visited)
    
    # Start search from position (1,1) which has 'e'
    print(f"\nStarting search from position (1,1) = '{board[1][1]}':")
    if board[1][1] in root.children:
        dfs_demo(1, 1, root, "", set())
    
    print(f"\nResult: {result}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_board(m: int, n: int) -> List[List[str]]:
        return [[random.choice(string.ascii_lowercase[:10]) for _ in range(n)] for _ in range(m)]
    
    def generate_words(count: int, max_length: int) -> List[str]:
        words = []
        for _ in range(count):
            length = random.randint(3, max_length)
            word = ''.join(random.choices(string.ascii_lowercase[:10], k=length))
            words.append(word)
        return list(set(words))  # Remove duplicates
    
    test_scenarios = [
        ("Small", generate_board(4, 4), generate_words(10, 6)),
        ("Medium", generate_board(6, 6), generate_words(20, 8)),
        ("Large", generate_board(8, 8), generate_words(50, 10)),
    ]
    
    approaches = [
        ("Trie + DFS", solution.findWords1),
        ("Optimized", solution.findWords2),
        ("Char Filter", solution.findWords3),
    ]
    
    for scenario_name, board, words in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Board: {len(board)}x{len(board[0])}, Words: {len(words)}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Make copy of board
            board_copy = [row[:] for row in board]
            result = method(board_copy, words)
            
            end_time = time.time()
            
            print(f"  {approach_name:12}: {(end_time - start_time)*1000:.2f}ms, found {len(result)} words")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Word puzzle solver
    print("1. Word Puzzle Solver:")
    puzzle = [
        ['C', 'A', 'T', 'S'],
        ['O', 'G', 'O', 'H'],
        ['D', 'O', 'G', 'S'],
        ['R', 'A', 'T', 'S']
    ]
    
    animal_words = ["CAT", "DOG", "CATS", "DOGS", "RATS", "GOAT"]
    
    found_animals = solution.findWords1(puzzle, animal_words)
    print(f"   Puzzle grid: {len(puzzle)}x{len(puzzle[0])}")
    print(f"   Animal words to find: {animal_words}")
    print(f"   Found animals: {found_animals}")
    
    # Application 2: DNA sequence analysis
    print("\n2. DNA Sequence Analysis:")
    dna_grid = [
        ['A', 'T', 'G', 'C'],
        ['T', 'G', 'C', 'A'],
        ['G', 'C', 'A', 'T'],
        ['C', 'A', 'T', 'G']
    ]
    
    genetic_patterns = ["ATG", "TAA", "TGA", "ATGC", "GCAT"]
    
    found_patterns = solution.findWords1(dna_grid, genetic_patterns)
    print(f"   DNA grid: {len(dna_grid)}x{len(dna_grid[0])}")
    print(f"   Genetic patterns: {genetic_patterns}")
    print(f"   Found patterns: {found_patterns}")
    
    # Application 3: Crossword validation
    print("\n3. Crossword Validation:")
    crossword = [
        ['H', 'E', 'L', 'L', 'O'],
        ['A', 'B', 'C', 'D', 'E'],
        ['P', 'Y', 'T', 'H', 'O', 'N'],
        ['W', 'O', 'R', 'L', 'D']
    ]
    
    # Note: This is a simplified example
    dictionary_words = ["HELLO", "WORLD", "PYTHON", "CODE"]
    
    valid_words = solution.findWords1(crossword, dictionary_words)
    print(f"   Crossword grid: {len(crossword)}x{len(crossword[0])}")
    print(f"   Dictionary check: {dictionary_words}")
    print(f"   Valid words found: {valid_words}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty inputs
        ([], [], []),
        ([["a"]], [], []),
        ([], ["word"], []),
        
        # Single character
        ([["a"]], ["a"], ["a"]),
        ([["a"]], ["b"], []),
        
        # Same character repeated
        ([["a", "a"], ["a", "a"]], ["aa", "aaa", "aaaa"], ["aa", "aaa", "aaaa"]),
        
        # Words longer than possible paths
        ([["a", "b"]], ["abc"], []),
        
        # Duplicate words in input
        ([["a", "b"]], ["ab", "ab"], ["ab"]),
        
        # Case sensitivity
        ([["A", "b"]], ["Ab"], []),
        
        # Very long word
        ([["a"] * 10], ["a" * 15], []),
    ]
    
    for i, (board, words, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: Board={board}, Words={words}")
        try:
            if board:  # Only run if board is not empty
                board_copy = [row[:] for row in board]
                result = solution.findWords1(board_copy, words)
            else:
                result = []
            
            status = "✓" if sorted(result) == sorted(expected) else "✗"
            print(f"  Result: {result}, Expected: {expected} {status}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_trie_construction()
    demonstrate_search_process()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()

"""
212. Word Search II demonstrates advanced trie-based board search:

1. Trie + DFS - Classic approach with backtracking
2. Optimized Trie - Enhanced with node pruning for better performance
3. Character Frequency - Pre-filtering based on character availability
4. Reverse Strategy - Bidirectional search for optimization
5. BFS Iterative - Queue-based alternative to recursive DFS
6. Multi-threaded - Conceptual parallel processing approach

Each approach shows different optimization strategies for the complex
problem of finding multiple words in a 2D character grid efficiently.
"""
