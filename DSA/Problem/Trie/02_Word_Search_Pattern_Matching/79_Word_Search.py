"""
79. Word Search - Multiple Approaches
Difficulty: Medium

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells 
are horizontally or vertically neighboring. The same letter cell may not be used more than once.

LeetCode Problem: https://leetcode.com/problems/word-search/

Example:
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
"""

from typing import List, Set, Tuple, Optional
from collections import deque

class TrieNode:
    """Trie node for optimized word search"""
    def __init__(self):
        self.children = {}
        self.is_word = False

class Solution:
    
    def exist1(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 1: DFS with Backtracking
        
        Standard backtracking approach exploring all paths.
        
        Time: O(m*n*4^L) where m,n are board dimensions, L is word length
        Space: O(L) for recursion stack
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, index: int) -> bool:
            # Base case: found the word
            if index == len(word):
                return True
            
            # Check bounds and character match
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] != word[index]):
                return False
            
            # Mark as visited
            char = board[i][j]
            board[i][j] = '#'
            
            # Explore all 4 directions
            found = (dfs(i+1, j, index+1) or dfs(i-1, j, index+1) or
                    dfs(i, j+1, index+1) or dfs(i, j-1, index+1))
            
            # Backtrack
            board[i][j] = char
            
            return found
        
        # Try starting from each cell
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        
        return False
    
    def exist2(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 2: DFS with Visited Set
        
        Use explicit visited set instead of modifying board.
        
        Time: O(m*n*4^L)
        Space: O(L + visited_cells) for recursion and visited set
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, index: int, visited: Set[Tuple[int, int]]) -> bool:
            if index == len(word):
                return True
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                (i, j) in visited or board[i][j] != word[index]):
                return False
            
            visited.add((i, j))
            
            # Explore all 4 directions
            found = (dfs(i+1, j, index+1, visited) or dfs(i-1, j, index+1, visited) or
                    dfs(i, j+1, index+1, visited) or dfs(i, j-1, index+1, visited))
            
            visited.remove((i, j))
            
            return found
        
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0, set()):
                    return True
        
        return False
    
    def exist3(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 3: Optimized with Early Termination
        
        Add character frequency check for early termination.
        
        Time: O(m*n + m*n*4^L) with early termination
        Space: O(alphabet_size + L)
        """
        if not board or not board[0] or not word:
            return False
        
        # Count character frequencies in board
        board_chars = {}
        for row in board:
            for char in row:
                board_chars[char] = board_chars.get(char, 0) + 1
        
        # Check if word can be formed
        word_chars = {}
        for char in word:
            word_chars[char] = word_chars.get(char, 0) + 1
        
        for char, count in word_chars.items():
            if board_chars.get(char, 0) < count:
                return False
        
        # Optimization: start from the less frequent character end
        if word_chars[word[0]] > word_chars[word[-1]]:
            word = word[::-1]
        
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, index: int) -> bool:
            if index == len(word):
                return True
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] != word[index]):
                return False
            
            char = board[i][j]
            board[i][j] = '#'
            
            found = (dfs(i+1, j, index+1) or dfs(i-1, j, index+1) or
                    dfs(i, j+1, index+1) or dfs(i, j-1, index+1))
            
            board[i][j] = char
            return found
        
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        
        return False
    
    def exist4(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 4: Trie-based Search
        
        Build trie for the word and use it for search optimization.
        
        Time: O(L + m*n*4^L)
        Space: O(L) for trie
        """
        if not board or not board[0] or not word:
            return False
        
        # Build trie for the word
        root = TrieNode()
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, node: TrieNode) -> bool:
            if node.is_word:
                return True
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] not in node.children):
                return False
            
            char = board[i][j]
            board[i][j] = '#'
            
            found = (dfs(i+1, j, node.children[char]) or dfs(i-1, j, node.children[char]) or
                    dfs(i, j+1, node.children[char]) or dfs(i, j-1, node.children[char]))
            
            board[i][j] = char
            return found
        
        for i in range(m):
            for j in range(n):
                if board[i][j] in root.children:
                    if dfs(i, j, root):
                        return True
        
        return False
    
    def exist5(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 5: BFS with Queue
        
        Use BFS instead of DFS for word search.
        
        Time: O(m*n*4^L)
        Space: O(m*n*L) for queue in worst case
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        
        # Queue: (row, col, word_index, visited_set)
        for start_i in range(m):
            for start_j in range(n):
                if board[start_i][start_j] == word[0]:
                    queue = deque([(start_i, start_j, 0, {(start_i, start_j)})])
                    
                    while queue:
                        i, j, index, visited = queue.popleft()
                        
                        if index == len(word) - 1:
                            return True
                        
                        # Explore neighbors
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            
                            if (0 <= ni < m and 0 <= nj < n and 
                                (ni, nj) not in visited and
                                board[ni][nj] == word[index + 1]):
                                
                                new_visited = visited | {(ni, nj)}
                                queue.append((ni, nj, index + 1, new_visited))
        
        return False
    
    def exist6(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 6: Bidirectional Search
        
        Search from both ends of the word simultaneously.
        
        Time: O(m*n*4^(L/2)) in best case
        Space: O(L)
        """
        if not board or not board[0] or not word:
            return False
        
        if len(word) == 1:
            for row in board:
                if word in row:
                    return True
            return False
        
        m, n = len(board), len(board[0])
        
        def dfs_forward(i: int, j: int, index: int, visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            """DFS from start, return all reachable positions at mid-point"""
            if index == len(word) // 2:
                return {(i, j)}
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                (i, j) in visited or board[i][j] != word[index]):
                return set()
            
            visited.add((i, j))
            reachable = set()
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                reachable.update(dfs_forward(ni, nj, index + 1, visited))
            
            visited.remove((i, j))
            return reachable
        
        def dfs_backward(i: int, j: int, index: int, visited: Set[Tuple[int, int]], 
                        forward_positions: Set[Tuple[int, int]]) -> bool:
            """DFS from end, check if we can meet forward search"""
            if index == len(word) // 2:
                return (i, j) in forward_positions
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                (i, j) in visited or board[i][j] != word[index]):
                return False
            
            visited.add((i, j))
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if dfs_backward(ni, nj, index - 1, visited, forward_positions):
                    visited.remove((i, j))
                    return True
            
            visited.remove((i, j))
            return False
        
        # Find all possible mid-points from forward search
        forward_positions = set()
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    forward_positions.update(dfs_forward(i, j, 0, set()))
        
        if not forward_positions:
            return False
        
        # Search backward from all possible end positions
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[-1]:
                    if dfs_backward(i, j, len(word) - 1, set(), forward_positions):
                        return True
        
        return False


def test_basic_cases():
    """Test basic word search functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode example 1
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED", True),
        
        # LeetCode example 2
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE", True),
        
        # LeetCode example 3
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB", False),
        
        # Single character
        ([["A"]], "A", True),
        ([["A"]], "B", False),
        
        # Path requires backtracking
        ([["A","B"],["C","D"]], "ABDC", True),
        
        # No valid path
        ([["A","B"],["C","D"]], "ABCD", False),
        
        # Longer word
        ([["C","A","A"],["A","A","A"],["B","C","D"]], "AAB", True),
    ]
    
    approaches = [
        ("DFS Backtrack", solution.exist1),
        ("DFS + Visited", solution.exist2),
        ("Optimized DFS", solution.exist3),
        ("Trie-based", solution.exist4),
        ("BFS", solution.exist5),
        ("Bidirectional", solution.exist6),
    ]
    
    for i, (board, word, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Board={board}, Word='{word}'")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                # Make copy of board since some methods modify it
                board_copy = [row[:] for row in board]
                result = method(board_copy, word)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_search_process():
    """Demonstrate the search process step by step"""
    print("\n=== Search Process Demo ===")
    
    board = [["A","B","C"],["D","E","F"],["G","H","I"]]
    word = "ABEHI"
    
    print(f"Board:")
    for i, row in enumerate(board):
        print(f"  {i}: {row}")
    
    print(f"\nSearching for word: '{word}'")
    
    m, n = len(board), len(board[0])
    
    def dfs_demo(i: int, j: int, index: int, path: List[Tuple[int, int]]) -> bool:
        print(f"  Step {len(path)}: at ({i},{j}) = '{board[i][j]}', looking for '{word[index]}', path: {path}")
        
        # Check bounds and character match
        if (i < 0 or i >= m or j < 0 or j >= n or 
            (i, j) in path or board[i][j] != word[index]):
            print(f"    Invalid: out of bounds, visited, or character mismatch")
            return False
        
        new_path = path + [(i, j)]
        
        # Found complete word
        if index == len(word) - 1:
            print(f"    SUCCESS! Found complete word with path: {new_path}")
            return True
        
        # Explore all 4 directions
        directions = [(0, 1, "right"), (1, 0, "down"), (0, -1, "left"), (-1, 0, "up")]
        
        for di, dj, direction in directions:
            ni, nj = i + di, j + dj
            print(f"    Trying {direction} to ({ni},{nj})")
            
            if dfs_demo(ni, nj, index + 1, new_path):
                return True
        
        print(f"    Backtracking from ({i},{j})")
        return False
    
    # Start search from position (0,0)
    print(f"\nStarting search from (0,0):")
    found = dfs_demo(0, 0, 0, [])
    print(f"\nResult: {'Found' if found else 'Not found'}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_board(m: int, n: int) -> List[List[str]]:
        return [[random.choice(string.ascii_uppercase[:10]) for _ in range(n)] for _ in range(m)]
    
    def generate_word(board: List[List[str]], length: int) -> str:
        # Generate word that might exist in board
        chars = []
        for row in board:
            chars.extend(row)
        return ''.join(random.choices(chars, k=length))
    
    test_scenarios = [
        ("Small", generate_board(3, 3), 4),
        ("Medium", generate_board(5, 5), 6),
        ("Large", generate_board(6, 6), 8),
    ]
    
    approaches = [
        ("DFS Backtrack", solution.exist1),
        ("Optimized DFS", solution.exist3),
        ("Trie-based", solution.exist4),
    ]
    
    for scenario_name, board, word_length in test_scenarios:
        word = generate_word(board, word_length)
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Board: {len(board)}x{len(board[0])}, Word: '{word}' (length {len(word)})")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(5):
                board_copy = [row[:] for row in board]
                result = method(board_copy, word)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms")


def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques"""
    print("\n=== Optimization Techniques Demo ===")
    
    # Technique 1: Character frequency check
    print("1. Character Frequency Optimization:")
    
    board = [["A","B","C"],["D","E","F"],["G","H","I"]]
    word = "XYZ"  # Doesn't exist in board
    
    # Count characters in board
    board_chars = {}
    for row in board:
        for char in row:
            board_chars[char] = board_chars.get(char, 0) + 1
    
    print(f"   Board characters: {board_chars}")
    print(f"   Word: '{word}'")
    
    # Check if word can be formed
    can_form = True
    for char in word:
        if char not in board_chars:
            can_form = False
            print(f"   Character '{char}' not in board - early termination")
            break
    
    print(f"   Can form word: {can_form}")
    
    # Technique 2: Starting from less frequent character
    print(f"\n2. Frequency-based Starting Point:")
    
    word2 = "AEI"
    word_chars = {}
    for char in word2:
        word_chars[char] = word_chars.get(char, 0) + 1
    
    print(f"   Word: '{word2}'")
    print(f"   First char '{word2[0]}' frequency in board: {board_chars.get(word2[0], 0)}")
    print(f"   Last char '{word2[-1]}' frequency in board: {board_chars.get(word2[-1], 0)}")
    
    if board_chars.get(word2[0], 0) > board_chars.get(word2[-1], 0):
        print(f"   Optimization: Start search from last character (less frequent)")
    else:
        print(f"   Start search from first character")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Word puzzle solver
    print("1. Word Puzzle Game:")
    
    puzzle = [
        ['C', 'A', 'T', 'S'],
        ['O', 'R', 'D', 'O'],
        ['G', 'E', 'A', 'M'],
        ['S', 'U', 'N', 'K']
    ]
    
    words_to_find = ["CAT", "DOG", "SUN", "GAME", "READ"]
    
    print(f"   Puzzle Grid:")
    for row in puzzle:
        print(f"     {row}")
    
    print(f"   Words to find: {words_to_find}")
    
    found_words = []
    for word in words_to_find:
        puzzle_copy = [row[:] for row in puzzle]
        if solution.exist1(puzzle_copy, word):
            found_words.append(word)
    
    print(f"   Found words: {found_words}")
    
    # Application 2: DNA sequence search
    print(f"\n2. DNA Sequence Analysis:")
    
    dna_grid = [
        ['A', 'T', 'G', 'C'],
        ['T', 'A', 'C', 'G'],
        ['G', 'C', 'A', 'T'],
        ['C', 'G', 'T', 'A']
    ]
    
    sequences = ["ATGC", "TACG", "CGTA", "INVALID"]
    
    print(f"   DNA Grid:")
    for row in dna_grid:
        print(f"     {row}")
    
    print(f"   Sequences to find: {sequences}")
    
    for sequence in sequences:
        dna_copy = [row[:] for row in dna_grid]
        found = solution.exist1(dna_copy, sequence)
        print(f"     '{sequence}': {'Found' if found else 'Not found'}")
    
    # Application 3: Maze pathfinding (character-based)
    print(f"\n3. Character-based Pathfinding:")
    
    maze = [
        ['S', 'P', 'P', 'E'],
        ['W', 'P', 'W', 'P'],
        ['W', 'P', 'P', 'P'],
        ['W', 'W', 'W', 'G']
    ]
    
    # S=Start, P=Path, W=Wall, E=Exit, G=Goal
    path_pattern = "SPPPPG"  # Valid path pattern
    
    print(f"   Maze (S=Start, P=Path, W=Wall, G=Goal):")
    for row in maze:
        print(f"     {row}")
    
    maze_copy = [row[:] for row in maze]
    path_exists = solution.exist1(maze_copy, path_pattern)
    print(f"   Path pattern '{path_pattern}': {'Valid' if path_exists else 'Invalid'}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty inputs
        ([], "", False),
        ([["A"]], "", False),
        ([], "A", False),
        
        # Single cell cases
        ([["A"]], "A", True),
        ([["A"]], "B", False),
        ([["A"]], "AA", False),
        
        # Word longer than board can accommodate
        ([["A", "B"]], "ABCD", False),
        
        # All same character
        ([["A", "A"], ["A", "A"]], "AAA", True),
        ([["A", "A"], ["A", "A"]], "AAAAA", False),
        
        # Spiral pattern
        ([["A", "B", "C"], ["H", "I", "D"], ["G", "F", "E"]], "ABCDEFGHI", True),
        
        # Requires complex backtracking
        ([["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]], "ABCFIHGED", True),
    ]
    
    for i, (board, word, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: Board={board}, Word='{word}'")
        try:
            if board and board[0]:  # Only test if board is valid
                board_copy = [row[:] for row in board]
                result = solution.exist1(board_copy, word)
            else:
                result = False
                
            status = "✓" if result == expected else "✗"
            print(f"  Result: {result}, Expected: {expected} {status}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_search_process()
    benchmark_approaches()
    demonstrate_optimization_techniques()
    demonstrate_real_world_applications()
    test_edge_cases()

"""
79. Word Search demonstrates multiple approaches for grid-based word finding:

1. DFS with Backtracking - Classic recursive approach with board modification
2. DFS with Visited Set - Explicit tracking without modifying input
3. Optimized DFS - Character frequency analysis and direction optimization
4. Trie-based Search - Structured approach for multiple word searches
5. BFS with Queue - Breadth-first alternative with state tracking
6. Bidirectional Search - Advanced optimization for longer words

Each approach shows different optimization strategies for the fundamental
problem of finding sequential character paths in a 2D grid.
"""
