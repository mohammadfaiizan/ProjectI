"""
79. Word Search
Difficulty: Medium

Problem:
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells 
are horizontally or vertically neighboring. The same letter cell may not be used more than once.

Examples:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false

Constraints:
- m == board.length
- n == board[i].length
- 1 <= m, n <= 6
- 1 <= word.length <= 15
- board and word consists of only lowercase and uppercase English letters
"""

from typing import List

class Solution:
    def exist_approach1_dfs_backtracking(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 1: DFS with Backtracking (Classic)
        
        Try starting from each cell and use DFS with backtracking to find word.
        Mark visited cells and unmark when backtracking.
        
        Time: O(M*N*4^L) where L = len(word)
        Space: O(L) - recursion stack depth
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        
        def dfs(i, j, index):
            """DFS to match word starting from position (i,j) at word[index]"""
            # Base case: found complete word
            if index == len(word):
                return True
            
            # Boundary checks and character match
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] != word[index] or board[i][j] == '#'):
                return False
            
            # Mark current cell as visited
            temp = board[i][j]
            board[i][j] = '#'
            
            # Try all 4 directions
            found = (dfs(i + 1, j, index + 1) or
                    dfs(i - 1, j, index + 1) or
                    dfs(i, j + 1, index + 1) or
                    dfs(i, j - 1, index + 1))
            
            # Backtrack: restore original character
            board[i][j] = temp
            
            return found
        
        # Try starting from each cell
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        
        return False
    
    def exist_approach2_visited_set(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 2: DFS with Visited Set (Non-destructive)
        
        Use separate visited set instead of modifying board.
        
        Time: O(M*N*4^L)
        Space: O(L) - recursion stack + visited set
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        
        def dfs(i, j, index, visited):
            if index == len(word):
                return True
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                (i, j) in visited or board[i][j] != word[index]):
                return False
            
            visited.add((i, j))
            
            # Try all 4 directions
            found = (dfs(i + 1, j, index + 1, visited) or
                    dfs(i - 1, j, index + 1, visited) or
                    dfs(i, j + 1, index + 1, visited) or
                    dfs(i, j - 1, index + 1, visited))
            
            visited.remove((i, j))  # Backtrack
            
            return found
        
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0, set()):
                    return True
        
        return False
    
    def exist_approach3_optimized_pruning(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 3: Optimized with Early Pruning
        
        Add optimizations like character frequency check and early termination.
        
        Time: O(M*N*4^L) - better average case
        Space: O(L)
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        
        # Optimization: Check if all characters in word exist in board
        from collections import Counter
        board_chars = Counter()
        for row in board:
            for char in row:
                board_chars[char] += 1
        
        word_chars = Counter(word)
        for char, count in word_chars.items():
            if board_chars[char] < count:
                return False
        
        # Optimization: Try both directions of word (sometimes reverse is faster)
        def search_word(target_word):
            def dfs(i, j, index):
                if index == len(target_word):
                    return True
                
                if (i < 0 or i >= m or j < 0 or j >= n or 
                    board[i][j] != target_word[index] or board[i][j] == '#'):
                    return False
                
                temp = board[i][j]
                board[i][j] = '#'
                
                found = (dfs(i + 1, j, index + 1) or
                        dfs(i - 1, j, index + 1) or
                        dfs(i, j + 1, index + 1) or
                        dfs(i, j - 1, index + 1))
                
                board[i][j] = temp
                return found
            
            for i in range(m):
                for j in range(n):
                    if dfs(i, j, 0):
                        return True
            return False
        
        # Try both forward and reverse (sometimes reverse is faster)
        return search_word(word) or search_word(word[::-1])
    
    def exist_approach4_iterative_dfs(self, board: List[List[str]], word: str) -> bool:
        """
        Approach 4: Iterative DFS to avoid recursion
        
        Use explicit stack for DFS implementation.
        
        Time: O(M*N*4^L)
        Space: O(L) - stack size
        """
        if not board or not board[0] or not word:
            return False
        
        m, n = len(board), len(board[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for start_i in range(m):
            for start_j in range(n):
                if board[start_i][start_j] != word[0]:
                    continue
                
                # Stack: (i, j, index, path_set)
                stack = [(start_i, start_j, 0, {(start_i, start_j)})]
                
                while stack:
                    i, j, index, visited = stack.pop()
                    
                    if index == len(word) - 1:
                        return True
                    
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        
                        if (0 <= ni < m and 0 <= nj < n and 
                            (ni, nj) not in visited and 
                            board[ni][nj] == word[index + 1]):
                            
                            new_visited = visited.copy()
                            new_visited.add((ni, nj))
                            stack.append((ni, nj, index + 1, new_visited))
        
        return False

def test_word_search():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (board, word, expected)
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED", True),
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE", True),
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB", False),
        ([["A"]], "A", True),
        ([["A"]], "B", False),
        ([["A","B"],["C","D"]], "ACDB", True),
        ([["A","B"],["C","D"]], "ABDC", True),
        ([["A","B"],["C","D"]], "AB", True),
    ]
    
    approaches = [
        ("DFS Backtracking", solution.exist_approach1_dfs_backtracking),
        ("Visited Set", solution.exist_approach2_visited_set),
        ("Optimized Pruning", solution.exist_approach3_optimized_pruning),
        ("Iterative DFS", solution.exist_approach4_iterative_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (board, word, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_board = [row[:] for row in board]
            result = func(test_board, word)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Board: {board}, Word: '{word}', Expected: {expected}, Got: {result}")

def demonstrate_word_search():
    """Demonstrate word search process"""
    print("\n=== Word Search Process Demo ===")
    
    board = [["A","B","C","E"],
             ["S","F","C","S"],
             ["A","D","E","E"]]
    word = "ABCCED"
    
    print("Board:")
    for i, row in enumerate(board):
        print(f"  Row {i}: {row}")
    
    print(f"\nSearching for word: '{word}'")
    
    m, n = len(board), len(board[0])
    
    def dfs_trace(i, j, index, path):
        """DFS with path tracking for demonstration"""
        if index == len(word):
            print(f"✓ Found complete word! Path: {path}")
            return True
        
        if (i < 0 or i >= m or j < 0 or j >= n or 
            board[i][j] != word[index] or (i, j) in [p[:2] for p in path]):
            return False
        
        path.append((i, j, board[i][j]))
        print(f"  Step {len(path)}: ({i},{j}) = '{board[i][j]}' matches '{word[index]}'")
        
        # Try all 4 directions
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if dfs_trace(i + di, j + dj, index + 1, path[:]):
                return True
        
        return False
    
    # Try starting from each cell
    found = False
    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0]:
                print(f"\nTrying starting position ({i},{j}) = '{board[i][j]}'")
                if dfs_trace(i, j, 0, []):
                    found = True
                    break
        if found:
            break
    
    if not found:
        print("Word not found in board")

def visualize_search_paths():
    """Visualize different search paths"""
    print("\n=== Search Path Visualization ===")
    
    board = [["A","B","C"],
             ["D","E","F"],
             ["G","H","I"]]
    
    print("Board:")
    for i, row in enumerate(board):
        print(f"  Row {i}: {row}")
    
    # Show different possible paths
    words_to_try = ["AEI", "ABC", "ADGBEH", "AEIDCBA"]
    
    for word in words_to_try:
        print(f"\nSearching for '{word}':")
        
        # Simple existence check
        def can_form_word():
            m, n = len(board), len(board[0])
            
            def dfs(i, j, index, visited):
                if index == len(word):
                    return True
                
                if (i < 0 or i >= m or j < 0 or j >= n or 
                    (i, j) in visited or board[i][j] != word[index]):
                    return False
                
                visited.add((i, j))
                found = any(dfs(i + di, j + dj, index + 1, visited) 
                           for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)])
                visited.remove((i, j))
                
                return found
            
            for i in range(m):
                for j in range(n):
                    if dfs(i, j, 0, set()):
                        return True
            return False
        
        result = can_form_word()
        print(f"  Result: {'Found' if result else 'Not found'}")

if __name__ == "__main__":
    test_word_search()
    demonstrate_word_search()
    visualize_search_paths()

"""
Graph Theory Concepts:
1. Path Finding with Constraints
2. DFS with Backtracking
3. Grid Traversal with Visited Tracking
4. Exhaustive Search with Pruning

Key Word Search Concepts:
- Path constraint: Must use each cell at most once
- Sequential matching: Characters must match in exact order
- Backtracking: Undo choices when dead end reached
- Multiple starting points: Try from every valid starting cell

Algorithm Techniques:
- DFS with backtracking: Natural recursive approach
- Visited tracking: Prevent cycles and reuse
- Early pruning: Character frequency optimization
- Path reconstruction: Track successful paths

Optimization Strategies:
- Character frequency check before search
- Bidirectional search (forward/reverse word)
- Early termination on first success
- Iterative implementation for large inputs

Real-world Applications:
- Word games (Boggle, word puzzles)
- Pattern matching in grids
- Path finding with constraints
- Sequence validation in 2D data
- Game AI for word-based games
"""

