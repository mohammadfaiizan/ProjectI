"""
212. Word Search II
Difficulty: Hard

Problem:
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent 
cells are horizontally or vertically neighboring. The same letter cell may not be used 
more than once in a word.

Examples:
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], 
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []

Constraints:
- m == board.length
- n == board[i].length
- 1 <= m, n <= 12
- board[i][j] is a lowercase English letter
- 1 <= words.length <= 3 * 10^4
- 1 <= words[i].length <= 10
- words[i] consists of lowercase English letters
- All the values of words are unique
"""

from typing import List, Set

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # Store complete word here when it's a valid end

class Solution:
    def findWords_approach1_trie_dfs(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 1: Trie + DFS (Optimal for Multiple Words)
        
        Build trie from words, then use single DFS to find all words simultaneously.
        This avoids redundant DFS calls for words with common prefixes.
        
        Time: O(M*N*4^L + W*L) where W = number of words, L = max word length
        Space: O(W*L) for trie + O(L) for recursion
        """
        if not board or not board[0] or not words:
            return []
        
        # Build Trie
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
        
        def dfs(i, j, node):
            """DFS with trie traversal"""
            # Get current character
            char = board[i][j]
            
            # Check if character exists in trie
            if char not in node.children:
                return
            
            # Move to next trie node
            node = node.children[char]
            
            # If we found a complete word, add to result
            if node.word:
                result.append(node.word)
                node.word = None  # Avoid duplicates
            
            # Mark current cell as visited
            board[i][j] = '#'
            
            # Explore 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#'):
                    dfs(ni, nj, node)
            
            # Backtrack: restore original character
            board[i][j] = char
        
        # Start DFS from each cell
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        
        return result
    
    def findWords_approach2_optimized_trie_pruning(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 2: Optimized Trie with Pruning
        
        Add optimizations like removing exhausted branches and early termination.
        
        Time: O(M*N*4^L + W*L) - better average case
        Space: O(W*L)
        """
        if not board or not board[0] or not words:
            return []
        
        # Build Trie with pruning capability
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
        
        def dfs(i, j, parent, node):
            """DFS with trie pruning"""
            char = board[i][j]
            
            if char not in node.children:
                return
            
            node = node.children[char]
            
            # Found a word
            if node.word:
                result.append(node.word)
                node.word = None  # Remove to avoid duplicates
            
            # Mark as visited
            board[i][j] = '#'
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#'):
                    dfs(ni, nj, node, node)
            
            # Backtrack
            board[i][j] = char
            
            # Optimization: Remove leaf nodes to prune trie
            if not node.children and not node.word:
                del parent.children[char]
        
        # Try each starting position
        for i in range(m):
            for j in range(n):
                if board[i][j] in root.children:
                    dfs(i, j, root, root)
        
        return result
    
    def findWords_approach3_naive_multiple_search(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 3: Naive Multiple Word Search (For Comparison)
        
        Search each word individually using Word Search I algorithm.
        Less efficient but simpler to understand.
        
        Time: O(W*M*N*4^L) - much slower for multiple words
        Space: O(L) for each search
        """
        if not board or not board[0] or not words:
            return []
        
        def exist(word):
            """Standard word search for single word"""
            m, n = len(board), len(board[0])
            
            def dfs(i, j, index):
                if index == len(word):
                    return True
                
                if (i < 0 or i >= m or j < 0 or j >= n or 
                    board[i][j] != word[index] or board[i][j] == '#'):
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
        
        result = []
        for word in words:
            if exist(word):
                result.append(word)
        
        return result
    
    def findWords_approach4_iterative_trie_dfs(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Approach 4: Iterative Trie DFS
        
        Use explicit stack to avoid recursion limits.
        
        Time: O(M*N*4^L + W*L)
        Space: O(W*L + M*N*L) - trie + stack
        """
        if not board or not board[0] or not words:
            return []
        
        # Build Trie
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
        
        # Use iterative DFS with stack
        for i in range(m):
            for j in range(n):
                if board[i][j] in root.children:
                    # Stack: (row, col, trie_node, visited_set)
                    stack = [(i, j, root.children[board[i][j]], {(i, j)})]
                    
                    while stack:
                        r, c, node, visited = stack.pop()
                        
                        # Check if we found a word
                        if node.word:
                            result.append(node.word)
                            node.word = None  # Avoid duplicates
                        
                        # Explore neighbors
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            
                            if (0 <= nr < m and 0 <= nc < n and 
                                (nr, nc) not in visited and
                                board[nr][nc] in node.children):
                                
                                new_visited = visited.copy()
                                new_visited.add((nr, nc))
                                
                                stack.append((nr, nc, node.children[board[nr][nc]], new_visited))
        
        return result

def test_word_search_ii():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (board, words, expected)
        ([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], 
         ["oath","pea","eat","rain"], ["eat","oath"]),
        ([["a","b"],["c","d"]], ["abcb"], []),
        ([["a","a"]], ["aa"], ["aa"]),
        ([["a"]], ["a"], ["a"]),
        ([["a","b","c"],["a","e","d"],["a","f","g"]], ["abcdefg","gfedcbaaa","eaabcdgfa","befa","dgc","ade"], ["abcdefg","befa","eaabcdgfa","gfedcbaaa"]),
    ]
    
    approaches = [
        ("Trie + DFS", solution.findWords_approach1_trie_dfs),
        ("Optimized Trie Pruning", solution.findWords_approach2_optimized_trie_pruning),
        ("Naive Multiple Search", solution.findWords_approach3_naive_multiple_search),
        ("Iterative Trie DFS", solution.findWords_approach4_iterative_trie_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (board, words, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_board = [row[:] for row in board]
            result = func(test_board, words)
            result_sorted = sorted(result)
            expected_sorted = sorted(expected)
            status = "✓" if result_sorted == expected_sorted else "✗"
            print(f"Test {i+1}: {status}")
            print(f"         Words: {words}")
            print(f"         Expected: {expected_sorted}")
            print(f"         Got: {result_sorted}")

def demonstrate_trie_optimization():
    """Demonstrate why Trie optimization is crucial"""
    print("\n=== Trie Optimization Demonstration ===")
    
    board = [["o","a","a","n"],
             ["e","t","a","e"],
             ["i","h","k","r"],
             ["i","f","l","v"]]
    
    words = ["oath", "oat", "oats", "eat", "tea", "tan", "ate", "nat"]
    
    print("Board:")
    for i, row in enumerate(board):
        print(f"  Row {i}: {row}")
    
    print(f"\nWords to search: {words}")
    
    # Build and visualize trie
    print(f"\nTrie structure:")
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    def print_trie(node, prefix="", depth=0):
        indent = "  " * depth
        if node.word:
            print(f"{indent}{prefix} -> WORD: {node.word}")
        
        for char, child in sorted(node.children.items()):
            print(f"{indent}{prefix}{char}")
            print_trie(child, prefix + char, depth + 1)
    
    print_trie(root)
    
    print(f"\nWhy Trie is efficient:")
    print("1. Words with common prefixes share trie paths")
    print("2. Single DFS can find multiple words simultaneously")
    print("3. Pruning eliminates unnecessary branches")
    print("4. No redundant searches for similar words")

if __name__ == "__main__":
    test_word_search_ii()
    demonstrate_trie_optimization()

"""
Graph Theory Concepts:
1. Multi-target Path Finding
2. Trie-based State Space Reduction
3. Efficient Backtracking with Shared Prefixes
4. Advanced DFS Optimization

Key Optimization Insights:
- Trie eliminates redundant prefix exploration
- Single DFS can find multiple words with shared prefixes
- Pruning removes exhausted branches dynamically
- Backtracking works seamlessly with trie traversal

Algorithm Complexity Analysis:
- Naive approach: O(W*M*N*4^L) - search each word separately
- Trie approach: O(M*N*4^L + W*L) - shared prefix optimization
- Space: O(W*L) for trie vs O(L) for single word search

Advanced Techniques:
- Dynamic trie pruning during search
- Word deduplication at trie level
- Iterative implementation for large inputs
- Memory-efficient visited tracking

Real-world Applications:
- Word game solvers (Boggle, word puzzles)
- Text pattern matching in grids
- Multi-pattern search algorithms
- Autocomplete systems with grid constraints
- Natural language processing on structured data

This problem showcases the power of combining data structures (Trie)
with graph algorithms (DFS) for optimal multi-target search.
"""

