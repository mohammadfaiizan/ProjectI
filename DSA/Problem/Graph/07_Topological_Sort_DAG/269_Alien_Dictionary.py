"""
269. Alien Dictionary - Multiple Approaches
Difficulty: Medium

There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.

A string s is lexicographically smaller than a string t if at the first letter where they differ, the letter in s comes before the letter in t in the alien language. If the first min(s.length, t.length) letters are the same, then s is lexicographically smaller if and only if s.length < t.length.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque

class AlienDictionary:
    """Multiple approaches to determine alien dictionary order"""
    
    def alienOrder_kahn_algorithm(self, words: List[str]) -> str:
        """
        Approach 1: Kahn's Algorithm (BFS Topological Sort)
        
        Build dependency graph and use BFS topological sort.
        
        Time: O(C) where C is total characters, Space: O(1) for English alphabet
        """
        # Step 1: Initialize graph and in-degree count
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Get all unique characters
        chars = set()
        for word in words:
            chars.update(word)
        
        # Initialize in-degree for all characters
        for char in chars:
            in_degree[char] = 0
        
        # Step 2: Build the graph by comparing adjacent words
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            
            # Check for invalid case: longer word is prefix of shorter word
            if len(word1) > len(word2) and word1.startswith(word2):
                return ""
            
            # Find first differing character
            for j in range(min(len(word1), len(word2))):
                if word1[j] != word2[j]:
                    # word1[j] comes before word2[j] in alien order
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                    break
        
        # Step 3: Kahn's algorithm
        queue = deque()
        for char in chars:
            if in_degree[char] == 0:
                queue.append(char)
        
        result = []
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all characters are processed (no cycle)
        return ''.join(result) if len(result) == len(chars) else ""
    
    def alienOrder_dfs_topological(self, words: List[str]) -> str:
        """
        Approach 2: DFS-based Topological Sort
        
        Use DFS with cycle detection for topological sorting.
        
        Time: O(C), Space: O(1)
        """
        # Build adjacency list
        graph = defaultdict(set)
        chars = set()
        
        for word in words:
            chars.update(word)
        
        # Build graph from word comparisons
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            
            # Invalid case check
            if len(word1) > len(word2) and word1.startswith(word2):
                return ""
            
            for j in range(min(len(word1), len(word2))):
                if word1[j] != word2[j]:
                    graph[word1[j]].add(word2[j])
                    break
        
        # DFS with cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {char: WHITE for char in chars}
        result = []
        
        def dfs(char: str) -> bool:
            if color[char] == GRAY:  # Cycle detected
                return False
            if color[char] == BLACK:  # Already processed
                return True
            
            color[char] = GRAY
            
            for neighbor in graph[char]:
                if not dfs(neighbor):
                    return False
            
            color[char] = BLACK
            result.append(char)
            return True
        
        # Run DFS from all unvisited characters
        for char in chars:
            if color[char] == WHITE:
                if not dfs(char):
                    return ""  # Cycle detected
        
        # Reverse to get correct topological order
        return ''.join(reversed(result))
    
    def alienOrder_iterative_dfs(self, words: List[str]) -> str:
        """
        Approach 3: Iterative DFS with Stack
        
        Use iterative DFS to avoid recursion depth issues.
        
        Time: O(C), Space: O(C)
        """
        # Build graph
        graph = defaultdict(set)
        chars = set()
        
        for word in words:
            chars.update(word)
        
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            
            if len(word1) > len(word2) and word1.startswith(word2):
                return ""
            
            for j in range(min(len(word1), len(word2))):
                if word1[j] != word2[j]:
                    graph[word1[j]].add(word2[j])
                    break
        
        # Iterative DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {char: WHITE for char in chars}
        result = []
        
        for start_char in chars:
            if color[start_char] != WHITE:
                continue
            
            stack = [start_char]
            path = []
            
            while stack:
                char = stack[-1]
                
                if color[char] == WHITE:
                    color[char] = GRAY
                    path.append(char)
                    
                    # Add neighbors to stack
                    for neighbor in graph[char]:
                        if color[neighbor] == GRAY:  # Cycle detected
                            return ""
                        if color[neighbor] == WHITE:
                            stack.append(neighbor)
                
                elif color[char] == GRAY:
                    # Finished processing this character
                    color[char] = BLACK
                    result.append(char)
                    stack.pop()
                    if path and path[-1] == char:
                        path.pop()
                
                else:  # BLACK
                    stack.pop()
        
        return ''.join(reversed(result))
    
    def alienOrder_constraint_satisfaction(self, words: List[str]) -> str:
        """
        Approach 4: Constraint Satisfaction Approach
        
        Model as constraint satisfaction problem.
        
        Time: O(C), Space: O(C)
        """
        # Collect all characters and constraints
        chars = set()
        constraints = []  # (before, after) pairs
        
        for word in words:
            chars.update(word)
        
        # Extract ordering constraints
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            
            if len(word1) > len(word2) and word1.startswith(word2):
                return ""
            
            for j in range(min(len(word1), len(word2))):
                if word1[j] != word2[j]:
                    constraints.append((word1[j], word2[j]))
                    break
        
        # Build graph from constraints
        graph = defaultdict(set)
        in_degree = {char: 0 for char in chars}
        
        for before, after in constraints:
            if after not in graph[before]:
                graph[before].add(after)
                in_degree[after] += 1
        
        # Topological sort
        queue = deque()
        for char in chars:
            if in_degree[char] == 0:
                queue.append(char)
        
        result = []
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return ''.join(result) if len(result) == len(chars) else ""
    
    def alienOrder_optimized_comparison(self, words: List[str]) -> str:
        """
        Approach 5: Optimized Word Comparison
        
        Optimize the word comparison process.
        
        Time: O(C), Space: O(1)
        """
        if not words:
            return ""
        
        # Step 1: Get all unique characters
        chars = set()
        for word in words:
            chars.update(word)
        
        # Step 2: Build graph with optimized comparison
        graph = defaultdict(set)
        in_degree = {char: 0 for char in chars}
        
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            min_len = min(len(word1), len(word2))
            
            # Find first difference
            diff_found = False
            for j in range(min_len):
                if word1[j] != word2[j]:
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                    diff_found = True
                    break
            
            # Check invalid case
            if not diff_found and len(word1) > len(word2):
                return ""
        
        # Step 3: Topological sort with early termination
        queue = deque()
        for char in chars:
            if in_degree[char] == 0:
                queue.append(char)
        
        result = []
        while queue:
            if len(queue) > 1:
                # Multiple valid orderings possible - just pick one
                pass
            
            char = queue.popleft()
            result.append(char)
            
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return ''.join(result) if len(result) == len(chars) else ""

def test_alien_dictionary():
    """Test alien dictionary algorithms"""
    solver = AlienDictionary()
    
    test_cases = [
        (["wrt","wrf","er","ett","rftt"], "wertf", "Example 1"),
        (["z","x"], "zx", "Simple two character order"),
        (["z","x","z"], "", "Invalid - contradiction"),
        (["abc","ab"], "", "Invalid - longer word prefix"),
        (["ac","ab","zc","zb"], "acbz", "Multiple constraints"),
    ]
    
    algorithms = [
        ("Kahn's Algorithm", solver.alienOrder_kahn_algorithm),
        ("DFS Topological", solver.alienOrder_dfs_topological),
        ("Iterative DFS", solver.alienOrder_iterative_dfs),
        ("Constraint Satisfaction", solver.alienOrder_constraint_satisfaction),
        ("Optimized Comparison", solver.alienOrder_optimized_comparison),
    ]
    
    print("=== Testing Alien Dictionary ===")
    
    for words, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Words: {words}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(words)
                # For valid results, check if it's a valid topological order
                is_valid = result == expected or (result != "" and expected != "" and len(result) == len(expected))
                status = "✓" if (result == expected or (result == "" and expected == "")) else ("?" if is_valid else "✗")
                print(f"{alg_name:22} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_alien_dictionary()

"""
Alien Dictionary demonstrates advanced topological sorting
for lexicographic ordering problems with constraint extraction
and cycle detection in character dependency graphs.
"""
