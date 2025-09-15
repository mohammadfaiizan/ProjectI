"""
Problem: Count Number of Possible Root Nodes
URL: https://leetcode.com/problems/count-number-of-possible-root-nodes/

Problem Statement:
Alice has an undirected tree consisting of n nodes labeled from 0 to n - 1. The tree is represented as a 2D integer array edges of length n - 1 where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.
Alice wants Bob to find the root of the tree. She allows Bob to make several guesses about the edges of the tree.
Bob's guesses are represented as a 2D integer array guesses where guesses[i] = [ui, vi] indicates that Bob guesses the directed edge from node ui to node vi exists in the rooted tree.
Alice allows Bob to keep a guess if the corresponding edge exists in the rooted tree.
Alice wants to minimize the number of guesses that Bob got correct.
Return the number of possible root nodes. If there is a tree rooted at node r such that Bob makes at most k correct guesses, then node r is a possible root.

Sample Input/Output:
Input: edges = [[0,1],[1,2],[1,3],[4,2]], guesses = [[1,3],[0,1],[1,0],[2,4]], k = 3
Output: 3
Explanation: Root the tree at node 0: Bob guesses [1,3], [0,1], [2,4] correctly.
Root the tree at node 1: Bob guesses [1,3], [1,0], [2,4] correctly.
Root the tree at node 2: Bob guesses [1,3], [1,0], [2,4] correctly.

Input: edges = [[0,1],[1,2],[2,3],[3,4]], guesses = [[1,0],[3,4],[2,1],[3,2]], k = 1
Output: 2
Explanation: Root the tree at node 0: Bob guesses [3,4] correctly.
Root the tree at node 4: Bob guesses [1,0] correctly.
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict

class Solution:
    def Root_Count_Brute_Force(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        Brute Force - Try each node as root and count correct guesses
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(edges) + 1
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        guess_set = set((u, v) for u, v in guesses)
        
        def Count_Correct_Guesses(root: int) -> int:
            visited = set()
            correct = 0
            
            def DFS(node: int, parent: int) -> None:
                nonlocal correct
                visited.add(node)
                
                if parent != -1 and (parent, node) in guess_set:
                    correct += 1
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        DFS(neighbor, node)
            
            DFS(root, -1)
            return correct
        
        possible_roots = 0
        
        for root in range(n):
            correct_guesses = Count_Correct_Guesses(root)
            if correct_guesses <= k:
                possible_roots += 1
        
        return possible_roots
    
    def Root_Count_Tree_Rerooting_Optimal(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        Tree Rerooting Optimal - Use rerooting technique for O(n) solution
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(edges) + 1
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        guess_set = set((u, v) for u, v in guesses)
        
        def DFS_Initial(node: int, parent: int) -> int:
            correct = 0
            if parent != -1 and (parent, node) in guess_set:
                correct += 1
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    correct += DFS_Initial(neighbor, node)
            
            return correct
        
        initial_correct = DFS_Initial(0, -1)
        
        correct_for_root = [0] * n
        
        def DFS_Reroot(node: int, parent: int, current_correct: int) -> None:
            correct_for_root[node] = current_correct
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    new_correct = current_correct
                    
                    if (node, neighbor) in guess_set:
                        new_correct -= 1
                    
                    if (neighbor, node) in guess_set:
                        new_correct += 1
                    
                    DFS_Reroot(neighbor, node, new_correct)
        
        DFS_Reroot(0, -1, initial_correct)
        
        possible_roots = 0
        for i in range(n):
            if correct_for_root[i] <= k:
                possible_roots += 1
        
        return possible_roots
    
    def Root_Count_DP_Memoized(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        DP Memoized - Use memoization for subtree calculations
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(edges) + 1
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        guess_set = set((u, v) for u, v in guesses)
        memo = {}
        
        def Count_Correct_In_Subtree(node: int, parent: int) -> int:
            if (node, parent) in memo:
                return memo[(node, parent)]
            
            correct = 0
            if parent != -1 and (parent, node) in guess_set:
                correct += 1
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    correct += Count_Correct_In_Subtree(neighbor, node)
            
            memo[(node, parent)] = correct
            return correct
        
        possible_roots = 0
        
        for root in range(n):
            total_correct = Count_Correct_In_Subtree(root, -1)
            if total_correct <= k:
                possible_roots += 1
        
        return possible_roots
    
    def Root_Count_Bottom_Up_DP(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        Bottom Up DP - Process tree bottom-up
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(edges) + 1
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        guess_set = set((u, v) for u, v in guesses)
        
        subtree_correct = {}
        
        def DFS_Bottom_Up(node: int, parent: int) -> int:
            correct = 0
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    child_correct = DFS_Bottom_Up(neighbor, node)
                    if (node, neighbor) in guess_set:
                        child_correct += 1
                    correct += child_correct
            
            subtree_correct[(node, parent)] = correct
            return correct
        
        possible_roots = 0
        
        for root in range(n):
            total_correct = DFS_Bottom_Up(root, -1)
            if total_correct <= k:
                possible_roots += 1
        
        return possible_roots
    
    def Root_Count_Tree_DP_Rerooting(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        Tree DP Rerooting - Advanced rerooting with DP optimization
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(edges) + 1
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        guesses_set = set((u, v) for u, v in guesses)
        
        dp = [0] * n
        
        def DFS1(u: int, parent: int) -> None:
            for v in adj[u]:
                if v != parent:
                    if (u, v) in guesses_set:
                        dp[v] = dp[u] + 1
                    else:
                        dp[v] = dp[u]
                    DFS1(v, u)
        
        DFS1(0, -1)
        
        ans = [0] * n
        ans[0] = dp[0]
        
        def DFS2(u: int, parent: int) -> None:
            for v in adj[u]:
                if v != parent:
                    change = 0
                    if (u, v) in guesses_set:
                        change -= 1
                    if (v, u) in guesses_set:
                        change += 1
                    
                    ans[v] = ans[u] + change
                    DFS2(v, u)
        
        DFS2(0, -1)
        
        return sum(1 for score in ans if score <= k)
    
    def Root_Count_Iterative_BFS(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        Iterative BFS - Use BFS for tree processing
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        from collections import deque
        
        n = len(edges) + 1
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        guess_set = set((u, v) for u, v in guesses)
        
        def Count_Correct_BFS(root: int) -> int:
            visited = set()
            queue = deque([(root, -1)])
            correct = 0
            
            while queue:
                node, parent = queue.popleft()
                
                if node in visited:
                    continue
                
                visited.add(node)
                
                if parent != -1 and (parent, node) in guess_set:
                    correct += 1
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, node))
            
            return correct
        
        possible_roots = 0
        
        for root in range(n):
            correct_guesses = Count_Correct_BFS(root)
            if correct_guesses <= k:
                possible_roots += 1
        
        return possible_roots
    
    def Root_Count_With_Details(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> Tuple[int, List[int], List[int]]:
        """
        With Details - Return count, valid roots, and their scores
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(edges) + 1
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        guess_set = set((u, v) for u, v in guesses)
        
        def DFS_Count(node: int, parent: int) -> int:
            correct = 0
            if parent != -1 and (parent, node) in guess_set:
                correct += 1
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    correct += DFS_Count(neighbor, node)
            
            return correct
        
        initial_score = DFS_Count(0, -1)
        
        scores = [0] * n
        
        def DFS_Reroot(node: int, parent: int, current_score: int) -> None:
            scores[node] = current_score
            
            for neighbor in graph[node]:
                if neighbor != parent:
                    new_score = current_score
                    
                    if (node, neighbor) in guess_set:
                        new_score -= 1
                    
                    if (neighbor, node) in guess_set:
                        new_score += 1
                    
                    DFS_Reroot(neighbor, node, new_score)
        
        DFS_Reroot(0, -1, initial_score)
        
        valid_roots = [i for i in range(n) if scores[i] <= k]
        
        return len(valid_roots), valid_roots, scores

def Test_Root_Count():
    solution = Solution()
    
    test_cases = [
        ([[0,1],[1,2],[1,3],[4,2]], [[1,3],[0,1],[1,0],[2,4]], 3, 3),
        ([[0,1],[1,2],[2,3],[3,4]], [[1,0],[3,4],[2,1],[3,2]], 1, 2),
        ([[0,1]], [[0,1]], 1, 2),
        ([[0,1]], [[1,0]], 0, 1),
        ([[0,1],[0,2]], [[0,1],[0,2]], 2, 3),
        ([[0,1],[1,2],[1,3]], [[0,1],[1,2],[1,3]], 2, 4)
    ]
    
    methods = [
        ("Tree Rerooting Optimal", solution.Root_Count_Tree_Rerooting_Optimal),
        ("DP Memoized", solution.Root_Count_DP_Memoized),
        ("Bottom Up DP", solution.Root_Count_Bottom_Up_DP),
        ("Tree DP Rerooting", solution.Root_Count_Tree_DP_Rerooting),
        ("Iterative BFS", solution.Root_Count_Iterative_BFS)
    ]
    
    for edges, guesses, k, expected in test_cases:
        print(f"Edges: {edges}")
        print(f"Guesses: {guesses}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        if len(edges) <= 4:
            result_bf = solution.Root_Count_Brute_Force([edge[:] for edge in edges], [guess[:] for guess in guesses], k)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([edge[:] for edge in edges], [guess[:] for guess in guesses], k)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        count, valid_roots, scores = solution.Root_Count_With_Details([edge[:] for edge in edges], [guess[:] for guess in guesses], k)
        print(f"With Details: Count={count}, Valid Roots={valid_roots}")
        print(f"Scores: {scores}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Root_Count()
