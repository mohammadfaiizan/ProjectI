"""
1462. Course Schedule IV - Multiple Approaches
Difficulty: Hard

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1] indicates that you have to take course 1 before you can take course 0.

Prerequisites can also be indirect. If course a is a prerequisite of course b, and course b is a prerequisite of course c, then course a is a prerequisite of course c.

You are also given an array queries where queries[i] = [ui, vi]. For the ith query, you should answer whether course ui is a prerequisite of course vi or not.

Return a boolean array answer, where answer[i] is the answer to the ith query.
"""

from typing import List
from collections import defaultdict, deque

class CourseScheduleIV:
    """Multiple approaches to determine course prerequisites"""
    
    def checkIfPrerequisite_floyd_warshall(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 1: Floyd-Warshall Algorithm
        
        Use Floyd-Warshall to find all-pairs reachability.
        
        Time: O(V^3), Space: O(V^2)
        """
        # Initialize reachability matrix
        reachable = [[False] * numCourses for _ in range(numCourses)]
        
        # Direct prerequisites
        for pre, course in prerequisites:
            reachable[pre][course] = True
        
        # Self-reachability (course is prerequisite of itself in some sense)
        for i in range(numCourses):
            reachable[i][i] = True
        
        # Floyd-Warshall: find transitive closure
        for k in range(numCourses):
            for i in range(numCourses):
                for j in range(numCourses):
                    reachable[i][j] = reachable[i][j] or (reachable[i][k] and reachable[k][j])
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(reachable[u][v])
        
        return result
    
    def checkIfPrerequisite_dfs_memoization(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 2: DFS with Memoization
        
        Use DFS with memoization to check reachability.
        
        Time: O(V^2 + E), Space: O(V^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for pre, course in prerequisites:
            graph[pre].append(course)
        
        # Memoization for reachability
        memo = {}
        
        def can_reach(start: int, target: int) -> bool:
            if start == target:
                return True
            
            if (start, target) in memo:
                return memo[(start, target)]
            
            # DFS to find if target is reachable from start
            visited = set()
            stack = [start]
            
            while stack:
                node = stack.pop()
                if node == target:
                    memo[(start, target)] = True
                    return True
                
                if node not in visited:
                    visited.add(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            memo[(start, target)] = False
            return False
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(can_reach(u, v))
        
        return result
    
    def checkIfPrerequisite_topological_sort(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 3: Topological Sort with Reachability
        
        Use topological sort to build reachability information.
        
        Time: O(V^2 + E), Space: O(V^2)
        """
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for pre, course in prerequisites:
            graph[pre].append(course)
            in_degree[course] += 1
        
        # Initialize reachability matrix
        reachable = [[False] * numCourses for _ in range(numCourses)]
        
        # Self-reachability
        for i in range(numCourses):
            reachable[i][i] = True
        
        # Topological sort with reachability propagation
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                # Propagate reachability
                for i in range(numCourses):
                    if reachable[i][node]:
                        reachable[i][neighbor] = True
                
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(reachable[u][v])
        
        return result
    
    def checkIfPrerequisite_bfs_from_each_node(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 4: BFS from Each Node
        
        Run BFS from each node to find all reachable nodes.
        
        Time: O(V * (V + E)), Space: O(V^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for pre, course in prerequisites:
            graph[pre].append(course)
        
        # For each node, find all reachable nodes using BFS
        reachable = [[False] * numCourses for _ in range(numCourses)]
        
        for start in range(numCourses):
            visited = set()
            queue = deque([start])
            visited.add(start)
            reachable[start][start] = True
            
            while queue:
                node = queue.popleft()
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        reachable[start][neighbor] = True
                        queue.append(neighbor)
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(reachable[u][v])
        
        return result
    
    def checkIfPrerequisite_optimized_dfs(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 5: Optimized DFS with Global Memoization
        
        Use optimized DFS with better memoization strategy.
        
        Time: O(V^2 + E), Space: O(V^2)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for pre, course in prerequisites:
            graph[pre].append(course)
        
        # Global reachability matrix
        reachable = {}
        
        def dfs(start: int) -> set:
            """DFS to find all nodes reachable from start"""
            if start in reachable:
                return reachable[start]
            
            visited = set([start])
            stack = [start]
            path_visited = set()
            
            while stack:
                node = stack.pop()
                
                if node not in path_visited:
                    path_visited.add(node)
                    
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
            
            reachable[start] = visited
            return visited
        
        # Precompute reachability for all nodes
        for i in range(numCourses):
            dfs(i)
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(v in reachable[u])
        
        return result
    
    def checkIfPrerequisite_matrix_multiplication(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 6: Matrix Multiplication Approach
        
        Use matrix multiplication to compute transitive closure.
        
        Time: O(V^3), Space: O(V^2)
        """
        # Initialize adjacency matrix
        adj = [[0] * numCourses for _ in range(numCourses)]
        
        for pre, course in prerequisites:
            adj[pre][course] = 1
        
        # Compute transitive closure using matrix multiplication concept
        # This is essentially Floyd-Warshall in matrix form
        result_matrix = [row[:] for row in adj]  # Copy
        
        # Add identity (self-loops)
        for i in range(numCourses):
            result_matrix[i][i] = 1
        
        # Compute transitive closure
        for k in range(numCourses):
            for i in range(numCourses):
                for j in range(numCourses):
                    result_matrix[i][j] = result_matrix[i][j] or (result_matrix[i][k] and result_matrix[k][j])
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(bool(result_matrix[u][v]))
        
        return result

def test_course_schedule_iv():
    """Test course schedule IV algorithms"""
    solver = CourseScheduleIV()
    
    test_cases = [
        (2, [[1,0]], [[0,1],[1,0]], [False, True], "Simple case"),
        (2, [], [[1,0],[0,1]], [False, False], "No prerequisites"),
        (3, [[1,2],[1,0],[2,0]], [[1,0],[1,2]], [True, True], "Transitive prerequisites"),
        (4, [[2,3],[2,1],[0,3],[0,1]], [[0,1],[0,3],[2,3],[3,0]], [True, True, True, False], "Complex case"),
    ]
    
    algorithms = [
        ("Floyd-Warshall", solver.checkIfPrerequisite_floyd_warshall),
        ("DFS Memoization", solver.checkIfPrerequisite_dfs_memoization),
        ("Topological Sort", solver.checkIfPrerequisite_topological_sort),
        ("BFS from Each Node", solver.checkIfPrerequisite_bfs_from_each_node),
        ("Optimized DFS", solver.checkIfPrerequisite_optimized_dfs),
        ("Matrix Multiplication", solver.checkIfPrerequisite_matrix_multiplication),
    ]
    
    print("=== Testing Course Schedule IV ===")
    
    for numCourses, prerequisites, queries, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Courses: {numCourses}, Prerequisites: {prerequisites}, Queries: {queries}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(numCourses, prerequisites, queries)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_course_schedule_iv()

"""
Course Schedule IV demonstrates advanced graph algorithms
for transitive closure computation and reachability queries
using Floyd-Warshall, DFS, and topological sorting techniques.
"""
