"""
1462. Course Schedule IV - Multiple Approaches
Difficulty: Medium

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

You will also be given an array queries where queries[i] = [ui, vi]. For the ith query, you should answer whether course ui is a prerequisite of course vi.

Return a boolean array answer, where answer[i] is the answer to the ith query.
"""

from typing import List, Set
from collections import defaultdict, deque

class CourseScheduleIV:
    """Multiple approaches to solve course schedule IV problem"""
    
    def checkIfPrerequisite_floyd_warshall(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 1: Floyd-Warshall Algorithm
        
        Use Floyd-Warshall to find all transitive relationships.
        
        Time: O(V³), Space: O(V²)
        """
        # Initialize reachability matrix
        reachable = [[False] * numCourses for _ in range(numCourses)]
        
        # Mark direct prerequisites
        for prereq, course in prerequisites:
            reachable[prereq][course] = True
        
        # Floyd-Warshall to find all transitive relationships
        for k in range(numCourses):
            for i in range(numCourses):
                for j in range(numCourses):
                    reachable[i][j] = reachable[i][j] or (reachable[i][k] and reachable[k][j])
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(reachable[u][v])
        
        return result
    
    def checkIfPrerequisite_topological_sort(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 2: Topological Sort with Ancestor Tracking
        
        Use topological sort to track all ancestors for each course.
        
        Time: O(V + E + Q), Space: O(V²)
        """
        # Build adjacency list and in-degree
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for prereq, course in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Track all prerequisites (ancestors) for each course
        all_prereqs = [set() for _ in range(numCourses)]
        
        # Topological sort
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            course = queue.popleft()
            
            # Process all courses that depend on current course
            for next_course in graph[course]:
                # Add current course as prerequisite
                all_prereqs[next_course].add(course)
                
                # Add all prerequisites of current course
                all_prereqs[next_course].update(all_prereqs[course])
                
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(u in all_prereqs[v])
        
        return result
    
    def checkIfPrerequisite_dfs_memoization(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 3: DFS with Memoization
        
        Use DFS with memoization to check reachability.
        
        Time: O(V + E + Q*V), Space: O(V²)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for prereq, course in prerequisites:
            graph[prereq].append(course)
        
        # Memoization for reachability
        memo = {}
        
        def is_reachable(start: int, target: int) -> bool:
            """Check if target is reachable from start"""
            if (start, target) in memo:
                return memo[(start, target)]
            
            if start == target:
                memo[(start, target)] = True
                return True
            
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
            result.append(is_reachable(u, v))
        
        return result
    
    def checkIfPrerequisite_bfs_reachability(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 4: BFS Reachability Matrix
        
        Use BFS to build complete reachability matrix.
        
        Time: O(V³), Space: O(V²)
        """
        # Build adjacency list
        graph = [[] for _ in range(numCourses)]
        for prereq, course in prerequisites:
            graph[prereq].append(course)
        
        # Build reachability matrix using BFS from each node
        reachable = [[False] * numCourses for _ in range(numCourses)]
        
        for start in range(numCourses):
            visited = set([start])
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                
                for neighbor in graph[node]:
                    reachable[start][neighbor] = True
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(reachable[u][v])
        
        return result
    
    def checkIfPrerequisite_optimized_topological(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 5: Optimized Topological Sort
        
        Optimized topological sort with bitset for prerequisites.
        
        Time: O(V + E + Q), Space: O(V²)
        """
        # Build graph
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        for prereq, course in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Use sets to track prerequisites more efficiently
        prerequisites_set = [set() for _ in range(numCourses)]
        
        # Topological sort
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            course = queue.popleft()
            
            for next_course in graph[course]:
                # Add current course and all its prerequisites
                prerequisites_set[next_course].add(course)
                prerequisites_set[next_course].update(prerequisites_set[course])
                
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(u in prerequisites_set[v])
        
        return result
    
    def checkIfPrerequisite_matrix_multiplication(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        Approach 6: Matrix Multiplication Approach
        
        Use matrix multiplication concept for transitive closure.
        
        Time: O(V³), Space: O(V²)
        """
        # Initialize adjacency matrix
        adj = [[False] * numCourses for _ in range(numCourses)]
        
        for prereq, course in prerequisites:
            adj[prereq][course] = True
        
        # Compute transitive closure using matrix multiplication concept
        # Keep multiplying until no changes
        changed = True
        while changed:
            changed = False
            new_adj = [row[:] for row in adj]  # Copy current matrix
            
            for i in range(numCourses):
                for j in range(numCourses):
                    if not adj[i][j]:
                        # Check if there's a path i -> k -> j
                        for k in range(numCourses):
                            if adj[i][k] and adj[k][j]:
                                new_adj[i][j] = True
                                changed = True
                                break
            
            adj = new_adj
        
        # Answer queries
        result = []
        for u, v in queries:
            result.append(adj[u][v])
        
        return result

def test_course_schedule_iv():
    """Test course schedule IV algorithms"""
    solver = CourseScheduleIV()
    
    test_cases = [
        (2, [[1,0]], [[0,1],[1,0]], [False, True], "Simple case"),
        (2, [], [[1,0],[0,1]], [False, False], "No prerequisites"),
        (3, [[1,2],[1,0],[2,0]], [[1,0],[1,2]], [True, True], "Transitive prerequisites"),
        (4, [[2,3],[2,1],[0,3],[0,1]], [[0,1],[0,3],[2,1],[2,3]], [True, True, True, True], "Complex case"),
        (5, [[0,1],[1,2],[2,3],[3,4]], [[0,4],[4,0],[1,3],[2,0]], [True, False, True, False], "Linear chain"),
    ]
    
    algorithms = [
        ("Floyd-Warshall", solver.checkIfPrerequisite_floyd_warshall),
        ("Topological Sort", solver.checkIfPrerequisite_topological_sort),
        ("DFS Memoization", solver.checkIfPrerequisite_dfs_memoization),
        ("BFS Reachability", solver.checkIfPrerequisite_bfs_reachability),
        ("Optimized Topological", solver.checkIfPrerequisite_optimized_topological),
        ("Matrix Multiplication", solver.checkIfPrerequisite_matrix_multiplication),
    ]
    
    print("=== Testing Course Schedule IV ===")
    
    for numCourses, prerequisites, queries, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Courses: {numCourses}, Prerequisites: {prerequisites}")
        print(f"Queries: {queries}")
        
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
Course Schedule IV demonstrates transitive closure computation
and reachability analysis in directed graphs using multiple
algorithmic approaches including Floyd-Warshall and topological sorting.
"""
