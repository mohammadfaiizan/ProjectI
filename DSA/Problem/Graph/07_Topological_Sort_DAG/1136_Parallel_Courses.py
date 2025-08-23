"""
1136. Parallel Courses - Multiple Approaches
Difficulty: Easy

You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei: course prevCoursei has to be taken before course nextCoursei.

In one semester, you can take any number of courses as long as you have taken all the prerequisites in the previous semesters for the courses you are taking.

Return the minimum number of semesters needed to take all courses. If there is no way to take all the courses due to cyclic prerequisites, return -1.
"""

from typing import List
from collections import defaultdict, deque

class ParallelCourses:
    """Multiple approaches to find minimum semesters for parallel courses"""
    
    def minimumSemesters_kahn_algorithm(self, n: int, relations: List[List[int]]) -> int:
        """
        Approach 1: Kahn's Algorithm (BFS-based Topological Sort)
        
        Use BFS to process courses level by level (semester by semester).
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)  # 1-indexed
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            in_degree[next_course] += 1
        
        # Find courses with no prerequisites (can take in first semester)
        queue = deque()
        for i in range(1, n + 1):
            if in_degree[i] == 0:
                queue.append(i)
        
        semesters = 0
        courses_taken = 0
        
        while queue:
            semesters += 1
            semester_size = len(queue)
            
            # Process all courses in current semester
            for _ in range(semester_size):
                course = queue.popleft()
                courses_taken += 1
                
                # Update prerequisites for dependent courses
                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        queue.append(next_course)
        
        # Check if all courses can be taken (no cycle)
        return semesters if courses_taken == n else -1
    
    def minimumSemesters_dfs_topological(self, n: int, relations: List[List[int]]) -> int:
        """
        Approach 2: DFS-based Topological Sort
        
        Use DFS to detect cycles and find topological order.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for prev, next_course in relations:
            graph[prev].append(next_course)
        
        # DFS with cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * (n + 1)
        topo_order = []
        
        def dfs(course: int) -> bool:
            if color[course] == GRAY:  # Back edge found - cycle detected
                return False
            if color[course] == BLACK:  # Already processed
                return True
            
            color[course] = GRAY
            
            for next_course in graph[course]:
                if not dfs(next_course):
                    return False
            
            color[course] = BLACK
            topo_order.append(course)
            return True
        
        # Run DFS from all unvisited nodes
        for i in range(1, n + 1):
            if color[i] == WHITE:
                if not dfs(i):
                    return -1  # Cycle detected
        
        # Reverse to get correct topological order
        topo_order.reverse()
        
        # Calculate minimum semesters using topological order
        course_to_semester = {}
        
        for course in topo_order:
            max_prereq_semester = 0
            
            # Find the maximum semester of all prerequisites
            for prev in range(1, n + 1):
                if course in graph[prev]:  # prev is prerequisite of course
                    if prev in course_to_semester:
                        max_prereq_semester = max(max_prereq_semester, course_to_semester[prev])
            
            course_to_semester[course] = max_prereq_semester + 1
        
        return max(course_to_semester.values())
    
    def minimumSemesters_longest_path(self, n: int, relations: List[List[int]]) -> int:
        """
        Approach 3: Longest Path in DAG
        
        Find longest path in DAG to determine minimum semesters.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            in_degree[next_course] += 1
        
        # Initialize distances (semesters)
        dist = [1] * (n + 1)  # Each course takes at least 1 semester
        
        # Topological sort with distance calculation
        queue = deque()
        for i in range(1, n + 1):
            if in_degree[i] == 0:
                queue.append(i)
        
        processed = 0
        
        while queue:
            course = queue.popleft()
            processed += 1
            
            for next_course in graph[course]:
                # Update distance to next course
                dist[next_course] = max(dist[next_course], dist[course] + 1)
                
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        # Check for cycles
        if processed != n:
            return -1
        
        # Return maximum distance (minimum semesters needed)
        return max(dist[1:])
    
    def minimumSemesters_dp_approach(self, n: int, relations: List[List[int]]) -> int:
        """
        Approach 4: Dynamic Programming Approach
        
        Use DP to calculate minimum semesters for each course.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for prev, next_course in relations:
            graph[prev].append(next_course)
        
        # Memoization for minimum semesters
        memo = {}
        visiting = set()  # For cycle detection
        
        def dp(course: int) -> int:
            if course in visiting:  # Cycle detected
                return -1
            if course in memo:
                return memo[course]
            
            visiting.add(course)
            
            max_prereq_semesters = 0
            
            # Check all courses that are prerequisites for current course
            for prev in range(1, n + 1):
                if course in graph[prev]:  # prev -> course
                    prereq_semesters = dp(prev)
                    if prereq_semesters == -1:  # Cycle in prerequisites
                        visiting.remove(course)
                        return -1
                    max_prereq_semesters = max(max_prereq_semesters, prereq_semesters)
            
            visiting.remove(course)
            memo[course] = max_prereq_semesters + 1
            return memo[course]
        
        # Calculate minimum semesters for all courses
        max_semesters = 0
        for i in range(1, n + 1):
            semesters = dp(i)
            if semesters == -1:
                return -1
            max_semesters = max(max_semesters, semesters)
        
        return max_semesters
    
    def minimumSemesters_iterative_relaxation(self, n: int, relations: List[List[int]]) -> int:
        """
        Approach 5: Iterative Distance Relaxation
        
        Iteratively relax distances until convergence.
        
        Time: O(V * E), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for prev, next_course in relations:
            graph[prev].append(next_course)
        
        # Initialize distances
        dist = [1] * (n + 1)  # Each course needs at least 1 semester
        
        # Relax distances iteratively
        for _ in range(n):  # At most n iterations needed
            updated = False
            
            for course in range(1, n + 1):
                for next_course in graph[course]:
                    if dist[next_course] < dist[course] + 1:
                        dist[next_course] = dist[course] + 1
                        updated = True
            
            if not updated:
                break
        
        # Check for negative cycles (shouldn't happen in this problem)
        # But we can detect if we need more than n semesters
        for course in range(1, n + 1):
            for next_course in graph[course]:
                if dist[next_course] < dist[course] + 1:
                    return -1  # Cycle detected
        
        return max(dist[1:])

def test_parallel_courses():
    """Test parallel courses algorithms"""
    solver = ParallelCourses()
    
    test_cases = [
        (3, [[1,3],[2,3]], 2, "Two prerequisites for course 3"),
        (3, [[1,2],[2,3],[3,1]], -1, "Cycle in prerequisites"),
        (1, [], 1, "Single course, no prerequisites"),
        (4, [[2,1],[3,1],[1,4]], 3, "Linear dependency chain"),
        (5, [[2,1],[3,1],[4,1],[1,5]], 3, "Multiple courses depend on course 1"),
    ]
    
    algorithms = [
        ("Kahn's Algorithm", solver.minimumSemesters_kahn_algorithm),
        ("DFS Topological", solver.minimumSemesters_dfs_topological),
        ("Longest Path", solver.minimumSemesters_longest_path),
        ("DP Approach", solver.minimumSemesters_dp_approach),
        ("Iterative Relaxation", solver.minimumSemesters_iterative_relaxation),
    ]
    
    print("=== Testing Parallel Courses ===")
    
    for n, relations, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, relations={relations}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, relations)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Semesters: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_parallel_courses()

"""
Parallel Courses demonstrates topological sorting applications
for scheduling problems with dependency constraints and
cycle detection in directed graphs.
"""
