"""
210. Course Schedule II
Difficulty: Medium

Problem:
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return the ordering of courses you should take to finish all courses. If there are many valid 
answers, return any of them. If it is impossible to finish all courses, return an empty array.

Examples:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3] (or [0,1,2,3])

Input: numCourses = 1, prerequisites = []
Output: [0]

Constraints:
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= numCourses * (numCourses - 1)
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- ai != bi
- All the pairs [ai, bi] are distinct
"""

from typing import List
from collections import defaultdict, deque

class Solution:
    def findOrder_approach1_kahns_algorithm(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Approach 1: Kahn's Algorithm (BFS-based Topological Sort)
        
        Generate topological ordering using BFS with in-degree tracking.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list and in-degree array
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Initialize queue with courses having no prerequisites
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        # Generate topological ordering
        topo_order = []
        
        while queue:
            current_course = queue.popleft()
            topo_order.append(current_course)
            
            # Process dependent courses
            for next_course in graph[current_course]:
                in_degree[next_course] -= 1
                
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        # Return ordering if all courses can be taken
        return topo_order if len(topo_order) == numCourses else []
    
    def findOrder_approach2_dfs_topological_sort(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Approach 2: DFS-based Topological Sort
        
        Generate topological ordering using DFS with post-order traversal.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # Color states: 0 = unvisited, 1 = visiting, 2 = visited
        color = [0] * numCourses
        topo_order = []
        
        def dfs(node):
            """DFS to generate topological order"""
            if color[node] == 1:  # Cycle detected
                return False
            
            if color[node] == 2:  # Already processed
                return True
            
            # Mark as visiting
            color[node] = 1
            
            # Explore neighbors
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    return False
            
            # Mark as visited and add to order (post-order)
            color[node] = 2
            topo_order.append(node)
            return True
        
        # Process all unvisited nodes
        for i in range(numCourses):
            if color[i] == 0:
                if not dfs(i):
                    return []  # Cycle detected
        
        # Reverse to get correct topological order
        return topo_order[::-1]
    
    def findOrder_approach3_priority_based_kahns(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Approach 3: Priority-based Kahn's Algorithm
        
        Use priority queue to get lexicographically smallest ordering.
        
        Time: O(V log V + E)
        Space: O(V + E)
        """
        import heapq
        
        # Build adjacency list and in-degree array
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Initialize min-heap with courses having no prerequisites
        heap = []
        for i in range(numCourses):
            if in_degree[i] == 0:
                heapq.heappush(heap, i)
        
        # Generate topological ordering
        topo_order = []
        
        while heap:
            current_course = heapq.heappop(heap)
            topo_order.append(current_course)
            
            # Process dependent courses
            for next_course in graph[current_course]:
                in_degree[next_course] -= 1
                
                if in_degree[next_course] == 0:
                    heapq.heappush(heap, next_course)
        
        return topo_order if len(topo_order) == numCourses else []
    
    def findOrder_approach4_iterative_dfs(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Approach 4: Iterative DFS with Stack
        
        Avoid recursion using explicit stack for DFS traversal.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # State tracking
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * numCourses
        topo_order = []
        
        def iterative_dfs(start):
            """Iterative DFS using explicit stack"""
            stack = [(start, False)]  # (node, processed)
            
            while stack:
                node, processed = stack.pop()
                
                if processed:
                    # Post-processing: add to topological order
                    topo_order.append(node)
                    color[node] = BLACK
                else:
                    if color[node] == GRAY:  # Cycle detected
                        return False
                    
                    if color[node] == BLACK:  # Already processed
                        continue
                    
                    # Pre-processing: mark as visiting
                    color[node] = GRAY
                    stack.append((node, True))  # Mark for post-processing
                    
                    # Add neighbors to stack
                    for neighbor in graph[node]:
                        if color[neighbor] != BLACK:
                            stack.append((neighbor, False))
            
            return True
        
        # Process all unvisited nodes
        for i in range(numCourses):
            if color[i] == WHITE:
                if not iterative_dfs(i):
                    return []  # Cycle detected
        
        # Reverse to get correct topological order
        return topo_order[::-1]
    
    def findOrder_approach5_modified_kahns_with_levels(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Approach 5: Modified Kahn's with Level Processing
        
        Process courses level by level for better understanding.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list and in-degree array
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Process level by level
        topo_order = []
        current_level = []
        
        # Find initial level (no prerequisites)
        for i in range(numCourses):
            if in_degree[i] == 0:
                current_level.append(i)
        
        while current_level:
            # Process all courses in current level
            current_level.sort()  # For consistent ordering
            next_level = []
            
            for course in current_level:
                topo_order.append(course)
                
                # Update dependent courses
                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    
                    if in_degree[next_course] == 0:
                        next_level.append(next_course)
            
            current_level = next_level
        
        return topo_order if len(topo_order) == numCourses else []

def test_course_schedule_ii():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (numCourses, prerequisites)
        (2, [[1,0]]),
        (4, [[1,0],[2,0],[3,1],[3,2]]),
        (1, []),
        (3, [[1,0],[1,2],[0,1]]),  # Cycle case
        (6, [[1,0],[2,0],[3,1],[4,1],[5,3],[5,4]]),
        (3, [[0,1],[0,2],[1,2]]),
    ]
    
    approaches = [
        ("Kahn's Algorithm", solution.findOrder_approach1_kahns_algorithm),
        ("DFS Topological Sort", solution.findOrder_approach2_dfs_topological_sort),
        ("Priority Kahn's", solution.findOrder_approach3_priority_based_kahns),
        ("Iterative DFS", solution.findOrder_approach4_iterative_dfs),
        ("Level-based Kahn's", solution.findOrder_approach5_modified_kahns_with_levels),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (numCourses, prerequisites) in enumerate(test_cases):
            result = func(numCourses, prerequisites[:])  # Deep copy
            status = "✓" if result else "✗" if not result else "✓"
            print(f"Test {i+1}: {status} courses={numCourses}, order={result}")

def demonstrate_topological_ordering():
    """Demonstrate topological ordering generation"""
    print("\n=== Topological Ordering Demo ===")
    
    numCourses = 6
    prerequisites = [[1,0],[2,0],[3,1],[4,1],[5,3],[5,4]]
    
    print(f"Courses: {numCourses}")
    print(f"Prerequisites: {prerequisites}")
    
    # Visualize dependencies
    print(f"\nDependency visualization:")
    for course, prereq in prerequisites:
        print(f"  Course {course} requires Course {prereq}")
    
    # Build graph
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    print(f"\nGraph structure:")
    for course in range(numCourses):
        if graph[course]:
            print(f"  Course {course} enables: {graph[course]}")
    
    print(f"\nIn-degrees: {in_degree}")
    
    # Generate ordering using Kahn's algorithm
    solution = Solution()
    order = solution.findOrder_approach1_kahns_algorithm(numCourses, prerequisites)
    
    print(f"\nPossible course order: {order}")
    
    # Verify ordering is valid
    if order:
        print(f"\nVerification:")
        taken = set()
        valid = True
        
        for course in order:
            # Check if all prerequisites are satisfied
            prereq_courses = []
            for c, p in prerequisites:
                if c == course:
                    prereq_courses.append(p)
            
            missing_prereqs = [p for p in prereq_courses if p not in taken]
            
            if missing_prereqs:
                print(f"  Course {course}: Missing prerequisites {missing_prereqs}")
                valid = False
            else:
                print(f"  Course {course}: All prerequisites satisfied")
            
            taken.add(course)
        
        print(f"\nOrdering is {'valid' if valid else 'invalid'}")

def demonstrate_dfs_vs_bfs_ordering():
    """Demonstrate difference between DFS and BFS topological ordering"""
    print("\n=== DFS vs BFS Ordering Comparison ===")
    
    numCourses = 4
    prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    
    print(f"Course dependencies:")
    for course, prereq in prerequisites:
        print(f"  {course} → {prereq}")
    
    solution = Solution()
    
    # Get orderings from different algorithms
    bfs_order = solution.findOrder_approach1_kahns_algorithm(numCourses, prerequisites)
    dfs_order = solution.findOrder_approach2_dfs_topological_sort(numCourses, prerequisites)
    priority_order = solution.findOrder_approach3_priority_based_kahns(numCourses, prerequisites)
    
    print(f"\nDifferent valid topological orderings:")
    print(f"  BFS (Kahn's): {bfs_order}")
    print(f"  DFS-based: {dfs_order}")
    print(f"  Priority-based: {priority_order}")
    
    print(f"\nWhy different orderings are possible:")
    print(f"• Course 0 has no prerequisites - can be taken first")
    print(f"• After Course 0, both Course 1 and Course 2 become available")
    print(f"• Different algorithms may choose different available courses")
    print(f"• All orderings are valid as long as dependencies are respected")

def analyze_algorithm_characteristics():
    """Analyze characteristics of different topological sort algorithms"""
    print("\n=== Algorithm Characteristics Analysis ===")
    
    print("Topological Sort Algorithm Comparison:")
    
    print("\n1. **Kahn's Algorithm (BFS-based):**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Uses in-degree tracking and queue")
    print("   • Natural for course scheduling")
    print("   • Produces one valid ordering")
    print("   • Easy to understand and implement")
    
    print("\n2. **DFS-based Topological Sort:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Uses recursion and post-order traversal")
    print("   • Natural for dependency analysis")
    print("   • Reverse post-order gives topological order")
    print("   • Better for theoretical analysis")
    
    print("\n3. **Priority-based Variants:**")
    print("   • Time: O(V log V + E)")
    print("   • Space: O(V + E)")
    print("   • Uses heap for lexicographic ordering")
    print("   • Deterministic output for same input")
    print("   • Useful when specific ordering preferred")
    
    print("\n4. **Level-based Processing:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Processes courses level by level")
    print("   • Good for parallel processing")
    print("   • Natural for semester planning")
    
    print("\nChoosing the Right Algorithm:")
    print("• **Kahn's Algorithm:** General purpose, easy to understand")
    print("• **DFS-based:** When you need path information")
    print("• **Priority-based:** When you need deterministic ordering")
    print("• **Level-based:** When you need to understand dependencies")

def demonstrate_practical_applications():
    """Demonstrate practical applications of course scheduling"""
    print("\n=== Practical Applications Demo ===")
    
    print("Real-World Course Scheduling Scenarios:")
    
    print("\n1. **University Degree Planning:**")
    courses = {
        "CS101": "Intro to Programming",
        "CS201": "Data Structures", 
        "CS301": "Algorithms",
        "CS302": "Database Systems",
        "CS401": "Software Engineering"
    }
    
    prerequisites = [
        ["CS201", "CS101"],
        ["CS301", "CS201"],
        ["CS302", "CS201"],
        ["CS401", "CS301"]
    ]
    
    print(f"  Courses: {courses}")
    print(f"  Prerequisites:")
    for course, prereq in prerequisites:
        print(f"    {course} requires {prereq}")
    
    print("\n2. **Build System Dependencies:**")
    modules = {
        "utils": "Utility functions",
        "database": "Database layer",
        "api": "API handlers",
        "frontend": "User interface",
        "tests": "Test suite"
    }
    
    build_deps = [
        ["database", "utils"],
        ["api", "database"],
        ["frontend", "api"],
        ["tests", "frontend"]
    ]
    
    print(f"  Modules: {modules}")
    print(f"  Build dependencies:")
    for module, dep in build_deps:
        print(f"    {module} depends on {dep}")
    
    print("\n3. **Project Task Scheduling:**")
    tasks = {
        "research": "Market research",
        "design": "Product design",
        "prototype": "Build prototype", 
        "testing": "User testing",
        "launch": "Product launch"
    }
    
    task_deps = [
        ["design", "research"],
        ["prototype", "design"],
        ["testing", "prototype"],
        ["launch", "testing"]
    ]
    
    print(f"  Tasks: {tasks}")
    print(f"  Task dependencies:")
    for task, dep in task_deps:
        print(f"    {task} requires {dep}")

if __name__ == "__main__":
    test_course_schedule_ii()
    demonstrate_topological_ordering()
    demonstrate_dfs_vs_bfs_ordering()
    analyze_algorithm_characteristics()
    demonstrate_practical_applications()

"""
Topological Sort and Course Ordering Concepts:
1. Kahn's Algorithm for BFS-based Topological Ordering
2. DFS-based Topological Sort with Post-order Traversal
3. Priority-based Ordering for Deterministic Results
4. Level-based Processing for Parallel Scheduling
5. Practical Applications in Course and Task Planning

Key Problem Insights:
- Generate valid course ordering respecting all prerequisites
- Multiple valid orderings possible for same dependency graph
- Cycle detection essential before generating ordering
- Different algorithms produce different but valid results

Algorithm Strategy:
1. Kahn's Algorithm: Process nodes with zero in-degree iteratively
2. DFS-based: Use post-order traversal, then reverse
3. Both approaches: O(V + E) time complexity
4. Return empty array if cycle detected

Topological Ordering Properties:
- Linear ordering of vertices in directed acyclic graph (DAG)
- For every directed edge (u,v), u appears before v in ordering
- Multiple valid orderings possible for same graph
- Ordering exists if and only if graph is acyclic

Algorithm Variations:
- Standard Kahn's: BFS with in-degree tracking
- DFS-based: Recursive traversal with post-order
- Priority-based: Use heap for lexicographic ordering
- Level-based: Process dependencies level by level

Real-world Applications:
- University course prerequisite planning
- Software build system dependencies
- Project management task scheduling
- Manufacturing process sequencing
- Data pipeline orchestration

This problem demonstrates practical topological sorting
for dependency resolution and scheduling optimization.
"""
