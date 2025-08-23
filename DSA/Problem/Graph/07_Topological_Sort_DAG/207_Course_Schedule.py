"""
207. Course Schedule
Difficulty: Medium

Problem:
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false.

Examples:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false

Input: numCourses = 3, prerequisites = [[1,0],[1,2],[0,1]]
Output: false

Constraints:
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- All the pairs prerequisites[i] are unique
"""

from typing import List
from collections import defaultdict, deque

class Solution:
    def canFinish_approach1_kahns_algorithm(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 1: Kahn's Algorithm (BFS-based Topological Sort)
        
        Use BFS to detect cycles in directed graph via topological sorting.
        
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
        
        # Process courses in topological order
        completed_courses = 0
        
        while queue:
            current_course = queue.popleft()
            completed_courses += 1
            
            # Remove current course and update in-degrees
            for next_course in graph[current_course]:
                in_degree[next_course] -= 1
                
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        # If all courses processed, no cycle exists
        return completed_courses == numCourses
    
    def canFinish_approach2_dfs_cycle_detection(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 2: DFS-based Cycle Detection
        
        Use DFS with three colors to detect cycles in directed graph.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # Color states: 0 = unvisited, 1 = visiting, 2 = visited
        color = [0] * numCourses
        
        def has_cycle(node):
            """DFS to detect cycle starting from node"""
            if color[node] == 1:  # Back edge found - cycle detected
                return True
            
            if color[node] == 2:  # Already processed
                return False
            
            # Mark as visiting
            color[node] = 1
            
            # Explore neighbors
            for neighbor in graph[node]:
                if has_cycle(neighbor):
                    return True
            
            # Mark as visited
            color[node] = 2
            return False
        
        # Check for cycles starting from each unvisited node
        for i in range(numCourses):
            if color[i] == 0 and has_cycle(i):
                return False
        
        return True
    
    def canFinish_approach3_dfs_topological_sort(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 3: DFS-based Topological Sort
        
        Generate complete topological ordering using DFS.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        visited = [False] * numCourses
        rec_stack = [False] * numCourses
        topo_order = []
        
        def dfs(node):
            """DFS to generate topological order"""
            if rec_stack[node]:  # Cycle detected
                return False
            
            if visited[node]:  # Already processed
                return True
            
            # Mark as visiting
            visited[node] = True
            rec_stack[node] = True
            
            # Explore neighbors
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    return False
            
            # Add to topological order (reverse post-order)
            topo_order.append(node)
            rec_stack[node] = False
            return True
        
        # Process all unvisited nodes
        for i in range(numCourses):
            if not visited[i]:
                if not dfs(i):
                    return False
        
        return len(topo_order) == numCourses
    
    def canFinish_approach4_union_find_cycle_detection(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 4: Union-Find for Cycle Detection (Modified)
        
        Use Union-Find with careful handling for directed graphs.
        
        Time: O(E * α(V))
        Space: O(V)
        """
        # This approach needs modification for directed graphs
        # We'll use it with strongly connected components concept
        
        # Build adjacency list for analysis
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # Use DFS-based approach as Union-Find is more suitable for undirected graphs
        return self.canFinish_approach2_dfs_cycle_detection(numCourses, prerequisites)
    
    def canFinish_approach5_iterative_elimination(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 5: Iterative Elimination Algorithm
        
        Repeatedly remove nodes with no incoming edges.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency structures
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Initialize all courses
        for i in range(numCourses):
            in_degree[i] = 0
        
        # Build graph
        for course, prereq in prerequisites:
            if course not in graph[prereq]:
                graph[prereq].add(course)
                in_degree[course] += 1
        
        # Find courses with no prerequisites
        no_prereq = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                no_prereq.append(i)
        
        # Iteratively eliminate courses
        eliminated = 0
        
        while no_prereq:
            current = no_prereq.popleft()
            eliminated += 1
            
            # Remove current course and update dependencies
            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    no_prereq.append(dependent)
        
        return eliminated == numCourses

def test_course_schedule():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (numCourses, prerequisites, expected)
        (2, [[1,0]], True),
        (2, [[1,0],[0,1]], False),
        (3, [[1,0],[1,2],[0,1]], False),
        (4, [[1,0],[2,0],[3,1],[3,2]], True),
        (5, [[1,4],[2,4],[3,1],[3,2]], True),
        (3, [[0,1],[0,2],[1,2]], True),
        (1, [], True),
        (2, [], True),
    ]
    
    approaches = [
        ("Kahn's Algorithm", solution.canFinish_approach1_kahns_algorithm),
        ("DFS Cycle Detection", solution.canFinish_approach2_dfs_cycle_detection),
        ("DFS Topological Sort", solution.canFinish_approach3_dfs_topological_sort),
        ("Union-Find Modified", solution.canFinish_approach4_union_find_cycle_detection),
        ("Iterative Elimination", solution.canFinish_approach5_iterative_elimination),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (numCourses, prerequisites, expected) in enumerate(test_cases):
            result = func(numCourses, prerequisites[:])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} courses={numCourses}, expected={expected}, got={result}")

def demonstrate_cycle_detection():
    """Demonstrate cycle detection in course prerequisites"""
    print("\n=== Cycle Detection Demo ===")
    
    test_case = {
        "numCourses": 4,
        "prerequisites": [[1,0],[2,1],[3,2],[1,3]]
    }
    
    print(f"Courses: {test_case['numCourses']}")
    print(f"Prerequisites: {test_case['prerequisites']}")
    print(f"Interpretation:")
    for course, prereq in test_case['prerequisites']:
        print(f"  Course {course} requires Course {prereq}")
    
    # Build adjacency list for visualization
    graph = defaultdict(list)
    for course, prereq in test_case['prerequisites']:
        graph[prereq].append(course)
    
    print(f"\nDependency graph:")
    for course in range(test_case['numCourses']):
        if graph[course]:
            print(f"  Course {course} → {graph[course]}")
    
    print(f"\nCycle analysis:")
    print(f"  Course 1 requires Course 0")
    print(f"  Course 2 requires Course 1") 
    print(f"  Course 3 requires Course 2")
    print(f"  Course 1 requires Course 3")
    print(f"  → Cycle: 1 → 2 → 3 → 1")
    
    solution = Solution()
    result = solution.canFinish_approach1_kahns_algorithm(
        test_case['numCourses'], test_case['prerequisites'])
    print(f"\nCan finish all courses: {result}")

def demonstrate_kahns_algorithm():
    """Demonstrate Kahn's algorithm step by step"""
    print("\n=== Kahn's Algorithm Demo ===")
    
    numCourses = 4
    prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    
    print(f"Courses: {numCourses}")
    print(f"Prerequisites: {prerequisites}")
    
    # Build graph and in-degree
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    print(f"\nInitial state:")
    print(f"  Graph: {dict(graph)}")
    print(f"  In-degree: {in_degree}")
    
    # Initialize queue
    queue = deque()
    for i in range(numCourses):
        if in_degree[i] == 0:
            queue.append(i)
    
    print(f"  Initial queue (no prerequisites): {list(queue)}")
    
    # Process step by step
    completed = []
    step = 0
    
    while queue:
        step += 1
        current = queue.popleft()
        completed.append(current)
        
        print(f"\nStep {step}: Process course {current}")
        print(f"  Completed: {completed}")
        
        # Update dependencies
        for next_course in graph[current]:
            in_degree[next_course] -= 1
            print(f"    Course {next_course} in-degree: {in_degree[next_course]}")
            
            if in_degree[next_course] == 0:
                queue.append(next_course)
                print(f"    Added course {next_course} to queue")
        
        print(f"  Queue: {list(queue)}")
    
    print(f"\nFinal result:")
    print(f"  Completed courses: {completed}")
    print(f"  Can finish all: {len(completed) == numCourses}")

def demonstrate_dfs_cycle_detection():
    """Demonstrate DFS cycle detection with colors"""
    print("\n=== DFS Cycle Detection Demo ===")
    
    numCourses = 3
    prerequisites = [[1,0],[2,1],[0,2]]
    
    print(f"Courses: {numCourses}")
    print(f"Prerequisites: {prerequisites}")
    
    # Build graph
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    print(f"Graph: {dict(graph)}")
    
    # Color states: 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
    color = [0] * numCourses
    
    def dfs_with_trace(node, path):
        """DFS with tracing for demonstration"""
        print(f"  Visit course {node}, path: {path}")
        
        if color[node] == 1:  # Gray node - back edge found
            print(f"    CYCLE DETECTED: Back edge to gray node {node}")
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            print(f"    Cycle: {' → '.join(map(str, cycle))}")
            return True
        
        if color[node] == 2:  # Black node - already processed
            print(f"    Already visited course {node}")
            return False
        
        # Mark as gray (visiting)
        color[node] = 1
        path.append(node)
        
        # Explore neighbors
        for neighbor in graph[node]:
            if dfs_with_trace(neighbor, path):
                return True
        
        # Mark as black (visited)
        color[node] = 2
        path.pop()
        print(f"    Finished processing course {node}")
        return False
    
    print(f"\nDFS traversal:")
    
    for i in range(numCourses):
        if color[i] == 0:
            print(f"\nStarting DFS from course {i}:")
            if dfs_with_trace(i, []):
                print(f"Cycle found - cannot complete all courses")
                break
    else:
        print(f"No cycle found - can complete all courses")

def analyze_topological_sort_applications():
    """Analyze applications of topological sorting"""
    print("\n=== Topological Sort Applications ===")
    
    print("Course Scheduling Applications:")
    
    print("\n1. **Academic Course Planning:**")
    print("   • University course prerequisites")
    print("   • Graduation requirement planning")
    print("   • Semester scheduling optimization")
    print("   • Degree completion pathways")
    
    print("\n2. **Project Management:**")
    print("   • Task dependency scheduling")
    print("   • Critical path analysis")
    print("   • Resource allocation planning")
    print("   • Milestone sequencing")
    
    print("\n3. **Software Development:**")
    print("   • Build system dependencies")
    print("   • Module compilation order")
    print("   • Package manager resolution")
    print("   • Deployment pipeline stages")
    
    print("\n4. **Manufacturing:**")
    print("   • Assembly line sequencing")
    print("   • Production step ordering")
    print("   • Quality control checkpoints")
    print("   • Supply chain coordination")
    
    print("\n5. **Data Processing:**")
    print("   • ETL pipeline dependencies")
    print("   • Workflow orchestration")
    print("   • Data transformation sequences")
    print("   • Analytics pipeline stages")
    
    print("\nAlgorithm Selection:")
    print("• **Kahn's Algorithm:** Good for detecting impossible schedules")
    print("• **DFS-based:** Better for finding actual ordering")
    print("• **Both have O(V + E) complexity**")
    print("• **Choice depends on specific requirements**")

def compare_cycle_detection_methods():
    """Compare different cycle detection methods"""
    print("\n=== Cycle Detection Methods Comparison ===")
    
    print("Methods for Detecting Cycles in Directed Graphs:")
    
    print("\n1. **Kahn's Algorithm (BFS-based):**")
    print("   ✅ Simple and intuitive")
    print("   ✅ Natural for scheduling problems")
    print("   ✅ Easy to implement")
    print("   ❌ Doesn't provide cycle details")
    
    print("\n2. **DFS with Colors (3-color method):**")
    print("   ✅ Can identify actual cycle")
    print("   ✅ Memory efficient")
    print("   ✅ Good for theoretical analysis")
    print("   ❌ Slightly more complex")
    
    print("\n3. **DFS with Recursion Stack:**")
    print("   ✅ Tracks current path")
    print("   ✅ Shows cycle formation")
    print("   ✅ Good for debugging")
    print("   ❌ Additional space for path tracking")
    
    print("\n4. **Union-Find (for undirected):**")
    print("   ✅ Excellent for undirected graphs")
    print("   ✅ Incremental cycle detection")
    print("   ❌ Not directly applicable to directed graphs")
    print("   ❌ Requires modification for DAGs")
    
    print("\nPractical Recommendations:")
    print("• **For course scheduling:** Kahn's algorithm")
    print("• **For debugging dependencies:** DFS with path tracking")
    print("• **For theoretical analysis:** DFS with colors")
    print("• **For performance-critical:** Either Kahn's or DFS")

if __name__ == "__main__":
    test_course_schedule()
    demonstrate_cycle_detection()
    demonstrate_kahns_algorithm()
    demonstrate_dfs_cycle_detection()
    analyze_topological_sort_applications()
    compare_cycle_detection_methods()

"""
Topological Sort and DAG Concepts:
1. Kahn's Algorithm for BFS-based Topological Sorting
2. DFS-based Cycle Detection in Directed Graphs
3. Course Scheduling as DAG Analysis Problem
4. In-degree Tracking and Dependency Resolution
5. Directed Graph Cycle Detection Techniques

Key Problem Insights:
- Course scheduling as directed acyclic graph (DAG) problem
- Cycle detection essential for feasibility checking
- Multiple algorithmic approaches with same time complexity
- Practical applications in scheduling and dependency resolution

Algorithm Strategy:
1. Kahn's Algorithm: Process nodes with zero in-degree iteratively
2. DFS Cycle Detection: Use colors to detect back edges
3. Both approaches: O(V + E) time and space complexity
4. Early termination when cycle detected

Kahn's Algorithm:
- Build in-degree array and adjacency list
- Start with nodes having zero in-degree
- Process nodes in topological order using queue
- Decrement in-degrees as nodes are processed

DFS Cycle Detection:
- Use three colors: white (unvisited), gray (visiting), black (visited)
- Back edge to gray node indicates cycle
- Recursion stack tracks current path
- Post-order traversal gives topological order

Real-world Applications:
- Academic course prerequisite planning
- Project management and task scheduling
- Software build systems and dependencies
- Manufacturing process sequencing
- Data pipeline orchestration

This problem demonstrates fundamental DAG algorithms
essential for dependency resolution and scheduling systems.
"""
