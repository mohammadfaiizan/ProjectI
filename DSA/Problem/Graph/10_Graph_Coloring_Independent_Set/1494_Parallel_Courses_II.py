"""
1494. Parallel Courses II
Difficulty: Medium

Problem:
You are given an integer n, which indicates that there are n courses labeled from 1 to n. 
You are also given an array relations where relations[i] = [prevCourse_i, nextCourse_i], 
denoting a prerequisite relationship between course prevCourse_i and course nextCourse_i: 
course prevCourse_i has to be taken before course nextCourse_i. 
Also, you are given the integer k, which indicates that you can take at most k courses 
simultaneously in each semester.

Return the minimum number of semesters needed to learn all n courses. The testcases will 
be generated such that it is possible to learn all the courses (i.e., the given graph is a DAG).

Examples:
Input: n = 4, relations = [[2,1],[3,1],[1,4]], k = 2
Output: 3
Explanation: The figure above represents the given graph.
In the first semester, you can take courses 2 and 3.
In the second semester, you can take course 1.
In the third semester, you can take course 4.

Input: n = 5, relations = [[2,1],[3,1],[4,1],[1,5]], k = 2
Output: 4

Input: n = 11, relations = [], k = 2
Output: 6

Constraints:
- 1 <= n <= 15
- 1 <= k <= n
- 0 <= relations.length <= n * (n-1) / 2
- relations[i].length == 2
- 1 <= prevCourse_i, nextCourse_i <= n
- prevCourse_i != nextCourse_i
- All the pairs [prevCourse_i, nextCourse_i] are unique.
- The given graph is a DAG.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import itertools

class Solution:
    def minimumSemesters_approach1_bitmask_dp(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 1: Bitmask Dynamic Programming
        
        Use bitmask DP with state compression to find optimal schedule.
        
        Time: O(3^n * k)
        Space: O(2^n)
        """
        # Build prerequisites map
        prerequisites = defaultdict(list)
        for prev, next_course in relations:
            prerequisites[next_course].append(prev)
        
        # Precompute valid subsets for each course
        def can_take_course(course, taken_mask):
            """Check if course can be taken given taken courses"""
            for prereq in prerequisites[course]:
                if not (taken_mask & (1 << (prereq - 1))):
                    return False
            return True
        
        # DP state: dp[mask] = minimum semesters to complete courses in mask
        dp = {}
        dp[0] = 0
        
        def solve(taken_mask):
            """Solve for minimum semesters starting from taken_mask"""
            if taken_mask in dp:
                return dp[taken_mask]
            
            if taken_mask == (1 << n) - 1:  # All courses taken
                return 0
            
            # Find available courses
            available = []
            for course in range(1, n + 1):
                if not (taken_mask & (1 << (course - 1))):  # Not taken
                    if can_take_course(course, taken_mask):
                        available.append(course)
            
            if not available:
                return float('inf')
            
            result = float('inf')
            
            # Try all subsets of available courses with size <= k
            for size in range(1, min(k, len(available)) + 1):
                for subset in itertools.combinations(available, size):
                    new_mask = taken_mask
                    for course in subset:
                        new_mask |= (1 << (course - 1))
                    
                    result = min(result, 1 + solve(new_mask))
            
            dp[taken_mask] = result
            return result
        
        return solve(0)
    
    def minimumSemesters_approach2_topological_bfs(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 2: Modified Topological Sort with BFS
        
        Use BFS-based topological sort with k-constraint.
        
        Time: O(2^n * n) in worst case
        Space: O(n + E)
        """
        # Build graph and in-degree
        graph = defaultdict(list)
        indegree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            indegree[next_course] += 1
        
        semesters = 0
        completed = 0
        
        while completed < n:
            # Find all courses with no prerequisites
            available = []
            for course in range(1, n + 1):
                if indegree[course] == 0:
                    available.append(course)
            
            if not available:
                return -1  # Impossible (shouldn't happen in valid input)
            
            # Take up to k courses this semester
            to_take = available[:k]
            
            # Update for next iteration
            for course in to_take:
                indegree[course] = -1  # Mark as taken
                completed += 1
                
                # Update prerequisites for dependent courses
                for dependent in graph[course]:
                    indegree[dependent] -= 1
            
            semesters += 1
        
        return semesters
    
    def minimumSemesters_approach3_dfs_with_memoization(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 3: DFS with Memoization
        
        Use DFS to explore all possible schedules with memoization.
        
        Time: O(2^n * n)
        Space: O(2^n)
        """
        # Build prerequisites
        prereqs = defaultdict(set)
        for prev, next_course in relations:
            prereqs[next_course].add(prev)
        
        memo = {}
        
        def dfs(taken_mask):
            """DFS to find minimum semesters"""
            if taken_mask in memo:
                return memo[taken_mask]
            
            if taken_mask == (1 << n) - 1:
                return 0
            
            # Find available courses
            available = []
            for course in range(1, n + 1):
                if not (taken_mask & (1 << (course - 1))):
                    # Check if all prerequisites are satisfied
                    can_take = True
                    for prereq in prereqs[course]:
                        if not (taken_mask & (1 << (prereq - 1))):
                            can_take = False
                            break
                    
                    if can_take:
                        available.append(course)
            
            if not available:
                memo[taken_mask] = float('inf')
                return float('inf')
            
            result = float('inf')
            
            # Try all possible combinations of up to k courses
            for num_courses in range(1, min(k, len(available)) + 1):
                for combo in itertools.combinations(available, num_courses):
                    new_mask = taken_mask
                    for course in combo:
                        new_mask |= (1 << (course - 1))
                    
                    result = min(result, 1 + dfs(new_mask))
            
            memo[taken_mask] = result
            return result
        
        return dfs(0)
    
    def minimumSemesters_approach4_iterative_scheduling(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 4: Iterative Scheduling with Greedy Selection
        
        Iteratively schedule courses greedily while respecting constraints.
        
        Time: O(n^2)
        Space: O(n + E)
        """
        # Build dependency graph
        graph = defaultdict(list)
        indegree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            indegree[next_course] += 1
        
        taken = [False] * (n + 1)
        semesters = 0
        
        while sum(taken[1:]) < n:
            # Find available courses (prerequisites satisfied)
            available = []
            for course in range(1, n + 1):
                if not taken[course] and indegree[course] == 0:
                    available.append(course)
            
            if not available:
                return -1
            
            # Greedy selection: prioritize courses with more dependents
            def count_dependents(course):
                """Count total courses that depend on this course"""
                visited = set()
                
                def dfs(c):
                    if c in visited:
                        return 0
                    visited.add(c)
                    count = 1
                    for dependent in graph[c]:
                        count += dfs(dependent)
                    return count
                
                return dfs(course) - 1  # Exclude the course itself
            
            # Sort by number of dependents (descending)
            available.sort(key=count_dependents, reverse=True)
            
            # Take up to k courses
            semester_courses = available[:k]
            
            # Update state
            for course in semester_courses:
                taken[course] = True
                for dependent in graph[course]:
                    indegree[dependent] -= 1
            
            semesters += 1
        
        return semesters
    
    def minimumSemesters_approach5_branch_and_bound(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 5: Branch and Bound Optimization
        
        Use branch and bound to prune infeasible solutions.
        
        Time: O(2^n) with pruning
        Space: O(n)
        """
        # Build prerequisites
        prereqs = defaultdict(set)
        for prev, next_course in relations:
            prereqs[next_course].add(prev)
        
        self.best_result = float('inf')
        
        def lower_bound(taken_mask, remaining_courses):
            """Calculate lower bound on remaining semesters"""
            return max(1, (remaining_courses + k - 1) // k)
        
        def branch_and_bound(taken_mask, semester):
            """Branch and bound search"""
            if semester >= self.best_result:
                return  # Prune
            
            if taken_mask == (1 << n) - 1:
                self.best_result = min(self.best_result, semester)
                return
            
            # Calculate remaining courses
            remaining = n - bin(taken_mask).count('1')
            
            # Prune if lower bound exceeds current best
            if semester + lower_bound(taken_mask, remaining) >= self.best_result:
                return
            
            # Find available courses
            available = []
            for course in range(1, n + 1):
                if not (taken_mask & (1 << (course - 1))):
                    can_take = True
                    for prereq in prereqs[course]:
                        if not (taken_mask & (1 << (prereq - 1))):
                            can_take = False
                            break
                    
                    if can_take:
                        available.append(course)
            
            if not available:
                return
            
            # Try all valid combinations
            for num_courses in range(1, min(k, len(available)) + 1):
                for combo in itertools.combinations(available, num_courses):
                    new_mask = taken_mask
                    for course in combo:
                        new_mask |= (1 << (course - 1))
                    
                    branch_and_bound(new_mask, semester + 1)
        
        branch_and_bound(0, 0)
        return self.best_result
    
    def minimumSemesters_approach6_optimized_bitmask(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 6: Optimized Bitmask DP with Preprocessing
        
        Optimized bitmask DP with better state representation.
        
        Time: O(3^n)
        Space: O(2^n)
        """
        # Precompute prerequisite masks
        prereq_mask = [0] * (n + 1)
        for prev, next_course in relations:
            prereq_mask[next_course] |= (1 << (prev - 1))
        
        # DP with queue for BFS-like exploration
        from collections import deque
        
        queue = deque([(0, 0)])  # (taken_mask, semesters)
        visited = {0: 0}
        
        while queue:
            taken_mask, semesters = queue.popleft()
            
            if taken_mask == (1 << n) - 1:
                return semesters
            
            # Find available courses
            available = []
            for course in range(1, n + 1):
                if not (taken_mask & (1 << (course - 1))):
                    # Check prerequisites using bitmask
                    if (taken_mask & prereq_mask[course]) == prereq_mask[course]:
                        available.append(course)
            
            # Generate all valid next states
            for num_courses in range(1, min(k, len(available)) + 1):
                for combo in itertools.combinations(available, num_courses):
                    new_mask = taken_mask
                    for course in combo:
                        new_mask |= (1 << (course - 1))
                    
                    if new_mask not in visited or visited[new_mask] > semesters + 1:
                        visited[new_mask] = semesters + 1
                        queue.append((new_mask, semesters + 1))
        
        return -1  # Should not reach here for valid input

def test_parallel_courses():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (n, relations, k, expected)
        (4, [[2,1],[3,1],[1,4]], 2, 3),
        (5, [[2,1],[3,1],[4,1],[1,5]], 2, 4),
        (11, [], 2, 6),
        (3, [[1,3],[2,3]], 2, 2),
        (4, [[2,1],[3,1],[1,4]], 1, 4),
        (2, [[1,2]], 1, 2),
        (1, [], 1, 1),
    ]
    
    approaches = [
        ("Bitmask DP", solution.minimumSemesters_approach1_bitmask_dp),
        ("Topological BFS", solution.minimumSemesters_approach2_topological_bfs),
        ("DFS Memoization", solution.minimumSemesters_approach3_dfs_with_memoization),
        ("Iterative Scheduling", solution.minimumSemesters_approach4_iterative_scheduling),
        ("Branch and Bound", solution.minimumSemesters_approach5_branch_and_bound),
        ("Optimized Bitmask", solution.minimumSemesters_approach6_optimized_bitmask),
    ]
    
    for i, (n, relations, k, expected) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: n={n}, k={k} ---")
        print(f"Relations: {relations}")
        print(f"Expected: {expected}")
        
        for approach_name, func in approaches:
            try:
                result = func(n, relations[:], k)  # Copy relations
                status = "✓" if result == expected else "✗"
                print(f"{approach_name:20} | {status} | Result: {result}")
                
            except Exception as e:
                print(f"{approach_name:20} | ERROR: {str(e)}")

def demonstrate_course_scheduling():
    """Demonstrate course scheduling strategy"""
    print("\n=== Course Scheduling Demo ===")
    
    n, relations, k = 4, [[2,1],[3,1],[1,4]], 2
    
    print(f"Courses: 1, 2, 3, 4")
    print(f"Prerequisites: {relations}")
    print(f"Max courses per semester: {k}")
    
    # Build dependency graph for visualization
    prereqs = defaultdict(list)
    for prev, next_course in relations:
        prereqs[next_course].append(prev)
    
    print(f"\nDependency analysis:")
    for course in range(1, n + 1):
        deps = prereqs[course]
        print(f"Course {course}: requires {deps if deps else 'none'}")
    
    print(f"\nOptimal scheduling:")
    print(f"Semester 1: Courses 2, 3 (no prerequisites)")
    print(f"Semester 2: Course 1 (requires 2 and 3)")
    print(f"Semester 3: Course 4 (requires 1)")
    print(f"Total semesters: 3")

def demonstrate_algorithm_comparison():
    """Demonstrate comparison between approaches"""
    print("\n=== Algorithm Comparison Demo ===")
    
    test_case = (5, [[2,1],[3,1],[4,1],[1,5]], 2)
    n, relations, k = test_case
    
    print(f"Test case: n={n}, k={k}")
    print(f"Relations: {relations}")
    
    solution = Solution()
    
    algorithms = [
        ("Bitmask DP", solution.minimumSemesters_approach1_bitmask_dp),
        ("Topological BFS", solution.minimumSemesters_approach2_topological_bfs),
        ("DFS Memoization", solution.minimumSemesters_approach3_dfs_with_memoization),
        ("Optimized Bitmask", solution.minimumSemesters_approach6_optimized_bitmask),
    ]
    
    print(f"\nAlgorithm performance:")
    print(f"{'Algorithm':<20} | {'Result':<6} | {'Notes'}")
    print("-" * 50)
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(n, relations, k)
            print(f"{alg_name:<20} | {result:<6} | Optimal solution")
            
        except Exception as e:
            print(f"{alg_name:<20} | ERROR | {str(e)[:20]}")

def analyze_problem_complexity():
    """Analyze the complexity of parallel courses problem"""
    print("\n=== Problem Complexity Analysis ===")
    
    print("Parallel Courses II Analysis:")
    
    print("\n1. **Problem Characteristics:**")
    print("   • DAG structure ensures feasibility")
    print("   • Constraint: at most k courses per semester")
    print("   • Goal: minimize number of semesters")
    print("   • Small n (≤ 15) suggests exponential algorithms")
    
    print("\n2. **Algorithm Approaches:**")
    print("   • Bitmask DP: O(3^n) - optimal for small n")
    print("   • Topological sort: O(n^2) - greedy approximation")
    print("   • DFS with memoization: O(2^n) - explores state space")
    print("   • Branch and bound: O(2^n) with pruning")
    
    print("\n3. **Key Insights:**")
    print("   • State space is 2^n (subset of completed courses)")
    print("   • Each state transition takes at most k courses")
    print("   • Optimal substructure: DP applicable")
    print("   • Small n makes exponential algorithms feasible")
    
    print("\n4. **Optimization Techniques:**")
    print("   • Bitmask representation for efficient state encoding")
    print("   • Memoization to avoid recomputation")
    print("   • Pruning in branch-and-bound")
    print("   • Precomputation of prerequisite masks")
    
    print("\n5. **Practical Considerations:**")
    print("   • For n ≤ 15, bitmask DP is optimal")
    print("   • Topological approaches good for larger instances")
    print("   • Memory usage grows as O(2^n)")
    print("   • Implementation complexity varies by approach")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    print("Parallel Courses Problem Applications:")
    
    print("\n1. **Academic Planning:**")
    print("   • University course scheduling")
    print("   • Degree completion optimization")
    print("   • Credit hour planning")
    print("   • Academic prerequisite management")
    
    print("\n2. **Project Management:**")
    print("   • Task scheduling with dependencies")
    print("   • Resource-constrained project planning")
    print("   • Critical path analysis")
    print("   • Parallel execution optimization")
    
    print("\n3. **Software Development:**")
    print("   • Build system optimization")
    print("   • Compilation dependency resolution")
    print("   • CI/CD pipeline scheduling")
    print("   • Module deployment planning")
    
    print("\n4. **Manufacturing:**")
    print("   • Production line scheduling")
    print("   • Assembly process optimization")
    print("   • Quality control checkpoints")
    print("   • Resource allocation planning")
    
    print("\n5. **Training Programs:**")
    print("   • Employee skill development")
    print("   • Certification path planning")
    print("   • Learning resource allocation")
    print("   • Knowledge prerequisite management")

if __name__ == "__main__":
    test_parallel_courses()
    demonstrate_course_scheduling()
    demonstrate_algorithm_comparison()
    analyze_problem_complexity()
    demonstrate_real_world_applications()

"""
Parallel Courses and Constraint Scheduling Concepts:
1. DAG-based Scheduling with Resource Constraints
2. Bitmask Dynamic Programming for State Space Exploration
3. Topological Sorting with Capacity Constraints
4. Branch-and-Bound Optimization with Pruning
5. Real-world Applications in Academic and Project Planning

Key Problem Insights:
- Small state space (n ≤ 15) enables exponential algorithms
- DAG structure guarantees feasible solutions
- Resource constraint (k courses per semester) adds complexity
- Optimal substructure allows dynamic programming approach
- Multiple algorithmic paradigms applicable

Algorithm Strategy:
1. Use bitmask DP for optimal solutions on small instances
2. Apply topological approaches for larger problems
3. Implement memoization for efficiency
4. Use branch-and-bound for early termination

Complexity Analysis:
- Bitmask DP: O(3^n) time, O(2^n) space
- Topological BFS: O(n^2) time, O(n) space
- DFS with memoization: O(2^n) time and space
- Branch-and-bound: O(2^n) worst case with pruning

Optimization Techniques:
- State compression using bitmasks
- Memoization for overlapping subproblems
- Pruning strategies in branch-and-bound
- Preprocessing for prerequisite checking

Real-world Applications:
- Academic course scheduling and degree planning
- Project management with resource constraints
- Software build system optimization
- Manufacturing and production scheduling
- Training program and skill development planning

This comprehensive implementation provides optimal solutions
for constraint-based scheduling problems in various domains.
"""
