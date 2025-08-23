"""
1494. Parallel Courses II - Multiple Approaches
Difficulty: Hard

You are given an integer n, which indicates that there are n courses labeled from 1 to n. 
You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], 
representing a prerequisite relationship between course prevCoursei and course nextCoursei: 
course prevCoursei has to be taken before course nextCoursei. Also, you are given the integer k.

In one semester, you can take at most k courses as long as you have taken all the prerequisites 
for the courses you are taking.

Return the minimum number of semesters needed to take all n courses.

The testcases will be generated such that it is possible to take every course.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq
from functools import lru_cache

class ParallelCoursesII:
    """Multiple approaches to solve parallel courses scheduling problem"""
    
    def minimumSemesters_bfs_topological(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 1: BFS Topological Sort with Greedy Selection
        
        Use topological sorting to identify available courses and greedily select k courses.
        
        Time: O(2^n * n) in worst case due to course selection
        Space: O(n + E)
        """
        # Build graph and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            in_degree[next_course] += 1
        
        semesters = 0
        completed = set()
        
        while len(completed) < n:
            # Find all courses with no prerequisites
            available = []
            for course in range(1, n + 1):
                if course not in completed and in_degree[course] == 0:
                    available.append(course)
            
            if not available:
                return -1  # Impossible (shouldn't happen per problem statement)
            
            # Take up to k courses this semester
            to_take = min(k, len(available))
            selected = available[:to_take]
            
            # Complete selected courses
            for course in selected:
                completed.add(course)
                # Update prerequisites for dependent courses
                for dependent in graph[course]:
                    in_degree[dependent] -= 1
            
            semesters += 1
        
        return semesters
    
    def minimumSemesters_dp_bitmask(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 2: Dynamic Programming with Bitmask
        
        Use bitmask DP to represent course completion states.
        
        Time: O(3^n * n) - for each state, try all subsets
        Space: O(2^n)
        """
        # Build prerequisite relationships
        prerequisites = [0] * (n + 1)  # Bitmask of prerequisites for each course
        
        for prev, next_course in relations:
            prerequisites[next_course] |= (1 << (prev - 1))
        
        @lru_cache(maxsize=None)
        def dp(completed_mask):
            """DP function: minimum semesters from current completion state"""
            if completed_mask == (1 << n) - 1:
                return 0  # All courses completed
            
            # Find available courses (prerequisites satisfied)
            available = []
            for course in range(1, n + 1):
                course_bit = 1 << (course - 1)
                if not (completed_mask & course_bit):  # Course not completed
                    if (completed_mask & prerequisites[course]) == prerequisites[course]:
                        available.append(course - 1)  # Convert to 0-indexed
            
            if not available:
                return float('inf')  # No available courses
            
            min_semesters = float('inf')
            
            # Try all possible combinations of up to k courses
            def generate_combinations(idx, current_combo, count):
                nonlocal min_semesters
                
                if count > 0:
                    # Try this combination
                    new_mask = completed_mask
                    for course_idx in current_combo:
                        new_mask |= (1 << course_idx)
                    
                    result = 1 + dp(new_mask)
                    min_semesters = min(min_semesters, result)
                
                if idx >= len(available) or count >= k:
                    return
                
                # Include current course
                current_combo.append(available[idx])
                generate_combinations(idx + 1, current_combo, count + 1)
                current_combo.pop()
                
                # Skip current course
                generate_combinations(idx + 1, current_combo, count)
            
            generate_combinations(0, [], 0)
            return min_semesters
        
        return dp(0)
    
    def minimumSemesters_optimized_bitmask(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 3: Optimized Bitmask DP
        
        More efficient bitmask DP with better subset enumeration.
        
        Time: O(3^n)
        Space: O(2^n)
        """
        # Build prerequisite bitmasks
        prereq = [0] * n
        for prev, next_course in relations:
            prereq[next_course - 1] |= (1 << (prev - 1))
        
        # DP array: dp[mask] = minimum semesters to complete courses in mask
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Find available courses
            available_mask = 0
            for i in range(n):
                if not (mask & (1 << i)) and (mask & prereq[i]) == prereq[i]:
                    available_mask |= (1 << i)
            
            # Try all subsets of available courses with size <= k
            submask = available_mask
            while submask > 0:
                if bin(submask).count('1') <= k:
                    new_mask = mask | submask
                    dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
                
                submask = (submask - 1) & available_mask
        
        return dp[(1 << n) - 1]
    
    def minimumSemesters_priority_queue(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 4: Priority Queue with Course Prioritization
        
        Use priority queue to select most "valuable" courses first.
        
        Time: O(n^2 * log n)
        Space: O(n + E)
        """
        # Build graph and calculate metrics
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)
        out_degree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            in_degree[next_course] += 1
            out_degree[prev] += 1
        
        def calculate_priority(course):
            """Calculate priority score for course selection"""
            # Higher priority for courses that unlock more courses
            return out_degree[course] * 10 + (n - course)  # Tie-breaker by course number
        
        semesters = 0
        completed = set()
        
        while len(completed) < n:
            # Find available courses with priorities
            available = []
            for course in range(1, n + 1):
                if course not in completed and in_degree[course] == 0:
                    priority = calculate_priority(course)
                    heapq.heappush(available, (-priority, course))  # Max heap
            
            if not available:
                return -1
            
            # Select up to k highest priority courses
            selected = []
            for _ in range(min(k, len(available))):
                if available:
                    _, course = heapq.heappop(available)
                    selected.append(course)
            
            # Complete selected courses
            for course in selected:
                completed.add(course)
                for dependent in graph[course]:
                    in_degree[dependent] -= 1
            
            semesters += 1
        
        return semesters
    
    def minimumSemesters_branch_and_bound(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 5: Branch and Bound with Pruning
        
        Use branch and bound to explore course selection with pruning.
        
        Time: O(exponential) but with pruning
        Space: O(n)
        """
        # Build prerequisites
        prereq = [set() for _ in range(n + 1)]
        dependents = defaultdict(list)
        
        for prev, next_course in relations:
            prereq[next_course].add(prev)
            dependents[prev].append(next_course)
        
        self.best_semesters = float('inf')
        
        def backtrack(completed, current_semester):
            if len(completed) == n:
                self.best_semesters = min(self.best_semesters, current_semester)
                return
            
            # Pruning: if current path already exceeds best, stop
            if current_semester >= self.best_semesters:
                return
            
            # Lower bound: remaining courses / k (optimistic)
            remaining = n - len(completed)
            lower_bound = current_semester + (remaining + k - 1) // k
            if lower_bound >= self.best_semesters:
                return
            
            # Find available courses
            available = []
            for course in range(1, n + 1):
                if course not in completed and prereq[course].issubset(completed):
                    available.append(course)
            
            if not available:
                return
            
            # Try different combinations of up to k courses
            def try_combinations(idx, current_selection):
                if len(current_selection) > 0:
                    # Try this selection
                    new_completed = completed | set(current_selection)
                    backtrack(new_completed, current_semester + 1)
                
                if idx >= len(available) or len(current_selection) >= k:
                    return
                
                # Include current course
                current_selection.append(available[idx])
                try_combinations(idx + 1, current_selection)
                current_selection.pop()
                
                # Skip current course (only if we have other options)
                if len(current_selection) > 0 or idx < len(available) - 1:
                    try_combinations(idx + 1, current_selection)
            
            try_combinations(0, [])
        
        backtrack(set(), 0)
        return self.best_semesters if self.best_semesters != float('inf') else -1
    
    def minimumSemesters_level_bfs(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 6: Level-wise BFS with State Compression
        
        BFS through different completion states level by level.
        
        Time: O(2^n * n)
        Space: O(2^n)
        """
        # Build prerequisite relationships
        prereq_mask = [0] * n
        for prev, next_course in relations:
            prereq_mask[next_course - 1] |= (1 << (prev - 1))
        
        # BFS queue: completion state (bitmask)
        queue = deque([0])
        visited = {0}
        semesters = 0
        
        while queue:
            # Process all states at current level
            level_size = len(queue)
            
            for _ in range(level_size):
                current_state = queue.popleft()
                
                if current_state == (1 << n) - 1:
                    return semesters
                
                # Find available courses
                available = []
                for i in range(n):
                    if not (current_state & (1 << i)):  # Course not taken
                        if (current_state & prereq_mask[i]) == prereq_mask[i]:
                            available.append(i)
                
                # Generate all valid next states (taking up to k courses)
                def generate_next_states(idx, current_selection, count):
                    if count > 0:
                        # Create new state
                        new_state = current_state
                        for course_idx in current_selection:
                            new_state |= (1 << course_idx)
                        
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append(new_state)
                    
                    if idx >= len(available) or count >= k:
                        return
                    
                    # Include current course
                    current_selection.append(available[idx])
                    generate_next_states(idx + 1, current_selection, count + 1)
                    current_selection.pop()
                    
                    # Skip current course
                    generate_next_states(idx + 1, current_selection, count)
                
                generate_next_states(0, [], 0)
            
            semesters += 1
        
        return -1  # Should not reach here per problem constraints

def test_parallel_courses_ii():
    """Test all approaches with various test cases"""
    solver = ParallelCoursesII()
    
    test_cases = [
        # (n, relations, k, expected, description)
        (4, [[2,1],[3,1],[1,4]], 2, 3, "Basic case"),
        (5, [[2,1],[3,1],[4,1],[1,5]], 2, 4, "Star dependency"),
        (11, [], 2, 6, "No dependencies"),
        (4, [[2,1],[3,1],[1,4]], 3, 2, "High capacity"),
        (3, [[1,3],[2,3]], 2, 2, "Parallel prerequisites"),
        (7, [[1,2],[1,3],[2,4],[3,4],[2,5],[3,6],[4,7],[5,7],[6,7]], 2, 4, "Complex DAG"),
    ]
    
    approaches = [
        ("BFS Topological", solver.minimumSemesters_bfs_topological),
        ("DP Bitmask", solver.minimumSemesters_dp_bitmask),
        ("Optimized Bitmask", solver.minimumSemesters_optimized_bitmask),
        ("Priority Queue", solver.minimumSemesters_priority_queue),
        ("Branch & Bound", solver.minimumSemesters_branch_and_bound),
        ("Level BFS", solver.minimumSemesters_level_bfs),
    ]
    
    print("=== Testing Parallel Courses II ===")
    
    for n, relations, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n={n}, k={k}, relations={relations}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(n, relations, k)
                status = "✓" if result == expected else "✗"
                print(f"{approach_name:18} | {status} | Result: {result}")
            except Exception as e:
                print(f"{approach_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_course_scheduling_analysis():
    """Demonstrate course scheduling problem analysis"""
    print("\n=== Course Scheduling Analysis ===")
    
    # Example problem
    n, k = 7, 2
    relations = [[1,2],[1,3],[2,4],[3,4],[2,5],[3,6],[4,7],[5,7],[6,7]]
    
    print(f"Problem: {n} courses, capacity {k} per semester")
    print(f"Prerequisites: {relations}")
    
    # Build dependency graph for visualization
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)
    
    for prev, next_course in relations:
        graph[prev].append(next_course)
        in_degree[next_course] += 1
    
    print(f"\nDependency Analysis:")
    for course in range(1, n + 1):
        deps = [prev for prev, next_c in relations if next_c == course]
        dependents = graph[course]
        print(f"Course {course}: requires {deps}, enables {dependents}")
    
    # Show semester-by-semester solution
    solver = ParallelCoursesII()
    result = solver.minimumSemesters_bfs_topological(n, relations, k)
    print(f"\nMinimum semesters needed: {result}")
    
    print(f"\nOptimal Strategy:")
    print(f"• Identify courses with no prerequisites first")
    print(f"• Prioritize courses that unlock many others")
    print(f"• Balance workload across semesters")
    print(f"• Consider long prerequisite chains early")

def analyze_algorithm_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **BFS Topological Sort:**")
    print("   • Time: O(V + E) per semester, O(V²) total")
    print("   • Space: O(V + E)")
    print("   • Pros: Simple, intuitive")
    print("   • Cons: Greedy may not be optimal")
    
    print("\n2. **DP with Bitmask:**")
    print("   • Time: O(3^n * n) - exponential")
    print("   • Space: O(2^n)")
    print("   • Pros: Guaranteed optimal")
    print("   • Cons: Exponential for large n")
    
    print("\n3. **Optimized Bitmask:**")
    print("   • Time: O(3^n) - better constant factors")
    print("   • Space: O(2^n)")
    print("   • Pros: More efficient subset enumeration")
    print("   • Cons: Still exponential")
    
    print("\n4. **Priority Queue:**")
    print("   • Time: O(V² log V)")
    print("   • Space: O(V + E)")
    print("   • Pros: Good heuristic performance")
    print("   • Cons: May not find optimal solution")
    
    print("\n5. **Branch and Bound:**")
    print("   • Time: Exponential with pruning")
    print("   • Space: O(V)")
    print("   • Pros: Optimal with good pruning")
    print("   • Cons: Worst-case exponential")
    
    print("\n6. **Level-wise BFS:**")
    print("   • Time: O(2^n * V)")
    print("   • Space: O(2^n)")
    print("   • Pros: Systematic state exploration")
    print("   • Cons: Memory intensive")
    
    print("\nRecommendations:")
    print("• n ≤ 15: Use bitmask DP for optimal solution")
    print("• n > 15: Use topological sort with heuristics")
    print("• Real-time: Priority queue with good heuristics")
    print("• Memory constrained: Branch and bound")

def demonstrate_optimization_strategies():
    """Demonstrate optimization strategies for course scheduling"""
    print("\n=== Optimization Strategies ===")
    
    print("Course Selection Heuristics:")
    
    print("\n1. **Prerequisite Chain Length:**")
    print("   • Prioritize courses in long dependency chains")
    print("   • Early completion prevents bottlenecks")
    print("   • Calculate critical path through dependencies")
    
    print("\n2. **Unlocking Potential:**")
    print("   • Favor courses that enable many others")
    print("   • High out-degree courses are valuable")
    print("   • Consider transitive dependencies")
    
    print("\n3. **Load Balancing:**")
    print("   • Distribute difficult courses across semesters")
    print("   • Avoid overloading any single semester")
    print("   • Consider course difficulty weights")
    
    print("\n4. **Flexibility Preservation:**")
    print("   • Keep multiple options available")
    print("   • Avoid creating unnecessary bottlenecks")
    print("   • Maintain scheduling flexibility")
    
    print("\n5. **Lookahead Planning:**")
    print("   • Consider future semester implications")
    print("   • Plan for prerequisite satisfaction")
    print("   • Anticipate capacity constraints")
    
    print("\nPractical Considerations:")
    print("• Course offering schedules (fall/spring only)")
    print("• Professor availability and capacity")
    print("• Student workload and difficulty balance")
    print("• Co-requisite and soft prerequisite handling")
    print("• Graduation timeline constraints")

if __name__ == "__main__":
    test_parallel_courses_ii()
    demonstrate_course_scheduling_analysis()
    analyze_algorithm_complexity()
    demonstrate_optimization_strategies()

"""
Parallel Courses II - Key Insights:

1. **Problem Structure:**
   - DAG scheduling with capacity constraints
   - Minimize makespan (total time) with limited parallelism
   - Prerequisite dependencies must be respected
   - Optimal substructure enables dynamic programming

2. **Algorithm Categories:**
   - Greedy: Fast but potentially suboptimal
   - Dynamic Programming: Optimal but exponential space
   - Branch and Bound: Optimal with pruning
   - Heuristic: Good performance with approximation

3. **Key Challenges:**
   - Exponential state space (2^n possible completions)
   - Course selection optimization within capacity
   - Prerequisite dependency management
   - Balance between optimality and efficiency

4. **Optimization Techniques:**
   - Bitmask representation for efficient state handling
   - Pruning strategies in branch and bound
   - Priority-based course selection heuristics
   - State compression and memoization

5. **Real-World Applications:**
   - Academic course planning and scheduling
   - Project task scheduling with dependencies
   - Manufacturing process optimization
   - Software build and deployment pipelines

The problem combines topological sorting with constrained
optimization, requiring careful balance of efficiency and optimality.
"""
