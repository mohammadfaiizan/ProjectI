"""
1494. Parallel Courses II - Multiple Approaches
Difficulty: Hard

You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei: course prevCoursei has to be taken before course nextCoursei. Furthermore, you are given the integer k.

In one semester, you can take at most k courses as long as you have taken all the prerequisites in the previous semesters for the courses you are taking.

Return the minimum number of semesters needed to take all courses. The testcases will be generated such that it is possible to take every course.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque
from functools import lru_cache

class ParallelCoursesII:
    """Multiple approaches to solve parallel courses with capacity constraint"""
    
    def minNumberOfSemesters_bitmask_dp(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 1: Bitmask Dynamic Programming
        
        Use bitmask to represent taken courses and DP to find minimum semesters.
        
        Time: O(3^n), Space: O(2^n)
        """
        # Build prerequisite mask for each course
        prereq = [0] * n
        for prev, next_course in relations:
            prereq[next_course - 1] |= (1 << (prev - 1))
        
        @lru_cache(maxsize=None)
        def dp(mask: int) -> int:
            """Return minimum semesters needed to complete remaining courses"""
            if mask == (1 << n) - 1:  # All courses taken
                return 0
            
            # Find courses that can be taken (prerequisites satisfied)
            available = []
            for i in range(n):
                if not (mask & (1 << i)) and (mask & prereq[i]) == prereq[i]:
                    available.append(i)
            
            if not available:
                return float('inf')  # Should not happen in valid input
            
            min_semesters = float('inf')
            
            # Try all possible combinations of up to k courses
            def generate_combinations(idx: int, current_combo: int, count: int):
                nonlocal min_semesters
                
                if count > 0:
                    # Try taking this combination
                    new_mask = mask | current_combo
                    min_semesters = min(min_semesters, 1 + dp(new_mask))
                
                if idx == len(available) or count == k:
                    return
                
                # Include current course
                generate_combinations(idx + 1, current_combo | (1 << available[idx]), count + 1)
                
                # Exclude current course
                generate_combinations(idx + 1, current_combo, count)
            
            generate_combinations(0, 0, 0)
            return min_semesters
        
        return dp(0)
    
    def minNumberOfSemesters_bfs_state_space(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 2: BFS on State Space
        
        Use BFS to explore all possible states of taken courses.
        
        Time: O(2^n * C(n,k)), Space: O(2^n)
        """
        # Build prerequisite masks
        prereq = [0] * n
        for prev, next_course in relations:
            prereq[next_course - 1] |= (1 << (prev - 1))
        
        # BFS
        queue = deque([(0, 0)])  # (mask, semesters)
        visited = set([0])
        target = (1 << n) - 1
        
        while queue:
            mask, semesters = queue.popleft()
            
            if mask == target:
                return semesters
            
            # Find available courses
            available = []
            for i in range(n):
                if not (mask & (1 << i)) and (mask & prereq[i]) == prereq[i]:
                    available.append(i)
            
            # Generate all valid combinations of up to k courses
            def generate_next_states(idx: int, current_mask: int, count: int):
                if count > 0:
                    new_mask = mask | current_mask
                    if new_mask not in visited:
                        visited.add(new_mask)
                        queue.append((new_mask, semesters + 1))
                
                if idx == len(available) or count == k:
                    return
                
                # Include current course
                generate_next_states(idx + 1, current_mask | (1 << available[idx]), count + 1)
                
                # Exclude current course
                generate_next_states(idx + 1, current_mask, count)
            
            generate_next_states(0, 0, 0)
        
        return -1  # Should not reach here
    
    def minNumberOfSemesters_greedy_with_backtrack(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 3: Greedy with Backtracking
        
        Use greedy selection with backtracking for optimization.
        
        Time: O(exponential), Space: O(n)
        """
        # Build graph
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)
        
        for prev, next_course in relations:
            graph[prev].append(next_course)
            in_degree[next_course] += 1
        
        def backtrack(taken: Set[int], semesters: int) -> int:
            if len(taken) == n:
                return semesters
            
            # Find available courses
            available = []
            for i in range(1, n + 1):
                if i not in taken:
                    # Check if prerequisites are satisfied
                    can_take = True
                    for prev in range(1, n + 1):
                        if i in graph[prev] and prev not in taken:
                            can_take = False
                            break
                    if can_take:
                        available.append(i)
            
            if not available:
                return float('inf')
            
            min_result = float('inf')
            
            # Try all combinations of up to k courses
            def try_combinations(idx: int, current_selection: List[int]):
                nonlocal min_result
                
                if current_selection:
                    new_taken = taken | set(current_selection)
                    result = backtrack(new_taken, semesters + 1)
                    min_result = min(min_result, result)
                
                if idx == len(available) or len(current_selection) == k:
                    return
                
                # Include current course
                try_combinations(idx + 1, current_selection + [available[idx]])
                
                # Exclude current course
                try_combinations(idx + 1, current_selection)
            
            try_combinations(0, [])
            return min_result
        
        return backtrack(set(), 0)
    
    def minNumberOfSemesters_optimized_bitmask(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 4: Optimized Bitmask DP
        
        Optimized version with better pruning and state management.
        
        Time: O(3^n), Space: O(2^n)
        """
        # Precompute prerequisites
        prereq = [0] * n
        for prev, next_course in relations:
            prereq[next_course - 1] |= (1 << (prev - 1))
        
        # Precompute all valid subsets of size <= k
        valid_subsets = []
        for mask in range(1 << n):
            if bin(mask).count('1') <= k:
                valid_subsets.append(mask)
        
        # DP with memoization
        dp = {}
        
        def solve(taken_mask: int) -> int:
            if taken_mask in dp:
                return dp[taken_mask]
            
            if taken_mask == (1 << n) - 1:
                return 0
            
            # Find available courses
            available_mask = 0
            for i in range(n):
                if not (taken_mask & (1 << i)) and (taken_mask & prereq[i]) == prereq[i]:
                    available_mask |= (1 << i)
            
            min_semesters = float('inf')
            
            # Try all valid subsets of available courses
            for subset in valid_subsets:
                if (subset & available_mask) == subset and subset != 0:
                    new_mask = taken_mask | subset
                    min_semesters = min(min_semesters, 1 + solve(new_mask))
            
            dp[taken_mask] = min_semesters
            return min_semesters
        
        return solve(0)
    
    def minNumberOfSemesters_iterative_dp(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        Approach 5: Iterative DP (Bottom-up)
        
        Use iterative DP to avoid recursion overhead.
        
        Time: O(3^n), Space: O(2^n)
        """
        # Build prerequisite masks
        prereq = [0] * n
        for prev, next_course in relations:
            prereq[next_course - 1] |= (1 << (prev - 1))
        
        # DP array
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        # Process all states
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Find available courses
            available = []
            for i in range(n):
                if not (mask & (1 << i)) and (mask & prereq[i]) == prereq[i]:
                    available.append(i)
            
            # Try all combinations of up to k available courses
            def generate_combinations(idx: int, current_combo: int, count: int):
                if count > 0:
                    new_mask = mask | current_combo
                    dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
                
                if idx == len(available) or count == k:
                    return
                
                # Include current course
                generate_combinations(idx + 1, current_combo | (1 << available[idx]), count + 1)
                
                # Exclude current course
                generate_combinations(idx + 1, current_combo, count)
            
            generate_combinations(0, 0, 0)
        
        return dp[(1 << n) - 1]

def test_parallel_courses_ii():
    """Test parallel courses II algorithms"""
    solver = ParallelCoursesII()
    
    test_cases = [
        (4, [[2,1],[3,1],[1,4]], 2, 3, "Example 1"),
        (5, [[2,1],[3,1],[4,1],[1,5]], 2, 4, "Example 2"),
        (11, [], 2, 6, "No prerequisites"),
        (3, [[1,3],[2,3]], 2, 2, "Simple case"),
    ]
    
    algorithms = [
        ("Bitmask DP", solver.minNumberOfSemesters_bitmask_dp),
        ("BFS State Space", solver.minNumberOfSemesters_bfs_state_space),
        ("Optimized Bitmask", solver.minNumberOfSemesters_optimized_bitmask),
        ("Iterative DP", solver.minNumberOfSemesters_iterative_dp),
    ]
    
    print("=== Testing Parallel Courses II ===")
    
    for n, relations, k, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, relations={relations}, k={k}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, relations, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Semesters: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_parallel_courses_ii()

"""
Parallel Courses II demonstrates advanced dynamic programming
with bitmask state representation for constrained scheduling
problems with capacity limitations.
"""
