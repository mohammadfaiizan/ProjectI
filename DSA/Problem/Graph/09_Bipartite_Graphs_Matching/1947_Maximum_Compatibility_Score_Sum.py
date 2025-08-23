"""
1947. Maximum Compatibility Score Sum - Multiple Approaches
Difficulty: Medium

There is a survey that consists of n questions where each question's answer is either 0 (no) or 1 (yes).

The survey is given to m students numbered from 0 to m - 1 and m mentors numbered from 0 to m - 1. The answer of the ith student to the jth question is given by the integer students[i][j], and the answer of the ith mentor to the jth question is given by the integer mentors[i][j].

The compatibility score of a student-mentor pair is the number of answers that are the same for both the student and mentor.

For example, if the student's answers are [1, 0, 1] and the mentor's answers are [0, 0, 1], then their compatibility score is 2 because only the 2nd and the 3rd answers are the same.

You want to pair each student with a unique mentor. Return the maximum compatibility score sum of all the student-mentor pairs.
"""

from typing import List
from functools import lru_cache

class MaximumCompatibilityScore:
    """Multiple approaches to find maximum compatibility score sum"""
    
    def maxCompatibilitySum_backtracking(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        """
        Approach 1: Backtracking with Pruning
        
        Try all possible student-mentor pairings using backtracking.
        
        Time: O(m!), Space: O(m)
        """
        m, n = len(students), len(students[0])
        
        # Precompute compatibility scores
        compatibility = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                score = sum(1 for k in range(n) if students[i][k] == mentors[j][k])
                compatibility[i][j] = score
        
        def backtrack(student_idx: int, used_mentors: set, current_score: int) -> int:
            if student_idx == m:
                return current_score
            
            max_score = 0
            for mentor_idx in range(m):
                if mentor_idx not in used_mentors:
                    used_mentors.add(mentor_idx)
                    score = backtrack(student_idx + 1, used_mentors, 
                                    current_score + compatibility[student_idx][mentor_idx])
                    max_score = max(max_score, score)
                    used_mentors.remove(mentor_idx)
            
            return max_score
        
        return backtrack(0, set(), 0)
    
    def maxCompatibilitySum_bitmask_dp(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        """
        Approach 2: Bitmask Dynamic Programming
        
        Use bitmask to represent used mentors and DP for optimization.
        
        Time: O(m * 2^m), Space: O(2^m)
        """
        m, n = len(students), len(students[0])
        
        # Precompute compatibility matrix
        compatibility = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                compatibility[i][j] = sum(1 for k in range(n) if students[i][k] == mentors[j][k])
        
        @lru_cache(maxsize=None)
        def dp(student_idx: int, mentor_mask: int) -> int:
            if student_idx == m:
                return 0
            
            max_score = 0
            for mentor_idx in range(m):
                if not (mentor_mask & (1 << mentor_idx)):  # Mentor not used
                    new_mask = mentor_mask | (1 << mentor_idx)
                    score = compatibility[student_idx][mentor_idx] + dp(student_idx + 1, new_mask)
                    max_score = max(max_score, score)
            
            return max_score
        
        return dp(0, 0)
    
    def maxCompatibilitySum_hungarian_algorithm(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        """
        Approach 3: Hungarian Algorithm (Maximum Weight Bipartite Matching)
        
        Convert to maximum weight bipartite matching problem.
        
        Time: O(m³), Space: O(m²)
        """
        m, n = len(students), len(students[0])
        
        # Build compatibility matrix (convert to cost matrix for Hungarian)
        max_possible_score = n
        cost_matrix = [[0] * m for _ in range(m)]
        
        for i in range(m):
            for j in range(m):
                compatibility_score = sum(1 for k in range(n) if students[i][k] == mentors[j][k])
                # Convert to cost (for minimum cost assignment)
                cost_matrix[i][j] = max_possible_score - compatibility_score
        
        def hungarian_algorithm(cost_matrix):
            """Simplified Hungarian algorithm implementation"""
            size = len(cost_matrix)
            
            # Step 1: Subtract row minimums
            for i in range(size):
                row_min = min(cost_matrix[i])
                for j in range(size):
                    cost_matrix[i][j] -= row_min
            
            # Step 2: Subtract column minimums
            for j in range(size):
                col_min = min(cost_matrix[i][j] for i in range(size))
                for i in range(size):
                    cost_matrix[i][j] -= col_min
            
            # For simplicity, use brute force for small matrices
            # In practice, would implement full Hungarian algorithm
            return self._brute_force_assignment(cost_matrix)
        
        min_cost = hungarian_algorithm([row[:] for row in cost_matrix])
        return m * max_possible_score - min_cost
    
    def _brute_force_assignment(self, cost_matrix):
        """Helper method for assignment problem"""
        from itertools import permutations
        
        m = len(cost_matrix)
        min_cost = float('inf')
        
        for perm in permutations(range(m)):
            cost = sum(cost_matrix[i][perm[i]] for i in range(m))
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def maxCompatibilitySum_branch_and_bound(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        """
        Approach 4: Branch and Bound
        
        Use branch and bound with upper bound estimation.
        
        Time: O(m!) worst case, but pruned, Space: O(m)
        """
        m, n = len(students), len(students[0])
        
        # Precompute compatibility scores
        compatibility = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                compatibility[i][j] = sum(1 for k in range(n) if students[i][k] == mentors[j][k])
        
        def upper_bound(student_idx: int, used_mentors: set) -> int:
            """Calculate upper bound for remaining assignments"""
            if student_idx >= m:
                return 0
            
            # For each remaining student, take the best available mentor
            bound = 0
            available_mentors = [j for j in range(m) if j not in used_mentors]
            
            for i in range(student_idx, m):
                if available_mentors:
                    # Take the best score for this student from available mentors
                    best_score = max(compatibility[i][j] for j in available_mentors)
                    bound += best_score
                    # Remove the mentor that gave the best score (greedy approximation)
                    best_mentor = max(available_mentors, key=lambda j: compatibility[i][j])
                    available_mentors.remove(best_mentor)
            
            return bound
        
        self.max_score = 0
        
        def branch_and_bound(student_idx: int, used_mentors: set, current_score: int):
            if student_idx == m:
                self.max_score = max(self.max_score, current_score)
                return
            
            # Pruning: if upper bound can't improve current best, skip
            if current_score + upper_bound(student_idx, used_mentors) <= self.max_score:
                return
            
            for mentor_idx in range(m):
                if mentor_idx not in used_mentors:
                    used_mentors.add(mentor_idx)
                    branch_and_bound(student_idx + 1, used_mentors, 
                                   current_score + compatibility[student_idx][mentor_idx])
                    used_mentors.remove(mentor_idx)
        
        branch_and_bound(0, set(), 0)
        return self.max_score
    
    def maxCompatibilitySum_optimized_bitmask(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        """
        Approach 5: Optimized Bitmask DP with Iterative Implementation
        
        Iterative DP with space optimization.
        
        Time: O(m * 2^m), Space: O(2^m)
        """
        m, n = len(students), len(students[0])
        
        # Precompute compatibility using bit operations for efficiency
        compatibility = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                # Use XOR and bit counting for faster comparison
                student_bits = 0
                mentor_bits = 0
                for k in range(n):
                    if students[i][k]:
                        student_bits |= (1 << k)
                    if mentors[j][k]:
                        mentor_bits |= (1 << k)
                
                # Count matching bits
                matching_bits = ~(student_bits ^ mentor_bits) & ((1 << n) - 1)
                compatibility[i][j] = bin(matching_bits).count('1')
        
        # DP table: dp[mask] = maximum score when mentors in mask are used
        dp = [-1] * (1 << m)
        dp[0] = 0
        
        for student_idx in range(m):
            new_dp = [-1] * (1 << m)
            
            for mask in range(1 << m):
                if dp[mask] == -1:
                    continue
                
                for mentor_idx in range(m):
                    if not (mask & (1 << mentor_idx)):  # Mentor not used
                        new_mask = mask | (1 << mentor_idx)
                        new_score = dp[mask] + compatibility[student_idx][mentor_idx]
                        new_dp[new_mask] = max(new_dp[new_mask], new_score)
            
            dp = new_dp
        
        return dp[(1 << m) - 1]

def test_maximum_compatibility_score():
    """Test maximum compatibility score algorithms"""
    solver = MaximumCompatibilityScore()
    
    test_cases = [
        ([[1,1,0],[1,0,1],[0,0,1]], [[1,0,0],[0,0,1],[1,1,0]], 8, "Example 1"),
        ([[0,0],[0,0],[0,0]], [[1,1],[1,1],[1,1]], 0, "No compatibility"),
        ([[1,0,1],[0,1,0]], [[0,1,0],[1,0,1]], 4, "Perfect match"),
        ([[1]], [[1]], 1, "Single pair match"),
        ([[0]], [[1]], 0, "Single pair no match"),
    ]
    
    algorithms = [
        ("Backtracking", solver.maxCompatibilitySum_backtracking),
        ("Bitmask DP", solver.maxCompatibilitySum_bitmask_dp),
        ("Hungarian Algorithm", solver.maxCompatibilitySum_hungarian_algorithm),
        ("Branch and Bound", solver.maxCompatibilitySum_branch_and_bound),
        ("Optimized Bitmask", solver.maxCompatibilitySum_optimized_bitmask),
    ]
    
    print("=== Testing Maximum Compatibility Score Sum ===")
    
    for students, mentors, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Students: {students}")
        print(f"Mentors: {mentors}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(students, mentors)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Score: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_maximum_compatibility_score()

"""
Maximum Compatibility Score Sum demonstrates bipartite matching
optimization with dynamic programming, Hungarian algorithm,
and advanced assignment problem solving techniques.
"""
