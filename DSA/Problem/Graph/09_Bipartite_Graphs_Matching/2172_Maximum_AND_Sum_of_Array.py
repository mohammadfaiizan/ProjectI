"""
2172. Maximum AND Sum of Array - Multiple Approaches
Difficulty: Hard

You are given an integer array nums of length n and an integer numSlots such that 2 * numSlots >= n. There are numSlots slots numbered from 1 to numSlots.

You have to place all n integers into the slots such that each slot contains at most 2 numbers. The AND sum of a given placement is the sum of the bitwise AND of every number with its slot number.

For example, the AND sum of placing the numbers [1, 3] into slot 2 is (1 AND 2) + (3 AND 2) = 0 + 2 = 2.

Return the maximum possible AND sum of nums.
"""

from typing import List
from functools import lru_cache

class MaximumANDSum:
    """Multiple approaches to find maximum AND sum"""
    
    def maximumANDSum_bitmask_dp(self, nums: List[int], numSlots: int) -> int:
        """
        Approach 1: Bitmask Dynamic Programming
        
        Use bitmask to represent slot occupancy state.
        
        Time: O(n * 3^numSlots), Space: O(3^numSlots)
        """
        n = len(nums)
        
        # Each slot can have 0, 1, or 2 numbers
        # Use base-3 representation: 0=empty, 1=one number, 2=two numbers
        @lru_cache(maxsize=None)
        def dp(idx: int, state: int) -> int:
            """
            dp(idx, state) = maximum AND sum placing nums[idx:] 
            with current slot state
            """
            if idx == n:
                return 0
            
            max_sum = 0
            temp_state = state
            
            # Try placing nums[idx] in each slot
            for slot in range(numSlots):
                # Extract slot occupancy (base-3 digit)
                slot_count = temp_state % 3
                temp_state //= 3
                
                if slot_count < 2:  # Slot has space
                    # Calculate AND contribution
                    and_value = nums[idx] & (slot + 1)
                    
                    # Update state for this slot
                    new_state = state + (3 ** slot)
                    
                    # Recurse
                    total_sum = and_value + dp(idx + 1, new_state)
                    max_sum = max(max_sum, total_sum)
            
            return max_sum
        
        return dp(0, 0)
    
    def maximumANDSum_optimized_bitmask(self, nums: List[int], numSlots: int) -> int:
        """
        Approach 2: Optimized Bitmask with Iterative DP
        
        Use iterative DP to avoid recursion overhead.
        
        Time: O(n * 3^numSlots), Space: O(3^numSlots)
        """
        n = len(nums)
        
        # Calculate total states (3^numSlots)
        total_states = 3 ** numSlots
        
        # DP table: dp[state] = maximum AND sum for this state
        dp = [-1] * total_states
        dp[0] = 0
        
        for idx in range(n):
            new_dp = [-1] * total_states
            
            for state in range(total_states):
                if dp[state] == -1:
                    continue
                
                # Try placing nums[idx] in each slot
                temp_state = state
                for slot in range(numSlots):
                    slot_count = temp_state % 3
                    temp_state //= 3
                    
                    if slot_count < 2:  # Slot has space
                        and_value = nums[idx] & (slot + 1)
                        new_state = state + (3 ** slot)
                        
                        new_dp[new_state] = max(new_dp[new_state], 
                                              dp[state] + and_value)
            
            dp = new_dp
        
        return max(val for val in dp if val != -1)
    
    def maximumANDSum_backtracking(self, nums: List[int], numSlots: int) -> int:
        """
        Approach 3: Backtracking with Pruning
        
        Use backtracking to explore all valid placements.
        
        Time: O(exponential), Space: O(numSlots)
        """
        n = len(nums)
        slots = [0] * numSlots  # slots[i] = number of items in slot i+1
        
        def backtrack(idx: int, current_sum: int) -> int:
            if idx == n:
                return current_sum
            
            max_sum = 0
            
            for slot in range(numSlots):
                if slots[slot] < 2:  # Slot has space
                    # Place nums[idx] in slot (slot+1)
                    and_value = nums[idx] & (slot + 1)
                    slots[slot] += 1
                    
                    total_sum = backtrack(idx + 1, current_sum + and_value)
                    max_sum = max(max_sum, total_sum)
                    
                    # Backtrack
                    slots[slot] -= 1
            
            return max_sum
        
        return backtrack(0, 0)
    
    def maximumANDSum_greedy_with_optimization(self, nums: List[int], numSlots: int) -> int:
        """
        Approach 4: Greedy with Local Optimization
        
        Use greedy placement with local optimization.
        
        Time: O(n * numSlots * log(numSlots)), Space: O(numSlots)
        """
        n = len(nums)
        
        # Sort numbers in descending order for better greedy choices
        nums_with_idx = [(nums[i], i) for i in range(n)]
        nums_with_idx.sort(reverse=True)
        
        slots = [0] * numSlots  # Track slot occupancy
        placement = {}  # Track where each number is placed
        
        total_sum = 0
        
        for num, original_idx in nums_with_idx:
            best_slot = -1
            best_and_value = -1
            
            # Find best slot for this number
            for slot in range(numSlots):
                if slots[slot] < 2:
                    and_value = num & (slot + 1)
                    if and_value > best_and_value:
                        best_and_value = and_value
                        best_slot = slot
            
            # Place number in best slot
            if best_slot != -1:
                slots[best_slot] += 1
                placement[original_idx] = best_slot + 1
                total_sum += best_and_value
        
        return total_sum
    
    def maximumANDSum_branch_and_bound(self, nums: List[int], numSlots: int) -> int:
        """
        Approach 5: Branch and Bound with Upper Bound Estimation
        
        Use branch and bound with intelligent pruning.
        
        Time: O(exponential), Space: O(numSlots)
        """
        n = len(nums)
        slots = [0] * numSlots
        
        # Precompute upper bounds for each number-slot combination
        and_values = [[nums[i] & (j + 1) for j in range(numSlots)] for i in range(n)]
        
        def upper_bound(idx: int) -> int:
            """Calculate upper bound for remaining placements"""
            if idx >= n:
                return 0
            
            bound = 0
            remaining_capacity = [2 - slots[j] for j in range(numSlots)]
            
            # For each remaining number, take the best possible AND value
            for i in range(idx, n):
                best_values = []
                for slot in range(numSlots):
                    if remaining_capacity[slot] > 0:
                        best_values.append(and_values[i][slot])
                
                if best_values:
                    best_value = max(best_values)
                    bound += best_value
                    
                    # Update remaining capacity (greedy approximation)
                    best_slot = max(range(numSlots), 
                                  key=lambda s: and_values[i][s] if remaining_capacity[s] > 0 else -1)
                    if remaining_capacity[best_slot] > 0:
                        remaining_capacity[best_slot] -= 1
            
            return bound
        
        self.max_sum = 0
        
        def branch_and_bound(idx: int, current_sum: int):
            if idx == n:
                self.max_sum = max(self.max_sum, current_sum)
                return
            
            # Pruning: if upper bound can't improve current best, skip
            if current_sum + upper_bound(idx) <= self.max_sum:
                return
            
            for slot in range(numSlots):
                if slots[slot] < 2:
                    and_value = and_values[idx][slot]
                    slots[slot] += 1
                    
                    branch_and_bound(idx + 1, current_sum + and_value)
                    
                    slots[slot] -= 1
        
        branch_and_bound(0, 0)
        return self.max_sum
    
    def maximumANDSum_hungarian_inspired(self, nums: List[int], numSlots: int) -> int:
        """
        Approach 6: Hungarian Algorithm Inspired Approach
        
        Model as maximum weight bipartite matching with duplicated slots.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(nums)
        
        # Create expanded slot list (each slot appears twice)
        expanded_slots = []
        for slot in range(1, numSlots + 1):
            expanded_slots.extend([slot, slot])
        
        # Ensure we have enough slots
        while len(expanded_slots) < n:
            expanded_slots.append(numSlots)  # Add dummy slots
        
        # Create cost matrix (negate for maximum weight matching)
        cost_matrix = []
        for i in range(n):
            row = []
            for j in range(len(expanded_slots)):
                if j < len(expanded_slots):
                    and_value = nums[i] & expanded_slots[j]
                    row.append(-and_value)  # Negate for min cost assignment
                else:
                    row.append(0)  # Dummy cost
            cost_matrix.append(row)
        
        # Use simplified Hungarian algorithm
        assignment = self._simple_assignment(cost_matrix)
        
        # Calculate total AND sum
        total_sum = 0
        for i, j in assignment:
            if i < n and j < len(expanded_slots):
                total_sum += nums[i] & expanded_slots[j]
        
        return total_sum
    
    def _simple_assignment(self, cost_matrix: List[List[int]]) -> List[tuple]:
        """Simplified assignment algorithm for small instances"""
        from itertools import permutations
        
        n = len(cost_matrix)
        m = len(cost_matrix[0]) if cost_matrix else 0
        
        if n == 0 or m == 0:
            return []
        
        # For small instances, use brute force
        if n <= 8:
            min_cost = float('inf')
            best_assignment = []
            
            for perm in permutations(range(m), n):
                cost = sum(cost_matrix[i][perm[i]] for i in range(n))
                if cost < min_cost:
                    min_cost = cost
                    best_assignment = [(i, perm[i]) for i in range(n)]
            
            return best_assignment
        
        # For larger instances, use greedy approach
        assignment = []
        used_cols = set()
        
        for i in range(n):
            best_j = -1
            best_cost = float('inf')
            
            for j in range(m):
                if j not in used_cols and cost_matrix[i][j] < best_cost:
                    best_cost = cost_matrix[i][j]
                    best_j = j
            
            if best_j != -1:
                assignment.append((i, best_j))
                used_cols.add(best_j)
        
        return assignment

def test_maximum_and_sum():
    """Test maximum AND sum algorithms"""
    solver = MaximumANDSum()
    
    test_cases = [
        ([1, 2, 3, 4, 5, 6], 3, 9, "Example 1"),
        ([1, 3, 10, 4, 7, 1], 4, 24, "Example 2"),
        ([1, 2], 1, 3, "Simple case"),
        ([5, 13, 2], 2, 23, "Small case"),
    ]
    
    algorithms = [
        ("Bitmask DP", solver.maximumANDSum_bitmask_dp),
        ("Optimized Bitmask", solver.maximumANDSum_optimized_bitmask),
        ("Backtracking", solver.maximumANDSum_backtracking),
        ("Greedy Optimization", solver.maximumANDSum_greedy_with_optimization),
        ("Branch and Bound", solver.maximumANDSum_branch_and_bound),
    ]
    
    print("=== Testing Maximum AND Sum ===")
    
    for nums, numSlots, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"nums: {nums}, numSlots: {numSlots}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, numSlots)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | AND Sum: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_maximum_and_sum()

"""
Maximum AND Sum demonstrates advanced dynamic programming
with bitmask state representation, backtracking optimization,
and bipartite matching applications for assignment problems.
"""
