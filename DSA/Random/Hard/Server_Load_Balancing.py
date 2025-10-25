"""
Problem: Server Load Balancing
URL: https://www.naukri.com/code360/contests/weekly-contest-201/20769517/problems/42563

Problem Statement:
A cloud provider has n servers. The i-th server currently handles nums[i] tasks. 
You may perform at most maxOperations repartition operations.

In one repartition operation you take a single server and split its tasks into two or more 
new servers (each new server must have a positive integer number of tasks).

After all operations you will have more servers (each split increases the total server count).

Let the final servers' loads be sorted descending; your penalty is the value of the K-th 
largest server load (i.e., the K-th bottleneck).

You want to minimize this penalty.

Return the minimum possible penalty (the smallest possible value of the K-th largest server load) 
after performing at most maxOperations repartitions.

Sample Input/Output:
Input: nums = [9], maxOperations = 2, K = 1
Output: 3
Explanation: Split 9 → 6 + 3 (1 operation), Split 6 → 3 + 3 (1 operation)
             Final loads: [3,3,3]. K=1 (largest) = 3. Used 2 operations ≤ maxOperations.

Input: nums = [1, 2, 1], maxOperations = 2, K = 1
Output: 1
Explanation: Already have servers with load 1, 2, 1. K=1 (largest) = 2. No operations needed.
"""

from typing import List
import math

class Solution:
    def Server_Load_Brute_Force(self, nums: List[int], maxOperations: int, K: int) -> int:
        """
        Brute Force Approach - Try all possible penalties
        Time Complexity: O(max(nums) * n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Can_Achieve_Penalty(penalty: int) -> bool:
            exceeding = []
            for load in nums:
                if load > penalty:
                    exceeding.append(load)
            
            if len(exceeding) <= K - 1:
                return True
            
            exceeding.sort(reverse=True)
            operations = 0
            for i in range(K - 1, len(exceeding)):
                operations += (exceeding[i] - 1) // penalty
                if operations > maxOperations:
                    return False
            return True
        
        max_load = max(nums)
        
        for penalty in range(1, max_load + 1):
            if Can_Achieve_Penalty(penalty):
                return penalty
        
        return max_load
    
    def Server_Load_Binary_Search(self, nums: List[int], maxOperations: int, K: int) -> int:
        """
        Binary Search Approach - Search for minimum penalty
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Can_Achieve_Penalty(penalty: int) -> bool:
            exceeding = []
            for load in nums:
                if load > penalty:
                    exceeding.append(load)
            
            if len(exceeding) <= K - 1:
                return True
            
            exceeding.sort(reverse=True)
            operations = 0
            for i in range(K - 1, len(exceeding)):
                operations += (exceeding[i] - 1) // penalty
                if operations > maxOperations:
                    return False
            return True
        
        left, right = 1, max(nums)
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if Can_Achieve_Penalty(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Server_Load_Binary_Search_Optimal(self, nums: List[int], maxOperations: int, K: int) -> int:
        """
        Binary Search with Optimized Validation - Optimal solution
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Operations_Needed(penalty: int) -> int:
            exceeding = []
            for load in nums:
                if load > penalty:
                    exceeding.append(load)
            
            if len(exceeding) <= K - 1:
                return 0
            
            exceeding.sort(reverse=True)
            operations = 0
            for i in range(K - 1, len(exceeding)):
                operations += math.ceil(exceeding[i] / penalty) - 1
            return operations
        
        left, right = 1, max(nums)
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if Operations_Needed(mid) <= maxOperations:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Server_Load_Binary_Search_Math(self, nums: List[int], maxOperations: int, K: int) -> int:
        """
        Binary Search with Math Formula
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Check_Feasible(penalty: int) -> bool:
            exceeding = []
            for load in nums:
                if load > penalty:
                    exceeding.append(load)
            
            if len(exceeding) <= K - 1:
                return True
            
            exceeding.sort(reverse=True)
            total_operations = 0
            for i in range(K - 1, len(exceeding)):
                load = exceeding[i]
                parts_needed = (load + penalty - 1) // penalty
                total_operations += parts_needed - 1
                if total_operations > maxOperations:
                    return False
            return True
        
        if not nums:
            return 0
        
        low, high = 1, max(nums)
        answer = high
        
        while low <= high:
            mid = low + (high - low) // 2
            
            if Check_Feasible(mid):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
    
    def Server_Load_Binary_Search_Greedy(self, nums: List[int], maxOperations: int, K: int) -> int:
        """
        Binary Search with Greedy Validation
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Is_Valid_Penalty(target: int) -> bool:
            exceeding = []
            for num in nums:
                if num > target:
                    exceeding.append(num)
            
            if len(exceeding) <= K - 1:
                return True
            
            exceeding.sort(reverse=True)
            ops_used = 0
            for i in range(K - 1, len(exceeding)):
                splits = math.ceil(exceeding[i] / target)
                ops_used += splits - 1
                if ops_used > maxOperations:
                    return False
            return True
        
        left, right = 1, max(nums)
        
        while left < right:
            mid = (left + right) // 2
            
            if Is_Valid_Penalty(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Server_Load_Binary_Search_Compact(self, nums: List[int], maxOperations: int, K: int) -> int:
        """
        Compact Binary Search Implementation
        Time Complexity: O(n * log(max(nums)))
        Space Complexity: O(1)
        """
        def Valid(penalty: int) -> bool:
            exceeding = sorted([num for num in nums if num > penalty], reverse=True)
            if len(exceeding) <= K - 1:
                return True
            return sum((exceeding[i] - 1) // penalty for i in range(K - 1, len(exceeding))) <= maxOperations
        
        lo, hi = 1, max(nums)
        
        while lo < hi:
            mid = (lo + hi) // 2
            if Valid(mid):
                hi = mid
            else:
                lo = mid + 1
        
        return lo

def Test_Server_Load():
    solution = Solution()
    
    test_cases = [
        ([9], 2, 1, 3),
        ([1, 2, 1], 2, 1, 1),
        ([10], 1, 1, 5),
        ([2, 4, 8, 2], 4, 1, 2),
        ([1], 0, 1, 1),
        ([7, 17], 2, 2, 7),
        ([10, 10, 10], 3, 2, 5),
        ([5, 19, 8, 1], 5, 2, 2)
    ]
    
    for nums, maxOps, K, expected in test_cases:
        result1 = solution.Server_Load_Brute_Force(nums.copy(), maxOps, K)
        result2 = solution.Server_Load_Binary_Search(nums.copy(), maxOps, K)
        result3 = solution.Server_Load_Binary_Search_Optimal(nums.copy(), maxOps, K)
        result4 = solution.Server_Load_Binary_Search_Math(nums.copy(), maxOps, K)
        result5 = solution.Server_Load_Binary_Search_Greedy(nums.copy(), maxOps, K)
        result6 = solution.Server_Load_Binary_Search_Compact(nums.copy(), maxOps, K)
        
        print(f"Servers: {nums}, MaxOps: {maxOps}, K: {K}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Binary Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Binary Search Math: {result4}")
        print(f"Binary Search Greedy: {result5}")
        print(f"Binary Search Compact: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Server_Load()

