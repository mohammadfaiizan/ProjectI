"""
Array Kadane's Algorithm and Subarray Problems
=============================================

Topics: Maximum subarray, Kadane's algorithm variations, subarray problems
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Medium to Hard
"""

from typing import List, Tuple
import sys

class ArrayKadaneSubarray:
    
    # ==========================================
    # 1. CLASSIC KADANE'S ALGORITHM
    # ==========================================
    
    def maximum_subarray_sum(self, nums: List[int]) -> int:
        """LC 53: Maximum Subarray (Kadane's Algorithm)
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        max_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            # Either extend existing subarray or start new one
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maximum_subarray_with_indices(self, nums: List[int]) -> Tuple[int, int, int]:
        """Maximum subarray sum with start and end indices
        Returns: (max_sum, start_index, end_index)
        """
        if not nums:
            return 0, 0, 0
        
        max_sum = current_sum = nums[0]
        start = end = temp_start = 0
        
        for i in range(1, len(nums)):
            if current_sum < 0:
                current_sum = nums[i]
                temp_start = i
            else:
                current_sum += nums[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        return max_sum, start, end
    
    def minimum_subarray_sum(self, nums: List[int]) -> int:
        """Minimum subarray sum (reverse Kadane's)
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        min_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = min(nums[i], current_sum + nums[i])
            min_sum = min(min_sum, current_sum)
        
        return min_sum
    
    # ==========================================
    # 2. KADANE'S VARIATIONS
    # ==========================================
    
    def maximum_circular_subarray(self, nums: List[int]) -> int:
        """LC 918: Maximum Sum Circular Subarray
        Time: O(n), Space: O(1)
        """
        def kadane_max(arr):
            max_sum = current_sum = arr[0]
            for i in range(1, len(arr)):
                current_sum = max(arr[i], current_sum + arr[i])
                max_sum = max(max_sum, current_sum)
            return max_sum
        
        def kadane_min(arr):
            min_sum = current_sum = arr[0]
            for i in range(1, len(arr)):
                current_sum = min(arr[i], current_sum + arr[i])
                min_sum = min(min_sum, current_sum)
            return min_sum
        
        # Case 1: Maximum subarray is non-circular
        max_kadane = kadane_max(nums)
        
        # Case 2: Maximum subarray is circular
        total_sum = sum(nums)
        min_kadane = kadane_min(nums)
        max_circular = total_sum - min_kadane
        
        # If all elements are negative, max_circular would be 0
        return max(max_kadane, max_circular) if max_circular != 0 else max_kadane
    
    def maximum_product_subarray(self, nums: List[int]) -> int:
        """LC 152: Maximum Product Subarray
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        max_prod = min_prod = result = nums[0]
        
        for i in range(1, len(nums)):
            # If current number is negative, swap max and min
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod
            
            # Update max and min products
            max_prod = max(nums[i], max_prod * nums[i])
            min_prod = min(nums[i], min_prod * nums[i])
            
            result = max(result, max_prod)
        
        return result
    
    def maximum_subarray_size_k(self, nums: List[int], k: int) -> int:
        """Maximum sum of subarray of fixed size k
        Time: O(n), Space: O(1)
        """
        if len(nums) < k:
            return 0
        
        # Calculate sum of first window
        window_sum = sum(nums[:k])
        max_sum = window_sum
        
        # Slide the window
        for i in range(k, len(nums)):
            window_sum = window_sum - nums[i - k] + nums[i]
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    def maximum_subarray_at_least_k(self, nums: List[int], k: int) -> int:
        """Maximum sum of subarray with length at least k
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        if n < k:
            return float('-inf')
        
        # Prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_sum = float('-inf')
        min_prefix = prefix[0]
        
        for i in range(k, n + 1):
            max_sum = max(max_sum, prefix[i] - min_prefix)
            min_prefix = min(min_prefix, prefix[i - k + 1])
        
        return max_sum
    
    # ==========================================
    # 3. SUBARRAY COUNT PROBLEMS
    # ==========================================
    
    def count_subarrays_sum_k(self, nums: List[int], k: int) -> int:
        """LC 560: Subarray Sum Equals K
        Time: O(n), Space: O(n)
        """
        from collections import defaultdict
        
        count = 0
        prefix_sum = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1
        
        for num in nums:
            prefix_sum += num
            count += sum_count[prefix_sum - k]
            sum_count[prefix_sum] += 1
        
        return count
    
    def count_subarrays_with_odd_sum(self, arr: List[int]) -> int:
        """LC 1524: Number of Sub-arrays With Odd Sum
        Time: O(n), Space: O(1)
        """
        MOD = 10**9 + 7
        odd_count = even_count = 0
        prefix_sum = 0
        result = 0
        
        for num in arr:
            prefix_sum += num
            
            if prefix_sum % 2 == 0:
                even_count += 1
                result = (result + odd_count) % MOD
            else:
                odd_count += 1
                result = (result + even_count + 1) % MOD
        
        return result
    
    def count_nice_subarrays(self, nums: List[int], k: int) -> int:
        """LC 1248: Count Number of Nice Subarrays (exactly k odd numbers)
        Time: O(n), Space: O(n)
        """
        from collections import defaultdict
        
        # Convert to binary array (1 for odd, 0 for even)
        prefix_sum = 0
        count = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1
        
        for num in nums:
            prefix_sum += num % 2
            count += sum_count[prefix_sum - k]
            sum_count[prefix_sum] += 1
        
        return count
    
    # ==========================================
    # 4. ADVANCED SUBARRAY PROBLEMS
    # ==========================================
    
    def shortest_subarray_sum_k(self, nums: List[int], k: int) -> int:
        """LC 862: Shortest Subarray with Sum at Least K
        Time: O(n), Space: O(n)
        """
        from collections import deque
        
        n = len(nums)
        prefix = [0] * (n + 1)
        
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        deq = deque()
        min_length = float('inf')
        
        for i in range(n + 1):
            # Check if we can form a subarray with sum >= k
            while deq and prefix[i] - prefix[deq[0]] >= k:
                min_length = min(min_length, i - deq.popleft())
            
            # Maintain increasing order of prefix sums
            while deq and prefix[i] <= prefix[deq[-1]]:
                deq.pop()
            
            deq.append(i)
        
        return min_length if min_length != float('inf') else -1
    
    def subarray_sum_divisible_by_k(self, nums: List[int], k: int) -> int:
        """LC 974: Subarray Sums Divisible by K
        Time: O(n), Space: O(k)
        """
        from collections import defaultdict
        
        remainder_count = defaultdict(int)
        remainder_count[0] = 1
        prefix_sum = 0
        count = 0
        
        for num in nums:
            prefix_sum += num
            remainder = prefix_sum % k
            count += remainder_count[remainder]
            remainder_count[remainder] += 1
        
        return count
    
    def continuous_subarray_sum_multiple_k(self, nums: List[int], k: int) -> bool:
        """LC 523: Continuous Subarray Sum (length >= 2, sum multiple of k)
        Time: O(n), Space: O(k)
        """
        if len(nums) < 2:
            return False
        
        remainder_map = {0: -1}
        prefix_sum = 0
        
        for i, num in enumerate(nums):
            prefix_sum += num
            remainder = prefix_sum % k if k != 0 else prefix_sum
            
            if remainder in remainder_map:
                if i - remainder_map[remainder] > 1:
                    return True
            else:
                remainder_map[remainder] = i
        
        return False
    
    def maximum_length_repeated_subarray(self, nums1: List[int], nums2: List[int]) -> int:
        """LC 718: Maximum Length of Repeated Subarray
        Time: O(m*n), Space: O(m*n)
        """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
        
        return max_length

# Test Examples
def run_examples():
    aks = ArrayKadaneSubarray()
    
    print("=== KADANE'S ALGORITHM AND SUBARRAY PROBLEMS ===\n")
    
    # Classic Kadane's
    print("1. CLASSIC KADANE'S ALGORITHM:")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = aks.maximum_subarray_sum(nums)
    max_sum_with_indices = aks.maximum_subarray_with_indices(nums)
    print(f"Array: {nums}")
    print(f"Maximum subarray sum: {max_sum}")
    print(f"With indices (sum, start, end): {max_sum_with_indices}")
    
    # Variations
    print("\n2. KADANE'S VARIATIONS:")
    circular_nums = [1, -2, 3, -2]
    print(f"Maximum circular subarray: {aks.maximum_circular_subarray(circular_nums)}")
    
    product_nums = [2, 3, -2, 4]
    print(f"Maximum product subarray: {aks.maximum_product_subarray(product_nums)}")
    
    # Count problems
    print("\n3. SUBARRAY COUNT PROBLEMS:")
    nums = [1, 1, 1]
    k = 2
    print(f"Count subarrays sum = {k}: {aks.count_subarrays_sum_k(nums, k)}")
    
    odd_nums = [1, 3, 5]
    print(f"Count subarrays with odd sum: {aks.count_subarrays_with_odd_sum(odd_nums)}")
    
    # Advanced problems
    print("\n4. ADVANCED PROBLEMS:")
    nums = [1, 0, 1, 0, 1]
    k = 2
    print(f"Count nice subarrays (k={k} odds): {aks.count_nice_subarrays(nums, k)}")
    
    nums = [2, -1, 2]
    k = 3
    print(f"Shortest subarray sum >= {k}: {aks.shortest_subarray_sum_k(nums, k)}")

if __name__ == "__main__":
    run_examples() 