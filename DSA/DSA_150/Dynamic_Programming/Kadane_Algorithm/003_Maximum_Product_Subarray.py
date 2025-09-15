"""
Problem: Maximum Product Subarray
URL: https://leetcode.com/problems/maximum-product-subarray/

Problem Statement:
Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.
The test cases are generated so that the answer will fit in a 32-bit integer.
A subarray is a contiguous subsequence of the array.

Sample Input/Output:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

Input: nums = [-2,3,-4]
Output: 24
Explanation: [-2,3,-4] has the largest product 24.
"""

from typing import List, Tuple

class Solution:
    def Max_Product_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Check all possible subarrays
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_product = float('-inf')
        
        for i in range(n):
            for j in range(i, n):
                current_product = 1
                for k in range(i, j + 1):
                    current_product *= nums[k]
                max_product = max(max_product, current_product)
        
        return max_product
    
    def Max_Product_Optimized_Brute_Force(self, nums: List[int]) -> int:
        """
        Optimized Brute Force - Avoid recalculating products
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_product = float('-inf')
        
        for i in range(n):
            current_product = 1
            for j in range(i, n):
                current_product *= nums[j]
                max_product = max(max_product, current_product)
        
        return max_product
    
    def Max_Product_Kadane_Modified_Optimal(self, nums: List[int]) -> int:
        """
        Kadane Modified Optimal - Track both max and min products
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_ending_here = min_ending_here = result = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_ending_here, min_ending_here = min_ending_here, max_ending_here
            
            max_ending_here = max(nums[i], max_ending_here * nums[i])
            min_ending_here = min(nums[i], min_ending_here * nums[i])
            
            result = max(result, max_ending_here)
        
        return result
    
    def Max_Product_DP_Array(self, nums: List[int]) -> int:
        """
        DP Array - Using DP arrays for max and min products
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(nums)
        
        max_dp = [0] * n
        min_dp = [0] * n
        
        max_dp[0] = min_dp[0] = nums[0]
        result = nums[0]
        
        for i in range(1, n):
            if nums[i] >= 0:
                max_dp[i] = max(nums[i], max_dp[i-1] * nums[i])
                min_dp[i] = min(nums[i], min_dp[i-1] * nums[i])
            else:
                max_dp[i] = max(nums[i], min_dp[i-1] * nums[i])
                min_dp[i] = min(nums[i], max_dp[i-1] * nums[i])
            
            result = max(result, max_dp[i])
        
        return result
    
    def Max_Product_Two_Pass(self, nums: List[int]) -> int:
        """
        Two Pass - Forward and backward passes
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_product = float('-inf')
        
        product = 1
        for i in range(n):
            product *= nums[i]
            max_product = max(max_product, product)
            if product == 0:
                product = 1
        
        product = 1
        for i in range(n - 1, -1, -1):
            product *= nums[i]
            max_product = max(max_product, product)
            if product == 0:
                product = 1
        
        return max_product
    
    def Max_Product_With_Indices(self, nums: List[int]) -> Tuple[int, int, int]:
        """
        With Indices - Return max product and start/end indices
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_ending_here = min_ending_here = result = nums[0]
        start = end = 0
        temp_start = 0
        
        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_ending_here, min_ending_here = min_ending_here, max_ending_here
            
            if max_ending_here * nums[i] < nums[i]:
                max_ending_here = nums[i]
                temp_start = i
            else:
                max_ending_here *= nums[i]
            
            min_ending_here = min(nums[i], min_ending_here * nums[i])
            
            if max_ending_here > result:
                result = max_ending_here
                start = temp_start
                end = i
        
        return result, start, end
    
    def Max_Product_Handle_Zeros(self, nums: List[int]) -> int:
        """
        Handle Zeros - Special handling for zeros in array
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        max_product = float('-inf')
        current_max = current_min = 1
        
        for num in nums:
            if num == 0:
                max_product = max(max_product, 0)
                current_max = current_min = 1
            elif num > 0:
                current_max *= num
                current_min *= num
                max_product = max(max_product, current_max)
                
                if current_min > current_max:
                    current_min = current_max
            else:
                temp = current_max
                current_max = max(num, current_min * num)
                current_min = min(num, temp * num)
                max_product = max(max_product, current_max)
        
        return max_product
    
    def Max_Product_Divide_By_Zeros(self, nums: List[int]) -> int:
        """
        Divide By Zeros - Split array by zeros and solve each part
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def Max_Product_No_Zeros(arr: List[int]) -> int:
            if not arr:
                return float('-inf')
            
            max_ending_here = min_ending_here = result = arr[0]
            
            for i in range(1, len(arr)):
                if arr[i] < 0:
                    max_ending_here, min_ending_here = min_ending_here, max_ending_here
                
                max_ending_here = max(arr[i], max_ending_here * arr[i])
                min_ending_here = min(arr[i], min_ending_here * arr[i])
                
                result = max(result, max_ending_here)
            
            return result
        
        max_product = 0 if 0 in nums else float('-inf')
        
        current_subarray = []
        
        for num in nums:
            if num == 0:
                if current_subarray:
                    max_product = max(max_product, Max_Product_No_Zeros(current_subarray))
                    current_subarray = []
            else:
                current_subarray.append(num)
        
        if current_subarray:
            max_product = max(max_product, Max_Product_No_Zeros(current_subarray))
        
        return max_product
    
    def Max_Product_All_Subarrays(self, nums: List[int]) -> Tuple[int, List[List[int]]]:
        """
        All Subarrays - Find all subarrays with maximum product
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        max_product = self.Max_Product_Kadane_Modified_Optimal(nums)
        max_subarrays = []
        
        n = len(nums)
        
        for i in range(n):
            current_product = 1
            for j in range(i, n):
                current_product *= nums[j]
                if current_product == max_product:
                    max_subarrays.append(nums[i:j+1])
        
        return max_product, max_subarrays
    
    def Max_Product_Negative_Count(self, nums: List[int]) -> int:
        """
        Negative Count - Handle based on negative number count
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        if 0 in nums:
            return max(0, self.Max_Product_Divide_By_Zeros(nums))
        
        negative_count = sum(1 for x in nums if x < 0)
        
        if negative_count % 2 == 0:
            product = 1
            for num in nums:
                product *= num
            return product
        else:
            left_product = 1
            for i in range(n):
                left_product *= nums[i]
                if nums[i] < 0:
                    break
            
            right_product = 1
            for i in range(n - 1, -1, -1):
                right_product *= nums[i]
                if nums[i] < 0:
                    break
            
            total_product = 1
            for num in nums:
                total_product *= num
            
            return max(total_product // left_product, total_product // right_product)

def Test_Max_Product():
    solution = Solution()
    
    test_cases = [
        ([2,3,-2,4], 6),
        ([-2,0,-1], 0),
        ([-2,3,-4], 24),
        ([2,-5,-2,-4,3], 24),
        ([0,2], 2),
        ([-1,-2,-3], 6),
        ([1,2,3,4], 24),
        ([-4,-3,-2], 12)
    ]
    
    methods = [
        ("Optimized Brute Force", solution.Max_Product_Optimized_Brute_Force),
        ("Kadane Modified Optimal", solution.Max_Product_Kadane_Modified_Optimal),
        ("DP Array", solution.Max_Product_DP_Array),
        ("Two Pass", solution.Max_Product_Two_Pass),
        ("Handle Zeros", solution.Max_Product_Handle_Zeros),
        ("Divide By Zeros", solution.Max_Product_Divide_By_Zeros),
        ("Negative Count", solution.Max_Product_Negative_Count)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 8:
            result_bf = solution.Max_Product_Brute_Force(nums.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_product, start, end = solution.Max_Product_With_Indices(nums.copy())
        print(f"With Indices: Product={max_product}, Start={start}, End={end}, Subarray={nums[start:end+1]}")
        
        if len(nums) <= 6:
            max_product, all_subarrays = solution.Max_Product_All_Subarrays(nums.copy())
            print(f"All Max Subarrays: Product={max_product}, Count={len(all_subarrays)}")
            for subarray in all_subarrays:
                print(f"  {subarray}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Product()
