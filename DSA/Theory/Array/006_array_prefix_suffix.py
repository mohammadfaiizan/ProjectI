"""
Array Prefix and Suffix Operations - Complete Implementation
===========================================================

Topics: Prefix sums, suffix operations, range queries, difference arrays
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Easy to Hard
"""

from typing import List

class ArrayPrefixSuffix:
    
    # ==========================================
    # 1. PREFIX SUM OPERATIONS
    # ==========================================
    
    def build_prefix_sum(self, arr: List[int]) -> List[int]:
        """Build prefix sum array
        Time: O(n), Space: O(n)
        """
        if not arr:
            return []
        
        prefix = [0] * (len(arr) + 1)
        for i in range(len(arr)):
            prefix[i + 1] = prefix[i] + arr[i]
        
        return prefix
    
    def range_sum_query(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        """LC 303: Range Sum Query - Immutable
        Time: O(n) build, O(1) query, Space: O(n)
        """
        prefix = self.build_prefix_sum(arr)
        results = []
        
        for left, right in queries:
            # Convert to 0-indexed if needed
            range_sum = prefix[right + 1] - prefix[left]
            results.append(range_sum)
        
        return results
    
    def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
        """LC 560: Subarray Sum Equals K
        Time: O(n), Space: O(n)
        """
        from collections import defaultdict
        
        count = 0
        prefix_sum = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1  # Empty prefix
        
        for num in nums:
            prefix_sum += num
            # If (prefix_sum - k) exists, we found subarrays
            count += sum_count[prefix_sum - k]
            sum_count[prefix_sum] += 1
        
        return count
    
    def continuous_subarray_sum(self, nums: List[int], k: int) -> bool:
        """LC 523: Continuous Subarray Sum (multiple of k)
        Time: O(n), Space: O(k)
        """
        if len(nums) < 2:
            return False
        
        remainder_map = {0: -1}  # remainder -> index
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
    
    def product_except_self(self, nums: List[int]) -> List[int]:
        """LC 238: Product of Array Except Self
        Time: O(n), Space: O(1) excluding output array
        """
        n = len(nums)
        result = [1] * n
        
        # Left pass: result[i] contains product of all elements to the left
        for i in range(1, n):
            result[i] = result[i - 1] * nums[i - 1]
        
        # Right pass: multiply with product of all elements to the right
        right_product = 1
        for i in range(n - 1, -1, -1):
            result[i] *= right_product
            right_product *= nums[i]
        
        return result
    
    # ==========================================
    # 2. 2D PREFIX SUM
    # ==========================================
    
    def build_2d_prefix_sum(self, matrix: List[List[int]]) -> List[List[int]]:
        """Build 2D prefix sum matrix
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix[i][j] = (matrix[i-1][j-1] + 
                               prefix[i-1][j] + 
                               prefix[i][j-1] - 
                               prefix[i-1][j-1])
        
        return prefix
    
    def range_sum_2d(self, matrix: List[List[int]], 
                     row1: int, col1: int, row2: int, col2: int) -> int:
        """LC 304: Range Sum Query 2D
        Time: O(1) query after O(m*n) preprocessing
        """
        prefix = self.build_2d_prefix_sum(matrix)
        
        return (prefix[row2 + 1][col2 + 1] - 
                prefix[row1][col2 + 1] - 
                prefix[row2 + 1][col1] + 
                prefix[row1][col1])
    
    def count_square_submatrices(self, matrix: List[List[int]]) -> int:
        """LC 1277: Count Square Submatrices with All Ones
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        result = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
                    result += dp[i][j]
        
        return result
    
    # ==========================================
    # 3. DIFFERENCE ARRAY
    # ==========================================
    
    def range_addition(self, length: int, updates: List[List[int]]) -> List[int]:
        """LC 370: Range Addition
        Time: O(n + k), Space: O(n)
        """
        diff = [0] * length
        
        # Apply updates to difference array
        for start, end, inc in updates:
            diff[start] += inc
            if end + 1 < length:
                diff[end + 1] -= inc
        
        # Convert difference array to actual array
        for i in range(1, length):
            diff[i] += diff[i - 1]
        
        return diff
    
    def car_pooling(self, trips: List[List[int]], capacity: int) -> bool:
        """LC 1094: Car Pooling
        Time: O(n), Space: O(1001)
        """
        diff = [0] * 1001
        
        # Apply passenger changes
        for passengers, start, end in trips:
            diff[start] += passengers
            diff[end] -= passengers
        
        # Check if capacity is exceeded
        current_passengers = 0
        for change in diff:
            current_passengers += change
            if current_passengers > capacity:
                return False
        
        return True
    
    def corporate_flight_bookings(self, bookings: List[List[int]], n: int) -> List[int]:
        """LC 1109: Corporate Flight Bookings
        Time: O(m + n), Space: O(n)
        """
        diff = [0] * (n + 1)
        
        for first, last, seats in bookings:
            diff[first - 1] += seats  # Convert to 0-indexed
            diff[last] -= seats
        
        # Convert to actual array
        for i in range(1, n):
            diff[i] += diff[i - 1]
        
        return diff[:n]
    
    # ==========================================
    # 4. ADVANCED PREFIX/SUFFIX PROBLEMS
    # ==========================================
    
    def maximum_subarray(self, nums: List[int]) -> int:
        """LC 53: Maximum Subarray (Kadane's Algorithm)
        Time: O(n), Space: O(1)
        """
        max_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maximum_product_subarray(self, nums: List[int]) -> int:
        """LC 152: Maximum Product Subarray
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        
        max_prod = min_prod = result = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod
            
            max_prod = max(nums[i], max_prod * nums[i])
            min_prod = min(nums[i], min_prod * nums[i])
            result = max(result, max_prod)
        
        return result
    
    def best_time_to_buy_sell_stock(self, prices: List[int]) -> int:
        """LC 121: Best Time to Buy and Sell Stock
        Time: O(n), Space: O(1)
        """
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        
        return max_profit
    
    def trapping_rain_water_prefix(self, height: List[int]) -> int:
        """LC 42: Trapping Rain Water using prefix/suffix
        Time: O(n), Space: O(n)
        """
        if not height:
            return 0
        
        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        
        # Fill left_max array
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], height[i])
        
        # Fill right_max array
        right_max[n - 1] = height[n - 1]
        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], height[i])
        
        # Calculate trapped water
        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]
        
        return water
    
    def largest_rectangle_histogram(self, heights: List[int]) -> int:
        """LC 84: Largest Rectangle in Histogram
        Time: O(n), Space: O(n)
        """
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area

# Test Examples
def run_examples():
    aps = ArrayPrefixSuffix()
    
    print("=== ARRAY PREFIX AND SUFFIX EXAMPLES ===\n")
    
    # Prefix sum
    print("1. PREFIX SUM OPERATIONS:")
    arr = [1, 2, 3, 4, 5]
    prefix = aps.build_prefix_sum(arr)
    print(f"Array: {arr}")
    print(f"Prefix sum: {prefix}")
    
    queries = [[0, 2], [1, 4], [2, 3]]
    results = aps.range_sum_query(arr, queries)
    print(f"Range sum queries {queries}: {results}")
    
    # Subarray sum equals k
    print("\n2. SUBARRAY SUM PROBLEMS:")
    nums = [1, 1, 1]
    k = 2
    print(f"Subarray sum equals {k} in {nums}: {aps.subarray_sum_equals_k(nums, k)}")
    
    # Product except self
    print("\n3. PRODUCT PROBLEMS:")
    nums = [1, 2, 3, 4]
    print(f"Product except self {nums}: {aps.product_except_self(nums)}")
    
    # 2D prefix sum
    print("\n4. 2D PREFIX SUM:")
    matrix = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5]]
    result = aps.range_sum_2d(matrix, 1, 1, 2, 3)
    print(f"2D range sum: {result}")
    
    # Difference array
    print("\n5. DIFFERENCE ARRAY:")
    updates = [[1, 3, 2], [2, 4, 3], [0, 2, -2]]
    result = aps.range_addition(5, updates)
    print(f"Range addition result: {result}")
    
    # Advanced problems
    print("\n6. ADVANCED PROBLEMS:")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Maximum subarray sum: {aps.maximum_subarray(nums)}")
    
    prices = [7, 1, 5, 3, 6, 4]
    print(f"Best time to buy/sell stock: {aps.best_time_to_buy_sell_stock(prices)}")

if __name__ == "__main__":
    run_examples() 