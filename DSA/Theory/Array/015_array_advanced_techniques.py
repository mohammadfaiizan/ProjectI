"""
Array Advanced Techniques and Optimizations
===========================================

Topics: Advanced array algorithms, optimization techniques, complex problems
Companies: Google, Facebook, Amazon, Microsoft, Apple, Netflix
Difficulty: Hard
"""

from typing import List, Tuple, Dict
import random
from collections import defaultdict

class ArrayAdvancedTechniques:
    
    # ==========================================
    # 1. DIVIDE AND CONQUER TECHNIQUES
    # ==========================================
    
    def count_inversions(self, arr: List[int]) -> int:
        """Count inversions using merge sort approach
        Time: O(n log n), Space: O(n)
        """
        def merge_and_count(arr, temp, left, mid, right):
            i, j, k = left, mid + 1, left
            inv_count = 0
            
            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    inv_count += (mid - i + 1)
                    j += 1
                k += 1
            
            while i <= mid:
                temp[k] = arr[i]
                i += 1
                k += 1
            
            while j <= right:
                temp[k] = arr[j]
                j += 1
                k += 1
            
            for i in range(left, right + 1):
                arr[i] = temp[i]
            
            return inv_count
        
        def merge_sort_and_count(arr, temp, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                inv_count += merge_sort_and_count(arr, temp, left, mid)
                inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
                inv_count += merge_and_count(arr, temp, left, mid, right)
            return inv_count
        
        temp = [0] * len(arr)
        return merge_sort_and_count(arr[:], temp, 0, len(arr) - 1)
    
    def closest_pair_sum(self, arr: List[int], target: int) -> Tuple[int, int]:
        """Find pair with sum closest to target
        Time: O(n log n), Space: O(1)
        """
        arr_with_idx = [(val, i) for i, val in enumerate(arr)]
        arr_with_idx.sort()
        
        left, right = 0, len(arr) - 1
        closest_sum = float('inf')
        result_pair = (0, 0)
        
        while left < right:
            current_sum = arr_with_idx[left][0] + arr_with_idx[right][0]
            
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum
                result_pair = (arr_with_idx[left][1], arr_with_idx[right][1])
            
            if current_sum < target:
                left += 1
            else:
                right -= 1
        
        return result_pair
    
    # ==========================================
    # 2. RANDOMIZED ALGORITHMS
    # ==========================================
    
    def quick_select(self, nums: List[int], k: int) -> int:
        """Find kth smallest element using QuickSelect
        Time: O(n) average, O(n²) worst, Space: O(1)
        """
        def partition(arr, low, high):
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1
        
        def quick_select_helper(arr, low, high, k):
            if low <= high:
                pi = partition(arr, low, high)
                
                if pi == k:
                    return arr[pi]
                elif pi > k:
                    return quick_select_helper(arr, low, pi - 1, k)
                else:
                    return quick_select_helper(arr, pi + 1, high, k)
        
        return quick_select_helper(nums[:], 0, len(nums) - 1, k - 1)
    
    def reservoir_sampling(self, stream: List[int], k: int) -> List[int]:
        """Reservoir sampling for random k elements from stream
        Time: O(n), Space: O(k)
        """
        reservoir = []
        
        for i, num in enumerate(stream):
            if len(reservoir) < k:
                reservoir.append(num)
            else:
                # Generate random index from 0 to i
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = num
        
        return reservoir
    
    # ==========================================
    # 3. GEOMETRIC ALGORITHMS
    # ==========================================
    
    def convex_hull_graham_scan(self, points: List[List[int]]) -> List[List[int]]:
        """Graham scan algorithm for convex hull
        Time: O(n log n), Space: O(n)
        """
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points = sorted(set(map(tuple, points)))
        
        if len(points) <= 1:
            return points
        
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return lower[:-1] + upper[:-1]
    
    def max_points_on_line(self, points: List[List[int]]) -> int:
        """LC 149: Max Points on a Line
        Time: O(n²), Space: O(n)
        """
        if len(points) <= 2:
            return len(points)
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        max_points = 0
        
        for i in range(len(points)):
            slopes = defaultdict(int)
            duplicate = 1
            local_max = 0
            
            for j in range(i + 1, len(points)):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                
                if dx == 0 and dy == 0:
                    duplicate += 1
                else:
                    g = gcd(dx, dy)
                    dx, dy = dx // g, dy // g
                    
                    if dx < 0:
                        dx, dy = -dx, -dy
                    elif dx == 0:
                        dy = abs(dy)
                    
                    slopes[(dx, dy)] += 1
                    local_max = max(local_max, slopes[(dx, dy)])
            
            max_points = max(max_points, local_max + duplicate)
        
        return max_points
    
    # ==========================================
    # 4. STRING MATCHING IN ARRAYS
    # ==========================================
    
    def kmp_search_in_array(self, text: List[int], pattern: List[int]) -> List[int]:
        """KMP pattern matching in integer arrays
        Time: O(n + m), Space: O(m)
        """
        def compute_lps(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            
            return lps
        
        if not pattern:
            return []
        
        lps = compute_lps(pattern)
        matches = []
        i = j = 0
        
        while i < len(text):
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == len(pattern):
                matches.append(i - j)
                j = lps[j - 1]
            elif i < len(text) and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    # ==========================================
    # 5. ADVANCED OPTIMIZATION PROBLEMS
    # ==========================================
    
    def max_profit_k_transactions(self, k: int, prices: List[int]) -> int:
        """LC 188: Best Time to Buy and Sell Stock IV
        Time: O(n*k), Space: O(k)
        """
        if not prices or k == 0:
            return 0
        
        n = len(prices)
        
        # If k is large enough, it's like unlimited transactions
        if k >= n // 2:
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i-1]:
                    profit += prices[i] - prices[i-1]
            return profit
        
        # DP with space optimization
        buy = [-prices[0]] * k
        sell = [0] * k
        
        for i in range(1, n):
            for j in range(k-1, -1, -1):
                sell[j] = max(sell[j], buy[j] + prices[i])
                if j == 0:
                    buy[j] = max(buy[j], -prices[i])
                else:
                    buy[j] = max(buy[j], sell[j-1] - prices[i])
        
        return sell[k-1]
    
    def shortest_unsorted_subarray(self, nums: List[int]) -> int:
        """LC 581: Shortest Unsorted Continuous Subarray
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        left, right = -1, -2
        min_val, max_val = nums[n-1], nums[0]
        
        for i in range(1, n):
            max_val = max(max_val, nums[i])
            min_val = min(min_val, nums[n-1-i])
            
            if nums[i] < max_val:
                right = i
            if nums[n-1-i] > min_val:
                left = n-1-i
        
        return right - left + 1
    
    def largest_rectangle_area_optimized(self, heights: List[int]) -> int:
        """Optimized largest rectangle using divide and conquer
        Time: O(n log n) average, Space: O(log n)
        """
        def divide_conquer(left, right):
            if left > right:
                return 0
            
            min_idx = left
            for i in range(left, right + 1):
                if heights[i] < heights[min_idx]:
                    min_idx = i
            
            return max(
                heights[min_idx] * (right - left + 1),
                divide_conquer(left, min_idx - 1),
                divide_conquer(min_idx + 1, right)
            )
        
        return divide_conquer(0, len(heights) - 1)

# Test Examples
def run_examples():
    aat = ArrayAdvancedTechniques()
    
    print("=== ARRAY ADVANCED TECHNIQUES EXAMPLES ===\n")
    
    # Divide and conquer
    print("1. DIVIDE AND CONQUER:")
    arr = [2, 4, 1, 3, 5]
    inversions = aat.count_inversions(arr)
    print(f"Inversions in {arr}: {inversions}")
    
    arr = [1, 3, -1, 2, 5]
    target = 4
    closest_pair = aat.closest_pair_sum(arr, target)
    print(f"Closest pair sum to {target}: indices {closest_pair}")
    
    # Randomized algorithms
    print("\n2. RANDOMIZED ALGORITHMS:")
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    kth_smallest = aat.quick_select(nums, k)
    print(f"{k}th smallest in {nums}: {kth_smallest}")
    
    # Geometric algorithms
    print("\n3. GEOMETRIC ALGORITHMS:")
    points = [[1, 1], [2, 2], [3, 3], [0, 0], [1, 3]]
    max_on_line = aat.max_points_on_line(points)
    print(f"Max points on line: {max_on_line}")
    
    # Advanced optimization
    print("\n4. ADVANCED OPTIMIZATION:")
    prices = [3, 2, 6, 5, 0, 3]
    k = 2
    max_profit = aat.max_profit_k_transactions(k, prices)
    print(f"Max profit with {k} transactions: {max_profit}")
    
    nums = [2, 6, 4, 8, 10, 9, 15]
    unsorted_len = aat.shortest_unsorted_subarray(nums)
    print(f"Shortest unsorted subarray length: {unsorted_len}")

if __name__ == "__main__":
    run_examples() 