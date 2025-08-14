"""
Advanced Searching Problems
==========================

Topics: 2D searches, peak finding, binary search on answer
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
LeetCode: 74, 240, 162, 658, 875, 1011
"""

from typing import List

class AdvancedSearchingProblems:
    
    def search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        """LC 74: Search 2D Matrix
        Time: O(log(m*n)), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            mid_val = matrix[mid // n][mid % n]
            
            if mid_val == target:
                return True
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    
    def search_matrix_ii(self, matrix: List[List[int]], target: int) -> bool:
        """LC 240: Search 2D Matrix II
        Time: O(m + n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        
        while row < m and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    def find_peak_element(self, nums: List[int]) -> int:
        """LC 162: Find Peak Element
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def find_k_closest_elements(self, arr: List[int], k: int, x: int) -> List[int]:
        """LC 658: Find K Closest Elements
        Time: O(log n + k), Space: O(1)
        """
        left, right = 0, len(arr) - k
        
        while left < right:
            mid = left + (right - left) // 2
            
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
        
        return arr[left:left + k]
    
    def koko_eating_bananas(self, piles: List[int], h: int) -> int:
        """LC 875: Koko Eating Bananas
        Time: O(n log(max(piles))), Space: O(1)
        """
        def can_eat_all(k):
            hours = sum((pile + k - 1) // k for pile in piles)
            return hours <= h
        
        left, right = 1, max(piles)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if can_eat_all(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def capacity_to_ship_packages(self, weights: List[int], days: int) -> int:
        """LC 1011: Capacity To Ship Packages
        Time: O(n log(sum(weights))), Space: O(1)
        """
        def can_ship_in_days(capacity):
            day_count = 1
            current_weight = 0
            
            for weight in weights:
                if current_weight + weight > capacity:
                    day_count += 1
                    current_weight = weight
                else:
                    current_weight += weight
            
            return day_count <= days
        
        left, right = max(weights), sum(weights)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if can_ship_in_days(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

# Test Examples
def run_examples():
    asp = AdvancedSearchingProblems()
    
    print("=== ADVANCED SEARCHING PROBLEMS ===\n")
    
    # 2D Matrix search
    matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]]
    print(f"Search 5 in matrix: {asp.search_matrix_ii(matrix, 5)}")
    
    # Peak finding
    nums = [1,2,3,1]
    peak = asp.find_peak_element(nums)
    print(f"Peak element: index {peak}")
    
    # K closest elements
    arr = [1,2,3,4,5]
    closest = asp.find_k_closest_elements(arr, 4, 3)
    print(f"4 closest to 3: {closest}")
    
    # Binary search on answer
    piles = [3,6,7,11]
    result = asp.koko_eating_bananas(piles, 8)
    print(f"Min eating speed: {result}")

if __name__ == "__main__":
    run_examples() 