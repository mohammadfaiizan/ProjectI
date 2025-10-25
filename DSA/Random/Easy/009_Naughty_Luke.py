"""
Problem: Naughty Luke
URL: https://www.naukri.com/code360/problems/naughty-luke

Problem Statement:
Luke is playing with numbers. He has an array of integers and wants to find out how many 
pairs (i, j) exist such that i < j and arr[i] > 2 * arr[j].

This is known as the "Reverse Pairs" problem.

Sample Input/Output:
Input: arr = [1,3,2,3,1]
Output: 2
Explanation: (3,1) appears twice (indices 1,4 and 3,4)

Input: arr = [2,4,3,5,1]
Output: 3
Explanation: (2,1), (4,1), (3,1)

Input: arr = [1,2,3,4,5]
Output: 0
"""

from typing import List

class Solution:
    def Naughty_Luke_Brute_Force(self, arr: List[int]) -> int:
        """
        Brute Force Approach - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        n = len(arr)
        
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > 2 * arr[j]:
                    count += 1
        
        return count
    
    def Naughty_Luke_Merge_Sort(self, arr: List[int]) -> int:
        """
        Merge Sort Approach - Optimal solution
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        def Merge_Sort_Count(nums: List[int], left: int, right: int) -> int:
            if left >= right:
                return 0
            
            mid = (left + right) // 2
            count = Merge_Sort_Count(nums, left, mid) + Merge_Sort_Count(nums, mid + 1, right)
            
            j = mid + 1
            for i in range(left, mid + 1):
                while j <= right and nums[i] > 2 * nums[j]:
                    j += 1
                count += (j - (mid + 1))
            
            nums[left:right + 1] = sorted(nums[left:right + 1])
            
            return count
        
        return Merge_Sort_Count(arr[:], 0, len(arr) - 1)
    
    def Naughty_Luke_Modified_Merge(self, arr: List[int]) -> int:
        """
        Modified Merge Sort with Manual Merge
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        def Merge(left: List[int], right: List[int]) -> tuple:
            count = 0
            j = 0
            
            for num in left:
                while j < len(right) and num > 2 * right[j]:
                    j += 1
                count += j
            
            merged = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
            
            merged.extend(left[i:])
            merged.extend(right[j:])
            
            return count, merged
        
        def Merge_Sort(nums: List[int]) -> tuple:
            if len(nums) <= 1:
                return 0, nums
            
            mid = len(nums) // 2
            left_count, left = Merge_Sort(nums[:mid])
            right_count, right = Merge_Sort(nums[mid:])
            
            merge_count, merged = Merge(left, right)
            
            return left_count + right_count + merge_count, merged
        
        count, _ = Merge_Sort(arr)
        return count
    
    def Naughty_Luke_Binary_Indexed_Tree(self, arr: List[int]) -> int:
        """
        Binary Indexed Tree Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        sorted_arr = sorted(set(arr + [2 * x for x in arr]))
        rank = {v: i + 1 for i, v in enumerate(sorted_arr)}
        
        class BIT:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (n + 1)
            
            def Update(self, i):
                while i <= self.n:
                    self.tree[i] += 1
                    i += i & (-i)
            
            def Query(self, i):
                s = 0
                while i > 0:
                    s += self.tree[i]
                    i -= i & (-i)
                return s
        
        bit = BIT(len(sorted_arr))
        count = 0
        
        for num in reversed(arr):
            count += bit.Query(rank[num] - 1)
            bit.Update(rank[2 * num])
        
        return count
    
    def Naughty_Luke_Optimized_Brute(self, arr: List[int]) -> int:
        """
        Optimized Brute Force with Early Termination
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        n = len(arr)
        
        for i in range(n - 1):
            threshold = arr[i] / 2.0
            for j in range(i + 1, n):
                if arr[j] < threshold:
                    count += 1
        
        return count

def Test_Naughty_Luke():
    solution = Solution()
    
    test_cases = [
        ([1,3,2,3,1], 2),
        ([2,4,3,5,1], 3),
        ([1,2,3,4,5], 0),
        ([5,4,3,2,1], 4),
        ([1], 0),
        ([2,1], 1)
    ]
    
    for arr, expected in test_cases:
        result1 = solution.Naughty_Luke_Brute_Force(arr.copy())
        result2 = solution.Naughty_Luke_Merge_Sort(arr.copy())
        result3 = solution.Naughty_Luke_Modified_Merge(arr.copy())
        result4 = solution.Naughty_Luke_Binary_Indexed_Tree(arr.copy())
        result5 = solution.Naughty_Luke_Optimized_Brute(arr.copy())
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Merge Sort: {result2}")
        print(f"Modified Merge: {result3}")
        print(f"Binary Indexed Tree: {result4}")
        print(f"Optimized Brute: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Naughty_Luke()

