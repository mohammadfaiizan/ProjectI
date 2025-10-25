"""
Problem: Merge Two Sorted Arrays
URL: https://leetcode.com/problems/merge-sorted-array/

Problem Statement:
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, 
and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

Sample Input/Output:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
"""

from typing import List

class Solution:
    def Merge_Extra_Space(self, nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
        """
        Extra Space Approach - Create new array and merge
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        result = []
        i, j = 0, 0
        
        while i < m and j < n:
            if nums1[i] <= nums2[j]:
                result.append(nums1[i])
                i += 1
            else:
                result.append(nums2[j])
                j += 1
        
        while i < m:
            result.append(nums1[i])
            i += 1
        
        while j < n:
            result.append(nums2[j])
            j += 1
        
        return result
    
    def Merge_Insert_Sort(self, nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
        """
        Insert and Sort Approach
        Time Complexity: O((m + n) log(m + n))
        Space Complexity: O(1)
        """
        for i in range(n):
            nums1[m + i] = nums2[i]
        
        nums1.sort()
        return nums1
    
    def Merge_Three_Pointers_Backward(self, nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
        """
        Three Pointers Backward - Optimal in-place solution
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1
        
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1
        
        return nums1
    
    def Merge_Forward_Pointers(self, nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
        """
        Forward Pointers with Copy
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        """
        nums1_copy = nums1[:m]
        
        p1, p2 = 0, 0
        
        for p in range(m + n):
            if p2 >= n or (p1 < m and nums1_copy[p1] <= nums2[p2]):
                nums1[p] = nums1_copy[p1]
                p1 += 1
            else:
                nums1[p] = nums2[p2]
                p2 += 1
        
        return nums1
    
    def Merge_Pythonic(self, nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
        """
        Pythonic Approach - Using slicing
        Time Complexity: O((m + n) log(m + n))
        Space Complexity: O(1)
        """
        nums1[m:] = nums2
        nums1.sort()
        return nums1

def Test_Merge_Arrays():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,0,0,0], 3, [2,5,6], 3, [1,2,2,3,5,6]),
        ([1], 1, [], 0, [1]),
        ([0], 0, [1], 1, [1]),
        ([4,5,6,0,0,0], 3, [1,2,3], 3, [1,2,3,4,5,6]),
        ([1,3,5,0,0,0], 3, [2,4,6], 3, [1,2,3,4,5,6])
    ]
    
    for nums1, m, nums2, n, expected in test_cases:
        result1 = solution.Merge_Extra_Space(nums1.copy(), m, nums2.copy(), n)
        result2 = solution.Merge_Insert_Sort(nums1.copy(), m, nums2.copy(), n)
        result3 = solution.Merge_Three_Pointers_Backward(nums1.copy(), m, nums2.copy(), n)
        result4 = solution.Merge_Forward_Pointers(nums1.copy(), m, nums2.copy(), n)
        result5 = solution.Merge_Pythonic(nums1.copy(), m, nums2.copy(), n)
        
        print(f"Nums1: {nums1[:m]}, Nums2: {nums2}")
        print(f"Expected: {expected}")
        print(f"Extra Space: {result1}")
        print(f"Insert Sort: {result2}")
        print(f"Three Pointers Backward: {result3}")
        print(f"Forward Pointers: {result4}")
        print(f"Pythonic: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Merge_Arrays()

