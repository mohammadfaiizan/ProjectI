"""
Problem: Majority Element
URL: https://leetcode.com/problems/majority-element/

Problem Statement:
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. 
You may assume that the majority element always exists in the array.

Sample Input/Output:
Input: nums = [3,2,3]
Output: 3

Input: nums = [2,2,1,1,1,2,2]
Output: 2

Input: nums = [1]
Output: 1
"""

from typing import List

class Solution:
    def Majority_Element_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force Approach - Count each element
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        majority_count = len(nums) // 2
        
        for num in nums:
            count = 0
            for elem in nums:
                if elem == num:
                    count += 1
            
            if count > majority_count:
                return num
        
        return -1
    
    def Majority_Element_Hash_Map(self, nums: List[int]) -> int:
        """
        Hash Map Approach - Count frequencies
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        count_map = {}
        majority_count = len(nums) // 2
        
        for num in nums:
            count_map[num] = count_map.get(num, 0) + 1
            if count_map[num] > majority_count:
                return num
        
        return -1
    
    def Majority_Element_Sorting(self, nums: List[int]) -> int:
        """
        Sorting Approach - Sort and return middle element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        nums.sort()
        return nums[len(nums) // 2]
    
    def Majority_Element_Boyer_Moore_Optimal(self, nums: List[int]) -> int:
        """
        Boyer-Moore Voting Algorithm - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        candidate = None
        count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
            
            count += (1 if num == candidate else -1)
        
        return candidate
    
    def Majority_Element_Divide_Conquer(self, nums: List[int]) -> int:
        """
        Divide and Conquer Approach
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        def Majority_Element_Rec(left: int, right: int) -> int:
            if left == right:
                return nums[left]
            
            mid = (left + right) // 2
            left_majority = Majority_Element_Rec(left, mid)
            right_majority = Majority_Element_Rec(mid + 1, right)
            
            if left_majority == right_majority:
                return left_majority
            
            left_count = sum(1 for i in range(left, right + 1) if nums[i] == left_majority)
            right_count = sum(1 for i in range(left, right + 1) if nums[i] == right_majority)
            
            return left_majority if left_count > right_count else right_majority
        
        return Majority_Element_Rec(0, len(nums) - 1)
    
    def Majority_Element_Counter(self, nums: List[int]) -> int:
        """
        Using Counter from collections
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        from collections import Counter
        counts = Counter(nums)
        return counts.most_common(1)[0][0]

def Test_Majority_Element():
    solution = Solution()
    
    test_cases = [
        ([3,2,3], 3),
        ([2,2,1,1,1,2,2], 2),
        ([1], 1),
        ([1,1,1,2,2], 1),
        ([5,5,5,5,1,2,3], 5)
    ]
    
    for nums, expected in test_cases:
        result1 = solution.Majority_Element_Brute_Force(nums.copy())
        result2 = solution.Majority_Element_Hash_Map(nums.copy())
        result3 = solution.Majority_Element_Sorting(nums.copy())
        result4 = solution.Majority_Element_Boyer_Moore_Optimal(nums.copy())
        result5 = solution.Majority_Element_Divide_Conquer(nums.copy())
        result6 = solution.Majority_Element_Counter(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Hash Map: {result2}")
        print(f"Sorting: {result3}")
        print(f"Boyer-Moore Optimal: {result4}")
        print(f"Divide Conquer: {result5}")
        print(f"Counter: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Majority_Element()

