"""
Problem: Remove Duplicates from Sorted Array
URL: https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/

Problem Statement:
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. 
The relative order of the elements should be kept the same. Then return the number of unique elements in nums.
Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:
- Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially.
- The remaining elements of nums are not important as well as the size of nums.
- Return k.

Sample Input/Output:
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
"""

from typing import List

class Solution:
    def Remove_Duplicates_Extra_Space(self, nums: List[int]) -> int:
        """
        Extra Space - Use set to track unique elements
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        unique_elements = []
        seen = set()
        
        for num in nums:
            if num not in seen:
                unique_elements.append(num)
                seen.add(num)
        
        for i in range(len(unique_elements)):
            nums[i] = unique_elements[i]
        
        return len(unique_elements)
    
    def Remove_Duplicates_Two_Pointers_Optimal(self, nums: List[int]) -> int:
        """
        Two Pointers Optimal - One pointer tracks unique position
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        write_index = 1
        
        for read_index in range(1, len(nums)):
            if nums[read_index] != nums[read_index - 1]:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index
    
    def Remove_Duplicates_Slow_Fast_Pointer(self, nums: List[int]) -> int:
        """
        Slow Fast Pointer - Classic two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        slow = 0
        
        for fast in range(1, len(nums)):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
        
        return slow + 1
    
    def Remove_Duplicates_Comparison_Based(self, nums: List[int]) -> int:
        """
        Comparison Based - Compare with previous unique element
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        unique_count = 1
        last_unique = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] != last_unique:
                nums[unique_count] = nums[i]
                last_unique = nums[i]
                unique_count += 1
        
        return unique_count
    
    def Remove_Duplicates_Single_Pass(self, nums: List[int]) -> int:
        """
        Single Pass - Process array in single iteration
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        insert_pos = 0
        
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i - 1]:
                nums[insert_pos] = nums[i]
                insert_pos += 1
        
        return insert_pos
    
    def Remove_Duplicates_While_Loop(self, nums: List[int]) -> int:
        """
        While Loop - Use while loop with index management
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        i = 0
        j = 1
        
        while j < len(nums):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
            j += 1
        
        return i + 1

def Test_Remove_Duplicates():
    solution = Solution()
    
    test_cases = [
        ([1,1,2], 2, [1,2]),
        ([0,0,1,1,1,2,2,3,3,4], 5, [0,1,2,3,4]),
        ([1], 1, [1]),
        ([1,1,1], 1, [1]),
        ([1,2,3,4,5], 5, [1,2,3,4,5])
    ]
    
    methods = [
        ("Extra Space", solution.Remove_Duplicates_Extra_Space),
        ("Two Pointers Optimal", solution.Remove_Duplicates_Two_Pointers_Optimal),
        ("Slow Fast Pointer", solution.Remove_Duplicates_Slow_Fast_Pointer),
        ("Comparison Based", solution.Remove_Duplicates_Comparison_Based),
        ("Single Pass", solution.Remove_Duplicates_Single_Pass),
        ("While Loop", solution.Remove_Duplicates_While_Loop)
    ]
    
    for nums, expected_length, expected_array in test_cases:
        print(f"Original: {nums}")
        print(f"Expected Length: {expected_length}")
        print(f"Expected Array: {expected_array}")
        
        for method_name, method in methods:
            test_nums = nums.copy()
            result_length = method(test_nums)
            result_array = test_nums[:result_length]
            print(f"{method_name}: Length={result_length}, Array={result_array}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Remove_Duplicates()
