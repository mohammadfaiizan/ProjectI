"""
Problem: Two Sum II - Input Array is Sorted
URL: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/

Problem Statement:
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, 
find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.
Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

Sample Input/Output:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].

Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore, index1 = 1, index2 = 3. We return [1, 3].

Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore, index1 = 1, index2 = 2. We return [1, 2].
"""

from typing import List

class Solution:
    def Two_Sum_Brute_Force(self, numbers: List[int], target: int) -> List[int]:
        """
        Brute Force - Check all possible pairs
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(numbers)
        
        for i in range(n):
            for j in range(i + 1, n):
                if numbers[i] + numbers[j] == target:
                    return [i + 1, j + 1]
        
        return []
    
    def Two_Sum_Two_Pointers_Optimal(self, numbers: List[int], target: int) -> List[int]:
        """
        Two Pointers Optimal - Start from both ends and adjust based on sum
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    def Two_Sum_Binary_Search(self, numbers: List[int], target: int) -> List[int]:
        """
        Binary Search - For each element, binary search for complement
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        def Binary_Search(arr: List[int], start: int, end: int, target_val: int) -> int:
            while start <= end:
                mid = (start + end) // 2
                if arr[mid] == target_val:
                    return mid
                elif arr[mid] < target_val:
                    start = mid + 1
                else:
                    end = mid - 1
            return -1
        
        for i in range(len(numbers)):
            complement = target - numbers[i]
            j = Binary_Search(numbers, i + 1, len(numbers) - 1, complement)
            if j != -1:
                return [i + 1, j + 1]
        
        return []
    
    def Two_Sum_Hash_Map(self, numbers: List[int], target: int) -> List[int]:
        """
        Hash Map - Store complements in hash map
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        num_to_index = {}
        
        for i, num in enumerate(numbers):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement] + 1, i + 1]
            num_to_index[num] = i
        
        return []
    
    def Two_Sum_Optimized_Pointers(self, numbers: List[int], target: int) -> List[int]:
        """
        Optimized Pointers - Skip unnecessary comparisons
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]
            elif current_sum < target:
                while left < right and numbers[left] == numbers[left]:
                    left += 1
            else:
                while left < right and numbers[right] == numbers[right]:
                    right -= 1
        
        return []
    
    def Two_Sum_Early_Termination(self, numbers: List[int], target: int) -> List[int]:
        """
        Early Termination - Terminate early when impossible
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(numbers) - 1
        
        while left < right:
            if numbers[left] + numbers[left + 1] > target:
                break
            if numbers[right - 1] + numbers[right] < target:
                break
            
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []

def Test_Two_Sum():
    solution = Solution()
    
    test_cases = [
        ([2,7,11,15], 9, [1,2]),
        ([2,3,4], 6, [1,3]),
        ([-1,0], -1, [1,2]),
        ([1,2,3,4,4,9,56,90], 8, [4,5]),
        ([5,25,75], 100, [2,3])
    ]
    
    methods = [
        ("Brute Force", solution.Two_Sum_Brute_Force),
        ("Two Pointers Optimal", solution.Two_Sum_Two_Pointers_Optimal),
        ("Binary Search", solution.Two_Sum_Binary_Search),
        ("Hash Map", solution.Two_Sum_Hash_Map),
        ("Optimized Pointers", solution.Two_Sum_Optimized_Pointers),
        ("Early Termination", solution.Two_Sum_Early_Termination)
    ]
    
    for numbers, target, expected in test_cases:
        print(f"Numbers: {numbers}, Target: {target}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            result = method(numbers.copy(), target)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Two_Sum()
