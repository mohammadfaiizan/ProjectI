"""
Problem: Nearest Greater to Right
URL: https://leetcode.com/problems/next-greater-element-i/description/

Problem Statement:
Given an array, find the nearest greater element to the right for each element.
If no greater element exists, return -1 for that element.

Sample Input/Output:
Input: arr = [1, 3, 2, 4]
Output: [3, 4, 4, -1]
Explanation: For 1->3, for 3->4, for 2->4, for 4->-1

Input: arr = [4, 5, 2, 25]
Output: [5, 25, 25, -1]
Explanation: For 4->5, for 5->25, for 2->25, for 25->-1
"""

from typing import List

class Solution:
    def Next_Greater_Element_Brute_Force(self, arr: List[int]) -> List[int]:
        """
        Brute Force Approach - Check all elements to the right
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = []
        
        for i in range(n):
            next_greater = -1
            for j in range(i + 1, n):
                if arr[j] > arr[i]:
                    next_greater = arr[j]
                    break
            result.append(next_greater)
        
        return result
    
    def Next_Greater_Element_Nested_Loop(self, arr: List[int]) -> List[int]:
        """
        Nested Loop Approach - Linear search for each element
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        result = [-1] * len(arr)
        
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[j] > arr[i]:
                    result[i] = arr[j]
                    break
        
        return result
    
    def Next_Greater_Element_Stack_Optimal(self, arr: List[int]) -> List[int]:
        """
        Stack Approach - Optimal solution using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n - 1, -1, -1):
            while stack and stack[-1] <= arr[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            
            stack.append(arr[i])
        
        return result
    
    def Next_Greater_Element_Stack_Forward(self, arr: List[int]) -> List[int]:
        """
        Stack Forward Traversal - Process from left to right
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and arr[stack[-1]] < arr[i]:
                index = stack.pop()
                result[index] = arr[i]
            stack.append(i)
        
        return result
    
    def Next_Greater_Element_Monotonic_Stack(self, arr: List[int]) -> List[int]:
        """
        Monotonic Stack Approach - Maintain decreasing stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = [-1] * len(arr)
        stack = []
        
        for i in range(len(arr) - 1, -1, -1):
            while stack and stack[-1] <= arr[i]:
                stack.pop()
            
            result[i] = stack[-1] if stack else -1
            stack.append(arr[i])
        
        return result

def Test_Next_Greater_Element():
    solution = Solution()
    
    test_cases = [
        ([1, 3, 2, 4], [3, 4, 4, -1]),
        ([4, 5, 2, 25], [5, 25, 25, -1]),
        ([13, 7, 6, 12], [-1, 12, 12, -1]),
        ([1, 2, 3, 4, 5], [2, 3, 4, 5, -1])
    ]
    
    for arr, expected in test_cases:
        result1 = solution.Next_Greater_Element_Brute_Force(arr.copy())
        result2 = solution.Next_Greater_Element_Nested_Loop(arr.copy())
        result3 = solution.Next_Greater_Element_Stack_Optimal(arr.copy())
        result4 = solution.Next_Greater_Element_Stack_Forward(arr.copy())
        result5 = solution.Next_Greater_Element_Monotonic_Stack(arr.copy())
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Stack Forward: {result4}")
        print(f"Monotonic Stack: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Next_Greater_Element()
