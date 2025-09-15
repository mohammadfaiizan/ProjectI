"""
Problem: Nearest Smaller to Left
URL: https://www.interviewbit.com/problems/nearest-smaller-element/

Problem Statement:
Given an array, find the nearest smaller element to the left for each element.
If no smaller element exists, return -1 for that element.

Sample Input/Output:
Input: arr = [4, 5, 2, 25]
Output: [-1, 4, -1, 2]
Explanation: For 4->-1, for 5->4, for 2->-1, for 25->2

Input: arr = [13, 7, 6, 12]
Output: [-1, -1, -1, 6]
Explanation: For 13->-1, for 7->-1, for 6->-1, for 12->6
"""

from typing import List

class Solution:
    def Previous_Smaller_Element_Brute_Force(self, arr: List[int]) -> List[int]:
        """
        Brute Force Approach - Check all elements to the left
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = []
        
        for i in range(n):
            prev_smaller = -1
            for j in range(i - 1, -1, -1):
                if arr[j] < arr[i]:
                    prev_smaller = arr[j]
                    break
            result.append(prev_smaller)
        
        return result
    
    def Previous_Smaller_Element_Nested_Loop(self, arr: List[int]) -> List[int]:
        """
        Nested Loop Approach - Linear search for each element
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        result = [-1] * len(arr)
        
        for i in range(len(arr)):
            for j in range(i - 1, -1, -1):
                if arr[j] < arr[i]:
                    result[i] = arr[j]
                    break
        
        return result
    
    def Previous_Smaller_Element_Stack_Optimal(self, arr: List[int]) -> List[int]:
        """
        Stack Approach - Optimal solution using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and stack[-1] >= arr[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            
            stack.append(arr[i])
        
        return result
    
    def Previous_Smaller_Element_Stack_Indices(self, arr: List[int]) -> List[int]:
        """
        Stack with Indices Approach - Store indices in stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and arr[stack[-1]] >= arr[i]:
                stack.pop()
            
            if stack:
                result[i] = arr[stack[-1]]
            
            stack.append(i)
        
        return result
    
    def Previous_Smaller_Element_Monotonic_Stack(self, arr: List[int]) -> List[int]:
        """
        Monotonic Stack Approach - Maintain increasing stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = [-1] * len(arr)
        stack = []
        
        for i in range(len(arr)):
            while stack and stack[-1] >= arr[i]:
                stack.pop()
            
            result[i] = stack[-1] if stack else -1
            stack.append(arr[i])
        
        return result
    
    def Previous_Smaller_Element_Stack_Backward(self, arr: List[int]) -> List[int]:
        """
        Stack Backward Processing - Alternative approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and stack[-1] >= arr[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            else:
                result[i] = -1
            
            stack.append(arr[i])
        
        return result

def Test_Previous_Smaller_Element():
    solution = Solution()
    
    test_cases = [
        ([4, 5, 2, 25], [-1, 4, -1, 2]),
        ([13, 7, 6, 12], [-1, -1, -1, 6]),
        ([1, 3, 2, 4], [-1, 1, 1, 2]),
        ([5, 4, 3, 2, 1], [-1, -1, -1, -1, -1])
    ]
    
    for arr, expected in test_cases:
        result1 = solution.Previous_Smaller_Element_Brute_Force(arr.copy())
        result2 = solution.Previous_Smaller_Element_Nested_Loop(arr.copy())
        result3 = solution.Previous_Smaller_Element_Stack_Optimal(arr.copy())
        result4 = solution.Previous_Smaller_Element_Stack_Indices(arr.copy())
        result5 = solution.Previous_Smaller_Element_Monotonic_Stack(arr.copy())
        result6 = solution.Previous_Smaller_Element_Stack_Backward(arr.copy())
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Stack Indices: {result4}")
        print(f"Monotonic Stack: {result5}")
        print(f"Stack Backward: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Previous_Smaller_Element()
