"""
Problem: Nearest Greater to Left
URL: No direct link found

Problem Statement:
Given an array, find the nearest greater element to the left for each element.
If no greater element exists, return -1 for that element.

Sample Input/Output:
Input: arr = [1, 3, 2, 4]
Output: [-1, -1, 3, -1]
Explanation: For 1->-1, for 3->-1, for 2->3, for 4->-1

Input: arr = [4, 5, 2, 25]
Output: [-1, -1, 5, -1]
Explanation: For 4->-1, for 5->-1, for 2->5, for 25->-1
"""

from typing import List

class Solution:
    def Previous_Greater_Element_Brute_Force(self, arr: List[int]) -> List[int]:
        """
        Brute Force Approach - Check all elements to the left
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = []
        
        for i in range(n):
            prev_greater = -1
            for j in range(i - 1, -1, -1):
                if arr[j] > arr[i]:
                    prev_greater = arr[j]
                    break
            result.append(prev_greater)
        
        return result
    
    def Previous_Greater_Element_Nested_Loop(self, arr: List[int]) -> List[int]:
        """
        Nested Loop Approach - Linear search for each element
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        result = [-1] * len(arr)
        
        for i in range(len(arr)):
            for j in range(i - 1, -1, -1):
                if arr[j] > arr[i]:
                    result[i] = arr[j]
                    break
        
        return result
    
    def Previous_Greater_Element_Stack_Optimal(self, arr: List[int]) -> List[int]:
        """
        Stack Approach - Optimal solution using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and stack[-1] <= arr[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            
            stack.append(arr[i])
        
        return result
    
    def Previous_Greater_Element_Stack_Indices(self, arr: List[int]) -> List[int]:
        """
        Stack with Indices Approach - Store indices in stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and arr[stack[-1]] <= arr[i]:
                stack.pop()
            
            if stack:
                result[i] = arr[stack[-1]]
            
            stack.append(i)
        
        return result
    
    def Previous_Greater_Element_Monotonic_Stack(self, arr: List[int]) -> List[int]:
        """
        Monotonic Stack Approach - Maintain decreasing stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = [-1] * len(arr)
        stack = []
        
        for i in range(len(arr)):
            while stack and stack[-1] <= arr[i]:
                stack.pop()
            
            result[i] = stack[-1] if stack else -1
            stack.append(arr[i])
        
        return result
    
    def Previous_Greater_Element_Stack_Reverse(self, arr: List[int]) -> List[int]:
        """
        Stack Reverse Processing - Process array in reverse order
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and stack[-1] < arr[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            else:
                result[i] = -1
            
            stack.append(arr[i])
        
        return result

def Test_Previous_Greater_Element():
    solution = Solution()
    
    test_cases = [
        ([1, 3, 2, 4], [-1, -1, 3, -1]),
        ([4, 5, 2, 25], [-1, -1, 5, -1]),
        ([13, 7, 6, 12], [-1, 13, 7, 13]),
        ([5, 4, 3, 2, 1], [-1, 5, 4, 3, 2])
    ]
    
    for arr, expected in test_cases:
        result1 = solution.Previous_Greater_Element_Brute_Force(arr.copy())
        result2 = solution.Previous_Greater_Element_Nested_Loop(arr.copy())
        result3 = solution.Previous_Greater_Element_Stack_Optimal(arr.copy())
        result4 = solution.Previous_Greater_Element_Stack_Indices(arr.copy())
        result5 = solution.Previous_Greater_Element_Monotonic_Stack(arr.copy())
        result6 = solution.Previous_Greater_Element_Stack_Reverse(arr.copy())
        
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Stack Indices: {result4}")
        print(f"Monotonic Stack: {result5}")
        print(f"Stack Reverse: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Previous_Greater_Element()
