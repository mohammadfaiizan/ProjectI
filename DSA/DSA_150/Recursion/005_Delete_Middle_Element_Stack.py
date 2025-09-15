"""
Problem: Delete middle Element from a Stack
URL: https://www.geeksforgeeks.org/problems/delete-middle-element-of-a-stack/1

Problem Statement:
Given a stack with push(), pop(), and empty() operations, delete the middle element of it without using any additional data structure.

Sample Input/Output:
Input: stack = [1, 2, 3, 4, 5], size = 5
Output: [1, 2, 4, 5] (middle element 3 deleted)
Explanation: Middle element at index 2 (0-indexed) is deleted

Input: stack = [1, 2, 3, 4], size = 4
Output: [1, 2, 4] (middle element 3 deleted)
"""

from typing import List

class Solution:
    def Delete_Middle_Recursive_Optimal(self, stack: List[int], size: int) -> List[int]:
        """
        Recursive Delete Middle - Optimal solution using recursion
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        def Delete_Middle_Helper(st: List[int], current: int, middle: int) -> None:
            if current == middle:
                st.pop()
                return
            
            temp = st.pop()
            Delete_Middle_Helper(st, current + 1, middle)
            st.append(temp)
        
        if not stack:
            return stack
        
        middle_index = size // 2
        Delete_Middle_Helper(stack, 0, middle_index)
        return stack
    
    def Delete_Middle_Using_Auxiliary_Stack(self, stack: List[int], size: int) -> List[int]:
        """
        Using Auxiliary Stack - Store elements temporarily
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not stack:
            return stack
        
        temp_stack = []
        middle_index = size // 2
        
        for i in range(middle_index):
            temp_stack.append(stack.pop())
        
        if stack:
            stack.pop()
        
        while temp_stack:
            stack.append(temp_stack.pop())
        
        return stack

def Test_Delete_Middle():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 3, 4, 5], 5),
        ([1, 2, 3, 4], 4),
        ([1, 2, 3], 3),
        ([1, 2], 2),
        ([1], 1)
    ]
    
    for stack, size in test_cases:
        result1 = solution.Delete_Middle_Recursive_Optimal(stack.copy(), size)
        result2 = solution.Delete_Middle_Using_Auxiliary_Stack(stack.copy(), size)
        
        print(f"Original Stack: {stack}, Size: {size}")
        print(f"Recursive Delete: {result1}")
        print(f"Auxiliary Stack: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Delete_Middle()
