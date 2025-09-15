"""
Problem: Sort a Stack
URL: https://www.geeksforgeeks.org/problems/sort-a-stack/

Problem Statement:
Given a stack, sort it using recursion. Use of any loop constructs like while, for..etc is not allowed.
We can only use the following ADT functions on Stack S: isEmpty(S), push(S, x), pop(S), top(S)

Sample Input/Output:
Input: stack = [3, 2, 1, 30, 5]
Output: [30, 5, 3, 2, 1] (top to bottom)
Explanation: Stack sorted in descending order from top

Input: stack = [1, 2, 3]
Output: [3, 2, 1] (top to bottom)
"""

from typing import List

class Solution:
    def Sort_Stack_Recursive_Optimal(self, stack: List[int]) -> List[int]:
        """
        Recursive Sort - Optimal solution using recursion only
        Time Complexity: O(n²)
        Space Complexity: O(n) - recursion stack
        """
        def Sort_Stack_Helper(st: List[int]) -> None:
            if not st:
                return
            
            temp = st.pop()
            Sort_Stack_Helper(st)
            Sorted_Insert(st, temp)
        
        def Sorted_Insert(st: List[int], element: int) -> None:
            if not st or st[-1] <= element:
                st.append(element)
                return
            
            temp = st.pop()
            Sorted_Insert(st, element)
            st.append(temp)
        
        Sort_Stack_Helper(stack)
        return stack
    
    def Sort_Stack_Using_Another_Stack(self, stack: List[int]) -> List[int]:
        """
        Using Another Stack - Iterative approach with auxiliary stack
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        temp_stack = []
        
        while stack:
            temp = stack.pop()
            
            while temp_stack and temp_stack[-1] > temp:
                stack.append(temp_stack.pop())
            
            temp_stack.append(temp)
        
        return temp_stack

def Test_Sort_Stack():
    solution = Solution()
    
    test_cases = [
        [3, 2, 1, 30, 5],
        [1, 2, 3],
        [5, 4, 3, 2, 1],
        [1],
        [10, 5, 15, 45]
    ]
    
    for stack in test_cases:
        result1 = solution.Sort_Stack_Recursive_Optimal(stack.copy())
        result2 = solution.Sort_Stack_Using_Another_Stack(stack.copy())
        
        print(f"Original Stack: {stack}")
        print(f"Recursive Sort: {result1}")
        print(f"Using Another Stack: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Sort_Stack()
