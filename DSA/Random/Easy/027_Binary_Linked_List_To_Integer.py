"""
Problem: Binary Linked List To Integer
URL: https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/

Problem Statement:
Given head which is a reference node to a singly-linked list. The value of each node in 
the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

Sample Input/Output:
Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10

Input: head = [0]
Output: 0

Input: head = [1]
Output: 1
"""

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Get_Decimal_Value_String(self, head: Optional[ListNode]) -> int:
        """
        String Conversion Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        binary_str = ""
        
        current = head
        while current:
            binary_str += str(current.val)
            current = current.next
        
        return int(binary_str, 2)
    
    def Get_Decimal_Value_Math(self, head: Optional[ListNode]) -> int:
        """
        Mathematical Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        values = []
        
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        result = 0
        for i, val in enumerate(reversed(values)):
            result += val * (2 ** i)
        
        return result
    
    def Get_Decimal_Value_Optimal(self, head: Optional[ListNode]) -> int:
        """
        Optimal Approach - Shift and add
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = 0
        
        current = head
        while current:
            result = result * 2 + current.val
            current = current.next
        
        return result
    
    def Get_Decimal_Value_Bit_Shift(self, head: Optional[ListNode]) -> int:
        """
        Bit Shift Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = 0
        
        current = head
        while current:
            result = (result << 1) | current.val
            current = current.next
        
        return result
    
    def Get_Decimal_Value_Recursive(self, head: Optional[ListNode]) -> int:
        """
        Recursive Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Count_Length(node):
            if not node:
                return 0
            return 1 + Count_Length(node.next)
        
        def Convert(node, length):
            if not node:
                return 0
            return node.val * (2 ** (length - 1)) + Convert(node.next, length - 1)
        
        length = Count_Length(head)
        return Convert(head, length)

def Create_List(values: List[int]) -> Optional[ListNode]:
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Test_Get_Decimal_Value():
    solution = Solution()
    
    test_cases = [
        ([1,0,1], 5),
        ([0], 0),
        ([1], 1),
        ([1,0,0,1,0,0,1,1,1,0,0,0,0,0,0], 18880),
        ([1,1,1], 7)
    ]
    
    for values, expected in test_cases:
        head = Create_List(values)
        
        result1 = solution.Get_Decimal_Value_String(Create_List(values))
        result2 = solution.Get_Decimal_Value_Math(Create_List(values))
        result3 = solution.Get_Decimal_Value_Optimal(Create_List(values))
        result4 = solution.Get_Decimal_Value_Bit_Shift(Create_List(values))
        result5 = solution.Get_Decimal_Value_Recursive(Create_List(values))
        
        print(f"Binary: {values}")
        print(f"Expected: {expected}")
        print(f"String: {result1}")
        print(f"Math: {result2}")
        print(f"Optimal: {result3}")
        print(f"Bit Shift: {result4}")
        print(f"Recursive: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Get_Decimal_Value()

