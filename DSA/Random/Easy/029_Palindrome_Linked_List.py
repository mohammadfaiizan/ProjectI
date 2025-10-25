"""
Problem: Palindrome Linked List
URL: https://leetcode.com/problems/palindrome-linked-list/

Problem Statement:
Given the head of a singly linked list, return true if it is a palindrome or false otherwise.

Sample Input/Output:
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false

Input: head = [1]
Output: true
"""

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Is_Palindrome_Array(self, head: Optional[ListNode]) -> bool:
        """
        Array Approach - Convert to array and check
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        values = []
        
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        return values == values[::-1]
    
    def Is_Palindrome_Reverse_List(self, head: Optional[ListNode]) -> bool:
        """
        Reverse List Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return True
        
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node
        
        left, right = head, prev
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True
    
    def Is_Palindrome_Stack(self, head: Optional[ListNode]) -> bool:
        """
        Stack Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        
        current = head
        while current:
            stack.append(current.val)
            current = current.next
        
        current = head
        while current:
            if current.val != stack.pop():
                return False
            current = current.next
        
        return True
    
    def Is_Palindrome_Recursive(self, head: Optional[ListNode]) -> bool:
        """
        Recursive Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        self.front = head
        
        def Recursively_Check(current):
            if not current:
                return True
            
            if not Recursively_Check(current.next):
                return False
            
            if self.front.val != current.val:
                return False
            
            self.front = self.front.next
            return True
        
        return Recursively_Check(head)
    
    def Is_Palindrome_Two_Pointer(self, head: Optional[ListNode]) -> bool:
        """
        Two Pointer with List
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        vals = []
        current = head
        
        while current:
            vals.append(current.val)
            current = current.next
        
        left, right = 0, len(vals) - 1
        
        while left < right:
            if vals[left] != vals[right]:
                return False
            left += 1
            right -= 1
        
        return True

def Create_List(values: List[int]) -> Optional[ListNode]:
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Test_Is_Palindrome():
    solution = Solution()
    
    test_cases = [
        ([1,2,2,1], True),
        ([1,2], False),
        ([1], True),
        ([1,2,3,2,1], True),
        ([1,2,3,4,5], False)
    ]
    
    for values, expected in test_cases:
        result1 = solution.Is_Palindrome_Array(Create_List(values))
        result2 = solution.Is_Palindrome_Reverse_List(Create_List(values))
        result3 = solution.Is_Palindrome_Stack(Create_List(values))
        result4 = solution.Is_Palindrome_Recursive(Create_List(values))
        result5 = solution.Is_Palindrome_Two_Pointer(Create_List(values))
        
        print(f"List: {values}")
        print(f"Expected: {expected}")
        print(f"Array: {result1}")
        print(f"Reverse List: {result2}")
        print(f"Stack: {result3}")
        print(f"Recursive: {result4}")
        print(f"Two Pointer: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Is_Palindrome()

