"""
Problem: Reverse Linked List
URL: https://leetcode.com/problems/reverse-linked-list/

Problem Statement:
Given the head of a singly linked list, reverse the list, and return the reversed list.

Sample Input/Output:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Input: head = [1,2]
Output: [2,1]

Input: head = []
Output: []
"""

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Reverse_List_Iterative_Optimal(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Iterative Optimal - Three pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    def Reverse_List_Recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Recursive Approach - Reverse using recursion
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return head
        
        reversed_head = self.Reverse_List_Recursive(head.next)
        head.next.next = head
        head.next = None
        
        return reversed_head
    
    def Reverse_List_Stack_Based(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Stack Based - Use stack to reverse
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head:
            return None
        
        stack = []
        current = head
        
        while current:
            stack.append(current)
            current = current.next
        
        new_head = stack.pop()
        current = new_head
        
        while stack:
            current.next = stack.pop()
            current = current.next
        
        current.next = None
        return new_head
    
    def Reverse_List_Two_Pointer(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pointer - Alternative two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        first = head
        second = head.next
        first.next = None
        
        while second:
            temp = second.next
            second.next = first
            first = second
            second = temp
        
        return first
    
    def Reverse_List_Helper_Function(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Helper Function - Use helper for recursion
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Reverse_Helper(prev: Optional[ListNode], current: Optional[ListNode]) -> Optional[ListNode]:
            if not current:
                return prev
            
            next_node = current.next
            current.next = prev
            return Reverse_Helper(current, next_node)
        
        return Reverse_Helper(None, head)

def Create_Linked_List(values):
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Linked_List_To_Array(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

def Test_Reverse_List():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5], [5,4,3,2,1]),
        ([1,2], [2,1]),
        ([], []),
        ([1], [1]),
        ([1,2,3], [3,2,1])
    ]
    
    methods = [
        ("Iterative Optimal", solution.Reverse_List_Iterative_Optimal),
        ("Recursive", solution.Reverse_List_Recursive),
        ("Stack Based", solution.Reverse_List_Stack_Based),
        ("Two Pointer", solution.Reverse_List_Two_Pointer),
        ("Helper Function", solution.Reverse_List_Helper_Function)
    ]
    
    for values, expected in test_cases:
        print(f"Input: {values}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            head = Create_Linked_List(values)
            result_head = method(head)
            result = Linked_List_To_Array(result_head)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Reverse_List()
