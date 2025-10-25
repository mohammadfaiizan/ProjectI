"""
Problem: Remove All Nodes with Value K
URL: https://leetcode.com/problems/remove-linked-list-elements/

Problem Statement:
Given the head of a linked list and an integer val, remove all the nodes of the linked 
list that has Node.val == val, and return the new head.

Sample Input/Output:
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]

Input: head = [], val = 1
Output: []

Input: head = [7,7,7,7], val = 7
Output: []
"""

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Remove_Elements_Iterative(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        Iterative Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        
        return dummy.next
    
    def Remove_Elements_Recursive(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        Recursive Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head:
            return None
        
        head.next = self.Remove_Elements_Recursive(head.next, val)
        
        return head.next if head.val == val else head
    
    def Remove_Elements_Two_Pointer(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        Two Pointer Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        while head and head.val == val:
            head = head.next
        
        if not head:
            return None
        
        prev, current = head, head.next
        
        while current:
            if current.val == val:
                prev.next = current.next
            else:
                prev = current
            current = current.next
        
        return head
    
    def Remove_Elements_Sentinel(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        Sentinel Node Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        sentinel = ListNode(0, head)
        pred, curr = sentinel, head
        
        while curr:
            if curr.val == val:
                pred.next = curr.next
            else:
                pred = curr
            curr = curr.next
        
        return sentinel.next
    
    def Remove_Elements_While_Loop(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """
        While Loop Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        while head and head.val == val:
            head = head.next
        
        current = head
        
        while current and current.next:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        
        return head

def Create_List(values: List[int]) -> Optional[ListNode]:
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Print_List(head: Optional[ListNode]) -> List[int]:
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result

def Test_Remove_Elements():
    solution = Solution()
    
    test_cases = [
        ([1,2,6,3,4,5,6], 6, [1,2,3,4,5]),
        ([], 1, []),
        ([7,7,7,7], 7, []),
        ([1,2,3,4,5], 6, [1,2,3,4,5]),
        ([1,1], 1, [])
    ]
    
    for values, val, expected in test_cases:
        result1 = Print_List(solution.Remove_Elements_Iterative(Create_List(values), val))
        result2 = Print_List(solution.Remove_Elements_Recursive(Create_List(values), val))
        result3 = Print_List(solution.Remove_Elements_Two_Pointer(Create_List(values), val))
        result4 = Print_List(solution.Remove_Elements_Sentinel(Create_List(values), val))
        result5 = Print_List(solution.Remove_Elements_While_Loop(Create_List(values), val))
        
        print(f"Input: {values}, Remove: {val}")
        print(f"Expected: {expected}")
        print(f"Iterative: {result1}")
        print(f"Recursive: {result2}")
        print(f"Two Pointer: {result3}")
        print(f"Sentinel: {result4}")
        print(f"While Loop: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Remove_Elements()

