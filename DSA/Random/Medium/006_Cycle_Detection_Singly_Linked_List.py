"""
Problem: Cycle Detection in a Singly Linked List
URL: https://leetcode.com/problems/linked-list-cycle/

Problem Statement:
Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached 
again by continuously following the next pointer. Internally, pos is used to denote the 
index of the node that tail's next pointer is connected to. Note that pos is not passed 
as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Sample Input/Output:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Input: head = [1,2], pos = 0
Output: true

Input: head = [1], pos = -1
Output: false
"""

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Has_Cycle_Hash_Set(self, head: Optional[ListNode]) -> bool:
        """
        Hash Set Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        seen = set()
        current = head
        
        while current:
            if current in seen:
                return True
            seen.add(current)
            current = current.next
        
        return False
    
    def Has_Cycle_Floyd_Optimal(self, head: Optional[ListNode]) -> bool:
        """
        Floyd's Cycle Detection (Tortoise and Hare) - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next
        
        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True
    
    def Has_Cycle_Two_Pointer(self, head: Optional[ListNode]) -> bool:
        """
        Two Pointer Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def Has_Cycle_Modification(self, head: Optional[ListNode]) -> bool:
        """
        Node Modification Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = head
        
        while current:
            if hasattr(current, 'visited'):
                return True
            current.visited = True
            current = current.next
        
        return False
    
    def Has_Cycle_Counter(self, head: Optional[ListNode]) -> bool:
        """
        Counter Approach - With limit
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = head
        count = 0
        max_nodes = 10000
        
        while current and count < max_nodes:
            current = current.next
            count += 1
        
        return current is not None

def Create_Cycle_List(values: List[int], pos: int) -> Optional[ListNode]:
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    cycle_node = None
    
    if pos == 0:
        cycle_node = head
    
    for i, val in enumerate(values[1:], 1):
        current.next = ListNode(val)
        current = current.next
        if i == pos:
            cycle_node = current
    
    if pos != -1 and cycle_node:
        current.next = cycle_node
    
    return head

def Test_Has_Cycle():
    solution = Solution()
    
    test_cases = [
        ([3,2,0,-4], 1, True),
        ([1,2], 0, True),
        ([1], -1, False),
        ([1,2,3,4,5], 2, True),
        ([1,2,3], -1, False)
    ]
    
    for values, pos, expected in test_cases:
        head1 = Create_Cycle_List(values, pos)
        head2 = Create_Cycle_List(values, pos)
        head3 = Create_Cycle_List(values, pos)
        head4 = Create_Cycle_List(values, pos)
        head5 = Create_Cycle_List(values, pos)
        
        result1 = solution.Has_Cycle_Hash_Set(head1)
        result2 = solution.Has_Cycle_Floyd_Optimal(head2)
        result3 = solution.Has_Cycle_Two_Pointer(head3)
        result4 = solution.Has_Cycle_Modification(head4)
        result5 = solution.Has_Cycle_Counter(head5)
        
        print(f"Values: {values}, Cycle at pos: {pos}")
        print(f"Expected: {expected}")
        print(f"Hash Set: {result1}")
        print(f"Floyd Optimal: {result2}")
        print(f"Two Pointer: {result3}")
        print(f"Modification: {result4}")
        print(f"Counter: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Has_Cycle()

