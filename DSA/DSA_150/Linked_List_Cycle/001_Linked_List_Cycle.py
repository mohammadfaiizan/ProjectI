"""
Problem: Linked List Cycle
URL: https://leetcode.com/problems/linked-list-cycle/

Problem Statement:
Given head, the head of a linked list, determine if the linked list has a cycle in it.
There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. 
Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
Return true if there is a cycle in the linked list. Otherwise, return false.

Sample Input/Output:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
"""

from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def Has_Cycle_Hash_Set(self, head: Optional[ListNode]) -> bool:
        """
        Hash Set - Track visited nodes
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        visited = set()
        current = head
        
        while current:
            if current in visited:
                return True
            visited.add(current)
            current = current.next
        
        return False
    
    def Has_Cycle_Floyd_Optimal(self, head: Optional[ListNode]) -> bool:
        """
        Floyd's Cycle Detection Optimal - Tortoise and Hare
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def Has_Cycle_Modified_List(self, head: Optional[ListNode]) -> bool:
        """
        Modified List - Mark visited nodes by modification
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return False
        
        VISITED_MARKER = float('inf')
        current = head
        
        while current:
            if current.val == VISITED_MARKER:
                return True
            
            original_val = current.val
            current.val = VISITED_MARKER
            current = current.next
        
        return False
    
    def Has_Cycle_Reverse_List(self, head: Optional[ListNode]) -> bool:
        """
        Reverse List - Reverse and check if we reach original head
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return False
        
        prev = None
        current = head
        
        while current:
            next_node = current.next
            current.next = prev
            
            if current.next == head:
                return True
            
            prev = current
            current = next_node
        
        return False
    
    def Has_Cycle_Runner_Technique(self, head: Optional[ListNode]) -> bool:
        """
        Runner Technique - Different speed pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return False
        
        slow = head
        fast = head
        
        try:
            while True:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    return True
                
        except AttributeError:
            return False
    
    def Has_Cycle_Length_Limit(self, head: Optional[ListNode]) -> bool:
        """
        Length Limit - Assume max length and timeout
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return False
        
        MAX_NODES = 10000
        current = head
        count = 0
        
        while current and count < MAX_NODES:
            current = current.next
            count += 1
        
        return current is not None

def Create_Cyclic_List(values, pos):
    if not values:
        return None
    
    nodes = []
    for val in values:
        nodes.append(ListNode(val))
    
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    if pos >= 0 and pos < len(nodes):
        nodes[-1].next = nodes[pos]
    
    return nodes[0]

def Test_Has_Cycle():
    solution = Solution()
    
    test_cases = [
        ([3,2,0,-4], 1, True),
        ([1,2], 0, True),
        ([1], -1, False),
        ([1,2,3,4,5], 2, True),
        ([1,2,3], -1, False)
    ]
    
    methods = [
        ("Hash Set", solution.Has_Cycle_Hash_Set),
        ("Floyd Optimal", solution.Has_Cycle_Floyd_Optimal),
        ("Runner Technique", solution.Has_Cycle_Runner_Technique),
        ("Length Limit", solution.Has_Cycle_Length_Limit)
    ]
    
    for values, pos, expected in test_cases:
        print(f"Values: {values}, Pos: {pos}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            head = Create_Cyclic_List(values, pos)
            try:
                result = method(head)
                print(f"{method_name}: {result}")
            except:
                print(f"{method_name}: Error (expected for some methods)")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Has_Cycle()
