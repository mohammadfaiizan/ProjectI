"""
Problem: Remove N-th Node From End of List
URL: https://leetcode.com/problems/remove-nth-node-from-end-of-list/

Problem Statement:
Given the head of a linked list, remove the nth node from the end of the list and return its head.

Sample Input/Output:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Input: head = [1], n = 1
Output: []

Input: head = [1,2], n = 1
Output: [1]
"""

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Remove_Nth_From_End_Two_Pass(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Two Pass - First pass to count, second to remove
        Time Complexity: O(L)
        Space Complexity: O(1)
        """
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        if n == length:
            return head.next
        
        current = head
        for _ in range(length - n - 1):
            current = current.next
        
        current.next = current.next.next
        return head
    
    def Remove_Nth_From_End_One_Pass_Optimal(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        One Pass Optimal - Two pointers with n gap
        Time Complexity: O(L)
        Space Complexity: O(1)
        """
        dummy = ListNode(0)
        dummy.next = head
        first = dummy
        second = dummy
        
        for _ in range(n + 1):
            first = first.next
        
        while first:
            first = first.next
            second = second.next
        
        second.next = second.next.next
        return dummy.next
    
    def Remove_Nth_From_End_Stack(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Stack Based - Use stack to track nodes
        Time Complexity: O(L)
        Space Complexity: O(L)
        """
        stack = []
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        
        while current:
            stack.append(current)
            current = current.next
        
        for _ in range(n):
            stack.pop()
        
        prev_node = stack[-1]
        prev_node.next = prev_node.next.next
        
        return dummy.next
    
    def Remove_Nth_From_End_Recursive(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Recursive - Count from end using recursion
        Time Complexity: O(L)
        Space Complexity: O(L)
        """
        def Remove_Helper(node: Optional[ListNode]) -> int:
            if not node:
                return 0
            
            count = Remove_Helper(node.next) + 1
            
            if count == n + 1:
                node.next = node.next.next
            
            return count
        
        dummy = ListNode(0)
        dummy.next = head
        Remove_Helper(dummy)
        return dummy.next
    
    def Remove_Nth_From_End_List_Storage(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        List Storage - Store all nodes in list
        Time Complexity: O(L)
        Space Complexity: O(L)
        """
        nodes = []
        current = head
        
        while current:
            nodes.append(current)
            current = current.next
        
        length = len(nodes)
        
        if n == length:
            return head.next if head.next else None
        
        target_index = length - n - 1
        nodes[target_index].next = nodes[target_index].next.next
        
        return head

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

def Test_Remove_Nth_From_End():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5], 2, [1,2,3,5]),
        ([1], 1, []),
        ([1,2], 1, [1]),
        ([1,2], 2, [2]),
        ([1,2,3], 3, [2,3])
    ]
    
    methods = [
        ("Two Pass", solution.Remove_Nth_From_End_Two_Pass),
        ("One Pass Optimal", solution.Remove_Nth_From_End_One_Pass_Optimal),
        ("Stack", solution.Remove_Nth_From_End_Stack),
        ("Recursive", solution.Remove_Nth_From_End_Recursive),
        ("List Storage", solution.Remove_Nth_From_End_List_Storage)
    ]
    
    for values, n, expected in test_cases:
        print(f"Input: {values}, n = {n}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            head = Create_Linked_List(values)
            result_head = method(head, n)
            result = Linked_List_To_Array(result_head)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Remove_Nth_From_End()
