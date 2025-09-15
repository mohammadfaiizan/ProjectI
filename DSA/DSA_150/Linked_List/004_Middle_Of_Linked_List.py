"""
Problem: Middle of the Linked List
URL: https://leetcode.com/problems/middle-of-the-linked-list/

Problem Statement:
Given the head of a singly linked list, return the middle node of the linked list.
If there are two middle nodes, return the second middle node.

Sample Input/Output:
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.
"""

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Middle_Node_Tortoise_Hare_Optimal(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Tortoise and Hare Optimal - Fast and slow pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def Middle_Node_Two_Pass(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pass - Count length then find middle
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        middle_index = length // 2
        current = head
        
        for _ in range(middle_index):
            current = current.next
        
        return current
    
    def Middle_Node_List_Storage(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        List Storage - Store all nodes then return middle
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        nodes = []
        current = head
        
        while current:
            nodes.append(current)
            current = current.next
        
        return nodes[len(nodes) // 2]
    
    def Middle_Node_Counter_Method(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Counter Method - Use counter with single pass
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = head
        middle = head
        count = 0
        
        while current:
            if count % 2 == 1:
                middle = middle.next
            current = current.next
            count += 1
        
        return middle
    
    def Middle_Node_Recursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Recursive - Find middle using recursion
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Get_Length(node: Optional[ListNode]) -> int:
            if not node:
                return 0
            return 1 + Get_Length(node.next)
        
        def Get_Middle(node: Optional[ListNode], target_index: int, current_index: int) -> Optional[ListNode]:
            if current_index == target_index:
                return node
            return Get_Middle(node.next, target_index, current_index + 1)
        
        length = Get_Length(head)
        middle_index = length // 2
        
        return Get_Middle(head, middle_index, 0)
    
    def Middle_Node_Stack_Based(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Stack Based - Use stack to find middle
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        current = head
        
        while current:
            stack.append(current)
            current = current.next
        
        middle_index = len(stack) // 2
        return stack[middle_index]

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

def Test_Middle_Node():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5], [3,4,5]),
        ([1,2,3,4,5,6], [4,5,6]),
        ([1], [1]),
        ([1,2], [2]),
        ([1,2,3], [2,3])
    ]
    
    methods = [
        ("Tortoise Hare Optimal", solution.Middle_Node_Tortoise_Hare_Optimal),
        ("Two Pass", solution.Middle_Node_Two_Pass),
        ("List Storage", solution.Middle_Node_List_Storage),
        ("Counter Method", solution.Middle_Node_Counter_Method),
        ("Recursive", solution.Middle_Node_Recursive),
        ("Stack Based", solution.Middle_Node_Stack_Based)
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
    Test_Middle_Node()
