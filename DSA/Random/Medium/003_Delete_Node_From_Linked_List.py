"""
Problem: Delete a Node from Linked List
URL: https://leetcode.com/problems/delete-node-in-a-linked-list/

Problem Statement:
There is a singly-linked list head and we want to delete a node node in it.

You are given the node to be deleted node. You will not be given access to the first 
node of head.

All the values of the linked list are unique, and it is guaranteed that the given node 
node is not the last node in the linked list.

Delete the given node. Note that by deleting the node, we do not mean removing it from 
memory. We mean:
- The value of the given node should not exist in the linked list.
- The number of nodes in the linked list should decrease by one.
- All the values before node should be in the same order.
- All the values after node should be in the same order.

Sample Input/Output:
Input: head = [4,5,1,9], node = 5
Output: [4,1,9]

Input: head = [4,5,1,9], node = 1
Output: [4,5,9]
"""

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Delete_Node_Copy_Next(self, node: ListNode) -> None:
        """
        Copy Next Node Approach - Optimal solution
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        node.val = node.next.val
        node.next = node.next.next
    
    def Delete_Node_Swap_Values(self, node: ListNode) -> None:
        """
        Swap Values Approach
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        next_node = node.next
        node.val = next_node.val
        node.next = next_node.next
    
    def Delete_Node_Direct(self, node: ListNode) -> None:
        """
        Direct Pointer Manipulation
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        node.val, node.next = node.next.val, node.next.next
    
    def Delete_Node_Temp(self, node: ListNode) -> None:
        """
        Using Temporary Variable
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        temp = node.next
        node.val = temp.val
        node.next = temp.next
        temp = None
    
    def Delete_Node_Multiple_Steps(self, node: ListNode) -> None:
        """
        Multiple Steps Approach
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        next_node = node.next
        node.val = next_node.val
        next_next = next_node.next
        node.next = next_next

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

def Get_Node(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    current = head
    
    while current:
        if current.val == val:
            return current
        current = current.next
    
    return None

def Test_Delete_Node():
    solution = Solution()
    
    test_cases = [
        ([4,5,1,9], 5, [4,1,9]),
        ([4,5,1,9], 1, [4,5,9])
    ]
    
    for values, node_val, expected in test_cases:
        for method_name in ['Copy_Next', 'Swap_Values', 'Direct', 'Temp', 'Multiple_Steps']:
            head = Create_List(values)
            node = Get_Node(head, node_val)
            
            if method_name == 'Copy_Next':
                solution.Delete_Node_Copy_Next(node)
            elif method_name == 'Swap_Values':
                solution.Delete_Node_Swap_Values(node)
            elif method_name == 'Direct':
                solution.Delete_Node_Direct(node)
            elif method_name == 'Temp':
                solution.Delete_Node_Temp(node)
            else:
                solution.Delete_Node_Multiple_Steps(node)
            
            result = Print_List(head)
            print(f"Method: {method_name}, Delete node: {node_val}")
            print(f"Expected: {expected}, Got: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Delete_Node()

