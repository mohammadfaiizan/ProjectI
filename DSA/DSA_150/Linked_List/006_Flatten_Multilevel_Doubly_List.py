"""
Problem: Flatten a Multilevel Doubly Linked List
URL: https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/

Problem Statement:
You are given a doubly linked list, which contains nodes that have a next pointer, a previous pointer, and an additional child pointer. 
This child pointer may or may not point to a separate doubly linked list, also containing these special nodes. 
These child lists may have one or more children of their own, and so on, to produce a multilevel data structure.
Given the head of the first level of the list, flatten the list so that all the nodes appear in a single-level, doubly linked list.

Sample Input/Output:
Input: head = [1,2,3,7,8,11,9,10,4,5,6,12]
Output: [1,2,3,7,8,11,9,10,4,5,6,12]
Explanation: The multilevel linked list in the input is shown. After flattening the multilevel linked list it becomes the output.
"""

from typing import Optional

class Node:
    def __init__(self, val, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

class Solution:
    def Flatten_DFS_Recursive_Optimal(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        DFS Recursive Optimal - Process child lists recursively
        Time Complexity: O(n)
        Space Complexity: O(d) where d is max depth
        """
        def Flatten_Helper(node: 'Optional[Node]') -> 'Optional[Node]':
            current = node
            last = None
            
            while current:
                if current.child:
                    child_last = Flatten_Helper(current.child)
                    
                    next_node = current.next
                    current.next = current.child
                    current.child.prev = current
                    
                    if next_node:
                        child_last.next = next_node
                        next_node.prev = child_last
                    
                    current.child = None
                    last = child_last
                else:
                    last = current
                
                current = current.next
            
            return last
        
        if head:
            Flatten_Helper(head)
        return head
    
    def Flatten_Stack_Iterative(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        Stack Iterative - Use stack to track next nodes
        Time Complexity: O(n)
        Space Complexity: O(d)
        """
        if not head:
            return head
        
        stack = []
        current = head
        
        while current:
            if current.child:
                if current.next:
                    stack.append(current.next)
                
                current.next = current.child
                current.child.prev = current
                current.child = None
            
            if not current.next and stack:
                next_node = stack.pop()
                current.next = next_node
                next_node.prev = current
            
            current = current.next
        
        return head
    
    def Flatten_Two_Pass(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        Two Pass - First collect all nodes, then rebuild
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Collect_Nodes(node: 'Optional[Node]', nodes: list) -> None:
            if not node:
                return
            
            nodes.append(node)
            
            if node.child:
                Collect_Nodes(node.child, nodes)
            
            if node.next:
                Collect_Nodes(node.next, nodes)
        
        if not head:
            return head
        
        all_nodes = []
        Collect_Nodes(head, all_nodes)
        
        for i in range(len(all_nodes)):
            node = all_nodes[i]
            node.prev = all_nodes[i - 1] if i > 0 else None
            node.next = all_nodes[i + 1] if i < len(all_nodes) - 1 else None
            node.child = None
        
        return all_nodes[0] if all_nodes else None
    
    def Flatten_Queue_BFS(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        Queue BFS - Use queue for level-order processing
        Time Complexity: O(n)
        Space Complexity: O(w) where w is max width
        """
        from collections import deque
        
        if not head:
            return head
        
        nodes = []
        queue = deque([head])
        
        while queue:
            node = queue.popleft()
            nodes.append(node)
            
            if node.next:
                queue.append(node.next)
            
            if node.child:
                queue.append(node.child)
        
        for i in range(len(nodes)):
            node = nodes[i]
            node.prev = nodes[i - 1] if i > 0 else None
            node.next = nodes[i + 1] if i < len(nodes) - 1 else None
            node.child = None
        
        return nodes[0] if nodes else None
    
    def Flatten_Preorder_Traversal(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        Preorder Traversal - Preorder DFS with explicit stack
        Time Complexity: O(n)
        Space Complexity: O(d)
        """
        if not head:
            return head
        
        result = []
        stack = [head]
        
        while stack:
            node = stack.pop()
            result.append(node)
            
            if node.next:
                stack.append(node.next)
            
            if node.child:
                stack.append(node.child)
        
        for i in range(len(result)):
            node = result[i]
            node.prev = result[i - 1] if i > 0 else None
            node.next = result[i + 1] if i < len(result) - 1 else None
            node.child = None
        
        return result[0] if result else None

def Create_Multilevel_List():
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node4 = Node(4)
    node5 = Node(5)
    node6 = Node(6)
    node7 = Node(7)
    node8 = Node(8)
    node9 = Node(9)
    node10 = Node(10)
    node11 = Node(11)
    node12 = Node(12)
    
    node1.next = node2
    node2.prev = node1
    node2.next = node3
    node3.prev = node2
    
    node3.child = node7
    node7.next = node8
    node8.prev = node7
    node8.next = node11
    node11.prev = node8
    node8.child = node9
    node9.next = node10
    node10.prev = node9
    
    node3.next = node4
    node4.prev = node3
    node4.next = node5
    node5.prev = node4
    node5.next = node6
    node6.prev = node5
    
    node11.child = node12
    
    return node1

def Print_Flattened_List(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

def Test_Flatten():
    solution = Solution()
    
    methods = [
        ("DFS Recursive Optimal", solution.Flatten_DFS_Recursive_Optimal),
        ("Stack Iterative", solution.Flatten_Stack_Iterative),
        ("Two Pass", solution.Flatten_Two_Pass),
        ("Queue BFS", solution.Flatten_Queue_BFS),
        ("Preorder Traversal", solution.Flatten_Preorder_Traversal)
    ]
    
    expected = [1,2,3,7,8,9,10,11,12,4,5,6]
    
    for method_name, method in methods:
        head = Create_Multilevel_List()
        result_head = method(head)
        result = Print_Flattened_List(result_head)
        print(f"{method_name}: {result}")
        print(f"Expected: {expected}")
        print(f"Match: {result == expected}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Flatten()
