"""
Problem: Delete Nodes Having Greater Value on Right Side
URL: https://practice.geeksforgeeks.org/problems/delete-nodes-having-greater-value-on-right/1

Problem Statement:
Given a singly linked list, remove all the nodes which have a greater value on their right side. The rightmost node is always kept.

Sample Input/Output:
Input: 12 -> 15 -> 10 -> 11 -> 5 -> 6 -> 2 -> 3 -> NULL
Output: 15 -> 11 -> 6 -> 3 -> NULL
Explanation: Nodes 12, 10, 5, 2 are deleted as they have greater values on right.
"""

class ListNode:
    def __init__(self, x):
        self.data = x
        self.next = None

def Create_List(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for i in range(1, len(arr)):
        current.next = ListNode(arr[i])
        current = current.next
    return head

def List_To_Array(head):
    result = []
    while head:
        result.append(head.data)
        head = head.next
    return result

def Print_List(head):
    arr = List_To_Array(head)
    print(" -> ".join(map(str, arr)) + " -> NULL")

class Solution:
    def Reverse_List(self, head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev

    def Delete_Nodes_Greater_Right_Reverse_Filter_Reverse(self, head):
        """
        Reverse list, filter nodes, reverse back
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if head is None or head.next is None:
            return head
        head = self.Reverse_List(head)
        current = head
        max_so_far = current.data
        while current and current.next:
            if current.next.data < max_so_far:
                current.next = current.next.next
            else:
                max_so_far = max(max_so_far, current.next.data)
                current = current.next
        return self.Reverse_List(head)

    def Delete_Nodes_Greater_Right_Recursive(self, head):
        """
        Recursive approach processing from right to left
        Time Complexity: O(n)
        Space Complexity: O(n) for recursion stack
        """
        if head is None or head.next is None:
            return head
        head.next = self.Delete_Nodes_Greater_Right_Recursive(head.next)
        if head.next and head.data < head.next.data:
            head = head.next
        return head

def Test_Delete_Nodes_Greater_Right():
    solution = Solution()
    
    list1 = Create_List([12, 15, 10, 11, 5, 6, 2, 3])
    result1 = solution.Delete_Nodes_Greater_Right_Reverse_Filter_Reverse(list1)
    print("Test 1 Reverse Filter: ", end="")
    Print_List(result1)
    
    list2 = Create_List([10, 20, 30, 40, 50])
    result2 = solution.Delete_Nodes_Greater_Right_Recursive(list2)
    print("Test 2 Recursive: ", end="")
    Print_List(result2)
    
    list3 = Create_List([5, 2, 13, 3, 8])
    result3 = solution.Delete_Nodes_Greater_Right_Reverse_Filter_Reverse(list3)
    print("Test 3 Mixed: ", end="")
    Print_List(result3)

if __name__ == "__main__":
    Test_Delete_Nodes_Greater_Right()
