"""
Problem: Middle of the Linked List
URL: https://leetcode.com/problems/middle-of-the-linked-list/

Problem Statement:
Given the head of a singly linked list, return the middle node of the linked list. If there are two middle nodes, return the second middle node.

Sample Input/Output:
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.
"""

class ListNode:
    def __init__(self, x):
        self.data = x
        self.next = None

def Create_List(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = ListNode(arr[i])
        curr = curr.next
    return head

def List_To_Array(head):
    result = []
    while head:
        result.append(head.data)
        head = head.next
    return result

def Print_List(head):
    arr = List_To_Array(head)
    print("->".join(map(str, arr)))

class Solution:
    def Middle_Node_Slow_Fast(self, head):
        """
        Slow-Fast pointer (Tortoise and Hare)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def Middle_Node_Count_Based(self, head):
        """
        Count-based approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count = 0
        temp = head
        while temp:
            count += 1
            temp = temp.next
        
        mid = count // 2
        temp = head
        while mid > 0:
            temp = temp.next
            mid -= 1
        
        return temp

def Test_Middle_Of_Linked_List():
    solution = Solution()
    
    arr = [1, 2, 3, 4, 5]
    head = Create_List(arr)
    result1 = solution.Middle_Node_Slow_Fast(head)
    print("Test 1 - Slow-Fast: ", end="")
    Print_List(result1)
    
    head = Create_List(arr)
    result2 = solution.Middle_Node_Count_Based(head)
    print("Test 1 - Count-Based: ", end="")
    Print_List(result2)
    
    arr2 = [1, 2, 3, 4, 5, 6]
    head = Create_List(arr2)
    result1 = solution.Middle_Node_Slow_Fast(head)
    print("Test 2 - Slow-Fast: ", end="")
    Print_List(result1)

if __name__ == "__main__":
    Test_Middle_Of_Linked_List()
