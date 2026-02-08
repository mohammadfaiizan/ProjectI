"""
Problem: Flatten a Linked List
URL: https://practice.geeksforgeeks.org/problems/flattening-a-linked-list/1

Problem Statement:
Given a linked list where every node has a next pointer and a bottom/down pointer. All bottom lists are sorted. Flatten the list into a single sorted list using bottom pointers.

Sample Input/Output:
Input: 5 -> 10 -> 19 -> 28
       |    |     |     |
       V    V     V     V
       7    20    22    35
       |          |     |
       V          V     V
       8          50    40
       |                |
       V                V
       30               45
Output: 5 -> 7 -> 8 -> 10 -> 19 -> 20 -> 22 -> 28 -> 30 -> 35 -> 40 -> 45 -> 50
Explanation: All nodes are merged using bottom pointers into a single sorted list.
"""

import heapq

class FlatNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.bottom = None

def Create_Flat_List(lists):
    if not lists:
        return None
    head = FlatNode(lists[0][0])
    current = head
    for i in range(len(lists)):
        row_head = None
        row_current = None
        for j in range(len(lists[i])):
            node = FlatNode(lists[i][j])
            if row_head is None:
                row_head = node
                row_current = node
            else:
                row_current.bottom = node
                row_current = row_current.bottom
        if i == 0:
            head = row_head
        else:
            current.next = row_head
        current = row_head
        while current.bottom:
            current = current.bottom
    return head

def List_To_Array(head):
    result = []
    while head:
        result.append(head.data)
        head = head.bottom
    return result

def Print_Flattened_List(head):
    arr = List_To_Array(head)
    print(" -> ".join(map(str, arr)))

class Solution:
    def Merge_Two_Sorted(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        result = None
        if a.data < b.data:
            result = a
            result.bottom = self.Merge_Two_Sorted(a.bottom, b)
        else:
            result = b
            result.bottom = self.Merge_Two_Sorted(a, b.bottom)
        result.next = None
        return result
    
    def Flatten_Recursive_Merge(self, root):
        """
        Recursive merge approach: Merge bottom lists recursively
        Time Complexity: O(n) where n is total nodes
        Space Complexity: O(1) excluding recursion stack
        """
        if root is None or root.next is None:
            return root
        root.next = self.Flatten_Recursive_Merge(root.next)
        root = self.Merge_Two_Sorted(root, root.next)
        return root

    def Flatten_Min_Heap(self, root):
        """
        Min heap approach: Use priority queue to merge all lists
        Time Complexity: O(n log k) where k is number of lists
        Space Complexity: O(k) for heap
        """
        if root is None:
            return None
        pq = []
        current = root
        while current:
            temp = current
            while temp:
                heapq.heappush(pq, (temp.data, temp))
                temp = temp.bottom
            current = current.next
        dummy = FlatNode(0)
        result = dummy
        while pq:
            node = heapq.heappop(pq)[1]
            dummy.bottom = node
            dummy = dummy.bottom
            dummy.next = None
        return result.bottom

def Test_Flatten_Linked_List():
    solution = Solution()
    
    test1 = [[5, 7, 8, 30], [10, 20], [19, 22, 50], [28, 35, 40, 45]]
    list1 = Create_Flat_List(test1)
    result1 = solution.Flatten_Recursive_Merge(list1)
    print("Test 1 Recursive: ", end="")
    Print_Flattened_List(result1)
    
    test2 = [[1, 3, 5], [2, 4]]
    list2 = Create_Flat_List(test2)
    result2 = solution.Flatten_Min_Heap(list2)
    print("Test 2 Min Heap: ", end="")
    Print_Flattened_List(result2)
    
    test3 = [[1]]
    list3 = Create_Flat_List(test3)
    result3 = solution.Flatten_Recursive_Merge(list3)
    print("Test 3 Single: ", end="")
    Print_Flattened_List(result3)

if __name__ == "__main__":
    Test_Flatten_Linked_List()
