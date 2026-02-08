"""
Problem: Merge K Sorted Linked Lists
URL: https://practice.geeksforgeeks.org/problems/merge-k-sorted-linked-lists/1

Problem Statement:
Given K sorted linked lists of different sizes. Merge them in such a way that after merging they will be a single sorted linked list.

Sample Input/Output:
Input: K = 3, Lists: 1->3->5->7, 2->4->6, 0->8->9
Output: 0->1->2->3->4->5->6->7->8->9
Explanation: All lists merged into one sorted list.
"""

import heapq

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
    def Merge_K_Sorted_Min_Heap(self, lists):
        """
        Use min heap to always get smallest element
        Time Complexity: O(n log k) where n is total nodes, k is number of lists
        Space Complexity: O(k) for heap
        """
        if not lists:
            return None
        pq = []
        for lst in lists:
            if lst:
                heapq.heappush(pq, (lst.data, lst))
        dummy = ListNode(0)
        current = dummy
        while pq:
            node = heapq.heappop(pq)[1]
            current.next = node
            current = current.next
            if node.next:
                heapq.heappush(pq, (node.next.data, node.next))
        result = dummy.next
        return result

    def Merge_Divide_Conquer_Helper(self, lists, left, right):
        if left == right:
            return lists[left]
        if left < right:
            mid = left + (right - left) // 2
            left_list = self.Merge_Divide_Conquer_Helper(lists, left, mid)
            right_list = self.Merge_Divide_Conquer_Helper(lists, mid + 1, right)
            return self.Merge_Two_Sorted(left_list, right_list)
        return None

    def Merge_Two_Sorted(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        result = None
        if a.data < b.data:
            result = a
            result.next = self.Merge_Two_Sorted(a.next, b)
        else:
            result = b
            result.next = self.Merge_Two_Sorted(a, b.next)
        return result

    def Merge_K_Sorted_Divide_Conquer(self, lists):
        """
        Divide and conquer: merge pairs recursively
        Time Complexity: O(n log k)
        Space Complexity: O(log k) for recursion stack
        """
        if not lists:
            return None
        return self.Merge_Divide_Conquer_Helper(lists, 0, len(lists) - 1)

def Test_Merge_K_Sorted_Lists():
    solution = Solution()
    
    test1 = []
    test1.append(Create_List([1, 3, 5, 7]))
    test1.append(Create_List([2, 4, 6]))
    test1.append(Create_List([0, 8, 9]))
    result1 = solution.Merge_K_Sorted_Min_Heap(test1)
    print("Test 1 Min Heap: ", end="")
    Print_List(result1)
    
    test2 = []
    test2.append(Create_List([1, 4, 5]))
    test2.append(Create_List([1, 3, 4]))
    test2.append(Create_List([2, 6]))
    result2 = solution.Merge_K_Sorted_Divide_Conquer(test2)
    print("Test 2 Divide Conquer: ", end="")
    Print_List(result2)
    
    test3 = []
    test3.append(Create_List([1]))
    test3.append(Create_List([0]))
    result3 = solution.Merge_K_Sorted_Min_Heap(test3)
    print("Test 3 Two Lists: ", end="")
    Print_List(result3)

if __name__ == "__main__":
    Test_Merge_K_Sorted_Lists()
