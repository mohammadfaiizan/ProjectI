"""
Problem: Sort a K-Sorted Doubly Linked List
URL: https://www.geeksforgeeks.org/sort-k-sorted-doubly-linked-list/

Problem Statement:
Given a K-sorted doubly linked list where each node is at most K positions away from its correct position, sort the list.

Sample Input/Output:
Input: List: 3 <-> 6 <-> 2 <-> 12 <-> 56 <-> 8, K = 2
Output: List: 2 <-> 3 <-> 6 <-> 8 <-> 12 <-> 56
Explanation: Each element is at most 2 positions away from its sorted position
"""

import heapq

class DLLNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.prev = None

def Create_DLL(arr):
    if not arr:
        return None
    head = DLLNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = DLLNode(arr[i])
        curr.next.prev = curr
        curr = curr.next
    return head

def Print_DLL(head):
    curr = head
    result = []
    while curr:
        result.append(str(curr.data))
        curr = curr.next
    print(" ".join(result))

class Solution:
    def Sort_K_Sorted_Insertion_Sort(self, head, k):
        """
        Insertion sort with swaps
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        curr = head.next
        while curr:
            temp = curr
            prev = curr.prev
            
            count = 0
            while prev and prev.data > temp.data and count < k:
                prev.data, temp.data = temp.data, prev.data
                temp = prev
                prev = prev.prev
                count += 1
            
            curr = curr.next
        
        return head
    
    def Sort_K_Sorted_Min_Heap(self, head, k):
        """
        Min heap approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        if not head:
            return None
        
        pq = []
        curr = head
        
        for i in range(k + 1):
            if curr:
                heapq.heappush(pq, (curr.data, curr))
                curr = curr.next
        
        new_head = None
        tail = None
        
        while pq:
            node = heapq.heappop(pq)[1]
            
            if not new_head:
                new_head = node
                tail = node
                new_head.prev = None
            else:
                tail.next = node
                node.prev = tail
                tail = node
            
            if curr:
                heapq.heappush(pq, (curr.data, curr))
                curr = curr.next
        
        if tail:
            tail.next = None
        return new_head

def Test_Sort_K_Sorted_DLL():
    solution = Solution()
    
    arr1 = [3, 6, 2, 12, 56, 8]
    head1 = Create_DLL(arr1)
    print("Original: ", end="")
    Print_DLL(head1)
    head1 = solution.Sort_K_Sorted_Insertion_Sort(head1, 2)
    print("Sorted (Insertion Sort): ", end="")
    Print_DLL(head1)
    
    arr2 = [3, 6, 2, 12, 56, 8]
    head2 = Create_DLL(arr2)
    print("Original: ", end="")
    Print_DLL(head2)
    head2 = solution.Sort_K_Sorted_Min_Heap(head2, 2)
    print("Sorted (Min Heap): ", end="")
    Print_DLL(head2)
    
    arr3 = [10, 9, 8, 7, 4, 70, 60, 50]
    head3 = Create_DLL(arr3)
    print("Original: ", end="")
    Print_DLL(head3)
    head3 = solution.Sort_K_Sorted_Min_Heap(head3, 4)
    print("Sorted (K=4): ", end="")
    Print_DLL(head3)

if __name__ == "__main__":
    Test_Sort_K_Sorted_DLL()
