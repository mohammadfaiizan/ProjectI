"""
Problem: Quick Sort on Linked List
URL: https://practice.geeksforgeeks.org/problems/quick-sort-on-linked-list/1

Problem Statement:
Sort the given Linked List using quicksort. which takes O(n^2) time in worst case and O(nLogn) in average and best cases, otherwise you may get TLE.

Sample Input/Output:
Input: N = 5, value[] = {3,5,2,4,1}
Output: 1->2->3->4->5
Explanation: After sorting the given linked list, the resultant will be 1->2->3->4->5.
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
    def Get_Tail(self, head):
        while head and head.next:
            head = head.next
        return head
    
    def Quick_Sort_Helper(self, head, tail):
        if not head or head == tail or not head.next:
            return head
        
        pivot = self.Partition(head, tail)
        
        if head != pivot:
            temp = head
            while temp.next != pivot:
                temp = temp.next
            temp.next = None
            head = self.Quick_Sort_Helper(head, temp)
            temp = self.Get_Tail(head)
            temp.next = pivot
        
        pivot.next = self.Quick_Sort_Helper(pivot.next, tail)
        return head
    
    def Partition(self, head, tail):
        pivot = tail
        prev = None
        curr = head
        end = tail
        
        while curr != pivot:
            if curr.data < pivot.data:
                if not prev:
                    prev = curr
                else:
                    prev = prev.next
                prev.data, curr.data = curr.data, prev.data
            curr = curr.next
        
        if not prev:
            prev = head
        else:
            prev = prev.next
        prev.data, pivot.data = pivot.data, prev.data
        return prev
    
    def Quick_Sort(self, head):
        """
        Quick Sort with last element as pivot
        Time Complexity: O(n log n) average, O(n^2) worst
        Space Complexity: O(log n) stack
        """
        if not head or not head.next:
            return head
        
        tail = self.Get_Tail(head)
        return self.Quick_Sort_Helper(head, tail)

def Test_Quick_Sort_Linked_List():
    solution = Solution()
    
    arr = [3, 5, 2, 4, 1]
    head = Create_List(arr)
    result = solution.Quick_Sort(head)
    print("Test 1: ", end="")
    Print_List(result)
    
    arr2 = [4, 2, 1, 3]
    head = Create_List(arr2)
    result = solution.Quick_Sort(head)
    print("Test 2: ", end="")
    Print_List(result)
    
    arr3 = [5, 1, 4, 2, 3]
    head = Create_List(arr3)
    result = solution.Quick_Sort(head)
    print("Test 3: ", end="")
    Print_List(result)

if __name__ == "__main__":
    Test_Quick_Sort_Linked_List()
