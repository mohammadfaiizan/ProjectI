"""
Problem: Merge Sort for Linked List
URL: https://practice.geeksforgeeks.org/problems/sort-a-linked-list/1

Problem Statement:
Given Pointer/Reference to the head of the linked list, the task is to Sort the given linked list using Merge Sort.

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
    def Get_Middle(self, head):
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    
    def Merge_Two_Sorted_Lists(self, list1, list2):
        dummy = ListNode(0)
        tail = dummy
        
        while list1 and list2:
            if list1.data <= list2.data:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        
        tail.next = list1 if list1 else list2
        return dummy.next
    
    def Merge_Sort_Recursive(self, head):
        """
        Merge Sort recursive with split
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        if not head or not head.next:
            return head
        
        mid = self.Get_Middle(head)
        next_to_mid = mid.next
        mid.next = None
        
        left = self.Merge_Sort_Recursive(head)
        right = self.Merge_Sort_Recursive(next_to_mid)
        
        return self.Merge_Two_Sorted_Lists(left, right)
    
    def Get_Length(self, head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    def Split_List(self, head, n):
        for i in range(1, n):
            if head:
                head = head.next
        if not head:
            return None
        next_node = head.next
        head.next = None
        return next_node
    
    def Merge_Sort_Iterative(self, head):
        """
        Iterative merge (bottom-up)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        length = self.Get_Length(head)
        dummy = ListNode(0)
        dummy.next = head
        
        size = 1
        while size < length:
            prev = dummy
            curr = dummy.next
            
            while curr:
                left = curr
                right = self.Split_List(left, size)
                curr = self.Split_List(right, size) if right else None
                
                prev.next = self.Merge_Two_Sorted_Lists(left, right)
                while prev.next:
                    prev = prev.next
            
            size *= 2
        
        return dummy.next

def Test_Merge_Sort_Linked_List():
    solution = Solution()
    
    arr = [3, 5, 2, 4, 1]
    head = Create_List(arr)
    result1 = solution.Merge_Sort_Recursive(head)
    print("Test 1 - Recursive: ", end="")
    Print_List(result1)
    
    arr = [3, 5, 2, 4, 1]
    head = Create_List(arr)
    result2 = solution.Merge_Sort_Iterative(head)
    print("Test 1 - Iterative: ", end="")
    Print_List(result2)
    
    arr2 = [4, 2, 1, 3]
    head = Create_List(arr2)
    result1 = solution.Merge_Sort_Recursive(head)
    print("Test 2 - Recursive: ", end="")
    Print_List(result1)

if __name__ == "__main__":
    Test_Merge_Sort_Linked_List()
