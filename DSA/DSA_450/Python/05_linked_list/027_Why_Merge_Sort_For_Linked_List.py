"""
Problem: Why Quick Sort is Preferred for Arrays and Merge Sort for Linked Lists
URL: https://www.geeksforgeeks.org/why-quick-sort-preferred-for-arrays-and-merge-sort-for-linked-lists/

Problem Statement:
Quick Sort is preferred for arrays due to cache locality and in-place partitioning. Merge Sort is preferred for linked lists since merge can be done without extra space and there's no random access penalty. This file demonstrates both algorithms on a linked list to compare.

Sample Input/Output:
Input: List: 3 -> 1 -> 4 -> 2 -> 5
Output (Merge Sort): 1 -> 2 -> 3 -> 4 -> 5
Output (Quick Sort): 1 -> 2 -> 3 -> 4 -> 5
Explanation: Both algorithms sort the list, but Merge Sort is more efficient for linked lists
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
    print(" ".join(map(str, arr)))

def Get_Middle(head):
    if not head:
        return None
    slow = head
    fast = head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def Merge_Two_Lists(left, right):
    dummy = ListNode(0)
    curr = dummy
    
    while left and right:
        if left.data <= right.data:
            curr.next = left
            left = left.next
        else:
            curr.next = right
            right = right.next
        curr = curr.next
    
    if left:
        curr.next = left
    if right:
        curr.next = right
    
    return dummy.next

class Solution:
    def Merge_Sort_Linked_List(self, head):
        """
        Merge Sort on linked list
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        if not head or not head.next:
            return head
        
        mid = Get_Middle(head)
        right = mid.next
        mid.next = None
        
        left_sorted = self.Merge_Sort_Linked_List(head)
        right_sorted = self.Merge_Sort_Linked_List(right)
        
        return Merge_Two_Lists(left_sorted, right_sorted)
    
    def Quick_Sort_Linked_List(self, head):
        """
        Quick Sort on linked list
        Time Complexity: O(n log n) average
        Space Complexity: O(log n)
        """
        if not head or not head.next:
            return head
        
        pivot = head
        smaller = None
        equal = None
        larger = None
        
        curr = head
        while curr:
            next_node = curr.next
            if curr.data < pivot.data:
                curr.next = smaller
                smaller = curr
            elif curr.data == pivot.data:
                curr.next = equal
                equal = curr
            else:
                curr.next = larger
                larger = curr
            curr = next_node
        
        smaller = self.Quick_Sort_Linked_List(smaller)
        larger = self.Quick_Sort_Linked_List(larger)
        
        result = None
        tail = None
        
        if smaller:
            result = smaller
            tail = smaller
            while tail.next:
                tail = tail.next
        
        if equal:
            if not result:
                result = equal
                tail = equal
            else:
                tail.next = equal
            while tail.next:
                tail = tail.next
        
        if larger:
            if not result:
                result = larger
            else:
                tail.next = larger
        
        return result

def Test_Why_Merge_Sort_For_Linked_List():
    solution = Solution()
    
    arr1 = [3, 1, 4, 2, 5]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    
    head1_merge = Create_List(arr1)
    head1_merge = solution.Merge_Sort_Linked_List(head1_merge)
    print("Merge Sort: ", end="")
    Print_List(head1_merge)
    
    head1_quick = Create_List(arr1)
    head1_quick = solution.Quick_Sort_Linked_List(head1_quick)
    print("Quick Sort: ", end="")
    Print_List(head1_quick)
    
    arr2 = [10, 5, 8, 3, 1, 9, 2]
    head2 = Create_List(arr2)
    print("Original: ", end="")
    Print_List(head2)
    
    head2_merge = Create_List(arr2)
    head2_merge = solution.Merge_Sort_Linked_List(head2_merge)
    print("Merge Sort: ", end="")
    Print_List(head2_merge)
    
    head2_quick = Create_List(arr2)
    head2_quick = solution.Quick_Sort_Linked_List(head2_quick)
    print("Quick Sort: ", end="")
    Print_List(head2_quick)

if __name__ == "__main__":
    Test_Why_Merge_Sort_For_Linked_List()
