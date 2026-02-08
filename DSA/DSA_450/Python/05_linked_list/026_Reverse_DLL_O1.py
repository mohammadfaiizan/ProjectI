"""
Problem: Can We Reverse a Linked List in Less Than O(n)?
URL: https://www.geeksforgeeks.org/can-we-reverse-a-linked-list-in-less-than-on/

Problem Statement:
A singly linked list cannot be reversed in less than O(n). However, a doubly linked list with head and tail pointers can be reversed in O(1) by swapping head and tail pointers (traversal direction changes via prev/next interpretation). This file demonstrates this concept.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 3 <-> 4 <-> 5 (head=1, tail=5)
Output: List: 5 <-> 4 <-> 3 <-> 2 <-> 1 (head=5, tail=1)
Explanation: By swapping head and tail, we effectively reverse the list in O(1)
"""

class DLLNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.prev = None

class DLL:
    def __init__(self):
        self.head = None
        self.tail = None

def Create_DLL_With_Tail(arr):
    dll = DLL()
    if not arr:
        return dll
    
    dll.head = DLLNode(arr[0])
    curr = dll.head
    for i in range(1, len(arr)):
        curr.next = DLLNode(arr[i])
        curr.next.prev = curr
        curr = curr.next
    dll.tail = curr
    return dll

def Print_DLL_Forward(head):
    curr = head
    result = []
    while curr:
        result.append(str(curr.data))
        curr = curr.next
    print(" ".join(result))

def Print_DLL_Backward(tail):
    curr = tail
    result = []
    while curr:
        result.append(str(curr.data))
        curr = curr.prev
    print(" ".join(result))

class Solution:
    def Reverse_DLL_O1(self, dll):
        """
        DLL O(1) reversal by swapping head/tail
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not dll or not dll.head:
            return
        temp = dll.head
        dll.head = dll.tail
        dll.tail = temp
    
    def Reverse_DLL_Standard(self, head):
        """
        Standard DLL reversal O(n)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        curr = head
        temp = None
        
        while curr:
            temp = curr.prev
            curr.prev = curr.next
            curr.next = temp
            curr = curr.prev
        
        if temp:
            head = temp.prev
        return head

def Test_Reverse_DLL_O1():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    dll1 = Create_DLL_With_Tail(arr1)
    print("Original (forward): ", end="")
    Print_DLL_Forward(dll1.head)
    print("Original (backward): ", end="")
    Print_DLL_Backward(dll1.tail)
    
    solution.Reverse_DLL_O1(dll1)
    print("After O(1) reversal (using head as start): ", end="")
    Print_DLL_Backward(dll1.head)
    print("After O(1) reversal (using tail as start): ", end="")
    Print_DLL_Forward(dll1.tail)
    
    arr2 = [10, 20, 30]
    head2 = Create_DLL_With_Tail(arr2).head
    print("Original: ", end="")
    Print_DLL_Forward(head2)
    head2 = solution.Reverse_DLL_Standard(head2)
    print("After O(n) standard reversal: ", end="")
    Print_DLL_Forward(head2)
    
    arr3 = [5]
    dll3 = Create_DLL_With_Tail(arr3)
    print("Original: ", end="")
    Print_DLL_Forward(dll3.head)
    solution.Reverse_DLL_O1(dll3)
    print("After O(1) reversal: ", end="")
    Print_DLL_Backward(dll3.head)

if __name__ == "__main__":
    Test_Reverse_DLL_O1()
