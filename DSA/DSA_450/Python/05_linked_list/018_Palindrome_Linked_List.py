"""
Problem: Check if Linked List is Palindrome
URL: https://practice.geeksforgeeks.org/problems/check-if-linked-list-is-pallindrome/1

Problem Statement:
Given a singly linked list of size N of integers. The task is to check if the given linked list is palindrome or not.

Sample Input/Output:
Input: N = 3, value[] = {1,2,1}
Output: 1
Explanation: The given linked list is 1->2->1, which is a palindrome and hence, the output is 1.
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
    def Reverse_List(self, head):
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev
    
    def Is_Palindrome_Reverse_Second_Half(self, head):
        """
        Reverse second half
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return True
        
        slow = head
        fast = head
        
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        second_half = slow.next
        slow.next = None
        second_half = self.Reverse_List(second_half)
        
        first = head
        second = second_half
        result = True
        
        while first and second:
            if first.data != second.data:
                result = False
                break
            first = first.next
            second = second.next
        
        second_half = self.Reverse_List(second_half)
        slow.next = second_half
        
        return result
    
    def Is_Palindrome_Stack(self, head):
        """
        Stack-based approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return True
        
        st = []
        curr = head
        while curr:
            st.append(curr.data)
            curr = curr.next
        
        curr = head
        while curr:
            if curr.data != st.pop():
                return False
            curr = curr.next
        
        return True
    
    def Is_Palindrome_Recursive_Helper(self, curr, front):
        if not curr:
            return True
        
        result = self.Is_Palindrome_Recursive_Helper(curr.next, front)
        if not result:
            return False
        
        if curr.data != front[0].data:
            return False
        front[0] = front[0].next
        return True
    
    def Is_Palindrome_Recursive(self, head):
        """
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        front = [head]
        return self.Is_Palindrome_Recursive_Helper(head, front)

def Test_Palindrome_Linked_List():
    solution = Solution()
    
    arr = [1, 2, 1]
    head = Create_List(arr)
    result1 = solution.Is_Palindrome_Reverse_Second_Half(head)
    print("Test 1 - Reverse Second Half:", result1)
    
    head = Create_List(arr)
    result2 = solution.Is_Palindrome_Stack(head)
    print("Test 1 - Stack:", result2)
    
    head = Create_List(arr)
    result3 = solution.Is_Palindrome_Recursive(head)
    print("Test 1 - Recursive:", result3)
    
    arr2 = [1, 2, 3]
    head = Create_List(arr2)
    result1 = solution.Is_Palindrome_Reverse_Second_Half(head)
    print("Test 2 - Reverse Second Half:", result1)

if __name__ == "__main__":
    Test_Palindrome_Linked_List()
