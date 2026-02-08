"""
Problem: Add 1 to a Number Represented as Linked List
URL: https://practice.geeksforgeeks.org/problems/add-1-to-a-number-represented-as-linked-list/1

Problem Statement:
Add 1 to a number represented as a linked list.

Sample Input/Output:
Input: 1->9->9->9->NULL
Output: 2->0->0->0->NULL
Explanation: 1999 + 1 = 2000
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
    print("->".join(map(str, arr)) + "->NULL")

def Reverse_List(head):
    prev = None
    curr = head
    next_node = None
    
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    return prev

class Solution:
    def Add_One_Reverse_Add_Reverse(self, head):
        """
        Reverse, Add, Reverse approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return ListNode(1)
        
        head = Reverse_List(head)
        curr = head
        carry = 1
        
        while curr and carry:
            sum_val = curr.data + carry
            curr.data = sum_val % 10
            carry = sum_val // 10
            
            if not curr.next and carry:
                curr.next = ListNode(carry)
                carry = 0
            
            curr = curr.next
        
        head = Reverse_List(head)
        return head
    
    def Add_One_Recursive_Helper(self, head):
        if not head:
            return 1
        
        carry = self.Add_One_Recursive_Helper(head.next)
        sum_val = head.data + carry
        head.data = sum_val % 10
        return sum_val // 10
    
    def Add_One_Recursive_Carry(self, head):
        """
        Recursive carry approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head:
            return ListNode(1)
        
        carry = self.Add_One_Recursive_Helper(head)
        
        if carry:
            new_head = ListNode(carry)
            new_head.next = head
            return new_head
        
        return head

def Test_Add_One_To_Number():
    solution = Solution()
    
    arr1 = [1, 9, 9, 9]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    head1 = solution.Add_One_Reverse_Add_Reverse(head1)
    print("After adding 1 (Reverse-Add-Reverse): ", end="")
    Print_List(head1)
    
    arr2 = [9, 9, 9]
    head2 = Create_List(arr2)
    print("\nOriginal: ", end="")
    Print_List(head2)
    head2 = solution.Add_One_Recursive_Carry(head2)
    print("After adding 1 (Recursive): ", end="")
    Print_List(head2)
    
    arr3 = [1, 2, 3]
    head3 = Create_List(arr3)
    print("\nOriginal: ", end="")
    Print_List(head3)
    head3 = solution.Add_One_Reverse_Add_Reverse(head3)
    print("After adding 1: ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Add_One_To_Number()
