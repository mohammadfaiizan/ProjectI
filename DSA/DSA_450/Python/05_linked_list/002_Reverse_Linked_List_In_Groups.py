"""
Problem: Reverse a Linked List in Groups of Given Size
URL: https://practice.geeksforgeeks.org/problems/reverse-a-linked-list-in-groups-of-given-size/1

Problem Statement:
Given a linked list, reverse every k nodes.

Sample Input/Output:
Input: 1->2->3->4->5->NULL, k=3
Output: 3->2->1->4->5->NULL
Explanation: First 3 nodes reversed, then next group.
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

class Solution:
    def Reverse_Groups_Recursive(self, head, k):
        """
        Recursive group reversal approach
        Time Complexity: O(n)
        Space Complexity: O(n/k)
        """
        if not head:
            return None
        
        curr = head
        prev = None
        next_node = None
        count = 0
        
        while curr and count < k:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
            count += 1
        
        if next_node:
            head.next = self.Reverse_Groups_Recursive(next_node, k)
        
        return prev
    
    def Reverse_Groups_Stack_Based(self, head, k):
        """
        Stack-based approach
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if not head:
            return None
        
        st = []
        curr = head
        prev = None
        
        while curr:
            count = 0
            while curr and count < k:
                st.append(curr)
                curr = curr.next
                count += 1
            
            while st:
                if not prev:
                    prev = st.pop()
                    head = prev
                else:
                    prev.next = st.pop()
                    prev = prev.next
        
        if prev:
            prev.next = None
        return head

def Test_Reverse_Linked_List_In_Groups():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    head1 = solution.Reverse_Groups_Recursive(head1, 3)
    print("Reversed in groups of 3 (Recursive): ", end="")
    Print_List(head1)
    
    arr2 = [1, 2, 3, 4, 5, 6, 7, 8]
    head2 = Create_List(arr2)
    print("\nOriginal: ", end="")
    Print_List(head2)
    head2 = solution.Reverse_Groups_Stack_Based(head2, 4)
    print("Reversed in groups of 4 (Stack): ", end="")
    Print_List(head2)
    
    arr3 = [1, 2]
    head3 = Create_List(arr3)
    print("\nOriginal: ", end="")
    Print_List(head3)
    head3 = solution.Reverse_Groups_Recursive(head3, 2)
    print("Reversed in groups of 2: ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Reverse_Linked_List_In_Groups()
