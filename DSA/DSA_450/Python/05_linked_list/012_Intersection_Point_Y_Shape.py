"""
Problem: Intersection Point in Y Shaped Linked Lists
URL: https://practice.geeksforgeeks.org/problems/intersection-point-in-y-shapped-linked-lists/1

Problem Statement:
There are two singly linked lists of size N and M in a system. But, due to some programming error the end node of one of the linked list got linked into the second list, forming an inverted Y shaped list. Write a program to get the point where two linked lists intersect.

Sample Input/Output:
Input: LinkList1 = 3->6->9->common, LinkList2 = 10->common, common = 15->30->NULL
Output: 15
Explanation: The Y shaped list ends after 15.
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

class Solution:
    def Intersect_Point_Hashing(self, head1, head2):
        """
        Hashing approach
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        """
        visited = set()
        while head1:
            visited.add(head1)
            head1 = head1.next
        while head2:
            if head2 in visited:
                return head2.data
            head2 = head2.next
        return -1
    
    def Intersect_Point_Difference(self, head1, head2):
        """
        Difference of node counts
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        len1 = 0
        len2 = 0
        temp1 = head1
        temp2 = head2
        
        while temp1:
            len1 += 1
            temp1 = temp1.next
        while temp2:
            len2 += 1
            temp2 = temp2.next
        
        diff = abs(len1 - len2)
        if len1 > len2:
            while diff > 0:
                head1 = head1.next
                diff -= 1
        else:
            while diff > 0:
                head2 = head2.next
                diff -= 1
        
        while head1 and head2:
            if head1 == head2:
                return head1.data
            head1 = head1.next
            head2 = head2.next
        return -1
    
    def Intersect_Point_Two_Pointer(self, head1, head2):
        """
        Two pointer technique
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        ptr1 = head1
        ptr2 = head2
        
        while ptr1 != ptr2:
            ptr1 = ptr1.next if ptr1 else head2
            ptr2 = ptr2.next if ptr2 else head1
        
        return ptr1.data if ptr1 else -1

def Test_Intersection_Point_Y_Shape():
    solution = Solution()
    
    common = ListNode(15)
    common.next = ListNode(30)
    
    head1 = ListNode(3)
    head1.next = ListNode(6)
    head1.next.next = ListNode(9)
    head1.next.next.next = common
    
    head2 = ListNode(10)
    head2.next = common
    
    result1 = solution.Intersect_Point_Hashing(head1, head2)
    print("Test 1 - Hashing:", result1)
    
    result2 = solution.Intersect_Point_Difference(head1, head2)
    print("Test 1 - Difference:", result2)
    
    result3 = solution.Intersect_Point_Two_Pointer(head1, head2)
    print("Test 1 - Two Pointer:", result3)

if __name__ == "__main__":
    Test_Intersection_Point_Y_Shape()
