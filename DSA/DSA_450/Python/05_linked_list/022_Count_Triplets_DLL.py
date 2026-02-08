"""
Problem: Count Triplets in Sorted Doubly Linked List whose Sum equals X
URL: https://www.geeksforgeeks.org/count-triplets-sorted-doubly-linked-list-whose-sum-equal-given-value-x/

Problem Statement:
Given a sorted doubly linked list and a target value X, count all triplets whose sum equals X.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 4 <-> 5 <-> 6 <-> 8 <-> 9, X = 17
Output: 2
Explanation: Triplets: (4, 5, 8) = 17, (2, 6, 9) = 17
"""

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
    def Count_Triplets_Brute_Force(self, head, target):
        """
        Brute force approach
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        """
        count = 0
        first = head
        while first:
            second = first.next
            while second:
                third = second.next
                while third:
                    if first.data + second.data + third.data == target:
                        count += 1
                    third = third.next
                second = second.next
            first = first.next
        return count
    
    def Count_Triplets_Hashing(self, head, target):
        """
        Hashing approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        count = 0
        first = head
        while first:
            second = first.next
            seen = set()
            while second:
                complement = target - first.data - second.data
                if complement in seen:
                    count += 1
                seen.add(second.data)
                second = second.next
            first = first.next
        return count
    
    def Count_Triplets_Two_Pointer(self, head, target):
        """
        Two pointer approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        first = head
        
        while first:
            second = first.next
            third = head
            while third.next:
                third = third.next
            
            while second and third and second != third and third.next != second:
                sum_val = first.data + second.data + third.data
                if sum_val == target:
                    count += 1
                    second = second.next
                    third = third.prev
                elif sum_val < target:
                    second = second.next
                else:
                    third = third.prev
            first = first.next
        
        return count

def Test_Count_Triplets_DLL():
    solution = Solution()
    
    arr1 = [1, 2, 4, 5, 6, 8, 9]
    head1 = Create_DLL(arr1)
    print("List: ", end="")
    Print_DLL(head1)
    count1 = solution.Count_Triplets_Brute_Force(head1, 17)
    print(f"Triplets with sum 17 (Brute Force): {count1}")
    
    count2 = solution.Count_Triplets_Hashing(head1, 17)
    print(f"Triplets with sum 17 (Hashing): {count2}")
    
    count3 = solution.Count_Triplets_Two_Pointer(head1, 17)
    print(f"Triplets with sum 17 (Two Pointer): {count3}")
    
    arr2 = [1, 2, 3, 4, 5]
    head2 = Create_DLL(arr2)
    print("List: ", end="")
    Print_DLL(head2)
    count4 = solution.Count_Triplets_Two_Pointer(head2, 6)
    print(f"Triplets with sum 6: {count4}")

if __name__ == "__main__":
    Test_Count_Triplets_DLL()
