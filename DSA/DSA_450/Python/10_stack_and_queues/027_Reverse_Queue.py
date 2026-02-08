"""
Problem: Reverse a Queue
URL: https://practice.geeksforgeeks.org/problems/queue-reversal/1

Problem Statement:
Reverse all elements in a queue.

Sample Input/Output:
Input: queue [1,2,3,4,5]
Output: [5,4,3,2,1]
"""

from collections import deque


class Solution:
    def Reverse_Queue_Recursion(self, q):
        """
        Reverse queue using recursion.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not q:
            return q
        front = q.popleft()
        reversed_q = self.Reverse_Queue_Recursion(q)
        reversed_q.append(front)
        return reversed_q

    def Reverse_Queue_Stack(self, q):
        """
        Reverse queue using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        while q:
            st.append(q.popleft())
        while st:
            q.append(st.pop())
        return q


def Test_Reverse_Queue_Recursion():
    solution = Solution()
    q1 = deque([1, 2, 3, 4, 5])
    
    reversed1 = solution.Reverse_Queue_Recursion(q1)
    print("Recursion - Reversed Queue: ", end="")
    while reversed1:
        print(reversed1.popleft(), end=" ")
    print()


def Test_Reverse_Queue_Stack():
    solution = Solution()
    q2 = deque([1, 2, 3, 4, 5])
    
    reversed2 = solution.Reverse_Queue_Stack(q2)
    print("Stack - Reversed Queue: ", end="")
    while reversed2:
        print(reversed2.popleft(), end=" ")
    print()


if __name__ == "__main__":
    Test_Reverse_Queue_Recursion()
    Test_Reverse_Queue_Stack()
