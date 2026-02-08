"""
Problem: Reverse First K Elements of Queue
URL: https://practice.geeksforgeeks.org/problems/reverse-first-k-elements-of-queue/1

Problem Statement:
Given an integer K and a queue of integers, reverse the order of the first K elements
of the queue, leaving the other elements in the same relative order.

Sample Input/Output:
Input: queue = [1, 2, 3, 4, 5], k = 3
Output: [3, 2, 1, 4, 5]

Input: queue = [4, 3, 2, 1], k = 4
Output: [1, 2, 3, 4]
"""

from collections import deque


class Solution:
    def Reverse_First_K_Stack(self, q, k):
        """
        Reverse first K elements using stack.
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if k <= 0 or k > len(q):
            return q
        st = []
        for i in range(k):
            st.append(q.popleft())
        while st:
            q.append(st.pop())
        remaining = len(q) - k
        for i in range(remaining):
            q.append(q.popleft())
        return q

    def Reverse_First_K_Recursive(self, q, k):
        """
        Reverse first K elements using recursion.
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if k <= 0:
            return q
        val = q.popleft()
        self.Reverse_First_K_Recursive(q, k - 1)
        q.append(val)
        remaining = len(q) - 1
        for i in range(remaining):
            q.append(q.popleft())
        return q


def Test_Reverse_First_K():
    solution = Solution()

    def Print_Queue(q):
        print("[", end="")
        first = True
        temp_q = deque(q)
        while temp_q:
            if not first:
                print(", ", end="")
            print(temp_q.popleft(), end="")
            first = False
        print("]")

    q1 = deque([1, 2, 3, 4, 5])
    print("Original: ", end="")
    Print_Queue(q1)
    result1 = solution.Reverse_First_K_Stack(deque([1, 2, 3, 4, 5]), 3)
    print("Reverse first 3 (Stack): ", end="")
    Print_Queue(result1)

    print("-" * 50)

    q2 = deque([4, 3, 2, 1])
    print("Original: ", end="")
    Print_Queue(q2)
    result2 = solution.Reverse_First_K_Stack(deque([4, 3, 2, 1]), 4)
    print("Reverse first 4 (Stack): ", end="")
    Print_Queue(result2)

    print("-" * 50)

    q3 = deque([10, 20, 30, 40, 50, 60])
    print("Original: ", end="")
    Print_Queue(q3)
    result3 = solution.Reverse_First_K_Stack(deque([10, 20, 30, 40, 50, 60]), 2)
    print("Reverse first 2 (Stack): ", end="")
    Print_Queue(result3)

    print("-" * 50)

    q4 = deque([1])
    print("Original: ", end="")
    Print_Queue(q4)
    result4 = solution.Reverse_First_K_Stack(deque([1]), 1)
    print("Reverse first 1 (Stack): ", end="")
    Print_Queue(result4)


if __name__ == "__main__":
    Test_Reverse_First_K()
