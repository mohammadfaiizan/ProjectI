"""
Problem: Interleave First Half of Queue with Second Half
URL: https://www.geeksforgeeks.org/interleave-first-half-queue-second-half/

Problem Statement:
Given a queue of even size, interleave first half with second half.

Sample Input/Output:
Input: [11,12,13,14,15,16,17,18,19,20]
Output: [11,16,12,17,13,18,14,19,15,20]
"""

from collections import deque


class Solution:
    def Interleave_Queue_Stack(self, q):
        """
        Interleave queue using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(q)
        half = n // 2
        st = []
        
        for i in range(half):
            st.append(q.popleft())
        
        while st:
            q.append(st.pop())
        
        for i in range(half):
            q.append(q.popleft())
        
        for i in range(half):
            st.append(q.popleft())
        
        while st:
            q.append(st.pop())
            q.append(q.popleft())
        
        return q

    def Interleave_Queue_Auxiliary(self, q):
        """
        Interleave queue using auxiliary queue.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(q)
        half = n // 2
        aux = deque()
        
        for i in range(half):
            aux.append(q.popleft())
        
        while aux:
            q.append(aux.popleft())
            q.append(q.popleft())
        
        return q


def Test_Interleave_Queue_Stack():
    solution = Solution()
    
    q1 = deque([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    result1 = solution.Interleave_Queue_Stack(q1)
    print("Stack - Interleaved Queue: ", end="")
    while result1:
        print(result1.popleft(), end=" ")
    print()


def Test_Interleave_Queue_Auxiliary():
    solution = Solution()
    
    q2 = deque([1, 2, 3, 4, 5, 6])
    result2 = solution.Interleave_Queue_Auxiliary(q2)
    print("Auxiliary - Interleaved Queue: ", end="")
    while result2:
        print(result2.popleft(), end=" ")
    print()


if __name__ == "__main__":
    Test_Interleave_Queue_Stack()
    Test_Interleave_Queue_Auxiliary()
