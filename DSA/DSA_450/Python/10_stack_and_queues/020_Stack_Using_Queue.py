"""
Problem: Implement Stack using Queues
URL: https://practice.geeksforgeeks.org/problems/stack-using-two-queues/1

Problem Statement:
Implement stack using two queues.

Sample Input/Output:
Input: push 1,2,3; pop -> 3,2,1
Output: Stack operations work correctly
"""

from collections import deque


class Stack_Using_Queue_Costly_Pop:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.q1.append(x)

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not self.q1:
            return -1
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        top = self.q1.popleft()
        self.q1, self.q2 = self.q2, self.q1
        return top

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not self.q1:
            return -1
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        top = self.q1[0]
        self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
        return top

    def Empty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.q1) == 0


class Stack_Using_Queue_Costly_Push:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.q1:
            return -1
        return self.q1.popleft()

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.q1:
            return -1
        return self.q1[0]

    def Empty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.q1) == 0


class Solution:
    def Test_Costly_Pop(self):
        st = Stack_Using_Queue_Costly_Pop()
        st.Push(1)
        st.Push(2)
        st.Push(3)
        print(f"Pop: {st.Pop()}")
        print(f"Pop: {st.Pop()}")
        print(f"Top: {st.Top()}")
        print(f"Pop: {st.Pop()}")
        print(f"Empty: {'true' if st.Empty() else 'false'}")

    def Test_Costly_Push(self):
        st = Stack_Using_Queue_Costly_Push()
        st.Push(1)
        st.Push(2)
        st.Push(3)
        print(f"Pop: {st.Pop()}")
        print(f"Pop: {st.Pop()}")
        print(f"Top: {st.Top()}")
        print(f"Pop: {st.Pop()}")
        print(f"Empty: {'true' if st.Empty() else 'false'}")


def Test_Stack_Using_Queue():
    solution = Solution()
    
    print("=== Costly Pop Approach ===")
    solution.Test_Costly_Pop()
    
    print("\n=== Costly Push Approach ===")
    solution.Test_Costly_Push()


if __name__ == "__main__":
    Test_Stack_Using_Queue()
