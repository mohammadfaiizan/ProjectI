"""
Problem: Implement Stack and Queue using Deque
URL: https://www.geeksforgeeks.org/implement-stack-queue-using-deque/

Problem Statement:
Implement both stack (LIFO) and queue (FIFO) using a doubly-linked-list-based deque.
Create a Deque class, then Stack and Queue classes that use it.

Sample Input/Output:
Input: Stack push(1), push(2), push(3), pop()
Output: 3

Input: Queue enqueue(1), enqueue(2), enqueue(3), dequeue()
Output: 1
"""

from collections import deque


class Deque:
    def __init__(self):
        self.dq = deque()

    def Push_Front(self, x):
        """
        Push element to front.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.dq.appendleft(x)

    def Push_Back(self, x):
        """
        Push element to back.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.dq.append(x)

    def Pop_Front(self):
        """
        Pop element from front.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.dq:
            return -1
        return self.dq.popleft()

    def Pop_Back(self):
        """
        Pop element from back.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.dq:
            return -1
        return self.dq.pop()

    def Is_Empty(self):
        """
        Check if deque is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.dq) == 0

    def Front(self):
        """
        Get front element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return -1 if not self.dq else self.dq[0]

    def Back(self):
        """
        Get back element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return -1 if not self.dq else self.dq[-1]


class Stack:
    def __init__(self):
        self.dq = Deque()

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.dq.Push_Back(x)

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.dq.Pop_Back()

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.dq.Back()

    def Is_Empty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.dq.Is_Empty()


class Queue:
    def __init__(self):
        self.dq = Deque()

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.dq.Push_Back(x)

    def Dequeue(self):
        """
        Remove element from queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.dq.Pop_Front()

    def Front(self):
        """
        Get front element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.dq.Front()

    def Is_Empty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.dq.Is_Empty()


class Solution:
    def Test_Stack_Queue_Using_Deque(self):
        stack = Stack()
        stack.Push(1)
        stack.Push(2)
        stack.Push(3)
        print(f"Stack Pop: {stack.Pop()}")
        print(f"Stack Top: {stack.Top()}")
        print(f"Stack Pop: {stack.Pop()}")

        queue = Queue()
        queue.Enqueue(1)
        queue.Enqueue(2)
        queue.Enqueue(3)
        print(f"Queue Dequeue: {queue.Dequeue()}")
        print(f"Queue Front: {queue.Front()}")
        print(f"Queue Dequeue: {queue.Dequeue()}")


def Test_Stack_Queue_Using_Deque():
    solution = Solution()
    solution.Test_Stack_Queue_Using_Deque()


if __name__ == "__main__":
    Test_Stack_Queue_Using_Deque()
