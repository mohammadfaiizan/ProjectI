"""
Problem: Implement Queue using Stacks
URL: https://practice.geeksforgeeks.org/problems/queue-using-two-stacks/1

Problem Statement:
Implement a queue using two stacks.

Sample Input/Output:
Input: enqueue(1), enqueue(2), enqueue(3), dequeue()
Output: 1
"""


class Queue_Using_Stacks_Costly_Enqueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        while self.s1:
            self.s2.append(self.s1.pop())
        self.s1.append(x)
        while self.s2:
            self.s1.append(self.s2.pop())

    def Dequeue(self):
        """
        Remove element from queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.s1:
            return -1
        return self.s1.pop()

    def Front(self):
        """
        Get front element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return -1 if not self.s1 else self.s1[-1]

    def Is_Empty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.s1) == 0


class Queue_Using_Stacks_Costly_Dequeue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.s1.append(x)

    def Dequeue(self):
        """
        Remove element from queue.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not self.s1 and not self.s2:
            return -1
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2.pop()

    def Front(self):
        """
        Get front element without removing.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return -1 if not self.s2 else self.s2[-1]

    def Is_Empty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.s1) == 0 and len(self.s2) == 0


class Queue_Using_Stacks_Recursion:
    def __init__(self):
        self.s = []

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.s.append(x)

    def Dequeue(self):
        """
        Remove element from queue using recursion.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not self.s:
            return -1
        x = self.s.pop()
        if not self.s:
            return x
        item = self.Dequeue()
        self.s.append(x)
        return item

    def Front(self):
        """
        Get front element without removing using recursion.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not self.s:
            return -1
        x = self.s.pop()
        if not self.s:
            self.s.append(x)
            return x
        item = self.Front()
        self.s.append(x)
        return item

    def Is_Empty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.s) == 0


class Solution:
    def Test_Queue_Using_Stacks(self):
        q1 = Queue_Using_Stacks_Costly_Enqueue()
        q1.Enqueue(1)
        q1.Enqueue(2)
        q1.Enqueue(3)
        print(f"Costly Enqueue - Dequeue: {q1.Dequeue()}")
        print(f"Costly Enqueue - Dequeue: {q1.Dequeue()}")

        q2 = Queue_Using_Stacks_Costly_Dequeue()
        q2.Enqueue(1)
        q2.Enqueue(2)
        q2.Enqueue(3)
        print(f"Costly Dequeue - Dequeue: {q2.Dequeue()}")
        print(f"Costly Dequeue - Dequeue: {q2.Dequeue()}")

        q3 = Queue_Using_Stacks_Recursion()
        q3.Enqueue(1)
        q3.Enqueue(2)
        q3.Enqueue(3)
        print(f"Recursion - Dequeue: {q3.Dequeue()}")
        print(f"Recursion - Dequeue: {q3.Dequeue()}")


def Test_Queue_Using_Stacks():
    solution = Solution()
    solution.Test_Queue_Using_Stacks()


if __name__ == "__main__":
    Test_Queue_Using_Stacks()
