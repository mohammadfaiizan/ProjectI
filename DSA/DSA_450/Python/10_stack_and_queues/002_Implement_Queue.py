"""
Problem: Implement Queue from Scratch
URL: https://www.geeksforgeeks.org/queue-set-1introduction-and-array-implementation/

Problem Statement:
Implement a queue data structure using array with operations: enqueue, dequeue, front, rear, isEmpty, isFull.

Sample Input/Output:
Input: enqueue(10), enqueue(20), front(), dequeue()
Output: front() returns 10, dequeue() removes 10
"""


class MyQueue_Array:
    def __init__(self, cap=100):
        self.capacity = cap
        self.arr = [0] * self.capacity
        self.frontIndex = 0
        self.rearIndex = -1
        self.queueSize = 0

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsFull():
            print("Queue Overflow")
            return
        self.rearIndex = (self.rearIndex + 1) % self.capacity
        self.arr[self.rearIndex] = x
        self.queueSize += 1

    def Dequeue(self):
        """
        Remove element from queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Queue Underflow")
            return -1
        val = self.arr[self.frontIndex]
        self.frontIndex = (self.frontIndex + 1) % self.capacity
        self.queueSize -= 1
        return val

    def Front(self):
        """
        Get front element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Queue is Empty")
            return -1
        return self.arr[self.frontIndex]

    def Rear(self):
        """
        Get rear element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Queue is Empty")
            return -1
        return self.arr[self.rearIndex]

    def IsEmpty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.queueSize == 0

    def IsFull(self):
        """
        Check if queue is full.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.queueSize == self.capacity

    def Size(self):
        """
        Get queue size.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.queueSize


class QueueNode:
    def __init__(self, val):
        self.data = val
        self.next = None


class MyQueue_LinkedList:
    def __init__(self):
        self.frontPtr = None
        self.rearPtr = None
        self.queueSize = 0

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        newNode = QueueNode(x)
        if self.rearPtr is None:
            self.frontPtr = self.rearPtr = newNode
        else:
            self.rearPtr.next = newNode
            self.rearPtr = newNode
        self.queueSize += 1

    def Dequeue(self):
        """
        Remove element from queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Queue Underflow")
            return -1
        temp = self.frontPtr
        val = temp.data
        self.frontPtr = self.frontPtr.next
        if self.frontPtr is None:
            self.rearPtr = None
        self.queueSize -= 1
        return val

    def Front(self):
        """
        Get front element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Queue is Empty")
            return -1
        return self.frontPtr.data

    def Rear(self):
        """
        Get rear element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Queue is Empty")
            return -1
        return self.rearPtr.data

    def IsEmpty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.frontPtr is None

    def Size(self):
        """
        Get queue size.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.queueSize


class Solution:
    def Test_Array_Queue(self):
        queue = MyQueue_Array(5)
        print("Array Queue Tests:")
        print(f"isEmpty: {queue.IsEmpty()}")
        queue.Enqueue(10)
        queue.Enqueue(20)
        queue.Enqueue(30)
        print(f"Size: {queue.Size()}")
        print(f"Front: {queue.Front()}")
        print(f"Rear: {queue.Rear()}")
        print(f"Dequeue: {queue.Dequeue()}")
        print(f"Front: {queue.Front()}")
        print(f"Size: {queue.Size()}")

    def Test_LinkedList_Queue(self):
        queue = MyQueue_LinkedList()
        print("\nLinked List Queue Tests:")
        print(f"isEmpty: {queue.IsEmpty()}")
        queue.Enqueue(10)
        queue.Enqueue(20)
        queue.Enqueue(30)
        print(f"Size: {queue.Size()}")
        print(f"Front: {queue.Front()}")
        print(f"Rear: {queue.Rear()}")
        print(f"Dequeue: {queue.Dequeue()}")
        print(f"Front: {queue.Front()}")
        print(f"Size: {queue.Size()}")


def Test_Implement_Queue():
    solution = Solution()
    solution.Test_Array_Queue()
    solution.Test_LinkedList_Queue()


if __name__ == "__main__":
    Test_Implement_Queue()
