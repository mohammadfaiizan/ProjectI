"""
Problem: Implement Circular Queue
URL: https://www.geeksforgeeks.org/circular-queue-set-1-introduction-array-implementation/

Problem Statement:
Implement circular queue with enqueue, dequeue, display using array with front/rear wrapping.

Sample Input/Output:
Input: enqueue(1), enqueue(2), enqueue(3), dequeue()
Output: 1
"""


class CircularQueue:
    def __init__(self, cap):
        self.capacity = cap
        self.arr = [0] * self.capacity
        self.front = -1
        self.rear = -1
        self.size = 0

    def Is_Full(self):
        """
        Check if queue is full.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.size == self.capacity

    def Is_Empty(self):
        """
        Check if queue is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.size == 0

    def Enqueue(self, x):
        """
        Add element to queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.Is_Full():
            print("Queue is full")
            return
        if self.Is_Empty():
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.capacity
        self.arr[self.rear] = x
        self.size += 1

    def Dequeue(self):
        """
        Remove element from queue.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.Is_Empty():
            print("Queue is empty")
            return -1
        val = self.arr[self.front]
        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return val

    def Front_Element(self):
        """
        Get front element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.Is_Empty():
            return -1
        return self.arr[self.front]

    def Rear_Element(self):
        """
        Get rear element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.Is_Empty():
            return -1
        return self.arr[self.rear]

    def Display(self):
        """
        Display queue elements.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if self.Is_Empty():
            print("Queue is empty")
            return
        i = self.front
        print("Queue: ", end="")
        while True:
            print(self.arr[i], end=" ")
            if i == self.rear:
                break
            i = (i + 1) % self.capacity
        print()


class Solution:
    def Test_Circular_Queue(self):
        cq = CircularQueue(5)
        
        cq.Enqueue(1)
        cq.Enqueue(2)
        cq.Enqueue(3)
        cq.Display()
        
        print(f"Dequeue: {cq.Dequeue()}")
        print(f"Front: {cq.Front_Element()}")
        print(f"Rear: {cq.Rear_Element()}")
        
        cq.Enqueue(4)
        cq.Enqueue(5)
        cq.Enqueue(6)
        cq.Display()
        
        print(f"Dequeue: {cq.Dequeue()}")
        print(f"Dequeue: {cq.Dequeue()}")
        cq.Display()


def Test_Circular_Queue():
    solution = Solution()
    solution.Test_Circular_Queue()


if __name__ == "__main__":
    Test_Circular_Queue()
