"""
Problem: Implement N Queues in a Single Array
URL: https://www.geeksforgeeks.org/efficiently-implement-k-queues-single-array/

Problem Statement:
Efficiently implement k queues in a single array using front[], rear[], next[] arrays and free list.

Sample Input/Output:
Input: enqueue(1, 0), enqueue(2, 0), enqueue(3, 1), dequeue(0)
Output: 1
"""


class KQueues:
    def __init__(self, k, n):
        self.n = n
        self.k = k
        self.arr = [0] * self.n
        self.front = [-1] * self.k
        self.rear = [-1] * self.k
        self.next = [0] * self.n
        
        self.free = 0
        for i in range(self.n - 1):
            self.next[i] = i + 1
        self.next[self.n - 1] = -1

    def Is_Full(self):
        """
        Check if all queues are full.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.free == -1

    def Is_Empty(self, qn):
        """
        Check if queue qn is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.front[qn] == -1

    def Enqueue(self, item, qn):
        """
        Add element to queue qn.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.Is_Full():
            print("Queue Overflow")
            return

        i = self.free
        self.free = self.next[i]

        if self.Is_Empty(qn):
            self.front[qn] = i
        else:
            self.next[self.rear[qn]] = i

        self.next[i] = -1
        self.rear[qn] = i
        self.arr[i] = item

    def Dequeue(self, qn):
        """
        Remove element from queue qn.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.Is_Empty(qn):
            print("Queue Underflow")
            return -1

        i = self.front[qn]
        self.front[qn] = self.next[i]
        self.next[i] = self.free
        self.free = i
        return self.arr[i]


class Solution:
    def Test_N_Queues_In_Array(self):
        kq = KQueues(3, 10)
        
        kq.Enqueue(15, 2)
        kq.Enqueue(45, 2)
        kq.Enqueue(17, 1)
        kq.Enqueue(49, 1)
        kq.Enqueue(39, 1)
        kq.Enqueue(11, 0)
        kq.Enqueue(9, 0)
        kq.Enqueue(7, 0)

        print(f"Dequeued from queue 2: {kq.Dequeue(2)}")
        print(f"Dequeued from queue 1: {kq.Dequeue(1)}")
        print(f"Dequeued from queue 0: {kq.Dequeue(0)}")


def Test_N_Queues_In_Array():
    solution = Solution()
    solution.Test_N_Queues_In_Array()


if __name__ == "__main__":
    Test_N_Queues_In_Array()
