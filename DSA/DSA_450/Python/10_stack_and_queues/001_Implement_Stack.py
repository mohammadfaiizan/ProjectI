"""
Problem: Implement Stack from Scratch
URL: https://www.geeksforgeeks.org/stack-data-structure-introduction-program/

Problem Statement:
Implement a stack data structure using array with operations: push, pop, top, isEmpty, size, isFull.

Sample Input/Output:
Input: push(10), push(20), top(), pop()
Output: top() returns 20, pop() removes 20
"""


class MyStack_Array:
    def __init__(self, cap=100):
        self.capacity = cap
        self.arr = [0] * self.capacity
        self.topIndex = -1

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsFull():
            print("Stack Overflow")
            return
        self.topIndex += 1
        self.arr[self.topIndex] = x

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Stack Underflow")
            return -1
        val = self.arr[self.topIndex]
        self.topIndex -= 1
        return val

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Stack is Empty")
            return -1
        return self.arr[self.topIndex]

    def IsEmpty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.topIndex == -1

    def IsFull(self):
        """
        Check if stack is full.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.topIndex == self.capacity - 1

    def Size(self):
        """
        Get stack size.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.topIndex + 1


class Node:
    def __init__(self, val):
        self.data = val
        self.next = None


class MyStack_LinkedList:
    def __init__(self):
        self.head = None
        self.stackSize = 0

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        newNode = Node(x)
        newNode.next = self.head
        self.head = newNode
        self.stackSize += 1

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Stack Underflow")
            return -1
        temp = self.head
        val = temp.data
        self.head = self.head.next
        self.stackSize -= 1
        return val

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty():
            print("Stack is Empty")
            return -1
        return self.head.data

    def IsEmpty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.head is None

    def Size(self):
        """
        Get stack size.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.stackSize


class Solution:
    def Test_Array_Stack(self):
        stack = MyStack_Array(5)
        print("Array Stack Tests:")
        print(f"isEmpty: {stack.IsEmpty()}")
        stack.Push(10)
        stack.Push(20)
        stack.Push(30)
        print(f"Size: {stack.Size()}")
        print(f"Top: {stack.Top()}")
        print(f"Pop: {stack.Pop()}")
        print(f"Top: {stack.Top()}")
        print(f"Size: {stack.Size()}")

    def Test_LinkedList_Stack(self):
        stack = MyStack_LinkedList()
        print("\nLinked List Stack Tests:")
        print(f"isEmpty: {stack.IsEmpty()}")
        stack.Push(10)
        stack.Push(20)
        stack.Push(30)
        print(f"Size: {stack.Size()}")
        print(f"Top: {stack.Top()}")
        print(f"Pop: {stack.Pop()}")
        print(f"Top: {stack.Top()}")
        print(f"Size: {stack.Size()}")


def Test_Implement_Stack():
    solution = Solution()
    solution.Test_Array_Stack()
    solution.Test_LinkedList_Stack()


if __name__ == "__main__":
    Test_Implement_Stack()
