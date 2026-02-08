"""
Problem: Find Middle Element of a Stack
URL: https://www.geeksforgeeks.org/design-a-stack-with-find-middle-operation/

Problem Statement:
Design a stack that supports findMiddle and deleteMiddle operations in O(1) time using a doubly linked list with a mid pointer.

Sample Input/Output:
Input: push(10), push(20), push(30), findMiddle(), deleteMiddle()
Output: findMiddle() returns 20, deleteMiddle() removes 20
"""


class DLLNode:
    def __init__(self, val):
        self.data = val
        self.prev = None
        self.next = None


class MiddleStack:
    def __init__(self):
        self.head = None
        self.mid = None
        self.count = 0

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        newNode = DLLNode(x)
        newNode.next = self.head
        
        if self.head is not None:
            self.head.prev = newNode
        
        self.head = newNode
        self.count += 1
        
        if self.count == 1:
            self.mid = newNode
        elif self.count % 2 == 0:
            self.mid = self.mid.prev

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.head is None:
            print("Stack Underflow")
            return -1
        
        temp = self.head
        val = temp.data
        self.head = self.head.next
        
        if self.head is not None:
            self.head.prev = None
        
        self.count -= 1
        
        if self.count == 0:
            self.mid = None
        elif self.count % 2 == 1:
            self.mid = self.mid.next
        
        return val

    def FindMiddle(self):
        """
        Find middle element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.mid is None:
            print("Stack is Empty")
            return -1
        return self.mid.data

    def DeleteMiddle(self):
        """
        Delete middle element.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.mid is None:
            print("Stack is Empty")
            return -1
        
        temp = self.mid
        val = temp.data
        
        if temp.prev is not None:
            temp.prev.next = temp.next
        if temp.next is not None:
            temp.next.prev = temp.prev
        
        if self.head == self.mid:
            self.head = self.mid.next
        
        self.count -= 1
        
        if self.count == 0:
            self.mid = None
            self.head = None
        elif self.count % 2 == 0:
            self.mid = self.mid.prev
        else:
            self.mid = self.mid.next
        
        return val

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.head is None:
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


class Solution:
    def Test_Middle_Stack(self):
        ms = MiddleStack()
        print("Middle Stack Tests:")
        
        ms.Push(10)
        ms.Push(20)
        ms.Push(30)
        ms.Push(40)
        ms.Push(50)
        
        print(f"Top: {ms.Top()}")
        print(f"Middle: {ms.FindMiddle()}")
        
        print(f"Delete Middle: {ms.DeleteMiddle()}")
        print(f"Top: {ms.Top()}")
        print(f"Middle: {ms.FindMiddle()}")
        
        print(f"Pop: {ms.Pop()}")
        print(f"Middle: {ms.FindMiddle()}")
        
        print(f"isEmpty: {ms.IsEmpty()}")


def Test_Middle_Element_Stack():
    solution = Solution()
    solution.Test_Middle_Stack()


if __name__ == "__main__":
    Test_Middle_Element_Stack()
