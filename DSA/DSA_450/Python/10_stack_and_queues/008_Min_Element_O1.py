"""
Problem: Get Minimum Element from Stack in O(1) Time and O(1) Space
URL: https://practice.geeksforgeeks.org/problems/special-stack/1

Problem Statement:
Design a special stack with getMin in O(1) time and O(1) extra space using the 2*val - min_ele encoding trick.

Sample Input/Output:
Input: push(10), push(20), push(5), getMin(), pop(), getMin()
Output: getMin() returns 5, after pop() getMin() returns 10
"""


class MinStack_O1Space:
    def __init__(self):
        self.st = []
        self.minEle = None

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.st:
            self.st.append(x)
            self.minEle = x
        else:
            if x >= self.minEle:
                self.st.append(x)
            else:
                self.st.append(2 * x - self.minEle)
                self.minEle = x

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.st:
            print("Stack Underflow")
            return -1
        top = self.st.pop()
        if top < self.minEle:
            actualTop = self.minEle
            self.minEle = 2 * self.minEle - top
            return actualTop
        return top

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.st:
            print("Stack is Empty")
            return -1
        top = self.st[-1]
        if top < self.minEle:
            return self.minEle
        return top

    def GetMin(self):
        """
        Get minimum element.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.st:
            print("Stack is Empty")
            return -1
        return self.minEle

    def IsEmpty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.st) == 0


class MinStack_Auxiliary:
    def __init__(self):
        self.st = []
        self.minSt = []

    def Push(self, x):
        """
        Push element onto stack.
        Time Complexity: O(1)
        Space Complexity: O(n)
        """
        self.st.append(x)
        if not self.minSt or x <= self.minSt[-1]:
            self.minSt.append(x)

    def Pop(self):
        """
        Pop element from stack.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.st:
            print("Stack Underflow")
            return -1
        top = self.st.pop()
        if top == self.minSt[-1]:
            self.minSt.pop()
        return top

    def Top(self):
        """
        Get top element without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.st:
            print("Stack is Empty")
            return -1
        return self.st[-1]

    def GetMin(self):
        """
        Get minimum element.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.minSt:
            print("Stack is Empty")
            return -1
        return self.minSt[-1]

    def IsEmpty(self):
        """
        Check if stack is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.st) == 0


class Solution:
    def Test_O1Space(self):
        ms = MinStack_O1Space()
        print("O(1) Space MinStack Tests:")
        
        ms.Push(10)
        ms.Push(20)
        ms.Push(5)
        ms.Push(15)
        
        print(f"Top: {ms.Top()}")
        print(f"Min: {ms.GetMin()}")
        
        print(f"Pop: {ms.Pop()}")
        print(f"Min: {ms.GetMin()}")
        
        print(f"Pop: {ms.Pop()}")
        print(f"Min: {ms.GetMin()}")

    def Test_Auxiliary(self):
        ms = MinStack_Auxiliary()
        print("\nAuxiliary Stack MinStack Tests:")
        
        ms.Push(10)
        ms.Push(20)
        ms.Push(5)
        ms.Push(15)
        
        print(f"Top: {ms.Top()}")
        print(f"Min: {ms.GetMin()}")
        
        print(f"Pop: {ms.Pop()}")
        print(f"Min: {ms.GetMin()}")
        
        print(f"Pop: {ms.Pop()}")
        print(f"Min: {ms.GetMin()}")


def Test_Min_Element_O1():
    solution = Solution()
    solution.Test_O1Space()
    solution.Test_Auxiliary()


if __name__ == "__main__":
    Test_Min_Element_O1()
