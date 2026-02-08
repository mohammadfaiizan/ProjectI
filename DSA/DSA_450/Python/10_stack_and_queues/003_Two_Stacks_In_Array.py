"""
Problem: Implement Two Stacks in an Array
URL: https://practice.geeksforgeeks.org/problems/implement-two-stacks-in-an-array/1

Problem Statement:
Use a single array to implement two stacks efficiently. One stack grows from left to right, the other from right to left.

Sample Input/Output:
Input: push1(10), push2(20), push1(30), pop1(), pop2()
Output: pop1() returns 30, pop2() returns 20
"""


class TwoStacks:
    def __init__(self, cap=100):
        self.capacity = cap
        self.arr = [0] * self.capacity
        self.top1 = -1
        self.top2 = cap

    def Push1(self, x):
        """
        Push element to stack 1.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.top1 >= self.top2 - 1:
            print("Stack Overflow")
            return
        self.top1 += 1
        self.arr[self.top1] = x

    def Push2(self, x):
        """
        Push element to stack 2.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.top1 >= self.top2 - 1:
            print("Stack Overflow")
            return
        self.top2 -= 1
        self.arr[self.top2] = x

    def Pop1(self):
        """
        Pop element from stack 1.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.top1 < 0:
            print("Stack1 Underflow")
            return -1
        val = self.arr[self.top1]
        self.top1 -= 1
        return val

    def Pop2(self):
        """
        Pop element from stack 2.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.top2 >= self.capacity:
            print("Stack2 Underflow")
            return -1
        val = self.arr[self.top2]
        self.top2 += 1
        return val

    def Peek1(self):
        """
        Get top element of stack 1 without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.top1 < 0:
            print("Stack1 is Empty")
            return -1
        return self.arr[self.top1]

    def Peek2(self):
        """
        Get top element of stack 2 without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.top2 >= self.capacity:
            print("Stack2 is Empty")
            return -1
        return self.arr[self.top2]

    def IsEmpty1(self):
        """
        Check if stack 1 is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.top1 < 0

    def IsEmpty2(self):
        """
        Check if stack 2 is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.top2 >= self.capacity


class Solution:
    def Test_Two_Stacks(self):
        ts = TwoStacks(10)
        print("Two Stacks in Array Tests:")
        
        ts.Push1(10)
        ts.Push1(20)
        ts.Push1(30)
        ts.Push2(100)
        ts.Push2(200)
        ts.Push2(300)
        
        print(f"Stack1 top: {ts.Peek1()}")
        print(f"Stack2 top: {ts.Peek2()}")
        
        print(f"Pop1: {ts.Pop1()}")
        print(f"Pop2: {ts.Pop2()}")
        
        print(f"Stack1 top: {ts.Peek1()}")
        print(f"Stack2 top: {ts.Peek2()}")
        
        print(f"isEmpty1: {ts.IsEmpty1()}")
        print(f"isEmpty2: {ts.IsEmpty2()}")


def Test_Two_Stacks_In_Array():
    solution = Solution()
    solution.Test_Two_Stacks()


if __name__ == "__main__":
    Test_Two_Stacks_In_Array()
