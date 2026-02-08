"""
Problem: Implement N Stacks in a Single Array
URL: https://www.geeksforgeeks.org/efficiently-implement-k-stacks-single-array/

Problem Statement:
Implement N stacks in a single array efficiently using arrays: arr[] for data, top[] for stack tops, next[] for free-list and next-in-stack chain.

Sample Input/Output:
Input: push(0, 10), push(1, 20), push(0, 30), pop(0), pop(1)
Output: pop(0) returns 30, pop(1) returns 20
"""


class NStacks:
    def __init__(self, numStacks, size):
        self.n = size
        self.k = numStacks
        self.arr = [0] * self.n
        self.top = [-1] * self.k
        self.next = [0] * self.n
        
        self.free = 0
        for i in range(self.n - 1):
            self.next[i] = i + 1
        self.next[self.n - 1] = -1

    def IsFull(self):
        """
        Check if all stacks are full.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.free == -1

    def IsEmpty(self, sn):
        """
        Check if stack sn is empty.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.top[sn] == -1

    def Push(self, sn, x):
        """
        Push element x to stack sn.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsFull():
            print("Stack Overflow")
            return
        
        i = self.free
        self.free = self.next[i]
        self.next[i] = self.top[sn]
        self.top[sn] = i
        self.arr[i] = x

    def Pop(self, sn):
        """
        Pop element from stack sn.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty(sn):
            print("Stack Underflow")
            return -1
        
        i = self.top[sn]
        self.top[sn] = self.next[i]
        self.next[i] = self.free
        self.free = i
        return self.arr[i]

    def Peek(self, sn):
        """
        Get top element of stack sn without removing.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.IsEmpty(sn):
            print("Stack is Empty")
            return -1
        return self.arr[self.top[sn]]


class Solution:
    def Test_N_Stacks(self):
        ns = NStacks(3, 10)
        print("N Stacks in Array Tests:")
        
        ns.Push(0, 10)
        ns.Push(0, 20)
        ns.Push(1, 100)
        ns.Push(1, 200)
        ns.Push(2, 1000)
        ns.Push(2, 2000)
        
        print(f"Stack 0 top: {ns.Peek(0)}")
        print(f"Stack 1 top: {ns.Peek(1)}")
        print(f"Stack 2 top: {ns.Peek(2)}")
        
        print(f"Pop Stack 0: {ns.Pop(0)}")
        print(f"Pop Stack 1: {ns.Pop(1)}")
        print(f"Pop Stack 2: {ns.Pop(2)}")
        
        print(f"Stack 0 top: {ns.Peek(0)}")
        print(f"Stack 1 top: {ns.Peek(1)}")
        print(f"Stack 2 top: {ns.Peek(2)}")
        
        print(f"isEmpty Stack 0: {ns.IsEmpty(0)}")


def Test_N_Stacks_In_Array():
    solution = Solution()
    solution.Test_N_Stacks()


if __name__ == "__main__":
    Test_N_Stacks_In_Array()
