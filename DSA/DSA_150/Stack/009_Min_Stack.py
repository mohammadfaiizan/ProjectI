"""
Problem: Implementing a Min Stack
URL: https://leetcode.com/problems/min-stack/

Problem Statement:
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
Implement the MinStack class with push(val), pop(), top(), getMin() operations.

Sample Input/Output:
Input: ["MinStack","push","push","push","getMin","pop","top","getMin"]
       [[],[-2],[0],[-3],[],[],[],[]]
Output: [null,null,null,null,-3,null,0,-2]
Explanation: MinStack minStack = new MinStack();
minStack.push(-2); minStack.push(0); minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop(); minStack.top(); // return 0
minStack.getMin(); // return -2
"""

from typing import List

class Min_Stack_Brute_Force:
    """
    Brute Force Approach - Find minimum by scanning entire stack
    Time Complexity: O(n) for getMin, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack = []
    
    def Push(self, val: int) -> None:
        self.stack.append(val)
    
    def Pop(self) -> None:
        if self.stack:
            self.stack.pop()
    
    def Top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def Get_Min(self) -> int:
        return min(self.stack) if self.stack else None

class Min_Stack_Two_Stacks:
    """
    Two Stacks Approach - Separate stack for minimums
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def Push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def Pop(self) -> None:
        if self.stack:
            popped = self.stack.pop()
            if self.min_stack and popped == self.min_stack[-1]:
                self.min_stack.pop()
    
    def Top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def Get_Min(self) -> int:
        return self.min_stack[-1] if self.min_stack else None

class Min_Stack_Single_Stack_Optimal:
    """
    Single Stack Approach - Store pairs (value, current_min)
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack = []
    
    def Push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def Pop(self) -> None:
        if self.stack:
            self.stack.pop()
    
    def Top(self) -> int:
        return self.stack[-1][0] if self.stack else None
    
    def Get_Min(self) -> int:
        return self.stack[-1][1] if self.stack else None

class Min_Stack_Space_Optimized:
    """
    Space Optimized Approach - Store only when new minimum found
    Time Complexity: O(1) for all operations
    Space Complexity: O(n) worst case, better average case
    """
    def __init__(self):
        self.stack = []
        self.min_val = float('inf')
    
    def Push(self, val: int) -> None:
        if val <= self.min_val:
            self.stack.append(self.min_val)
            self.min_val = val
        self.stack.append(val)
    
    def Pop(self) -> None:
        if self.stack:
            popped = self.stack.pop()
            if popped == self.min_val:
                self.min_val = self.stack.pop()
    
    def Top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def Get_Min(self) -> int:
        return self.min_val if self.min_val != float('inf') else None

class Min_Stack_Difference_Encoding:
    """
    Difference Encoding Approach - Store differences from minimum
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack = []
        self.min_val = None
    
    def Push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.min_val = val
        else:
            self.stack.append(val - self.min_val)
            if val < self.min_val:
                self.min_val = val
    
    def Pop(self) -> None:
        if self.stack:
            top = self.stack.pop()
            if top < 0:
                self.min_val = self.min_val - top
    
    def Top(self) -> int:
        if not self.stack:
            return None
        top = self.stack[-1]
        return self.min_val if top < 0 else self.min_val + top
    
    def Get_Min(self) -> int:
        return self.min_val

class Min_Stack_Linked_List:
    """
    Linked List Approach - Each node stores value and current minimum
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    class Node:
        def __init__(self, val: int, min_val: int, next_node=None):
            self.val = val
            self.min_val = min_val
            self.next = next_node
    
    def __init__(self):
        self.head = None
    
    def Push(self, val: int) -> None:
        if not self.head:
            self.head = self.Node(val, val)
        else:
            min_val = min(val, self.head.min_val)
            self.head = self.Node(val, min_val, self.head)
    
    def Pop(self) -> None:
        if self.head:
            self.head = self.head.next
    
    def Top(self) -> int:
        return self.head.val if self.head else None
    
    def Get_Min(self) -> int:
        return self.head.min_val if self.head else None

def Test_Min_Stack():
    implementations = [
        ("Brute Force", Min_Stack_Brute_Force),
        ("Two Stacks", Min_Stack_Two_Stacks),
        ("Single Stack Optimal", Min_Stack_Single_Stack_Optimal),
        ("Space Optimized", Min_Stack_Space_Optimized),
        ("Difference Encoding", Min_Stack_Difference_Encoding),
        ("Linked List", Min_Stack_Linked_List)
    ]
    
    operations = [
        ("push", -2),
        ("push", 0),
        ("push", -3),
        ("getMin", None),
        ("pop", None),
        ("top", None),
        ("getMin", None)
    ]
    
    expected_results = [None, None, None, -3, None, 0, -2]
    
    for name, StackClass in implementations:
        print(f"Testing {name}:")
        stack = StackClass()
        results = []
        
        for i, (op, val) in enumerate(operations):
            if op == "push":
                result = stack.Push(val)
                results.append(result)
            elif op == "pop":
                result = stack.Pop()
                results.append(result)
            elif op == "top":
                result = stack.Top()
                results.append(result)
            elif op == "getMin":
                result = stack.Get_Min()
                results.append(result)
        
        print(f"Results: {results}")
        print(f"Expected: {expected_results}")
        print(f"Match: {results == expected_results}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Stack()
