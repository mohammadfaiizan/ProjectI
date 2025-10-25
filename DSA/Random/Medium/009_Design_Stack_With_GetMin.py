"""
Problem: Design a Stack that Supports getMin() in O(1) Time and O(1) Extra Space
URL: https://leetcode.com/problems/min-stack/

Problem Statement:
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.

Sample Input/Output:
Input: ["MinStack","push","push","push","getMin","pop","top","getMin"]
       [[],[-2],[0],[-3],[],[],[],[]]
Output: [null,null,null,null,-3,null,0,-2]
"""

from typing import List, Optional

class Min_Stack_Two_Stacks:
    """
    Two Stacks Approach
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
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            self.stack.pop()
    
    def Top(self) -> int:
        return self.stack[-1] if self.stack else -1
    
    def Get_Min(self) -> int:
        return self.min_stack[-1] if self.min_stack else -1

class Min_Stack_Single_Stack_Tuple:
    """
    Single Stack with Tuples
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
        return self.stack[-1][0] if self.stack else -1
    
    def Get_Min(self) -> int:
        return self.stack[-1][1] if self.stack else -1

class Min_Stack_Encoding:
    """
    Encoding Approach - O(1) extra space
    Time Complexity: O(1) for all operations
    Space Complexity: O(1) extra space
    """
    def __init__(self):
        self.stack = []
        self.min_val = float('inf')
    
    def Push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(val)
            self.min_val = val
        else:
            if val < self.min_val:
                self.stack.append(2 * val - self.min_val)
                self.min_val = val
            else:
                self.stack.append(val)
    
    def Pop(self) -> None:
        if not self.stack:
            return
        
        top = self.stack.pop()
        if top < self.min_val:
            self.min_val = 2 * self.min_val - top
    
    def Top(self) -> int:
        if not self.stack:
            return -1
        
        top = self.stack[-1]
        return self.min_val if top < self.min_val else top
    
    def Get_Min(self) -> int:
        return self.min_val if self.stack else -1

class Min_Stack_Difference:
    """
    Difference Approach
    Time Complexity: O(1) for all operations
    Space Complexity: O(1) extra space
    """
    def __init__(self):
        self.stack = []
        self.minimum = None
    
    def Push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.minimum = val
        else:
            self.stack.append(val - self.minimum)
            if val < self.minimum:
                self.minimum = val
    
    def Pop(self) -> None:
        if not self.stack:
            return
        
        diff = self.stack.pop()
        if diff < 0:
            self.minimum = self.minimum - diff
    
    def Top(self) -> int:
        if not self.stack:
            return -1
        
        diff = self.stack[-1]
        return self.minimum if diff < 0 else self.minimum + diff
    
    def Get_Min(self) -> int:
        return self.minimum if self.stack else -1

class Node:
    def __init__(self, val: int, min_val: int):
        self.val = val
        self.min_val = min_val
        self.next = None

class Min_Stack_Linked_List:
    """
    Linked List Approach
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.head = None
    
    def Push(self, val: int) -> None:
        if not self.head:
            self.head = Node(val, val)
        else:
            new_node = Node(val, min(val, self.head.min_val))
            new_node.next = self.head
            self.head = new_node
    
    def Pop(self) -> None:
        if self.head:
            self.head = self.head.next
    
    def Top(self) -> int:
        return self.head.val if self.head else -1
    
    def Get_Min(self) -> int:
        return self.head.min_val if self.head else -1

def Test_Min_Stack():
    for approach_name, StackClass in [
        ("Two Stacks", Min_Stack_Two_Stacks),
        ("Single Stack Tuple", Min_Stack_Single_Stack_Tuple),
        ("Encoding", Min_Stack_Encoding),
        ("Difference", Min_Stack_Difference),
        ("Linked List", Min_Stack_Linked_List)
    ]:
        print(f"Testing {approach_name} Approach:")
        stack = StackClass()
        
        stack.Push(-2)
        stack.Push(0)
        stack.Push(-3)
        print(f"After pushing -2, 0, -3: getMin() = {stack.Get_Min()}")
        
        stack.Pop()
        print(f"After pop: top() = {stack.Top()}, getMin() = {stack.Get_Min()}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Stack()

