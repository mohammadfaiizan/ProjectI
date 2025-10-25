"""
Problem: Queue Using Stack
URL: https://leetcode.com/problems/implement-queue-using-stacks/

Problem Statement:
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue 
should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:
- void push(int x) Pushes element x to the back of the queue.
- int pop() Removes the element from the front of the queue and returns it.
- int peek() Returns the element at the front of the queue.
- boolean empty() Returns true if the queue is empty, false otherwise.

Sample Input/Output:
Input: ["MyQueue", "push", "push", "peek", "pop", "empty"]
       [[], [1], [2], [], [], []]
Output: [null, null, null, 1, 1, false]
"""

from typing import List

class Queue_Using_Two_Stacks:
    """
    Two Stacks Approach
    Time Complexity: O(1) amortized for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack_in = []
        self.stack_out = []
    
    def Push(self, x: int) -> None:
        self.stack_in.append(x)
    
    def Pop(self) -> int:
        self.Peek()
        return self.stack_out.pop() if self.stack_out else -1
    
    def Peek(self) -> int:
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        
        return self.stack_out[-1] if self.stack_out else -1
    
    def Empty(self) -> bool:
        return not self.stack_in and not self.stack_out

class Queue_Using_Two_Stacks_Push_Expensive:
    """
    Two Stacks with Push Expensive
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    
    def Push(self, x: int) -> None:
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        
        self.stack1.append(x)
        
        while self.stack2:
            self.stack1.append(self.stack2.pop())
    
    def Pop(self) -> int:
        return self.stack1.pop() if self.stack1 else -1
    
    def Peek(self) -> int:
        return self.stack1[-1] if self.stack1 else -1
    
    def Empty(self) -> bool:
        return len(self.stack1) == 0

class Queue_Using_One_Stack_Recursive:
    """
    One Stack with Recursion
    Time Complexity: O(n) for pop/peek
    Space Complexity: O(n)
    """
    def __init__(self):
        self.stack = []
    
    def Push(self, x: int) -> None:
        self.stack.append(x)
    
    def Pop(self) -> int:
        if len(self.stack) == 1:
            return self.stack.pop()
        
        temp = self.stack.pop()
        result = self.Pop()
        self.stack.append(temp)
        
        return result
    
    def Peek(self) -> int:
        if len(self.stack) == 1:
            return self.stack[-1]
        
        temp = self.stack.pop()
        result = self.Peek()
        self.stack.append(temp)
        
        return result
    
    def Empty(self) -> bool:
        return len(self.stack) == 0

class Queue_Using_Stack_List:
    """
    Using Lists as Stacks
    Time Complexity: O(1) amortized
    Space Complexity: O(n)
    """
    def __init__(self):
        self.input_stack = []
        self.output_stack = []
    
    def Push(self, x: int) -> None:
        self.input_stack.append(x)
    
    def Transfer(self) -> None:
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
    
    def Pop(self) -> int:
        self.Transfer()
        return self.output_stack.pop() if self.output_stack else -1
    
    def Peek(self) -> int:
        self.Transfer()
        return self.output_stack[-1] if self.output_stack else -1
    
    def Empty(self) -> bool:
        return not self.input_stack and not self.output_stack

class Queue_Using_Stack_Front_Tracking:
    """
    Stack with Front Element Tracking
    Time Complexity: O(1) amortized
    Space Complexity: O(n)
    """
    def __init__(self):
        self.s1 = []
        self.s2 = []
        self.front = None
    
    def Push(self, x: int) -> None:
        if not self.s1:
            self.front = x
        self.s1.append(x)
    
    def Pop(self) -> int:
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        
        val = self.s2.pop() if self.s2 else -1
        
        return val
    
    def Peek(self) -> int:
        if self.s2:
            return self.s2[-1]
        return self.front if self.s1 else -1
    
    def Empty(self) -> bool:
        return not self.s1 and not self.s2

def Test_Queue_Using_Stack():
    for approach_name, QueueClass in [
        ("Two Stacks", Queue_Using_Two_Stacks),
        ("Two Stacks Push Expensive", Queue_Using_Two_Stacks_Push_Expensive),
        ("One Stack Recursive", Queue_Using_One_Stack_Recursive),
        ("Stack List", Queue_Using_Stack_List),
        ("Front Tracking", Queue_Using_Stack_Front_Tracking)
    ]:
        print(f"Testing {approach_name} Approach:")
        queue = QueueClass()
        
        queue.Push(1)
        queue.Push(2)
        print(f"After pushing 1, 2: peek() = {queue.Peek()}")
        print(f"pop() = {queue.Pop()}")
        print(f"empty() = {queue.Empty()}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Queue_Using_Stack()

