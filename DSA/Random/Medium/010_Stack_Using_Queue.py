"""
Problem: Stack Using Queue
URL: https://leetcode.com/problems/implement-stack-using-queues/

Problem Statement:
Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack 
should support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:
- void push(int x) Pushes element x to the top of the stack.
- int pop() Removes the element on the top of the stack and returns it.
- int top() Returns the element on the top of the stack.
- boolean empty() Returns true if the stack is empty, false otherwise.

Sample Input/Output:
Input: ["MyStack", "push", "push", "top", "pop", "empty"]
       [[], [1], [2], [], [], []]
Output: [null, null, null, 2, 2, false]
"""

from typing import List
from collections import deque

class Stack_Using_Two_Queues:
    """
    Two Queues Approach
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def Push(self, x: int) -> None:
        self.q2.append(x)
        
        while self.q1:
            self.q2.append(self.q1.popleft())
        
        self.q1, self.q2 = self.q2, self.q1
    
    def Pop(self) -> int:
        return self.q1.popleft() if self.q1 else -1
    
    def Top(self) -> int:
        return self.q1[0] if self.q1 else -1
    
    def Empty(self) -> bool:
        return len(self.q1) == 0

class Stack_Using_One_Queue:
    """
    One Queue Approach - Optimal
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.q = deque()
    
    def Push(self, x: int) -> None:
        self.q.append(x)
        
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
    
    def Pop(self) -> int:
        return self.q.popleft() if self.q else -1
    
    def Top(self) -> int:
        return self.q[0] if self.q else -1
    
    def Empty(self) -> bool:
        return len(self.q) == 0

class Stack_Using_Queue_List:
    """
    Using List as Queue
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.queue = []
    
    def Push(self, x: int) -> None:
        self.queue.append(x)
        size = len(self.queue)
        
        while size > 1:
            self.queue.append(self.queue.pop(0))
            size -= 1
    
    def Pop(self) -> int:
        return self.queue.pop(0) if self.queue else -1
    
    def Top(self) -> int:
        return self.queue[0] if self.queue else -1
    
    def Empty(self) -> bool:
        return len(self.queue) == 0

class Stack_Using_Queue_Reverse:
    """
    Queue with Reverse on Pop
    Time Complexity: O(1) for push, O(n) for pop/top
    Space Complexity: O(n)
    """
    def __init__(self):
        self.q = deque()
    
    def Push(self, x: int) -> None:
        self.q.append(x)
    
    def Pop(self) -> int:
        if not self.q:
            return -1
        
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
        
        return self.q.popleft()
    
    def Top(self) -> int:
        if not self.q:
            return -1
        
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
        
        top_val = self.q[0]
        self.q.append(self.q.popleft())
        
        return top_val
    
    def Empty(self) -> bool:
        return len(self.q) == 0

class Stack_Using_Queue_Size:
    """
    Queue with Size Tracking
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.q = deque()
        self.top_element = None
    
    def Push(self, x: int) -> None:
        self.q.append(x)
        self.top_element = x
        
        size = len(self.q)
        for _ in range(size - 1):
            self.q.append(self.q.popleft())
    
    def Pop(self) -> int:
        if not self.q:
            return -1
        
        val = self.q.popleft()
        
        if self.q:
            self.top_element = self.q[0]
        
        return val
    
    def Top(self) -> int:
        return self.top_element if self.q else -1
    
    def Empty(self) -> bool:
        return len(self.q) == 0

def Test_Stack_Using_Queue():
    for approach_name, StackClass in [
        ("Two Queues", Stack_Using_Two_Queues),
        ("One Queue", Stack_Using_One_Queue),
        ("Queue List", Stack_Using_Queue_List),
        ("Queue Reverse", Stack_Using_Queue_Reverse),
        ("Queue Size", Stack_Using_Queue_Size)
    ]:
        print(f"Testing {approach_name} Approach:")
        stack = StackClass()
        
        stack.Push(1)
        stack.Push(2)
        print(f"After pushing 1, 2: top() = {stack.Top()}")
        print(f"pop() = {stack.Pop()}")
        print(f"empty() = {stack.Empty()}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Stack_Using_Queue()

