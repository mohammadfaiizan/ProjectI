"""
Problem: Implementing a Stack using Queues
URL: https://leetcode.com/problems/implement-stack-using-queues/description/

Problem Statement:
Implement a last-in-first-out (LIFO) stack using only two queues. 
The implemented stack should support push, top, pop, and empty operations.

Sample Input/Output:
Input: ["MyStack", "push", "push", "top", "pop", "empty"]
       [[], [1], [2], [], [], []]
Output: [null, null, null, 2, 2, false]
Explanation: MyStack myStack = new MyStack();
myStack.push(1); myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False
"""

from collections import deque
from typing import List

class My_Stack_Two_Queues_Brute_Force:
    """
    Two Queues Brute Force - Move elements between queues for each pop
    Time Complexity: O(n) for pop, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def Push(self, x: int) -> None:
        self.q1.append(x)
    
    def Pop(self) -> int:
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        result = self.q1.popleft() if self.q1 else None
        self.q1, self.q2 = self.q2, self.q1
        return result
    
    def Top(self) -> int:
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        result = self.q1[0] if self.q1 else None
        if self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
        return result
    
    def Empty(self) -> bool:
        return len(self.q1) == 0

class My_Stack_Two_Queues_Optimized:
    """
    Two Queues Optimized - Make push expensive instead of pop
    Time Complexity: O(n) for push, O(1) for pop
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
        return self.q1.popleft() if self.q1 else None
    
    def Top(self) -> int:
        return self.q1[0] if self.q1 else None
    
    def Empty(self) -> bool:
        return len(self.q1) == 0

class My_Stack_Single_Queue_Optimal:
    """
    Single Queue Optimal - Rotate queue after each push
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.queue = deque()
    
    def Push(self, x: int) -> None:
        self.queue.append(x)
        
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
    
    def Pop(self) -> int:
        return self.queue.popleft() if self.queue else None
    
    def Top(self) -> int:
        return self.queue[0] if self.queue else None
    
    def Empty(self) -> bool:
        return len(self.queue) == 0

class My_Stack_Recursive_Approach:
    """
    Recursive Approach - Use recursion to simulate stack behavior
    Time Complexity: O(n) for pop/top
    Space Complexity: O(n)
    """
    def __init__(self):
        self.queue = deque()
    
    def Push(self, x: int) -> None:
        self.queue.append(x)
    
    def Pop(self) -> int:
        if not self.queue:
            return None
        
        if len(self.queue) == 1:
            return self.queue.popleft()
        
        item = self.queue.popleft()
        result = self.Pop()
        self.queue.append(item)
        return result
    
    def Top(self) -> int:
        result = self.Pop()
        if result is not None:
            self.Push(result)
        return result
    
    def Empty(self) -> bool:
        return len(self.queue) == 0

class My_Stack_Array_Based:
    """
    Array Based Implementation - Using list as queue
    Time Complexity: O(n) for push, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        self.arr = []
    
    def Push(self, x: int) -> None:
        self.arr.append(x)
        
        for i in range(len(self.arr) - 1):
            self.arr.append(self.arr.pop(0))
    
    def Pop(self) -> int:
        return self.arr.pop(0) if self.arr else None
    
    def Top(self) -> int:
        return self.arr[0] if self.arr else None
    
    def Empty(self) -> bool:
        return len(self.arr) == 0

class My_Stack_Priority_Queue:
    """
    Priority Queue Approach - Use counter for LIFO order
    Time Complexity: O(log n) for push/pop, O(1) for others
    Space Complexity: O(n)
    """
    def __init__(self):
        import heapq
        self.heap = []
        self.counter = 0
    
    def Push(self, x: int) -> None:
        import heapq
        heapq.heappush(self.heap, (-self.counter, x))
        self.counter += 1
    
    def Pop(self) -> int:
        import heapq
        if self.heap:
            return heapq.heappop(self.heap)[1]
        return None
    
    def Top(self) -> int:
        return self.heap[0][1] if self.heap else None
    
    def Empty(self) -> bool:
        return len(self.heap) == 0

def Test_Stack_Using_Queues():
    implementations = [
        ("Two Queues Brute Force", My_Stack_Two_Queues_Brute_Force),
        ("Two Queues Optimized", My_Stack_Two_Queues_Optimized),
        ("Single Queue Optimal", My_Stack_Single_Queue_Optimal),
        ("Recursive Approach", My_Stack_Recursive_Approach),
        ("Array Based", My_Stack_Array_Based),
        ("Priority Queue", My_Stack_Priority_Queue)
    ]
    
    operations = [
        ("push", 1),
        ("push", 2),
        ("top", None),
        ("pop", None),
        ("empty", None)
    ]
    
    expected_results = [None, None, 2, 2, False]
    
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
            elif op == "empty":
                result = stack.Empty()
                results.append(result)
        
        print(f"Results: {results}")
        print(f"Expected: {expected_results}")
        print(f"Match: {results == expected_results}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Stack_Using_Queues()
