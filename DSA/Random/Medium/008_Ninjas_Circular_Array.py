"""
Problem: Ninja's Circular Array
URL: https://www.naukri.com/code360/problems/ninja-s-circular-array

Problem Statement:
Design a circular array that supports the following operations:
- addAtEnd(value): Add an element at the end
- addAtStart(value): Add an element at the start
- removeFromEnd(): Remove an element from the end
- removeFromStart(): Remove an element from the start
- getFirst(): Get the first element
- getLast(): Get the last element

Sample Input/Output:
Input: ["CircularArray", "addAtEnd", "addAtEnd", "getFirst", "getLast", "removeFromStart"]
       [[], [1], [2], [], [], []]
Output: [null, null, null, 1, 2, null]
"""

from typing import List, Optional
from collections import deque

class Circular_Array_List:
    """
    List Approach
    Time Complexity: O(1) for end operations, O(n) for start operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.arr = []
    
    def Add_At_End(self, value: int) -> None:
        self.arr.append(value)
    
    def Add_At_Start(self, value: int) -> None:
        self.arr.insert(0, value)
    
    def Remove_From_End(self) -> Optional[int]:
        return self.arr.pop() if self.arr else None
    
    def Remove_From_Start(self) -> Optional[int]:
        return self.arr.pop(0) if self.arr else None
    
    def Get_First(self) -> Optional[int]:
        return self.arr[0] if self.arr else None
    
    def Get_Last(self) -> Optional[int]:
        return self.arr[-1] if self.arr else None

class Circular_Array_Deque:
    """
    Deque Approach - Optimal solution
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.dq = deque()
    
    def Add_At_End(self, value: int) -> None:
        self.dq.append(value)
    
    def Add_At_Start(self, value: int) -> None:
        self.dq.appendleft(value)
    
    def Remove_From_End(self) -> Optional[int]:
        return self.dq.pop() if self.dq else None
    
    def Remove_From_Start(self) -> Optional[int]:
        return self.dq.popleft() if self.dq else None
    
    def Get_First(self) -> Optional[int]:
        return self.dq[0] if self.dq else None
    
    def Get_Last(self) -> Optional[int]:
        return self.dq[-1] if self.dq else None

class Circular_Array_Fixed_Size:
    """
    Fixed Size Circular Array
    Time Complexity: O(1) for all operations
    Space Complexity: O(capacity)
    """
    def __init__(self, capacity: int = 1000):
        self.arr = [0] * capacity
        self.capacity = capacity
        self.front = 0
        self.rear = 0
        self.size = 0
    
    def Add_At_End(self, value: int) -> None:
        if self.size == self.capacity:
            return
        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
    
    def Add_At_Start(self, value: int) -> None:
        if self.size == self.capacity:
            return
        self.front = (self.front - 1 + self.capacity) % self.capacity
        self.arr[self.front] = value
        self.size += 1
    
    def Remove_From_End(self) -> Optional[int]:
        if self.size == 0:
            return None
        self.rear = (self.rear - 1 + self.capacity) % self.capacity
        value = self.arr[self.rear]
        self.size -= 1
        return value
    
    def Remove_From_Start(self) -> Optional[int]:
        if self.size == 0:
            return None
        value = self.arr[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return value
    
    def Get_First(self) -> Optional[int]:
        return self.arr[self.front] if self.size > 0 else None
    
    def Get_Last(self) -> Optional[int]:
        return self.arr[(self.rear - 1 + self.capacity) % self.capacity] if self.size > 0 else None

class Node:
    def __init__(self, val: int):
        self.val = val
        self.prev = None
        self.next = None

class Circular_Array_Doubly_Linked_List:
    """
    Doubly Linked List Approach
    Time Complexity: O(1) for all operations
    Space Complexity: O(n)
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def Add_At_End(self, value: int) -> None:
        new_node = Node(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.size += 1
    
    def Add_At_Start(self, value: int) -> None:
        new_node = Node(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
    
    def Remove_From_End(self) -> Optional[int]:
        if not self.tail:
            return None
        value = self.tail.val
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        self.size -= 1
        return value
    
    def Remove_From_Start(self) -> Optional[int]:
        if not self.head:
            return None
        value = self.head.val
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        self.size -= 1
        return value
    
    def Get_First(self) -> Optional[int]:
        return self.head.val if self.head else None
    
    def Get_Last(self) -> Optional[int]:
        return self.tail.val if self.tail else None

def Test_Circular_Array():
    for approach_name, ArrayClass in [
        ("List", Circular_Array_List),
        ("Deque", Circular_Array_Deque),
        ("Doubly Linked List", Circular_Array_Doubly_Linked_List)
    ]:
        print(f"Testing {approach_name} Approach:")
        arr = ArrayClass()
        
        arr.Add_At_End(1)
        arr.Add_At_End(2)
        print(f"After adding 1, 2 at end: First={arr.Get_First()}, Last={arr.Get_Last()}")
        
        arr.Add_At_Start(0)
        print(f"After adding 0 at start: First={arr.Get_First()}, Last={arr.Get_Last()}")
        
        arr.Remove_From_End()
        print(f"After removing from end: First={arr.Get_First()}, Last={arr.Get_Last()}")
        
        arr.Remove_From_Start()
        print(f"After removing from start: First={arr.Get_First()}, Last={arr.Get_Last()}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Circular_Array()

