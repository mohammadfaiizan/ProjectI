# Stack Basic Operations

## Push

Insert element at the top of the stack.

```python
def push(self, x):
    self.arr[self.top] = x
    self.top += 1
```

## Pop

Remove and return the top element. Raises if empty.

```python
def pop(self):
    if self.isEmpty():
        raise IndexError("pop from empty stack")
    self.top -= 1
    return self.arr[self.top]
```

## Peek / Top

Return the top element without removing it.

```python
def peek(self):
    if self.isEmpty():
        raise IndexError("peek from empty stack")
    return self.arr[self.top - 1]
```

## Check isEmpty

Return True if stack has no elements.

```python
def isEmpty(self):
    return self.top == 0
```

## Check isFull (Array-Based)

Return True if stack has reached capacity. Only applicable for fixed-size array stack.

```python
def isFull(self):
    return self.top == self.capacity
```

## Get Size

Return the number of elements in the stack.

```python
def size(self):
    return self.top
```

## Stack Using Array (Fixed Size)

```python
class StackArray:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arr = [None] * capacity
        self.top = 0

    def push(self, x):
        if self.isFull():
            raise OverflowError("stack overflow")
        self.arr[self.top] = x
        self.top += 1

    def pop(self):
        if self.isEmpty():
            raise IndexError("pop from empty stack")
        self.top -= 1
        return self.arr[self.top]

    def peek(self):
        if self.isEmpty():
            raise IndexError("peek from empty stack")
        return self.arr[self.top - 1]

    def isEmpty(self):
        return self.top == 0

    def isFull(self):
        return self.top == self.capacity

    def size(self):
        return self.top
```

## Stack Using Dynamic Array

```python
class StackDynamicArray:
    def __init__(self):
        self.arr = []
        self.top = 0

    def push(self, x):
        if self.top == len(self.arr):
            self.arr.append(x)
        else:
            self.arr[self.top] = x
        self.top += 1

    def pop(self):
        if self.isEmpty():
            raise IndexError("pop from empty stack")
        self.top -= 1
        return self.arr[self.top]

    def peek(self):
        if self.isEmpty():
            raise IndexError("peek from empty stack")
        return self.arr[self.top - 1]

    def isEmpty(self):
        return self.top == 0

    def size(self):
        return self.top
```

Alternative using Python list directly (list.append and list.pop at end are O(1) amortized):

```python
class StackList:
    def __init__(self):
        self.arr = []

    def push(self, x):
        self.arr.append(x)

    def pop(self):
        if not self.arr:
            raise IndexError("pop from empty stack")
        return self.arr.pop()

    def peek(self):
        if not self.arr:
            raise IndexError("peek from empty stack")
        return self.arr[-1]

    def isEmpty(self):
        return len(self.arr) == 0

    def size(self):
        return len(self.arr)
```

## Stack Using Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class StackLinkedList:
    def __init__(self):
        self.head = None
        self._size = 0

    def push(self, x):
        node = Node(x)
        node.next = self.head
        self.head = node
        self._size += 1

    def pop(self):
        if self.isEmpty():
            raise IndexError("pop from empty stack")
        val = self.head.data
        self.head = self.head.next
        self._size -= 1
        return val

    def peek(self):
        if self.isEmpty():
            raise IndexError("peek from empty stack")
        return self.head.data

    def isEmpty(self):
        return self.head is None

    def size(self):
        return self._size
```

## Stack Using Single Queue

Push is O(n): enqueue new element, then rotate n-1 elements to bring it to front. Pop is O(1): dequeue from front.

```python
from collections import deque

class StackSingleQueue:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        if not self.q:
            raise IndexError("pop from empty stack")
        return self.q.popleft()

    def peek(self):
        if not self.q:
            raise IndexError("peek from empty stack")
        return self.q[0]

    def isEmpty(self):
        return len(self.q) == 0

    def size(self):
        return len(self.q)
```

## Stack Using Two Queues

Push O(1): enqueue to q1. Pop O(n): move n-1 elements from q1 to q2, dequeue the last from q1 (top), swap q1 and q2.

```python
from collections import deque

class StackTwoQueues:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        self.q1.append(x)

    def pop(self):
        if not self.q1:
            raise IndexError("pop from empty stack")
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        val = self.q1.popleft()
        self.q1, self.q2 = self.q2, self.q1
        return val

    def peek(self):
        if not self.q1:
            raise IndexError("peek from empty stack")
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        val = self.q1[0]
        self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
        return val

    def isEmpty(self):
        return len(self.q1) == 0

    def size(self):
        return len(self.q1)
```

## Display Stack Contents

Print elements from top to bottom without modifying the stack.

```python
def display(self):
    if self.isEmpty():
        print("Stack is empty")
        return
    for i in range(self.top - 1, -1, -1):
        print(self.arr[i])
```

For linked list:

```python
def display(self):
    if self.isEmpty():
        print("Stack is empty")
        return
    curr = self.head
    while curr:
        print(curr.data)
        curr = curr.next
```

## Clear Stack

Remove all elements from the stack.

```python
def clear(self):
    self.top = 0
```

For linked list:

```python
def clear(self):
    self.head = None
    self._size = 0
```

For Python list-based:

```python
def clear(self):
    self.arr.clear()
```
