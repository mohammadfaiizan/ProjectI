# Queue Basic Operations

## Enqueue

Insert element at the rear of the queue.

```python
def enqueue(self, x):
    if self.isFull():
        raise OverflowError("queue overflow")
    self.arr[self.rear] = x
    self.rear = (self.rear + 1) % self.capacity
    self.size += 1
```

## Dequeue

Remove and return the element at the front. Raises if empty.

```python
def dequeue(self):
    if self.isEmpty():
        raise IndexError("dequeue from empty queue")
    val = self.arr[self.front]
    self.front = (self.front + 1) % self.capacity
    self.size -= 1
    return val
```

## Peek Front

Return the front element without removing it.

```python
def peek_front(self):
    if self.isEmpty():
        raise IndexError("peek from empty queue")
    return self.arr[self.front]
```

## Peek Rear

Return the rear element without removing it.

```python
def peek_rear(self):
    if self.isEmpty():
        raise IndexError("peek from empty queue")
    idx = (self.rear - 1 + self.capacity) % self.capacity
    return self.arr[idx]
```

## Check isEmpty

Return True if queue has no elements.

```python
def isEmpty(self):
    return self.size == 0
```

## Check isFull

Return True if queue has reached capacity. Applicable for fixed-size implementations.

```python
def isFull(self):
    return self.size == self.capacity
```

## Get Size

Return the number of elements in the queue.

```python
def get_size(self):
    return self.size
```

## Queue Using Array (Linear)

Uses front and rear indices. Rear advances on enqueue; front advances on dequeue. May waste space when front moves forward.

```python
class QueueArray:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arr = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0

    def enqueue(self, x):
        if self.isFull():
            raise OverflowError("queue overflow")
        self.rear += 1
        self.arr[self.rear] = x
        self.size += 1

    def dequeue(self):
        if self.isEmpty():
            raise IndexError("dequeue from empty queue")
        val = self.arr[self.front]
        self.front += 1
        self.size -= 1
        return val

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        return self.arr[self.front]

    def peek_rear(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        return self.arr[self.rear]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity

    def get_size(self):
        return self.size
```

## Circular Queue Using Array

Uses modulo arithmetic to wrap indices. One slot is sacrificed to distinguish full from empty, or a size counter is maintained.

```python
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arr = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0

    def enqueue(self, x):
        if self.isFull():
            raise OverflowError("queue overflow")
        self.arr[self.rear] = x
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        if self.isEmpty():
            raise IndexError("dequeue from empty queue")
        val = self.arr[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return val

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        return self.arr[self.front]

    def peek_rear(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        idx = (self.rear - 1 + self.capacity) % self.capacity
        return self.arr[idx]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity

    def get_size(self):
        return self.size
```

## Queue Using Linked List

Front is the head; rear is the tail. Enqueue adds at tail; dequeue removes from head. O(1) for both operations.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class QueueLinkedList:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def enqueue(self, x):
        node = Node(x)
        if self.rear is None:
            self.front = self.rear = node
        else:
            self.rear.next = node
            self.rear = node
        self._size += 1

    def dequeue(self):
        if self.isEmpty():
            raise IndexError("dequeue from empty queue")
        val = self.front.data
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self._size -= 1
        return val

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        return self.front.data

    def peek_rear(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        return self.rear.data

    def isEmpty(self):
        return self.front is None

    def get_size(self):
        return self._size
```

## Queue Using Two Stacks (Amortized O(1))

Stack1 receives enqueues. Stack2 is used for dequeues. When stack2 is empty, pop all from stack1 and push to stack2. Each element is pushed and popped at most twice, giving amortized O(1) per operation.

```python
class QueueTwoStacks:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def enqueue(self, x):
        self.stack_in.append(x)

    def dequeue(self):
        if self.isEmpty():
            raise IndexError("dequeue from empty queue")
        self._transfer()
        return self.stack_out.pop()

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        self._transfer()
        return self.stack_out[-1]

    def _transfer(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())

    def isEmpty(self):
        return not self.stack_in and not self.stack_out

    def get_size(self):
        return len(self.stack_in) + len(self.stack_out)
```

## Queue Using Single Stack (Recursive)

Dequeue uses recursion to reach the bottom of the stack, then returns the bottom element while unwinding. Enqueue is O(1); dequeue is O(n) due to recursion depth.

```python
class QueueSingleStack:
    def __init__(self):
        self.stack = []

    def enqueue(self, x):
        self.stack.append(x)

    def dequeue(self):
        if self.isEmpty():
            raise IndexError("dequeue from empty queue")
        if len(self.stack) == 1:
            return self.stack.pop()
        top = self.stack.pop()
        result = self.dequeue()
        self.stack.append(top)
        return result

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty queue")
        if len(self.stack) == 1:
            return self.stack[-1]
        top = self.stack.pop()
        result = self.peek_front()
        self.stack.append(top)
        return result

    def isEmpty(self):
        return len(self.stack) == 0

    def get_size(self):
        return len(self.stack)
```

## Deque Using Array

Circular array with front and rear. Supports add/remove at both ends.

```python
class DequeArray:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arr = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0

    def _prev(self, i):
        return (i - 1 + self.capacity) % self.capacity

    def _next(self, i):
        return (i + 1) % self.capacity

    def push_front(self, x):
        if self.isFull():
            raise OverflowError("deque overflow")
        self.front = self._prev(self.front)
        self.arr[self.front] = x
        self.size += 1

    def push_back(self, x):
        if self.isFull():
            raise OverflowError("deque overflow")
        self.arr[self.rear] = x
        self.rear = self._next(self.rear)
        self.size += 1

    def pop_front(self):
        if self.isEmpty():
            raise IndexError("pop from empty deque")
        val = self.arr[self.front]
        self.front = self._next(self.front)
        self.size -= 1
        return val

    def pop_back(self):
        if self.isEmpty():
            raise IndexError("pop from empty deque")
        self.rear = self._prev(self.rear)
        val = self.arr[self.rear]
        self.size -= 1
        return val

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty deque")
        return self.arr[self.front]

    def peek_back(self):
        if self.isEmpty():
            raise IndexError("peek from empty deque")
        idx = self._prev(self.rear)
        return self.arr[idx]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity

    def get_size(self):
        return self.size
```

## Deque Using Doubly Linked List

Each node has prev and next pointers. O(1) for all operations at both ends.

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DequeDoublyLinked:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def push_front(self, x):
        node = DNode(x)
        if self.front is None:
            self.front = self.rear = node
        else:
            node.next = self.front
            self.front.prev = node
            self.front = node
        self._size += 1

    def push_back(self, x):
        node = DNode(x)
        if self.rear is None:
            self.front = self.rear = node
        else:
            node.prev = self.rear
            self.rear.next = node
            self.rear = node
        self._size += 1

    def pop_front(self):
        if self.isEmpty():
            raise IndexError("pop from empty deque")
        val = self.front.data
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        else:
            self.front.prev = None
        self._size -= 1
        return val

    def pop_back(self):
        if self.isEmpty():
            raise IndexError("pop from empty deque")
        val = self.rear.data
        self.rear = self.rear.prev
        if self.rear is None:
            self.front = None
        else:
            self.rear.next = None
        self._size -= 1
        return val

    def peek_front(self):
        if self.isEmpty():
            raise IndexError("peek from empty deque")
        return self.front.data

    def peek_back(self):
        if self.isEmpty():
            raise IndexError("peek from empty deque")
        return self.rear.data

    def isEmpty(self):
        return self.front is None

    def get_size(self):
        return self._size
```

## Display Queue Contents

Print elements from front to rear without modifying the queue.

```python
def display(self):
    if self.isEmpty():
        print("Queue is empty")
        return
    i = self.front
    for _ in range(self.size):
        print(self.arr[i])
        i = (i + 1) % self.capacity
```

For linked list:

```python
def display(self):
    if self.isEmpty():
        print("Queue is empty")
        return
    curr = self.front
    while curr:
        print(curr.data)
        curr = curr.next
```

## Clear Queue

Remove all elements from the queue.

```python
def clear(self):
    self.front = 0
    self.rear = 0
    self.size = 0
```

For linked list:

```python
def clear(self):
    self.front = None
    self.rear = None
    self._size = 0
```

For two-stacks implementation:

```python
def clear(self):
    self.stack_in.clear()
    self.stack_out.clear()
```
