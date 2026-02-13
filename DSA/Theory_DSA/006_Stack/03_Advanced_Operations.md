# Stack Advanced Operations

## Min Stack (getMin O(1) Time and O(1) Space)

Store value and min-so-far together. On push, min = min(new_val, current_min). On pop, no extra work. O(1) space because we only store one min per element.

```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, x):
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, min(x, self.stack[-1][1])))

    def pop(self):
        if not self.stack:
            raise IndexError("pop from empty stack")
        return self.stack.pop()[0]

    def top(self):
        if not self.stack:
            raise IndexError("top from empty stack")
        return self.stack[-1][0]

    def getMin(self):
        if not self.stack:
            raise IndexError("getMin from empty stack")
        return self.stack[-1][1]
```

## Max Stack (getMax O(1))

Same idea as min stack: store (value, max_so_far) per element.

```python
class MaxStack:
    def __init__(self):
        self.stack = []

    def push(self, x):
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, max(x, self.stack[-1][1])))

    def pop(self):
        if not self.stack:
            raise IndexError("pop from empty stack")
        return self.stack.pop()[0]

    def top(self):
        if not self.stack:
            raise IndexError("top from empty stack")
        return self.stack[-1][0]

    def getMax(self):
        if not self.stack:
            raise IndexError("getMax from empty stack")
        return self.stack[-1][1]
```

## Stack with Middle Element Access O(1)

Use doubly linked list with a pointer to the middle node. On push/pop, update middle: move middle forward when size goes from odd to even (after push), backward when even to odd (after pop). Middle access is O(1).

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class StackWithMiddle:
    def __init__(self):
        self.head = None
        self.mid = None
        self.count = 0

    def push(self, x):
        node = Node(x)
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        self.count += 1
        if self.count == 1:
            self.mid = node
        elif self.count % 2 == 1:
            self.mid = self.mid.prev

    def pop(self):
        if not self.head:
            raise IndexError("pop from empty stack")
        val = self.head.data
        self.head = self.head.next
        if self.head:
            self.head.prev = None
        self.count -= 1
        if self.count == 0:
            self.mid = None
        elif self.count % 2 == 0:
            self.mid = self.mid.next if self.mid else None
        return val

    def findMiddle(self):
        if not self.mid:
            raise IndexError("middle of empty stack")
        return self.mid.data

    def deleteMiddle(self):
        if not self.mid:
            raise IndexError("delete middle of empty stack")
        val = self.mid.data
        if self.mid.prev:
            self.mid.prev.next = self.mid.next
        if self.mid.next:
            self.mid.next.prev = self.mid.prev
        if self.head == self.mid:
            self.head = self.mid.next
        self.count -= 1
        if self.count % 2 == 0:
            self.mid = self.mid.next if self.mid else None
        else:
            self.mid = self.mid.prev if self.mid else None
        return val
```

## Sort a Stack Using Recursion

Pop all elements recursively, then insert each in sorted order at the bottom.

```python
def sorted_insert(stack, x):
    if not stack or stack[-1] <= x:
        stack.append(x)
        return
    top = stack.pop()
    sorted_insert(stack, x)
    stack.append(top)

def sort_stack_recursion(stack):
    if not stack:
        return
    x = stack.pop()
    sort_stack_recursion(stack)
    sorted_insert(stack, x)
```

## Sort a Stack Using Temporary Stack

Use an auxiliary stack. Pop from main, while auxiliary top is greater than current, push back to main. Then push current to auxiliary. Repeat until main is empty. Auxiliary becomes sorted (ascending from bottom).

```python
def sort_stack_temp(stack):
    temp = []
    while stack:
        x = stack.pop()
        while temp and temp[-1] > x:
            stack.append(temp.pop())
        temp.append(x)
    while temp:
        stack.append(temp.pop())
```

## Reverse a Stack Using Recursion (No Extra DS)

Pop all elements recursively, then insert each at the bottom. Insert-at-bottom: pop all, push x, push back the popped elements.

```python
def insert_at_bottom(stack, x):
    if not stack:
        stack.append(x)
        return
    top = stack.pop()
    insert_at_bottom(stack, x)
    stack.append(top)

def reverse_stack_recursion(stack):
    if not stack:
        return
    x = stack.pop()
    reverse_stack_recursion(stack)
    insert_at_bottom(stack, x)
```

## Insert at Bottom of Stack

```python
def insert_at_bottom(stack, x):
    if not stack:
        stack.append(x)
        return
    top = stack.pop()
    insert_at_bottom(stack, x)
    stack.append(top)
```

## Delete Middle Element of Stack

Recursively pop and count. When we reach middle (count == n//2), don't push back. Otherwise push back.

```python
def delete_middle(stack, n=None, curr=0):
    if n is None:
        n = len(stack)
    if not stack:
        return
    x = stack.pop()
    delete_middle(stack, n, curr + 1)
    if curr != n // 2:
        stack.append(x)
```

## Implement Two Stacks in One Array

Stack1 grows from left (top at start), Stack2 grows from right (top at end). They meet when top1 + 1 == top2.

```python
class TwoStacks:
    def __init__(self, n):
        self.size = n
        self.arr = [None] * n
        self.top1 = -1
        self.top2 = n

    def push1(self, x):
        if self.top1 + 1 >= self.top2:
            raise OverflowError("stack overflow")
        self.top1 += 1
        self.arr[self.top1] = x

    def push2(self, x):
        if self.top2 - 1 <= self.top1:
            raise OverflowError("stack overflow")
        self.top2 -= 1
        self.arr[self.top2] = x

    def pop1(self):
        if self.top1 < 0:
            raise IndexError("pop from empty stack")
        val = self.arr[self.top1]
        self.top1 -= 1
        return val

    def pop2(self):
        if self.top2 >= self.size:
            raise IndexError("pop from empty stack")
        val = self.arr[self.top2]
        self.top2 += 1
        return val
```

## Implement K Stacks in One Array

Use one array for values, one for next index of each stack's top, one for top of each stack, one for free list. Each slot stores value and next pointer.

```python
class KStacks:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.arr = [0] * n
        self.next_idx = list(range(1, n)) + [-1]
        self.top = [-1] * k
        self.free = 0

    def isFull(self):
        return self.free == -1

    def isEmpty(self, sn):
        return self.top[sn] == -1

    def push(self, sn, x):
        if self.isFull():
            raise OverflowError("stack overflow")
        i = self.free
        self.free = self.next_idx[i]
        self.next_idx[i] = self.top[sn]
        self.top[sn] = i
        self.arr[i] = x

    def pop(self, sn):
        if self.isEmpty(sn):
            raise IndexError("pop from empty stack")
        i = self.top[sn]
        self.top[sn] = self.next_idx[i]
        self.next_idx[i] = self.free
        self.free = i
        return self.arr[i]
```

## Stock Span Using Stack

For each day, span = 1 + count of consecutive previous days with price <= current. Use stack of indices. Pop while stack not empty and price[stack[-1]] <= price[i]. Span = i - stack[-1] if stack else i + 1.

```python
def stock_span(prices):
    n = len(prices)
    span = [0] * n
    stack = []
    for i in range(n):
        while stack and prices[stack[-1]] <= prices[i]:
            stack.pop()
        span[i] = i - stack[-1] if stack else i + 1
        stack.append(i)
    return span
```

## Celebrity Problem Using Stack

Celebrity: everyone knows them, they know nobody. Push all indices. Pop two (a, b): if a knows b, a is not celebrity, push b; else b is not celebrity, push a. Last remaining candidate: verify they know nobody and everyone knows them.

```python
def knows(matrix, a, b):
    return matrix[a][b] == 1

def find_celebrity(matrix, n):
    stack = list(range(n))
    while len(stack) > 1:
        a = stack.pop()
        b = stack.pop()
        if knows(matrix, a, b):
            stack.append(b)
        else:
            stack.append(a)
    c = stack.pop()
    for i in range(n):
        if i != c and (knows(matrix, c, i) or not knows(matrix, i, c)):
            return -1
    return c
```

## Iterative Tower of Hanoi

Use three stacks for source, auxiliary, dest. Legal move: move disk from one rod to another only if the destination is empty or has a larger disk. Total moves = 2^n - 1. Simulate with stack operations.

```python
def tower_of_hanoi_iterative(n, src_name='A', aux_name='B', dest_name='C'):
    src, aux, dest = list(range(n, 0, -1)), [], []
    total = 2**n - 1
    if n % 2 == 0:
        aux, dest = dest, aux
        aux_name, dest_name = dest_name, aux_name
    for move in range(1, total + 1):
        if move % 3 == 1:
            if not src:
                src.append(dest.pop())
                print(f"Move disk from {dest_name} to {src_name}")
            elif not dest or src[-1] < dest[-1]:
                dest.append(src.pop())
                print(f"Move disk from {src_name} to {dest_name}")
            else:
                src.append(dest.pop())
                print(f"Move disk from {dest_name} to {src_name}")
        elif move % 3 == 2:
            if not src:
                src.append(aux.pop())
                print(f"Move disk from {aux_name} to {src_name}")
            elif not aux or src[-1] < aux[-1]:
                aux.append(src.pop())
                print(f"Move disk from {src_name} to {aux_name}")
            else:
                src.append(aux.pop())
                print(f"Move disk from {aux_name} to {src_name}")
        else:
            if not aux:
                aux.append(dest.pop())
                print(f"Move disk from {dest_name} to {aux_name}")
            elif not dest or aux[-1] < dest[-1]:
                dest.append(aux.pop())
                print(f"Move disk from {aux_name} to {dest_name}")
            else:
                aux.append(dest.pop())
                print(f"Move disk from {dest_name} to {aux_name}")
```

## Reverse String Using Stack

Push each character, then pop to build reversed string.

```python
def reverse_string(s):
    stack = list(s)
    return ''.join(stack.pop() for _ in range(len(stack)))
```

## Check Balanced Parentheses

Push opening brackets. On closing, pop and check match. Stack must be empty at end.

```python
def is_balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0
```
