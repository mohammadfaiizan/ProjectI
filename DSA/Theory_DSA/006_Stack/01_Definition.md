# Stack Definition and Fundamentals

## LIFO Principle

A stack is a linear data structure that follows the **Last In, First Out (LIFO)** principle. The element inserted last is the first one to be removed. Insertion and deletion occur only at one end, called the **top** of the stack.

```
Push order: A, B, C, D
Pop order:  D, C, B, A

Top -> [D] <- most recently added
       [C]
       [B]
       [A] <- bottom
```

## Stack Abstract Data Type (ADT)

The stack ADT defines the following operations without specifying implementation:

| Operation | Description |
|-----------|-------------|
| push(x) | Insert element x at the top |
| pop() | Remove and return the top element |
| peek() / top() | Return the top element without removing |
| isEmpty() | Return true if stack has no elements |
| size() | Return the number of elements |

## Array-Based Stack

Elements are stored in a contiguous array. A variable `top` (or index) tracks the position of the top element. Push increments top and stores; pop returns element at top and decrements.

```
Array: [A][B][C][ ][ ][ ]
        0  1  2  3  4  5
        top = 2
```

**Advantages**: Cache-friendly, O(1) access, simple implementation.
**Disadvantages**: Fixed capacity (unless dynamic array), may require reallocation.

## Linked List-Based Stack

The top of the stack is the head of the linked list. Push adds a new node at the head; pop removes the head. Each node stores data and a pointer to the next node.

```
Top -> [D|next] -> [C|next] -> [B|next] -> [A|None]
```

**Advantages**: No fixed size limit, O(1) push and pop at head.
**Disadvantages**: Extra memory for pointers, cache-unfriendly.

## Function Call Stack

When a function is called, its return address and local variables are pushed onto the system call stack. When the function returns, the frame is popped. Recursion uses the same mechanism: each recursive call pushes a new frame.

```
main() calls f()
f() calls g()
g() returns
f() returns
main() continues

Stack frames (top to bottom): [g's frame][f's frame][main's frame]
```

## Stack Overflow

Occurs when push is attempted on a full stack (array-based) or when the call stack exceeds available memory (recursion depth too high). In recursion, infinite recursion or very deep recursion causes stack overflow.

## Stack Underflow

Occurs when pop or peek is attempted on an empty stack. Must be handled by checking isEmpty before pop/peek, or by defining pop/peek to return a sentinel/raise exception when empty.

## When to Use Stack

| Use Case | Why Stack |
|----------|-----------|
| Matching | Pairs (parentheses, brackets) - push opening, pop on closing match |
| Nesting | Track nested structure (tags, blocks) - depth = stack size |
| Undo | Last action undone first - push actions, pop to undo |
| DFS | Depth-first traversal - push neighbors, pop to backtrack |
| Expression evaluation | Postfix/prefix - operands and operators in LIFO order |
| Monotonic problems | Next greater/smaller - maintain monotonic order |

## Time Complexity Table

| Operation | Array-Based | Linked List-Based | Notes |
|-----------|-------------|-------------------|-------|
| push | O(1) amortized | O(1) | Array: O(n) worst if resize |
| pop | O(1) | O(1) | |
| peek / top | O(1) | O(1) | |
| isEmpty | O(1) | O(1) | |
| search | O(n) | O(n) | Linear scan from top |
| get size | O(1) | O(1) | If size maintained |

## Space Complexity

- Array-based: O(n) for n elements, plus O(1) for top index. May over-allocate.
- Linked list-based: O(n) for n nodes, each with data + pointer overhead.
- Auxiliary stack (e.g., for recursion simulation): O(depth) for recursion depth or problem-specific.
