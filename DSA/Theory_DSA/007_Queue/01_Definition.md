# Queue Definition and Fundamentals

## FIFO Principle

A queue is a linear data structure that follows the **First In, First Out (FIFO)** principle. The element inserted first is the first one to be removed. Insertion occurs at one end called the **rear** (or back), and removal occurs at the other end called the **front** (or head).

```
Enqueue order: A, B, C, D
Dequeue order: A, B, C, D

Front -> [A] <- first to be removed
         [B]
         [C]
         [D] <- rear (last added)
```

## Queue Abstract Data Type (ADT)

The queue ADT defines the following operations without specifying implementation:

| Operation | Description |
|-----------|-------------|
| enqueue(x) | Insert element x at the rear |
| dequeue() | Remove and return the element at the front |
| front() / peek() | Return the front element without removing |
| rear() | Return the rear element without removing |
| isEmpty() | Return true if queue has no elements |
| size() | Return the number of elements |

## Simple Queue (Linear Queue)

A basic queue where elements are stored sequentially. In an array-based implementation, the front index moves forward as elements are dequeued, potentially leaving unused space at the beginning. This can lead to "false overflow" when the rear reaches the end but space exists at the front.

```
Array: [ ][ ][C][D][E]
        0  1  2  3  4
        front=2, rear=4
```

**Problem**: After several dequeues, front advances. When rear reaches capacity, the queue appears full even if space exists before front.

## Circular Queue

A queue implemented as a circular buffer where the rear wraps around to the beginning when it reaches the end. Uses modulo arithmetic to compute indices: `(front + 1) % capacity` and `(rear + 1) % capacity`. Eliminates wasted space from the linear queue.

```
Capacity 5:
  [A][B][C][D][ ]
   0  1  2  3  4
  front=0, rear=3

After dequeue A, enqueue E:
  [E][B][C][D][ ]
   0  1  2  3  4
  front=1, rear=4
```

**Advantages**: Efficient use of space, O(1) enqueue and dequeue.
**Disadvantages**: Fixed capacity, requires careful handling of full/empty distinction (often by sacrificing one slot or maintaining a count).

## Deque (Double-Ended Queue)

A deque allows insertion and deletion at both ends. Operations: push_front, push_back, pop_front, pop_back, peek_front, peek_back. Can simulate both stack and queue behavior.

```
Front <-> [A][B][C][D] <-> Rear
          push_front/pop_front    push_back/pop_back
```

**Use cases**: Sliding window problems, palindrome checking, undo-redo with both ends, level-order traversal with front/back manipulation.

## Priority Queue Overview

A priority queue orders elements by priority. The element with highest (or lowest) priority is removed first, regardless of insertion order. Typically implemented using a heap (binary heap, Fibonacci heap). Detailed coverage in module 008.

| Operation | Time Complexity (Binary Heap) |
|-----------|-------------------------------|
| insert | O(log n) |
| extract max/min | O(log n) |
| peek | O(1) |

## When to Use Queue

| Use Case | Why Queue |
|----------|-----------|
| BFS | Breadth-first search processes nodes level by level; queue ensures FIFO order |
| Scheduling | Task scheduling (CPU, printer) where first-come-first-served is required |
| Buffering | Network packets, I/O streams - process in arrival order |
| Order Processing | E-commerce orders, ticket systems - maintain fair ordering |
| Level-order traversal | Binary tree levels processed left-to-right |
| Sliding window | Deque for maintaining useful indices in window problems |

## Time Complexity Table

| Operation | Array (Linear) | Circular Array | Linked List | Deque (collections.deque) |
|-----------|----------------|----------------|-------------|---------------------------|
| enqueue | O(1) | O(1) | O(1) | O(1) |
| dequeue | O(1) | O(1) | O(1) | O(1) |
| front | O(1) | O(1) | O(1) | O(1) |
| rear | O(1) | O(1) | O(1) | O(1) |
| isEmpty | O(1) | O(1) | O(1) | O(1) |
| isFull | O(1) | O(1) | N/A | N/A |
| size | O(1) | O(1) | O(1) | O(1) |

**Note**: Linear array dequeue can be O(n) if we shift elements to fill the gap. Using front/rear pointers without shifting gives O(1) but wastes space. Circular queue achieves O(1) without shifting.

## Space Complexity

- Array-based (linear or circular): O(n) for n elements, fixed or dynamic capacity.
- Linked list-based: O(n) for n nodes, each with data and pointer(s).
- Deque (doubly linked list or circular buffer): O(n).
