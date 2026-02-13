# Linked List Definition and Fundamentals

## Singly Linked List

A singly linked list consists of nodes where each node contains:
- **data**: The value stored in the node
- **next**: A reference (pointer) to the next node in the sequence

The last node has `next = None` (or null), indicating the end of the list. Only forward traversal is possible.

```
[data|next] -> [data|next] -> [data|next] -> None
```

## Doubly Linked List

A doubly linked list node contains:
- **prev**: Reference to the previous node
- **data**: The value stored
- **next**: Reference to the next node

The first node has `prev = None`, the last has `next = None`. Bidirectional traversal is possible.

```
None <-> [prev|data|next] <-> [prev|data|next] <-> [prev|data|next] <-> None
```

## Circular Singly Linked List

Same structure as singly linked list, but the last node's `next` points back to the first node instead of None. Forms a cycle. No explicit "end" without a designated head.

```
[data|next] -> [data|next] -> [data|next] -+
    ^                                      |
    +--------------------------------------+
```

## Circular Doubly Linked List

Combines doubly linked list with circular structure. The last node's `next` points to the first, and the first node's `prev` points to the last. Full bidirectional traversal in a ring.

```
    +--------------------------------------+
    v                                      |
[prev|data|next] <-> [prev|data|next] <-> [prev|data|next]
    ^                                      |
    +--------------------------------------+
```

## Memory Layout

### Non-Contiguous Allocation

Unlike arrays, linked list nodes are not stored in consecutive memory. Each node is allocated separately (typically on the heap). Nodes can be scattered across memory.

### Heap Allocation

Each node requires a separate allocation. In Python, objects are heap-allocated. The list head (or a reference to it) is the only direct handle; reaching any node requires traversing from the head.

### Pointer Overhead

Each node incurs overhead for the pointer(s). Singly: 1 pointer per node. Doubly: 2 pointers per node. For small data types, overhead can exceed data size.

## Linked List vs Array Comparison

| Operation | Array | Singly Linked List | Doubly Linked List |
|-----------|-------|-------------------|---------------------|
| Access by index | O(1) | O(n) | O(n) |
| Search by value | O(n) | O(n) | O(n) |
| Insert at head | O(n) | O(1) | O(1) |
| Insert at tail | O(1) amortized | O(1) if tail known, else O(n) | O(1) if tail known |
| Insert at position k | O(n) | O(k) | O(k) |
| Delete at head | O(n) | O(1) | O(1) |
| Delete at tail | O(1) | O(n) unless tail ptr | O(1) if tail known |
| Delete at position k | O(n) | O(k) | O(k) |
| Memory | Contiguous, minimal overhead | Non-contiguous, pointer overhead | Non-contiguous, 2x pointer overhead |
| Cache performance | Excellent (spatial locality) | Poor (random access) | Poor |
| Extra space for growth | May need reallocation | O(1) per new node | O(1) per new node |

## Sentinel and Dummy Nodes

### Dummy Node (Head Sentinel)

A dummy node placed before the actual first element. Its `next` points to the real head. Benefits:
- Eliminates special case for empty list (dummy always exists)
- Simplifies insert-at-head: new node's next = dummy.next, dummy.next = new node
- No need to update head pointer variable when inserting at beginning

```python
dummy = Node(0)
dummy.next = head
```

### Tail Sentinel

A sentinel at the end can simplify append operations in some implementations.

### When to Use

Use dummy nodes when:
- Multiple operations might change the head (e.g., delete head, insert at head)
- Merging or building a new list from scratch
- Avoiding null checks in loop bodies

## When to Use Linked List

**Prefer linked list when:**
- Frequent insertions/deletions at beginning (O(1) vs O(n) for array)
- Unknown or highly variable size (no reallocation)
- Implementing queues, stacks, or adjacency lists for graphs
- Need to splice sublists without copying (pointer manipulation)
- Implementing certain algorithms (LRU cache, polynomial arithmetic)

**Prefer array when:**
- Random access by index is required
- Memory and cache efficiency matter
- Size is known or predictable
- Iteration performance is critical

## Time Complexity Summary

| Operation | Singly | Doubly | Notes |
|-----------|--------|--------|-------|
| Access by index | O(n) | O(n) | Must traverse |
| Search | O(n) | O(n) | Linear scan |
| Insert at head | O(1) | O(1) | |
| Insert at tail | O(1)* | O(1)* | *If tail pointer maintained |
| Insert at position k | O(k) | O(k) | |
| Delete head | O(1) | O(1) | |
| Delete tail | O(n)* | O(1)* | *Singly needs traversal |
| Delete at k | O(k) | O(k) | |
| Reverse | O(n) | O(n) | |
| Merge two sorted | O(n+m) | O(n+m) | |

## Space Complexity

- n nodes: O(n) for data + O(n) for pointers (singly: 1 ptr/node, doubly: 2 ptrs/node)
- Recursive traversal: O(n) call stack for recursion depth
- Iterative operations: O(1) auxiliary space typically
