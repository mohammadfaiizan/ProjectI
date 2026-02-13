# Heap and Priority Queue - Definition and Fundamentals

## Complete Binary Tree Property

A heap is a complete binary tree. A complete binary tree has the following properties:

1. All levels are fully filled except possibly the last level
2. The last level has all nodes as left as possible
3. Every node has at most two children

This structure allows efficient array representation without pointers.

## Heap Property

### Min-Heap

In a min-heap, for every node i (except root):
- `parent(i) <= child(i)`
- The smallest element is always at the root

### Max-Heap

In a max-heap, for every node i (except root):
- `parent(i) >= child(i)`
- The largest element is always at the root

## Array Representation

A heap can be stored in a zero-indexed array where:

| Node at index i | Parent | Left Child | Right Child |
|-----------------|--------|------------|--------------|
| Index formula   | `(i-1)//2` | `2*i+1` | `2*i+2` |

Example: Array `[10, 20, 15, 30, 40]` as min-heap (conceptually):
```
        10 (index 0)
       /  \
     20    15 (indices 1, 2)
    /  \
  30    40 (indices 3, 4)
```

## Priority Queue as Abstract Type

A priority queue is an abstract data type that supports:
- Insert element with priority
- Extract element with highest (or lowest) priority
- Peek at top element without removing

A binary heap is the most common implementation of a priority queue due to O(log n) insert and extract operations.

## Min-Heap vs Max-Heap vs Sorted Array vs BST

| Operation | Min/Max Heap | Sorted Array | BST (Balanced) |
|-----------|--------------|--------------|----------------|
| Insert | O(log n) | O(n) | O(log n) |
| Extract Min/Max | O(log n) | O(1) or O(n) | O(log n) |
| Peek | O(1) | O(1) | O(log n) |
| Build from n elements | O(n) | O(n log n) | O(n log n) |
| Find kth smallest | O(k log n) | O(1) | O(k) |
| Space | O(n) | O(n) | O(n) |

## When to Use Heap

| Use Case | Description |
|----------|-------------|
| Top-K | Find k largest/smallest elements efficiently |
| Merge K Sorted | Merge k sorted lists/arrays in O(n log k) |
| Median | Maintain running median with two heaps |
| Scheduling | Process tasks by priority (CPU scheduling, task queues) |
| Dijkstra | Priority queue for shortest path |
| Huffman Coding | Build optimal prefix codes |

## Time Complexity Table

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Insert | O(log n) | Sift up after append |
| Extract Min/Max | O(log n) | Sift down after swap |
| Peek | O(1) | Root element |
| Build Heap | O(n) | Bottom-up construction |
| Decrease/Increase Key | O(log n) | Sift up or down |
| Delete | O(log n) | Decrease to -inf then extract |
| Merge | O(n) | Concatenate and rebuild |

## Heap Invariants Summary

1. **Shape property**: Complete binary tree
2. **Heap property**: Parent dominates children (min or max)
3. **Array mapping**: Implicit parent-child relationships via index formulas
