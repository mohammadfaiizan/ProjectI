# Array Definition and Fundamentals

## Static vs Dynamic Arrays

**Static Array**: Fixed size determined at allocation. Cannot grow or shrink. Memory is allocated once.

**Dynamic Array**: Size can change at runtime. Internally reallocates when capacity is exceeded. Python lists are dynamic arrays.

## Contiguous Memory Layout

Arrays store elements in consecutive memory locations. If base address is B and each element occupies W bytes, element at index i is at address: B + (i * W). This enables:
- Predictable memory access patterns
- Efficient cache utilization (spatial locality)
- Simple pointer arithmetic

## Zero-Based Indexing

Indices start at 0. Element at index i is the (i+1)-th element. Benefits:
- Direct mapping: index i implies i elements before it
- Simpler loop bounds: range(n) covers indices 0 to n-1
- Consistent with pointer arithmetic in low-level languages

## Random Access O(1)

Given index i, accessing arr[i] requires one address computation and one memory read. No traversal needed. Time complexity: O(1).

## Cache-Friendliness

Contiguous layout means sequential access loads adjacent elements into cache lines. Iterating an array has excellent cache hit rates compared to linked structures where nodes may be scattered.

## When to Use Arrays vs Linked Lists vs Hash Maps

| Use Case | Array | Linked List | Hash Map |
|----------|-------|-------------|----------|
| Random access by index | O(1) - Best | O(n) | N/A |
| Insert/delete at end | O(1) amortized | O(1) | O(1) |
| Insert/delete at beginning | O(n) | O(1) | O(1) |
| Insert/delete at middle | O(n) | O(1) if node known | O(1) |
| Search by value | O(n) | O(n) | O(1) average |
| Search by key | N/A | N/A | O(1) average |
| Memory overhead | Minimal | Extra pointers | Hash table overhead |
| Order preservation | Yes | Yes | No (unordered) |

**Choose Array when**: Need random access, index-based operations, or ordered traversal. Good for fixed or predictable size.

**Choose Linked List when**: Frequent insertions/deletions at head or middle, unknown size, no random access needed.

**Choose Hash Map when**: Key-value lookups, membership tests, counting frequencies.

## Python List Internals

### Over-Allocation

Python lists allocate more memory than needed to avoid reallocation on every append. Growth pattern: when list is full, new capacity = (current_size * 3) // 2 + 1 (or similar). This amortizes reallocation cost.

### Dynamic Resizing

When append() exceeds capacity, Python allocates a new larger array, copies elements, and frees the old one. Amortized O(1) per append because reallocations are rare.

## Time Complexity Table

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Access by index | O(1) | Random access |
| Search (linear) | O(n) | Must scan elements |
| Insert at end | O(1) amortized | May trigger reallocation |
| Insert at beginning | O(n) | Shift all elements |
| Insert at index | O(n) | Shift elements from index onward |
| Delete from end | O(1) | Pop last |
| Delete from beginning | O(n) | Shift remaining elements |
| Delete at index | O(n) | Shift elements |
| Append | O(1) amortized | Same as insert at end |

## Space Complexity

- Array of n elements: O(n) for storing elements
- Auxiliary space for operations: typically O(1) for in-place, O(n) for algorithms requiring copies
