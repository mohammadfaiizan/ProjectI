# C++ Data Structures and Algorithms - Complete Guide

This directory contains comprehensive C++ implementations of fundamental data structures and algorithms, designed for interview preparation and learning purposes.

## Files Overview

### 1. Array and Vector Operations
**File**: `01_array_complete.cpp` (13KB, 444 lines)
- Multiple initialization methods for arrays and vectors
- All vector operations (push_back, insert, erase, etc.)
- STL algorithms (sort, find, binary_search, etc.)
- 2D array operations and matrix handling
- Common patterns: two pointers, sliding window
- Performance considerations and memory management

### 2. String Operations and Algorithms
**File**: `02_string_complete.cpp` (26KB, 930 lines)
- 10 different string initialization methods
- String access, modification, and manipulation
- Search operations (find, rfind, find_first_of, etc.)
- String comparison and sorting techniques
- Splitting and joining strings
- Advanced algorithms: palindrome, anagram, pattern matching
- KMP algorithm, longest common subsequence, edit distance
- Regular expressions and performance optimization

### 3. Linked List Implementations
**File**: `03_linked_list_complete.cpp` (29KB, 1033 lines)
- Singly, doubly, and circular linked lists
- Multiple insertion methods (beginning, end, position)
- All deletion techniques
- Traversal patterns (forward, backward, recursive)
- Advanced operations: reversal, merging, cycle detection
- Memory management and best practices

### 4. Stack Implementations
**File**: `04_stack_complete.cpp` (24KB, 812 lines)
- Array-based and linked-list based stacks
- STL stack operations
- Applications: balanced parentheses, expression evaluation
- Advanced patterns: monotonic stack, next greater element
- Memory management and performance considerations

### 5. Queue Implementations
**File**: `05_queue_complete.cpp` (28KB, 956 lines)
- Array-based, linked-list based, and circular queues
- Priority queue implementations
- Deque operations
- Applications: BFS, sliding window maximum
- Performance analysis and use cases

### 6. Tree Operations
**File**: `06_tree_complete.cpp` (28KB, 961 lines)
- Binary tree and binary search tree implementations
- All traversal methods (preorder, inorder, postorder)
- Both recursive and iterative approaches
- Tree construction from traversals
- Advanced algorithms: LCA, diameter, path sum
- AVL tree basics and balancing concepts

### 7. Hash Table Operations
**File**: `07_hash_complete.cpp` (30KB, 980 lines)
- Hash table implementation with collision handling
- Linear probing, quadratic probing, and chaining
- STL containers: unordered_map, unordered_set
- Hash functions and load factor management
- Applications: two sum, LRU cache, word frequency

### 8. Graph Algorithms
**File**: `08_graph_complete.cpp` (28KB, 910 lines)
- Multiple graph representations (adjacency matrix, list, edge list)
- Traversal algorithms: DFS, BFS
- Shortest path: Dijkstra, Bellman-Ford, Floyd-Warshall
- MST algorithms: Kruskal, Prim
- Topological sorting and strongly connected components
- Union-Find data structure

### 9. Heap Operations
**File**: `09_heap_complete.cpp` (28KB, 973 lines)
- Min-heap and max-heap implementations
- Generic heap with custom comparators
- STL priority_queue operations
- Heap sort implementation
- Applications: running median, top K elements
- Memory management and performance optimization

## Compilation Instructions

### Basic Compilation
```bash
g++ -std=c++17 -o program_name file_name.cpp
```

### With Debugging
```bash
g++ -std=c++17 -g -Wall -Wextra -o program_name file_name.cpp
```

### Optimized Build
```bash
g++ -std=c++17 -O2 -o program_name file_name.cpp
```

### Example
```bash
# Compile and run array operations
g++ -std=c++17 -o array_demo 01_array_complete.cpp
./array_demo

# Compile and run string operations
g++ -std=c++17 -o string_demo 02_string_complete.cpp
./string_demo
```

## Learning Path

### Beginner Level
1. **Arrays** (`01_array_complete.cpp`) - Start with basic operations
2. **Strings** (`02_string_complete.cpp`) - Learn string manipulation
3. **Linked Lists** (`03_linked_list_complete.cpp`) - Understand pointers and references

### Intermediate Level
4. **Stacks** (`04_stack_complete.cpp`) - Learn LIFO principles
5. **Queues** (`05_queue_complete.cpp`) - Understand FIFO and priority concepts
6. **Trees** (`06_tree_complete.cpp`) - Master hierarchical structures

### Advanced Level
7. **Hash Tables** (`07_hash_complete.cpp`) - Optimize lookup operations
8. **Graphs** (`08_graph_complete.cpp`) - Solve complex relationship problems
9. **Heaps** (`09_heap_complete.cpp`) - Master priority-based operations

## Key Features

- **Comprehensive Coverage**: Each file covers multiple implementation approaches
- **Beginner-Friendly**: Detailed comments and step-by-step explanations
- **Real-World Applications**: Practical examples and use cases
- **Performance Analysis**: Time and space complexity discussions
- **Modern C++**: Uses C++17 features and best practices
- **Memory Management**: Proper handling of dynamic memory
- **STL Integration**: Shows both custom implementations and STL usage

## Time Complexity Reference

| Operation | Array | Linked List | Stack | Queue | BST | Hash Table | Heap |
|-----------|-------|-------------|-------|-------|-----|------------|------|
| Access    | O(1)  | O(n)        | O(n)  | O(n)  | O(log n) | O(1) avg | O(n) |
| Search    | O(n)  | O(n)        | O(n)  | O(n)  | O(log n) | O(1) avg | O(n) |
| Insert    | O(n)  | O(1)        | O(1)  | O(1)  | O(log n) | O(1) avg | O(log n) |
| Delete    | O(n)  | O(1)        | O(1)  | O(1)  | O(log n) | O(1) avg | O(log n) |

## Space Complexity Reference

| Data Structure | Space Complexity |
|---------------|------------------|
| Array         | O(n)            |
| Linked List   | O(n)            |
| Stack         | O(n)            |
| Queue         | O(n)            |
| Binary Tree   | O(n)            |
| Hash Table    | O(n)            |
| Heap          | O(n)            |
| Graph         | O(V + E)        |

## Usage Tips

1. **Start with main()**: Each file has a comprehensive main() function demonstrating all features
2. **Read Comments**: Detailed explanations for each algorithm and technique
3. **Modify and Experiment**: Try different inputs and observe behavior
4. **Time Your Code**: Use chrono library to measure performance
5. **Memory Profiling**: Use tools like valgrind to check for memory leaks
6. **Practice**: Implement variations and solve related problems

## Additional Resources

- **Complexity Analysis**: Each file includes time and space complexity analysis
- **Best Practices**: Modern C++ coding standards and conventions
- **Interview Questions**: Common patterns and problems for technical interviews
- **Performance Tips**: Optimization techniques and memory management

## Prerequisites

- **C++ Compiler**: GCC 7.0+ or Clang 5.0+ with C++17 support
- **Basic C++ Knowledge**: Understanding of pointers, references, and OOP concepts
- **STL Familiarity**: Knowledge of standard template library is helpful

## Contributing

Feel free to:
- Add more examples and test cases
- Improve documentation and comments
- Optimize existing algorithms
- Add new data structures and algorithms
- Fix bugs and improve code quality

---

**Total Code**: 200+ KB of comprehensive C++ educational content  
**Files**: 9 complete implementations  
**Lines of Code**: 6000+ lines with detailed comments  
**Learning Hours**: 50+ hours of structured learning material 