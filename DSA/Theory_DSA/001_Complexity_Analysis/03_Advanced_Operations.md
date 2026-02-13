# Multi-Variable and Advanced Complexity

## Multi-Variable Complexity O(n*m)

When input has multiple dimensions, complexity is expressed in terms of all relevant variables.

```python
def matrix_traversal(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    for i in range(rows):
        for j in range(cols):
            process(matrix[i][j])
```

- Time: O(rows * cols) or O(n * m)
- For a graph with V vertices and E edges: O(V + E) for adjacency list traversal

```python
def common_elements(arr1, arr2):
    set2 = set(arr2)
    result = []
    for x in arr1:
        if x in set2:
            result.append(x)
    return result
```

- Building set: O(m) where m = len(arr2)
- Loop over arr1: O(n) iterations, each `in` check O(1)
- Total: O(n + m)

## Hidden Costs

**String concatenation**:

```python
def bad_concat(strings):
    result = ""
    for s in strings:
        result += s
    return result
```

- Each `+=` may copy the entire result string (strings are immutable in Python)
- Total: O(n^2) for n total characters

```python
def good_concat(strings):
    return "".join(strings)
```

- join allocates once and copies: O(n)

**List copying**:

```python
def copy_and_append(arr, x):
    new_arr = arr[:]
    new_arr.append(x)
    return new_arr
```

- Slicing arr[:] copies all elements: O(n)

**List slicing**:

```python
def process_slice(arr, i, j):
    return sum(arr[i:j])
```

- arr[i:j] creates a new list of length j-i: O(j-i) space and time for the slice

## Python Built-in Operations Reference Table

| Operation | Time | Notes |
|-----------|------|-------|
| list[i] | O(1) | Indexing |
| list.append(x) | O(1) amortized | Amortized due to occasional resize |
| list.insert(0, x) | O(n) | Shifts all elements |
| list.pop() | O(1) | Remove from end |
| list.pop(0) | O(n) | Shifts all elements |
| list[i:j] | O(j-i) | Creates new list |
| x in list | O(n) | Linear search |
| list.sort() | O(n log n) | Timsort |
| dict[key] | O(1) average | Hash lookup |
| key in dict | O(1) average | Hash lookup |
| dict[key] = val | O(1) average | Hash insert |
| x in set | O(1) average | Hash lookup |
| set.add(x) | O(1) average | Hash insert |
| s[i] | O(1) | Indexing |
| s1 + s2 | O(len(s1)+len(s2)) | Concatenation creates new string |
| x in s | O(n) | Linear search |
| s.find(sub) | O(n*m) worst | n=len(s), m=len(sub) |
| s.split() | O(n) | Single pass |

## Space-Time Tradeoffs

**Caching**: Store computed results to avoid recomputation. Increases space, decreases time.

```python
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

- Time: O(n), Space: O(n) for memo dict and call stack

**Lookup tables**: Precompute and store for O(1) access.

```python
FACTORIAL_TABLE = [1]
for i in range(1, 101):
    FACTORIAL_TABLE.append(FACTORIAL_TABLE[-1] * i)
```

- Precomputation: O(n), lookup: O(1)

**Hash set for O(1) lookup**:

```python
def two_sum(arr, target):
    seen = set()
    for x in arr:
        if target - x in seen:
            return True
        seen.add(x)
    return False
```

- O(n) time, O(n) space vs O(n^2) brute force

## NP-Completeness Overview

**P (Polynomial time)**: Problems solvable in O(n^k) for some constant k. Examples: sorting, shortest path, linear programming.

**NP (Nondeterministic Polynomial)**: Problems whose solutions can be verified in polynomial time. P is a subset of NP. Examples: Hamiltonian path, SAT, vertex cover.

**NP-hard**: A problem H is NP-hard if every problem in NP can be reduced to H in polynomial time. NP-hard problems are at least as hard as the hardest problems in NP. They may not be in NP (e.g., halting problem).

**NP-complete**: A problem is NP-complete if it is in NP and NP-hard. NP-complete problems are the hardest problems in NP. If any NP-complete problem has a polynomial-time algorithm, then P = NP.

Examples of NP-complete problems: Boolean satisfiability (SAT), clique, graph coloring, traveling salesman (decision version), subset sum, knapsack (decision version).

## Complexity Classes Summary

| Class | Definition | Examples |
|-------|------------|----------|
| P | Solvable in polynomial time | Sorting, Dijkstra, GCD |
| NP | Verifiable in polynomial time | SAT, Hamiltonian path |
| NP-hard | At least as hard as NP | SAT, TSP, halting |
| NP-complete | In NP and NP-hard | SAT, clique, vertex cover |
| EXPTIME | Solvable in exponential time | Chess (generalized) |
| PSPACE | Solvable with polynomial space | QBF, geography |
