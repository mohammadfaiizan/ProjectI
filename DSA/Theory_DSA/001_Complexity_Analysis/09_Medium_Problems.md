# Medium Complexity Analysis Problems

## Problem 1

```python
def nested_varying_bounds(n):
    count = 0
    for i in range(n):
        for j in range(i, n):
            count += 1
    return count
```

What is the time complexity?

**Answer**: O(n^2). Inner loop runs (n - i) times for i = 0, 1, ..., n-1. Total iterations: n + (n-1) + ... + 1 = n(n+1)/2 = Theta(n^2).

---

## Problem 2

```python
def triple_nested(n):
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                count += 1
    return count
```

What is the time complexity?

**Answer**: O(n^3). Three nested loops, each running n times. Total: n * n * n = n^3 iterations.

---

## Problem 3

```python
def recursive_power(base, exp):
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = recursive_power(base, exp // 2)
        return half * half
    return base * recursive_power(base, exp - 1)
```

What is the time complexity in terms of exp?

**Answer**: O(log exp). When exp is even, we recurse on exp/2. When odd, we do one multiplication and recurse on exp-1 (which becomes even). The number of recursive calls is O(log exp). Each call does O(1) arithmetic. Total: O(log exp).

---

## Problem 4

```python
def process_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    for i in range(rows):
        for j in range(cols):
            matrix[i][j] *= 2
```

What is the time complexity?

**Answer**: O(rows * cols) or O(n * m). Two nested loops over dimensions of the matrix. Each cell is visited once with O(1) work.

---

## Problem 5

```python
def merge_sorted(a, b):
    result = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
```

What is the time complexity? Let n = len(a), m = len(b).

**Answer**: O(n + m). Each element from both arrays is appended to result exactly once. The extend operations copy remaining elements. Total: O(n + m).

---

## Problem 6

```python
def build_prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix
```

What is the space complexity?

**Answer**: O(n). The prefix array has n+1 elements. Auxiliary space is Theta(n). Input is O(n), so total is O(n) but we report auxiliary as O(n).

---

## Problem 7

```python
def amortized_append_sequence(n):
    arr = []
    for i in range(n):
        arr.append(i)
    return arr
```

What is the amortized time per append? What is the total time for n appends?

**Answer**: Amortized O(1) per append. Total O(n). Dynamic array doubles on resize. Resizes occur at sizes 1, 2, 4, 8, ... Total copy cost is O(n). Plus n appends. Total O(n), so amortized O(1) per operation.

---

## Problem 8

```python
def recursive_divide(n):
    if n <= 1:
        return 1
    return recursive_divide(n // 2) + recursive_divide(n // 2) + n
```

What is the recurrence and solution?

**Answer**: T(n) = 2T(n/2) + n. By Master Theorem: a=2, b=2, f(n)=n = Theta(n^1) = Theta(n^{log_b a}). Case 2. T(n) = Theta(n log n).

---

## Problem 9

```python
def uneven_split(n):
    if n <= 1:
        return 1
    return uneven_split(n // 3) + uneven_split(2 * n // 3) + n
```

What is the time complexity?

**Answer**: O(n log n). Recursion tree: each level does O(n) work. The depth is determined by the longer branch: (2/3)^k * n = 1 gives k = log_{3/2} n. Total: O(n) * O(log n) = O(n log n). Can also use Akra-Bazzi.

---

## Problem 10

```python
def nested_with_inner_condition(arr):
    count = 0
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i] == arr[j]:
                count += 1
    return count
```

What is the time complexity?

**Answer**: O(n^2). Two nested loops over n elements. The condition and increment are O(1). Regardless of how often the condition is true, we always do n^2 iterations of the loop structure.

---

## Problem 11

```python
def binary_search_recursive(arr, target, lo, hi):
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, hi)
    return binary_search_recursive(arr, target, lo, mid - 1)
```

What is the space complexity?

**Answer**: O(log n). The recursion depth is the number of times we halve the range: log2(n) levels. Each frame uses O(1) space. Total: O(log n).

---

## Problem 12

```python
def process_pairs(arr):
    seen = set()
    for x in arr:
        for y in arr:
            if (x, y) not in seen:
                seen.add((x, y))
```

What is the time complexity? What is the space complexity?

**Answer**: Time O(n^2). Inner loop runs n times for each of n outer iterations. Set lookup and add are O(1) average. Space O(n^2) in worst case when all pairs are distinct.

---

## Problem 13

```python
def strassen_style(n):
    if n <= 1:
        return 1
    return 7 * strassen_style(n // 2) + n * n
```

What is the solution? (Assume n is a power of 2.)

**Answer**: T(n) = 7T(n/2) + n^2. Master Theorem: a=7, b=2, log_b a = log_2 7 approx 2.81. f(n)=n^2 = O(n^2.81 - epsilon). Case 1. T(n) = Theta(n^{log_2 7}).

---

## Problem 14

```python
def hash_table_ops(keys):
    table = {}
    for k in keys:
        table[k] = table.get(k, 0) + 1
    return table
```

What is the time complexity? Let n = len(keys).

**Answer**: O(n) average. n iterations. Each dict get and set is O(1) average. Total: O(n). Worst case with many collisions: O(n^2), but typical hash tables avoid this.

---

## Problem 15

```python
def substring_search(s, sub):
    n, m = len(s), len(sub)
    for i in range(n - m + 1):
        if s[i:i+m] == sub:
            return i
    return -1
```

What is the time complexity?

**Answer**: O(n * m). Outer loop: n - m + 1 = O(n) iterations. Each slice s[i:i+m] creates a new string of length m: O(m). Comparison is O(m). Total: O(n * m).

---

## Problem 16

```python
def recursive_fib(n):
    if n <= 1:
        return n
    return recursive_fib(n - 1) + recursive_fib(n - 2)
```

What is the time complexity? What is the space complexity?

**Answer**: Time O(2^n). The recursion tree has roughly 2^n nodes (each call branches into two, depth n). Space O(n) for the maximum depth of the call stack.

---

## Problem 17

```python
def sort_then_scan(arr):
    arr.sort()
    for i in range(len(arr) - 1):
        if arr[i] == arr[i + 1]:
            return True
    return False
```

What is the time complexity?

**Answer**: O(n log n). Sort is O(n log n). The scan is O(n). Dominated by sort: O(n log n).

---

## Problem 18

```python
def nested_log(n):
    count = 0
    i = 1
    while i < n:
        j = 1
        while j < n:
            count += 1
            j *= 2
        i *= 2
    return count
```

What is the time complexity?

**Answer**: O((log n)^2). Outer loop: i = 1, 2, 4, ..., n. Runs log n times. Inner loop: j = 1, 2, 4, ..., n. Runs log n times per outer iteration. Total: (log n) * (log n) = O((log n)^2).

---

## Problem 19

```python
def process_graph(adj_list):
    visited = set()
    for v in adj_list:
        if v not in visited:
            dfs(v, adj_list, visited)
```

Assume DFS visits each vertex and edge once. What is the time complexity? Let V = vertices, E = edges.

**Answer**: O(V + E). Each vertex is processed once. Each edge is traversed once (in undirected) or once per direction (in directed). Total: O(V + E).

---

## Problem 20

```python
def dynamic_programming(n):
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

What is the time and space complexity?

**Answer**: Time O(n). Space O(n). Single loop over n values. Each iteration does O(1) work. The dp array uses O(n) space.
