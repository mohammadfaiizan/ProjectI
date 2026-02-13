# How to Count Operations in Code

Operation counting is the foundation of complexity analysis. Each primitive operation (comparison, arithmetic, assignment, function call with known cost) contributes to the total. The goal is to express the total count as a function of input size n and simplify to asymptotic notation.

## Constant Time O(1)

Operations that execute a fixed number of steps regardless of input size.

```python
def get_first(arr):
    return arr[0]

def swap(a, b):
    a, b = b, a
    return a, b
```

Array indexing, arithmetic, variable assignment, and simple conditionals are typically O(1). The key is that the number of operations does not depend on n.

## Single Loop Analysis O(n)

A loop that runs n times, with O(1) work per iteration, yields O(n).

```python
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total
```

- Loop executes n times (n = len(arr))
- Each iteration: one addition, one assignment
- Total: O(n) operations

```python
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
```

- Loop runs n - 1 times, asymptotically O(n)
- Per iteration: one comparison, possibly one assignment
- Total: O(n)

## Nested Loop Analysis O(n^2) and O(n^3)

Two nested loops over the same range produce O(n^2).

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

- Outer loop: n iterations
- Inner loop: (n-1) + (n-2) + ... + 1 = n(n-1)/2 iterations
- Total: O(n^2)

```python
def matrix_multiply(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
```

- Three nested loops, each O(n)
- Total: O(n^3)

## Consecutive Loops

When loops are sequential (not nested), add their complexities.

```python
def process(arr):
    for x in arr:
        print(x)
    for x in arr:
        print(x * 2)
```

- First loop: O(n)
- Second loop: O(n)
- Total: O(n) + O(n) = O(n)

## Linear Time O(n)

Any algorithm that touches each input element a constant number of times is O(n).

```python
def linear_search(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1
```

## Quadratic Time O(n^2)

Common in comparison-based sorting, all-pairs algorithms, and naive string matching.

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
```

## Logarithmic Time O(log n)

Occurs when the problem size is halved (or reduced by a constant factor) each step.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

- Each iteration halves the search space
- Maximum iterations: ceil(log2(n)) + 1
- Total: O(log n)

## Simple Recursion Analysis

**Factorial O(n)**:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

- n recursive calls, each doing O(1) work
- Total: O(n)

**Fibonacci O(2^n)**:

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

- Recursion tree has roughly 2^n nodes (many redundant)
- Total: O(2^n) time

## Comparing Two Algorithms

Given Algorithm A: O(n^2) and Algorithm B: O(n log n):

- For small n, A might be faster due to lower constant factors
- For large n, B dominates
- Crossover point: solve n^2 = c * n log n for n; typically n is in hundreds or thousands depending on c

## Step-by-Step Code Snippet Analysis

**Example 1**:

```python
def mystery(arr):
    n = len(arr)
    result = 0
    for i in range(n):
        for j in range(i, n):
            result += arr[j]
    return result
```

- Outer loop: i = 0 to n-1
- Inner loop: for i=0 runs n times, i=1 runs n-1 times, ..., i=n-1 runs 1 time
- Total inner iterations: n + (n-1) + ... + 1 = n(n+1)/2 = O(n^2)
- Each inner iteration: O(1)
- Total: O(n^2)

**Example 2**:

```python
def example(arr):
    n = len(arr)
    for i in range(n):
        if arr[i] % 2 == 0:
            for j in range(n):
                print(arr[i] + arr[j])
```

- Worst case: all elements even, inner loop always runs
- Outer: n iterations
- Inner: n iterations per outer
- Total: O(n^2)
- Best case: all elements odd, inner loop never runs: O(n)

**Example 3**:

```python
def mixed(arr):
    arr.sort()
    for x in arr:
        if x > 0:
            print(x)
```

- sort(): O(n log n)
- Loop: O(n)
- Total: O(n log n) + O(n) = O(n log n)
