# Space Complexity

Space complexity measures the amount of memory an algorithm uses as a function of input size. It includes the space for the algorithm's variables, data structures, and call stack.

## Auxiliary Space vs Total Space

**Auxiliary space**: Extra space used by the algorithm beyond the input. This is what we typically report.

**Total space**: Input space + auxiliary space. For in-place algorithms, total space equals input space.

When we say "space complexity O(1)", we usually mean O(1) auxiliary space. The input itself may take O(n) space, but we do not count it as part of the algorithm's space usage unless we modify it in place.

## In-Place Algorithms

Algorithms that use only O(1) auxiliary space. They modify the input structure without allocating significant extra memory.

**Reverse array**:

```python
def reverse_in_place(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
```

- Auxiliary space: O(1) (only a few variables)
- Modifies input in place

**Dutch National Flag**:

```python
def dutch_national_flag(arr):
    low, mid, high = 0, 0, len(arr) - 1
    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
```

- Auxiliary space: O(1)
- Three pointers, no extra arrays

## Stack Space in Recursion

Each recursive call pushes a new stack frame. Space complexity = maximum depth of the call stack * space per frame.

**Factorial**:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

- Depth: n
- Space per frame: O(1)
- Total stack space: O(n)

**Binary search**:

```python
def binary_search(arr, target, lo, hi):
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search(arr, target, mid + 1, hi)
    return binary_search(arr, target, lo, mid - 1)
```

- Depth: O(log n)
- Space: O(log n)

## Recursive vs Iterative Space

**Fibonacci recursive**:

```python
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)
```

- Stack depth: O(n) (along longest path)
- Space: O(n)

**Fibonacci iterative**:

```python
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

- Space: O(1)

Iterative often uses less space by avoiding the call stack.

## Tail Call Optimization

A tail call is a recursive call that is the last operation in a function. Some languages (e.g., Scheme, Scala) optimize tail calls by reusing the current stack frame instead of pushing a new one. This reduces space from O(n) to O(1) for tail-recursive functions.

Python does not perform tail call optimization. A tail-recursive function in Python still uses O(n) stack space.

```python
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, n * acc)
```

- In a language with TCO: O(1) space
- In Python: O(n) space

## Space-Time Tradeoffs

**Caching / Memoization**: Store computed results to avoid recomputation.

```python
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

- Time: O(n)
- Space: O(n) for memo dict + O(n) stack = O(n)

**Lookup tables**: Precompute and store for O(1) access.

```python
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_small(n):
    return n in PRIMES
```

- Space: O(k) for k precomputed values
- Time: O(1) with set, O(k) with list

**Hash set for O(1) lookup**:

```python
def has_duplicate(arr):
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False
```

- Space: O(n) for set
- Time: O(n) vs O(n^2) brute force

## Comparing Space Usage

**Problem**: Find two elements that sum to target.

**Brute force**: O(1) space, O(n^2) time.

**Hash map**: O(n) space, O(n) time.

**Sort + two pointers**: O(1) auxiliary if sort in place, O(log n) for sort stack; O(n log n) time.

**Problem**: Compute nth Fibonacci number.

**Naive recursion**: O(n) stack space, O(2^n) time.

**Memoization**: O(n) space, O(n) time.

**Iterative**: O(1) space, O(n) time.

**Matrix exponentiation**: O(1) space, O(log n) time.
