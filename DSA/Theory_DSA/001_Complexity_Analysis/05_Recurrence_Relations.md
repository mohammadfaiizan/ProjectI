# Recurrence Relations

A recurrence relation defines a function in terms of its value on smaller inputs. Recurrences arise naturally when analyzing recursive algorithms.

## What is a Recurrence Relation

A recurrence has the form T(n) = (expression involving T(n-1), T(n-2), ..., T(n/k), etc.) + (non-recursive work).

Example: T(n) = 2T(n/2) + n describes merge sort: split into two halves, solve each (recursively), merge in O(n).

## How to Write Recurrence from Code

**Merge sort**:

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

- Two recursive calls on n/2 elements each
- Merge is O(n)
- T(n) = 2T(n/2) + Theta(n)

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

- One recursive call on half the array
- O(1) work per call
- T(n) = T(n/2) + Theta(1)

## Substitution Method

Guess the form of the solution and prove it by induction.

**Example**: T(n) = 2T(n/2) + n, T(1) = 1. Guess T(n) = O(n log n).

Assume T(k) <= c * k log k for k < n. Then:

T(n) = 2T(n/2) + n <= 2 * c * (n/2) * log(n/2) + n = cn log(n/2) + n = cn log n - cn + n

We need cn log n - cn + n <= cn log n, i.e., -cn + n <= 0, so c >= 1. Choose c = 1. Base case T(1) = 1 <= 1*0 = 0 fails. Adjust: for n >= 2, T(n) <= n log n works with c = 1.

## Recursion Tree Method

Draw a tree where each node represents the cost at that level. Sum costs level by level.

**Example**: T(n) = 2T(n/2) + n

```
Level 0: n
Level 1: n/2 + n/2 = n
Level 2: n/4 + n/4 + n/4 + n/4 = n
...
Level log n: n leaves (base cases)
```

- log n levels, each costing n
- Total: n * log n = Theta(n log n)

**Example**: T(n) = T(n/3) + T(2n/3) + n

- Tree is unbalanced but each level sums to at most n
- Longest path: n -> 2n/3 -> (2/3)^2 n -> ... -> 1. Depth = log_{3/2} n
- Total: O(n log n)

## Master Theorem

For recurrences of the form T(n) = aT(n/b) + f(n) where a >= 1, b > 1:

Compare f(n) with n^(log_b a):

**Case 1**: If f(n) = O(n^{log_b a - epsilon}) for some epsilon > 0, then T(n) = Theta(n^{log_b a}).

**Case 2**: If f(n) = Theta(n^{log_b a}), then T(n) = Theta(n^{log_b a} * log n).

**Case 3**: If f(n) = Omega(n^{log_b a + epsilon}) for some epsilon > 0, and a*f(n/b) <= c*f(n) for some c < 1 (regularity), then T(n) = Theta(f(n)).

**Merge sort**: T(n) = 2T(n/2) + n. a=2, b=2, log_b a = 1. f(n)=n = Theta(n^1). Case 2. T(n) = Theta(n log n).

**Binary search**: T(n) = T(n/2) + 1. a=1, b=2, log_b a = 0. f(n)=1 = Theta(n^0). Case 2. T(n) = Theta(log n).

**Strassen's matrix multiplication**: T(n) = 7T(n/2) + Theta(n^2). a=7, b=2, log_b a = log_2 7 approx 2.81. f(n)=n^2 = O(n^2.81 - epsilon). Case 1. T(n) = Theta(n^{log_2 7}).

## Extended Master Theorem

For T(n) = aT(n/b) + Theta(n^k * log^p n):

- If log_b a > k: T(n) = Theta(n^{log_b a})
- If log_b a = k: T(n) = Theta(n^k * log^{p+1} n)
- If log_b a < k: T(n) = Theta(n^k * log^p n)

## Akra-Bazzi Method Overview

For recurrences of the form T(n) = sum a_i * T(b_i * n + h_i(n)) + g(n) where 0 < b_i < 1:

Find p such that sum a_i * b_i^p = 1. Then T(n) = Theta(n^p * (1 + integral_1^n g(u)/u^{p+1} du)).

Handles non-standard splits (e.g., T(n) = T(n/3) + T(2n/3) + n) and floor/ceiling effects.

## Common Recurrences Table

| Algorithm | Recurrence | Solution |
|-----------|------------|----------|
| Merge sort | T(n) = 2T(n/2) + Theta(n) | Theta(n log n) |
| Quick sort (avg) | T(n) = 2T(n/2) + Theta(n) | Theta(n log n) |
| Quick sort (worst) | T(n) = T(n-1) + Theta(n) | Theta(n^2) |
| Binary search | T(n) = T(n/2) + Theta(1) | Theta(log n) |
| Fibonacci (naive) | T(n) = T(n-1) + T(n-2) + Theta(1) | Theta(2^n) |
| Tower of Hanoi | T(n) = 2T(n-1) + Theta(1) | Theta(2^n) |
| Karatsuba | T(n) = 3T(n/2) + Theta(n) | Theta(n^{log_2 3}) |
| Strassen | T(n) = 7T(n/2) + Theta(n^2) | Theta(n^{log_2 7}) |
