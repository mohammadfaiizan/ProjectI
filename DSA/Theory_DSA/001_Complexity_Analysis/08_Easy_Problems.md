# Easy Complexity Analysis Problems

## Problem 1

What is the time complexity of iterating through every element in an array of size n?

**Answer**: O(n). Each element is visited exactly once. Single loop over n elements with O(1) work per iteration.

---

## Problem 2

What is the time complexity of accessing the element at index i in an array?

**Answer**: O(1). Array indexing is a constant-time operation. The memory address is computed as base + i * element_size.

---

## Problem 3

```python
def foo(n):
    for i in range(n):
        print(i)
```

What is the time complexity?

**Answer**: O(n). The loop runs n times. Each iteration does O(1) work (print). Total: O(n).

---

## Problem 4

```python
def bar(n):
    x = 0
    for i in range(10):
        x += i
    return x
```

What is the time complexity?

**Answer**: O(1). The loop runs a fixed 10 times regardless of n. The parameter n is unused. Constant number of operations.

---

## Problem 5

What is the best-case time complexity of linear search in an array of n elements?

**Answer**: O(1). Best case occurs when the target is at the first index. One comparison and return.

---

## Problem 6

What is the worst-case time complexity of linear search?

**Answer**: O(n). Worst case occurs when the target is not in the array or at the last index. All n elements must be checked.

---

## Problem 7

```python
def baz(arr):
    return arr[0] + arr[-1]
```

What is the time complexity?

**Answer**: O(1). Two array accesses and one addition. Constant time regardless of array length.

---

## Problem 8

```python
def sum_first_k(arr, k):
    total = 0
    for i in range(min(k, len(arr))):
        total += arr[i]
    return total
```

What is the time complexity when k is a constant (e.g., k=5)?

**Answer**: O(1). The loop runs at most min(k, n) times. When k is constant, this is O(1). When k is a variable, it would be O(min(k, n)).

---

## Problem 9

What is the time complexity of binary search on a sorted array of n elements?

**Answer**: O(log n). Each step halves the search space. After k steps, remaining size is n/2^k. When n/2^k = 1, k = log n.

---

## Problem 10

```python
def double_loop(n):
    count = 0
    for i in range(n):
        for j in range(n):
            count += 1
    return count
```

What is the time complexity?

**Answer**: O(n^2). Inner loop runs n times for each of n outer iterations. Total: n * n = n^2 iterations.

---

## Problem 11

```python
def consecutive_loops(n):
    for i in range(n):
        pass
    for j in range(n):
        pass
```

What is the time complexity?

**Answer**: O(n). Two sequential loops, each O(n). O(n) + O(n) = O(n).

---

## Problem 12

What is the time complexity of finding the maximum element in an unsorted array of n elements?

**Answer**: O(n). Must examine every element at least once to guarantee correctness. Single pass with O(1) work per element.

---

## Problem 13

```python
def mystery(n):
    if n <= 0:
        return 0
    return 1 + mystery(n - 1)
```

What is the time complexity?

**Answer**: O(n). The function makes n recursive calls (n, n-1, ..., 1). Each call does O(1) work. Total: O(n).

---

## Problem 14

What is the space complexity of the recursive function in Problem 13?

**Answer**: O(n). The call stack has depth n. Each frame uses O(1) space. Total: O(n).

---

## Problem 15

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

What is the time complexity?

**Answer**: O(n). n recursive calls, each with O(1) work. Total: O(n).

---

## Problem 16

What is the time complexity of appending an element to a Python list?

**Answer**: O(1) amortized. Most appends are O(1). Occasionally a resize occurs (O(n)), but over n appends the amortized cost is O(1) per append.

---

## Problem 17

```python
def check_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True
```

What is the best-case time complexity?

**Answer**: O(1). Best case: first two elements are out of order (arr[0] > arr[1]). One comparison and return False.

---

## Problem 18

What is the worst-case time complexity of the function in Problem 17?

**Answer**: O(n). Worst case: array is sorted. All n-1 comparisons are performed. O(n).

---

## Problem 19

```python
def count_zeros(arr):
    return arr.count(0)
```

What is the time complexity? (Assume arr is a list.)

**Answer**: O(n). The count method scans the entire list to count occurrences. Single pass over n elements.

---

## Problem 20

What is the time complexity of checking if a value exists in a Python set of n elements?

**Answer**: O(1) average. Hash-based lookup. Worst case O(n) with many collisions, but average case is constant.
