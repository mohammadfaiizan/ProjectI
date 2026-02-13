# Basic Recursion Operations

## Factorial

**Iterative**:
```python
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

**Recursive**:
```python
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)
```

## Fibonacci

**Naive recursive** (O(2^n)):
```python
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)
```

**Memoized** (O(n)):
```python
def fib_memoized(n, memo=None):
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib_memoized(n - 1, memo) + fib_memoized(n - 2, memo)
    return memo[n]
```

## Power / Exponentiation

**Naive** (O(n)):
```python
def power_naive(base, exp):
    if exp == 0:
        return 1
    return base * power_naive(base, exp - 1)
```

**Fast power** (O(log n)):
```python
def power_fast(base, exp):
    if exp == 0:
        return 1
    half = power_fast(base, exp // 2)
    if exp % 2 == 0:
        return half * half
    return base * half * half
```

## Sum of Digits

```python
def sum_of_digits(n):
    if n < 10:
        return n
    return n % 10 + sum_of_digits(n // 10)
```

## Reverse a String Recursively

```python
def reverse_string(s):
    if len(s) <= 1:
        return s
    return reverse_string(s[1:]) + s[0]
```

## Check Palindrome Recursively

```python
def is_palindrome(s, left=0, right=None):
    if right is None:
        right = len(s) - 1
    if left >= right:
        return True
    if s[left] != s[right]:
        return False
    return is_palindrome(s, left + 1, right - 1)
```

## Print 1 to N

```python
def print_1_to_n(n):
    if n < 1:
        return
    print_1_to_n(n - 1)
    print(n)
```

## Print N to 1

```python
def print_n_to_1(n):
    if n < 1:
        return
    print(n)
    print_n_to_1(n - 1)
```

## Sum of Array

```python
def sum_of_array(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + sum_of_array(arr, index + 1)
```

## Find Max in Array

```python
def find_max(arr, index=0):
    if index >= len(arr):
        return float('-inf')
    return max(arr[index], find_max(arr, index + 1))
```

## Count Occurrences

```python
def count_occurrences(arr, target, index=0):
    if index >= len(arr):
        return 0
    add = 1 if arr[index] == target else 0
    return add + count_occurrences(arr, target, index + 1)
```

## Binary Search Recursively

```python
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    return binary_search(arr, target, left, mid - 1)
```

## Tower of Hanoi

**Theory**: Move n disks from source A to destination C using auxiliary B. Rules: only one disk at a time; never place larger disk on smaller. Minimum moves = 2^n - 1.

**Implementation**:
```python
def tower_of_hanoi(n, source, auxiliary, destination):
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    tower_of_hanoi(n - 1, source, destination, auxiliary)
    print(f"Move disk {n} from {source} to {destination}")
    tower_of_hanoi(n - 1, auxiliary, source, destination)
```

**Count moves**:
```python
def hanoi_moves(n):
    return (1 << n) - 1
```

## Print All Subsequences of String

```python
def print_subsequences(s, current="", index=0):
    if index == len(s):
        print(current)
        return
    print_subsequences(s, current, index + 1)
    print_subsequences(s, current + s[index], index + 1)
```

## Print All Subsets of Array

```python
def print_subsets(arr, current=None, index=0):
    if current is None:
        current = []
    if index == len(arr):
        print(current)
        return
    print_subsets(arr, current, index + 1)
    print_subsets(arr, current + [arr[index]], index + 1)
```
