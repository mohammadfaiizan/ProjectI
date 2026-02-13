# Recursion Patterns

## Linear Recursion

Single recursive call per invocation. Problem size reduced by one (or constant) each step.

**Factorial**:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**Sum**:
```python
def sum_linear(arr, n):
    if n <= 0:
        return 0
    return sum_linear(arr, n - 1) + arr[n - 1]
```

## Binary Recursion

Two recursive calls per invocation. Problem splits into two subproblems.

**Fibonacci**:
```python
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

**Merge Sort**:
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

## Tail Recursion

The recursive call is the last operation. No computation after the call. Can be optimized to iteration by compilers (Python does not optimize).

```python
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, n * acc)
```

## Non-Tail to Tail Conversion

Original factorial (non-tail): work done after recursive call (multiplication).
Tail version: pass accumulated result as parameter, do work before/during call.

```python
def factorial_nontail(n):
    if n <= 1:
        return 1
    return n * factorial_nontail(n - 1)

def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, acc * n)
```

## Head Recursion vs Tail Recursion

**Head recursion**: Recursive call before processing. Processes on way back from base case.
```python
def head_recursion(n):
    if n > 0:
        head_recursion(n - 1)
        print(n)
```

**Tail recursion**: Recursive call is last. Processing done before call.
```python
def tail_recursion(n):
    if n > 0:
        print(n)
        tail_recursion(n - 1)
```

## Mutual Recursion

Two or more functions call each other.

```python
def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)
```

## Nested Recursion

The argument to the recursive call is itself a recursive call.

```python
def ackermann(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ackermann(m - 1, 1)
    return ackermann(m - 1, ackermann(m, n - 1))
```

## Tree Recursion

Multiple branches from each call. Each call spawns multiple recursive calls.

```python
def fib_tree(n):
    if n <= 1:
        return n
    return fib_tree(n - 1) + fib_tree(n - 2)
```

## Indirect Recursion

A calls B, B calls A (or longer chain).

```python
def func_a(n):
    if n <= 0:
        return
    print("A", n)
    func_b(n - 1)

def func_b(n):
    if n <= 0:
        return
    print("B", n)
    func_a(n - 1)
```

## Recursion to Iteration (Explicit Stack)

```python
def factorial_iterative(n):
    stack = [n]
    result = 1
    while stack:
        x = stack.pop()
        if x <= 1:
            continue
        result *= x
        stack.append(x - 1)
    return result

def dfs_iterative(root):
    if root is None:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

## Iteration to Recursion

```python
def sum_iterative(arr):
    total = 0
    for x in arr:
        total += x
    return total

def sum_recursive(arr, i=0):
    if i >= len(arr):
        return 0
    return arr[i] + sum_recursive(arr, i + 1)
```

## Recursion with Accumulator

Pass accumulated value through recursive calls to avoid work on return.

```python
def sum_accumulator(arr, i=0, acc=0):
    if i >= len(arr):
        return acc
    return sum_accumulator(arr, i + 1, acc + arr[i])
```

## Recursion with Helper Function Pattern

Use inner helper when extra parameters (e.g., index, accumulator) are needed but should not be part of public API.

```python
def reverse_string(s):
    def helper(left, right):
        if left >= right:
            return
        s_list[left], s_list[right] = s_list[right], s_list[left]
        helper(left + 1, right - 1)

    s_list = list(s)
    helper(0, len(s) - 1)
    return ''.join(s_list)
```
