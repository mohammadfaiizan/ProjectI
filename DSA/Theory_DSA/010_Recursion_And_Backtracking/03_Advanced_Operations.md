# Advanced Recursion Operations

## Generate All Binary Strings of Length N

```python
def generate_binary_strings(n, current=""):
    if len(current) == n:
        print(current)
        return
    generate_binary_strings(n, current + "0")
    generate_binary_strings(n, current + "1")
```

## Generate All Strings of Length N from Set

```python
def generate_strings_from_set(chars, n, current=""):
    if len(current) == n:
        print(current)
        return
    for c in chars:
        generate_strings_from_set(chars, n, current + c)
```

## Print All Permutations (Without Duplicates)

```python
def permutations_unique(arr, current=None, used=None):
    if current is None:
        current = []
    if used is None:
        used = [False] * len(arr)
    if len(current) == len(arr):
        print(current)
        return
    for i in range(len(arr)):
        if not used[i]:
            used[i] = True
            current.append(arr[i])
            permutations_unique(arr, current, used)
            current.pop()
            used[i] = False
```

## Print All Permutations (With Duplicates)

```python
def permutations_with_duplicates(arr):
    from collections import Counter
    count = Counter(arr)
    result = []

    def backtrack(path):
        if len(path) == len(arr):
            result.append(path[:])
            return
        for c in count:
            if count[c] > 0:
                count[c] -= 1
                path.append(c)
                backtrack(path)
                path.pop()
                count[c] += 1

    backtrack([])
    return result
```

## Flood Fill Recursive

```python
def flood_fill(image, sr, sc, new_color):
    old_color = image[sr][sc]
    if old_color == new_color:
        return image
    rows, cols = len(image), len(image[0])

    def fill(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or image[r][c] != old_color:
            return
        image[r][c] = new_color
        fill(r + 1, c)
        fill(r - 1, c)
        fill(r, c + 1)
        fill(r, c - 1)

    fill(sr, sc)
    return image
```

## Recursive Tree Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder(root):
    if root is None:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def preorder(root):
    if root is None:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

def postorder(root):
    if root is None:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)
```

## Recursive Linked List Operations

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    if head is None or head.next is None:
        return head
    new_head = reverse_list(head.next)
    head.next.next = head
    head.next = None
    return new_head

def search_list(head, target):
    if head is None:
        return False
    if head.val == target:
        return True
    return search_list(head.next, target)

def list_length(head):
    if head is None:
        return 0
    return 1 + list_length(head.next)
```

## Josephus Problem

```python
def josephus(n, k):
    if n == 1:
        return 0
    return (josephus(n - 1, k) + k) % n
```

## Staircase Problem (Ways to climb n steps, 1 or 2 at a time)

```python
def climb_stairs(n):
    if n <= 2:
        return n
    return climb_stairs(n - 1) + climb_stairs(n - 2)
```

## Digit Sum Until Single Digit

```python
def digit_sum_single(n):
    if n < 10:
        return n
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return digit_sum_single(total)
```

## Recursive String Compression

```python
def compress_string(s, index=0):
    if index >= len(s):
        return ""
    count = 1
    while index + 1 < len(s) and s[index] == s[index + 1]:
        count += 1
        index += 1
    if count > 1:
        return s[index] + str(count) + compress_string(s, index + 1)
    return s[index] + compress_string(s, index + 1)
```

## Mutual Recursion

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

## Ackermann Function

```python
def ackermann(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ackermann(m - 1, 1)
    return ackermann(m - 1, ackermann(m, n - 1))
```

## Memoization Wrapper (Generic Decorator)

```python
def memoize(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper

@memoize
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```
