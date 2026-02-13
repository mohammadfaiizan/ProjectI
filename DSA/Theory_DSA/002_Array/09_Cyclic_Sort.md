# Cyclic Sort

When array contains numbers in range [1, n] or [0, n-1], each element has a correct index (value - 1 or value). Cyclic sort places each element at its correct position by swapping. After sorting, arr[i] == i+1 (or arr[i] == i). Used for finding missing, duplicate, or corrupt elements.

## Cyclic Sort (Place at Correct Index)

For range [1, n], correct index of value v is v-1. Swap arr[i] with arr[arr[i]-1] until arr[i] is at correct place. Time O(n), Space O(1).

```python
def cyclic_sort(arr):
    i = 0
    while i < len(arr):
        correct_idx = arr[i] - 1
        if 0 <= correct_idx < len(arr) and arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    return arr
```

## Find Missing Number 0 to n

Array of n elements with values in [0, n]. One number missing. Sum approach: expected sum - actual sum. Or XOR: xor all indices and values, result is missing. Or cyclic sort and find index where arr[i] != i. Time O(n), Space O(1).

```python
def find_missing_number(arr):
    n = len(arr)
    expected = n * (n + 1) // 2
    actual = sum(arr)
    return expected - actual

def find_missing_number_xor(arr):
    n = len(arr)
    xor = 0
    for i in range(n + 1):
        xor ^= i
    for x in arr:
        xor ^= x
    return xor
```

## Find All Missing Numbers

Array of n elements, values in [1, n]. Some missing. Cyclic sort, then indices where arr[i] != i+1 are missing. Time O(n), Space O(1) excluding output.

```python
def find_all_missing(arr):
    i = 0
    n = len(arr)
    while i < n:
        correct_idx = arr[i] - 1
        if 0 <= correct_idx < n and arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    return [i + 1 for i in range(n) if arr[i] != i + 1]
```

## Find Duplicate Number

Array of n+1 elements, values in [1, n]. One duplicate. Cyclic sort, duplicate will be at index 0 or where arr[i] != i+1. Or Floyd cycle detection (treat as linked list). Time O(n), Space O(1).

```python
def find_duplicate(arr):
    i = 0
    n = len(arr)
    while i < n:
        correct_idx = arr[i] - 1
        if arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    for i in range(n):
        if arr[i] != i + 1:
            return arr[i]
    return arr[0]

def find_duplicate_floyd(arr):
    slow = fast = arr[0]
    while True:
        slow = arr[slow]
        fast = arr[arr[fast]]
        if slow == fast:
            break
    slow = arr[0]
    while slow != fast:
        slow = arr[slow]
        fast = arr[fast]
    return slow
```

## Find All Duplicates

Array of n elements, values in [1, n]. Some appear twice, others once. Cyclic sort, elements not at correct position after sorting might be duplicates. Or use negative marking: for each x, mark arr[x-1] as negative; if already negative, x is duplicate. Time O(n), Space O(1).

```python
def find_all_duplicates(arr):
    result = []
    for x in arr:
        idx = abs(x) - 1
        if arr[idx] < 0:
            result.append(abs(x))
        else:
            arr[idx] = -arr[idx]
    return result
```

## Find Corrupt Pair

Array of n elements, values in [1, n]. One duplicate and one missing. Cyclic sort, then find index where arr[i] != i+1: duplicate = arr[i], missing = i+1. Time O(n), Space O(1).

```python
def find_corrupt_pair(arr):
    i = 0
    n = len(arr)
    while i < n:
        correct_idx = arr[i] - 1
        if 0 <= correct_idx < n and arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    for i in range(n):
        if arr[i] != i + 1:
            return [arr[i], i + 1]
    return []
```

## First Missing Positive

Array of integers. Find smallest positive integer not in array. Values in [1, n+1] possible. Cyclic sort for values in [1, n]. Place each positive x at index x-1 if 1 <= x <= n. Then scan for first index where arr[i] != i+1. Time O(n), Space O(1).

```python
def first_missing_positive(arr):
    n = len(arr)
    i = 0
    while i < n:
        correct_idx = arr[i] - 1
        if 1 <= arr[i] <= n and arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    for i in range(n):
        if arr[i] != i + 1:
            return i + 1
    return n + 1
```

## Find k Missing Positives

Array and k. Find kth smallest positive integer not in array. Cyclic sort to place valid positives. Then count missing until we reach k. Time O(n), Space O(1).

```python
def find_kth_missing_positive(arr, k):
    n = len(arr)
    i = 0
    while i < n:
        correct_idx = arr[i] - 1
        if 1 <= arr[i] <= n and arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    missing_count = 0
    for i in range(n):
        if arr[i] != i + 1:
            missing_count += 1
            if missing_count == k:
                return i + 1
    return n + k
```

## Set Mismatch

Array of n elements, values in [1, n]. One duplicate and one missing. Same as corrupt pair. Return [duplicate, missing]. Time O(n), Space O(1).

```python
def set_mismatch(arr):
    i = 0
    n = len(arr)
    while i < n:
        correct_idx = arr[i] - 1
        if arr[i] != arr[correct_idx]:
            arr[i], arr[correct_idx] = arr[correct_idx], arr[i]
        else:
            i += 1
    for i in range(n):
        if arr[i] != i + 1:
            return [arr[i], i + 1]
    return []
```
