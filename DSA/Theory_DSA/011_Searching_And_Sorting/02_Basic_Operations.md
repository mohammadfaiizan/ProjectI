# Basic Searching and Sorting Operations

## Linear Search (Unsorted Array)

**Idea:** Scan each element from left to right until target is found or array ends.

**Steps:**
1. Start at index 0
2. Compare each element with target
3. Return index if found
4. Return -1 if end of array reached

**Time:** O(n), **Space:** O(1)

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

---

## Sentinel Linear Search

**Idea:** Place target at the end as sentinel to avoid boundary check in loop. Guarantees loop terminates when target is found (either in array or at sentinel).

**Steps:**
1. Store last element
2. Replace last element with target (sentinel)
3. Loop until arr[i] == target (always terminates)
4. Restore last element
5. Return i if i < n-1 else -1

**Benefit:** One comparison per iteration instead of two (no explicit bounds check).

```python
def sentinel_linear_search(arr, target):
    n = len(arr)
    if n == 0:
        return -1
    last = arr[n - 1]
    arr[n - 1] = target
    i = 0
    while arr[i] != target:
        i += 1
    arr[n - 1] = last
    if i < n - 1 or last == target:
        return i
    return -1
```

---

## Binary Search Iterative (Sorted Array)

**Idea:** Repeatedly halve the search space by comparing target with middle element.

**Steps:**
1. left = 0, right = n - 1
2. While left <= right: mid = (left + right) // 2
3. If arr[mid] == target, return mid
4. If arr[mid] < target, left = mid + 1
5. Else right = mid - 1
6. Return -1 if not found

**Time:** O(log n), **Space:** O(1)

```python
def binary_search_iterative(arr, target):
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

---

## Binary Search Recursive

**Idea:** Same as iterative but implemented recursively. Base case: empty range or found. Recurse on left or right half.

```python
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    return binary_search_recursive(arr, target, left, mid - 1)
```

**Time:** O(log n), **Space:** O(log n) for recursion stack

---

## Selection Sort

**Idea:** Repeatedly find minimum in unsorted portion and swap with current position.

**Steps:**
1. For i from 0 to n-2: find min index in arr[i:n]
2. Swap arr[i] with arr[min_index]
3. Sorted portion grows from left

**Properties:** O(n^2), not stable, in-place, not adaptive

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
```

---

## Bubble Sort

**Idea:** Repeatedly swap adjacent elements if wrong order. Largest elements "bubble" to end. Use flag to stop early if no swaps (adaptive).

**Steps:**
1. For i from 0 to n-1: for j from 0 to n-1-i
2. If arr[j] > arr[j+1], swap
3. If no swaps in a pass, break (adaptive)

**Properties:** O(n^2), stable, in-place, adaptive with flag

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
```

---

## Insertion Sort

**Idea:** Build sorted portion from left. For each new element, shift larger elements right and insert.

**Steps:**
1. For i from 1 to n-1: key = arr[i]
2. j = i - 1, shift arr[j] right while arr[j] > key
3. Place key at arr[j+1]

**Properties:** O(n^2), stable, in-place, adaptive (O(n) for nearly sorted)

```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
```

---

## Comparison of Basic Sorts

| Sort | Best | Avg | Worst | Space | Stable | Adaptive | Use Case |
|------|------|-----|-------|-------|--------|----------|----------|
| Selection | O(n^2) | O(n^2) | O(n^2) | O(1) | No | No | Minimize swaps |
| Bubble | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Educational |
| Insertion | O(n) | O(n^2) | O(n^2) | O(1) | Yes | Yes | Small n, nearly sorted |

**When to use:**
- **Selection:** When swap cost is high (e.g., large objects); O(n) swaps
- **Bubble:** Rarely; mainly for teaching
- **Insertion:** Small arrays (n < 50), nearly sorted data, hybrid sorts (tim sort)
