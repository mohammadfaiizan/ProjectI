# Advanced Searching and Sorting Operations

## Merge Sort

**Idea:** Divide array into halves, sort each recursively, merge two sorted halves.

**Steps:**
1. Base case: length <= 1, return
2. Mid = n // 2, sort left and right halves
3. Merge: two pointers, copy smaller to result
4. Copy result back to original

**Properties:** O(n log n), stable, O(n) space

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    merge_sort(left)
    merge_sort(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
```

---

## Quick Sort (Lomuto Partition)

**Idea:** Choose pivot (typically last), partition so elements <= pivot are left, > pivot are right. Pivot goes to final position. Recurse on both sides.

**Lomuto:** Single pointer scans, maintains boundary of "less than" region.

```python
def lomuto_partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort_lomuto(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = lomuto_partition(arr, low, high)
        quick_sort_lomuto(arr, low, pi - 1)
        quick_sort_lomuto(arr, pi + 1, high)
```

---

## Quick Sort (Hoare Partition)

**Idea:** Two pointers from both ends, swap when both find wrong elements. Pivot (first) may not land at final position; partition returns meeting point.

```python
def hoare_partition(arr, low, high):
    pivot = arr[low]
    i, j = low - 1, high + 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]

def quick_sort_hoare(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = hoare_partition(arr, low, high)
        quick_sort_hoare(arr, low, pi)
        quick_sort_hoare(arr, pi + 1, high)
```

---

## Randomized Quick Sort

**Idea:** Pick random pivot to avoid O(n^2) on sorted/reverse-sorted input. Expected O(n log n).

```python
import random

def randomized_partition(arr, low, high):
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
    return lomuto_partition(arr, low, high)

def randomized_quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = randomized_partition(arr, low, high)
        randomized_quick_sort(arr, low, pi - 1)
        randomized_quick_sort(arr, pi + 1, high)
```

---

## Three-Way Quick Sort (Dutch National Flag)

**Idea:** Partition into three regions: < pivot, = pivot, > pivot. Handles duplicates efficiently.

```python
def three_way_partition(arr, low, high):
    pivot = arr[low]
    lt, i, gt = low, low, high
    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1
    return lt, gt

def three_way_quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        lt, gt = three_way_partition(arr, low, high)
        three_way_quick_sort(arr, low, lt - 1)
        three_way_quick_sort(arr, gt + 1, high)
```

---

## Heap Sort

**Idea:** Build max-heap, repeatedly extract max (swap with last, heapify down). Sorted region grows from right.

**Steps:**
1. Build max-heap: heapify from last non-leaf up
2. For i from n-1 down to 1: swap arr[0] with arr[i], heapify(0, i)

**Properties:** O(n log n), in-place O(1), not stable

```python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
```

---

## Shell Sort

**Idea:** Gap-based insertion sort. Start with large gap, reduce until gap=1. Elements far apart are compared first.

```python
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
```

---

## Tim Sort Overview

**Idea:** Hybrid of merge sort and insertion sort. Python's default sort.

**Mechanism:**
- Find "runs" (ascending or descending segments)
- Use insertion sort for small runs (minrun, typically 32-64)
- Merge runs using merge sort logic
- Adaptive: O(n) for nearly sorted, O(n log n) worst case
- Stable

---

## Introsort Overview

**Idea:** Hybrid used as C++ std::sort. Combines quicksort, heapsort, and insertion sort.

**Mechanism:**
- Start with quicksort
- If recursion depth exceeds 2*log(n), switch to heapsort (avoid O(n^2))
- Use insertion sort for small subarrays (typically n < 16)
- O(n log n) worst case guaranteed

---

## Comparison of Advanced Sorts

| Sort | Time (avg) | Time (worst) | Space | Stable | Adaptive | Cache |
|------|------------|--------------|-------|--------|----------|-------|
| Merge | O(n log n) | O(n log n) | O(n) | Yes | No | Poor (scattered access) |
| Quick | O(n log n) | O(n^2) | O(log n) | No | No | Good (local access) |
| Heap | O(n log n) | O(n log n) | O(1) | No | No | Poor |
| Shell | O(n^1.3) | O(n^2) | O(1) | No | Some | Good |
| Tim | O(n log n) | O(n log n) | O(n) | Yes | Yes | Good |
| Introsort | O(n log n) | O(n log n) | O(log n) | No | No | Good |
