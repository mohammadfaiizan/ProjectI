# Comparison-Based Sorting

## Merge Sort

**Algorithm:** Divide array into halves, recursively sort each, merge two sorted halves.

**Merge procedure:** Two pointers, copy smaller element to result. Append remaining.

**Complexity:** O(n log n) time, O(n) space. Stable. Good for linked lists (O(1) extra space with linked merge).

```python
def merge(arr, left, mid, right):
    L = arr[left:mid+1]
    R = arr[mid+1:right+1]
    i = j = 0
    k = left
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)
```

---

## Quick Sort

**Lomuto partition:** Pivot at end. Single pointer, swap smaller elements to left.

**Hoare partition:** Two pointers from ends, swap when both find wrong elements.

**Pivot strategies:**
- First/last: O(n^2) on sorted
- Median-of-three: better
- Random: expected O(n log n)

**Complexity:** O(n log n) average, O(n^2) worst. Not stable. Cache-friendly (local access).

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

def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = lomuto_partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)
```

---

## Heap Sort

**Algorithm:** Build max-heap (heapify from last non-leaf up). Repeatedly extract max: swap root with last, heapify down. Sorted region grows from right.

**Complexity:** O(n log n). O(1) space. Not stable.

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

**Idea:** Gap-based insertion sort. Gap sequences: Shell (n/2, n/4, ...), Knuth (1, 4, 13, 40, ...), Sedgewick.

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

## Tim Sort

**Mechanism:** Find runs (ascending or descending). Use insertion sort for small runs (minrun ~32-64). Merge runs. Galloping merge when one run dominates. Stable, adaptive.

---

## Comparison Lower Bound Proof

**Decision tree model:** Each comparison is a node with two children (yes/no). Leaves are permutations. Correct sort must reach distinct leaf for each input.

**Proof:**
- n! possible orderings
- Binary tree with n! leaves has height >= log2(n!)
- log2(n!) = Theta(n log n) by Stirling
- Therefore Omega(n log n) comparisons required

---

## When to Use Each Sort

| Sort | Use when |
|------|----------|
| Merge | Stable needed, linked lists, external sort |
| Quick | General purpose, in-place, cache matters |
| Heap | O(1) space required, worst-case guarantee |
| Shell | Simple improvement over insertion |
| Tim | Default for mixed data (Python) |

---

## Comprehensive Comparison Table

| Sort | Best | Avg | Worst | Space | Stable | Adaptive | Cache |
|------|------|-----|-------|-------|--------|----------|-------|
| Merge | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No | Poor |
| Quick | O(n log n) | O(n log n) | O(n^2) | O(log n) | No | No | Good |
| Heap | O(n log n) | O(n log n) | O(n log n) | O(1) | No | No | Poor |
| Shell | O(n log n) | O(n^1.3) | O(n^2) | O(1) | No | Some | Good |
| Tim | O(n) | O(n log n) | O(n log n) | O(n) | Yes | Yes | Good |
