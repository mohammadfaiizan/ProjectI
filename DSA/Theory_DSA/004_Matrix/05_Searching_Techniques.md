# Matrix Searching Techniques

## Search in Row-Sorted Matrix (Binary Search Per Row)

Each row is sorted. Binary search each row independently. Time O(m log n).

```python
def search_row_sorted(matrix, target):
    for i in range(len(matrix)):
        lo, hi = 0, len(matrix[0]) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if matrix[i][mid] == target:
                return True
            if matrix[i][mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
    return False
```

## Search in Row-Col Sorted Matrix (Staircase O(m+n))

Each row sorted ascending, each column sorted ascending. Start from top-right or bottom-left. If current > target, move left; if current < target, move down.

```
Matrix (row and col sorted):    Start top-right (3). Target=5.
  1  4  7 11                   3 < 5 -> down to 6
  2  5  8 12                   6 > 5 -> left to 5. Found.
  3  6  9 16
 10 13 14 17
```

```python
def search_row_col_sorted(matrix, target):
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    i, j = 0, n - 1
    while i < m and j >= 0:
        if matrix[i][j] == target:
            return True
        if matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False
```

## Search in Fully Sorted Matrix (1D Binary Search)

Matrix sorted in row-major order (each row's last element <= next row's first). Treat as 1D sorted array. Index mapping: mid // n gives row, mid % n gives col.

```python
def search_sorted_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    lo, hi = 0, m * n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        r, c = mid // n, mid % n
        if matrix[r][c] == target:
            return True
        if matrix[r][c] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False
```

## Kth Smallest in Sorted Matrix

### Min-Heap Approach

Push first element of each row (or first row). Pop k times. When popping (r, c), push (r, c+1) if exists. Time O(k log(min(k, m))).

```python
import heapq

def kth_smallest_min_heap(matrix, k):
    if not matrix or not matrix[0]:
        return None
    m, n = len(matrix), len(matrix[0])
    heap = [(matrix[i][0], i, 0) for i in range(min(k, m))]
    heapq.heapify(heap)
    for _ in range(k - 1):
        val, r, c = heapq.heappop(heap)
        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
    return heap[0][0]
```

### Binary Search Approach

Binary search on value range [min, max]. For each mid, count elements <= mid. If count >= k, answer <= mid; else answer > mid.

```python
def kth_smallest_binary_search(matrix, k):
    if not matrix or not matrix[0]:
        return None
    m, n = len(matrix), len(matrix[0])
    lo, hi = matrix[0][0], matrix[m - 1][n - 1]
    while lo < hi:
        mid = (lo + hi) // 2
        count = 0
        j = n - 1
        for i in range(m):
            while j >= 0 and matrix[i][j] > mid:
                j -= 1
            count += j + 1
        if count < k:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

## Median of Row-Wise Sorted Matrix

Binary search on value range. For each mid, count elements <= mid. Median is the element at position (m*n)//2. Find smallest value such that count of elements <= value >= (m*n)//2 + 1.

```python
def median_row_sorted(matrix):
    if not matrix or not matrix[0]:
        return None
    m, n = len(matrix), len(matrix[0])
    total = m * n
    lo, hi = matrix[0][0], matrix[0][-1]
    for i in range(m):
        lo = min(lo, matrix[i][0])
        hi = max(hi, matrix[i][-1])
    desired = total // 2 + 1
    while lo < hi:
        mid = (lo + hi) // 2
        count = 0
        j = n - 1
        for i in range(m):
            while j >= 0 and matrix[i][j] > mid:
                j -= 1
            count += j + 1
        if count < desired:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

## Count Negatives in Sorted Matrix

Each row and column non-increasing. Start from top-right. For each row, find first negative (or end). All elements to the right are negative.

```python
def count_negatives(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    count = 0
    j = n - 1
    for i in range(m):
        while j >= 0 and matrix[i][j] < 0:
            j -= 1
        count += n - 1 - j
    return count
```

## Count Elements Less Than or Equal to Given Value

In row-col sorted matrix, for each row use binary search to find rightmost element <= value.

```python
def count_less_equal(matrix, value):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    count = 0
    j = n - 1
    for i in range(m):
        while j >= 0 and matrix[i][j] > value:
            j -= 1
        count += j + 1
    return count
```
