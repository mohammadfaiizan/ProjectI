# Dutch Flag Partitioning

Dijkstra's three-way partitioning: divide array into three regions (e.g., low, mid, high) in a single pass using three pointers. Used for sorting 0s, 1s, 2s and variations.

## Sort 0s 1s 2s (Dijkstra's DNF)

Three pointers: low (next 0 position), mid (current), high (next 2 position). If arr[mid]==0, swap with low. If arr[mid]==2, swap with high. If arr[mid]==1, advance mid. Time O(n), Space O(1).

```python
def sort_012_dnf(arr):
    low, mid, high = 0, 0, len(arr) - 1
    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    return arr
```

## Three-Way Partition Around Pivot

Partition into elements < pivot, == pivot, > pivot. Same DNF logic with pivot value. Time O(n), Space O(1).

```python
def three_way_partition(arr, pivot):
    low, mid, high = 0, 0, len(arr) - 1
    while mid <= high:
        if arr[mid] < pivot:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == pivot:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    return arr
```

## Segregate 0s and 1s

Two-way partition. Two pointers: left finds 1, right finds 0, swap. Time O(n), Space O(1).

```python
def segregate_01(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        while left < len(arr) and arr[left] == 0:
            left += 1
        while right >= 0 and arr[right] == 1:
            right -= 1
        if left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    return arr
```

## Segregate Negatives and Positives

Two pointers: left finds positive, right finds negative, swap. Time O(n), Space O(1).

```python
def segregate_negatives_positives(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        while left < len(arr) and arr[left] < 0:
            left += 1
        while right >= 0 and arr[right] >= 0:
            right -= 1
        if left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    return arr
```

## Segregate Even and Odd

Two pointers: left finds odd, right finds even, swap. Time O(n), Space O(1).

```python
def segregate_even_odd(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        while left < len(arr) and arr[left] % 2 == 0:
            left += 1
        while right >= 0 and arr[right] % 2 == 1:
            right -= 1
        if left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    return arr
```

## Sort with Three Distinct Values

Generalization of DNF. If array has exactly three distinct values (e.g., 'R','G','B'), use DNF with low_val, mid_val, high_val. Time O(n), Space O(1).

```python
def sort_three_distinct(arr, low_val, mid_val, high_val):
    low, mid, high = 0, 0, len(arr) - 1
    while mid <= high:
        if arr[mid] == low_val:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == mid_val:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    return arr
```

## Three-Way Quicksort Partition

Partition for quicksort that groups equal elements. Returns (low_end, high_start) such that elements in [low, low_end) are less, [low_end, high_start) are equal, [high_start, high] are greater. Time O(n), Space O(1).

```python
def three_way_quicksort_partition(arr, low, high):
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
```

## Move All Negatives to Beginning

Two pointers: write index for next negative. Scan and swap negatives to front. Time O(n), Space O(1).

```python
def move_negatives_to_beginning(arr):
    write = 0
    for read in range(len(arr)):
        if arr[read] < 0:
            arr[write], arr[read] = arr[read], arr[write]
            write += 1
    return arr
```
