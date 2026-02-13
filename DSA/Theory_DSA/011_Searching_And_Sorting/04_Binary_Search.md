# Binary Search: Theory and Variations

## Standard Template

**Idea:** Maintain left and right boundaries. Mid = (left + right) // 2. Adjust boundaries based on comparison. Terminate when left > right.

```python
def binary_search(arr, target):
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

## Find First Occurrence

**Idea:** When target found, do not return. Set right = mid - 1 to search left. Return left when loop ends (or -1 if not found).

```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

---

## Find Last Occurrence

**Idea:** When target found, set left = mid + 1 to search right. Return result when loop ends.

```python
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

---

## Count Occurrences

**Idea:** Find first and last occurrence. Count = last - first + 1 if found, else 0.

```python
def count_occurrences(arr, target):
    first = find_first(arr, target)
    if first == -1:
        return 0
    last = find_last(arr, target)
    return last - first + 1
```

---

## Search Insert Position (Lower Bound)

**Idea:** Find leftmost index where arr[i] >= target. Return left if found, else len(arr).

```python
def search_insert_position(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left
```

---

## Search in Rotated Sorted Array

**Idea:** One half is always sorted. Check which half contains target. If mid in left sorted: target in [left, mid] if left <= target <= mid else right half. If mid in right sorted: target in [mid, right] if mid <= target <= right else left half.

```python
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

---

## Search in Rotated Sorted Array II (Duplicates)

**Idea:** Same as above but when arr[left] == arr[mid] == arr[right], we cannot tell which half is sorted. Shrink: left += 1, right -= 1.

```python
def search_rotated_ii(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        if arr[left] == arr[mid] == arr[right]:
            left += 1
            right -= 1
        elif arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False
```

---

## Find Minimum in Rotated Sorted Array

**Idea:** Minimum is in unsorted half. Compare arr[mid] with arr[right]. If arr[mid] > arr[right], min is in right half. Else min is in left half (including mid).

```python
def find_min_rotated(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    return arr[left]
```

---

## Find Minimum in Rotated Sorted Array II (Duplicates)

**Idea:** When arr[mid] == arr[right], we cannot discard either half. Decrement right.

```python
def find_min_rotated_ii(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[right]:
            left = mid + 1
        elif arr[mid] < arr[right]:
            right = mid
        else:
            right -= 1
    return arr[left]
```

---

## Find Peak Element

**Idea:** Peak is arr[i] where arr[i] > arr[i-1] and arr[i] > arr[i+1]. If arr[mid] < arr[mid+1], peak is in right half. Else peak is in left half (including mid).

```python
def find_peak_element(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left
```

---

## Find Peak in Mountain Array

**Idea:** Mountain: strictly increasing then strictly decreasing. Same as find peak: if arr[mid] < arr[mid+1], peak is right; else left.

```python
def peak_index_in_mountain_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left
```

---

## Search a 2D Matrix (Row-Col Sorted)

**Idea:** Treat as 1D sorted array. Row = mid // cols, Col = mid % cols.

```python
def search_2d_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    while left <= right:
        mid = (left + right) // 2
        r, c = mid // cols, mid % cols
        val = matrix[r][c]
        if val == target:
            return True
        if val < target:
            left = mid + 1
        else:
            right = mid - 1
    return False
```

---

## Search a 2D Matrix II (Each Row/Col Sorted)

**Idea:** Start from top-right. If target > matrix[r][c], move down. If target < matrix[r][c], move left. Else found.

```python
def search_2d_matrix_ii(matrix, target):
    if not matrix or not matrix[0]:
        return False
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        val = matrix[r][c]
        if val == target:
            return True
        if val < target:
            r += 1
        else:
            c -= 1
    return False
```

---

## Single Element in Sorted Array

**Idea:** All pairs (0,1), (2,3), (4,5)... before single are same. After single, pairs shift. Check pair parity: if mid is even, pair is (mid, mid+1). If arr[mid] == arr[mid+1], single is right. Else left.

```python
def single_non_duplicate(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if mid % 2 == 1:
            mid -= 1
        if arr[mid] == arr[mid + 1]:
            left = mid + 2
        else:
            right = mid
    return arr[left]
```

---

## Smallest Letter Greater Than Target

**Idea:** Find first letter > target. Wrap around: if none found, return first letter.

```python
def next_greatest_letter(letters, target):
    left, right = 0, len(letters) - 1
    while left <= right:
        mid = (left + right) // 2
        if letters[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    return letters[left % len(letters)]
```

---

## Find First and Last Position

**Idea:** Two binary searches: first occurrence and last occurrence.

```python
def search_range(arr, target):
    def find_first():
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                right = mid - 1
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

    def find_last():
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                left = mid + 1
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

    return [find_first(), find_last()]
```

---

## Median of Two Sorted Arrays

**Idea:** Partition both arrays so left half has (n+m+1)//2 elements. Binary search partition in smaller array. Check max_left <= min_right.

```python
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2
    left, right = 0, m
    while left <= right:
        i = (left + right) // 2
        j = half - i
        left1 = float('-inf') if i == 0 else nums1[i - 1]
        right1 = float('inf') if i == m else nums1[i]
        left2 = float('-inf') if j == 0 else nums2[j - 1]
        right2 = float('inf') if j == n else nums2[j]
        if left1 <= right2 and left2 <= right1:
            if (m + n) % 2 == 0:
                return (max(left1, left2) + min(right1, right2)) / 2
            return max(left1, left2)
        if left1 > right2:
            right = i - 1
        else:
            left = i + 1
    return 0.0
```

---

## H-Index Binary Search

**Idea:** Sort citations descending. Find largest i such that citations[i] >= i+1. Binary search for i.

```python
def h_index(citations):
    citations.sort(reverse=True)
    left, right = 0, len(citations) - 1
    n = len(citations)
    result = 0
    while left <= right:
        mid = (left + right) // 2
        if citations[mid] >= mid + 1:
            result = mid + 1
            left = mid + 1
        else:
            right = mid - 1
    return result
```

---

## Time-Based Key-Value Store

**Idea:** Store (timestamp, value) per key. For get, binary search largest timestamp <= given timestamp.

```python
from collections import defaultdict

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        arr = self.store[key]
        left, right = 0, len(arr) - 1
        result = ""
        while left <= right:
            mid = (left + right) // 2
            if arr[mid][0] <= timestamp:
                result = arr[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        return result
```

---

## Find Right Interval

**Idea:** For each interval, find smallest start_j >= end_i. Sort by start, store original indices. Binary search for each end_i.

```python
def find_right_interval(intervals):
    n = len(intervals)
    sorted_starts = [(intervals[i][0], i) for i in range(n)]
    sorted_starts.sort(key=lambda x: x[0])
    starts = [s[0] for s in sorted_starts]
    result = []
    for i in range(n):
        end = intervals[i][1]
        left, right = 0, n - 1
        idx = -1
        while left <= right:
            mid = (left + right) // 2
            if sorted_starts[mid][0] >= end:
                idx = sorted_starts[mid][1]
                right = mid - 1
            else:
                left = mid + 1
        result.append(idx)
    return result
```

---

## Find K Closest Elements

**Idea:** Binary search for left boundary of window of size k. Compare arr[mid] and arr[mid+k] distance to x. If arr[mid] is closer, shrink right; else shrink left.

```python
def find_closest_elements(arr, k, x):
    left, right = 0, len(arr) - k
    while left < right:
        mid = (left + right) // 2
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]
```

---

## Count of Smaller Numbers After Self

**Idea:** Binary search insertion: process from right, maintain sorted list of seen elements. For each num, binary search position = count of smaller. Insert num in sorted order.

```python
import bisect

def count_smaller(nums):
    sorted_nums = []
    result = []
    for num in reversed(nums):
        idx = bisect.bisect_left(sorted_nums, num)
        result.append(idx)
        bisect.insort(sorted_nums, num)
    return result[::-1]
```

---

## Count Negative Numbers in Sorted Matrix

**Idea:** Grid sorted row-wise and col-wise (non-increasing). Start top-right. If negative, all below are negative; count += rows - r, move left. Else move down.

```python
def count_negatives(grid):
    rows, cols = len(grid), len(grid[0])
    r, c = 0, cols - 1
    count = 0
    while r < rows and c >= 0:
        if grid[r][c] < 0:
            count += rows - r
            c -= 1
        else:
            r += 1
    return count
```

---

## Successful Pairs of Spells and Potions

**Idea:** Sort potions. For each spell, need min potion such that spell * potion >= success. Binary search for ceil(success / spell) in potions.

```python
def successful_pairs(spells, potions, success):
    potions.sort()
    n = len(potions)
    result = []
    for spell in spells:
        need = (success + spell - 1) // spell
        left, right = 0, n - 1
        idx = n
        while left <= right:
            mid = (left + right) // 2
            if potions[mid] >= need:
                idx = mid
                right = mid - 1
            else:
                left = mid + 1
        result.append(n - idx)
    return result
```
