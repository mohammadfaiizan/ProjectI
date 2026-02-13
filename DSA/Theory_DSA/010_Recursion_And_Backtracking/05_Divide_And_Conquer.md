# Divide and Conquer

## Template

1. **Divide**: Split problem into smaller subproblems (usually 2)
2. **Conquer**: Solve subproblems recursively (base case when trivial)
3. **Combine**: Merge solutions of subproblems into solution for original

```python
def divide_conquer(problem):
    if base_case(problem):
        return solve_directly(problem)
    subproblems = divide(problem)
    sub_solutions = [divide_conquer(sp) for sp in subproblems]
    return combine(sub_solutions)
```

## Merge Sort

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

## Quick Sort

```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot = partition(arr, low, high)
        quick_sort(arr, low, pivot - 1)
        quick_sort(arr, pivot + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

## Count Inversions (Modified Merge Sort)

```python
def count_inversions(arr):
    def merge_count(left, right):
        result = []
        i = j = count = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                count += len(left) - i
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result, count

    def sort_count(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, c1 = sort_count(a[:mid])
        right, c2 = sort_count(a[mid:])
        merged, c3 = merge_count(left, right)
        return merged, c1 + c2 + c3

    _, count = sort_count(arr)
    return count
```

## Count Smaller Numbers After Self

```python
def count_smaller(nums):
    n = len(nums)
    result = [0] * n
    indexed = list(enumerate(nums))

    def merge_count(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_count(arr[:mid])
        right = merge_count(arr[mid:])
        i = j = 0
        merged = []
        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                result[left[i][0]] += j
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        while i < len(left):
            result[left[i][0]] += j
            merged.append(left[i])
            i += 1
        merged.extend(right[j:])
        return merged

    merge_count(indexed)
    return result
```

## Closest Pair of Points

```python
def closest_pair(points):
    points_sorted_x = sorted(points, key=lambda p: p[0])
    points_sorted_y = sorted(points, key=lambda p: p[1])

    def dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def closest(px, py):
        n = len(px)
        if n <= 3:
            min_d = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_d = min(min_d, dist(px[i], px[j]))
            return min_d
        mid = n // 2
        mid_point = px[mid]
        left_x = px[:mid]
        right_x = px[mid:]
        left_y = [p for p in py if p[0] <= mid_point[0]]
        right_y = [p for p in py if p[0] > mid_point[0]]
        d_left = closest(left_x, left_y)
        d_right = closest(right_x, right_y)
        d = min(d_left, d_right)
        strip = [p for p in py if abs(p[0] - mid_point[0]) < d]
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < d:
                d = min(d, dist(strip[i], strip[j]))
                j += 1
        return d

    return closest(points_sorted_x, points_sorted_y)
```

## Maximum Subarray Sum (D&C O(n log n))

```python
def max_subarray_dc(arr):
    def max_crossing(low, mid, high):
        left_sum = float('-inf')
        total = 0
        for i in range(mid, low - 1, -1):
            total += arr[i]
            left_sum = max(left_sum, total)
        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, high + 1):
            total += arr[i]
            right_sum = max(right_sum, total)
        return left_sum + right_sum

    def max_sub(low, high):
        if low == high:
            return arr[low]
        mid = (low + high) // 2
        return max(max_sub(low, mid), max_sub(mid + 1, high), max_crossing(low, mid, high))

    return max_sub(0, len(arr) - 1)
```

## Median of Two Sorted Arrays

```python
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    low, high = 0, m
    while low <= high:
        i = (low + high) // 2
        j = (m + n + 1) // 2 - i
        left1 = float('-inf') if i == 0 else nums1[i - 1]
        right1 = float('inf') if i == m else nums1[i]
        left2 = float('-inf') if j == 0 else nums2[j - 1]
        right2 = float('inf') if j == n else nums2[j]
        if left1 <= right2 and left2 <= right1:
            if (m + n) % 2 == 0:
                return (max(left1, left2) + min(right1, right2)) / 2
            return max(left1, left2)
        if left1 > right2:
            high = i - 1
        else:
            low = i + 1
```

## Kth Largest Element (Quickselect O(n) average)

```python
def find_kth_largest(nums, k):
    k = len(nums) - k

    def quickselect(left, right):
        pivot = nums[right]
        p = left
        for i in range(left, right):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[right] = nums[right], nums[p]
        if p == k:
            return nums[p]
        if p < k:
            return quickselect(p + 1, right)
        return quickselect(left, p - 1)

    return quickselect(0, len(nums) - 1)
```

## Different Ways to Add Parentheses

```python
def diff_ways_to_compute(expression):
    if expression.isdigit():
        return [int(expression)]
    results = []
    for i, c in enumerate(expression):
        if c in '+-*':
            left = diff_ways_to_compute(expression[:i])
            right = diff_ways_to_compute(expression[i + 1:])
            for l in left:
                for r in right:
                    if c == '+':
                        results.append(l + r)
                    elif c == '-':
                        results.append(l - r)
                    else:
                        results.append(l * r)
    return results
```

## Skyline Problem

```python
import heapq

def get_skyline(buildings):
    points = []
    for left, right, height in buildings:
        points.append((left, -height, right))
        points.append((right, 0, 0))
    points.sort(key=lambda x: (x[0], x[1]))
    result = []
    heap = [(0, float('inf'))]
    for x, neg_h, r in points:
        while heap[0][1] <= x:
            heapq.heappop(heap)
        if neg_h:
            heapq.heappush(heap, (neg_h, r))
        max_h = -heap[0][0]
        if not result or result[-1][1] != max_h:
            result.append([x, max_h])
    return result
```

## Count of Range Sum

```python
def count_range_sum(nums, lower, upper):
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)

    def merge_count(arr):
        if len(arr) <= 1:
            return 0
        mid = len(arr) // 2
        count = merge_count(arr[:mid]) + merge_count(arr[mid:])
        i = j = mid
        for left in arr[:mid]:
            while i < len(arr) and arr[i] - left < lower:
                i += 1
            while j < len(arr) and arr[j] - left <= upper:
                j += 1
            count += j - i
        arr[:] = sorted(arr)
        return count

    return merge_count(prefix)
```
