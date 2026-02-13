# Advanced Array Operations

## Reverse In-Place (Two Pointers)

Swap elements from both ends moving toward center. Time O(n), Space O(1).

```python
def reverse_inplace(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
```

## Rotate Left by k (Juggling Algorithm)

Move elements in cycles. GCD(n, k) cycles, each of length n/GCD(n, k). Time O(n), Space O(1).

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def rotate_left_juggling(arr, k):
    n = len(arr)
    if n == 0:
        return arr
    k = k % n
    if k == 0:
        return arr
    g = gcd(n, k)
    for start in range(g):
        temp = arr[start]
        j = start
        while True:
            next_j = (j + k) % n
            if next_j == start:
                break
            arr[j] = arr[next_j]
            j = next_j
        arr[j] = temp
    return arr
```

## Rotate Left by k (Reversal Algorithm)

Reverse first k, reverse rest, reverse entire array. Time O(n), Space O(1).

```python
def reverse_range(arr, left, right):
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

def rotate_left_reversal(arr, k):
    n = len(arr)
    if n == 0:
        return arr
    k = k % n
    if k == 0:
        return arr
    reverse_range(arr, 0, k - 1)
    reverse_range(arr, k, n - 1)
    reverse_range(arr, 0, n - 1)
    return arr
```

## Rotate Left by k (Cyclic)

```python
def rotate_left_cyclic(arr, k):
    n = len(arr)
    if n == 0:
        return arr
    k = k % n
    return arr[k:] + arr[:k]
```

## Rotate Right by k

Right rotate by k equals left rotate by n - k.

```python
def rotate_right(arr, k):
    n = len(arr)
    if n == 0:
        return arr
    k = k % n
    return arr[n - k:] + arr[:n - k]

def rotate_right_inplace(arr, k):
    n = len(arr)
    if n == 0:
        return arr
    k = k % n
    if k == 0:
        return arr
    reverse_range(arr, 0, n - 1)
    reverse_range(arr, 0, k - 1)
    reverse_range(arr, k, n - 1)
    return arr
```

## Merge Two Sorted Arrays (With Extra Space)

Two pointers, compare and place. Time O(m+n), Space O(m+n).

```python
def merge_sorted_extra(arr1, arr2):
    result = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result
```

## Merge Two Sorted Arrays (Without Extra Space)

Merge in-place when first array has space. Use gap method or insertion. Gap method: start with gap = ceil((m+n)/2), reduce until 1, compare elements gap apart.

```python
def next_gap(gap):
    if gap <= 1:
        return 0
    return (gap // 2) + (gap % 2)

def merge_sorted_inplace(arr1, arr2):
    m, n = len(arr1), len(arr2)
    gap = m + n
    gap = next_gap(gap)
    while gap > 0:
        i = 0
        while i + gap < m:
            if arr1[i] > arr1[i + gap]:
                arr1[i], arr1[i + gap] = arr1[i + gap], arr1[i]
            i += 1
        j = gap - m if gap > m else 0
        while i < m and j < n:
            if arr1[i] > arr2[j]:
                arr1[i], arr2[j] = arr2[j], arr1[i]
            i += 1
            j += 1
        if j < n:
            j = 0
            while j + gap < n:
                if arr2[j] > arr2[j + gap]:
                    arr2[j], arr2[j + gap] = arr2[j + gap], arr2[j]
                j += 1
        gap = next_gap(gap)
    return arr1, arr2
```

## Merge Unsorted

```python
def merge_unsorted(arr1, arr2):
    return arr1 + arr2

def merge_unsorted_sorted(arr1, arr2):
    return sorted(arr1 + arr2)
```

## Remove Duplicates from Sorted (In-Place)

Two pointers: one for writing position, one for reading. Time O(n), Space O(1).

```python
def remove_duplicates_sorted(arr):
    if not arr:
        return 0
    write = 1
    for read in range(1, len(arr)):
        if arr[read] != arr[read - 1]:
            arr[write] = arr[read]
            write += 1
    return write
```

## Remove Duplicates from Unsorted

Use set to track seen. Time O(n), Space O(n).

```python
def remove_duplicates_unsorted(arr):
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def remove_duplicates_unsorted_inplace(arr):
    seen = set()
    write = 0
    for read in range(len(arr)):
        if arr[read] not in seen:
            seen.add(arr[read])
            arr[write] = arr[read]
            write += 1
    return write
```

## Remove All Occurrences of Value

```python
def remove_all_occurrences(arr, value):
    return [x for x in arr if x != value]

def remove_all_occurrences_inplace(arr, value):
    write = 0
    for read in range(len(arr)):
        if arr[read] != value:
            arr[write] = arr[read]
            write += 1
    return write
```

## Move Zeros to End

Two pointers: write position for non-zeros. Time O(n), Space O(1).

```python
def move_zeros_to_end(arr):
    write = 0
    for read in range(len(arr)):
        if arr[read] != 0:
            arr[write] = arr[read]
            write += 1
    for i in range(write, len(arr)):
        arr[i] = 0
    return arr
```

## Rearrange Positive/Negative Alternately

Partition negatives and positives, then interleave. Or use two pointers after partitioning.

```python
def rearrange_positive_negative(arr):
    neg = [x for x in arr if x < 0]
    pos = [x for x in arr if x >= 0]
    result = []
    i, j = 0, 0
    while i < len(neg) and j < len(pos):
        result.append(neg[i])
        result.append(pos[j])
        i += 1
        j += 1
    result.extend(neg[i:])
    result.extend(pos[j:])
    return result
```

## Rearrange arr[i] = i

Place each element at its value index if possible. Time O(n), Space O(1) with in-place.

```python
def rearrange_arri_equals_i(arr):
    n = len(arr)
    for i in range(n):
        while arr[i] >= 0 and arr[i] < n and arr[i] != i:
            val = arr[i]
            arr[i], arr[val] = arr[val], arr[i]
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

## Find Second Largest Without Sorting

Two passes or single pass tracking first and second. Time O(n), Space O(1).

```python
def find_second_largest(arr):
    if len(arr) < 2:
        return None
    first = second = float('-inf')
    for x in arr:
        if x > first:
            second = first
            first = x
        elif x > second and x != first:
            second = x
    return second if second != float('-inf') else None
```

## Leaders in Array

Element is leader if it is greater than all elements to its right. Traverse from right, track max.

```python
def leaders_in_array(arr):
    if not arr:
        return []
    result = []
    max_right = arr[-1]
    result.append(max_right)
    for i in range(len(arr) - 2, -1, -1):
        if arr[i] >= max_right:
            max_right = arr[i]
            result.append(arr[i])
    return result[::-1]
```

## Replace with Next Greatest

Replace each element with the greatest element to its right. Rightmost becomes -1. Traverse from right.

```python
def replace_with_next_greatest(arr):
    n = len(arr)
    if n == 0:
        return arr
    max_right = -1
    for i in range(n - 1, -1, -1):
        temp = arr[i]
        arr[i] = max_right
        max_right = max(max_right, temp)
    return arr
```

## Sort 0s 1s 2s (Dutch National Flag)

Three-way partition: low, mid, high. mid traverses. Time O(n), Space O(1).

```python
def sort_012(arr):
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

## Equilibrium Index

Index where sum of left elements equals sum of right. Prefix sum approach: total sum, then iterate checking left_sum == total - left_sum - arr[i].

```python
def equilibrium_index(arr):
    total = sum(arr)
    left_sum = 0
    for i in range(len(arr)):
        if left_sum == total - left_sum - arr[i]:
            return i
        left_sum += arr[i]
    return -1
```

## Majority Element (Boyer-Moore)

Element appearing more than n/2 times. Cancel pairs of different elements; remaining candidate is majority. Verify in second pass.

```python
def majority_element(arr):
    if not arr:
        return None
    candidate = arr[0]
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == candidate:
            count += 1
        else:
            count -= 1
            if count == 0:
                candidate = arr[i]
                count = 1
    count = sum(1 for x in arr if x == candidate)
    return candidate if count > len(arr) // 2 else None
```

## Check Sorted

```python
def check_sorted(arr):
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return False
    return True

def check_sorted_descending(arr):
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            return False
    return True
```

## Make Unique with Min Increments

Sort, then ensure each element is strictly greater than previous. Increment as needed.

```python
def make_unique_min_increments(arr):
    if not arr:
        return 0
    arr.sort()
    ops = 0
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            diff = arr[i - 1] - arr[i] + 1
            arr[i] += diff
            ops += diff
    return ops
```
