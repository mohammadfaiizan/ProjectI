# Two Pointers Technique

Two pointers maintain indices that traverse the array, often from opposite ends or at different speeds. Used for sorted arrays, pair finding, and in-place transformations.

## Two Sum (Sorted)

Array is sorted. Use left and right pointers. If sum < target, move left right; if sum > target, move right left. Time O(n), Space O(1).

```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return [left, right]
        if s < target:
            left += 1
        else:
            right -= 1
    return []
```

## Two Sum (Unsorted Pair)

Use hashmap to store seen values. For each x, check if target - x exists. Time O(n), Space O(n).

```python
def two_sum_unsorted(arr, target):
    seen = {}
    for i, x in enumerate(arr):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```

## Three Sum

Sort array. Fix first element, use two pointers for remaining two. Skip duplicates. Time O(n^2), Space O(1).

```python
def three_sum(arr, target):
    arr = sorted(arr)
    result = []
    for i in range(len(arr) - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        left, right = i + 1, len(arr) - 1
        while left < right:
            s = arr[i] + arr[left] + arr[right]
            if s == target:
                result.append([arr[i], arr[left], arr[right]])
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
    return result
```

## Three Sum Closest

Similar to three sum. Track closest sum. Time O(n^2), Space O(1).

```python
def three_sum_closest(arr, target):
    arr = sorted(arr)
    closest = float('inf')
    for i in range(len(arr) - 2):
        left, right = i + 1, len(arr) - 1
        while left < right:
            s = arr[i] + arr[left] + arr[right]
            if abs(s - target) < abs(closest - target):
                closest = s
            if s < target:
                left += 1
            elif s > target:
                right -= 1
            else:
                return s
    return closest
```

## Four Sum

Sort. Fix two elements, use two pointers for remaining two. Time O(n^3), Space O(1).

```python
def four_sum(arr, target):
    arr = sorted(arr)
    result = []
    for i in range(len(arr) - 3):
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        for j in range(i + 1, len(arr) - 2):
            if j > i + 1 and arr[j] == arr[j - 1]:
                continue
            left, right = j + 1, len(arr) - 1
            while left < right:
                s = arr[i] + arr[j] + arr[left] + arr[right]
                if s == target:
                    result.append([arr[i], arr[j], arr[left], arr[right]])
                    while left < right and arr[left] == arr[left + 1]:
                        left += 1
                    while left < right and arr[right] == arr[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif s < target:
                    left += 1
                else:
                    right -= 1
    return result
```

## Container With Most Water

Two pointers at ends. Area = min(height[left], height[right]) * (right - left). Move pointer with smaller height. Time O(n), Space O(1).

```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        max_water = max(max_water, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
```

## Trapping Rain Water

For each position, water = min(max_left, max_right) - height[i]. Two passes or two pointers with running max. Time O(n), Space O(1) with two pointers.

```python
def trap_rain_water(height):
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
```

## Remove Duplicates Sorted

Two pointers: read and write. Write only when element differs from previous. Time O(n), Space O(1).

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

## Remove Element

Two pointers: write position for non-val elements. Time O(n), Space O(1).

```python
def remove_element(arr, val):
    write = 0
    for read in range(len(arr)):
        if arr[read] != val:
            arr[write] = arr[read]
            write += 1
    return write
```

## Move Zeros

Same as remove element for 0, then fill remaining with zeros. Or swap non-zero to front. Time O(n), Space O(1).

```python
def move_zeros(arr):
    write = 0
    for read in range(len(arr)):
        if arr[read] != 0:
            arr[write], arr[read] = arr[read], arr[write]
            write += 1
    return arr
```

## Sort Colors

Dutch National Flag: three pointers low, mid, high. Time O(n), Space O(1).

```python
def sort_colors(arr):
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

## Squares of Sorted Array

Array has negatives. Two pointers at ends, compare squares, place larger at end of result. Time O(n), Space O(n).

```python
def sorted_squares(arr):
    n = len(arr)
    result = [0] * n
    left, right = 0, n - 1
    pos = n - 1
    while left <= right:
        if abs(arr[left]) > abs(arr[right]):
            result[pos] = arr[left] ** 2
            left += 1
        else:
            result[pos] = arr[right] ** 2
            right -= 1
        pos -= 1
    return result
```

## Pair with Given Difference

Sort and use two pointers. Or use hashmap. Time O(n log n) with sort, O(n) with hashmap.

```python
def pair_with_difference(arr, diff):
    arr = sorted(arr)
    left, right = 0, 1
    while right < len(arr):
        d = arr[right] - arr[left]
        if d == diff:
            return [arr[left], arr[right]]
        if d < diff:
            right += 1
        else:
            left += 1
            if left == right:
                right += 1
    return []
```

## Count Pairs with Sum

Sort, two pointers. For each left, find count of rights such that arr[left] + arr[right] == target. Handle duplicates. Time O(n log n), Space O(1).

```python
def count_pairs_with_sum(arr, target):
    arr = sorted(arr)
    left, right = 0, len(arr) - 1
    count = 0
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            if arr[left] == arr[right]:
                n = right - left + 1
                count += n * (n - 1) // 2
                break
            left_count = 1
            while left + 1 < right and arr[left] == arr[left + 1]:
                left_count += 1
                left += 1
            right_count = 1
            while right - 1 > left and arr[right] == arr[right - 1]:
                right_count += 1
                right -= 1
            count += left_count * right_count
            left += 1
            right -= 1
        elif s < target:
            left += 1
        else:
            right -= 1
    return count
```

## Dutch National Flag

Three-way partition. See Sort Colors above.

## Merge Sorted Arrays

Two pointers, compare and merge. When one array has extra space, merge from end. Time O(m+n), Space O(1) when merging into larger array.

```python
def merge_sorted(arr1, m, arr2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while i >= 0 and j >= 0:
        if arr1[i] > arr2[j]:
            arr1[k] = arr1[i]
            i -= 1
        else:
            arr1[k] = arr2[j]
            j -= 1
        k -= 1
    while j >= 0:
        arr1[k] = arr2[j]
        j -= 1
        k -= 1
    return arr1
```

## Boats to Save People

Sort. Heaviest with lightest if sum <= limit, else heaviest alone. Two pointers. Time O(n log n), Space O(1).

```python
def num_rescue_boats(people, limit):
    people = sorted(people)
    left, right = 0, len(people) - 1
    boats = 0
    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1
    return boats
```

## Minimize Max Pair Sum

Sort. Pair smallest with largest. Two pointers. Time O(n log n), Space O(1).

```python
def min_pair_sum(arr):
    arr = sorted(arr)
    left, right = 0, len(arr) - 1
    max_sum = 0
    while left < right:
        max_sum = max(max_sum, arr[left] + arr[right])
        left += 1
        right -= 1
    return max_sum
```

## Bag of Tokens

Sort. Use smallest tokens for power, largest for score. Two pointers. Time O(n log n), Space O(1).

```python
def bag_of_tokens_score(tokens, power):
    tokens = sorted(tokens)
    left, right = 0, len(tokens) - 1
    score = 0
    max_score = 0
    while left <= right:
        if power >= tokens[left]:
            power -= tokens[left]
            score += 1
            left += 1
            max_score = max(max_score, score)
        elif score > 0:
            power += tokens[right]
            score -= 1
            right -= 1
        else:
            break
    return max_score
```

## Partition Labels

For each char, track last occurrence. Two pointers: start and end of current partition. Extend end when we see chars with larger last index. Time O(n), Space O(1) for 26 chars.

```python
def partition_labels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result
```
