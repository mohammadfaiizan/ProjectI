# Medium Array Problems

## 1. Three Sum

Find all unique triplets with sum 0. Sort, fix one, two pointers for rest.

```python
def three_sum(nums):
    nums.sort()
    out = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        j, k = i + 1, len(nums) - 1
        while j < k:
            s = nums[i] + nums[j] + nums[k]
            if s == 0:
                out.append([nums[i], nums[j], nums[k]])
                while j < k and nums[j] == nums[j + 1]:
                    j += 1
                while j < k and nums[k] == nums[k - 1]:
                    k -= 1
                j += 1
                k -= 1
            elif s < 0:
                j += 1
            else:
                k -= 1
    return out
```

Time: O(n^2) | Space: O(1)

---

## 2. Container With Most Water

Two lines form container. Two pointers at ends, move smaller inward.

```python
def max_area(height):
    i, j, best = 0, len(height) - 1, 0
    while i < j:
        best = max(best, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return best
```

Time: O(n) | Space: O(1)

---

## 3. Product of Array Except Self

Output[i] = product of all except arr[i]. Prefix and suffix products.

```python
def product_except_self(nums):
    n = len(nums)
    out = [1] * n
    for i in range(1, n):
        out[i] = out[i - 1] * nums[i - 1]
    suf = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suf
        suf *= nums[i]
    return out
```

Time: O(n) | Space: O(1)

---

## 4. Maximum Product Subarray

Contiguous subarray with max product. Track max and min (for negative).

```python
def max_product(nums):
    best = mx = mn = nums[0]
    for x in nums[1:]:
        mx, mn = max(x, mx * x, mn * x), min(x, mx * x, mn * x)
        best = max(best, mx)
    return best
```

Time: O(n) | Space: O(1)

---

## 5. Find Minimum in Rotated Sorted Array

Sorted array rotated. Binary search: compare mid with right to decide direction.

```python
def find_min_rotated(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid
    return nums[lo]
```

Time: O(log n) | Space: O(1)

---

## 6. Search in Rotated Sorted Array

Binary search with rotation handling. Compare mid with left/right to find sorted half.

```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

Time: O(log n) | Space: O(1)

---

## 7. Find First and Last Position of Element

Binary search for leftmost and rightmost occurrence.

```python
def search_range(nums, target):
    def bisect_left():
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    def bisect_right():
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    left = bisect_left()
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, bisect_right() - 1]
```

Time: O(log n) | Space: O(1)

---

## 8. Combination Sum

Find all combinations that sum to target. Backtracking with pruning.

```python
def combination_sum(candidates, target):
    out = []
    def bt(start, path, rem):
        if rem == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] <= rem:
                path.append(candidates[i])
                bt(i, path, rem - candidates[i])
                path.pop()
    bt(0, [], target)
    return out
```

Time: O(2^n) | Space: O(target)

---

## 9. Combination Sum II

Same but each element used once, no duplicate combinations. Sort and skip duplicates.

```python
def combination_sum2(candidates, target):
    candidates.sort()
    out = []
    def bt(start, path, rem):
        if rem == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            if candidates[i] <= rem:
                path.append(candidates[i])
                bt(i + 1, path, rem - candidates[i])
                path.pop()
    bt(0, [], target)
    return out
```

Time: O(2^n) | Space: O(n)

---

## 10. Jump Game

Can you reach last index? Track max reachable, greedy.

```python
def can_jump(nums):
    reach = 0
    for i, x in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + x)
        if reach >= len(nums) - 1:
            return True
    return True
```

Time: O(n) | Space: O(1)

---

## 11. Jump Game II

Minimum jumps to reach end. BFS or greedy: extend reach each step.

```python
def jump(nums):
    jumps = end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == end:
            jumps += 1
            end = farthest
    return jumps
```

Time: O(n) | Space: O(1)

---

## 12. Merge Intervals

Merge overlapping intervals. Sort by start, merge if overlap.

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
```

Time: O(n log n) | Space: O(n)

---

## 13. Insert Interval

Insert new interval into sorted non-overlapping intervals. Find position, merge.

```python
def insert_interval(intervals, new):
    out = []
    for s, e in intervals:
        if e < new[0]:
            out.append([s, e])
        elif s > new[1]:
            out.append(new)
            new = [s, e]
        else:
            new = [min(s, new[0]), max(e, new[1])]
    out.append(new)
    return out
```

Time: O(n) | Space: O(n)

---

## 14. Spiral Matrix

Traverse matrix in spiral order. Layer by layer with boundaries.

```python
def spiral_order(matrix):
    if not matrix:
        return []
    r1, r2, c1, c2 = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    out = []
    while r1 <= r2 and c1 <= c2:
        for c in range(c1, c2 + 1):
            out.append(matrix[r1][c])
        for r in range(r1 + 1, r2 + 1):
            out.append(matrix[r][c2])
        if r1 < r2 and c1 < c2:
            for c in range(c2 - 1, c1 - 1, -1):
                out.append(matrix[r2][c])
            for r in range(r2 - 1, r1, -1):
                out.append(matrix[r][c1])
        r1, r2, c1, c2 = r1 + 1, r2 - 1, c1 + 1, c2 - 1
    return out
```

Time: O(m * n) | Space: O(1)

---

## 15. Rotate Image

Rotate matrix 90 degrees in-place. Transpose then reverse rows.

```python
def rotate_image(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
```

Time: O(n^2) | Space: O(1)

---

## 16. Group Anagrams

Group strings by anagram. Use sorted string or char count as key.

```python
def group_anagrams(strs):
    from collections import defaultdict
    d = defaultdict(list)
    for s in strs:
        d[tuple(sorted(s))].append(s)
    return list(d.values())
```

Time: O(n * k log k) | Space: O(n)

---

## 17. Subarray Sum Equals K

Count subarrays with sum k. Prefix sum + hashmap.

```python
def subarray_sum(nums, k):
    from collections import defaultdict
    pre, cnt, d = 0, 0, defaultdict(int)
    d[0] = 1
    for x in nums:
        pre += x
        cnt += d.get(pre - k, 0)
        d[pre] += 1
    return cnt
```

Time: O(n) | Space: O(n)

---

## 18. Longest Substring Without Repeating Characters

Sliding window with hashset for seen chars.

```python
def length_of_longest_substring(s):
    seen = {}
    start = best = 0
    for i, c in enumerate(s):
        if c in seen and seen[c] >= start:
            start = seen[c] + 1
        seen[c] = i
        best = max(best, i - start + 1)
    return best
```

Time: O(n) | Space: O(min(n, charset))

---

## 19. Longest Palindromic Substring

Expand around center for each position. Odd and even length.

```python
def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]
    best = ""
    for i in range(len(s)):
        best = max(expand(i, i), expand(i, i + 1), best, key=len)
    return best
```

Time: O(n^2) | Space: O(1)

---

## 20. Next Permutation

In-place next lexicographic permutation. Find first decreasing from right, swap with next larger, reverse suffix.

```python
def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = reversed(nums[i + 1:])
```

Time: O(n) | Space: O(1)

---

## 21. Sort Colors (Dutch National Flag)

Three-way partition for 0, 1, 2.

```python
def sort_colors(nums):
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1
            mid += 1
        elif nums[mid] == 2:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
        else:
            mid += 1
```

Time: O(n) | Space: O(1)

---

## 22. Top K Frequent Elements

Bucket sort by frequency or heap. O(n) with bucket sort.

```python
def top_k_frequent(nums, k):
    from collections import Counter
    cnt = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for x, f in cnt.items():
        buckets[f].append(x)
    out = []
    for b in reversed(buckets):
        out.extend(b)
        if len(out) >= k:
            return out[:k]
    return out
```

Time: O(n) | Space: O(n)

---

## 23. Kth Largest Element

Quickselect or heap. Partition around pivot.

```python
def find_kth_largest(nums, k):
    import heapq
    return heapq.nlargest(k, nums)[-1]
```

Time: O(n log k) | Space: O(k)

---

## 24. Find Peak Element

Binary search: if mid < mid+1, peak in right half; else left.

```python
def find_peak_element(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

Time: O(log n) | Space: O(1)

---

## 25. Search a 2D Matrix

Sorted matrix. Binary search treating as 1D.

```python
def search_matrix(matrix, target):
    if not matrix:
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

Time: O(log(m * n)) | Space: O(1)

---

## 26. Set Matrix Zeroes

Set row and col to 0 if element is 0. Use first row/col as markers.

```python
def set_zeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    row0 = col0 = False
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                if i == 0:
                    row0 = True
                if j == 0:
                    col0 = True
                matrix[i][0] = matrix[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if row0:
        for j in range(n):
            matrix[0][j] = 0
    if col0:
        for i in range(m):
            matrix[i][0] = 0
```

Time: O(m * n) | Space: O(1)

---

## 27. Spiral Matrix II

Generate n x n matrix with values 1 to n^2 in spiral order.

```python
def generate_spiral(n):
    mat = [[0] * n for _ in range(n)]
    r1, r2, c1, c2, v = 0, n - 1, 0, n - 1, 1
    while r1 <= r2 and c1 <= c2:
        for c in range(c1, c2 + 1):
            mat[r1][c] = v
            v += 1
        for r in range(r1 + 1, r2 + 1):
            mat[r][c2] = v
            v += 1
        if r1 < r2 and c1 < c2:
            for c in range(c2 - 1, c1 - 1, -1):
                mat[r2][c] = v
                v += 1
            for r in range(r2 - 1, r1, -1):
                mat[r][c1] = v
                v += 1
        r1, r2, c1, c2 = r1 + 1, r2 - 1, c1 + 1, c2 - 1
    return mat
```

Time: O(n^2) | Space: O(1)

---

## 28. Unique Paths

Grid paths from top-left to bottom-right. DP or combinatorics.

```python
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
```

Time: O(m * n) | Space: O(n)

---

## 29. Minimum Path Sum

Min sum path in grid. DP with min of up/left.

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    return grid[-1][-1]
```

Time: O(m * n) | Space: O(1)

---

## 30. Rotate Array

Rotate right by k. Reversal algorithm or cyclic.

```python
def rotate(nums, k):
    k %= len(nums)
    def rev(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
    rev(0, len(nums) - 1)
    rev(0, k - 1)
    rev(k, len(nums) - 1)
```

Time: O(n) | Space: O(1)
