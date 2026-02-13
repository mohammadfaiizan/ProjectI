# Hard Array Problems

## 1. Trapping Rain Water

Water trapped between bars. Two pointers or stack. Track left_max and right_max.

```python
def trap(height):
    if not height:
        return 0
    l, r = 0, len(height) - 1
    lm, rm = 0, 0
    water = 0
    while l < r:
        if height[l] < height[r]:
            lm = max(lm, height[l])
            water += lm - height[l]
            l += 1
        else:
            rm = max(rm, height[r])
            water += rm - height[r]
            r -= 1
    return water
```

Time: O(n) | Space: O(1)

---

## 2. First Missing Positive

Smallest positive integer not in array. Cyclic sort for values in [1,n].

```python
def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1
```

Time: O(n) | Space: O(1)

---

## 3. Merge k Sorted Lists/Arrays

Merge k sorted arrays. Min-heap of (value, array_index, element_index).

```python
def merge_k_sorted(arrays):
    import heapq
    h = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(h, (arr[0], i, 0))
    out = []
    while h:
        val, ai, ei = heapq.heappop(h)
        out.append(val)
        if ei + 1 < len(arrays[ai]):
            heapq.heappush(h, (arrays[ai][ei + 1], ai, ei + 1))
    return out
```

Time: O(N log k) | Space: O(k)

---

## 4. Median of Two Sorted Arrays

Find median of two sorted arrays. Binary search on smaller array for partition.

```python
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = (m + n + 1) // 2 - i
        l1 = float('-inf') if i == 0 else nums1[i - 1]
        r1 = float('inf') if i == m else nums1[i]
        l2 = float('-inf') if j == 0 else nums2[j - 1]
        r2 = float('inf') if j == n else nums2[j]
        if l1 <= r2 and l2 <= r1:
            mid = max(l1, l2)
            if (m + n) % 2:
                return mid
            return (mid + min(r1, r2)) / 2
        if l1 > r2:
            hi = i - 1
        else:
            lo = i + 1
```

Time: O(log(min(m,n))) | Space: O(1)

---

## 5. Sliding Window Maximum

Max in each sliding window of size k. Monotonic deque.

```python
def max_sliding_window(nums, k):
    from collections import deque
    dq = deque()
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
```

Time: O(n) | Space: O(k)

---

## 6. Minimum Window Substring

Smallest substring of s containing all chars of t. Sliding window with frequency maps.

```python
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    have = 0
    need_cnt = len(need)
    start, length = 0, float('inf')
    l = 0
    for r, c in enumerate(s):
        if c in need:
            need[c] -= 1
            if need[c] == 0:
                have += 1
        while have == need_cnt:
            if r - l + 1 < length:
                start, length = l, r - l + 1
            if s[l] in need:
                need[s[l]] += 1
                if need[s[l]] > 0:
                    have -= 1
            l += 1
    return s[start:start + length] if length != float('inf') else ""
```

Time: O(n + m) | Space: O(m)

---

## 7. Substring with Concatenation of All Words

Find start indices where substring is concatenation of all words. Sliding window per word length.

```python
def find_substring(s, words):
    from collections import Counter
    if not words:
        return []
    wlen, nw = len(words[0]), len(words)
    target = Counter(words)
    out = []
    for start in range(wlen):
        seen = Counter()
        cnt = 0
        for i in range(start, len(s) - wlen + 1, wlen):
            w = s[i:i + wlen]
            if w in target:
                seen[w] += 1
                cnt += 1
                while seen[w] > target[w]:
                    first = s[start:start + wlen]
                    seen[first] -= 1
                    cnt -= 1
                    start += wlen
                if cnt == nw:
                    out.append(start)
            else:
                seen.clear()
                cnt = 0
                start = i + wlen
    return out
```

Time: O(n * wlen) | Space: O(nw)

---

## 8. Longest Consecutive Sequence

Longest consecutive integer sequence. Union-Find or set with expansion.

```python
def longest_consecutive(nums):
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:
            cur = 1
            while x + cur in s:
                cur += 1
            best = max(best, cur)
    return best
```

Time: O(n) | Space: O(n)

---

## 9. Two Sum - Data Structure Design

Add and find. Hashmap for values, on find check complement.

```python
class TwoSum:
    def __init__(self):
        self.cnt = {}
    def add(self, val):
        self.cnt[val] = self.cnt.get(val, 0) + 1
    def find(self, target):
        for x in self.cnt:
            y = target - x
            if y in self.cnt and (x != y or self.cnt[x] > 1):
                return True
        return False
```

Time: O(1) add, O(n) find | Space: O(n)

---

## 10. Max Points on a Line

Max collinear points. For each point, count slopes to others. Handle duplicates.

```python
def max_points(points):
    from collections import defaultdict
    from math import gcd
    n = len(points)
    if n <= 2:
        return n
    best = 0
    for i in range(n):
        slopes = defaultdict(int)
        dup = 0
        for j in range(n):
            if i == j:
                continue
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                dup += 1
                continue
            g = gcd(dx, dy)
            if g:
                dx, dy = dx // g, dy // g
            slopes[(dx, dy)] += 1
        best = max(best, 1 + dup + (max(slopes.values()) if slopes else 0))
    return best
```

Time: O(n^2) | Space: O(n)

---

## 11. Candy

Distribute candy: adjacent ratings get different amounts. Two passes: left-to-right and right-to-left.

```python
def candy(ratings):
    n = len(ratings)
    candies = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    return sum(candies)
```

Time: O(n) | Space: O(n)

---

## 12. Product of Array Except Self (with division restriction)

No division. Prefix and suffix products in two passes.

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

## 13. Maximum Gap

Max difference between successive elements in sorted form. Bucket sort with n+1 buckets.

```python
def maximum_gap(nums):
    if len(nums) < 2:
        return 0
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return 0
    n = len(nums)
    size = (hi - lo) // n + 1
    buckets = [[float('inf'), float('-inf')] for _ in range(n + 1)]
    for x in nums:
        b = (x - lo) // size
        buckets[b][0] = min(buckets[b][0], x)
        buckets[b][1] = max(buckets[b][1], x)
    prev = lo
    best = 0
    for mn, mx in buckets:
        if mn != float('inf'):
            best = max(best, mn - prev)
            prev = mx
    return best
```

Time: O(n) | Space: O(n)

---

## 14. Create Maximum Number

Form k-digit number from two arrays. Greedy: for each split, take largest subsequence from each, merge.

```python
def max_number(nums1, nums2, k):
    def pick(arr, k):
        drop = len(arr) - k
        st = []
        for x in arr:
            while drop and st and st[-1] < x:
                st.pop()
                drop -= 1
            st.append(x)
        return st[:k]
    def merge(a, b):
        return [max(a, b).pop(0) for _ in a + b]
    best = []
    for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
        cand = merge(pick(nums1, i), pick(nums2, k - i))
        best = max(best, cand)
    return best
```

Time: O(k * (n + m)) | Space: O(k)

---

## 15. Count of Smaller Numbers After Self

For each element, count smaller elements to the right. Merge sort with inversion count or BST.

```python
def count_smaller(nums):
    from bisect import bisect_left
    sorted_vals = []
    out = []
    for x in reversed(nums):
        i = bisect_left(sorted_vals, x)
        out.append(i)
        sorted_vals.insert(i, x)
    return out[::-1]
```

Time: O(n^2) | Space: O(n)

---

## 16. Sliding Window Median

Median in each sliding window. Two heaps (max heap for lower half, min for upper) or multiset.

```python
def median_sliding_window(nums, k):
    import heapq
    lo, hi = [], []
    def add(x):
        heapq.heappush(lo, -x)
        heapq.heappush(hi, -heapq.heappop(lo))
        if len(lo) < len(hi):
            heapq.heappush(lo, -heapq.heappop(hi))
    def remove(x):
        if x <= -lo[0]:
            lo.remove(-x)
            heapq.heapify(lo)
        else:
            hi.remove(x)
            heapq.heapify(hi)
    def median():
        return -lo[0] if k % 2 else (-lo[0] + hi[0]) / 2
    for i in range(k):
        add(nums[i])
    out = [median()]
    for i in range(k, len(nums)):
        remove(nums[i - k])
        add(nums[i])
        out.append(median())
    return out
```

Time: O(n log k) | Space: O(k)

---

## 17. Shortest Subarray with Sum at Least K

Array may have negatives. Monotonic deque of prefix sums.

```python
def shortest_subarray(nums, k):
    from collections import deque
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)
    dq = deque()
    best = float('inf')
    for i, p in enumerate(pre):
        while dq and p - pre[dq[0]] >= k:
            best = min(best, i - dq.popleft())
        while dq and pre[dq[-1]] >= p:
            dq.pop()
        dq.append(i)
    return best if best != float('inf') else -1
```

Time: O(n) | Space: O(n)

---

## 18. Subarray Sum Equals K (with negatives)

Prefix sum + hashmap. Same approach works with negatives.

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

## 19. Longest Increasing Path in Matrix

DFS with memoization. Explore from each cell, cache longest path from that cell.

```python
def longest_increasing_path(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}
    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        best = 1
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        memo[(i, j)] = best
        return best
    return max(dfs(i, j) for i in range(m) for j in range(n))
```

Time: O(m * n) | Space: O(m * n)

---

## 20. Max Sum of Rectangle No Larger Than K

2D Kadane + TreeSet for subarray sum <= k. Iterate row ranges, compress to 1D.

```python
def max_sum_submatrix(matrix, k):
    import bisect
    m, n = len(matrix), len(matrix[0])
    best = float('-inf')
    for r1 in range(m):
        row_sums = [0] * n
        for r2 in range(r1, m):
            for c in range(n):
                row_sums[c] += matrix[r2][c]
            pre = 0
            seen = [0]
            for x in row_sums:
                pre += x
                i = bisect.bisect_left(seen, pre - k)
                if i < len(seen):
                    best = max(best, pre - seen[i])
                bisect.insort(seen, pre)
    return best
```

Time: O(m^2 * n log n) | Space: O(n)
