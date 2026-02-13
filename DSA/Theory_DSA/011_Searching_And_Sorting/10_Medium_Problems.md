# Medium and Hard Searching and Sorting Problems

## Medium Problems

## 1. Search in Rotated Sorted Array

Sorted array rotated at unknown pivot. Find target in O(log n). Binary search. One half is always sorted. Check which half contains target.

```python
def search(nums, target):
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

## 2. Search in Rotated Sorted Array II

Same as above with duplicates. Return boolean. Handle arr[left]==arr[mid]==arr[right] by shrinking both ends.

```python
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return True
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1
            hi -= 1
        elif nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False
```

Time: O(n) | Space: O(1)

---

## 3. Find Minimum in Rotated Sorted Array

Rotated sorted array, find minimum element. Binary search. Compare arr[mid] with arr[right]. Min in unsorted half.

```python
def findMin(nums):
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

## 4. Find First and Last Position of Element

Sorted array with duplicates. Find [first, last] index of target. Two binary searches: first occurrence and last occurrence.

```python
def searchRange(nums, target):
    def left_bound():
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def right_bound():
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    left = left_bound()
    if left >= len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, right_bound()]
```

Time: O(log n) | Space: O(1)

---

## 5. Search a 2D Matrix

2D matrix sorted row-wise (each row > previous). Find target. Flatten to 1D, binary search. row=mid//cols, col=mid%cols.

```python
def searchMatrix(matrix, target):
    if not matrix:
        return False
    m, n = len(matrix), len(matrix[0])
    lo, hi = 0, m * n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = matrix[mid // n][mid % n]
        if val == target:
            return True
        if val < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False
```

Time: O(log(mn)) | Space: O(1)

---

## 6. Search a 2D Matrix II

Each row and column sorted. Find target. Start top-right. If target > val, move down. If target < val, move left.

```python
def searchMatrix(matrix, target):
    if not matrix:
        return False
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target:
            return True
        if matrix[r][c] < target:
            r += 1
        else:
            c -= 1
    return False
```

Time: O(m + n) | Space: O(1)

---

## 7. Koko Eating Bananas

Piles, h hours. Min eating speed k to finish all. Binary search on answer. Feasible(k): sum(ceil(pile/k)) <= h.

```python
def minEatingSpeed(piles, h):
    def feasible(k):
        return sum((p + k - 1) // k for p in piles) <= h

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Time: O(n log max) | Space: O(1)

---

## 8. Capacity to Ship Packages Within D Days

Weights, d days. Min capacity to ship all. Binary search on capacity. Feasible: greedy load, count days.

```python
def shipWithinDays(weights, days):
    def feasible(cap):
        cur, d = 0, 1
        for w in weights:
            if cur + w > cap:
                d += 1
                cur = w
            else:
                cur += w
        return d <= days

    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Time: O(n log sum) | Space: O(1)

---

## 9. Split Array Largest Sum

Split into k subarrays. Minimize largest sum. Binary search on max sum. Feasible: greedy split.

```python
def splitArray(nums, k):
    def feasible(mx):
        cur, cnt = 0, 1
        for x in nums:
            if cur + x > mx:
                cnt += 1
                cur = x
            else:
                cur += x
        return cnt <= k

    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Time: O(n log sum) | Space: O(1)

---

## 10. Find Peak Element

Array with arr[i] != arr[i+1]. Find any peak index. Binary search. If arr[mid] < arr[mid+1], peak in right half.

```python
def findPeakElement(nums):
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

## 11. Find Right Interval

Intervals. For each, find smallest start_j >= end_i. Sort by start with original index. Binary search for each end_i.

```python
def findRightInterval(intervals):
    sorted_arr = sorted((intervals[i][0], i) for i in range(len(intervals)))
    res = []
    for _, _, end in intervals:
        lo, hi = 0, len(sorted_arr) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_arr[mid][0] >= end:
                idx = sorted_arr[mid][1]
                hi = mid - 1
            else:
                lo = mid + 1
        res.append(idx)
    return res
```

Time: O(n log n) | Space: O(n)

---

## 12. Find K Closest Elements

Sorted array, x, k. Return k closest elements to x. Binary search for left boundary of optimal window of size k.

```python
def findClosestElements(arr, k, x):
    lo, hi = 0, len(arr) - k
    while lo < hi:
        mid = (lo + hi) // 2
        if x - arr[mid] > arr[mid + k] - x:
            lo = mid + 1
        else:
            hi = mid
    return arr[lo:lo + k]
```

Time: O(log n) | Space: O(1)

---

## 13. Single Element in Sorted Array

Every element appears twice except one. Find it in O(log n). Binary search on pair parity. Before single, pairs at (even, odd). After, (odd, even).

```python
def singleNonDuplicate(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid % 2 == 1:
            mid -= 1
        if nums[mid] == nums[mid + 1]:
            lo = mid + 2
        else:
            hi = mid
    return nums[lo]
```

Time: O(log n) | Space: O(1)

---

## 14. Successful Pairs of Spells and Potions

spell[i]*potion[j] >= success. Count pairs per spell. Sort potions. Binary search for min potion: ceil(success/spell).

```python
def successfulPairs(spells, potions, success):
    potions.sort()
    n = len(potions)
    res = []
    for s in spells:
        need = (success + s - 1) // s
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if potions[mid] < need:
                lo = mid + 1
            else:
                hi = mid
        res.append(n - lo)
    return res
```

Time: O(m log n) | Space: O(1)

---

## 15. Time Based Key-Value Store

set(key, value, timestamp), get(key, timestamp) returns value with largest timestamp <= given. Store list of (timestamp, value) per key. Binary search for get.

```python
from bisect import bisect_right

class TimeMap:
    def __init__(self):
        self.store = {}

    def set(self, key, value, timestamp):
        if key not in self.store:
            self.store[key] = []
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        if key not in self.store:
            return ""
        arr = self.store[key]
        i = bisect_right(arr, (timestamp, chr(127))) - 1
        return arr[i][1] if i >= 0 else ""
```

Time: O(log n) | Space: O(n)

---

## 16. H-Index

Citations array. Find h: h papers have >= h citations. Sort descending. Binary search for largest i with citations[i] >= i+1.

```python
def hIndex(citations):
    citations.sort(reverse=True)
    for i, c in enumerate(citations):
        if c < i + 1:
            return i
    return len(citations)
```

Time: O(n log n) | Space: O(1)

---

## 17. Count of Smaller Numbers After Self

For each element, count how many smaller elements to the right. Merge sort with inversion count, or binary search insertion from right.

```python
def countSmaller(nums):
    import bisect
    sorted_nums = []
    res = []
    for x in reversed(nums):
        i = bisect.bisect_left(sorted_nums, x)
        res.append(i)
        bisect.insort(sorted_nums, x)
    return res[::-1]
```

Time: O(n log n) | Space: O(n)

---

## 18. Sort List

Sort linked list in O(n log n) time, O(1) space. Merge sort on linked list. Find mid with slow/fast, merge two halves.

```python
def sortList(head):
    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    mid = slow.next
    slow.next = None
    left = sortList(head)
    right = sortList(mid)
    dummy = ListNode()
    cur = dummy
    while left and right:
        if left.val <= right.val:
            cur.next = left
            left = left.next
        else:
            cur.next = right
            right = right.next
        cur = cur.next
    cur.next = left or right
    return dummy.next
```

Time: O(n log n) | Space: O(log n)

---

## 19. Sort Colors (Dutch National Flag)

Array of 0, 1, 2. Sort in-place one pass. Three pointers: low, mid, high. Swap based on arr[mid].

```python
def sortColors(nums):
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

## 20. Top K Frequent Elements

Return k most frequent elements. Count frequency. Bucket sort by frequency or quickselect.

```python
def topKFrequent(nums, k):
    from collections import Counter
    buckets = [[] for _ in range(len(nums) + 1)]
    for x, c in Counter(nums).items():
        buckets[c].append(x)
    out = []
    for i in range(len(nums), 0, -1):
        out.extend(buckets[i])
        if len(out) >= k:
            return out[:k]
    return out
```

Time: O(n) | Space: O(n)

---

## 21. Kth Largest Element in Array

Find kth largest element. Quickselect (partition like quicksort). Or heap of size k.

```python
def findKthLargest(nums, k):
    def partition(lo, hi):
        pivot = nums[hi]
        i = lo
        for j in range(lo, hi):
            if nums[j] >= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    lo, hi = 0, len(nums) - 1
    k = k - 1
    while True:
        p = partition(lo, hi)
        if p == k:
            return nums[p]
        if p < k:
            lo = p + 1
        else:
            hi = p - 1
```

Time: O(n) avg | Space: O(1)

---

## 22. Merge Intervals

Overlapping intervals, merge all overlapping. Sort by start. Iterate, merge if current overlaps with last in result.

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res
```

Time: O(n log n) | Space: O(1)

---

## 23. Non-overlapping Intervals

Remove minimum intervals to make non-overlapping. Sort by end. Greedy: keep interval if start >= last_end.

```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    end, count = float('-inf'), 0
    for s, e in intervals:
        if s >= end:
            end = e
        else:
            count += 1
    return count
```

Time: O(n log n) | Space: O(1)

---

## 24. Insert Interval

Non-overlapping intervals sorted. Insert new interval, merge if needed. Binary search for position. Merge overlapping. Or linear scan.

```python
def insert(intervals, newInterval):
    res = []
    for s, e in intervals:
        if e < newInterval[0]:
            res.append([s, e])
        elif s > newInterval[1]:
            res.append(newInterval)
            newInterval = [s, e]
        else:
            newInterval = [min(s, newInterval[0]), max(e, newInterval[1])]
    res.append(newInterval)
    return res
```

Time: O(n) | Space: O(1)

---

## 25. Minimum Number of Arrows to Burst Balloons

Intervals (balloons). Find min arrows (points) to hit all. Sort by end. Greedy: arrow at first end, skip all containing it.

```python
def findMinArrowShots(points):
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows, end = 1, points[0][1]
    for s, e in points[1:]:
        if s > end:
            arrows += 1
            end = e
    return arrows
```

Time: O(n log n) | Space: O(1)

---

## Hard Problems

## 1. Median of Two Sorted Arrays

Find median of two sorted arrays in O(log(min(n,m))). Binary search partition in smaller array. Partition larger so left half has (n+m+1)//2 elements. Check max_left <= min_right.

```python
def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = (m + n + 1) // 2 - i
        left1 = nums1[i-1] if i else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j-1] if j else float('-inf')
        right2 = nums2[j] if j < n else float('inf')
        if left1 <= right2 and left2 <= right1:
            if (m + n) % 2:
                return max(left1, left2)
            return (max(left1, left2) + min(right1, right2)) / 2
        if left1 > right2:
            hi = i - 1
        else:
            lo = i + 1
```

Time: O(log(min(m,n))) | Space: O(1)

---

## 2. Find Minimum in Rotated Sorted Array II

Rotated sorted with duplicates. Find minimum. When arr[mid]==arr[right], right--. Cannot discard half.

```python
def findMin(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1
        elif nums[mid] < nums[hi]:
            hi = mid
        else:
            hi -= 1
    return nums[lo]
```

Time: O(n) | Space: O(1)

---

## 3. Count of Range Sum

Count ranges [i,j] where lower <= sum <= upper. Prefix sums. Merge sort with counting. For each left prefix, count right prefixes in [lower-left, upper-left].

```python
def countRangeSum(nums, lower, upper):
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)

    def merge_count(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        count = merge_count(lo, mid) + merge_count(mid, hi)
        i = j = mid
        for k in range(lo, mid):
            while i < hi and pre[i] - pre[k] < lower:
                i += 1
            while j < hi and pre[j] - pre[k] <= upper:
                j += 1
            count += j - i
        pre[lo:hi] = sorted(pre[lo:hi])
        return count
    return merge_count(0, len(pre))
```

Time: O(n log n) | Space: O(n)

---

## 4. Reverse Pairs

Count pairs (i,j) with i<j and arr[i] > 2*arr[j]. Merge sort. During merge, for each left element, count right elements < left/2.

```python
def reversePairs(nums):
    def merge_count(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        count = merge_count(lo, mid) + merge_count(mid, hi)
        j = mid
        for i in range(lo, mid):
            while j < hi and nums[j] < nums[i] / 2:
                j += 1
            count += j - mid
        nums[lo:hi] = sorted(nums[lo:hi])
        return count
    return merge_count(0, len(nums))
```

Time: O(n log n) | Space: O(n)

---

## 5. Max Sum of Rectangle No Larger Than K

2D matrix. Find max sum subrectangle <= k. Fix left and right columns, compute row sums. Find max subarray sum <= k using prefix and binary search.

```python
def maxSumSubmatrix(matrix, k):
    import bisect
    m, n = len(matrix), len(matrix[0])
    res = float('-inf')
    for c1 in range(n):
        row_sum = [0] * m
        for c2 in range(c1, n):
            for r in range(m):
                row_sum[r] += matrix[r][c2]
            pre = [0]
            for s in row_sum:
                pre.append(pre[-1] + s)
            for i in range(len(pre)):
                target = pre[i] - k
                j = bisect.bisect_left(pre, target)
                if j < len(pre):
                    res = max(res, pre[i] - pre[j])
    return res
```

Time: O(n^2 * m log m) | Space: O(m)

---

## 6. Minimum Window Substring

String s, t. Find min substring of s containing all chars of t. Sliding window. Expand until valid, then contract. Track char counts.

```python
def minWindow(s, t):
    from collections import Counter
    need = Counter(t)
    have, need_count = 0, len(need)
    res, res_len = "", float('inf')
    l = 0
    for r, c in enumerate(s):
        if c in need:
            need[c] -= 1
            if need[c] == 0:
                have += 1
        while have == need_count:
            if r - l + 1 < res_len:
                res_len = r - l + 1
                res = s[l:r+1]
            if s[l] in need:
                need[s[l]] += 1
                if need[s[l]] > 0:
                    have -= 1
            l += 1
    return res
```

Time: O(n + m) | Space: O(m)

---

## 7. Sliding Window Maximum

Array, window size k. Max in each window. Deque maintaining decreasing order. Front is max. Remove indices outside window.

```python
def maxSlidingWindow(nums, k):
    from collections import deque
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

Time: O(n) | Space: O(k)

---

## 8. Find Median from Data Stream

Add numbers, return median. Two heaps: max-heap for lower half, min-heap for upper half. Balance sizes.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []
        self.hi = []

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

Time: O(log n) | Space: O(n)

---

## 9. Merge k Sorted Lists

k linked lists, merge into one sorted list. Min-heap of (val, list_idx, node). Extract min, add next from same list.

```python
def mergeKLists(lists):
    import heapq
    heap = []
    for i, L in enumerate(lists):
        if L:
            heapq.heappush(heap, (L.val, i, L))
    dummy = ListNode()
    cur = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

Time: O(n log k) | Space: O(k)

---

## 10. Kth Smallest in Sorted Matrix

n*n matrix sorted row and column. Find kth smallest. Binary search on value. Count elements <= mid. If count >= k, answer <= mid.

```python
def kthSmallest(matrix, k):
    n = len(matrix)
    def count_le(mid):
        r, c, cnt = 0, n - 1, 0
        while r < n and c >= 0:
            if matrix[r][c] <= mid:
                cnt += c + 1
                r += 1
            else:
                c -= 1
        return cnt

    lo, hi = matrix[0][0], matrix[-1][-1]
    while lo < hi:
        mid = (lo + hi) // 2
        if count_le(mid) < k:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

Time: O(n log(max-min)) | Space: O(1)

---

## 11. Split Array Largest Sum (Hard variant)

Same as medium but with additional constraints. Binary search on answer with feasibility check.

```python
def splitArray(nums, k):
    def feasible(mx):
        cur, cnt = 0, 1
        for x in nums:
            if cur + x > mx:
                cnt += 1
                cur = x
            else:
                cur += x
        return cnt <= k

    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Time: O(n log sum) | Space: O(1)

---

## 12. Minimum Cost to Hire K Workers

Workers with quality and wage. Hire k such that ratio wage/quality is same. Minimize total cost. Sort by ratio. For each worker as captain, take k-1 workers with smallest quality from those with lower ratio. Use heap.

```python
def mincostToHireWorkers(quality, wage, k):
    import heapq
    workers = sorted((w / q, q, w) for q, w in zip(quality, wage))
    heap = []
    sum_q = 0
    res = float('inf')
    for r, q, w in workers:
        heapq.heappush(heap, -q)
        sum_q += q
        if len(heap) > k:
            sum_q += heapq.heappop(heap)
        if len(heap) == k:
            res = min(res, sum_q * r)
    return res
```

Time: O(n log n) | Space: O(k)

---

## 13. Count of Smaller Numbers After Self (BIT/Fenwick)

Same as medium but optimize with BIT. Coordinate compression + Fenwick tree. Process from right, query prefix sum.

```python
def countSmaller(nums):
    rank = {v: i + 1 for i, v in enumerate(sorted(set(nums)))}
    n = len(nums)
    tree = [0] * (n + 2)

    def update(i):
        while i <= n + 1:
            tree[i] += 1
            i += i & -i

    def query(i):
        s = 0
        while i:
            s += tree[i]
            i -= i & -i
        return s

    res = []
    for x in reversed(nums):
        r = rank[x]
        res.append(query(r - 1))
        update(r)
    return res[::-1]
```

Time: O(n log n) | Space: O(n)

---

## 14. Russian Doll Envelopes

Envelopes (w,h). Fit one inside another if both dimensions smaller. Max chain. Sort by width asc, height desc. LIS on heights (binary search).

```python
def maxEnvelopes(envelopes):
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    from bisect import bisect_left
    dp = []
    for _, h in envelopes:
        i = bisect_left(dp, h)
        if i == len(dp):
            dp.append(h)
        else:
            dp[i] = h
    return len(dp)
```

Time: O(n log n) | Space: O(n)

---

## 15. Minimum Number of Operations to Make Array Continuous

Replace elements to make array contiguous [x, x+1, ..., x+n-1]. Min replacements. Sort and deduplicate. For each unique value as start, binary search how many in range [start, start+n-1]. Max window = min operations.

```python
def minOperations(nums):
    from bisect import bisect_right
    n = len(nums)
    nums = sorted(set(nums))
    res = n
    for i, start in enumerate(nums):
        end = start + n - 1
        j = bisect_right(nums, end) - 1
        in_range = j - i + 1
        res = min(res, n - in_range)
    return res
```

Time: O(n log n) | Space: O(n)
