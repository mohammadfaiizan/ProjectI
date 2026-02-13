# Easy Heap and Priority Queue Problems

## 1. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Min-heap of size k. Push each element; when size exceeds k, pop. Root is kth largest. O(n log k).

```python
import heapq
def findKthLargest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)
    return heap[0]
```

Time: O(n log k) | Space: O(k)

---

## 2. Last Stone Weight

Smash two largest stones; if unequal, push difference. Return last remaining weight. Max-heap (negate for Python heapq). Pop two, push difference until one or none left.

```python
import heapq
def lastStoneWeight(stones):
    heap = [-s for s in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        if a != b:
            heapq.heappush(heap, a - b)
    return -heap[0] if heap else 0
```

Time: O(n log n) | Space: O(n)

---

## 3. Merge K Sorted Lists

Merge k sorted linked lists into one sorted list. Min-heap of (val, list_id, node). Pop min, advance that list, push next. O(n log k).

```python
import heapq
def mergeKLists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    dummy = ListNode(0)
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

## 4. K Closest Points to Origin

Return k points closest to (0,0). Max-heap of size k by distance squared. Pop when size > k. O(n log k).

```python
import heapq
def kClosest(points, k):
    heap = []
    for x, y in points:
        d = -(x*x + y*y)
        if len(heap) < k:
            heapq.heappush(heap, (d, x, y))
        elif d > heap[0][0]:
            heapq.heapreplace(heap, (d, x, y))
    return [[x, y] for _, x, y in heap]
```

Time: O(n log k) | Space: O(k)

---

## 5. Top K Frequent Elements

Return k most frequent elements. Count frequencies. Min-heap of (freq, num) of size k. O(n log k).

```python
import heapq
from collections import Counter
def topKFrequent(nums, k):
    cnt = Counter(nums)
    heap = []
    for num, freq in cnt.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heapreplace(heap, (freq, num))
    return [num for _, num in heap]
```

Time: O(n log k) | Space: O(n)

---

## 6. Sort Characters by Frequency

Sort string so more frequent chars come first. Count, max-heap by freq, pop and append. O(n log n).

```python
import heapq
from collections import Counter
def frequencySort(s):
    cnt = Counter(s)
    heap = [(-freq, c) for c, freq in cnt.items()]
    heapq.heapify(heap)
    return ''.join(c * (-freq) for freq, c in heap)
```

Time: O(n log n) | Space: O(n)

---

## 7. Third Maximum Number

Return third distinct maximum; if fewer than three, return max. Min-heap of size 3 for three largest, or use set + heap.

```python
import heapq
def thirdMax(nums):
    seen = set()
    heap = []
    for x in nums:
        if x in seen:
            continue
        seen.add(x)
        if len(heap) < 3:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return heap[0] if len(heap) == 3 else max(heap)
```

Time: O(n) | Space: O(1)

---

## 8. Relative Ranks

Assign "Gold", "Silver", "Bronze" to top 3 scores; others get rank number. Heap of (score, index). Pop in order, assign ranks.

```python
import heapq
def findRelativeRanks(score):
    heap = [(-s, i) for i, s in enumerate(score)]
    heapq.heapify(heap)
    res = [''] * len(score)
    medals = ['Gold Medal', 'Silver Medal', 'Bronze Medal']
    for r in range(len(score)):
        _, i = heapq.heappop(heap)
        res[i] = medals[r] if r < 3 else str(r + 1)
    return res
```

Time: O(n log n) | Space: O(n)

---

## 9. Kth Largest in Stream

Design class to add numbers and return kth largest. Min-heap of size k. Same as kth largest in array but streaming.

```python
import heapq
class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = nums[:]
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heapreplace(self.heap, val)
        return self.heap[0]
```

Time: O(log k) add | Space: O(k)

---

## 10. Minimum Sum of Four Digit Number After Splitting

Split 4-digit number into two 2-digit numbers to minimize sum. Sort digits, form smallest two numbers. Heap not strictly needed but can use for general "min sum of k numbers from digits".

```python
def minimumSum(num):
    digits = sorted(str(num))
    return int(digits[0] + digits[2]) + int(digits[1] + digits[3])
```

Time: O(1) | Space: O(1)

---

## 11. Maximum Product of Two Elements in an Array

Find max of (nums[i]-1)*(nums[j]-1). Two largest elements. Max-heap or single pass to track two max.

```python
import heapq
def maxProduct(nums):
    heap = [-x for x in nums]
    heapq.heapify(heap)
    a = -heapq.heappop(heap)
    b = -heapq.heappop(heap)
    return (a - 1) * (b - 1)
```

Time: O(n) | Space: O(n)

---

## 12. Largest Number After Digit Swaps by Parity

Swap digits of same parity arbitrarily; find largest possible number. Separate odd and even digits, sort descending, interleave. Heap for general "k largest" variant.

```python
def largestInteger(num):
    s = str(num)
    odd = sorted([c for c in s if int(c) % 2], reverse=True)
    even = sorted([c for c in s if int(c) % 2 == 0], reverse=True)
    io, ie = 0, 0
    res = []
    for c in s:
        if int(c) % 2:
            res.append(odd[io])
            io += 1
        else:
            res.append(even[ie])
            ie += 1
    return int(''.join(res))
```

Time: O(n log n) | Space: O(n)

---

## 13. Make Array Zero by Subtracting Equal Amounts

Each move subtract same positive number from all non-zero elements. Min moves to make all zero. Count distinct non-zero values. Min-heap to process smallest first.

```python
def minimumOperations(nums):
    return len(set(nums) - {0})
```

Time: O(n) | Space: O(n)

---

## 14. Maximum Units on a Truck

Truck has box capacity. Each box type has (boxes, units per box). Maximize units. Sort by units per box descending. Greedy take. Heap for "k largest units" variant.

```python
def maximumUnits(boxTypes, truckSize):
    boxTypes.sort(key=lambda x: -x[1])
    total = 0
    for boxes, units in boxTypes:
        take = min(boxes, truckSize)
        total += take * units
        truckSize -= take
        if truckSize == 0:
            break
    return total
```

Time: O(n log n) | Space: O(1)

---

## 15. Minimum Cost of Buying Candies With Discount

Buy n candies, every 3rd is free (cheapest). Minimize cost. Sort descending. Every 3rd is free. Max-heap to always pick two paid, one free.

```python
def minimumCost(cost):
    cost.sort(reverse=True)
    total = 0
    for i, c in enumerate(cost):
        if (i + 1) % 3 != 0:
            total += c
    return total
```

Time: O(n log n) | Space: O(1)

---

## 16. Remove Stones to Minimize Total

Apply operation k times: pick pile, remove floor(pile/2) stones. Minimize total. Max-heap of piles. Pop, halve, push. Repeat k times.

```python
import heapq
def minStoneSum(piles, k):
    heap = [-p for p in piles]
    heapq.heapify(heap)
    for _ in range(k):
        p = -heapq.heappop(heap)
        heapq.heappush(heap, -(p - p // 2))
    return -sum(heap)
```

Time: O(k log n) | Space: O(n)

---

## 17. Construct String With Repeat Limit

Construct string with at most repeatLimit consecutive same chars, using char counts. Max-heap by count. Pop two different chars alternately to avoid consecutive limit.

```python
import heapq
from collections import Counter
def repeatLimitedString(s, repeatLimit):
    cnt = Counter(s)
    heap = [(-ord(c), cnt[c]) for c in cnt]
    heapq.heapify(heap)
    res = []
    prev = None
    while heap:
        neg_ord, count = heapq.heappop(heap)
        c = chr(-neg_ord)
        if c == prev:
            if not heap:
                break
            n2, c2 = heapq.heappop(heap)
            res.append(chr(-n2))
            c2 -= 1
            if c2:
                heapq.heappush(heap, (n2, c2))
            heapq.heappush(heap, (neg_ord, count))
            prev = chr(-n2)
        else:
            take = min(count, repeatLimit)
            res.append(c * take)
            count -= take
            if count:
                heapq.heappush(heap, (neg_ord, count))
            prev = c
    return ''.join(res)
```

Time: O(n log n) | Space: O(n)

---

## 18. Maximum Score From Removing Stones

Three piles. Each move remove 1 from two piles. Max moves. Max-heap. Pop two, decrement, push back if > 0. Count moves.

```python
import heapq
def maximumScore(a, b, c):
    heap = [-a, -b, -c]
    heapq.heapify(heap)
    moves = 0
    while len(heap) >= 2:
        x = -heapq.heappop(heap)
        y = -heapq.heappop(heap)
        moves += 1
        if x > 1:
            heapq.heappush(heap, -(x - 1))
        if y > 1:
            heapq.heappush(heap, -(y - 1))
    return moves
```

Time: O(n) | Space: O(1)

---

## 19. Minimum Sum of Squared Difference

Change nums1 to reduce sum of (nums1[i]-nums2[i])^2 with limited operations. Diff array, max-heap by abs(diff). Greedy reduce largest diffs first.

```python
import heapq
def minSumSquareDiff(nums1, nums2, k1, k2):
    diffs = [-abs(a - b) for a, b in zip(nums1, nums2)]
    heapq.heapify(diffs)
    k = k1 + k2
    while k and diffs and diffs[0] != 0:
        d = heapq.heappop(diffs)
        d = -d
        reduce = min(k, 1)
        d -= reduce
        k -= reduce
        heapq.heappush(diffs, -d if d > 0 else 0)
    return sum(d * d for d in diffs if d != 0)
```

Time: O(n log n) | Space: O(n)

---

## 20. Divide Array Into Equal Pairs

Can array be partitioned into n pairs with equal sums? Sort or count. Heap for "k smallest pairs" variant.

```python
def divideArray(nums):
    from collections import Counter
    return all(v % 2 == 0 for v in Counter(nums).values())
```

Time: O(n) | Space: O(n)

---

## 21. Maximum Bags With Full Capacity of Rocks

Bags have capacity and current rocks. Add additional rocks to maximize full bags. Min-heap of (capacity - current) for each bag. Fill smallest gaps first.

```python
def maximumBags(capacity, rocks, additionalRocks):
    gaps = sorted(c - r for c, r in zip(capacity, rocks))
    full = 0
    for g in gaps:
        if additionalRocks >= g:
            additionalRocks -= g
            full += 1
        else:
            break
    return full
```

Time: O(n log n) | Space: O(n)

---

## 22. Total Cost to Hire K Workers

Hire k workers from two ends of candidate list with cost. Two min-heaps for left and right. Pop k times from cheaper side.

```python
import heapq
def totalCost(costs, k, candidates):
    n = len(costs)
    left = costs[:candidates]
    right = costs[max(candidates, n - candidates):]
    heapq.heapify(left)
    heapq.heapify(right)
    li, ri = candidates, n - candidates - 1
    total = 0
    for _ in range(k):
        if not right or (left and left[0] <= right[0]):
            total += heapq.heappop(left)
            if li <= ri:
                heapq.heappush(left, costs[li])
                li += 1
        else:
            total += heapq.heappop(right)
            if li <= ri:
                heapq.heappush(right, costs[ri])
                ri -= 1
    return total
```

Time: O((k + n) log n) | Space: O(n)

---

## 23. Minimum Amount of Time to Fill Cups

Fill cups one or two at a time. Min seconds to fill all. Max-heap of remaining amounts. Each second reduce one or two largest.

```python
import heapq
def fillCups(amount):
    heap = [-a for a in amount]
    heapq.heapify(heap)
    sec = 0
    while heap[0] != 0:
        a = -heapq.heappop(heap)
        b = -heapq.heappop(heap) if len(heap) > 0 else 0
        a -= 1
        if b > 0:
            b -= 1
        if a > 0:
            heapq.heappush(heap, -a)
        if b > 0:
            heapq.heappush(heap, -b)
        sec += 1
    return sec
```

Time: O(n log n) | Space: O(1)

---

## 24. Apply Operations to an Array

Double zeros, shift. Return resulting array. Heap not primary; array simulation. Heap for "top k after operations" variant.

```python
def applyOperations(nums):
    for i in range(len(nums) - 1):
        if nums[i] == nums[i + 1]:
            nums[i] *= 2
            nums[i + 1] = 0
    write = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write] = nums[i]
            write += 1
    for i in range(write, len(nums)):
        nums[i] = 0
    return nums
```

Time: O(n) | Space: O(1)

---

## 25. Design a Number Container System

Support change index, find median or min index for a number. Map number to indices. Heap for "smallest index" queries.

```python
from collections import defaultdict
import heapq
class NumberContainers:
    def __init__(self):
        self.idx_to_num = {}
        self.num_to_idxs = defaultdict(list)

    def change(self, index, number):
        self.idx_to_num[index] = number
        heapq.heappush(self.num_to_idxs[number], index)

    def find(self, number):
        while self.num_to_idxs[number] and self.idx_to_num.get(self.num_to_idxs[number][0]) != number:
            heapq.heappop(self.num_to_idxs[number])
        return self.num_to_idxs[number][0] if self.num_to_idxs[number] else -1
```

Time: O(log n) change, O(1) amortized find | Space: O(n)
