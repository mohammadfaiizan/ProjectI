# Medium Hashing Problems

## 1. Three Sum

Find all unique triplets that sum to zero. Sort, fix one element, two pointers or hash for remaining pair. Skip duplicates.

```python
def threeSum(nums):
    nums.sort()
    n, res = len(nums), []
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        j, k = i + 1, n - 1
        while j < k:
            s = nums[i] + nums[j] + nums[k]
            if s == 0:
                res.append([nums[i], nums[j], nums[k]])
                while j < k and nums[j] == nums[j+1]: j += 1
                while j < k and nums[k] == nums[k-1]: k -= 1
                j += 1
                k -= 1
            elif s < 0:
                j += 1
            else:
                k -= 1
    return res
```

Time: O(n^2) | Space: O(1)

---

## 2. Group Anagrams

Group strings that are anagrams of each other. Use sorted string or character count tuple as hash key.

```python
def groupAnagrams(strs):
    from collections import defaultdict
    d = defaultdict(list)
    for s in strs:
        d[tuple(sorted(s))].append(s)
    return list(d.values())
```

Time: O(n * k log k) | Space: O(n)

---

## 3. Longest Substring Without Repeating Characters

Find longest substring with all unique characters. Sliding window with set or dict storing char to index.

```python
def lengthOfLongestSubstring(s):
    seen = {}
    start, best = 0, 0
    for i, c in enumerate(s):
        if c in seen and seen[c] >= start:
            start = seen[c] + 1
        seen[c] = i
        best = max(best, i - start + 1)
    return best
```

Time: O(n) | Space: O(min(n, charset))

---

## 4. Subarray Sum Equals K

Count contiguous subarrays with sum k. Prefix sum hash; prefix[j]-prefix[i]=k means prefix[i]=prefix[j]-k.

```python
def subarraySum(nums, k):
    from collections import defaultdict
    pre, cnt, cur = defaultdict(int), 0, 0
    pre[0] = 1
    for x in nums:
        cur += x
        cnt += pre.get(cur - k, 0)
        pre[cur] += 1
    return cnt
```

Time: O(n) | Space: O(n)

---

## 5. Top K Frequent Elements

Return k most frequent elements. Counter for frequencies, then bucket sort or heap.

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

## 6. Longest Consecutive Sequence

Longest consecutive integer sequence length. Put all in set; for each potential sequence start (x-1 not in set), count forward.

```python
def longestConsecutive(nums):
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

## 7. Contiguous Array

Longest subarray with equal 0s and 1s. Treat 0 as -1; prefix sum hash for sum 0.

```python
def findMaxLength(nums):
    pre, best, cur = {0: -1}, 0, 0
    for i, x in enumerate(nums):
        cur += 1 if x else -1
        if cur in pre:
            best = max(best, i - pre[cur])
        else:
            pre[cur] = i
    return best
```

Time: O(n) | Space: O(n)

---

## 8. Design Underground System

Track check-in/out and compute average time between stations. Hash check-in by id; hash (start, end) to (total_time, count).

```python
class UndergroundSystem:
    def __init__(self):
        self.check_in = {}
        self.trips = {}

    def checkIn(self, id, stationName, t):
        self.check_in[id] = (stationName, t)

    def checkOut(self, id, stationName, t):
        start, t0 = self.check_in.pop(id)
        key = (start, stationName)
        if key not in self.trips:
            self.trips[key] = [0, 0]
        self.trips[key][0] += t - t0
        self.trips[key][1] += 1

    def getAverageTime(self, startStation, endStation):
        total, cnt = self.trips[(startStation, endStation)]
        return total / cnt
```

Time: O(1) | Space: O(n)

---

## 9. LRU Cache

Cache with get/put in O(1), evict least recently used when full. Hash map + doubly linked list for order.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

Time: O(1) | Space: O(capacity)

---

## 10. Encode and Decode TinyURL

Shorten and expand URLs. Hash long URL to short code; store bidirectional mapping.

```python
class Codec:
    def __init__(self):
        self.long_to_short = {}
        self.short_to_long = {}
        self.counter = 0

    def encode(self, longUrl):
        if longUrl not in self.long_to_short:
            short = str(self.counter)
            self.counter += 1
            self.long_to_short[longUrl] = short
            self.short_to_long[short] = longUrl
        return "http://tinyurl.com/" + self.long_to_short[longUrl]

    def decode(self, shortUrl):
        key = shortUrl.split("/")[-1]
        return self.short_to_long.get(key, "")
```

Time: O(1) | Space: O(n)

---

## 11. Insert Delete GetRandom O(1)

Data structure with insert, remove, getRandom all O(1). List + dict; on remove swap with last and pop.

```python
import random

class RandomizedSet:
    def __init__(self):
        self.vals = []
        self.idx = {}

    def insert(self, val):
        if val in self.idx:
            return False
        self.idx[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val):
        if val not in self.idx:
            return False
        i = self.idx[val]
        last = self.vals[-1]
        self.vals[i] = last
        self.idx[last] = i
        self.vals.pop()
        del self.idx[val]
        return True

    def getRandom(self):
        return random.choice(self.vals)
```

Time: O(1) | Space: O(n)

---

## 12. Time Based Key-Value Store

Store multiple versions per key; get(key, timestamp) returns value at or before timestamp. Dict of key to list of (timestamp, value); binary search for floor.

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

## 13. Snapshot Array

Array with set, snap, and get(index, snap_id). Each index stores list of (snap_id, value); binary search for snap_id.

```python
from bisect import bisect_right

class SnapshotArray:
    def __init__(self, length):
        self.snaps = [[(0, 0)] for _ in range(length)]
        self.snap_id = 0

    def set(self, index, val):
        self.snaps[index].append((self.snap_id, val))

    def snap(self):
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index, snap_id):
        arr = self.snaps[index]
        i = bisect_right(arr, (snap_id, 10**9)) - 1
        return arr[i][1]
```

Time: O(log n) | Space: O(n)

---

## 14. 4Sum

Find all unique quadruplets with sum target. Sort, fix two elements, two pointers or hash for remaining pair.

```python
def fourSum(nums, target):
    nums.sort()
    n, res = len(nums), []
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
            lo, hi = j + 1, n - 1
            while lo < hi:
                s = nums[i] + nums[j] + nums[lo] + nums[hi]
                if s == target:
                    res.append([nums[i], nums[j], nums[lo], nums[hi]])
                    while lo < hi and nums[lo] == nums[lo+1]: lo += 1
                    while lo < hi and nums[hi] == nums[hi-1]: hi -= 1
                    lo += 1
                    hi -= 1
                elif s < target:
                    lo += 1
                else:
                    hi -= 1
    return res
```

Time: O(n^3) | Space: O(1)

---

## 15. 4Sum II

Four arrays; count tuples (i,j,k,l) with A[i]+B[j]+C[k]+D[l]=0. Hash sums of A+B; for each sum of C+D, count -sum in hash.

```python
def fourSumCount(A, B, C, D):
    from collections import Counter
    ab = Counter(a + b for a in A for b in B)
    return sum(ab.get(-c - d, 0) for c in C for d in D)
```

Time: O(n^2) | Space: O(n^2)

---

## 16. Sort Characters by Frequency

Sort string by character frequency descending. Counter, then most_common or sort by count.

```python
def frequencySort(s):
    from collections import Counter
    return ''.join(c * n for c, n in Counter(s).most_common())
```

Time: O(n log n) | Space: O(n)

---

## 17. Find All Duplicates in an Array

Array 1..n, some appear twice; return all duplicates. Mark indices by negating; already negative means duplicate.

```python
def findDuplicates(nums):
    out = []
    for x in nums:
        i = abs(x) - 1
        if nums[i] < 0:
            out.append(abs(x))
        else:
            nums[i] *= -1
    return out
```

Time: O(n) | Space: O(1)

---

## 18. Contains Duplicate III

Check if |nums[i]-nums[j]| <= t and |i-j| <= k. Bucket hash or sliding window with sorted container.

```python
def containsNearbyAlmostDuplicate(nums, k, t):
    from collections import OrderedDict
    if t < 0:
        return False
    bucket = OrderedDict()
    w = t + 1
    for i, x in enumerate(nums):
        b = x // w
        if b in bucket or (b-1 in bucket and x - bucket[b-1] <= t) or (b+1 in bucket and bucket[b+1] - x <= t):
            return True
        bucket[b] = x
        if i >= k:
            bucket.popitem(last=False)
    return False
```

Time: O(n) | Space: O(min(n, k))

---

## 19. Majority Element II

Find elements appearing more than n/3 times. Extended Boyer-Moore; at most two candidates.

```python
def majorityElement(nums):
    c1, c2, v1, v2 = 0, 0, None, None
    for x in nums:
        if x == v1:
            c1 += 1
        elif x == v2:
            c2 += 1
        elif c1 == 0:
            v1, c1 = x, 1
        elif c2 == 0:
            v2, c2 = x, 1
        else:
            c1 -= 1
            c2 -= 1
    return [x for x in (v1, v2) if x is not None and nums.count(x) > len(nums) // 3]
```

Time: O(n) | Space: O(1)

---

## 20. Longest Harmonious Subsequence

Longest subsequence where max-min=1. Count frequencies; for each x, add count(x)+count(x+1) if both exist.

```python
def findLHS(nums):
    from collections import Counter
    c = Counter(nums)
    return max((c[x] + c[x+1] for x in c if x + 1 in c), default=0)
```

Time: O(n) | Space: O(n)

---

## 21. Brick Wall

Least bricks to cross (vertical line). Hash gap positions; max gaps across rows = least bricks.

```python
def leastBricks(wall):
    from collections import Counter
    gaps = Counter()
    for row in wall:
        pos = 0
        for w in row[:-1]:
            pos += w
            gaps[pos] += 1
    return len(wall) - max(gaps.values()) if gaps else len(wall)
```

Time: O(n * m) | Space: O(n)

---

## 22. Number of Boomerangs

Triples (i,j,k) where dist(i,j)=dist(i,k). For each point, count distances; each count c adds c*(c-1).

```python
def numberOfBoomerangs(points):
    from collections import Counter
    def dist(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    total = 0
    for p in points:
        c = Counter(dist(p, q) for q in points if q != p)
        total += sum(v * (v - 1) for v in c.values())
    return total
```

Time: O(n^2) | Space: O(n)

---

## 23. Find the Duplicate Number

Array 1..n with one duplicate; find it in O(1) space. Floyd cycle detection (linked list) or binary search on count.

```python
def findDuplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

Time: O(n) | Space: O(1)

---

## 24. Subarray Sum Divisible by K

Count subarrays with sum divisible by k. Prefix sum mod k; same remainder means divisible subarray.

```python
def subarraysDivByK(nums, k):
    from collections import defaultdict
    pre, cnt, cur = defaultdict(int), 0, 0
    pre[0] = 1
    for x in nums:
        cur = (cur + x) % k
        cnt += pre[cur]
        pre[cur] += 1
    return cnt
```

Time: O(n) | Space: O(k)

---

## 25. Maximum Size Subarray Sum Equals K

Longest subarray with sum k. Prefix sum hash; store first occurrence of each prefix for longest.

```python
def maxSubArrayLen(nums, k):
    pre, best, cur = {0: -1}, 0, 0
    for i, x in enumerate(nums):
        cur += x
        if cur - k in pre:
            best = max(best, i - pre[cur - k])
        if cur not in pre:
            pre[cur] = i
    return best
```

Time: O(n) | Space: O(n)

---

## 26. Copy List with Random Pointer

Deep copy linked list with random pointer. Hash map old node to new node; two passes.

```python
def copyRandomList(head):
    if not head:
        return None
    m = {}
    cur = head
    while cur:
        m[cur] = Node(cur.val)
        cur = cur.next
    cur = head
    while cur:
        if cur.next:
            m[cur].next = m[cur.next]
        if cur.random:
            m[cur].random = m[cur.random]
        cur = cur.next
    return m[head]
```

Time: O(n) | Space: O(n)

---

## 27. Reconstruct Original Digits from English

Given string with jumbled digits in words, return digits in order. Count unique chars; some digits have unique letters (e.g., z only in zero).

```python
def originalDigits(s):
    from collections import Counter
    c = Counter(s)
    cnt = [0] * 10
    cnt[0] = c.get('z', 0)
    cnt[2] = c.get('w', 0)
    cnt[4] = c.get('u', 0)
    cnt[6] = c.get('x', 0)
    cnt[8] = c.get('g', 0)
    cnt[1] = c.get('o', 0) - cnt[0] - cnt[2] - cnt[4]
    cnt[3] = c.get('h', 0) - cnt[8]
    cnt[5] = c.get('f', 0) - cnt[4]
    cnt[7] = c.get('s', 0) - cnt[6]
    cnt[9] = c.get('i', 0) - cnt[5] - cnt[6] - cnt[8]
    return ''.join(str(i) * cnt[i] for i in range(10))
```

Time: O(n) | Space: O(1)

---

## 28. Design Authentication Manager

Token with expiry; count unexpired tokens. Hash token to expiry time; on count, filter by current time.

```python
class AuthenticationManager:
    def __init__(self, timeToLive):
        self.ttl = timeToLive
        self.tokens = {}

    def generate(self, tokenId, currentTime):
        self.tokens[tokenId] = currentTime + self.ttl

    def renew(self, tokenId, currentTime):
        if tokenId in self.tokens and self.tokens[tokenId] > currentTime:
            self.tokens[tokenId] = currentTime + self.ttl

    def countUnexpiredTokens(self, currentTime):
        self.tokens = {k: v for k, v in self.tokens.items() if v > currentTime}
        return len(self.tokens)
```

Time: O(n) | Space: O(n)

---

## 29. Insert Delete GetRandom O(1) with Duplicates

Same as RandomizedSet but allow duplicates. Dict of value to set of indices; list for storage; swap with last on remove.

```python
import random

class RandomizedCollection:
    def __init__(self):
        self.vals = []
        self.idx = {}

    def insert(self, val):
        self.vals.append(val)
        if val not in self.idx:
            self.idx[val] = set()
        self.idx[val].add(len(self.vals) - 1)
        return len(self.idx[val]) == 1

    def remove(self, val):
        if val not in self.idx or not self.idx[val]:
            return False
        i = self.idx[val].pop()
        last = self.vals[-1]
        self.vals[i] = last
        self.idx[last].add(i)
        self.idx[last].discard(len(self.vals) - 1)
        self.vals.pop()
        return True

    def getRandom(self):
        return random.choice(self.vals)
```

Time: O(1) | Space: O(n)

---

## 30. LFU Cache

Cache evicting least frequently used. Hash key to node; freq to doubly linked list; min_freq tracker.

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.min_freq = 0
        self.freq = defaultdict(OrderedDict)
        self.key_to_freq = {}

    def get(self, key):
        if key not in self.key_to_freq:
            return -1
        f = self.key_to_freq[key]
        val = self.freq[f].pop(key)
        if not self.freq[f] and f == self.min_freq:
            self.min_freq += 1
        self.freq[f+1][key] = val
        self.key_to_freq[key] = f + 1
        return val

    def put(self, key, value):
        if self.cap == 0:
            return
        if key in self.key_to_freq:
            self.get(key)
            self.freq[self.key_to_freq[key]][key] = value
            return
        if len(self.key_to_freq) >= self.cap:
            k = next(iter(self.freq[self.min_freq]))
            del self.freq[self.min_freq][k]
            del self.key_to_freq[k]
        self.min_freq = 1
        self.freq[1][key] = value
        self.key_to_freq[key] = 1
```

Time: O(1) | Space: O(capacity)
