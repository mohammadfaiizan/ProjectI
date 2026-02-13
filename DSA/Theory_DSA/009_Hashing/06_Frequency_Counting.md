# Frequency Counting and Hash-Based Problems

## Two Sum

Find two indices such that nums[i] + nums[j] = target. Store complement (target - nums[i]) in hash; when we see complement, return pair.

```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        comp = target - x
        if comp in seen:
            return [seen[comp], i]
        seen[x] = i
    return []
```

## Three Sum

Find all unique triplets that sum to 0. Sort, fix one element, use two pointers or hash for remaining pair.

```python
def three_sum(nums):
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        target = -nums[i]
        seen = set()
        for j in range(i + 1, n):
            comp = target - nums[j]
            if comp in seen:
                res.append([nums[i], comp, nums[j]])
                while j + 1 < n and nums[j] == nums[j + 1]:
                    j += 1
            seen.add(nums[j])
    return res
```

## Four Sum

Find all unique quadruplets that sum to target. Sort, fix two elements, use hash or two pointers for remaining pair.

```python
def four_sum(nums, target):
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            t = target - nums[i] - nums[j]
            seen = set()
            for k in range(j + 1, n):
                comp = t - nums[k]
                if comp in seen:
                    res.append([nums[i], nums[j], comp, nums[k]])
                    while k + 1 < n and nums[k] == nums[k + 1]:
                        k += 1
                seen.add(nums[k])
    return res
```

## Subarray Sum Equals K

Count subarrays with sum k. Prefix sum: if prefix[j] - prefix[i] = k, then prefix[i] = prefix[j] - k. Store prefix counts.

```python
def subarray_sum(nums, k):
    prefix_count = {0: 1}
    total = 0
    count = 0
    for x in nums:
        total += x
        count += prefix_count.get(total - k, 0)
        prefix_count[total] = prefix_count.get(total, 0) + 1
    return count
```

## Contiguous Array (Equal 0s and 1s)

Treat 0 as -1. Subarray with equal 0s and 1s has sum 0. Use prefix sum hash.

```python
def find_max_length(nums):
    d = {0: -1}
    total = 0
    ans = 0
    for i, x in enumerate(nums):
        total += 1 if x else -1
        if total in d:
            ans = max(ans, i - d[total])
        else:
            d[total] = i
    return ans
```

## Longest Consecutive Sequence

Find longest consecutive sequence length. Put all in set, for each num check if num-1 not in set (start of sequence), then count forward.

```python
def longest_consecutive(nums):
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:
            curr = 0
            while x + curr in s:
                curr += 1
            best = max(best, curr)
    return best
```

## Longest Subarray with Sum K

For non-negative array, sliding window. For array with negatives, prefix sum hash: store first occurrence of each prefix.

```python
def longest_subarray_sum_k(nums, k):
    d = {0: -1}
    total = 0
    ans = -1
    for i, x in enumerate(nums):
        total += x
        if total - k in d:
            ans = max(ans, i - d[total - k])
        if total not in d:
            d[total] = i
    return ans if ans >= 0 else 0
```

## Group Anagrams

Group strings that are anagrams. Key = sorted string or character count tuple.

```python
def group_anagrams(strs):
    from collections import defaultdict
    d = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        d[key].append(s)
    return list(d.values())
```

## First Unique Character

Find index of first non-repeating character. Count frequency, then scan for first with count 1.

```python
def first_uniq_char(s):
    from collections import Counter
    cnt = Counter(s)
    for i, c in enumerate(s):
        if cnt[c] == 1:
            return i
    return -1
```

## First Non-Repeating in Stream

Maintain DLL of unique chars and hash of char to node. On new char, add to DLL or remove if seen again.

```python
from collections import OrderedDict

class FirstUnique:
    def __init__(self, nums):
        self.od = OrderedDict()
        self.seen = set()
        for x in nums:
            self.add(x)

    def add(self, value):
        if value in self.seen:
            if value in self.od:
                del self.od[value]
        else:
            self.seen.add(value)
            self.od[value] = None

    def showFirstUnique(self):
        return next(iter(self.od), -1)
```

## Top K Frequent Elements

Count frequencies, then use heap or bucket sort.

```python
def top_k_frequent(nums, k):
    from collections import Counter
    cnt = Counter(nums)
    return [x for x, _ in cnt.most_common(k)]
```

## Sort Characters by Frequency

Count, sort by count descending, build result string.

```python
def frequency_sort(s):
    from collections import Counter
    cnt = Counter(s)
    return ''.join(c * n for c, n in cnt.most_common())
```

## Intersection of Two Arrays

Return unique elements in both. Use two sets, intersect.

```python
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))
```

## Intersection II (With Duplicates)

Return intersection with multiplicity. Count one array, decrement when found in second.

```python
def intersect(nums1, nums2):
    from collections import Counter
    c = Counter(nums1)
    res = []
    for x in nums2:
        if c[x] > 0:
            res.append(x)
            c[x] -= 1
    return res
```

## Isomorphic Strings

Map s to t and t to s. Check consistency.

```python
def is_isomorphic(s, t):
    m1, m2 = {}, {}
    for a, b in zip(s, t):
        if (a in m1 and m1[a] != b) or (b in m2 and m2[b] != a):
            return False
        m1[a] = b
        m2[b] = a
    return True
```

## Word Pattern

Same as isomorphic: pattern chars map to words bijectively.

```python
def word_pattern(pattern, s):
    words = s.split()
    if len(pattern) != len(words):
        return False
    m1, m2 = {}, {}
    for p, w in zip(pattern, words):
        if (p in m1 and m1[p] != w) or (w in m2 and m2[w] != p):
            return False
        m1[p] = w
        m2[w] = p
    return True
```

## Find All Duplicates

Array of 1..n, some appear twice. Mark visited indices.

```python
def find_duplicates(nums):
    res = []
    for x in nums:
        i = abs(x) - 1
        if nums[i] < 0:
            res.append(abs(x))
        else:
            nums[i] = -nums[i]
    return res
```

## Find Disappeared Numbers

Array of 1..n, find missing. Mark indices, collect unmarked.

```python
def find_disappeared(nums):
    for x in nums:
        i = abs(x) - 1
        if nums[i] > 0:
            nums[i] = -nums[i]
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]
```

## Valid Sudoku

Check rows, cols, boxes. Use sets or arrays for each.

```python
def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            c = board[i][j]
            if c == '.':
                continue
            if c in rows[i] or c in cols[j] or c in boxes[i//3*3+j//3]:
                return False
            rows[i].add(c)
            cols[j].add(c)
            boxes[i//3*3+j//3].add(c)
    return True
```

## Contains Duplicate

Any duplicate exists.

```python
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
```

## Contains Duplicate II (Within K)

Duplicate within distance k. Sliding window + set.

```python
def contains_nearby_duplicate(nums, k):
    seen = {}
    for i, x in enumerate(nums):
        if x in seen and i - seen[x] <= k:
            return True
        seen[x] = i
    return False
```

## Contains Duplicate III (Within K and T)

|nums[i]-nums[j]| <= t and |i-j| <= k. Use ordered structure (bucket or BST).

```python
def contains_nearby_almost_duplicate(nums, k, t):
    if t < 0:
        return False
    from collections import OrderedDict
    def get_id(x, w):
        return x // w if x >= 0 else (x + 1) // w - 1
    w = t + 1
    d = {}
    for i, x in enumerate(nums):
        bid = get_id(x, w)
        if bid in d:
            return True
        if bid - 1 in d and abs(x - d[bid - 1]) <= t:
            return True
        if bid + 1 in d and abs(x - d[bid + 1]) <= t:
            return True
        d[bid] = x
        if i >= k:
            del d[get_id(nums[i - k], w)]
    return False
```

## Majority Element

Element appearing more than n/2 times. Boyer-Moore or count.

```python
def majority_element(nums):
    cand, count = None, 0
    for x in nums:
        if count == 0:
            cand = x
        count += 1 if x == cand else -1
    return cand
```

## Majority Element II (n/3)

Elements appearing more than n/3. At most two such elements. Extended Boyer-Moore.

```python
def majority_element_ii(nums):
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
    return [x for x in [v1, v2] if x is not None and nums.count(x) > len(nums)//3]
```

## Count Pairs with Absolute Diff K

```python
def count_k_diff_pairs(nums, k):
    from collections import Counter
    c = Counter(nums)
    if k == 0:
        return sum(v*(v-1)//2 for v in c.values())
    return sum(c[x] * c.get(x + k, 0) for x in c)
```

## Number of Good Pairs

Pairs (i,j) with nums[i]==nums[j] and i<j.

```python
def num_identical_pairs(nums):
    from collections import Counter
    return sum(v*(v-1)//2 for v in Counter(nums).values())
```

## Ransom Note

Can magazine provide all chars for ransom? Count magazine, decrement for ransom.

```python
def can_construct(ransom, magazine):
    from collections import Counter
    c = Counter(magazine)
    for ch in ransom:
        if c[ch] <= 0:
            return False
        c[ch] -= 1
    return True
```

## Find Common Characters

Common chars across all strings with multiplicity.

```python
def common_chars(words):
    from collections import Counter
    cnt = Counter(words[0])
    for w in words[1:]:
        cnt &= Counter(w)
    return list(cnt.elements())
```

## Jewels and Stones

Count stones that are jewels.

```python
def num_jewels_in_stones(jewels, stones):
    j = set(jewels)
    return sum(1 for s in stones if s in j)
```

## Subdomain Visit Count

Parse "cnt domain", split domain by dots, add count to domain and all parent subdomains.

```python
def subdomain_visits(cpdomains):
    from collections import Counter
    c = Counter()
    for s in cpdomains:
        cnt, domain = s.split()
        cnt = int(cnt)
        parts = domain.split('.')
        for i in range(len(parts)):
            sub = '.'.join(parts[i:])
            c[sub] += cnt
    return [f"{v} {k}" for k, v in c.items()]
```

## Most Common Word

Most frequent word not in banned list.

```python
def most_common_word(paragraph, banned):
    import re
    from collections import Counter
    words = re.findall(r'\w+', paragraph.lower())
    banned = set(banned)
    c = Counter(w for w in words if w not in banned)
    return c.most_common(1)[0][0]
```

## Longest Harmonious Subsequence

Subsequence where max-min=1. Count frequencies, for each x check x and x+1.

```python
def find_lhs(nums):
    from collections import Counter
    c = Counter(nums)
    ans = 0
    for x in c:
        if x + 1 in c:
            ans = max(ans, c[x] + c[x + 1])
    return ans
```

## 4Sum II (Count Tuples)

Four arrays. Count (i,j,k,l) with A[i]+B[j]+C[k]+D[l]=0. Hash sums of A+B, count -(C[k]+D[l]).

```python
def four_sum_count(A, B, C, D):
    from collections import Counter
    ab = Counter(a + b for a in A for b in B)
    return sum(ab.get(-c - d, 0) for c in C for d in D)
```

## Brick Wall

Least bricks crossed. Hash positions of gaps (prefix sums per row), find position with max gaps.

```python
def least_bricks(wall):
    from collections import Counter
    c = Counter()
    for row in wall:
        pos = 0
        for w in row[:-1]:
            pos += w
            c[pos] += 1
    return len(wall) - max(c.values()) if c else len(wall)
```

## Palindrome Pairs (Hash Approach)

Pairs (i,j) where words[i]+words[j] is palindrome. For each word, check if reverse of prefix/suffix exists and remainder is palindrome.

```python
def palindrome_pairs(words):
    d = {w: i for i, w in enumerate(words)}
    res = []
    for i, w in enumerate(words):
        for j in range(len(w) + 1):
            pre, suf = w[:j], w[j:]
            rev_pre, rev_suf = pre[::-1], suf[::-1]
            if rev_suf in d and d[rev_suf] != i and pre == pre[::-1]:
                res.append([d[rev_suf], i])
            if j > 0 and rev_pre in d and d[rev_pre] != i and suf == suf[::-1]:
                res.append([i, d[rev_pre]])
    return res
```

## Number of Boomerangs

Triples (i,j,k) where dist(i,j)=dist(i,k). For each point, count distances, add count*(count-1).

```python
def number_of_boomerangs(points):
    def dist(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    ans = 0
    for p in points:
        c = {}
        for q in points:
            d = dist(p, q)
            c[d] = c.get(d, 0) + 1
        for v in c.values():
            ans += v * (v - 1)
    return ans
```
