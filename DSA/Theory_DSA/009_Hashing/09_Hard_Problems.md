# Hard Hashing Problems

## 1. Palindrome Pairs

Find all pairs (i,j) where words[i] + words[j] is palindrome. For each word, consider all splits; if reverse of one part exists and other part is palindrome, valid pair. Hash words to indices.

```python
def palindromePairs(words):
    d = {w: i for i, w in enumerate(words)}
    res = []
    for i, w in enumerate(words):
        for j in range(len(w) + 1):
            pre, suf = w[:j], w[j:]
            if pre == pre[::-1] and suf[::-1] in d and d[suf[::-1]] != i:
                res.append([d[suf[::-1]], i])
            if j > 0 and suf == suf[::-1] and pre[::-1] in d and d[pre[::-1]] != i:
                res.append([i, d[pre[::-1]]])
    return res
```

Time: O(n * k^2) | Space: O(n)

---

## 2. Substring with Concatenation of All Words

Find all starting indices where substring is concatenation of every word exactly once. Sliding window with word length; hash word counts; check each starting position.

```python
def findSubstring(s, words):
    from collections import Counter
    if not words:
        return []
    n, k, total = len(s), len(words[0]), len(words) * len(words[0])
    target = Counter(words)
    res = []
    for start in range(k):
        seen = Counter()
        count, left = 0, start
        for i in range(start, n - k + 1, k):
            w = s[i:i+k]
            if w in target:
                seen[w] += 1
                count += 1
                while seen[w] > target[w]:
                    lw = s[left:left+k]
                    seen[lw] -= 1
                    count -= 1
                    left += k
                if count == len(words):
                    res.append(left)
                    lw = s[left:left+k]
                    seen[lw] -= 1
                    count -= 1
                    left += k
            else:
                seen.clear()
                count = 0
                left = i + k
    return res
```

Time: O(n * k) | Space: O(m)

---

## 3. Minimum Window Substring

Smallest substring of s containing all chars of t. Sliding window; hash to track char counts; expand/contract to satisfy.

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

## 4. Longest Substring with At Most K Distinct Characters

Longest substring with at most k distinct chars. Sliding window with hash for char counts; shrink when distinct > k.

```python
def lengthOfLongestSubstringKDistinct(s, k):
    from collections import defaultdict
    if k == 0:
        return 0
    cnt = defaultdict(int)
    l, best = 0, 0
    for r, c in enumerate(s):
        cnt[c] += 1
        while len(cnt) > k:
            cnt[s[l]] -= 1
            if cnt[s[l]] == 0:
                del cnt[s[l]]
            l += 1
        best = max(best, r - l + 1)
    return best
```

Time: O(n) | Space: O(k)

---

## 5. Count Unique Characters of All Substrings

Sum over all substrings of "unique char count" in each substring. For each char, count substrings where it is unique (contribution method).

```python
def uniqueLetterString(s):
    idx = {c: [-1, -1] for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    res = 0
    for i, c in enumerate(s):
        k, j = idx[c]
        res += (i - j) * (j - k)
        idx[c] = [j, i]
    for c in idx:
        k, j = idx[c]
        res += (len(s) - j) * (j - k)
    return res
```

Time: O(n) | Space: O(1)

---

## 6. Subarrays with K Different Integers

Count subarrays with exactly k distinct integers. (At most K) - (At most K-1) using sliding window.

```python
def subarraysWithKDistinct(nums, k):
    def atMost(k):
        cnt = {}
        l, res = 0, 0
        for r, x in enumerate(nums):
            cnt[x] = cnt.get(x, 0) + 1
            while len(cnt) > k:
                cnt[nums[l]] -= 1
                if cnt[nums[l]] == 0:
                    del cnt[nums[l]]
                l += 1
            res += r - l + 1
        return res
    return atMost(k) - atMost(k - 1)
```

Time: O(n) | Space: O(k)

---

## 7. Minimum Window Subsequence

Shortest substring of s that has t as subsequence. DP or two pointers; hash for next occurrence can help.

```python
def minWindow(s, t):
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        dp[0][j] = float('inf')
    best_len, best_start = float('inf'), -1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = 1 + (dp[i-1][j-1] if dp[i-1][j-1] != float('inf') else 0)
            else:
                dp[i][j] = 1 + dp[i-1][j] if dp[i-1][j] != float('inf') else float('inf')
        if dp[i][m] < best_len:
            best_len = dp[i][m]
            best_start = i - best_len
    return s[best_start:best_start+best_len] if best_len != float('inf') else ""
```

Time: O(n * m) | Space: O(n * m)

---

## 8. Max Points on a Line

Max collinear points. For each point, hash slope (dx, dy) normalized; count max same slope.

```python
def maxPoints(points):
    from math import gcd
    from collections import Counter
    n = len(points)
    if n <= 2:
        return n
    best = 0
    for i in range(n):
        slopes = Counter()
        for j in range(n):
            if i == j:
                continue
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = gcd(dx, dy)
            if g:
                dx, dy = dx // g, dy // g
            slopes[(dx, dy)] += 1
        best = max(best, 1 + max(slopes.values()) if slopes else 1)
    return best
```

Time: O(n^2) | Space: O(n)

---

## 9. First Missing Positive

Find smallest missing positive integer in O(n) time O(1) space. Index mapping; place each positive at its index; scan for first mismatch.

```python
def firstMissingPositive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1
```

Time: O(n) | Space: O(1)

---

## 10. Trapping Rain Water II

3D version; water trapped in elevation map. Min-heap from boundary; hash or visited set for processed cells.

```python
def trapRainWater(heightMap):
    import heapq
    if not heightMap:
        return 0
    m, n = len(heightMap), len(heightMap[0])
    vis = [[False] * n for _ in range(m)]
    heap = []
    for i in range(m):
        for j in range(n):
            if i in (0, m-1) or j in (0, n-1):
                heapq.heappush(heap, (heightMap[i][j], i, j))
                vis[i][j] = True
    res = 0
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and not vis[ni][nj]:
                vis[ni][nj] = True
                res += max(0, h - heightMap[ni][nj])
                heapq.heappush(heap, (max(h, heightMap[ni][nj]), ni, nj))
    return res
```

Time: O(m * n * log(m * n)) | Space: O(m * n)

---

## 11. Word Squares

Arrange words so each row and column reads same word. Backtrack; hash prefix to list of words; build square row by row.

```python
def wordSquares(words):
    from collections import defaultdict
    n = len(words[0])
    pref = defaultdict(list)
    for w in words:
        for i in range(1, n + 1):
            pref[w[:i]].append(w)

    def backtrack(square):
        if len(square) == n:
            res.append(square[:])
            return
        prefix = ''.join(square[i][len(square)] for i in range(len(square)))
        for w in pref[prefix]:
            square.append(w)
            backtrack(square)
            square.pop()

    res = []
    for w in words:
        backtrack([w])
    return res
```

Time: O(n * m^n) | Space: O(n * m)

---

## 12. Palindrome Pairs (Optimized)

Same as above; optimize with Trie or rolling hash. Trie of reversed words; for each word traverse and check remainder palindrome.

```python
def palindromePairs(words):
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.idx = -1
            self.pal_suffix = []

    root = TrieNode()
    for i, w in enumerate(words):
        node = root
        for j, c in enumerate(reversed(w)):
            if w[:len(w)-j] == w[:len(w)-j][::-1]:
                node.pal_suffix.append(i)
            node = node.children.setdefault(c, TrieNode())
        node.idx = i
        node.pal_suffix.append(i)

    res = []
    for i, w in enumerate(words):
        node = root
        for j, c in enumerate(w):
            if node.idx >= 0 and node.idx != i and w[j:] == w[j:][::-1]:
                res.append([i, node.idx])
            if c not in node.children:
                break
            node = node.children[c]
        else:
            for k in node.pal_suffix:
                if k != i and words[k] == words[k][::-1]:
                    res.append([i, k])
                elif k != i:
                    suf = words[k][:len(words[k])-len(w)]
                    if suf == suf[::-1]:
                        res.append([i, k])
    return res
```

Time: O(n * k^2) | Space: O(n * k)

---

## 13. Count of Smaller Numbers After Self

For each element, count smaller elements to the right. Merge sort or Fenwick/BIT; hashing for coordinate compression.

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

## 14. Maximum XOR of Two Numbers in Array

Find max XOR of any pair. Trie with bits; for each number greedily choose opposite bit when possible.

```python
def findMaximumXOR(nums):
    root = {}
    for x in nums:
        node = root
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            node = node.setdefault(b, {})
    best = 0
    for x in nums:
        node, cur = root, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            if want in node:
                cur |= (1 << i)
                node = node[want]
            else:
                node = node[b]
        best = max(best, cur)
    return best
```

Time: O(n * 32) | Space: O(n * 32)

---

## 15. Number of Matching Subsequences

Count words that are subsequences of s. Precompute next occurrence of each char; hash word to pointer in s.

```python
def numMatchingSubseq(s, words):
    from collections import defaultdict
    waiting = defaultdict(list)
    for w in words:
        waiting[w[0]].append(iter(w[1:]))
    for c in s:
        for it in waiting.pop(c, []):
            nxt = next(it, None)
            if nxt is None:
                continue
            waiting[nxt].append(it)
    return len(words) - sum(len(v) for v in waiting.values())
```

Time: O(len(s) + sum(len(w))) | Space: O(n)

---

## 16. Minimum Number of Refueling Stops

Min stops to reach target with fuel stations. Max-heap of fuel at passed stations; hash for station positions.

```python
def minRefuelStops(target, startFuel, stations):
    import heapq
    heap = []
    fuel, stops, i = startFuel, 0, 0
    while fuel < target:
        while i < len(stations) and stations[i][0] <= fuel:
            heapq.heappush(heap, -stations[i][1])
            i += 1
        if not heap:
            return -1
        fuel += -heapq.heappop(heap)
        stops += 1
    return stops
```

Time: O(n log n) | Space: O(n)

---

## 17. Longest Duplicate Substring

Longest substring that appears at least twice. Binary search on length; rolling hash (Rabin-Karp) for O(n) check.

```python
def longestDupSubstring(s):
    n = len(s)
    base, mod = 26, 2**63 - 1

    def check(L):
        h = 0
        for i in range(L):
            h = (h * base + (ord(s[i]) - 97)) % mod
        seen = {h}
        p = pow(base, L - 1, mod)
        for i in range(L, n):
            h = (h - (ord(s[i-L]) - 97) * p) % mod
            h = (h * base + (ord(s[i]) - 97)) % mod
            if h in seen:
                return s[i-L+1:i+1]
            seen.add(h)
        return ""

    lo, hi = 1, n
    res = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        substr = check(mid)
        if substr:
            res = substr
            lo = mid + 1
        else:
            hi = mid - 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## 18. Count Distinct Substrings

Count distinct substrings of string. Suffix array or Trie; hash set of substrings for simpler O(n^2) approach.

```python
def countDistinctSubstrings(s):
    return len({s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)})
```

Time: O(n^2) | Space: O(n^2)

---

## 19. Repeated DNA Sequences

Find 10-char sequences that appear more than once. Rolling hash or encode as 2 bits per char; hash to count.

```python
def findRepeatedDnaSequences(s):
    from collections import Counter
    if len(s) < 10:
        return []
    cnt = Counter(s[i:i+10] for i in range(len(s) - 9))
    return [seq for seq, c in cnt.items() if c > 1]
```

Time: O(n) | Space: O(n)

---

## 20. Group Shifted Strings

Group strings that are shifts of each other (abc, bcd, cde). Normalize by first char; use difference sequence as key.

```python
def groupStrings(strings):
    from collections import defaultdict
    def key(s):
        if not s:
            return ()
        base = ord(s[0])
        return tuple((ord(c) - base) % 26 for c in s)
    d = defaultdict(list)
    for s in strings:
        d[key(s)].append(s)
    return list(d.values())
```

Time: O(n * k) | Space: O(n)
