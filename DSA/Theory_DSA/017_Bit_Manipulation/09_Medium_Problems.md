# Medium Bit Manipulation Problems

## 1. Single Number II

**Description:** Every element appears three times except one. Find the unique element.

**Approach:** Bit counting: for each bit position, count mod 3; or use ones/twos state machine.

```python
def singleNumber(nums):
    ones = twos = 0
    for x in nums:
        ones = (ones ^ x) & ~twos
        twos = (twos ^ x) & ~ones
    return ones
```

Time: O(n) | Space: O(1)

---

## 2. Single Number III

**Description:** Every element appears twice except two. Find both unique elements.

**Approach:** XOR all to get a^b; use rightmost set bit to partition into two groups; XOR each group.

```python
def singleNumber(nums):
    xor_all = 0
    for x in nums:
        xor_all ^= x
    diff = xor_all & -xor_all
    a = b = 0
    for x in nums:
        if x & diff:
            a ^= x
        else:
            b ^= x
    return [a, b]
```

Time: O(n) | Space: O(1)

---

## 3. Maximum XOR of Two Numbers in an Array

**Description:** Find maximum XOR of any pair in array.

**Approach:** Build binary trie; for each number traverse trie greedily choosing opposite bit when available.

```python
def findMaximumXOR(nums):
    trie = {}
    for x in nums:
        node = trie
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            node = node.setdefault(b, {})
    res = 0
    for x in nums:
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur <<= 1
            if want in node:
                cur += 1
                node = node[want]
            else:
                node = node[b]
        res = max(res, cur)
    return res
```

Time: O(n * 32) | Space: O(n * 32)

---

## 4. Subsets

**Description:** Generate all subsets of array.

**Approach:** Iterate mask 0 to 2^n-1; include arr[i] if bit i set.

```python
def subsets(nums):
    res = []
    for mask in range(1 << len(nums)):
        res.append([nums[i] for i in range(len(nums)) if (mask >> i) & 1])
    return res
```

Time: O(n * 2^n) | Space: O(2^n)

---

## 5. Subsets II

**Description:** Generate all subsets with duplicates (no duplicate subsets).

**Approach:** Sort first; bitmask with duplicate handling or backtracking.

```python
def subsetsWithDup(nums):
    nums.sort()
    res = set()
    for mask in range(1 << len(nums)):
        res.add(tuple(nums[i] for i in range(len(nums)) if (mask >> i) & 1))
    return [list(s) for s in res]
```

Time: O(n * 2^n) | Space: O(2^n)

---

## 6. Partition to K Equal Sum Subsets

**Description:** Can array be partitioned into k subsets with equal sum?

**Approach:** Bitmask DP; dp[mask] = (current subset sum, subsets used).

```python
def canPartitionKSubsets(nums, k):
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    nums.sort(reverse=True)
    n = len(nums)
    dp = [None] * (1 << n)
    dp[0] = (0, 0)
    for mask in range(1 << n):
        if dp[mask] is None:
            continue
        cur_sum, subsets = dp[mask]
        for i in range(n):
            if (mask >> i) & 1:
                continue
            new_sum = cur_sum + nums[i]
            new_mask = mask | (1 << i)
            if new_sum > target:
                continue
            ns = new_sum if new_sum < target else 0
            nsub = subsets + (1 if new_sum == target else 0)
            if dp[new_mask] is None or nsub > dp[new_mask][1]:
                dp[new_mask] = (ns, nsub)
    return dp[-1] and dp[-1][1] == k
```

Time: O(n * 2^n) | Space: O(2^n)

---

## 7. Matchsticks to Square

**Description:** Can matchsticks form a square?

**Approach:** Bitmask to try all partitions into 4 groups.

```python
def makesquare(matchsticks):
    total = sum(matchsticks)
    if total % 4:
        return False
    side = total // 4
    n = len(matchsticks)
    dp = {}
    def dfs(mask, sides_done, cur_sum):
        if mask in dp:
            return dp[mask]
        if sides_done == 4:
            return True
        if cur_sum == side:
            return dfs(mask, sides_done + 1, 0)
        for i in range(n):
            if (mask >> i) & 1 or cur_sum + matchsticks[i] > side:
                continue
            if dfs(mask | (1 << i), sides_done, cur_sum + matchsticks[i]):
                dp[mask] = True
                return True
        dp[mask] = False
        return False
    return dfs(0, 0, 0)
```

Time: O(n * 2^n) | Space: O(2^n)

---

## 8. Maximum Product of Word Lengths

**Description:** Max len(word[i]) * len(word[j]) where words share no letters.

**Approach:** Bitmask per word; iterate pairs, check mask_i & mask_j == 0.

```python
def maxProduct(words):
    masks = []
    for w in words:
        m = 0
        for c in w:
            m |= 1 << (ord(c) - 97)
        masks.append((m, len(w)))
    res = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if not (masks[i][0] & masks[j][0]):
                res = max(res, masks[i][1] * masks[j][1])
    return res
```

Time: O(n^2) | Space: O(n)

---

## 9. Gray Code

**Description:** Generate n-bit gray code sequence.

**Approach:** Gray code formula: i ^ (i >> 1) for i in 0..2^n-1.

```python
def grayCode(n):
    return [i ^ (i >> 1) for i in range(1 << n)]
```

Time: O(2^n) | Space: O(1)

---

## 10. Repeated DNA Sequences

**Description:** Find 10-char sequences that appear more than once.

**Approach:** Encode sequence as 2 bits per char (A=00,C=01,G=10,T=11); use rolling hash or bitmask.

```python
def findRepeatedDnaSequences(s):
    if len(s) < 10:
        return []
    m = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seen, res = set(), set()
    cur = 0
    for i in range(10):
        cur = (cur << 2) | m[s[i]]
    seen.add(cur)
    for i in range(10, len(s)):
        cur = ((cur << 2) | m[s[i]]) & 0xFFFFF
        if cur in seen:
            res.add(s[i-9:i+1])
        seen.add(cur)
    return list(res)
```

Time: O(n) | Space: O(n)

---

## 11. Total Hamming Distance

**Description:** Sum of hamming distances between all pairs.

**Approach:** For each bit position, count ones; contribution = count * (n - count).

```python
def totalHammingDistance(nums):
    res = 0
    for i in range(32):
        ones = sum((x >> i) & 1 for x in nums)
        res += ones * (len(nums) - ones)
    return res
```

Time: O(n * 32) | Space: O(1)

---

## 12. Find the Duplicate Number

**Description:** Array of n+1 numbers in [1,n]; exactly one duplicate.

**Approach:** Floyd cycle detection or XOR with 1..n (if space allows).

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

## 13. Decode XORed Permutation

**Description:** Reconstruct perm [1..n] from encoded where encoded[i] = perm[i] XOR perm[i+1].

**Approach:** XOR of 1..n known; encoded at odd indices gives perm[0]; reconstruct.

```python
def decode(encoded):
    n = len(encoded) + 1
    total = 0
    for i in range(1, n + 1):
        total ^= i
    first = total
    for i in range(1, len(encoded), 2):
        first ^= encoded[i]
    res = [first]
    for x in encoded:
        res.append(res[-1] ^ x)
    return res
```

Time: O(n) | Space: O(1)

---

## 14. Minimum XOR Sum of Two Arrays

**Description:** Permute arr2 to minimize sum of (arr1[i] XOR arr2[perm[i]]).

**Approach:** Bitmask DP; dp[mask] = min XOR sum for first popcount(mask) elements of arr1.

```python
def minimumXORSum(arr1, arr2):
    n = len(arr1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        j = bin(mask).count('1')
        if j >= n:
            continue
        for i in range(n):
            if (mask >> i) & 1:
                continue
            dp[mask | (1 << i)] = min(dp[mask | (1 << i)], dp[mask] + (arr1[j] ^ arr2[i]))
    return dp[-1]
```

Time: O(n^2 * 2^n) | Space: O(2^n)

---

## 15. Number of Valid Words for Each Puzzle

**Description:** For each puzzle, count words that are subsets of puzzle and contain first letter.

**Approach:** Bitmask words; for each puzzle enumerate submasks containing first letter.

```python
def findNumOfValidWords(words, puzzles):
    from collections import Counter
    count = Counter()
    for w in words:
        mask = 0
        for c in w:
            mask |= 1 << (ord(c) - 97)
        count[mask] += 1
    res = []
    for p in puzzles:
        first = 1 << (ord(p[0]) - 97)
        mask = 0
        for c in p:
            mask |= 1 << (ord(c) - 97)
        submask = mask
        total = 0
        while submask:
            if submask & first:
                total += count.get(submask, 0)
            submask = (submask - 1) & mask
        res.append(total)
    return res
```

Time: O(w * L + p * 2^7) | Space: O(w)

---

## 16. Can I Win

**Description:** Two players pick 1..maxChoosable without replacement; first to reach desiredTotal wins.

**Approach:** Bitmask state (chosen numbers); memoized DFS.

```python
def canIWin(maxChoosableInteger, desiredTotal):
    if desiredTotal <= maxChoosableInteger:
        return True
    if (maxChoosableInteger + 1) * maxChoosableInteger // 2 < desiredTotal:
        return False
    memo = {}
    def dfs(mask, total):
        if mask in memo:
            return memo[mask]
        for i in range(1, maxChoosableInteger + 1):
            if (mask >> (i-1)) & 1:
                continue
            if total + i >= desiredTotal:
                memo[mask] = True
                return True
            if not dfs(mask | (1 << (i-1)), total + i):
                memo[mask] = True
                return True
        memo[mask] = False
        return False
    return dfs(0, 0)
```

Time: O(2^n) | Space: O(2^n)

---

## 17. Partition Equal Subset Sum

**Description:** Can array be partitioned into two subsets with equal sum?

**Approach:** Bitset DP or bitmask; check if sum/2 achievable.

```python
def canPartition(nums):
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = 1
    for n in nums:
        dp |= dp << n
    return (dp >> target) & 1
```

Time: O(n * sum) | Space: O(sum)

---

## 18. Letter Tile Possibilities

**Description:** Count distinct sequences from tiles (with duplicates).

**Approach:** Bitmask for chosen positions; backtrack with duplicate handling.

```python
def numTilePossibilities(tiles):
    from collections import Counter
    def dfs(count):
        total = 0
        for c in count:
            if count[c] > 0:
                count[c] -= 1
                total += 1 + dfs(count)
                count[c] += 1
        return total
    return dfs(Counter(tiles))
```

Time: O(n!) | Space: O(n)

---

## 19. Find Kth Bit in Nth Binary String

**Description:** Recursive binary string; find kth character.

**Approach:** Pattern: S(n) = S(n-1) + "1" + reverse(invert(S(n-1))).

```python
def findKthBit(n, k):
    if n == 1:
        return "0"
    mid = (1 << n) // 2
    if k == mid:
        return "1"
    if k < mid:
        return findKthBit(n - 1, k)
    return "1" if findKthBit(n - 1, (1 << n) - k) == "0" else "0"
```

Time: O(n) | Space: O(n)

---

## 20. Minimum Number of Operations to Make Array Continuous

**Description:** Replace elements to make array contiguous [x, x+1, ..., x+n-1].

**Approach:** Sort, sliding window; not primarily bit manipulation.

```python
def minOperations(nums):
    n = len(nums)
    nums = sorted(set(nums))
    res = n
    j = 0
    for i in range(len(nums)):
        while j < len(nums) and nums[j] < nums[i] + n:
            j += 1
        res = min(res, n - (j - i))
    return res
```

Time: O(n log n) | Space: O(n)

---

## 21. Count Pairs With Given XOR

**Description:** Count pairs (i,j) such that arr[i] XOR arr[j] == target.

**Approach:** For each a, count occurrences of a XOR target; use hash map.

```python
def countPairs(nums, target):
    from collections import Counter
    c = Counter(nums)
    return sum(c.get(x ^ target, 0) for x in nums) // 2
```

Time: O(n) | Space: O(n)

---

## 22. Longest Nice Substring

**Description:** Longest substring where every letter has both upper and lower.

**Approach:** Bitmask for seen chars; divide and conquer on first "bad" char.

```python
def longestNiceSubstring(s):
    if len(s) < 2:
        return ""
    chars = set(s)
    for i, c in enumerate(s):
        if c.swapcase() not in chars:
            left = longestNiceSubstring(s[:i])
            right = longestNiceSubstring(s[i+1:])
            return max(left, right, key=len)
    return s
```

Time: O(n^2) | Space: O(n)

---

## 23. Find All Duplicates in an Array

**Description:** Array of n elements, each in [1,n]; elements appear once or twice.

**Approach:** Use index as flag (negate or add n); bit manipulation for in-place.

```python
def findDuplicates(nums):
    res = []
    for x in nums:
        i = abs(x) - 1
        if nums[i] < 0:
            res.append(abs(x))
        else:
            nums[i] = -nums[i]
    return res
```

Time: O(n) | Space: O(1)

---

## 24. Bitwise AND of Numbers Range

**Description:** AND of all numbers in [left, right].

**Approach:** Find common prefix of left and right in binary; result is that prefix with rest zeros.

```python
def rangeBitwiseAnd(left, right):
    shift = 0
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift
```

Time: O(32) | Space: O(1)

---

## 25. UTF-8 Validation

**Description:** Check if byte sequence is valid UTF-8.

**Approach:** Use bit masks to check leading bits of each byte.

```python
def validUtf8(data):
    n = 0
    for x in data:
        if n > 0:
            if (x >> 6) != 0b10:
                return False
            n -= 1
        elif (x >> 7) == 0:
            n = 0
        elif (x >> 5) == 0b110:
            n = 1
        elif (x >> 4) == 0b1110:
            n = 2
        elif (x >> 3) == 0b11110:
            n = 3
        else:
            return False
    return n == 0
```

Time: O(n) | Space: O(1)

---

# Hard Problems

## 1. Maximum XOR With an Element From Array

**Description:** Queries: max XOR of xi with any arr[j] where arr[j] <= mi.

**Approach:** Offline: sort queries by mi; trie with numbers <= mi; query max XOR.

```python
def maximizeXor(nums, queries):
    nums.sort()
    qs = sorted(enumerate(queries), key=lambda x: x[1][1])
    trie, res, j = {}, [0] * len(queries), 0
    for idx, (x, m) in qs:
        while j < len(nums) and nums[j] <= m:
            node = trie
            for i in range(31, -1, -1):
                b = (nums[j] >> i) & 1
                node = node.setdefault(b, {})
            j += 1
        if not trie:
            res[idx] = -1
            continue
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur = (cur << 1) + (1 if want in node else 0)
            node = node[want] if want in node else node[b]
        res[idx] = cur
    return res
```

Time: O((n+q) * 32) | Space: O(n * 32)

---

## 2. Number of Ways to Wear Different Hats to Each Other

**Description:** n people, each has list of hats; assign distinct hat to each.

**Approach:** Bitmask DP on people; dp[mask][hat] = ways to assign hats to mask people using first hat types.

```python
def numberWays(hats):
    from collections import defaultdict
    n = len(hats)
    people_by_hat = defaultdict(list)
    for i, h in enumerate(hats):
        for hat in h:
            people_by_hat[hat].append(i)
    dp = {0: 1}
    for hat in range(1, 41):
        for person in people_by_hat.get(hat, []):
            ndp = {}
            for mask, ways in dp.items():
                ndp[mask] = ndp.get(mask, 0) + ways
                if not (mask >> person) & 1:
                    nm = mask | (1 << person)
                    ndp[nm] = ndp.get(nm, 0) + ways
            dp = ndp
    return dp.get((1 << n) - 1, 0) % (10**9 + 7)
```

Time: O(40 * n * 2^n) | Space: O(2^n)

---

## 3. Minimum Cost to Connect Two Groups of Points

**Description:** Connect left group to right with minimum cost; each point must be connected.

**Approach:** Bitmask for right group coverage; DP over left group and mask.

```python
def connectTwoGroups(cost):
    m, n = len(cost), len(cost[0])
    min_right = [min(cost[i][j] for i in range(m)) for j in range(n)]
    dp = {}
    def dfs(i, mask):
        if (i, mask) in dp:
            return dp[(i, mask)]
        if i == m:
            return sum(min_right[j] for j in range(n) if not (mask >> j) & 1)
        res = float('inf')
        for j in range(n):
            res = min(res, cost[i][j] + dfs(i + 1, mask | (1 << j)))
        dp[(i, mask)] = res
        return res
    return dfs(0, 0)
```

Time: O(m * n * 2^n) | Space: O(m * 2^n)

---

## 4. Maximum Score Words Formed by Letters

**Description:** Choose words to maximize score; each letter has limited count.

**Approach:** Bitmask over words; for each subset check letter feasibility.

```python
def maxScoreWords(words, letters, score):
    from collections import Counter
    count = Counter(letters)
    def feasible(mask):
        c = Counter()
        for i in range(len(words)):
            if (mask >> i) & 1:
                c.update(words[i])
        return all(c[x] <= count[x] for x in c)
    def word_score(mask):
        s = 0
        for i in range(len(words)):
            if (mask >> i) & 1:
                for c in words[i]:
                    s += score[ord(c) - 97]
        return s
    res = 0
    for mask in range(1 << len(words)):
        if feasible(mask):
            res = max(res, word_score(mask))
    return res
```

Time: O(2^n * L) | Space: O(n)

---

## 5. Number of Ways to Build Sturdy Brick Wall

**Description:** Build wall with bricks; avoid certain boundaries.

**Approach:** Bitmask for row patterns; DP with compatibility check.

```python
def buildWall(height, width, bricks):
    masks = []
    def dfs(cur, pos, mask):
        if pos == width:
            masks.append(mask)
            return
        for b in bricks:
            if pos + b <= width:
                npos = pos + b
                nmask = mask if npos == width else mask | (1 << (npos - 1))
                dfs(cur, npos, nmask)
    dfs(0, 0, 0)
    from collections import defaultdict
    dp = defaultdict(int)
    for m in masks:
        dp[m] = 1
    for _ in range(height - 1):
        ndp = defaultdict(int)
        for m1 in dp:
            for m2 in masks:
                if not (m1 & m2):
                    ndp[m2] += dp[m1]
        dp = ndp
    return sum(dp.values()) % (10**9 + 7)
```

Time: O(h * patterns^2) | Space: O(patterns)

---

## 6. Maximum AND Sum of Array

**Description:** Assign numbers to slots; maximize sum of (num AND slot_index).

**Approach:** Bitmask DP; assign numbers to slots represented by mask.

```python
def maximumANDSum(nums, numSlots):
    n = len(nums)
    dp = [0] * (1 << (2 * numSlots))
    for mask in range(1 << (2 * numSlots)):
        idx = bin(mask).count('1')
        if idx >= n:
            continue
        for slot in range(numSlots):
            for count in range(1, 3):
                b = 2 * slot + count - 1
                if (mask >> b) & 1:
                    continue
                new_mask = mask | (1 << b)
                dp[new_mask] = max(dp[new_mask], dp[mask] + (nums[idx] & (slot + 1)))
    return max(dp)
```

Time: O(n * slots * 2^(2*slots)) | Space: O(2^(2*slots))

---

## 7. Maximum Compatibility Score Sum

**Description:** Assign students to mentors; maximize compatibility sum.

**Approach:** Bitmask DP; dp[mask] = max score assigning to first popcount(mask) students.

```python
def maxCompatibilitySum(students, mentors):
    m = len(students)
    def score(s, t):
        return sum(a == b for a, b in zip(s, t))
    dp = [0] * (1 << m)
    for mask in range(1 << m):
        j = bin(mask).count('1')
        if j >= m:
            continue
        for i in range(m):
            if (mask >> i) & 1:
                continue
            dp[mask | (1 << i)] = max(dp[mask | (1 << i)], dp[mask] + score(students[j], mentors[i]))
    return dp[(1 << m) - 1]
```

Time: O(m^2 * 2^m) | Space: O(2^m)

---

## 8. Minimum Cost to Cut a Stick

**Description:** Cut stick at given positions; cost = stick length.

**Approach:** Interval DP; can use bitmask for cut positions.

```python
def minCost(n, cuts):
    cuts = sorted([0] + cuts + [n])
    m = len(cuts)
    dp = [[0] * m for _ in range(m)]
    for L in range(2, m):
        for i in range(m - L):
            j = i + L
            dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i+1, j)) + cuts[j] - cuts[i]
    return dp[0][m-1]
```

Time: O(m^3) | Space: O(m^2)

---

## 9. Number of Ways to Form a Target String

**Description:** Form target by picking one char per column from matrix.

**Approach:** DP with frequency precomputation; bitmask for column selection in variants.

```python
def numWays(words, target):
    n, m = len(words[0]), len(target)
    freq = [[0] * 26 for _ in range(n)]
    for w in words:
        for i, c in enumerate(w):
            freq[i][ord(c) - 97] += 1
    dp = [1] + [0] * len(target)
    for i in range(n):
        for j in range(len(target) - 1, -1, -1):
            dp[j+1] = (dp[j+1] + dp[j] * freq[i][ord(target[j]) - 97]) % (10**9 + 7)
    return dp[len(target)]
```

Time: O(n * m * 26) | Space: O(m)

---

## 10. Maximum Number of Achievable Transfer Requests

**Description:** Buildings have employees; transfer requests (from, to); maximize balanced requests.

**Approach:** Bitmask over requests; for each subset check if net flow is zero for all buildings.

```python
def maximumRequests(n, requests):
    res = 0
    for mask in range(1 << len(requests)):
        flow = [0] * n
        for i, (a, b) in enumerate(requests):
            if (mask >> i) & 1:
                flow[a] -= 1
                flow[b] += 1
        if all(f == 0 for f in flow):
            res = max(res, bin(mask).count('1'))
    return res
```

Time: O(2^r * n) | Space: O(n)

---

## 11. Find Minimum Time to Finish All Jobs

**Description:** n jobs, k workers; minimize max time (each job to one worker).

**Approach:** Bitmask DP; dp[mask][k] = min max time for mask jobs with k workers.

```python
def minimumTimeRequired(jobs, k):
    n = len(jobs)
    dp = [[float('inf')] * (k + 1) for _ in range(1 << n)]
    dp[0][0] = 0
    for mask in range(1 << n):
        for w in range(k):
            if dp[mask][w] == float('inf'):
                continue
            submask = ((1 << n) - 1) ^ mask
            s = submask
            while s:
                cost = sum(jobs[i] for i in range(n) if (s >> i) & 1)
                nm = mask | s
                dp[nm][w+1] = min(dp[nm][w+1], max(dp[mask][w], cost))
                s = (s - 1) & submask
    return dp[-1][k]
```

Time: O(3^n * k) | Space: O(2^n * k)

---

## 12. Maximum Students Taking Exam

**Description:** Seating in grid; students cannot be adjacent; maximize count.

**Approach:** Bitmask DP per row; dp[row][mask] = max students for first row rows with row having mask.

```python
def maxStudents(seats):
    m, n = len(seats), len(seats[0])
    def valid(mask, row):
        for j in range(n):
            if (mask >> j) & 1:
                if not seats[row][j] or (j and (mask >> (j-1)) & 1):
                    return False
        return True
    dp = {0: 0}
    for i in range(m):
        ndp = {}
        for mask in range(1 << n):
            if not valid(mask, i):
                continue
            cnt = bin(mask).count('1')
            for pmask, total in dp.items():
                if any((mask >> j) & (pmask >> (j+1)) or (mask >> (j+1)) & (pmask >> j) for j in range(n-1)):
                    continue
                ndp[mask] = max(ndp.get(mask, 0), total + cnt)
        dp = ndp
    return max(dp.values()) if dp else 0
```

Time: O(m * 4^n) | Space: O(2^n)

---

## 13. Minimum Number of Work Sessions to Finish Tasks

**Description:** Tasks with time; sessions have limit; minimize sessions.

**Approach:** Bitmask DP; dp[mask] = min sessions for mask tasks.

```python
def minSessions(tasks, sessionTime):
    n = len(tasks)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        time = 0
        for i in range(n):
            if (mask >> i) & 1:
                continue
            if time + tasks[i] <= sessionTime:
                time += tasks[i]
                dp[mask | (1 << i)] = min(dp[mask | (1 << i)], dp[mask])
            else:
                dp[mask | (1 << i)] = min(dp[mask | (1 << i)], dp[mask] + 1)
                time = tasks[i]
                for j in range(i + 1, n):
                    if not (mask >> j) & 1 and time + tasks[j] <= sessionTime:
                        time += tasks[j]
                        dp[mask | (1 << i) | (1 << j)] = min(dp.get(mask | (1<<i)|(1<<j), float('inf')), dp[mask])
                break
    return dp[-1] if dp[-1] != float('inf') else 0
```

Time: O(n * 2^n) | Space: O(2^n)

---

## 14. Count Number of Maximum Bitwise-OR Subsets

**Description:** Count subsets that achieve maximum OR value.

**Approach:** Find max OR; DP counting subsets with each OR value.

```python
def countMaxOrSubsets(nums):
    max_or = 0
    for x in nums:
        max_or |= x
    dp = {0: 1}
    for x in nums:
        ndp = dict(dp)
        for mask, count in dp.items():
            nm = mask | x
            ndp[nm] = ndp.get(nm, 0) + count
        dp = ndp
    return dp.get(max_or, 0)
```

Time: O(n * 2^20) | Space: O(2^20)

---

## 15. Number of Wonderful Substrings

**Description:** Wonderful = at most one letter appears odd times.

**Approach:** Prefix XOR of parity bits; for each prefix count previous prefixes with same or 1-bit-different XOR.

```python
def wonderfulSubstrings(word):
    count = {0: 1}
    xor = 0
    res = 0
    for c in word:
        xor ^= 1 << (ord(c) - 97)
        res += count.get(xor, 0)
        for i in range(10):
            res += count.get(xor ^ (1 << i), 0)
        count[xor] = count.get(xor, 0) + 1
    return res
```

Time: O(n * 10) | Space: O(2^10)
