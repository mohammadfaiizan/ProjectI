# Easy Greedy Problems

## 1. Best Time to Buy and Sell Stock

**Description**: One buy and one sell; maximize profit.

**Approach**: Track min price seen; at each day, profit = price - min_so_far. Take max.

```python
def maxProfit(prices):
    min_p, res = float('inf'), 0
    for p in prices:
        min_p = min(min_p, p)
        res = max(res, p - min_p)
    return res
```

Time: O(n) | Space: O(1)

---

## 2. Assign Cookies

**Description**: Assign cookies to children; each child has greed factor, each cookie has size. Child satisfied if cookie size >= greed.

**Approach**: Sort both arrays; two pointers. Assign smallest cookie that satisfies each child.

```python
def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = j = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            i += 1
        j += 1
    return i
```

Time: O(n log n) | Space: O(1)

---

## 3. Lemonade Change

**Description**: Customers pay 5, 10, or 20. Must give correct change. Return if possible.

**Approach**: Greedy change: for 10 give one 5; for 20 give one 10+5 or three 5s (prefer 10+5).

```python
def lemonadeChange(bills):
    five = ten = 0
    for b in bills:
        if b == 5:
            five += 1
        elif b == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        else:
            if ten and five:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    return True
```

Time: O(n) | Space: O(1)

---

## 4. Valid Parentheses

**Description**: Check if string has matching brackets.

**Approach**: Stack; push open, pop on close and match. Greedy: match immediately.

```python
def isValid(s):
    stack = []
    m = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in m:
            if not stack or stack[-1] != m[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return len(stack) == 0
```

Time: O(n) | Space: O(n)

---

## 5. Merge Sorted Array

**Description**: Merge two sorted arrays in place.

**Approach**: Two pointers from end; place larger element at end of result.

```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

Time: O(m + n) | Space: O(1)

---

## 6. Majority Element

**Description**: Find element appearing more than n/2 times.

**Approach**: Boyer-Moore voting: cancel different pairs; survivor is majority.

```python
def majorityElement(nums):
    cand, count = None, 0
    for x in nums:
        if count == 0:
            cand = x
        count += 1 if x == cand else -1
    return cand
```

Time: O(n) | Space: O(1)

---

## 7. Maximum Subarray

**Description**: Find contiguous subarray with maximum sum.

**Approach**: Kadane: at each position, either extend previous subarray or start new. Greedy: extend if sum stays positive.

```python
def maxSubArray(nums):
    cur = res = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        res = max(res, cur)
    return res
```

Time: O(n) | Space: O(1)

---

## 8. Jump Game

**Description**: Can you reach last index? Each element is max jump from that position.

**Approach**: Track max reachable index; if current index exceeds it, return false.

```python
def canJump(nums):
    reach = 0
    for i, x in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + x)
    return True
```

Time: O(n) | Space: O(1)

---

## 9. Climbing Stairs

**Description**: n steps; climb 1 or 2 at a time. Count ways.

**Approach**: DP (Fibonacci). Not strictly greedy but simple recurrence.

```python
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```

Time: O(n) | Space: O(1)

---

## 10. Best Time to Buy and Sell Stock II

**Description**: Unlimited buys and sells; maximize profit.

**Approach**: Greedy: add profit whenever price[i] > price[i-1] (buy day before, sell today).

```python
def maxProfit(prices):
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
```

Time: O(n) | Space: O(1)

---

## 11. Is Subsequence

**Description**: Check if s is subsequence of t.

**Approach**: Two pointers; match chars of s in t left to right.

```python
def isSubsequence(s, t):
    i = 0
    for c in t:
        if i < len(s) and s[i] == c:
            i += 1
    return i == len(s)
```

Time: O(n) | Space: O(1)

---

## 12. Can Place Flowers

**Description**: Flower bed with 0s and 1s; can't plant adjacent. Place n new flowers.

**Approach**: Greedy: plant at each valid 0 (both neighbors 0); count until n reached.

```python
def canPlaceFlowers(flowerbed, n):
    f = [0] + flowerbed + [0]
    for i in range(1, len(f) - 1):
        if f[i-1] == f[i] == f[i+1] == 0:
            f[i] = 1
            n -= 1
        if n <= 0:
            return True
    return n <= 0
```

Time: O(n) | Space: O(1)

---

## 13. Minimum Cost to Move Chips

**Description**: Chips at positions; move 2 positions cost 0, 1 position cost 1. Min cost to stack all.

**Approach**: min(count_odd, count_even) - move all to one parity at 0 cost, then min group moves 1.

```python
def minCostToMoveChips(position):
    odd = sum(1 for p in position if p % 2)
    return min(odd, len(position) - odd)
```

Time: O(n) | Space: O(1)

---

## 14. Maximum Units on a Truck

**Description**: Box types with (count, units per box). Truck holds limited boxes. Maximize units.

**Approach**: Sort by units per box descending; take boxes greedily.

```python
def maximumUnits(boxTypes, truckSize):
    boxTypes.sort(key=lambda x: -x[1])
    units = 0
    for count, u in boxTypes:
        take = min(truckSize, count)
        units += take * u
        truckSize -= take
        if truckSize == 0:
            break
    return units
```

Time: O(n log n) | Space: O(1)

---

## 15. Two City Scheduling

**Description**: 2n people; cost to send to city A or B. Send n to each. Minimize cost.

**Approach**: Sort by (costA - costB); first n to A, rest to B.

```python
def twoCitySchedCost(costs):
    costs.sort(key=lambda x: x[0] - x[1])
    n = len(costs) // 2
    return sum(c[0] for c in costs[:n]) + sum(c[1] for c in costs[n:])
```

Time: O(n log n) | Space: O(1)

---

## 16. Split a String in Balanced Strings

**Description**: String of L and R. Split into balanced (equal L and R) substrings. Max count.

**Approach**: Count L and R; whenever they equal, that's a split. Greedy count.

```python
def balancedStringSplit(s):
    bal = count = 0
    for c in s:
        bal += 1 if c == 'L' else -1
        if bal == 0:
            count += 1
    return count
```

Time: O(n) | Space: O(1)

---

## 17. Minimum Add to Make Parentheses Valid

**Description**: Add minimum parentheses to make string valid.

**Approach**: Count open; if close without open, need to add open. At end add remaining open.

```python
def minAddToMakeValid(s):
    open_needed = close_needed = 0
    for c in s:
        if c == '(':
            close_needed += 1
        else:
            if close_needed:
                close_needed -= 1
            else:
                open_needed += 1
    return open_needed + close_needed
```

Time: O(n) | Space: O(1)

---

## 18. Partition Labels

**Description**: Partition string so each letter appears in at most one part. Minimize number of parts.

**Approach**: Track last index of each char; extend partition until current index equals last of all chars in partition.

```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    start = end = 0
    res = []
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            res.append(end - start + 1)
            start = i + 1
    return res
```

Time: O(n) | Space: O(1)

---

## 19. Score After Flipping Matrix

**Description**: Binary matrix; flip rows or columns. Maximize score (each row as binary number).

**Approach**: Greedy: ensure first column all 1s (flip rows); then flip columns where 0s > 1s.

```python
def matrixScore(grid):
    m, n = len(grid), len(grid[0])
    for i in range(m):
        if grid[i][0] == 0:
            for j in range(n):
                grid[i][j] ^= 1
    res = 0
    for j in range(n):
        ones = sum(grid[i][j] for i in range(m))
        res = res * 2 + max(ones, m - ones)
    return res
```

Time: O(m * n) | Space: O(1)

---

## 20. DI String Match

**Description**: String of I and D. Permutation of 0..n where I means increase, D means decrease.

**Approach**: I: assign smallest unused; D: assign largest unused. Two pointers at ends.

```python
def diStringMatch(s):
    lo, hi, res = 0, len(s), []
    for c in s:
        res.append(lo if c == 'I' else hi)
        lo, hi = (lo + 1, hi) if c == 'I' else (lo, hi - 1)
    res.append(lo)
    return res
```

Time: O(n) | Space: O(1)

---

## 21. Play with Chips

**Description**: Same as minimum cost to move chips.

**Approach**: min(odd_count, even_count).

```python
def minCostToMoveChips(chips):
    odd = sum(1 for c in chips if c % 2)
    return min(odd, len(chips) - odd)
```

Time: O(n) | Space: O(1)

---

## 22. Maximize Sum Of Array After K Negations

**Description**: Negate exactly k elements. Maximize sum.

**Approach**: Sort; negate negatives first. If k left, negate smallest absolute value repeatedly.

```python
def largestSumAfterKNegations(nums, k):
    nums.sort()
    for i in range(len(nums)):
        if k and nums[i] < 0:
            nums[i] *= -1
            k -= 1
    if k % 2:
        nums[nums.index(min(nums))] *= -1
    return sum(nums)
```

Time: O(n log n) | Space: O(1)

---

## 23. Last Stone Weight

**Description**: Repeatedly smash two largest stones (difference remains). Final stone weight.

**Approach**: Max-heap; pop two, push difference until one or zero left.

```python
def lastStoneWeight(stones):
    import heapq
    h = [-s for s in stones]
    heapq.heapify(h)
    while len(h) > 1:
        a, b = -heapq.heappop(h), -heapq.heappop(h)
        if a != b:
            heapq.heappush(h, -(a - b))
    return -h[0] if h else 0
```

Time: O(n log n) | Space: O(n)

---

## 24. Array Partition

**Description**: Partition 2n numbers into n pairs. Maximize sum of min of each pair.

**Approach**: Sort; pair consecutive. Greedy: min of pair is maximized when we pair smallest with second smallest, etc. So sort and take every even index.

```python
def arrayPairSum(nums):
    nums.sort()
    return sum(nums[::2])
```

Time: O(n log n) | Space: O(1)

---

## 25. Monotonic Array

**Description**: Check if array is monotonic (non-decreasing or non-increasing).

**Approach**: One pass; track direction; reject if violates.

```python
def isMonotonic(nums):
    inc = dec = True
    for i in range(1, len(nums)):
        inc &= nums[i] >= nums[i-1]
        dec &= nums[i] <= nums[i-1]
    return inc or dec
```

Time: O(n) | Space: O(1)

---

## 26. Largest Perimeter Triangle

**Description**: Form triangle from three array elements. Max perimeter.

**Approach**: Sort descending; check triplets (a, b, c) for a < b + c. First valid is max perimeter.

```python
def largestPerimeter(nums):
    nums.sort(reverse=True)
    for i in range(len(nums) - 2):
        if nums[i] < nums[i+1] + nums[i+2]:
            return nums[i] + nums[i+1] + nums[i+2]
    return 0
```

Time: O(n log n) | Space: O(1)

---

## 27. Distribute Candies

**Description**: 2n candies, n types. Sister gets n candies. Max distinct types she can get.

**Approach**: min(unique_count, n).

```python
def distributeCandies(candyType):
    return min(len(set(candyType)), len(candyType) // 2)
```

Time: O(n) | Space: O(n)

---

## 28. Non-decreasing Array

**Description**: Can you make array non-decreasing by changing at most one element?

**Approach**: Count inversions; if more than one, check if single fix works (e.g., lower prev or raise current).

```python
def checkPossibility(nums):
    err = -1
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            if err != -1:
                return False
            err = i
    if err in [-1, 0, len(nums) - 2]:
        return True
    return nums[err - 1] <= nums[err + 1] or nums[err] <= nums[err + 2]
```

Time: O(n) | Space: O(1)
