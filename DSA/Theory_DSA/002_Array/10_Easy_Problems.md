# Easy Array Problems

## 1. Two Sum

Find two indices such that arr[i] + arr[j] = target. Use hashmap: for each x, check if target - x exists.

```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```

Time: O(n) | Space: O(n)

---

## 2. Remove Duplicates from Sorted Array

In-place, return new length. Two pointers: write index for unique elements.

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    w = 1
    for r in range(1, len(nums)):
        if nums[r] != nums[r - 1]:
            nums[w] = nums[r]
            w += 1
    return w
```

Time: O(n) | Space: O(1)

---

## 3. Remove Element

Remove all occurrences of val in-place. Two pointers.

```python
def remove_element(nums, val):
    w = 0
    for x in nums:
        if x != val:
            nums[w] = x
            w += 1
    return w
```

Time: O(n) | Space: O(1)

---

## 4. Search Insert Position

Find index where target would be inserted in sorted array. Binary search.

```python
def search_insert(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

Time: O(log n) | Space: O(1)

---

## 5. Maximum Subarray (Kadane)

Find contiguous subarray with largest sum. Kadane's algorithm.

```python
def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```

Time: O(n) | Space: O(1)

---

## 6. Plus One

Add 1 to number represented as array of digits. Handle carry from right.

```python
def plus_one(digits):
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += 1
        if digits[i] < 10:
            return digits
        digits[i] = 0
    return [1] + digits
```

Time: O(n) | Space: O(1)

---

## 7. Merge Sorted Array

Merge two sorted arrays. nums1 has extra space. Two pointers from end.

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

## 8. Pascal's Triangle

Generate first n rows. Each row: sum of adjacent elements from previous row.

```python
def generate_pascal(n):
    tri = [[1]]
    for _ in range(n - 1):
        row = [1] + [tri[-1][i] + tri[-1][i + 1] for i in range(len(tri[-1]) - 1)] + [1]
        tri.append(row)
    return tri
```

Time: O(n^2) | Space: O(n^2)

---

## 9. Best Time to Buy and Sell Stock

One buy, one sell. Max profit = max price - min price before it. Track min so far.

```python
def max_profit(prices):
    best, mn = 0, float('inf')
    for p in prices:
        mn = min(mn, p)
        best = max(best, p - mn)
    return best
```

Time: O(n) | Space: O(1)

---

## 10. Single Number

Every element appears twice except one. XOR all elements.

```python
def single_number(nums):
    res = 0
    for x in nums:
        res ^= x
    return res
```

Time: O(n) | Space: O(1)

---

## 11. Majority Element

Element appearing more than n/2 times. Boyer-Moore voting.

```python
def majority_element(nums):
    cand, cnt = None, 0
    for x in nums:
        if cnt == 0:
            cand, cnt = x, 1
        elif x == cand:
            cnt += 1
        else:
            cnt -= 1
    return cand
```

Time: O(n) | Space: O(1)

---

## 12. Contains Duplicate

Check if array has duplicates. Use set.

```python
def contains_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
```

Time: O(n) | Space: O(n)

---

## 13. Missing Number

Array [0,n] with one missing. Sum or XOR approach.

```python
def missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)
```

Time: O(n) | Space: O(1)

---

## 14. Move Zeroes

Move all zeros to end in-place. Two pointers.

```python
def move_zeroes(nums):
    w = 0
    for x in nums:
        if x != 0:
            nums[w] = x
            w += 1
    for i in range(w, len(nums)):
        nums[i] = 0
```

Time: O(n) | Space: O(1)

---

## 15. Intersection of Two Arrays

Find common elements. Use set intersection.

```python
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))
```

Time: O(n + m) | Space: O(n + m)

---

## 16. Intersection of Two Arrays II

Find common elements with frequency. Use Counter, decrement on match.

```python
def intersect(nums1, nums2):
    from collections import Counter
    c = Counter(nums1)
    out = []
    for x in nums2:
        if c[x] > 0:
            out.append(x)
            c[x] -= 1
    return out
```

Time: O(n + m) | Space: O(min(n, m))

---

## 17. Third Maximum Number

Find third distinct maximum. Track top three or use set and sort.

```python
def third_max(nums):
    distinct = sorted(set(nums), reverse=True)
    return distinct[2] if len(distinct) >= 3 else distinct[0]
```

Time: O(n) | Space: O(n)

---

## 18. Find All Numbers Disappeared in Array

Values [1,n], some missing. Negative marking or cyclic sort.

```python
def find_disappeared(nums):
    for x in nums:
        i = abs(x) - 1
        if nums[i] > 0:
            nums[i] *= -1
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]
```

Time: O(n) | Space: O(1)

---

## 19. Assign Cookies

Greedy: sort both, assign smallest cookie to smallest child that fits.

```python
def find_content_children(g, s):
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

## 20. Island Perimeter

Count perimeter of island (1s). Each 1 contributes 4, subtract 2 per adjacent 1.

```python
def island_perimeter(grid):
    perim = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]:
                perim += 4
                if i > 0 and grid[i - 1][j]:
                    perim -= 2
                if j > 0 and grid[i][j - 1]:
                    perim -= 2
    return perim
```

Time: O(m * n) | Space: O(1)

---

## 21. Max Consecutive Ones

Longest contiguous segment of 1s. Single pass, reset on 0.

```python
def find_max_consecutive_ones(nums):
    best = cur = 0
    for x in nums:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best
```

Time: O(n) | Space: O(1)

---

## 22. Teemo Attacking

Merge overlapping time intervals. Track end of poison.

```python
def find_poisoned_duration(time_series, duration):
    if not time_series:
        return 0
    total = duration
    for i in range(1, len(time_series)):
        total += min(duration, time_series[i] - time_series[i - 1])
    return total
```

Time: O(n) | Space: O(1)

---

## 23. Next Greater Element I

For each element in nums1, find next greater in nums2. Monotonic stack on nums2.

```python
def next_greater_element(nums1, nums2):
    st, nxt = [], {}
    for x in nums2:
        while st and st[-1] < x:
            nxt[st.pop()] = x
        st.append(x)
    return [nxt.get(x, -1) for x in nums1]
```

Time: O(n + m) | Space: O(m)

---

## 24. Keyboard Row

Filter words that can be typed using one keyboard row. Set intersection.

```python
def find_words(words):
    rows = [set("qwertyuiop"), set("asdfghjkl"), set("zxcvbnm")]
    out = []
    for w in words:
        ws = set(w.lower())
        if any(ws <= r for r in rows):
            out.append(w)
    return out
```

Time: O(n * L) | Space: O(1)

---

## 25. Find Mode in Binary Search Tree

Find most frequent value. In-order traversal, track frequency.

```python
def find_mode(root):
    from collections import Counter
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    cnt = Counter(inorder(root))
    mx = max(cnt.values()) if cnt else 0
    return [k for k, v in cnt.items() if v == mx]
```

Time: O(n) | Space: O(n)

---

## 26. Relative Ranks

Assign ranks to sorted scores. Use index mapping.

```python
def find_relative_ranks(score):
    order = sorted(range(len(score)), key=lambda i: -score[i])
    rank = [""] * len(score)
    medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    for r, i in enumerate(order):
        rank[i] = medals[r] if r < 3 else str(r + 1)
    return rank
```

Time: O(n log n) | Space: O(n)

---

## 27. Array Partition I

Pair elements to maximize sum of mins. Sort, take every other element.

```python
def array_pair_sum(nums):
    nums.sort()
    return sum(nums[::2])
```

Time: O(n log n) | Space: O(1)

---

## 28. Reshape the Matrix

Reshape matrix to new dimensions. Row-major traversal.

```python
def matrix_reshape(mat, r, c):
    flat = [x for row in mat for x in row]
    if len(flat) != r * c:
        return mat
    return [flat[i:i + c] for i in range(0, len(flat), c)]
```

Time: O(m * n) | Space: O(m * n)

---

## 29. Distribute Candies

Max distinct candy types for n/2 people. min(unique_count, n/2).

```python
def distribute_candies(candy_type):
    return min(len(set(candy_type)), len(candy_type) // 2)
```

Time: O(n) | Space: O(n)

---

## 30. Longest Harmonious Subsequence

Find longest subsequence where max-min=1. Count frequency, check adjacent counts.

```python
def find_lhs(nums):
    from collections import Counter
    c = Counter(nums)
    best = 0
    for x in c:
        if x + 1 in c:
            best = max(best, c[x] + c[x + 1])
    return best
```

Time: O(n) | Space: O(n)
