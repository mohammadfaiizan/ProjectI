# Easy Searching and Sorting Problems

## 1. Binary Search

Given sorted array and target, return index of target or -1. Standard binary search. left=0, right=n-1, mid=(left+right)//2, adjust boundaries.

```python
def binarySearch(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

Time: O(log n) | Space: O(1)

---

## 2. First Bad Version

n versions, first bad causes all after to be bad. Find first bad with API isBadVersion(i). Binary search for first occurrence where isBadVersion(mid) is True.

```python
def firstBadVersion(n):
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if isBadVersion(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

Time: O(log n) | Space: O(1)

---

## 3. Search Insert Position

Sorted array, find index to insert target. Return index where it would be or exists. Lower bound binary search. Find leftmost index where arr[i] >= target.

```python
def searchInsert(nums, target):
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

## 4. Squares of a Sorted Array

Sorted array (can have negatives), return sorted squares. Two pointers from ends (abs values), merge into result.

```python
def sortedSquares(nums):
    n = len(nums)
    res = [0] * n
    l, r, k = 0, n - 1, n - 1
    while l <= r:
        if abs(nums[l]) > abs(nums[r]):
            res[k] = nums[l] ** 2
            l += 1
        else:
            res[k] = nums[r] ** 2
            r -= 1
        k -= 1
    return res
```

Time: O(n) | Space: O(n)

---

## 5. Merge Sorted Array

Two sorted arrays, merge into first (has extra space). Three pointers from end. Compare and place larger at end of first array.

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

## 6. Intersection of Two Arrays

Return intersection of two arrays (unique elements). Sort both, two pointers. Or use set for O(n) average.

```python
def intersection(nums1, nums2):
    nums1.sort()
    nums2.sort()
    i, j, res = 0, 0, []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            if not res or res[-1] != nums1[i]:
                res.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return res
```

Time: O(n log n) | Space: O(1)

---

## 7. Intersection of Two Arrays II

Return intersection with multiplicity (count matters). Sort both, two pointers. Advance pointer with smaller value.

```python
def intersect(nums1, nums2):
    nums1.sort()
    nums2.sort()
    i, j, res = 0, 0, []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            res.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return res
```

Time: O(n log n) | Space: O(1)

---

## 8. Valid Anagram

Check if two strings are anagrams. Sort both and compare. Or count frequency of each character.

```python
def isAnagram(s, t):
    return sorted(s) == sorted(t)
```

Time: O(n log n) | Space: O(n)

---

## 9. Two Sum (Sorted)

Sorted array, find two numbers that sum to target. Two pointers at start and end. If sum < target, left++. If sum > target, right--.

```python
def twoSum(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == target:
            return [l + 1, r + 1]
        if s < target:
            l += 1
        else:
            r -= 1
    return []
```

Time: O(n) | Space: O(1)

---

## 10. Remove Duplicates from Sorted Array

In-place remove duplicates, return new length. Two pointers. Slow for unique position, fast scans. Copy when different.

```python
def removeDuplicates(nums):
    if not nums:
        return 0
    k = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[k - 1]:
            nums[k] = nums[i]
            k += 1
    return k
```

Time: O(n) | Space: O(1)

---

## 11. Merge Two Sorted Lists

Merge two sorted linked lists. Dummy node, compare heads, attach smaller, advance.

```python
def mergeTwoLists(l1, l2):
    dummy = ListNode()
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

Time: O(n + m) | Space: O(1)

---

## 12. Find Smallest Letter Greater Than Target

Sorted letters (wrap around), find smallest > target. Binary search for first letter > target. Return letters[left % n].

```python
def nextGreatestLetter(letters, target):
    lo, hi = 0, len(letters)
    while lo < hi:
        mid = (lo + hi) // 2
        if letters[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return letters[lo % len(letters)]
```

Time: O(log n) | Space: O(1)

---

## 13. Peak Index in a Mountain Array

Array increases then decreases. Find peak index. Binary search. If arr[mid] < arr[mid+1], peak in right half.

```python
def peakIndexInMountainArray(arr):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < arr[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

Time: O(log n) | Space: O(1)

---

## 14. Find the Distance Value Between Two Arrays

Count arr1 elements such that no arr2 element within d. Sort arr2. For each arr1 element, binary search nearest in arr2, check distance.

```python
def findTheDistanceValue(arr1, arr2, d):
    arr2.sort()
    count = 0
    for x in arr1:
        lo, hi = 0, len(arr2) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if arr2[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if abs(arr2[lo] - x) > d and (lo == 0 or abs(arr2[lo-1] - x) > d):
            count += 1
    return count
```

Time: O(n log m) | Space: O(1)

---

## 15. Check If N and Its Double Exist

Check if there exist i,j with arr[i] == 2*arr[j]. Sort. For each element, binary search for 2*x or x/2.

```python
def checkIfExist(arr):
    arr.sort()
    for i, x in enumerate(arr):
        target = 2 * x if x >= 0 else x // 2
        if x < 0 and x % 2 == 1:
            continue
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target and mid != i:
                return True
            if arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
    return False
```

Time: O(n log n) | Space: O(1)

---

## 16. Relative Sort Array

Sort arr1 by order defined in arr2. Elements not in arr2 go at end, sorted. Count frequency of arr1. Traverse arr2, output count times. Append remaining sorted.

```python
def relativeSortArray(arr1, arr2):
    from collections import Counter
    cnt = Counter(arr1)
    res = []
    for x in arr2:
        res.extend([x] * cnt.pop(x, 0))
    for x in sorted(cnt.keys()):
        res.extend([x] * cnt[x])
    return res
```

Time: O(n log n) | Space: O(n)

---

## 17. Sort Array By Parity

Move all even elements before odd. Order among evens/odds does not matter. Two pointers. Swap when left odd and right even.

```python
def sortArrayByParity(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        if nums[l] % 2 > nums[r] % 2:
            nums[l], nums[r] = nums[r], nums[l]
        if nums[l] % 2 == 0:
            l += 1
        if nums[r] % 2 == 1:
            r -= 1
    return nums
```

Time: O(n) | Space: O(1)

---

## 18. Sort Array By Parity II

Reorder so even indices have even values, odd indices have odd. Two pointers for even and odd positions. Swap when both wrong.

```python
def sortArrayByParityII(nums):
    j = 1
    for i in range(0, len(nums), 2):
        if nums[i] % 2:
            while nums[j] % 2:
                j += 2
            nums[i], nums[j] = nums[j], nums[i]
    return nums
```

Time: O(n) | Space: O(1)

---

## 19. Largest Perimeter Triangle

From sides array, find largest perimeter of valid triangle. Sort descending. For consecutive triple (a,b,c) with a<b+c, return a+b+c.

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

## 20. Height Checker

Students in row by height. Return count of positions where expected != actual. Sort copy, compare with original. Count mismatches.

```python
def heightChecker(heights):
    expected = sorted(heights)
    return sum(1 for a, b in zip(heights, expected) if a != b)
```

Time: O(n log n) | Space: O(n)

---

## 21. Third Maximum Number

Return third distinct maximum. If less than 3 distinct, return max. Track three largest. Or sort, deduplicate, return third from end.

```python
def thirdMax(nums):
    nums = sorted(set(nums), reverse=True)
    return nums[2] if len(nums) >= 3 else nums[0]
```

Time: O(n log n) | Space: O(n)

---

## 22. Find Target Indices After Sorting

Sort array, return all indices where target appears. Sort, linear scan for target. Or count elements < target and = target.

```python
def targetIndices(nums, target):
    nums.sort()
    return [i for i, x in enumerate(nums) if x == target]
```

Time: O(n log n) | Space: O(1)

---

## 23. Count Negative Numbers in Sorted Matrix

Grid sorted non-increasing row and column wise. Count negatives. Start top-right. If negative, all below negative; count += rows-r; move left.

```python
def countNegatives(grid):
    m, n = len(grid), len(grid[0])
    r, c, count = 0, n - 1, 0
    while r < m and c >= 0:
        if grid[r][c] < 0:
            count += m - r
            c -= 1
        else:
            r += 1
    return count
```

Time: O(m + n) | Space: O(1)

---

## 24. Kth Missing Positive Number

Strictly increasing array. Find kth missing positive integer. For each index i, missing count = arr[i] - i - 1. Binary search for first index where missing >= k.

```python
def findKthPositive(arr, k):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] - mid - 1 < k:
            lo = mid + 1
        else:
            hi = mid
    return lo + k
```

Time: O(log n) | Space: O(1)

---

## 25. Fair Candy Swap

Two arrays (candy sizes). Swap one element each so both have equal total. Compute sum difference. For each a in A, need b = a + diff/2 in B. Use set for B.

```python
def fairCandySwap(aliceSizes, bobSizes):
    diff = (sum(aliceSizes) - sum(bobSizes)) // 2
    bob_set = set(bobSizes)
    for a in aliceSizes:
        if a - diff in bob_set:
            return [a, a - diff]
```

Time: O(n + m) | Space: O(m)

---

## 26. Find the Town Judge

n people, trust pairs. Judge trusts nobody, everyone trusts judge. Count in-degree and out-degree. Judge has in=n-1, out=0.

```python
def findJudge(n, trust):
    deg = [0] * (n + 1)
    for a, b in trust:
        deg[a] -= 1
        deg[b] += 1
    for i in range(1, n + 1):
        if deg[i] == n - 1:
            return i
    return -1
```

Time: O(n) | Space: O(n)

---

## 27. Contains Duplicate

Check if array has duplicate. Sort and adjacent compare. Or use set.

```python
def containsDuplicate(nums):
    nums.sort()
    return any(nums[i] == nums[i+1] for i in range(len(nums)-1))
```

Time: O(n log n) | Space: O(1)

---

## 28. Missing Number

Array of n distinct numbers from [0,n]. Find missing one. Sum 0+1+...+n minus array sum. Or XOR all with 0..n.

```python
def missingNumber(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)
```

Time: O(n) | Space: O(1)
