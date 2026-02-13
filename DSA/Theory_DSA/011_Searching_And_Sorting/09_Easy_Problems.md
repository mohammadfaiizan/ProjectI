# Easy Searching and Sorting Problems

## Problem List with Approach Hints

### 1. Binary Search
**Description:** Given sorted array and target, return index of target or -1.
**Approach:** Standard binary search. left=0, right=n-1, mid=(left+right)//2, adjust boundaries.

### 2. First Bad Version
**Description:** n versions, first bad causes all after to be bad. Find first bad with API isBadVersion(i).
**Approach:** Binary search for first occurrence where isBadVersion(mid) is True.

### 3. Search Insert Position
**Description:** Sorted array, find index to insert target. Return index where it would be or exists.
**Approach:** Lower bound binary search. Find leftmost index where arr[i] >= target.

### 4. Squares of a Sorted Array
**Description:** Sorted array (can have negatives), return sorted squares.
**Approach:** Two pointers from ends (abs values), merge into result. Or square and sort O(n log n).

### 5. Merge Sorted Array
**Description:** Two sorted arrays, merge into first (has extra space).
**Approach:** Three pointers from end. Compare and place larger at end of first array.

### 6. Intersection of Two Arrays
**Description:** Return intersection of two arrays (unique elements).
**Approach:** Sort both, two pointers. Or use set for O(n) average.

### 7. Intersection of Two Arrays II
**Description:** Return intersection with multiplicity (count matters).
**Approach:** Sort both, two pointers. Advance pointer with smaller value.

### 8. Valid Anagram
**Description:** Check if two strings are anagrams.
**Approach:** Sort both and compare. Or count frequency of each character.

### 9. Two Sum (Sorted)
**Description:** Sorted array, find two numbers that sum to target.
**Approach:** Two pointers at start and end. If sum < target, left++. If sum > target, right--.

### 10. Remove Duplicates from Sorted Array
**Description:** In-place remove duplicates, return new length.
**Approach:** Two pointers. Slow for unique position, fast scans. Copy when different.

### 11. Merge Two Sorted Lists
**Description:** Merge two sorted linked lists.
**Approach:** Dummy node, compare heads, attach smaller, advance.

### 12. Find Smallest Letter Greater Than Target
**Description:** Sorted letters (wrap around), find smallest > target.
**Approach:** Binary search for first letter > target. Return letters[left % n].

### 13. Peak Index in a Mountain Array
**Description:** Array increases then decreases. Find peak index.
**Approach:** Binary search. If arr[mid] < arr[mid+1], peak in right half.

### 14. Find the Distance Value Between Two Arrays
**Description:** Count arr1 elements such that no arr2 element within d.
**Approach:** Sort arr2. For each arr1 element, binary search nearest in arr2, check distance.

### 15. Check If N and Its Double Exist
**Description:** Check if there exist i,j with arr[i] == 2*arr[j].
**Approach:** Sort. For each element, binary search for 2*x or x/2.

### 16. Relative Sort Array
**Description:** Sort arr1 by order defined in arr2. Elements not in arr2 go at end, sorted.
**Approach:** Count frequency of arr1. Traverse arr2, output count times. Append remaining sorted.

### 17. Sort Array By Parity
**Description:** Move all even elements before odd. Order among evens/odds does not matter.
**Approach:** Two pointers. Swap when left odd and right even.

### 18. Sort Array By Parity II
**Description:** Reorder so even indices have even values, odd indices have odd.
**Approach:** Two pointers for even and odd positions. Swap when both wrong.

### 19. Largest Perimeter Triangle
**Description:** From sides array, find largest perimeter of valid triangle.
**Approach:** Sort descending. For consecutive triple (a,b,c) with a<b+c, return a+b+c.

### 20. Height Checker
**Description:** Students in row by height. Return count of positions where expected != actual.
**Approach:** Sort copy, compare with original. Count mismatches.

### 21. Third Maximum Number
**Description:** Return third distinct maximum. If less than 3 distinct, return max.
**Approach:** Track three largest. Or sort, deduplicate, return third from end.

### 22. Find Target Indices After Sorting
**Description:** Sort array, return all indices where target appears.
**Approach:** Sort, linear scan for target. Or count elements < target and = target.

### 23. Count Negative Numbers in Sorted Matrix
**Description:** Grid sorted non-increasing row and column wise. Count negatives.
**Approach:** Start top-right. If negative, all below negative; count += rows-r; move left.

### 24. Kth Missing Positive Number
**Description:** Strictly increasing array. Find kth missing positive integer.
**Approach:** For each index i, missing count = arr[i] - i - 1. Binary search for first index where missing >= k.

### 25. Fair Candy Swap
**Description:** Two arrays (candy sizes). Swap one element each so both have equal total.
**Approach:** Compute sum difference. For each a in A, need b = a + diff/2 in B. Use set for B.

### 26. Find the Town Judge
**Description:** n people, trust pairs. Judge trusts nobody, everyone trusts judge.
**Approach:** Count in-degree and out-degree. Judge has in=n-1, out=0.

### 27. Contains Duplicate
**Description:** Check if array has duplicate.
**Approach:** Sort and adjacent compare. Or use set.

### 28. Missing Number
**Description:** Array of n distinct numbers from [0,n]. Find missing one.
**Approach:** Sum 0+1+...+n minus array sum. Or XOR all with 0..n.
