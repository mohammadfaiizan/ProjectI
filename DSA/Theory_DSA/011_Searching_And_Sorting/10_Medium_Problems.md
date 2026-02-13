# Medium and Hard Searching and Sorting Problems

## Medium Problems

### 1. Search in Rotated Sorted Array
**Description:** Sorted array rotated at unknown pivot. Find target in O(log n).
**Approach:** Binary search. One half is always sorted. Check which half contains target.

### 2. Search in Rotated Sorted Array II
**Description:** Same as above with duplicates. Return boolean.
**Approach:** Handle arr[left]==arr[mid]==arr[right] by shrinking both ends.

### 3. Find Minimum in Rotated Sorted Array
**Description:** Rotated sorted array, find minimum element.
**Approach:** Binary search. Compare arr[mid] with arr[right]. Min in unsorted half.

### 4. Find First and Last Position of Element
**Description:** Sorted array with duplicates. Find [first, last] index of target.
**Approach:** Two binary searches: first occurrence and last occurrence.

### 5. Search a 2D Matrix
**Description:** 2D matrix sorted row-wise (each row > previous). Find target.
**Approach:** Flatten to 1D, binary search. row=mid//cols, col=mid%cols.

### 6. Search a 2D Matrix II
**Description:** Each row and column sorted. Find target.
**Approach:** Start top-right. If target > val, move down. If target < val, move left.

### 7. Koko Eating Bananas
**Description:** Piles, h hours. Min eating speed k to finish all.
**Approach:** Binary search on answer. Feasible(k): sum(ceil(pile/k)) <= h.

### 8. Capacity to Ship Packages Within D Days
**Description:** Weights, d days. Min capacity to ship all.
**Approach:** Binary search on capacity. Feasible: greedy load, count days.

### 9. Split Array Largest Sum
**Description:** Split into k subarrays. Minimize largest sum.
**Approach:** Binary search on max sum. Feasible: greedy split.

### 10. Find Peak Element
**Description:** Array with arr[i] != arr[i+1]. Find any peak index.
**Approach:** Binary search. If arr[mid] < arr[mid+1], peak in right half.

### 11. Find Right Interval
**Description:** Intervals. For each, find smallest start_j >= end_i.
**Approach:** Sort by start with original index. Binary search for each end_i.

### 12. Find K Closest Elements
**Description:** Sorted array, x, k. Return k closest elements to x.
**Approach:** Binary search for left boundary of optimal window of size k.

### 13. Single Element in Sorted Array
**Description:** Every element appears twice except one. Find it in O(log n).
**Approach:** Binary search on pair parity. Before single, pairs at (even, odd). After, (odd, even).

### 14. Successful Pairs of Spells and Potions
**Description:** spell[i]*potion[j] >= success. Count pairs per spell.
**Approach:** Sort potions. Binary search for min potion: ceil(success/spell).

### 15. Time Based Key-Value Store
**Description:** set(key, value, timestamp), get(key, timestamp) returns value with largest timestamp <= given.
**Approach:** Store list of (timestamp, value) per key. Binary search for get.

### 16. H-Index
**Description:** Citations array. Find h: h papers have >= h citations.
**Approach:** Sort descending. Binary search for largest i with citations[i] >= i+1.

### 17. Count of Smaller Numbers After Self
**Description:** For each element, count how many smaller elements to the right.
**Approach:** Merge sort with inversion count, or binary search insertion from right.

### 18. Sort List
**Description:** Sort linked list in O(n log n) time, O(1) space.
**Approach:** Merge sort on linked list. Find mid with slow/fast, merge two halves.

### 19. Sort Colors (Dutch National Flag)
**Description:** Array of 0, 1, 2. Sort in-place one pass.
**Approach:** Three pointers: low, mid, high. Swap based on arr[mid].

### 20. Top K Frequent Elements
**Description:** Return k most frequent elements.
**Approach:** Count frequency. Bucket sort by frequency or quickselect.

### 21. Kth Largest Element in Array
**Description:** Find kth largest element.
**Approach:** Quickselect (partition like quicksort). Or heap of size k.

### 22. Merge Intervals
**Description:** Overlapping intervals, merge all overlapping.
**Approach:** Sort by start. Iterate, merge if current overlaps with last in result.

### 23. Non-overlapping Intervals
**Description:** Remove minimum intervals to make non-overlapping.
**Approach:** Sort by end. Greedy: keep interval if start >= last_end.

### 24. Insert Interval
**Description:** Non-overlapping intervals sorted. Insert new interval, merge if needed.
**Approach:** Binary search for position. Merge overlapping. Or linear scan.

### 25. Minimum Number of Arrows to Burst Balloons
**Description:** Intervals (balloons). Find min arrows (points) to hit all.
**Approach:** Sort by end. Greedy: arrow at first end, skip all containing it.

---

## Hard Problems

### 1. Median of Two Sorted Arrays
**Description:** Find median of two sorted arrays in O(log(min(n,m))).
**Approach:** Binary search partition in smaller array. Partition larger so left half has (n+m+1)//2 elements. Check max_left <= min_right.

### 2. Find Minimum in Rotated Sorted Array II
**Description:** Rotated sorted with duplicates. Find minimum.
**Approach:** When arr[mid]==arr[right], right--. Cannot discard half.

### 3. Count of Range Sum
**Description:** Count ranges [i,j] where lower <= sum <= upper.
**Approach:** Prefix sums. Merge sort with counting. For each left prefix, count right prefixes in [lower-left, upper-left].

### 4. Reverse Pairs
**Description:** Count pairs (i,j) with i<j and arr[i] > 2*arr[j].
**Approach:** Merge sort. During merge, for each left element, count right elements < left/2.

### 5. Max Sum of Rectangle No Larger Than K
**Description:** 2D matrix. Find max sum subrectangle <= k.
**Approach:** Fix left and right columns, compute row sums. Find max subarray sum <= k using prefix and binary search.

### 6. Minimum Window Substring
**Description:** String s, t. Find min substring of s containing all chars of t.
**Approach:** Sliding window. Expand until valid, then contract. Track char counts.

### 7. Sliding Window Maximum
**Description:** Array, window size k. Max in each window.
**Approach:** Deque maintaining decreasing order. Front is max. Remove indices outside window.

### 8. Find Median from Data Stream
**Description:** Add numbers, return median.
**Approach:** Two heaps: max-heap for lower half, min-heap for upper half. Balance sizes.

### 9. Merge k Sorted Lists
**Description:** k linked lists, merge into one sorted list.
**Approach:** Min-heap of (val, list_idx, node). Extract min, add next from same list.

### 10. Kth Smallest in Sorted Matrix
**Description:** n*n matrix sorted row and column. Find kth smallest.
**Approach:** Binary search on value. Count elements <= mid. If count >= k, answer <= mid.

### 11. Split Array Largest Sum (Hard variant)
**Description:** Same as medium but with additional constraints.
**Approach:** Binary search on answer with feasibility check.

### 12. Minimum Cost to Hire K Workers
**Description:** Workers with quality and wage. Hire k such that ratio wage/quality is same. Minimize total cost.
**Approach:** Sort by ratio. For each worker as captain, take k-1 workers with smallest quality from those with lower ratio. Use heap.

### 13. Count of Smaller Numbers After Self (BIT/Fenwick)
**Description:** Same as medium but optimize with BIT.
**Approach:** Coordinate compression + Fenwick tree. Process from right, query prefix sum.

### 14. Russian Doll Envelopes
**Description:** Envelopes (w,h). Fit one inside another if both dimensions smaller. Max chain.
**Approach:** Sort by width asc, height desc. LIS on heights (binary search).

### 15. Minimum Number of Operations to Make Array Continuous
**Description:** Replace elements to make array contiguous [x, x+1, ..., x+n-1]. Min replacements.
**Approach:** Sort and deduplicate. For each unique value as start, binary search how many in range [start, start+n-1]. Max window = min operations.
