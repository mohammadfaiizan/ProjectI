# Hard Array Problems

## 1. Trapping Rain Water

Water trapped between bars. Two pointers or stack. Track left_max and right_max.

## 2. First Missing Positive

Smallest positive integer not in array. Cyclic sort for values in [1,n].

## 3. Merge k Sorted Lists/Arrays

Merge k sorted arrays. Min-heap of (value, array_index, element_index).

## 4. Median of Two Sorted Arrays

Find median of two sorted arrays. Binary search on smaller array for partition.

## 5. Sliding Window Maximum

Max in each sliding window of size k. Monotonic deque.

## 6. Minimum Window Substring

Smallest substring of s containing all chars of t. Sliding window with frequency maps.

## 7. Substring with Concatenation of All Words

Find start indices where substring is concatenation of all words. Sliding window per word length.

## 8. Longest Consecutive Sequence

Longest consecutive integer sequence. Union-Find or set with expansion.

## 9. Two Sum - Data Structure Design

Add and find. Hashmap for values, on find check complement.

## 10. Max Points on a Line

Max collinear points. For each point, count slopes to others. Handle duplicates.

## 11. Candy

Distribute candy: adjacent ratings get different amounts. Two passes: left-to-right and right-to-left.

## 12. Product of Array Except Self (with division restriction)

No division. Prefix and suffix products in two passes.

## 13. Maximum Gap

Max difference between successive elements in sorted form. Bucket sort with n+1 buckets.

## 14. Create Maximum Number

Form k-digit number from two arrays. Greedy: for each split, take largest subsequence from each, merge.

## 15. Count of Smaller Numbers After Self

For each element, count smaller elements to the right. Merge sort with inversion count or BST.

## 16. Sliding Window Median

Median in each sliding window. Two heaps (max heap for lower half, min for upper) or multiset.

## 17. Shortest Subarray with Sum at Least K

Array may have negatives. Monotonic deque of prefix sums.

## 18. Subarray Sum Equals K (with negatives)

Prefix sum + hashmap. Same approach works with negatives.

## 19. Longest Increasing Path in Matrix

DFS with memoization. Explore from each cell, cache longest path from that cell.

## 20. Max Sum of Rectangle No Larger Than K

2D Kadane + TreeSet for subarray sum <= k. Iterate row ranges, compress to 1D.
