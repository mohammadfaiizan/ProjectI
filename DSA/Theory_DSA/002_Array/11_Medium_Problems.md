# Medium Array Problems

## 1. Three Sum

Find all unique triplets with sum 0. Sort, fix one, two pointers for rest.

## 2. Container With Most Water

Two lines form container. Two pointers at ends, move smaller inward.

## 3. Product of Array Except Self

Output[i] = product of all except arr[i]. Prefix and suffix products.

## 4. Maximum Product Subarray

Contiguous subarray with max product. Track max and min (for negative).

## 5. Find Minimum in Rotated Sorted Array

Sorted array rotated. Binary search: compare mid with right to decide direction.

## 6. Search in Rotated Sorted Array

Binary search with rotation handling. Compare mid with left/right to find sorted half.

## 7. Find First and Last Position of Element

Binary search for leftmost and rightmost occurrence.

## 8. Combination Sum

Find all combinations that sum to target. Backtracking with pruning.

## 9. Combination Sum II

Same but each element used once, no duplicate combinations. Sort and skip duplicates.

## 10. Jump Game

Can you reach last index? Track max reachable, greedy.

## 11. Jump Game II

Minimum jumps to reach end. BFS or greedy: extend reach each step.

## 12. Merge Intervals

Merge overlapping intervals. Sort by start, merge if overlap.

## 13. Insert Interval

Insert new interval into sorted non-overlapping intervals. Find position, merge.

## 14. Spiral Matrix

Traverse matrix in spiral order. Layer by layer with boundaries.

## 15. Rotate Image

Rotate matrix 90 degrees in-place. Transpose then reverse rows.

## 16. Group Anagrams

Group strings by anagram. Use sorted string or char count as key.

## 17. Subarray Sum Equals K

Count subarrays with sum k. Prefix sum + hashmap.

## 18. Longest Substring Without Repeating Characters

Sliding window with hashset for seen chars.

## 19. Longest Palindromic Substring

Expand around center for each position. Odd and even length.

## 20. Next Permutation

In-place next lexicographic permutation. Find first decreasing from right, swap with next larger, reverse suffix.

## 21. Sort Colors (Dutch National Flag)

Three-way partition for 0, 1, 2.

## 22. Top K Frequent Elements

Bucket sort by frequency or heap. O(n) with bucket sort.

## 23. Kth Largest Element

Quickselect or heap. Partition around pivot.

## 24. Find Peak Element

Binary search: if mid < mid+1, peak in right half; else left.

## 25. Search a 2D Matrix

Sorted matrix. Binary search treating as 1D.

## 26. Set Matrix Zeroes

Set row and col to 0 if element is 0. Use first row/col as markers.

## 27. Spiral Matrix II

Generate n x n matrix with values 1 to n^2 in spiral order.

## 28. Unique Paths

Grid paths from top-left to bottom-right. DP or combinatorics.

## 29. Minimum Path Sum

Min sum path in grid. DP with min of up/left.

## 30. Rotate Array

Rotate right by k. Reversal algorithm or cyclic.
