# Easy Array Problems

## 1. Two Sum

Find two indices such that arr[i] + arr[j] = target. Use hashmap: for each x, check if target - x exists.

## 2. Remove Duplicates from Sorted Array

In-place, return new length. Two pointers: write index for unique elements.

## 3. Remove Element

Remove all occurrences of val in-place. Two pointers.

## 4. Search Insert Position

Find index where target would be inserted in sorted array. Binary search.

## 5. Maximum Subarray (Kadane)

Find contiguous subarray with largest sum. Kadane's algorithm.

## 6. Plus One

Add 1 to number represented as array of digits. Handle carry from right.

## 7. Merge Sorted Array

Merge two sorted arrays. nums1 has extra space. Two pointers from end.

## 8. Pascal's Triangle

Generate first n rows. Each row: sum of adjacent elements from previous row.

## 9. Best Time to Buy and Sell Stock

One buy, one sell. Max profit = max price - min price before it. Track min so far.

## 10. Single Number

Every element appears twice except one. XOR all elements.

## 11. Majority Element

Element appearing more than n/2 times. Boyer-Moore voting.

## 12. Contains Duplicate

Check if array has duplicates. Use set.

## 13. Missing Number

Array [0,n] with one missing. Sum or XOR approach.

## 14. Move Zeroes

Move all zeros to end in-place. Two pointers.

## 15. Intersection of Two Arrays

Find common elements. Use set intersection.

## 16. Intersection of Two Arrays II

Find common elements with frequency. Use Counter, decrement on match.

## 17. Third Maximum Number

Find third distinct maximum. Track top three or use set and sort.

## 18. Find All Numbers Disappeared in Array

Values [1,n], some missing. Negative marking or cyclic sort.

## 19. Assign Cookies

Greedy: sort both, assign smallest cookie to smallest child that fits.

## 20. Island Perimeter

Count perimeter of island (1s). Each 1 contributes 4, subtract 2 per adjacent 1.

## 21. Max Consecutive Ones

Longest contiguous segment of 1s. Single pass, reset on 0.

## 22. Teemo Attacking

Merge overlapping time intervals. Track end of poison.

## 23. Next Greater Element I

For each element in nums1, find next greater in nums2. Monotonic stack on nums2.

## 24. Keyboard Row

Filter words that can be typed using one keyboard row. Set intersection.

## 25. Find Mode in Binary Search Tree

Find most frequent value. In-order traversal, track frequency.

## 26. Relative Ranks

Assign ranks to sorted scores. Use index mapping.

## 27. Array Partition I

Pair elements to maximize sum of mins. Sort, take every other element.

## 28. Reshape the Matrix

Reshape matrix to new dimensions. Row-major traversal.

## 29. Distribute Candies

Max distinct candy types for n/2 people. min(unique_count, n/2).

## 30. Longest Harmonious Subsequence

Find longest subsequence where max-min=1. Count frequency, check adjacent counts.
