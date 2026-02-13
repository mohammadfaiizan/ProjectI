# Easy Heap and Priority Queue Problems

## 1. Kth Largest Element in an Array

**Description**: Find the kth largest element in an unsorted array.

**Approach**: Min-heap of size k. Push each element; when size exceeds k, pop. Root is kth largest. O(n log k).

---

## 2. Last Stone Weight

**Description**: Smash two largest stones; if unequal, push difference. Return last remaining weight.

**Approach**: Max-heap (negate for Python heapq). Pop two, push difference until one or none left.

---

## 3. Merge K Sorted Lists

**Description**: Merge k sorted linked lists into one sorted list.

**Approach**: Min-heap of (val, list_id, node). Pop min, advance that list, push next. O(n log k).

---

## 4. K Closest Points to Origin

**Description**: Return k points closest to (0,0).

**Approach**: Max-heap of size k by distance squared. Pop when size > k. O(n log k).

---

## 5. Top K Frequent Elements

**Description**: Return k most frequent elements.

**Approach**: Count frequencies. Min-heap of (freq, num) of size k. O(n log k).

---

## 6. Sort Characters by Frequency

**Description**: Sort string so more frequent chars come first.

**Approach**: Count, max-heap by freq, pop and append. O(n log n).

---

## 7. Third Maximum Number

**Description**: Return third distinct maximum; if fewer than three, return max.

**Approach**: Min-heap of size 3 for three largest, or use set + heap.

---

## 8. Relative Ranks

**Description**: Assign "Gold", "Silver", "Bronze" to top 3 scores; others get rank number.

**Approach**: Heap of (score, index). Pop in order, assign ranks.

---

## 9. Kth Largest in Stream

**Description**: Design class to add numbers and return kth largest.

**Approach**: Min-heap of size k. Same as kth largest in array but streaming.

---

## 10. Minimum Sum of Four Digit Number After Splitting

**Description**: Split 4-digit number into two 2-digit numbers to minimize sum.

**Approach**: Sort digits, form smallest two numbers. Heap not strictly needed but can use for general "min sum of k numbers from digits".

---

## 11. Maximum Product of Two Elements in an Array

**Description**: Find max of (nums[i]-1)*(nums[j]-1).

**Approach**: Two largest elements. Max-heap or single pass to track two max.

---

## 12. Largest Number After Digit Swaps by Parity

**Description**: Swap digits of same parity arbitrarily; find largest possible number.

**Approach**: Separate odd and even digits, sort descending, interleave. Heap for general "k largest" variant.

---

## 13. Make Array Zero by Subtracting Equal Amounts

**Description**: Each move subtract same positive number from all non-zero elements. Min moves to make all zero.

**Approach**: Count distinct non-zero values. Min-heap to process smallest first.

---

## 14. Maximum Units on a Truck

**Description**: Truck has box capacity. Each box type has (boxes, units per box). Maximize units.

**Approach**: Sort by units per box descending. Greedy take. Heap for "k largest units" variant.

---

## 15. Minimum Cost of Buying Candies With Discount

**Description**: Buy n candies, every 3rd is free (cheapest). Minimize cost.

**Approach**: Sort descending. Every 3rd is free. Max-heap to always pick two paid, one free.

---

## 16. Remove Stones to Minimize Total

**Description**: Apply operation k times: pick pile, remove floor(pile/2) stones. Minimize total.

**Approach**: Max-heap of piles. Pop, halve, push. Repeat k times.

---

## 17. Construct String With Repeat Limit

**Description**: Construct string with at most repeatLimit consecutive same chars, using char counts.

**Approach**: Max-heap by count. Pop two different chars alternately to avoid consecutive limit.

---

## 18. Maximum Score From Removing Stones

**Description**: Three piles. Each move remove 1 from two piles. Max moves.

**Approach**: Max-heap. Pop two, decrement, push back if > 0. Count moves.

---

## 19. Minimum Sum of Squared Difference

**Description**: Change nums1 to reduce sum of (nums1[i]-nums2[i])^2 with limited operations.

**Approach**: Diff array, max-heap by abs(diff). Greedy reduce largest diffs first.

---

## 20. Divide Array Into Equal Pairs

**Description**: Can array be partitioned into n pairs with equal sums?

**Approach**: Sort or count. Heap for "k smallest pairs" variant.

---

## 21. Maximum Bags With Full Capacity of Rocks

**Description**: Bags have capacity and current rocks. Add additional rocks to maximize full bags.

**Approach**: Min-heap of (capacity - current) for each bag. Fill smallest gaps first.

---

## 22. Total Cost to Hire K Workers

**Description**: Hire k workers from two ends of candidate list with cost.

**Approach**: Two min-heaps for left and right. Pop k times from cheaper side.

---

## 23. Minimum Amount of Time to Fill Cups

**Description**: Fill cups one or two at a time. Min seconds to fill all.

**Approach**: Max-heap of remaining amounts. Each second reduce one or two largest.

---

## 24. Apply Operations to an Array

**Description**: Double zeros, shift. Return resulting array.

**Approach**: Heap not primary; array simulation. Heap for "top k after operations" variant.

---

## 25. Design a Number Container System

**Description**: Support change index, find median or min index for a number.

**Approach**: Map number to indices. Heap for "smallest index" queries.
