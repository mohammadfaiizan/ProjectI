# Easy Greedy Problems

## 1. Best Time to Buy and Sell Stock

**Description**: One buy and one sell; maximize profit.

**Approach**: Track min price seen; at each day, profit = price - min_so_far. Take max.

---

## 2. Assign Cookies

**Description**: Assign cookies to children; each child has greed factor, each cookie has size. Child satisfied if cookie size >= greed.

**Approach**: Sort both arrays; two pointers. Assign smallest cookie that satisfies each child.

---

## 3. Lemonade Change

**Description**: Customers pay 5, 10, or 20. Must give correct change. Return if possible.

**Approach**: Greedy change: for 10 give one 5; for 20 give one 10+5 or three 5s (prefer 10+5).

---

## 4. Valid Parentheses

**Description**: Check if string has matching brackets.

**Approach**: Stack; push open, pop on close and match. Greedy: match immediately.

---

## 5. Merge Sorted Array

**Description**: Merge two sorted arrays in place.

**Approach**: Two pointers from end; place larger element at end of result.

---

## 6. Majority Element

**Description**: Find element appearing more than n/2 times.

**Approach**: Boyer-Moore voting: cancel different pairs; survivor is majority.

---

## 7. Maximum Subarray

**Description**: Find contiguous subarray with maximum sum.

**Approach**: Kadane: at each position, either extend previous subarray or start new. Greedy: extend if sum stays positive.

---

## 8. Jump Game

**Description**: Can you reach last index? Each element is max jump from that position.

**Approach**: Track max reachable index; if current index exceeds it, return false.

---

## 9. Climbing Stairs

**Description**: n steps; climb 1 or 2 at a time. Count ways.

**Approach**: DP (Fibonacci). Not strictly greedy but simple recurrence.

---

## 10. Best Time to Buy and Sell Stock II

**Description**: Unlimited buys and sells; maximize profit.

**Approach**: Greedy: add profit whenever price[i] > price[i-1] (buy day before, sell today).

---

## 11. Is Subsequence

**Description**: Check if s is subsequence of t.

**Approach**: Two pointers; match chars of s in t left to right.

---

## 12. Can Place Flowers

**Description**: Flower bed with 0s and 1s; can't plant adjacent. Place n new flowers.

**Approach**: Greedy: plant at each valid 0 (both neighbors 0); count until n reached.

---

## 13. Minimum Cost to Move Chips

**Description**: Chips at positions; move 2 positions cost 0, 1 position cost 1. Min cost to stack all.

**Approach**: min(count_odd, count_even) - move all to one parity at 0 cost, then min group moves 1.

---

## 14. Maximum Units on a Truck

**Description**: Box types with (count, units per box). Truck holds limited boxes. Maximize units.

**Approach**: Sort by units per box descending; take boxes greedily.

---

## 15. Two City Scheduling

**Description**: 2n people; cost to send to city A or B. Send n to each. Minimize cost.

**Approach**: Sort by (costA - costB); first n to A, rest to B.

---

## 16. Split a String in Balanced Strings

**Description**: String of L and R. Split into balanced (equal L and R) substrings. Max count.

**Approach**: Count L and R; whenever they equal, that's a split. Greedy count.

---

## 17. Minimum Add to Make Parentheses Valid

**Description**: Add minimum parentheses to make string valid.

**Approach**: Count open; if close without open, need to add open. At end add remaining open.

---

## 18. Partition Labels

**Description**: Partition string so each letter appears in at most one part. Minimize number of parts.

**Approach**: Track last index of each char; extend partition until current index equals last of all chars in partition.

---

## 19. Score After Flipping Matrix

**Description**: Binary matrix; flip rows or columns. Maximize score (each row as binary number).

**Approach**: Greedy: ensure first column all 1s (flip rows); then flip columns where 0s > 1s.

---

## 20. DI String Match

**Description**: String of I and D. Permutation of 0..n where I means increase, D means decrease.

**Approach**: I: assign smallest unused; D: assign largest unused. Two pointers at ends.

---

## 21. Play with Chips

**Description**: Same as minimum cost to move chips.

**Approach**: min(odd_count, even_count).

---

## 22. Maximize Sum Of Array After K Negations

**Description**: Negate exactly k elements. Maximize sum.

**Approach**: Sort; negate negatives first. If k left, negate smallest absolute value repeatedly.

---

## 23. Last Stone Weight

**Description**: Repeatedly smash two largest stones (difference remains). Final stone weight.

**Approach**: Max-heap; pop two, push difference until one or zero left.

---

## 24. Array Partition

**Description**: Partition 2n numbers into n pairs. Maximize sum of min of each pair.

**Approach**: Sort; pair consecutive. Greedy: min of pair is maximized when we pair smallest with second smallest, etc. So sort and take every even index.

---

## 25. Monotonic Array

**Description**: Check if array is monotonic (non-decreasing or non-increasing).

**Approach**: One pass; track direction; reject if violates.

---

## 26. Largest Perimeter Triangle

**Description**: Form triangle from three array elements. Max perimeter.

**Approach**: Sort descending; check triplets (a, b, c) for a < b + c. First valid is max perimeter.

---

## 27. Distribute Candies

**Description**: 2n candies, n types. Sister gets n candies. Max distinct types she can get.

**Approach**: min(unique_count, n).

---

## 28. Non-decreasing Array

**Description**: Can you make array non-decreasing by changing at most one element?

**Approach**: Count inversions; if more than one, check if single fix works (e.g., lower prev or raise current).
