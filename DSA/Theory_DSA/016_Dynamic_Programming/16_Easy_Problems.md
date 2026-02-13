# Easy DP Problems

## 1. Climbing Stairs

**Description:** Count ways to reach the nth stair using 1 or 2 steps.

**Approach:** Linear recurrence. `dp[i] = dp[i-1] + dp[i-2]`. Space: O(1) with two variables.

## 2. Min Cost Climbing Stairs

**Description:** Given cost array, find minimum cost to reach top (start at index 0 or 1).

**Approach:** `dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])`. Return `dp[n]`.

## 3. House Robber

**Description:** Max money from non-adjacent houses.

**Approach:** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`. Two variables for O(1) space.

## 4. Pascal's Triangle

**Description:** Generate Pascal's triangle rows.

**Approach:** Each row: `row[j] = prev[j-1] + prev[j]` with 1s at boundaries.

## 5. Maximum Subarray (Kadane)

**Description:** Find contiguous subarray with largest sum.

**Approach:** `dp[i] = max(nums[i], dp[i-1] + nums[i])`. Track global max.

## 6. Best Time to Buy and Sell Stock

**Description:** One transaction, max profit.

**Approach:** Track min price seen, max profit = price - min_price.

## 7. Best Time to Buy and Sell Stock II

**Description:** Unlimited transactions.

**Approach:** Greedy - add all positive price differences.

## 8. Divisor Game

**Description:** Two players, subtract divisor of n. Can first player win?

**Approach:** `dp[i]` = first wins for n=i. Try all divisors; if any move leads to loss for opponent, win.

## 9. Range Sum Query - Immutable

**Description:** Answer sum queries for subarray.

**Approach:** Prefix sum array. Query [i,j] = prefix[j+1] - prefix[i].

## 10. Counting Bits

**Description:** For each i in [0,n], count 1-bits in binary representation.

**Approach:** `dp[i] = dp[i >> 1] + (i & 1)`.

## 11. Decode Ways

**Description:** Count ways to decode digit string to letters (1-26).

**Approach:** `dp[i]` = ways for prefix of length i. Check 1-digit and 2-digit valid decodings.

## 12. Unique Paths

**Description:** Count paths from top-left to bottom-right in m x n grid.

**Approach:** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`. First row/col = 1.

## 13. Unique Paths II

**Description:** Same with obstacles.

**Approach:** Skip obstacle cells. Initialize first row/col accounting for obstacles.

## 14. Minimum Path Sum

**Description:** Min sum path in grid from top-left to bottom-right.

**Approach:** `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

## 15. Triangle (Minimum Total)

**Description:** Min path sum from top to bottom of triangle.

**Approach:** Bottom-up. `dp[j] = triangle[i][j] + min(dp[j], dp[j+1])`.

## 16. Maximum Product Subarray

**Description:** Max product of contiguous subarray.

**Approach:** Track max and min (negative * negative = positive). `max_dp = max(nums[i], max_dp * nums[i], min_dp * nums[i])`.

## 17. Word Break

**Description:** Can string be segmented into dictionary words?

**Approach:** `dp[i]` = can segment prefix of length i. For each j < i, if dp[j] and s[j:i] in dict, dp[i] = True.

## 18. Coin Change

**Description:** Min coins to make amount.

**Approach:** `dp[i] = min(dp[i-c] + 1)` for each coin c. Initialize dp[0] = 0.

## 19. Longest Increasing Subsequence (O(n^2))

**Description:** Length of longest strictly increasing subsequence.

**Approach:** `dp[i] = 1 + max(dp[j])` for j < i where nums[j] < nums[i].

## 20. Is Subsequence

**Description:** Is s subsequence of t?

**Approach:** Two pointers or DP. `dp[i][j]` = is s[:i] subseq of t[:j].

## 21. Nth Tribonacci Number

**Description:** Tribonacci: T(n) = T(n-1) + T(n-2) + T(n-3).

**Approach:** Three variables, iterate.

## 22. Get Maximum in Generated Array

**Description:** Generate array with rules, return max.

**Approach:** Simulate generation, track max.

## 23. Delete and Earn

**Description:** Pick nums, delete num and num-1. Maximize sum.

**Approach:** Convert to points array (index = value, value = sum of that value). House robber on points.

## 24. Perfect Squares

**Description:** Min number of perfect squares that sum to n.

**Approach:** `dp[i] = min(dp[i - j*j] + 1)` for j*j <= i.

## 25. Maximum Sum of Non-Adjacent Elements (1D)

**Description:** Max sum picking non-adjacent elements.

**Approach:** Same as house robber.
