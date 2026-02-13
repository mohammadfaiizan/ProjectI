# Medium DP Problems

## 1. Longest Palindromic Substring

**Description:** Find longest palindromic substring.

**Approach:** Expand around center or 2D DP. `dp[i][j]` = is s[i:j+1] palindrome. Fill by length.

## 2. Longest Palindromic Subsequence

**Description:** Length of longest palindromic subsequence.

**Approach:** Interval DP. `dp[i][j] = 2 + dp[i+1][j-1]` if match, else `max(dp[i+1][j], dp[i][j-1])`.

## 3. Longest Common Subsequence

**Description:** Length of LCS of two strings.

**Approach:** 2D DP. `dp[i][j]` = LCS of s1[:i], s2[:j]. Match or skip.

## 4. Edit Distance (Levenshtein)

**Description:** Min insert/delete/replace to transform s1 to s2.

**Approach:** `dp[i][j]` = min edits for s1[:i], s2[:j]. Three transitions: insert, delete, replace.

## 5. Coin Change II

**Description:** Count ways to make amount with coins.

**Approach:** `dp[i] += dp[i-c]` for each coin. Order: iterate coins first to avoid permutation count.

## 6. Target Sum

**Description:** Assign + or - to each number to reach target.

**Approach:** Convert to subset sum. Find subsets with sum = (total + target) / 2.

## 7. Partition Equal Subset Sum

**Description:** Can array be partitioned into two equal-sum subsets?

**Approach:** Subset sum with target = total/2.

## 8. Word Break II

**Description:** Return all possible sentence formations.

**Approach:** Memoized DFS. For each prefix that is a word, recurse on suffix and combine.

## 9. Unique Paths II (Obstacles)

**Description:** Unique paths with obstacles.

**Approach:** Same as unique paths, skip obstacle cells.

## 10. Minimum Path Sum

**Description:** Min sum path in grid.

**Approach:** `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

## 11. House Robber II

**Description:** House robber with circular arrangement.

**Approach:** Run house robber on nums[1:] and nums[:-1], take max.

## 12. Maximum Product Subarray

**Description:** Max product of contiguous subarray.

**Approach:** Track max and min. Negatives can flip to max.

## 13. Longest Increasing Subsequence (O(n log n))

**Description:** LIS length with binary search.

**Approach:** Maintain tails array. For each x, binary search position, extend or replace.

## 14. Maximum Length of Repeated Subarray

**Description:** Max length subarray that appears in both arrays.

**Approach:** 2D DP. `dp[i][j] = 1 + dp[i-1][j-1]` if match. Track max.

## 15. Count Square Submatrices with All Ones

**Description:** Count all square submatrices of 1s.

**Approach:** `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` if grid[i][j]=1. Sum all dp.

## 16. Minimum Cost for Tickets

**Description:** Min cost to travel on given days with 1/7/30 day passes.

**Approach:** `dp[i]` = min cost for first i days. For each day, try 1/7/30 day pass.

## 17. Integer Break

**Description:** Break n into positive integers, maximize product.

**Approach:** `dp[i] = max(j * dp[i-j], j * (i-j))` for j in 1..i-1.

## 18. Combination Sum IV

**Description:** Count combinations that sum to target (order matters).

**Approach:** `dp[i] += dp[i-n]` for each n in nums. Unbounded, order matters.

## 19. Decode Ways II

**Description:** Decode with * wildcard (1-9).

**Approach:** Extend decode ways. Handle * for single and double digit.

## 20. Maximum Sum Circular Subarray

**Description:** Max sum subarray in circular array.

**Approach:** Either max subarray in linear array, or total - min subarray (wrap-around case).

## 21. Best Time to Buy and Sell Stock with Cooldown

**Description:** One day cooldown after sell.

**Approach:** State machine: sold, held, rest. `sold = held + price`, `held = max(held, rest - price)`, `rest = max(rest, sold)`.

## 22. Best Time to Buy and Sell Stock with Transaction Fee

**Description:** Fee per transaction.

**Approach:** `cash = max(cash, hold + price - fee)`, `hold = max(hold, cash - price)`.

## 23. Interleaving String

**Description:** Is s3 interleaving of s1 and s2?

**Approach:** `dp[i][j]` = can form s3[:i+j] from s1[:i], s2[:j]. Match s1[i-1] or s2[j-1] with s3[i+j-1].

## 24. Distinct Subsequences

**Description:** Count distinct subsequences of s equal to t.

**Approach:** `dp[i][j]` = count for s[:i], t[:j]. If s[i-1]==t[j-1], add dp[i-1][j-1]. Always add dp[i-1][j].

## 25. Palindrome Partitioning II

**Description:** Min cuts to partition string into palindromes.

**Approach:** Precompute is_pal[i][j]. `dp[i] = min(dp[j] + 1)` for j where s[j:i] is palindrome.

---

# Hard Problems

## 1. Edit Distance

**Description:** Min operations to transform one string to another.

**Approach:** Standard Levenshtein DP.

## 2. Longest Valid Parentheses

**Description:** Length of longest valid parentheses substring.

**Approach:** `dp[i]` = longest valid ending at i. Track matching open bracket.

## 3. Wildcard Matching

**Description:** Match string with * (any sequence) and ? (any char).

**Approach:** 2D DP. * can match empty or extend. Handle * carefully.

## 4. Regular Expression Matching

**Description:** Match with . and * (zero or more of preceding).

**Approach:** 2D DP. Handle * by matching zero or one+ of preceding char.

## 5. Minimum Insertion Steps to Make a String Palindrome

**Description:** Min insertions to make palindrome.

**Approach:** n - longest palindromic subsequence.

## 6. Burst Balloons

**Description:** Burst balloons to maximize coins. Coin = left * curr * right.

**Approach:** Interval DP. Add 1s at ends. `dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i]*nums[k]*nums[j])`.

## 7. Cherry Pickup II

**Description:** Two robots from top to bottom, collect cherries, no double count.

**Approach:** 3D DP. State: (row, col1, col2). Both move down, try all column pairs.

## 8. Maximum Profit in Job Scheduling

**Description:** Jobs with start, end, profit. Max profit non-overlapping.

**Approach:** Sort by end. `dp[i] = max(dp[i-1], profit[i] + dp[j])` where j is latest non-overlapping.

## 9. Minimum Cost to Merge Stones

**Description:** Merge k consecutive piles, min cost.

**Approach:** Interval DP. Merge only when (length-1) % (k-1) == 0.

## 10. Dungeon Game

**Description:** Min initial HP to reach princess in dungeon.

**Approach:** Reverse DP. `dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])`.

## 11. Count Different Palindromic Subsequences

**Description:** Count distinct palindromic subsequences.

**Approach:** Interval DP with character tracking. Handle duplicates carefully.

## 12. Minimum Cost to Cut a Stick

**Description:** Cut stick at given positions, cost = stick length. Min total cost.

**Approach:** Interval DP. Sort cuts. `dp[i][j] = cuts[j]-cuts[i] + min(dp[i][k] + dp[k][j])`.

## 13. Maximum Score from Performing Multiplication Operations

**Description:** Pick from ends of array, multiply by multiplier[i], max score.

**Approach:** `dp[i][j]` = max score with i picks from left, j from right. Or dp[left][right] = remaining operations.

## 14. Number of Ways to Stay in the Same Place

**Description:** Steps, arrLen. Ways to stay at 0 after steps.

**Approach:** `dp[step][pos]` = ways. Can move left, right, or stay. Space optimize to 1D.

## 15. Minimum Number of Refueling Stops

**Description:** Car with tank, stations with fuel. Min stops to reach target.

**Approach:** `dp[i]` = max distance with i stops. Greedy with heap or DP.
