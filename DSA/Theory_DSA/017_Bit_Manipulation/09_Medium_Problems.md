# Medium Bit Manipulation Problems

## 1. Single Number II
**Description:** Every element appears three times except one. Find the unique element.
**Approach:** Bit counting: for each bit position, count mod 3; or use ones/twos state machine.

## 2. Single Number III
**Description:** Every element appears twice except two. Find both unique elements.
**Approach:** XOR all to get a^b; use rightmost set bit to partition into two groups; XOR each group.

## 3. Maximum XOR of Two Numbers in an Array
**Description:** Find maximum XOR of any pair in array.
**Approach:** Build binary trie; for each number traverse trie greedily choosing opposite bit when available.

## 4. Subsets
**Description:** Generate all subsets of array.
**Approach:** Iterate mask 0 to 2^n-1; include arr[i] if bit i set.

## 5. Subsets II
**Description:** Generate all subsets with duplicates (no duplicate subsets).
**Approach:** Sort first; bitmask with duplicate handling or backtracking.

## 6. Partition to K Equal Sum Subsets
**Description:** Can array be partitioned into k subsets with equal sum?
**Approach:** Bitmask DP; dp[mask] = (current subset sum, subsets used).

## 7. Matchsticks to Square
**Description:** Can matchsticks form a square?
**Approach:** Bitmask to try all partitions into 4 groups.

## 8. Maximum Product of Word Lengths
**Description:** Max len(word[i]) * len(word[j]) where words share no letters.
**Approach:** Bitmask per word; iterate pairs, check mask_i & mask_j == 0.

## 9. Gray Code
**Description:** Generate n-bit gray code sequence.
**Approach:** Gray code formula: i ^ (i >> 1) for i in 0..2^n-1.

## 10. Repeated DNA Sequences
**Description:** Find 10-char sequences that appear more than once.
**Approach:** Encode sequence as 2 bits per char (A=00,C=01,G=10,T=11); use rolling hash or bitmask.

## 11. Total Hamming Distance
**Description:** Sum of hamming distances between all pairs.
**Approach:** For each bit position, count ones; contribution = count * (n - count).

## 12. Find the Duplicate Number
**Description:** Array of n+1 numbers in [1,n]; exactly one duplicate.
**Approach:** Floyd cycle detection or XOR with 1..n (if space allows).

## 13. Decode XORed Permutation
**Description:** Reconstruct perm [1..n] from encoded where encoded[i] = perm[i] XOR perm[i+1].
**Approach:** XOR of 1..n known; encoded at odd indices gives perm[0]; reconstruct.

## 14. Minimum XOR Sum of Two Arrays
**Description:** Permute arr2 to minimize sum of (arr1[i] XOR arr2[perm[i]]).
**Approach:** Bitmask DP; dp[mask] = min XOR sum for first popcount(mask) elements of arr1.

## 15. Number of Valid Words for Each Puzzle
**Description:** For each puzzle, count words that are subsets of puzzle and contain first letter.
**Approach:** Bitmask words; for each puzzle enumerate submasks containing first letter.

## 16. Can I Win
**Description:** Two players pick 1..maxChoosable without replacement; first to reach desiredTotal wins.
**Approach:** Bitmask state (chosen numbers); memoized DFS.

## 17. Partition Equal Subset Sum
**Description:** Can array be partitioned into two subsets with equal sum?
**Approach:** Bitset DP or bitmask; check if sum/2 achievable.

## 18. Letter Tile Possibilities
**Description:** Count distinct sequences from tiles (with duplicates).
**Approach:** Bitmask for chosen positions; backtrack with duplicate handling.

## 19. Find Kth Bit in Nth Binary String
**Description:** Recursive binary string; find kth character.
**Approach:** Pattern: S(n) = S(n-1) + "1" + reverse(invert(S(n-1))).

## 20. Minimum Number of Operations to Make Array Continuous
**Description:** Replace elements to make array contiguous [x, x+1, ..., x+n-1].
**Approach:** Sort, sliding window; not primarily bit manipulation.

## 21. Count Pairs With Given XOR
**Description:** Count pairs (i,j) such that arr[i] XOR arr[j] == target.
**Approach:** For each a, count occurrences of a XOR target; use hash map.

## 22. Longest Nice Substring
**Description:** Longest substring where every letter has both upper and lower.
**Approach:** Bitmask for seen chars; divide and conquer on first "bad" char.

## 23. Find All Duplicates in an Array
**Description:** Array of n elements, each in [1,n]; elements appear once or twice.
**Approach:** Use index as flag (negate or add n); bit manipulation for in-place.

## 24. Bitwise AND of Numbers Range
**Description:** AND of all numbers in [left, right].
**Approach:** Find common prefix of left and right in binary; result is that prefix with rest zeros.

## 25. UTF-8 Validation
**Description:** Check if byte sequence is valid UTF-8.
**Approach:** Use bit masks to check leading bits of each byte.

---

# Hard Problems

## 1. Maximum XOR With an Element From Array
**Description:** Queries: max XOR of xi with any arr[j] where arr[j] <= mi.
**Approach:** Offline: sort queries by mi; trie with numbers <= mi; query max XOR.

## 2. Number of Ways to Wear Different Hats to Each Other
**Description:** n people, each has list of hats; assign distinct hat to each.
**Approach:** Bitmask DP on people; dp[mask][hat] = ways to assign hats to mask people using first hat types.

## 3. Minimum Cost to Connect Two Groups of Points
**Description:** Connect left group to right with minimum cost; each point must be connected.
**Approach:** Bitmask for right group coverage; DP over left group and mask.

## 4. Maximum Score Words Formed by Letters
**Description:** Choose words to maximize score; each letter has limited count.
**Approach:** Bitmask over words; for each subset check letter feasibility.

## 5. Number of Ways to Build Sturdy Brick Wall
**Description:** Build wall with bricks; avoid certain boundaries.
**Approach:** Bitmask for row patterns; DP with compatibility check.

## 6. Maximum AND Sum of Array
**Description:** Assign numbers to slots; maximize sum of (num AND slot_index).
**Approach:** Bitmask DP; assign numbers to slots represented by mask.

## 7. Maximum Compatibility Score Sum
**Description:** Assign students to mentors; maximize compatibility sum.
**Approach:** Bitmask DP; dp[mask] = max score assigning to first popcount(mask) students.

## 8. Minimum Cost to Cut a Stick
**Description:** Cut stick at given positions; cost = stick length.
**Approach:** Interval DP; can use bitmask for cut positions.

## 9. Number of Ways to Form a Target String
**Description:** Form target by picking one char per column from matrix.
**Approach:** DP with frequency precomputation; bitmask for column selection in variants.

## 10. Maximum Number of Achievable Transfer Requests
**Description:** Buildings have employees; transfer requests (from, to); maximize balanced requests.
**Approach:** Bitmask over requests; for each subset check if net flow is zero for all buildings.

## 11. Find Minimum Time to Finish All Jobs
**Description:** n jobs, k workers; minimize max time (each job to one worker).
**Approach:** Bitmask DP; dp[mask][k] = min max time for mask jobs with k workers.

## 12. Maximum Students Taking Exam
**Description:** Seating in grid; students cannot be adjacent; maximize count.
**Approach:** Bitmask DP per row; dp[row][mask] = max students for first row rows with row having mask.

## 13. Minimum Number of Work Sessions to Finish Tasks
**Description:** Tasks with time; sessions have limit; minimize sessions.
**Approach:** Bitmask DP; dp[mask] = min sessions for mask tasks.

## 14. Count Number of Maximum Bitwise-OR Subsets
**Description:** Count subsets that achieve maximum OR value.
**Approach:** Find max OR; DP counting subsets with each OR value.

## 15. Number of Wonderful Substrings
**Description:** Wonderful = at most one letter appears odd times.
**Approach:** Prefix XOR of parity bits; for each prefix count previous prefixes with same or 1-bit-different XOR.
