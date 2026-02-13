# Medium Recursion and Backtracking Problems

## Medium Problems (20+)

1. **Subsets II** - All subsets with duplicates. Approach: Sort, skip duplicate elements at same recursion level.

2. **Permutations** - All permutations of array. Approach: Backtrack with used array, swap or append.

3. **Permutations II** - Permutations with duplicates. Approach: Use frequency map, iterate over unique elements.

4. **Combination Sum** - Combinations summing to target, reuse allowed. Approach: Backtrack with same index allowed.

5. **Combination Sum II** - Each candidate once, no duplicate combinations. Approach: Sort, skip duplicates when not first at level.

6. **Combination Sum III** - k numbers from 1-9 summing to n. Approach: Backtrack 1-9, track count and remainder.

7. **Letter Combinations of a Phone Number** - Map digits to letters. Approach: Backtrack digit by digit.

8. **Generate Parentheses** - All valid n pairs. Approach: Backtrack with open < n, close < open.

9. **Word Search** - Find word in 2D grid. Approach: DFS from each cell, mark visited.

10. **Palindrome Partitioning** - Partition string into palindromic substrings. Approach: At each cut, if prefix palindrome recurse on rest.

11. **Restore IP Addresses** - Valid IP from string. Approach: Place 3 dots, validate each segment.

12. **N-Queens** - Place n queens, no attacks. Approach: Backtrack row by row, track columns and diagonals.

13. **N-Queens II** - Count N-Queens solutions. Approach: Same as above, increment count instead of storing.

14. **Sudoku Solver** - Fill valid sudoku. Approach: Try digits 1-9 in empty cells, check row/col/box.

15. **Partition to K Equal Sum Subsets** - Split array into k equal-sum subsets. Approach: Backtrack to fill k buckets.

16. **Matchsticks to Square** - Form square with matchsticks. Approach: Four sides, backtrack stick assignment.

17. **Fair Distribution of Cookies** - Distribute to k children, minimize max. Approach: Assign each cookie to child, prune.

18. **Beautiful Arrangement** - Count permutations where pos divides val or val divides pos. Approach: Backtrack position, try unused numbers.

19. **Expression Add Operators** - Insert +,-,* to reach target. Approach: Backtrack with current value and previous operand for multiplication.

20. **Different Ways to Add Parentheses** - All results of adding parentheses to expression. Approach: D&C at each operator.

21. **Unique Paths III** - Path visiting every empty cell. Approach: Count empty cells, DFS with visited set.

22. **Word Search II** - Find all dictionary words in grid. Approach: Trie + backtracking.

23. **All Paths From Source to Target** - All paths in DAG from 0 to n-1. Approach: DFS backtrack.

24. **Splitting String into Descending Consecutive Values** - Split so each part is prev-1. Approach: Backtrack first segment length.

25. **Count Numbers with Unique Digits** - Count numbers with all unique digits. Approach: Combinatorial or backtrack.

---

## Hard Problems (15+)

1. **Word Search II** - Multiple words in grid. Approach: Trie + backtracking, remove found words from trie.

2. **N-Queens** - Return all board configurations. Approach: Full backtracking with diagonal tracking.

3. **Sudoku Solver** - Solve any valid sudoku. Approach: Backtrack with constraint propagation.

4. **Expression Add Operators** - All expressions reaching target. Approach: Handle multiplication by tracking previous operand.

5. **Partition to K Equal Sum Subsets** - NP-complete. Approach: Backtrack with pruning (empty bucket optimization).

6. **Matchsticks to Square** - Partition into 4 equal sides. Approach: Sort descending, backtrack to 4 buckets.

7. **Unique Paths III** - Visit every cell exactly once. Approach: Count empties, DFS with backtrack.

8. **Robot Room Cleaner** - Clean unknown room. Approach: DFS with relative coordinates, backtrack and turn.

9. **Word Pattern II** - Pattern matches string with bijection. Approach: Backtrack pattern char to substring mapping.

10. **Palindrome Partitioning II** - Min cuts for all palindromic. Approach: DP preferred; backtrack for enumeration.

11. **Count of Range Sum** - Count subarrays in range. Approach: Merge sort with counting.

12. **Different Ways to Add Parentheses** - All expression evaluations. Approach: D&C at operators.

13. **The Skyline Problem** - Building silhouettes. Approach: Divide and conquer or sweep line.

14. **Closest Pair of Points** - O(n log n) closest pair. Approach: D&C with strip optimization.

15. **Median of Two Sorted Arrays** - O(log(min(m,n))). Approach: Binary search on partition.

16. **Kth Largest Element** - Quickselect. Approach: Partition, recurse on one half.

17. **Count Inversions** - Pairs i<j with arr[i]>arr[j]. Approach: Modified merge sort.

18. **Count Smaller Numbers After Self** - For each element count smaller to right. Approach: Merge sort with index tracking.
