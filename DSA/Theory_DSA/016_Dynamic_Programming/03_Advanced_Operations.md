# Advanced DP Operations

## Space Optimization Techniques

### 1D from 2D

When `dp[i][j]` depends only on `dp[i-1][*]`, replace the 2D table with a 1D array. Iterate in reverse order for capacity-like dimensions to avoid overwriting values needed for the current row.

```python
def knapsack_1d(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

### Two Rows

When row `i` depends only on row `i-1`, use two arrays and swap.

### Two Variables

When the recurrence uses only the last few values, use a constant number of variables.

## DP on Strings

Use 2D DP when comparing or matching two strings. State is typically `(i, j)` for prefixes. Common patterns: LCS, edit distance, interleaving, distinct subsequences. Transitions often involve matching characters, inserting, deleting, or replacing.

## DP on Trees

Process trees with DFS in post-order: solve for children first, then combine at the parent. State often includes whether a node is selected or not (e.g., house robber III). Use recursion with memoization keyed by `(node, flag)`.

## DP with Bitmask

State includes a bitmask representing a subset (e.g., which cities visited in TSP). State: `dp[mask][i]` = optimal value when subset `mask` is processed and we end at position `i`. Transitions: extend the mask by flipping bits. Used for assignment, TSP, subset covering.

## Digit DP

Count numbers in range `[L, R]` satisfying digit constraints. State: `(position, tight, leading_zeros, ...)`. `tight` means prefix equals the upper bound prefix. Process digits left to right. Used for counting numbers with no repeated digits, no consecutive ones, etc.

## Interval DP

Subproblems are intervals `[i, j]`. Fill by length: length 1, then 2, then 3, etc. Used for matrix chain multiplication, burst balloons, optimal BST, palindrome partitioning. Recurrence: try all split points `k` in `[i, j]` and combine.

## Probability DP

State represents a probability or expected value. Transitions combine probabilities. Used for dice games, random walks, expected steps. May require solving linear equations for some formulations.

## DP with Data Structures

### Segment Tree

When the recurrence needs range queries or range updates over previous DP values. Example: `dp[i] = max(dp[j] + cost for j in range)` with efficient range max.

### Binary Indexed Tree (BIT)

Similar to segment tree for prefix/suffix queries. Used when `dp[i]` depends on `max/min/sum` of `dp[0..i-1]` with some condition.

## Convex Hull Trick (Overview)

Optimizes recurrences of the form `dp[i] = min(dp[j] + cost(j, i))` when `cost(j, i)` is linear in `i`. Maintains a convex hull of lines; query gives the minimum at a given x. Reduces complexity from O(n^2) to O(n log n).

## Divide and Conquer Optimization (Overview)

For recurrences `dp[i][j] = min over k of (dp[i-1][k] + cost(k, j))` where the optimal `k` for `dp[i][j]` is monotonic in `j`. Use divide and conquer to find optimal split points. Reduces from O(n^2 * k) to O(n * k * log n).

## Knuth's Optimization (Overview)

For recurrences like matrix chain multiplication where the cost function satisfies the quadrangle inequality. The optimal split point is monotonic. Used to reduce the search space for the split point from O(n) to O(1) per cell.

## Aliens Trick (Overview)

For problems that are easier when the objective has a linear penalty term. Binary search on the penalty; each iteration solves a simpler DP. Used when the original problem has complex constraints that simplify with a Lagrange multiplier.

## DP on DAG

Process nodes in topological order. State: `dp[v]` = optimal value to reach or from node `v`. Transitions follow edges. Used for longest path in DAG, critical path, scheduling.

## Profile DP (Overview)

Used for tiling and grid problems. State encodes the "profile" of the current row (which cells are filled, connectivity). Transition: try all valid ways to extend the profile. Complex but handles problems like domino tiling, Hamiltonian path in grid.
