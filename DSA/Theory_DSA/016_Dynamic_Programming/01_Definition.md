# Dynamic Programming: Definition and Fundamentals

## What is Dynamic Programming?

Dynamic Programming (DP) is an algorithmic paradigm that solves complex problems by breaking them into simpler overlapping subproblems and storing the results of subproblems to avoid redundant computation. The key insight is that optimal solutions to larger problems can be constructed from optimal solutions to smaller subproblems.

## Overlapping Subproblems

A problem exhibits overlapping subproblems when the same subproblem is solved multiple times during the computation. For example, in computing Fibonacci numbers, `fib(5)` requires `fib(4)` and `fib(3)`, and `fib(4)` requires `fib(3)` and `fib(2)`. The subproblem `fib(3)` is computed multiple times.

**Characteristics:**
- The recursive solution would recompute the same subproblems repeatedly
- Memoization or tabulation stores results to avoid redundant work
- Without overlapping subproblems, divide-and-conquer suffices (e.g., merge sort)

## Optimal Substructure

A problem has optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems. For example, in the shortest path problem, if the optimal path from A to C goes through B, then the segments A-to-B and B-to-C must be optimal paths themselves.

**Characteristics:**
- The optimal solution can be expressed in terms of optimal solutions to smaller instances
- This property allows us to build solutions bottom-up or top-down
- Greedy algorithms also require optimal substructure but with a stronger local-choice property

## Memoization vs Tabulation

### Memoization (Top-Down)

- Recursive approach with caching
- Compute only when needed (lazy evaluation)
- Natural translation from recursive formulation
- Call stack depth can be an issue for large inputs
- May compute only a subset of states (useful when not all states are reachable)

### Tabulation (Bottom-Up)

- Iterative approach with a table
- Fill table in dependency order (all subproblems before larger ones)
- No recursion overhead
- Always computes all states in the table
- Often allows space optimization (rolling arrays)

**When to use which:**
- Memoization: When state space is sparse or irregular
- Tabulation: When all states are needed and order is clear; better for space optimization

## State Definition

The state in DP represents the subproblem being solved. A well-defined state must:

1. **Uniquely identify** the subproblem
2. **Contain all information** needed to compute the answer
3. **Be minimal** (no redundant information)

**Examples:**
- Fibonacci: state = index `n`
- 0/1 Knapsack: state = `(index, remaining_capacity)`
- LCS: state = `(i, j)` for prefixes of two strings

## DP vs Greedy vs Divide-and-Conquer

| Aspect | DP | Greedy | Divide-and-Conquer |
|--------|-----|--------|---------------------|
| Subproblems | Overlapping | No overlap (typically) | Non-overlapping |
| Choice | Explore multiple choices, pick best | Make locally optimal choice | Combine subproblem results |
| Optimal substructure | Yes | Yes (with greedy choice property) | Yes |
| Storage | Memo/table for subproblems | Usually none | Recursion stack |
| Examples | Knapsack, LCS | Huffman, Dijkstra | Merge sort, binary search |

**Greedy:** Makes one choice per step; works when local optimality implies global optimality.
**DP:** Considers all choices; stores results to avoid recomputation.
**Divide-and-Conquer:** Splits into independent subproblems; no overlap.

## When to Use DP

### 1. Optimization Problems

- Maximize or minimize some value (profit, cost, length)
- Multiple choices at each step
- Need to compare all feasible choices

Examples: Knapsack, shortest path, max profit

### 2. Counting Problems

- Count number of ways to achieve a goal
- Often "how many" questions

Examples: Coin change II, distinct subsequences, unique paths

### 3. Feasibility Problems

- Determine if a solution exists
- Boolean outcome

Examples: Subset sum, word break, partition equal subset sum

### 4. Recognition Heuristics

Consider DP when you see:
- "Maximum/minimum" with choices
- "Number of ways"
- "Is it possible"
- Recursive structure with overlapping subproblems
- Decisions that affect future choices (e.g., take or skip)

## Summary

Dynamic Programming applies when a problem has overlapping subproblems and optimal substructure. Choose memoization for sparse or irregular state spaces, and tabulation when a clear fill order exists and space optimization is desired. State definition is critical: it must uniquely and minimally capture the subproblem. DP differs from greedy (which makes irrevocable local choices) and divide-and-conquer (which has non-overlapping subproblems). Use DP for optimization, counting, and feasibility problems where subproblems repeat.
