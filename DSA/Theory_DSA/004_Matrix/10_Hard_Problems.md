# Hard Matrix Problems

## 1. Dungeon Game

**Description**: Knight at (0,0), princess at (m-1,n-1). Each cell adds/subtracts health. Find minimum initial health so knight never drops below 1.

**Approach**: Reverse DP from bottom-right. dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j]).

---

## 2. Cherry Pickup

**Description**: Two paths from (0,0) to (n-1,n-1). Collect cherries. Same cell counted once. Maximize total.

**Approach**: DP by step s = i+j. State (r1, r2) for both paths at same step. dp[s][r1][r2] = max over 4 prev states + cherries.

---

## 3. Cherry Pickup II

**Description**: Two robots at (0,0) and (0,n-1). Move down each step, can move to adjacent column. Maximize cherries.

**Approach**: DP by row. dp[i][j1][j2] = max over 9 prev (j1-1,0,+1) x (j2-1,0,+1) + cherries (same cell counted once).

---

## 4. Sudoku Solver

**Description**: Solve 9x9 Sudoku. Fill empty cells ('.') with 1-9.

**Approach**: Backtracking. For each empty cell, try 1-9. Check row, col, 3x3 box. Recurse. Prune invalid.

---

## 5. N-Queens

**Description**: Place n queens on n x n board so no two attack. Return all distinct solutions.

**Approach**: Backtracking. Track cols, positive diagonal (r+c), negative diagonal (r-c). Place queen per row, recurse.

---

## 6. N-Queens II

**Description**: Count distinct solutions to N-Queens.

**Approach**: Same backtracking, count instead of storing boards.

---

## 7. Minimum Cost to Make at Least One Valid Path in a Grid

**Description**: Grid with arrows. Change arrow costs 1. Find min cost to go from (0,0) to (m-1,n-1).

**Approach**: 0-1 BFS. Following arrow = cost 0, changing = cost 1. Deque: appendleft for 0, append for 1.

---

## 8. Trapping Rain Water II

**Description**: 2D elevation map. Water trapped when surrounded by higher bars. Total trapped water.

**Approach**: Min-heap of boundary cells. Pop minimum, add water that can be trapped at that cell (bounded by min neighbor). Push new boundary.

---

## 9. Max Sum of Rectangle No Larger Than K

**Description**: 2D matrix. Find max sum of subrectangle with sum <= k.

**Approach**: Fix left and right columns. Compute row-wise prefix sum. For each (l,r), array of prefix sums. Find max subarray sum <= k using ordered set (binary search on prefix - k).

---

## 10. Number of Distinct Islands

**Description**: Count distinct island shapes (by relative positions).

**Approach**: DFS each island. Encode shape as sequence of directions (with backtrack marker). Hash shapes.

---

## 11. Minimum Number of Days to Disconnect Island

**Description**: 1=land, 0=water. Change min number of 1s to 0 so grid becomes disconnected (more than 1 island or 0 islands).

**Approach**: If already disconnected, 0. If articulation point exists (single 1 removal disconnects), 1. Else 2 (corner case: 2x2 full grid).

---

## 12. Critical Connections in a Network

**Description**: Graph as connections. Find bridges (edges whose removal increases connected components).

**Approach**: Tarjan's algorithm. DFS with discovery time and low link. Bridge if low[v] > disc[u].

---

## 13. Minimum Cost to Connect Two Groups of Points

**Description**: Two groups of points. Connect each point in group 1 to at least one in group 2. Minimize total cost.

**Approach**: DP with bitmask for group 2 coverage. dp[i][mask] = min cost to connect first i of group 1 and cover mask of group 2.

---

## 14. Maximum Points in an Archery Competition

**Description**: Two players, 12 arrows. Each section 0-11 has score. aliceArrows[i] arrows Alice used. Maximize Bob's score while beating Alice.

**Approach**: Bitmask or DP. For each section, Bob can use 0 to (total - alice[i]) arrows. Maximize sum of (arrows * section) where Bob wins.

---

## 15. Count Submatrices With All Ones

**Description**: Binary matrix. Count submatrices with all 1s.

**Approach**: For each cell (i,j), compute consecutive 1s to the left. For each row, use histogram approach: for each column, count rectangles ending at (i,j) with height from 1 to i.

---

## 16. Minimum Falling Path Sum II

**Description**: Matrix. From top row, move to any cell in next row except same column. Minimize path sum.

**Approach**: DP. For each (i,j), dp[i][j] = matrix[i][j] + min(dp[i-1][k] for k != j). Optimize: track min and second min of previous row.

---

## 17. Best Meeting Point

**Description**: 2D grid with 1s (people) and 0s (empty). Find point that minimizes total Manhattan distance from all people.

**Approach**: Optimal meeting point is median of x-coordinates and median of y-coordinates. Sort row and col indices of 1s, take median.

---

## 18. Smallest Rectangle Enclosing Black Pixels

**Description**: Binary image. Find area of smallest axis-aligned rectangle containing all black pixels. Given one black pixel.

**Approach**: BFS/DFS from given pixel to find all black pixels. Track min/max row and col.

---

## 19. Matrix Chain Multiplication

**Description**: Sequence of matrices with dimensions. Find minimum scalar multiplications to compute product.

**Approach**: DP. dp[i][j] = min cost to multiply matrices i through j. Try all split points k.

---

## 20. Minimum Swaps to Arrange a Binary Grid

**Description**: n x n grid. Each row has trailing 1s. Row i needs at least (n-1-i) trailing zeros. Find min adjacent row swaps.

**Approach**: Greedy. For each row from top, find nearest row with enough trailing zeros. Bubble it up, count swaps.
