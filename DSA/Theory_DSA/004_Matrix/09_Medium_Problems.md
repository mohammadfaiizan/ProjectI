# Medium Matrix Problems

## 1. Set Matrix Zeroes

**Description**: If element is 0, set entire row and column to 0. O(1) space.

**Approach**: Use first row and first column as markers. Handle (0,0) overlap with separate flags for first row and first column.

---

## 2. Spiral Matrix

**Description**: Return elements in spiral order (top, right, bottom, left, repeat).

**Approach**: Layer-by-layer with four boundaries. Shrink boundaries after each side.

---

## 3. Spiral Matrix II

**Description**: Generate n x n matrix filled with 1 to n^2 in spiral order.

**Approach**: Same layer approach. Fill while moving boundaries.

---

## 4. Rotate Image

**Description**: Rotate n x n matrix 90 degrees clockwise in-place.

**Approach**: Transpose then reverse each row. Or rotate in 4-way swaps for each element in top-left quadrant.

---

## 5. Search a 2D Matrix

**Description**: Matrix sorted row-wise (each row's last <= next row's first). Search target.

**Approach**: Treat as 1D sorted array. Binary search with index mapping mid//n, mid%n.

---

## 6. Search a 2D Matrix II

**Description**: Each row sorted, each column sorted. Search target.

**Approach**: Staircase from top-right. If current > target, move left; if current < target, move down. O(m+n).

---

## 7. Game of Life

**Description**: Apply Conway's rules simultaneously. In-place with O(1) extra space.

**Approach**: Encode next state in second bit. 0b01=live->dead, 0b10=dead->live. Right shift after.

---

## 8. Unique Paths

**Description**: Robot at (0,0), move right or down to (m-1,n-1). Count paths.

**Approach**: DP. dp[i][j] = dp[i-1][j] + dp[i][j-1]. Space optimize to single row.

---

## 9. Unique Paths II

**Description**: Same with obstacles. 1 blocks, 0 allows.

**Approach**: DP with obstacle check. dp[i][j]=0 if obstacle else dp[i-1][j]+dp[i][j-1].

---

## 10. Minimum Path Sum

**Description**: Path from top-left to bottom-right minimizing sum.

**Approach**: DP. dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1]).

---

## 11. Triangle (Minimum Path Sum)

**Description**: Triangular grid. Move to adjacent in next row. Minimize path sum.

**Approach**: DP from bottom. dp[j] = triangle[i][j] + min(dp[j], dp[j+1]).

---

## 12. Maximal Square

**Description**: Binary matrix. Find largest square of 1s.

**Approach**: DP. dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) if matrix[i][j]==1.

---

## 13. Maximal Rectangle

**Description**: Binary matrix. Find largest rectangle of 1s.

**Approach**: For each row, build histogram of consecutive 1s from above. Max area in histogram (stack) for each row.

---

## 14. Number of Islands

**Description**: Count connected components of 1s (4-direction).

**Approach**: DFS/BFS. Mark visited by flipping to 0. Count DFS calls.

---

## 15. Surrounded Regions

**Description**: Flip O to X if surrounded. Border O's and connected stay.

**Approach**: DFS from all border O's, mark as temporary. Flip remaining O to X, restore temporary.

---

## 16. Pacific Atlantic Water Flow

**Description**: Cells that can flow to both Pacific (top/left) and Atlantic (bottom/right).

**Approach**: DFS from Pacific border and Atlantic border. Intersection of reachable sets.

---

## 17. Longest Increasing Path in a Matrix

**Description**: Longest strictly increasing path (any direction).

**Approach**: DFS with memoization. For each cell, try 4 directions, memo longest path from that cell.

---

## 18. Word Search

**Description**: Find word by moving adjacent. No cell reuse.

**Approach**: Backtracking. For each starting cell, DFS with visited set (or mark in-place).

---

## 19. Word Search II

**Description**: Find all words from dictionary on board.

**Approach**: Build Trie of words. For each cell, DFS with Trie. When reaching word end, add to result.

---

## 20. Kth Smallest Element in a Sorted Matrix

**Description**: Each row and column sorted. Find kth smallest.

**Approach**: Min-heap of first element per row. Pop k times, push next in row. Or binary search on value range.

---

## 21. 01 Matrix (Distance to Nearest Zero)

**Description**: Binary matrix. For each cell, find distance to nearest 0.

**Approach**: Multi-source BFS from all 0s. Or two passes (from top-left and bottom-right).

---

## 22. Shortest Path in Binary Matrix

**Description**: 0s passable, 1s blocked. 8-direction. Shortest path from (0,0) to (n-1,n-1).

**Approach**: BFS. Queue (r, c, dist). Mark visited. Return dist when reaching bottom-right.

---

## 23. Rotting Oranges

**Description**: 0=empty, 1=fresh, 2=rotten. Each minute rotten oranges rot adjacent. Minutes to rot all.

**Approach**: Multi-source BFS from rotten. Track minutes per level. Check if any fresh remains.

---

## 24. As Far from Land as Possible

**Description**: 0=water, 1=land. Find water cell with maximum distance to nearest land.

**Approach**: Multi-source BFS from all land. Max distance in BFS is answer.

---

## 25. Number of Closed Islands

**Description**: 0=land, 1=water. Count islands not touching border.

**Approach**: DFS from border 0s to mark as "not closed". Count remaining connected 0 components.

---

## 26. Count Sub Islands

**Description**: grid1 and grid2. Count islands in grid2 fully covered by 1s in grid1.

**Approach**: For each island in grid2, DFS and check all cells have grid1[i][j]==1. If yes, count.

---

## 27. Number of Enclaves

**Description**: Count 1s that cannot reach boundary.

**Approach**: DFS from all border 1s to mark reachable. Count unmarked 1s.

---

## 28. Shortest Bridge

**Description**: Two islands of 1s. Minimum 0s to flip to connect.

**Approach**: DFS to find and mark first island. BFS from first island to reach second. Return BFS distance.

---

## 29. Walls and Gates

**Description**: -1=wall, 0=gate, INF=room. Fill each room with distance to nearest gate.

**Approach**: Multi-source BFS from all gates.

---

## 30. Valid Sudoku

**Description**: Check if 9x9 partially filled Sudoku is valid (no duplicates in row, col, 3x3 box).

**Approach**: Use sets for rows, cols, boxes. box_id = (r//3)*3 + c//3.
