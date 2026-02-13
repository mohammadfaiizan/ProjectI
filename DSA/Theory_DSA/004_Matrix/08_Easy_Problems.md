# Easy Matrix Problems

## 1. Transpose Matrix

**Description**: Return transpose of matrix. Swap rows and columns.

**Approach**: Create new matrix of size n x m. result[j][i] = matrix[i][j].

---

## 2. Reshape the Matrix

**Description**: Reshape matrix from r x c to new_r x new_c. Fill row by row. Return original if impossible.

**Approach**: Flatten to 1D, then fill new matrix. Check r*c == new_r*new_c.

---

## 3. Flipping an Image

**Description**: Flip image horizontally, then invert (0 to 1, 1 to 0).

**Approach**: For each row, reverse then XOR each element with 1.

---

## 4. Toeplitz Matrix

**Description**: Check if matrix is Toeplitz (each diagonal has same elements).

**Approach**: For each (i,j) with i>0 and j>0, check matrix[i][j] == matrix[i-1][j-1].

---

## 5. Image Smoother

**Description**: Replace each pixel with average of 3x3 neighborhood (floor).

**Approach**: Create new matrix. For each cell, sum 9 neighbors (or fewer at edges), divide by count.

---

## 6. Flood Fill

**Description**: Replace connected region of same color from (sr,sc) with new color.

**Approach**: DFS or BFS from start. Only recurse when neighbor has same color as original.

---

## 7. Find the Town Judge

**Description**: In trust array [a,b] meaning a trusts b, find person trusted by all except themselves who trusts nobody. (Graph problem, not strictly matrix but similar structure.)

**Approach**: Count in-degree and out-degree. Judge has in-degree n-1, out-degree 0.

---

## 8. Matrix Diagonal Sum

**Description**: Sum elements on both diagonals of square matrix. Do not double-count center.

**Approach**: Sum matrix[i][i] and matrix[i][n-1-i]. If n odd, subtract center once.

---

## 9. Cells with Odd Values in a Matrix

**Description**: Start with zeros. Apply operations: increment all cells in row i, or in col j. Count cells with odd values.

**Approach**: Track which rows and cols are incremented odd times. Cell (i,j) odd iff (row_i XOR col_j) is 1.

---

## 10. Special Positions in a Binary Matrix

**Description**: Position (i,j) is special if matrix[i][j]==1 and all other elements in row i and col j are 0.

**Approach**: Precompute row sums and col sums. Special if matrix[i][j]==1 and row_sum[i]==1 and col_sum[j]==1.

---

## 11. Maximum Population Year

**Description**: Logs [birth, death]. Find year with maximum population. (Range/prefix sum, not 2D matrix but similar.)

**Approach**: Create array of size 101 (1950-2050). For each log, increment birth year, decrement death year. Prefix sum, find max.

---

## 12. Row With Maximum Ones

**Description**: Binary matrix. Find row index with maximum number of 1s.

**Approach**: Linear scan each row, count 1s. Or binary search for first 1 in each row (if sorted).

---

## 13. Richest Customer Wealth

**Description**: accounts[i][j] is money in bank j of customer i. Find max sum over all banks for any customer.

**Approach**: For each row, sum elements. Return max row sum.

---

## 14. Check if Every Row and Column Contains All Numbers

**Description**: n x n matrix. Check if each row and each column contains exactly 1 to n.

**Approach**: For each row and col, use set to check distinct values in range [1,n].

---

## 15. Minimum Time Visiting All Points

**Description**: Array of points. Find minimum time to visit all in order. Can move 1 unit horizontally, vertically, or diagonally per second.

**Approach**: For consecutive points (x1,y1) and (x2,y2), time = max(|x2-x1|, |y2-y1|). Sum over pairs.

---

## 16. Lucky Numbers in a Matrix

**Description**: Lucky number is minimum in its row and maximum in its column.

**Approach**: Precompute row mins and col maxs. Check each cell.

---

## 17. Count Negative Numbers in Sorted Matrix

**Description**: Each row and column sorted non-increasing. Count negatives.

**Approach**: Start top-right. For each row, find first negative. All to the right are negative. Move left as we go down.

---

## 18. Sort the Matrix Diagonally

**Description**: Sort each diagonal (top-left to bottom-right) of matrix.

**Approach**: Group elements by (i-j). Sort each group. Place back.

---

## 19. Find Winner on a Tic Tac Toe Game

**Description**: Moves array. Determine winner of 3x3 game (A, B, or pending/draw).

**Approach**: Build 3x3 board. Check rows, cols, diagonals for 3 in a row.

---

## 20. Available Captures for Rook

**Description**: Chess board. Find number of pawns rook can capture (first piece in each direction).

**Approach**: Find rook position. Scan 4 directions until piece or edge. Count pawns ('p').

---

## 21. Projection Area of 3D Shapes

**Description**: grid[i][j] = height of tower. Find total projection area (top + front + side).

**Approach**: Top: count non-zero cells. Front: max per column. Side: max per row. Sum all.

---

## 22. Shift 2D Grid

**Description**: Shift grid right by k positions (circular).

**Approach**: Flatten to 1D, rotate by k (reverse trick or modulo), reshape.

---

## 23. Delete Greatest Value in Each Row

**Description**: Repeatedly pick max from each row (delete after), add to score. Find max total score.

**Approach**: Sort each row. For each column, add max of that column across rows.

---

## 24. Equal Row and Column Pairs

**Description**: Count pairs (i,j) where row i equals column j.

**Approach**: Hash row tuples. For each column, count how many rows match.

---

## 25. Sum of Matrix After Queries

**Description**: n x n matrix of zeros. Queries: set row i to val, or set col j to val. Sum final matrix.

**Approach**: Process queries in reverse. Track which rows/cols are overwritten. Later queries override earlier. Sum = sum of (val * cells not overwritten by later ops).
