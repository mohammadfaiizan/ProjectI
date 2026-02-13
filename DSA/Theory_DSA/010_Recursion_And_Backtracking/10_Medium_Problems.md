# Medium Recursion and Backtracking Problems

## Medium Problems

## 1. Subsets II

All subsets with duplicates. Sort, skip duplicate elements at same recursion level.

```python
def subsetsWithDup(nums):
    nums.sort()
    res = []
    def bt(i, path):
        res.append(path[:])
        for j in range(i, len(nums)):
            if j > i and nums[j] == nums[j-1]:
                continue
            path.append(nums[j])
            bt(j + 1, path)
            path.pop()
    bt(0, [])
    return res
```

Time: O(2^n) | Space: O(n)

---

## 2. Permutations

All permutations of array. Backtrack with used array, swap or append.

```python
def permute(nums):
    res = []
    def bt(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            bt(path, used)
            path.pop()
            used[i] = False
    bt([], [False] * len(nums))
    return res
```

Time: O(n!) | Space: O(n)

---

## 3. Permutations II

Permutations with duplicates. Use frequency map, iterate over unique elements.

```python
def permuteUnique(nums):
    from collections import Counter
    res = []
    def bt(path, cnt):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for x in cnt:
            if cnt[x] > 0:
                cnt[x] -= 1
                path.append(x)
                bt(path, cnt)
                path.pop()
                cnt[x] += 1
    bt([], Counter(nums))
    return res
```

Time: O(n!) | Space: O(n)

---

## 4. Combination Sum

Combinations summing to target, reuse allowed. Backtrack with same index allowed.

```python
def combinationSum(candidates, target):
    res = []
    def bt(i, path, rem):
        if rem == 0:
            res.append(path[:])
            return
        if rem < 0 or i >= len(candidates):
            return
        path.append(candidates[i])
        bt(i, path, rem - candidates[i])
        path.pop()
        bt(i + 1, path, rem)
    bt(0, [], target)
    return res
```

Time: O(2^target) | Space: O(target)

---

## 5. Combination Sum II

Each candidate once, no duplicate combinations. Sort, skip duplicates when not first at level.

```python
def combinationSum2(candidates, target):
    candidates.sort()
    res = []
    def bt(i, path, rem):
        if rem == 0:
            res.append(path[:])
            return
        if rem < 0 or i >= len(candidates):
            return
        for j in range(i, len(candidates)):
            if j > i and candidates[j] == candidates[j-1]:
                continue
            if candidates[j] > rem:
                break
            path.append(candidates[j])
            bt(j + 1, path, rem - candidates[j])
            path.pop()
    bt(0, [], target)
    return res
```

Time: O(2^n) | Space: O(n)

---

## 6. Combination Sum III

k numbers from 1-9 summing to n. Backtrack 1-9, track count and remainder.

```python
def combinationSum3(k, n):
    res = []
    def bt(start, path, rem):
        if len(path) == k and rem == 0:
            res.append(path[:])
            return
        if len(path) >= k or rem <= 0:
            return
        for x in range(start, 10):
            if x > rem:
                break
            path.append(x)
            bt(x + 1, path, rem - x)
            path.pop()
    bt(1, [], n)
    return res
```

Time: O(9 choose k) | Space: O(k)

---

## 7. Letter Combinations of a Phone Number

Map digits to letters. Backtrack digit by digit.

```python
def letterCombinations(digits):
    if not digits:
        return []
    m = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    res = []
    def bt(i, path):
        if i == len(digits):
            res.append(''.join(path))
            return
        for c in m[digits[i]]:
            path.append(c)
            bt(i + 1, path)
            path.pop()
    bt(0, [])
    return res
```

Time: O(4^n) | Space: O(n)

---

## 8. Generate Parentheses

All valid n pairs. Backtrack with open < n, close < open.

```python
def generateParenthesis(n):
    res = []
    def bt(s, o, c):
        if len(s) == 2 * n:
            res.append(s)
            return
        if o < n:
            bt(s + '(', o + 1, c)
        if c < o:
            bt(s + ')', o, c + 1)
    bt('', 0, 0)
    return res
```

Time: O(4^n / sqrt(n)) | Space: O(n)

---

## 9. Word Search

Find word in 2D grid. DFS from each cell, mark visited.

```python
def exist(board, word):
    m, n = len(board), len(board[0])
    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp, board[i][j] = board[i][j], '#'
        for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
            if dfs(i+di, j+dj, k+1):
                return True
        board[i][j] = tmp
        return False
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

Time: O(m * n * 4^len(word)) | Space: O(len(word))

---

## 10. Palindrome Partitioning

Partition string into palindromic substrings. At each cut, if prefix palindrome recurse on rest.

```python
def partition(s):
    res = []
    def bt(i, path):
        if i == len(s):
            res.append(path[:])
            return
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            if sub == sub[::-1]:
                path.append(sub)
                bt(j, path)
                path.pop()
    bt(0, [])
    return res
```

Time: O(n * 2^n) | Space: O(n)

---

## 11. Restore IP Addresses

Valid IP from string. Place 3 dots, validate each segment.

```python
def restoreIpAddresses(s):
    res = []
    def bt(i, path):
        if len(path) == 4 and i == len(s):
            res.append('.'.join(path))
            return
        if len(path) >= 4 or i >= len(s):
            return
        for j in range(1, 4):
            if i + j > len(s):
                break
            seg = s[i:i+j]
            if (seg[0] == '0' and len(seg) > 1) or int(seg) > 255:
                continue
            path.append(seg)
            bt(i + j, path)
            path.pop()
    bt(0, [])
    return res
```

Time: O(1) | Space: O(1)

---

## 12. N-Queens

Place n queens, no attacks. Backtrack row by row, track columns and diagonals.

```python
def solveNQueens(n):
    res = []
    col, diag1, diag2 = set(), set(), set()
    def bt(row, path):
        if row == n:
            res.append(['.' * c + 'Q' + '.' * (n - c - 1) for c in path])
            return
        for c in range(n):
            if c in col or row - c in diag1 or row + c in diag2:
                continue
            col.add(c)
            diag1.add(row - c)
            diag2.add(row + c)
            path.append(c)
            bt(row + 1, path)
            path.pop()
            col.discard(c)
            diag1.discard(row - c)
            diag2.discard(row + c)
    bt(0, [])
    return res
```

Time: O(n!) | Space: O(n)

---

## 13. N-Queens II

Count N-Queens solutions. Same as above, increment count instead of storing.

```python
def totalNQueens(n):
    count = 0
    col, diag1, diag2 = set(), set(), set()
    def bt(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for c in range(n):
            if c in col or row - c in diag1 or row + c in diag2:
                continue
            col.add(c)
            diag1.add(row - c)
            diag2.add(row + c)
            bt(row + 1)
            col.discard(c)
            diag1.discard(row - c)
            diag2.discard(row + c)
    bt(0)
    return count
```

Time: O(n!) | Space: O(n)

---

## 14. Sudoku Solver

Fill valid sudoku. Try digits 1-9 in empty cells, check row/col/box.

```python
def solveSudoku(board):
    def valid(r, c, d):
        for i in range(9):
            if board[r][i] == d or board[i][c] == d:
                return False
            if board[3*(r//3)+i//3][3*(c//3)+i%3] == d:
                return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for d in '123456789':
                        if valid(i, j, d):
                            board[i][j] = d
                            if solve():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    solve()
```

Time: O(9^m) | Space: O(1)

---

## 15. Partition to K Equal Sum Subsets

Split array into k equal-sum subsets. Backtrack to fill k buckets.

```python
def canPartitionKSubsets(nums, k):
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    nums.sort(reverse=True)
    buckets = [0] * k

    def bt(i):
        if i == len(nums):
            return True
        for j in range(k):
            if buckets[j] + nums[i] <= target:
                buckets[j] += nums[i]
                if bt(i + 1):
                    return True
                buckets[j] -= nums[i]
            if buckets[j] == 0:
                break
        return False
    return bt(0)
```

Time: O(k^n) | Space: O(n)

---

## 16. Matchsticks to Square

Form square with matchsticks. Four sides, backtrack stick assignment.

```python
def makesquare(matchsticks):
    total = sum(matchsticks)
    if total % 4:
        return False
    side = total // 4
    matchsticks.sort(reverse=True)
    sides = [0] * 4

    def bt(i):
        if i == len(matchsticks):
            return True
        for j in range(4):
            if sides[j] + matchsticks[i] <= side:
                sides[j] += matchsticks[i]
                if bt(i + 1):
                    return True
                sides[j] -= matchsticks[i]
            if sides[j] == 0:
                break
        return False
    return bt(0)
```

Time: O(4^n) | Space: O(n)

---

## 17. Fair Distribution of Cookies

Distribute to k children, minimize max. Assign each cookie to child, prune.

```python
def distributeCookies(cookies, k):
    n = len(cookies)
    kids = [0] * k
    res = float('inf')

    def bt(i):
        nonlocal res
        if i == n:
            res = min(res, max(kids))
            return
        for j in range(k):
            kids[j] += cookies[i]
            if kids[j] < res:
                bt(i + 1)
            kids[j] -= cookies[i]
    bt(0)
    return res
```

Time: O(k^n) | Space: O(k)

---

## 18. Beautiful Arrangement

Count permutations where pos divides val or val divides pos. Backtrack position, try unused numbers.

```python
def countArrangement(n):
    count = 0
    used = [False] * (n + 1)

    def bt(pos):
        nonlocal count
        if pos > n:
            count += 1
            return
        for val in range(1, n + 1):
            if not used[val] and (pos % val == 0 or val % pos == 0):
                used[val] = True
                bt(pos + 1)
                used[val] = False
    bt(1)
    return count
```

Time: O(k) k = valid permutations | Space: O(n)

---

## 19. Expression Add Operators

Insert +,-,* to reach target. Backtrack with current value and previous operand for multiplication.

```python
def addOperators(num, target):
    res = []
    def bt(i, path, val, prev):
        if i == len(num):
            if val == target:
                res.append(path)
            return
        for j in range(i + 1, len(num) + 1):
            s = num[i:j]
            if len(s) > 1 and s[0] == '0':
                break
            nxt = int(s)
            if i == 0:
                bt(j, s, nxt, nxt)
            else:
                bt(j, path + '+' + s, val + nxt, nxt)
                bt(j, path + '-' + s, val - nxt, -nxt)
                bt(j, path + '*' + s, val - prev + prev * nxt, prev * nxt)
    bt(0, '', 0, 0)
    return res
```

Time: O(4^n) | Space: O(n)

---

## 20. Different Ways to Add Parentheses

All results of adding parentheses to expression. D&C at each operator.

```python
def diffWaysToCompute(expression):
    if expression.isdigit():
        return [int(expression)]
    res = []
    for i, c in enumerate(expression):
        if c in '+-*':
            left = diffWaysToCompute(expression[:i])
            right = diffWaysToCompute(expression[i+1:])
            for a in left:
                for b in right:
                    res.append(a + b if c == '+' else a - b if c == '-' else a * b)
    return res
```

Time: O(Catalan) | Space: O(n)

---

## 21. Unique Paths III

Path visiting every empty cell. Count empty cells, DFS with visited set.

```python
def uniquePathsIII(grid):
    m, n = len(grid), len(grid[0])
    empty = 1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                sr, sc = i, j
            elif grid[i][j] == 0:
                empty += 1

    def dfs(r, c, rem):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] == -1:
            return 0
        if grid[r][c] == 2:
            return 1 if rem == 0 else 0
        grid[r][c] = -1
        total = sum(dfs(r+dr, c+dc, rem-1) for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)])
        grid[r][c] = 0
        return total
    return dfs(sr, sc, empty)
```

Time: O(4^(m*n)) | Space: O(m * n)

---

## 22. Word Search II

Find all dictionary words in grid. Trie + backtracking.

```python
def findWords(board, words):
    from collections import defaultdict
    Trie = lambda: defaultdict(Trie)
    root = Trie()
    for w in words:
        node = root
        for c in w:
            node = node[c]
        node['$'] = w

    m, n = len(board), len(board[0])
    res = []

    def dfs(r, c, node):
        ch = board[r][c]
        if ch not in node:
            return
        node = node[ch]
        if '$' in node:
            res.append(node['$'])
            del node['$']
        board[r][c] = '#'
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != '#':
                dfs(nr, nc, node)
        board[r][c] = ch

    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    return res
```

Time: O(m * n * 4^L) | Space: O(total word len)

---

## 23. All Paths From Source to Target

All paths in DAG from 0 to n-1. DFS backtrack.

```python
def allPathsSourceTarget(graph):
    n = len(graph)
    res = []
    def dfs(i, path):
        if i == n - 1:
            res.append(path[:])
            return
        for j in graph[i]:
            path.append(j)
            dfs(j, path)
            path.pop()
    dfs(0, [0])
    return res
```

Time: O(2^n) | Space: O(n)

---

## 24. Splitting String into Descending Consecutive Values

Split so each part is prev-1. Backtrack first segment length.

```python
def splitString(s):
    def bt(i, prev):
        if i == len(s):
            return True
        for j in range(i + 1, len(s) + 1):
            cur = int(s[i:j])
            if prev is None:
                if bt(j, cur):
                    return True
            elif cur == prev - 1:
                if bt(j, cur):
                    return True
        return False
    return bt(0, None)
```

Time: O(n^2) | Space: O(n)

---

## 25. Count Numbers with Unique Digits

Count numbers with all unique digits. Combinatorial or backtrack.

```python
def countNumbersWithUniqueDigits(n):
    if n == 0:
        return 1
    total = 10
    unique = 9
    for i in range(2, n + 1):
        unique *= (11 - i)
        total += unique
    return total
```

Time: O(n) | Space: O(1)

---

## Hard Problems

## 1. Word Search II (Hard)

Multiple words in grid. Trie + backtracking, remove found words from trie.

```python
def findWords(board, words):
    from collections import defaultdict
    Trie = lambda: defaultdict(Trie)
    root = Trie()
    for w in words:
        node = root
        for c in w:
            node = node[c]
        node['$'] = w

    m, n, res = len(board), len(board[0]), []
    def dfs(r, c, node):
        ch = board[r][c]
        if ch not in node:
            return
        node = node[ch]
        if '$' in node:
            res.append(node['$'])
            del node['$']
        board[r][c] = '#'
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != '#':
                dfs(nr, nc, node)
        board[r][c] = ch

    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    return res
```

Time: O(m * n * 4^L) | Space: O(total)

---

## 2. N-Queens (Hard)

Return all board configurations. Full backtracking with diagonal tracking.

```python
def solveNQueens(n):
    res = []
    col, diag1, diag2 = set(), set(), set()
    def bt(row, path):
        if row == n:
            res.append(['.' * c + 'Q' + '.' * (n - c - 1) for c in path])
            return
        for c in range(n):
            if c in col or row - c in diag1 or row + c in diag2:
                continue
            col.add(c)
            diag1.add(row - c)
            diag2.add(row + c)
            path.append(c)
            bt(row + 1, path)
            path.pop()
            col.discard(c)
            diag1.discard(row - c)
            diag2.discard(row + c)
    bt(0, [])
    return res
```

Time: O(n!) | Space: O(n)

---

## 3. Sudoku Solver (Hard)

Solve any valid sudoku. Backtrack with constraint propagation.

```python
def solveSudoku(board):
    def valid(r, c, d):
        for i in range(9):
            if board[r][i] == d or board[i][c] == d:
                return False
            if board[3*(r//3)+i//3][3*(c//3)+i%3] == d:
                return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for d in '123456789':
                        if valid(i, j, d):
                            board[i][j] = d
                            if solve():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    solve()
```

Time: O(9^m) | Space: O(1)

---

## 4. Expression Add Operators (Hard)

All expressions reaching target. Handle multiplication by tracking previous operand.

```python
def addOperators(num, target):
    res = []
    def bt(i, path, val, prev):
        if i == len(num):
            if val == target:
                res.append(path)
            return
        for j in range(i + 1, len(num) + 1):
            s = num[i:j]
            if len(s) > 1 and s[0] == '0':
                break
            nxt = int(s)
            if i == 0:
                bt(j, s, nxt, nxt)
            else:
                bt(j, path + '+' + s, val + nxt, nxt)
                bt(j, path + '-' + s, val - nxt, -nxt)
                bt(j, path + '*' + s, val - prev + prev * nxt, prev * nxt)
    bt(0, '', 0, 0)
    return res
```

Time: O(4^n) | Space: O(n)

---

## 5. Partition to K Equal Sum Subsets (Hard)

NP-complete. Backtrack with pruning (empty bucket optimization).

```python
def canPartitionKSubsets(nums, k):
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    nums.sort(reverse=True)
    buckets = [0] * k

    def bt(i):
        if i == len(nums):
            return True
        for j in range(k):
            if buckets[j] + nums[i] <= target:
                buckets[j] += nums[i]
                if bt(i + 1):
                    return True
                buckets[j] -= nums[i]
            if buckets[j] == 0:
                break
        return False
    return bt(0)
```

Time: O(k^n) | Space: O(n)

---

## 6. Matchsticks to Square (Hard)

Partition into 4 equal sides. Sort descending, backtrack to 4 buckets.

```python
def makesquare(matchsticks):
    total = sum(matchsticks)
    if total % 4:
        return False
    side = total // 4
    matchsticks.sort(reverse=True)
    sides = [0] * 4

    def bt(i):
        if i == len(matchsticks):
            return True
        for j in range(4):
            if sides[j] + matchsticks[i] <= side:
                sides[j] += matchsticks[i]
                if bt(i + 1):
                    return True
                sides[j] -= matchsticks[i]
            if sides[j] == 0:
                break
        return False
    return bt(0)
```

Time: O(4^n) | Space: O(n)

---

## 7. Unique Paths III (Hard)

Visit every cell exactly once. Count empties, DFS with backtrack.

```python
def uniquePathsIII(grid):
    m, n = len(grid), len(grid[0])
    empty = 1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                sr, sc = i, j
            elif grid[i][j] == 0:
                empty += 1

    def dfs(r, c, rem):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] == -1:
            return 0
        if grid[r][c] == 2:
            return 1 if rem == 0 else 0
        grid[r][c] = -1
        total = sum(dfs(r+dr, c+dc, rem-1) for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)])
        grid[r][c] = 0
        return total
    return dfs(sr, sc, empty)
```

Time: O(4^(m*n)) | Space: O(m * n)

---

## 8. Robot Room Cleaner

Clean unknown room. DFS with relative coordinates, backtrack and turn.

```python
def cleanRoom(robot):
    dirs = [(-1,0),(0,1),(1,0),(0,-1)]
    vis = set()

    def dfs(r, c, d):
        robot.clean()
        vis.add((r, c))
        for _ in range(4):
            nd = (d + _) % 4
            nr, nc = r + dirs[nd][0], c + dirs[nd][1]
            if (nr, nc) not in vis and robot.move():
                dfs(nr, nc, nd)
                robot.turnRight()
                robot.turnRight()
                robot.move()
                robot.turnRight()
                robot.turnRight()
            robot.turnRight()
    dfs(0, 0, 0)
```

Time: O(4^(m*n)) | Space: O(m * n)

---

## 9. Word Pattern II

Pattern matches string with bijection. Backtrack pattern char to substring mapping.

```python
def wordPatternMatch(pattern, s):
    def bt(pi, si, pmap, smap):
        if pi == len(pattern) and si == len(s):
            return True
        if pi >= len(pattern) or si >= len(s):
            return False
        p = pattern[pi]
        if p in pmap:
            w = pmap[p]
            if s[si:si+len(w)] != w:
                return False
            return bt(pi + 1, si + len(w), pmap, smap)
        for j in range(si + 1, len(s) + 1):
            w = s[si:j]
            if w in smap:
                continue
            pmap[p] = w
            smap[w] = p
            if bt(pi + 1, j, pmap, smap):
                return True
            del pmap[p]
            del smap[w]
        return False
    return bt(0, 0, {}, {})
```

Time: O(n^m) | Space: O(m + n)

---

## 10. Palindrome Partitioning II

Min cuts for all palindromic. DP preferred; backtrack for enumeration.

```python
def minCut(s):
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for i in range(n):
        pal[i][i] = True
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            pal[i][j] = (s[i] == s[j]) and (L == 2 or pal[i+1][j-1])
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = i
        for j in range(i):
            if pal[j][i-1]:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n] - 1
```

Time: O(n^2) | Space: O(n^2)

---

## 11. Count of Range Sum

Count subarrays in range. Merge sort with counting.

```python
def countRangeSum(nums, lower, upper):
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)

    def merge_count(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        count = merge_count(lo, mid) + merge_count(mid, hi)
        i = j = mid
        for k in range(lo, mid):
            while i < hi and pre[i] - pre[k] < lower:
                i += 1
            while j < hi and pre[j] - pre[k] <= upper:
                j += 1
            count += j - i
        pre[lo:hi] = sorted(pre[lo:hi])
        return count
    return merge_count(0, len(pre))
```

Time: O(n log n) | Space: O(n)

---

## 12. Different Ways to Add Parentheses (Hard)

All expression evaluations. D&C at operators.

```python
def diffWaysToCompute(expression):
    if expression.isdigit():
        return [int(expression)]
    res = []
    for i, c in enumerate(expression):
        if c in '+-*':
            left = diffWaysToCompute(expression[:i])
            right = diffWaysToCompute(expression[i+1:])
            for a in left:
                for b in right:
                    res.append(a + b if c == '+' else a - b if c == '-' else a * b)
    return res
```

Time: O(Catalan) | Space: O(n)

---

## 13. The Skyline Problem

Building silhouettes. Divide and conquer or sweep line.

```python
def getSkyline(buildings):
    from heapq import heappush, heappop
    events = []
    for L, R, H in buildings:
        events.append((L, -H, R))
        events.append((R, 0, 0))
    events.sort()
    res = []
    heap = [(0, float('inf'))]
    for x, neg_h, R in events:
        while heap[0][1] <= x:
            heappop(heap)
        if neg_h:
            heappush(heap, (neg_h, R))
        if not res or res[-1][1] != -heap[0][0]:
            res.append([x, -heap[0][0]])
    return res
```

Time: O(n log n) | Space: O(n)

---

## 14. Closest Pair of Points

O(n log n) closest pair. D&C with strip optimization.

```python
def closestPair(points):
    import math
    pts = sorted(points, key=lambda p: p[0])
    def dist(p, q):
        return math.hypot(p[0]-q[0], p[1]-q[1])

    def dc(pts):
        n = len(pts)
        if n <= 3:
            return min(dist(pts[i], pts[j]) for i in range(n) for j in range(i+1, n))
        mid = n // 2
        d = min(dc(pts[:mid]), dc(pts[mid:]))
        strip = [p for p in pts if abs(p[0] - pts[mid][0]) < d]
        strip.sort(key=lambda p: p[1])
        for i in range(len(strip)):
            for j in range(i+1, min(i+8, len(strip))):
                d = min(d, dist(strip[i], strip[j]))
        return d
    return dc(pts)
```

Time: O(n log n) | Space: O(n)

---

## 15. Median of Two Sorted Arrays

O(log(min(m,n))). Binary search on partition.

```python
def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = (m + n + 1) // 2 - i
        left1 = nums1[i-1] if i else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j-1] if j else float('-inf')
        right2 = nums2[j] if j < n else float('inf')
        if left1 <= right2 and left2 <= right1:
            if (m + n) % 2:
                return max(left1, left2)
            return (max(left1, left2) + min(right1, right2)) / 2
        if left1 > right2:
            hi = i - 1
        else:
            lo = i + 1
```

Time: O(log(min(m,n))) | Space: O(1)

---

## 16. Kth Largest Element

Quickselect. Partition, recurse on one half.

```python
def findKthLargest(nums, k):
    def partition(lo, hi):
        pivot = nums[hi]
        i = lo
        for j in range(lo, hi):
            if nums[j] >= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    lo, hi = 0, len(nums) - 1
    k = k - 1
    while True:
        p = partition(lo, hi)
        if p == k:
            return nums[p]
        if p < k:
            lo = p + 1
        else:
            hi = p - 1
```

Time: O(n) avg | Space: O(1)

---

## 17. Count Inversions

Pairs i<j with arr[i]>arr[j]. Modified merge sort.

```python
def countInversions(arr):
    def merge_count(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        count = merge_count(lo, mid) + merge_count(mid, hi)
        i, j = lo, mid
        merged = []
        while i < mid and j < hi:
            if arr[i] <= arr[j]:
                merged.append(arr[i])
                i += 1
            else:
                merged.append(arr[j])
                count += mid - i
                j += 1
        merged.extend(arr[i:mid])
        merged.extend(arr[j:hi])
        arr[lo:hi] = merged
        return count
    return merge_count(0, len(arr))
```

Time: O(n log n) | Space: O(n)

---

## 18. Count Smaller Numbers After Self

For each element count smaller to right. Merge sort with index tracking.

```python
def countSmaller(nums):
    res = [0] * len(nums)
    def merge(inds, lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        merge(inds, lo, mid)
        merge(inds, mid, hi)
        i, j = lo, mid
        tmp = []
        while i < mid and j < hi:
            if nums[inds[i]] <= nums[inds[j]]:
                res[inds[i]] += j - mid
                tmp.append(inds[i])
                i += 1
            else:
                tmp.append(inds[j])
                j += 1
        while i < mid:
            res[inds[i]] += j - mid
            tmp.append(inds[i])
            i += 1
        tmp.extend(inds[j:hi])
        inds[lo:hi] = tmp
    merge(list(range(len(nums))), 0, len(nums))
    return res
```

Time: O(n log n) | Space: O(n)
