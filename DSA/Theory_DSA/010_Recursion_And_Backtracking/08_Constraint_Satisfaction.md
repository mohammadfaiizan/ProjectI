# Constraint Satisfaction Problems

## N-Queens (Place N Queens No Attacks)

**Theory**: Place n queens on nxn board so no two attack. Queens attack same row, column, or diagonal. Backtrack: place queen per row, check column and diagonal constraints.

```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
```

## N-Queens II (Count Solutions)

**Theory**: Same as N-Queens but return count only.

```python
def total_n_queens(n):
    count = [0]
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        if row == n:
            count[0] += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count[0]
```

## Solve Sudoku

**Theory**: Fill 9x9 grid so each row, column, 3x3 box has 1-9. Backtrack: try each empty cell with valid digits.

```python
def solve_sudoku(board):
    def valid(row, col, num):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if board[i][j] == num:
                    return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if valid(i, j, num):
                            board[i][j] = num
                            if solve():
                                return True
                            board[i][j] = '.'
                    return False
        return True

    solve()
```

## Word Search in Grid

**Theory**: Find if word exists in 2D grid. Move adjacent (up, down, left, right). Each cell used once per path.

```python
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, index):
        if index == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[index]:
            return False
        temp = board[r][c]
        board[r][c] = '#'
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if backtrack(r + dr, c + dc, index + 1):
                board[r][c] = temp
                return True
        board[r][c] = temp
        return False

    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    return False
```

## Word Search II (Multiple Words - Trie + Backtracking)

**Theory**: Find all words from dictionary in grid. Build trie from words. For each cell, DFS with trie. When word found, add to result and continue (remove from trie to avoid duplicates).

```python
def find_words(board, words):
    from collections import defaultdict

    class TrieNode:
        def __init__(self):
            self.children = defaultdict(TrieNode)
            self.word = None

    root = TrieNode()
    for w in words:
        node = root
        for c in w:
            node = node.children[c]
        node.word = w

    result = []
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, node):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        char = board[r][c]
        if char not in node.children:
            return
        node = node.children[char]
        if node.word:
            result.append(node.word)
            node.word = None
        board[r][c] = '#'
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            backtrack(r + dr, c + dc, node)
        board[r][c] = char

    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, root)
    return result
```

## Crossword Puzzle Solver

**Theory**: Fill crossword grid with words. Words must fit in slots. Backtrack: try each word in each slot, check constraints.

```python
def solve_crossword(grid, words):
    rows, cols = len(grid), len(grid[0])
    slots = []

    def get_slots():
        for i in range(rows):
            j = 0
            while j < cols:
                if grid[i][j] == '-' or grid[i][j].isalpha():
                    start = j
                    while j < cols and (grid[i][j] == '-' or grid[i][j].isalpha()):
                        j += 1
                    if j - start > 1:
                        slots.append(('h', i, start, j - start))
                j += 1
        for j in range(cols):
            i = 0
            while i < rows:
                if grid[i][j] == '-' or grid[i][j].isalpha():
                    start = i
                    while i < rows and (grid[i][j] == '-' or grid[i][j].isalpha()):
                        i += 1
                    if i - start > 1:
                        slots.append(('v', start, j, i - start))
                i += 1

    get_slots()
    used = [False] * len(words)

    def fits(slot, word):
        if len(word) != slot[3]:
            return False
        t, r, c, length = slot
        for k in range(length):
            if t == 'h':
                cell = grid[r][c + k]
            else:
                cell = grid[r + k][c]
            if cell != '-' and cell != word[k]:
                return False
        return True

    def place(slot, word):
        t, r, c, length = slot
        old = []
        for k in range(length):
            if t == 'h':
                old.append(grid[r][c + k])
                grid[r][c + k] = word[k]
            else:
                old.append(grid[r + k][c])
                grid[r + k][c] = word[k]
        return old

    def unplace(slot, old):
        t, r, c, length = slot
        for k in range(length):
            if t == 'h':
                grid[r][c + k] = old[k]
            else:
                grid[r + k][c] = old[k]

    def backtrack(slot_idx):
        if slot_idx == len(slots):
            return True
        for i, word in enumerate(words):
            if used[i] or not fits(slots[slot_idx], word):
                continue
            used[i] = True
            old = place(slots[slot_idx], word)
            if backtrack(slot_idx + 1):
                return True
            unplace(slots[slot_idx], old)
            used[i] = False
        return False

    backtrack(0)
    return grid
```

## Rat in a Maze (All Paths)

**Theory**: Find all paths from (0,0) to (n-1,n-1). Move down or right. 1 = open, 0 = blocked.

```python
def find_path_maze(maze):
    rows, cols = len(maze), len(maze[0])
    result = []

    def backtrack(r, c, path):
        if r == rows - 1 and c == cols - 1:
            result.append(path[:])
            return
        if r < 0 or r >= rows or c < 0 or c >= cols or maze[r][c] == 0:
            return
        maze[r][c] = 0
        for dr, dc, d in [(1, 0, 'D'), (0, 1, 'R'), (-1, 0, 'U'), (0, -1, 'L')]:
            path.append(d)
            backtrack(r + dr, c + dc, path)
            path.pop()
        maze[r][c] = 1

    backtrack(0, 0, [])
    return result
```

## Knight's Tour

**Theory**: Knight visits every cell exactly once. 8 possible moves. Backtrack with move order.

```python
def knight_tour(n):
    board = [[-1] * n for _ in range(n)]
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

    def valid(r, c):
        return 0 <= r < n and 0 <= c < n and board[r][c] == -1

    def backtrack(r, c, count):
        board[r][c] = count
        if count == n * n - 1:
            return True
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if valid(nr, nc):
                if backtrack(nr, nc, count + 1):
                    return True
        board[r][c] = -1
        return False

    return backtrack(0, 0, 0)
```

## Graph Coloring (M-Coloring)

**Theory**: Color vertices with m colors so no adjacent vertices same color. Backtrack: assign color to each vertex.

```python
def graph_coloring(graph, m):
    n = len(graph)
    colors = [0] * n

    def valid(v, c):
        for u in graph[v]:
            if colors[u] == c:
                return False
        return True

    def backtrack(v):
        if v == n:
            return True
        for c in range(1, m + 1):
            if valid(v, c):
                colors[v] = c
                if backtrack(v + 1):
                    return True
                colors[v] = 0
        return False

    return backtrack(0)
```

## Hamiltonian Path

**Theory**: Path visiting every vertex exactly once. Backtrack: extend path by unvisited neighbors.

```python
def hamiltonian_path(graph, n):
    path = []
    visited = [False] * n

    def backtrack(v):
        path.append(v)
        if len(path) == n:
            return True
        visited[v] = True
        for u in graph[v]:
            if not visited[u]:
                if backtrack(u):
                    return True
        path.pop()
        visited[v] = False
        return False

    for start in range(n):
        if backtrack(start):
            return path
    return None
```

## Hamiltonian Cycle

**Theory**: Hamiltonian path that returns to start. Check if last vertex has edge to first.

```python
def hamiltonian_cycle(graph, n):
    path = []
    visited = [False] * n

    def backtrack(v, start):
        path.append(v)
        if len(path) == n:
            if start in graph[v]:
                return True
            path.pop()
            return False
        visited[v] = True
        for u in graph[v]:
            if not visited[u]:
                if backtrack(u, start):
                    return True
        path.pop()
        visited[v] = False
        return False

    for start in range(n):
        if backtrack(start, start):
            return path + [start]
    return None
```

## Cryptarithmetic Solver (SEND+MORE=MONEY)

**Theory**: Assign digits to letters. Each letter unique digit. Leading digits non-zero. SEND+MORE=MONEY.

```python
def solve_cryptarithmetic():
    letters = ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']
    used = [False] * 10
    mapping = {}

    def to_num(word):
        return sum(mapping[c] * (10 ** (len(word) - 1 - i)) for i, c in enumerate(word))

    def backtrack(idx):
        if idx == len(letters):
            if mapping['S'] != 0 and mapping['M'] != 0:
                send = to_num("SEND")
                more = to_num("MORE")
                money = to_num("MONEY")
                if send + more == money:
                    return True
            return False
        for d in range(10):
            if used[d]:
                continue
            if d == 0 and letters[idx] in ('S', 'M'):
                continue
            used[d] = True
            mapping[letters[idx]] = d
            if backtrack(idx + 1):
                return True
            used[d] = False
        return False

    backtrack(0)
    return mapping
```

## Tug of War

**Theory**: Divide array into two groups of equal size (or nearly) minimizing difference of sums.

```python
def tug_of_war(arr):
    n = len(arr)
    half = n // 2
    result = [float('inf'), None, None]

    def backtrack(i, g1, g2, sum1, sum2):
        if i == n:
            if abs(len(g1) - len(g2)) <= 1 and abs(sum1 - sum2) < result[0]:
                result[0] = abs(sum1 - sum2)
                result[1] = g1[:]
                result[2] = g2[:]
            return
        if len(g1) < half + 1:
            g1.append(arr[i])
            backtrack(i + 1, g1, g2, sum1 + arr[i], sum2)
            g1.pop()
        if len(g2) < half + 1:
            g2.append(arr[i])
            backtrack(i + 1, g1, g2, sum1, sum2 + arr[i])
            g2.pop()

    backtrack(0, [], [], 0, 0)
    return result[1], result[2]
```

## Expression Add Operators (Insert +,-,* to Reach Target)

**Theory**: Insert +, -, * between digits. No leading zeros. Evaluate to target.

```python
def add_operators(num, target):
    result = []

    def backtrack(index, path, value, prev):
        if index == len(num):
            if value == target:
                result.append(path)
            return
        for i in range(index + 1, len(num) + 1):
            s = num[index:i]
            if len(s) > 1 and s[0] == '0':
                break
            curr = int(s)
            if index == 0:
                backtrack(i, s, curr, curr)
            else:
                backtrack(i, path + '+' + s, value + curr, curr)
                backtrack(i, path + '-' + s, value - curr, -curr)
                backtrack(i, path + '*' + s, value - prev + prev * curr, prev * curr)

    backtrack(0, '', 0, 0)
    return result
```

## Unique Paths III (Visit Every Non-Obstacle)

**Theory**: Grid with start, end, obstacles. Count paths that visit every empty cell exactly once.

```python
def unique_paths_iii(grid):
    rows, cols = len(grid), len(grid[0])
    empty = 0
    start = None
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                empty += 1
            elif grid[i][j] == 1:
                start = (i, j)
    result = [0]
    seen = set()

    def backtrack(r, c, steps):
        if grid[r][c] == 2:
            if steps == empty + 1:
                result[0] += 1
            return
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
                key = (nr, nc)
                if key not in seen:
                    seen.add(key)
                    backtrack(nr, nc, steps + 1)
                    seen.remove(key)

    seen.add(start)
    backtrack(start[0], start[1], 0)
    return result[0]
```

## Robot Room Cleaner

**Theory**: Robot with unknown room layout. Clean all cells. API: move(), turnLeft(), turnRight(), clean(). Backtrack with DFS, explore all 4 directions.

```python
def clean_room(robot):
    def go_back():
        robot.turnRight()
        robot.turnRight()
        robot.move()
        robot.turnRight()
        robot.turnRight()

    visited = set()

    def backtrack(r, c, d):
        visited.add((r, c))
        robot.clean()
        for _ in range(4):
            if d == 0:
                nr, nc = r - 1, c
            elif d == 1:
                nr, nc = r, c + 1
            elif d == 2:
                nr, nc = r + 1, c
            else:
                nr, nc = r, c - 1
            if (nr, nc) not in visited and robot.move():
                backtrack(nr, nc, d)
                go_back()
            robot.turnRight()
            d = (d + 1) % 4

    backtrack(0, 0, 0)
```
