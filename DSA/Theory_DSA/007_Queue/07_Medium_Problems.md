# Medium Queue Problems

## 1. Binary Tree Zigzag Level Order Traversal

Return level-order traversal but alternate left-to-right and right-to-left per level. BFS with level tracking. Use a flag; when flag is false, reverse the level before appending to result.

```python
from collections import deque
def zigzagLevelOrder(root):
    if not root:
        return []
    q = deque([root])
    res = []
    left_to_right = True
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return res
```

Time: O(n) | Space: O(n)

---

## 2. Binary Tree Right Side View

Return the rightmost node value at each level. BFS. For each level, take the last node processed (rightmost).

```python
from collections import deque
def rightSideView(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        res.append(q[-1].val)
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return res
```

Time: O(n) | Space: O(n)

---

## 3. Binary Tree Left Side View

Return the leftmost node value at each level. BFS. For each level, take the first node (leftmost).

```python
from collections import deque
def leftSideView(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        res.append(q[0].val)
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return res
```

Time: O(n) | Space: O(n)

---

## 4. Populating Next Right Pointers in Each Node

Perfect binary tree. Populate each node's next pointer to point to its next right node on the same level. BFS level-order. For each level, link nodes from left to right. Last node in level has next = null.

```python
from collections import deque
def connect(root):
    if not root:
        return root
    q = deque([root])
    while q:
        for i in range(len(q) - 1):
            q[i].next = q[i + 1]
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return root
```

Time: O(n) | Space: O(n)

---

## 5. Populating Next Right Pointers in Each Node II

Same as above but tree may not be perfect (missing nodes). BFS. Track previous node in level; set previous.next = current. Handle levels with gaps.

```python
from collections import deque
def connect(root):
    if not root:
        return root
    q = deque([root])
    while q:
        prev = None
        for _ in range(len(q)):
            node = q.popleft()
            if prev:
                prev.next = node
            prev = node
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return root
```

Time: O(n) | Space: O(n)

---

## 6. Number of Islands

Grid of '1' and '0'. Count connected components of '1'. BFS (or DFS) from each unvisited '1'. Mark all reachable '1's as visited. Count number of BFS starts.

```python
from collections import deque
def numIslands(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                q = deque([(i, j)])
                grid[i][j] = '0'
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                            grid[nr][nc] = '0'
                            q.append((nr, nc))
    return count
```

Time: O(m*n) | Space: O(m*n)

---

## 7. Rotting Oranges

Grid with fresh (1) and rotten (2) oranges. Each minute, rotten rot adjacent fresh. Return minutes to rot all, or -1. Multi-source BFS from all rotten. Expand one minute at a time. Track fresh count; return -1 if any remain.

```python
from collections import deque
def orangesRotting(grid):
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    if fresh == 0:
        return 0
    mins = 0
    while q:
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    q.append((nr, nc))
        mins += 1
    return mins - 1 if fresh == 0 else -1
```

Time: O(m*n) | Space: O(m*n)

---

## 8. Shortest Path in Binary Matrix

NxN grid of 0s and 1s. Shortest path from (0,0) to (n-1,n-1) through 0s only. 8-direction. BFS from (0,0). Enqueue unvisited 0 neighbors. Return depth when (n-1,n-1) is reached.

```python
from collections import deque
def shortestPathBinaryMatrix(grid):
    n = len(grid)
    if grid[0][0] or grid[n-1][n-1]:
        return -1
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    while q:
        r, c, d = q.popleft()
        if r == n - 1 and c == n - 1:
            return d
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc, d + 1))
    return -1
```

Time: O(n^2) | Space: O(n^2)

---

## 9. Word Ladder

Transform beginWord to endWord by changing one letter at a time. Each intermediate must be in wordList. Return shortest sequence length. BFS. For each word, try all one-letter changes. Use set for O(1) lookup. Return depth when endWord found.

```python
from collections import deque
def ladderLength(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return 0
    q = deque([(beginWord, 1)])
    seen = {beginWord}
    while q:
        word, d = q.popleft()
        if word == endWord:
            return d
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                nw = word[:i] + c + word[i+1:]
                if nw in words and nw not in seen:
                    seen.add(nw)
                    q.append((nw, d + 1))
    return 0
```

Time: O(n * m^2) | Space: O(n)

---

## 10. Word Ladder II

Same as Word Ladder but return all shortest transformation sequences. BFS in layers. Build paths. When a word is found at a new layer, add all paths to it. Use layer-by-layer to avoid longer paths.

```python
from collections import deque, defaultdict
def findLadders(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return []
    layer = {beginWord: [[beginWord]]}
    while layer:
        new_layer = defaultdict(list)
        for word in layer:
            if word == endWord:
                return layer[word]
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = word[:i] + c + word[i+1:]
                    if nw in words:
                        for path in layer[word]:
                            new_layer[nw].append(path + [nw])
        words -= set(new_layer.keys())
        layer = new_layer
    return []
```

Time: O(n * m^2) | Space: O(n)

---

## 11. Open the Lock

4-digit lock. Start "0000", target given. Deadends forbidden. Each move rotates one wheel by 1. Return minimum moves. BFS with state as string. For each digit, try +1 and -1 (mod 10). Skip deadends and visited.

```python
from collections import deque
def openLock(deadends, target):
    dead = set(deadends)
    if '0000' in dead:
        return -1
    q = deque([('0000', 0)])
    seen = {'0000'}
    while q:
        state, moves = q.popleft()
        if state == target:
            return moves
        for i in range(4):
            for d in (1, -1):
                nd = (int(state[i]) + d) % 10
                nstate = state[:i] + str(nd) + state[i+1:]
                if nstate not in dead and nstate not in seen:
                    seen.add(nstate)
                    q.append((nstate, moves + 1))
    return -1
```

Time: O(10^4) | Space: O(10^4)

---

## 12. Snakes and Ladders

Board game. Boustrophedon numbering. Some cells have snakes/ladders. Minimum dice rolls from 1 to N*N. BFS. State = board position. From each position, try all 6 dice outcomes. Apply snake/ladder if destination has one.

```python
from collections import deque
def snakesAndLadders(board):
    n = len(board)
    def to_pos(sq):
        r = (sq - 1) // n
        c = (sq - 1) % n
        if r % 2:
            c = n - 1 - c
        r = n - 1 - r
        return r, c

    q = deque([(1, 0)])
    seen = {1}
    while q:
        sq, moves = q.popleft()
        if sq == n * n:
            return moves
        for d in range(1, 7):
            nsq = min(sq + d, n * n)
            r, c = to_pos(nsq)
            if board[r][c] != -1:
                nsq = board[r][c]
            if nsq not in seen:
                seen.add(nsq)
                q.append((nsq, moves + 1))
    return -1
```

Time: O(n^2) | Space: O(n^2)

---

## 13. 01 Matrix (Multidirectional BFS)

Matrix of 0s and 1s. For each cell, return distance to nearest 0. Multi-source BFS from all 0s. Propagate distance to neighbors.

```python
from collections import deque
def updateMatrix(mat):
    m, n = len(mat), len(mat[0])
    q = deque()
    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                q.append((i, j))
            else:
                mat[i][j] = -1
    while q:
        r, c = q.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and mat[nr][nc] == -1:
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
    return mat
```

Time: O(m*n) | Space: O(m*n)

---

## 14. As Far from Land as Possible

NxN grid of land (1) and water (0). Find water cell with maximum distance to nearest land. Multi-source BFS from all land. Last water cell to be reached has max distance.

```python
from collections import deque
def maxDistance(grid):
    m, n = len(grid), len(grid[0])
    q = deque()
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                q.append((i, j))
    if len(q) == 0 or len(q) == m * n:
        return -1
    dist = 0
    while q:
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    q.append((nr, nc))
        dist += 1
    return dist - 1
```

Time: O(m*n) | Space: O(m*n)

---

## 15. Sliding Window Maximum

For each window of size k, return the maximum element. Monotonic decreasing deque. Store indices. Remove from back when new element is larger. Front is current max.

```python
from collections import deque
def maxSlidingWindow(nums, k):
    dq = deque()
    res = []
    for i in range(len(nums)):
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

Time: O(n) | Space: O(k)

---

## 16. Design Task Scheduler

Schedule tasks with cooldown n between same tasks. Minimize total time. Max-heap for task counts. Queue for (task, next_available_time). Each slot: if heap has task, pop and schedule; push to queue.

```python
import heapq
from collections import deque, Counter
def leastInterval(tasks, n):
    cnt = Counter(tasks)
    heap = [-c for c in cnt.values()]
    heapq.heapify(heap)
    q = deque()
    time = 0
    while heap or q:
        time += 1
        if heap:
            c = heapq.heappop(heap) + 1
            if c:
                q.append((c, time + n))
        if q and q[0][1] == time:
            heapq.heappush(heap, q.popleft()[0])
    return time
```

Time: O(n) | Space: O(n)

---

## 17. LRU Cache

Design LRU cache with get and put. Evict least recently used when full. OrderedDict (move_to_end on access) or HashMap + deque of keys. On get: move to end. On put: add/update, evict from front if full.

```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

Time: O(1) | Space: O(capacity)

---

## 18. First Unique Character in Data Stream

Stream of integers. Support add() and showFirstUnique(). Return first unique in stream. Queue for candidates. HashMap for frequency. On add: increment freq, enqueue. Remove from front while freq[front] > 1.

```python
from collections import deque
class FirstUnique:
    def __init__(self, nums):
        self.q = deque()
        self.freq = {}
        for x in nums:
            self.add(x)

    def add(self, value):
        self.freq[value] = self.freq.get(value, 0) + 1
        self.q.append(value)
        while self.q and self.freq[self.q[0]] > 1:
            self.q.popleft()

    def showFirstUnique(self):
        return self.q[0] if self.q else -1
```

Time: O(1) amortized | Space: O(n)

---

## 19. Interleave First and Second Half of Queue

Queue [1,2,3,4,5,6] becomes [1,4,2,5,3,6]. Use only queue and stack. Push first half to stack. Enqueue stack to queue. Rotate. Push new first half to stack. Interleave: pop from stack, dequeue from queue, alternate.

```python
from collections import deque
def interleaveQueue(q):
    n = len(q)
    st = []
    for _ in range(n // 2):
        st.append(q.popleft())
    while st:
        q.append(st.pop())
    for _ in range(n // 2):
        q.append(q.popleft())
    for _ in range(n // 2):
        st.append(q.popleft())
    while st:
        q.append(st.pop())
        q.append(q.popleft())
    return q
```

Time: O(n) | Space: O(n)

---

## 20. Sort a Queue

Sort a queue using only queue operations (and possibly one extra queue). Repeatedly find minimum by rotating through queue, remove it, append to result queue. Repeat until empty.

```python
from collections import deque
def sortQueue(q):
    res = deque()
    while q:
        mn = min(q)
        for _ in range(len(q)):
            x = q.popleft()
            if x == mn:
                res.append(x)
                break
            q.append(x)
    return res
```

Time: O(n^2) | Space: O(n)

---

## 21. Jump Game III (BFS)

From start index, can jump to start+arr[start] or start-arr[start]. Return true if can reach any index with value 0. BFS with visited set. Enqueue valid indices. Return true when we land on 0.

```python
from collections import deque
def canReach(arr, start):
    n = len(arr)
    q = deque([start])
    seen = {start}
    while q:
        i = q.popleft()
        if arr[i] == 0:
            return True
        for j in [i + arr[i], i - arr[i]]:
            if 0 <= j < n and j not in seen:
                seen.add(j)
                q.append(j)
    return False
```

Time: O(n) | Space: O(n)

---

## 22. Minimum Knight Moves

Minimum moves for knight from (0,0) to (x,y) on infinite board. BFS. Knight moves: 8 L-shaped positions. Use symmetry to limit search to one quadrant.

```python
from collections import deque
def minKnightMoves(x, y):
    x, y = abs(x), abs(y)
    moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    q = deque([(0, 0, 0)])
    seen = {(0, 0)}
    while q:
        r, c, d = q.popleft()
        if r == x and c == y:
            return d
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in seen and nr >= -2 and nc >= -2:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1
```

Time: O(|x|*|y|) | Space: O(|x|*|y|)

---

## 23. Shortest Path with Alternating Colors

Directed graph with red and blue edges. Shortest path from 0 to n-1 with alternating edge colors. BFS with state (node, last_color). From red edge we take blue; from blue we take red. Track (node, color) in visited.

```python
from collections import deque
def shortestAlternatingPaths(n, redEdges, blueEdges):
    red = [[] for _ in range(n)]
    blue = [[] for _ in range(n)]
    for a, b in redEdges:
        red[a].append(b)
    for a, b in blueEdges:
        blue[a].append(b)
    res = [-1] * n
    q = deque([(0, 0, None)])
    seen = {(0, None)}
    while q:
        node, d, color = q.popleft()
        if res[node] == -1:
            res[node] = d
        if color != 0:
            for v in red[node]:
                if (v, 0) not in seen:
                    seen.add((v, 0))
                    q.append((v, d + 1, 0))
        if color != 1:
            for v in blue[node]:
                if (v, 1) not in seen:
                    seen.add((v, 1))
                    q.append((v, d + 1, 1))
    return res
```

Time: O(n + e) | Space: O(n)

---

## 24. Nearest Exit from Entrance in Maze

Maze with '.' and '+'. Shortest path from entrance to any border (exit). BFS from entrance. Return depth when we reach a border cell (except entrance).

```python
from collections import deque
def nearestExit(maze, entrance):
    m, n = len(maze), len(maze[0])
    q = deque([(entrance[0], entrance[1], 0)])
    maze[entrance[0]][entrance[1]] = '+'
    while q:
        r, c, d = q.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and maze[nr][nc] == '.':
                if nr == 0 or nr == m-1 or nc == 0 or nc == n-1:
                    return d + 1
                maze[nr][nc] = '+'
                q.append((nr, nc, d + 1))
    return -1
```

Time: O(m*n) | Space: O(m*n)

---

## 25. Map of Highest Peak

Grid of land and water. Assign heights so adjacent differ by at most 1, water is 0. Maximize heights. Multi-source BFS from water cells. Assign distance to each land cell.

```python
from collections import deque
def highestPeak(isWater):
    m, n = len(isWater), len(isWater[0])
    res = [[-1] * n for _ in range(m)]
    q = deque()
    for i in range(m):
        for j in range(n):
            if isWater[i][j]:
                q.append((i, j, 0))
                res[i][j] = 0
    while q:
        r, c, h = q.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and res[nr][nc] == -1:
                res[nr][nc] = h + 1
                q.append((nr, nc, h + 1))
    return res
```

Time: O(m*n) | Space: O(m*n)

---

## 26. Bus Routes

Array of bus routes (each is list of stops). Minimum buses from source to target. BFS. Build stop_to_buses map. From each stop, try all buses. Track buses taken. Each bus ride = 1.

```python
from collections import deque
def numBusesToDestination(routes, source, target):
    if source == target:
        return 0
    stop_to_buses = {}
    for i, route in enumerate(routes):
        for stop in route:
            if stop not in stop_to_buses:
                stop_to_buses[stop] = []
            stop_to_buses[stop].append(i)
    q = deque([(source, 0)])
    seen_buses = set()
    seen_stops = {source}
    while q:
        stop, buses = q.popleft()
        if stop == target:
            return buses
        for bus in stop_to_buses[stop]:
            if bus not in seen_buses:
                seen_buses.add(bus)
                for s in routes[bus]:
                    if s not in seen_stops:
                        seen_stops.add(s)
                        q.append((s, buses + 1))
    return -1
```

Time: O(n * m) | Space: O(n * m)

---

## 27. Sliding Puzzle

2x3 board. Minimum moves to reach [[1,2,3],[4,5,0]]. BFS over board states (as string). Swap 0 with neighbors. Return depth when target reached.

```python
from collections import deque
def slidingPuzzle(board):
    target = '123450'
    start = ''.join(str(c) for row in board for c in row)
    if start == target:
        return 0
    neighbors = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
    q = deque([(start, 0)])
    seen = {start}
    while q:
        state, moves = q.popleft()
        i = state.index('0')
        for j in neighbors[i]:
            arr = list(state)
            arr[i], arr[j] = arr[j], arr[i]
            nstate = ''.join(arr)
            if nstate == target:
                return moves + 1
            if nstate not in seen:
                seen.add(nstate)
                q.append((nstate, moves + 1))
    return -1
```

Time: O(6!) | Space: O(6!)

---

## 28. Shortest Subarray with Sum at Least K

Array (may have negatives). Shortest subarray with sum >= k. Prefix sum + monotonic increasing deque. For each prefix[i], find smallest j with prefix[j] <= prefix[i] - k.

```python
from collections import deque
def shortestSubarray(nums, k):
    n = len(nums)
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    dq = deque()
    res = float('inf')
    for i in range(n + 1):
        while dq and prefix[i] - prefix[dq[0]] >= k:
            res = min(res, i - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[i]:
            dq.pop()
        dq.append(i)
    return res if res != float('inf') else -1
```

Time: O(n) | Space: O(n)

---

## 29. Constrained Subsequence Sum

Subsequence with no two elements within k indices. Maximize sum. DP with dp[i] = nums[i] + max(dp[i-k]..dp[i-1]). Monotonic deque for max over sliding window.

```python
from collections import deque
def constrainedSubsetSum(nums, k):
    dq = deque([0])
    dp = [0] * len(nums)
    for i in range(len(nums)):
        dp[i] = nums[i] + max(0, dp[dq[0]])
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
    return max(dp)
```

Time: O(n) | Space: O(n)

---

## 30. Longest Continuous Subarray with Absolute Diff <= Limit

Longest subarray where max - min <= limit. Two monotonic deques (max and min). Expand right; when max-min > limit, shrink left.

```python
from collections import deque
def longestSubarray(nums, limit):
    max_dq = deque()
    min_dq = deque()
    left = 0
    res = 0
    for right in range(len(nums)):
        while max_dq and nums[max_dq[-1]] < nums[right]:
            max_dq.pop()
        while min_dq and nums[min_dq[-1]] > nums[right]:
            min_dq.pop()
        max_dq.append(right)
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            if max_dq[0] == left:
                max_dq.popleft()
            if min_dq[0] == left:
                min_dq.popleft()
            left += 1
        res = max(res, right - left + 1)
    return res
```

Time: O(n) | Space: O(n)

---

# Hard Problems

## 1. Word Ladder II (Optimized)

All shortest transformation sequences from beginWord to endWord. Avoid TLE. Bidirectional BFS or single BFS with layer tracking. Build graph of parent pointers; DFS to reconstruct paths.

```python
from collections import deque, defaultdict
def findLadders(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return []
    layer = {beginWord: [[beginWord]]}
    while layer:
        new_layer = defaultdict(list)
        for word in layer:
            if word == endWord:
                return layer[word]
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = word[:i] + c + word[i+1:]
                    if nw in words:
                        for path in layer[word]:
                            new_layer[nw].append(path + [nw])
        words -= set(new_layer.keys())
        layer = new_layer
    return []
```

Time: O(n * m^2) | Space: O(n)

---

## 2. Sliding Window Maximum (Deque)

Same as medium; ensure O(n) solution. Monotonic decreasing deque. Each element pushed and popped at most once.

```python
from collections import deque
def maxSlidingWindow(nums, k):
    dq = deque()
    res = []
    for i in range(len(nums)):
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

Time: O(n) | Space: O(k)

---

## 3. Shortest Subarray with Sum at Least K

Array with negatives. Shortest subarray with sum >= k. O(n) required. Prefix sum + monotonic deque. Maintain increasing prefix indices; for each i, pop from front while valid, pop from back while prefix[back] >= prefix[i].

```python
from collections import deque
def shortestSubarray(nums, k):
    n = len(nums)
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    dq = deque()
    res = float('inf')
    for i in range(n + 1):
        while dq and prefix[i] - prefix[dq[0]] >= k:
            res = min(res, i - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[i]:
            dq.pop()
        dq.append(i)
    return res if res != float('inf') else -1
```

Time: O(n) | Space: O(n)

---

## 4. Constrained Subsequence Sum

Subsequence with no two elements within k indices. Maximize sum. O(n) required. DP + monotonic deque for max over sliding window of dp values.

```python
from collections import deque
def constrainedSubsetSum(nums, k):
    dq = deque([0])
    dp = [0] * len(nums)
    for i in range(len(nums)):
        dp[i] = nums[i] + max(0, dp[dq[0]])
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
    return max(dp)
```

Time: O(n) | Space: O(n)

---

## 5. Jump Game VI

From index 0, jump at most k steps. Maximize sum of landed indices. DP with dp[i] = nums[i] + max(dp[i-k]..dp[i-1]). Monotonic deque for max.

```python
from collections import deque
def maxResult(nums, k):
    dq = deque([0])
    for i in range(1, len(nums)):
        nums[i] += nums[dq[0]]
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
    return nums[-1]
```

Time: O(n) | Space: O(k)

---

## 6. Longest Continuous Subarray with Absolute Diff <= Limit

Longest subarray where max - min <= limit. O(n). Two deques (max, min). Sliding window. Shrink left when max - min > limit.

```python
from collections import deque
def longestSubarray(nums, limit):
    max_dq = deque()
    min_dq = deque()
    left = 0
    res = 0
    for right in range(len(nums)):
        while max_dq and nums[max_dq[-1]] < nums[right]:
            max_dq.pop()
        while min_dq and nums[min_dq[-1]] > nums[right]:
            min_dq.pop()
        max_dq.append(right)
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            if max_dq[0] == left:
                max_dq.popleft()
            if min_dq[0] == left:
                min_dq.popleft()
            left += 1
        res = max(res, right - left + 1)
    return res
```

Time: O(n) | Space: O(n)

---

## 7. Max Value of Equation

Points (x_i, y_i) sorted by x. Max of y_i + y_j + |x_i - x_j| for |x_i - x_j| <= k. Rewrite as (y_i - x_i) + (y_j + x_j). For each j, max over i in range of (y_i - x_i). Monotonic deque.

```python
from collections import deque
def findMaxValueOfEquation(points, k):
    dq = deque()
    res = float('-inf')
    for x, y in points:
        while dq and x - dq[0][1] > k:
            dq.popleft()
        if dq:
            res = max(res, dq[0][0] + x + y)
        while dq and dq[-1][0] <= y - x:
            dq.pop()
        dq.append((y - x, x))
    return res
```

Time: O(n) | Space: O(n)

---

## 8. Minimum Cost to Make at Least One Valid Path in a Grid

Grid with arrows. Change cost of 1 per cell. Minimum cost to reach bottom-right. 0-1 BFS. Moving in arrow direction costs 0; changing direction costs 1. Deque: push to front for 0, back for 1.

```python
from collections import deque
def minCost(grid):
    m, n = len(grid), len(grid[0])
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    q = deque([(0, 0, 0)])
    dist = {(0, 0): 0}
    while q:
        r, c, d = q.popleft()
        if r == m - 1 and c == n - 1:
            return d
        for i, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            cost = 0 if grid[r][c] == i + 1 else 1
            nd = d + cost
            if 0 <= nr < m and 0 <= nc < n and ((nr, nc) not in dist or dist[(nr, nc)] > nd):
                dist[(nr, nc)] = nd
                if cost == 0:
                    q.appendleft((nr, nc, nd))
                else:
                    q.append((nr, nc, nd))
    return -1
```

Time: O(m*n) | Space: O(m*n)

---

## 9. Shortest Path in a Grid with Obstacles Elimination

Grid with obstacles. Can eliminate at most k obstacles. Shortest path from (0,0) to (m-1,n-1). BFS with state (r, c, k_remaining). When hitting obstacle, decrement k if k > 0. Track visited as (r, c, k).

```python
from collections import deque
def shortestPath(grid, k):
    m, n = len(grid), len(grid[0])
    if k >= m + n - 2:
        return m + n - 2
    q = deque([(0, 0, k, 0)])
    seen = {(0, 0, k)}
    while q:
        r, c, k_rem, steps = q.popleft()
        if r == m - 1 and c == n - 1:
            return steps
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                nk = k_rem - grid[nr][nc]
                if nk >= 0 and (nr, nc, nk) not in seen:
                    seen.add((nr, nc, nk))
                    q.append((nr, nc, nk, steps + 1))
    return -1
```

Time: O(m*n*k) | Space: O(m*n*k)

---

## 10. Bus Routes (Hard variant)

Large number of routes and stops. Optimize for memory and time. BFS on buses. Build stop_to_buses. Only expand to unvisited buses. Track visited stops per bus.

```python
from collections import deque
def numBusesToDestination(routes, source, target):
    if source == target:
        return 0
    stop_to_buses = {}
    for i, route in enumerate(routes):
        for stop in route:
            if stop not in stop_to_buses:
                stop_to_buses[stop] = []
            stop_to_buses[stop].append(i)
    q = deque([(source, 0)])
    seen_buses = set()
    seen_stops = {source}
    while q:
        stop, buses = q.popleft()
        if stop == target:
            return buses
        for bus in stop_to_buses[stop]:
            if bus not in seen_buses:
                seen_buses.add(bus)
                for s in routes[bus]:
                    if s not in seen_stops:
                        seen_stops.add(s)
                        q.append((s, buses + 1))
    return -1
```

Time: O(n * m) | Space: O(n * m)

---

## 11. Sliding Puzzle (3x2 or 3x3)

3x2 or 3x3 sliding puzzle. Minimum moves. BFS over state space. State = flattened board string. Swap blank with neighbors. Use A* for 3x3 to reduce states.

```python
from collections import deque
def slidingPuzzle(board):
    m, n = len(board), len(board[0])
    target = tuple(range(1, m * n)) + (0,)
    start = []
    for row in board:
        for c in row:
            start.append(c)
    start = tuple(start)
    if start == target:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        state, moves = q.popleft()
        i = state.index(0)
        r, c = i // n, i % n
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                j = nr * n + nc
                arr = list(state)
                arr[i], arr[j] = arr[j], arr[i]
                nstate = tuple(arr)
                if nstate == target:
                    return moves + 1
                if nstate not in seen:
                    seen.add(nstate)
                    q.append((nstate, moves + 1))
    return -1
```

Time: O((m*n)!) | Space: O((m*n)!)

---

## 12. Open the Lock (Double-ended)

Open the lock with minimum moves. May have multiple targets or constraints. BFS. Bidirectional BFS can reduce search space for large state spaces.

```python
from collections import deque
def openLock(deadends, target):
    dead = set(deadends)
    if '0000' in dead:
        return -1
    q = deque([('0000', 0)])
    seen = {'0000'}
    while q:
        state, moves = q.popleft()
        if state == target:
            return moves
        for i in range(4):
            for d in (1, -1):
                nd = (int(state[i]) + d) % 10
                nstate = state[:i] + str(nd) + state[i+1:]
                if nstate not in dead and nstate not in seen:
                    seen.add(nstate)
                    q.append((nstate, moves + 1))
    return -1
```

Time: O(10^4) | Space: O(10^4)

---

## 13. Word Ladder (Optimized)

Word Ladder with very long wordList. Avoid TLE. Bidirectional BFS. Or use character-level wildcard indexing: for "hot" try "*ot", "h*t", "ho*" to find neighbors in O(word_length) per word.

```python
from collections import deque
def ladderLength(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return 0
    s1, s2 = {beginWord}, {endWord}
    d = 1
    while s1 and s2:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        next_level = set()
        for word in s1:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = word[:i] + c + word[i+1:]
                    if nw in s2:
                        return d + 1
                    if nw in words:
                        words.discard(nw)
                        next_level.add(nw)
        s1 = next_level
        d += 1
    return 0
```

Time: O(n * m^2) | Space: O(n)

---

## 14. Minimum Knight Moves (Optimized)

Minimum knight moves with large coordinates. Avoid MLE. BFS with symmetry. Only search in first quadrant. Use (abs(x), abs(y)) and bound search space.

```python
from collections import deque
def minKnightMoves(x, y):
    x, y = abs(x), abs(y)
    if x + y == 0:
        return 0
    moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    q = deque([(0, 0, 0)])
    seen = {(0, 0)}
    while q:
        r, c, d = q.popleft()
        if r == x and c == y:
            return d
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in seen and nr >= -2 and nc >= -2:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1
```

Time: O(|x|*|y|) | Space: O(|x|*|y|)

---

## 15. Critical Connections in a Network (BFS variant)

Find bridges in graph. BFS can be used in some approaches. Tarjan's DFS is standard. BFS-based level graph for max-flow can identify critical edges.

```python
from collections import deque
def criticalConnections(n, connections):
    graph = [[] for _ in range(n)]
    for a, b in connections:
        graph[a].append(b)
        graph[b].append(a)
    low = [-1] * n
    disc = [-1] * n
    parent = [-1] * n
    res = []
    time = [0]
    def dfs(u):
        disc[u] = low[u] = time[0]
        time[0] += 1
        for v in graph[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    res.append([u, v])
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    dfs(0)
    return res
```

Time: O(V + E) | Space: O(V)

---

## 16. Reconstruct Itinerary (BFS/Queue)

Lexicographically smallest Euler path. Uses queue for managing adjacency. Hierholzer's algorithm. Use heap or sorted structure for neighbors. DFS with backtracking; queue for remaining edges.

```python
def findItinerary(tickets):
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in tickets:
        graph[a].append(b)
    for k in graph:
        graph[k].sort(reverse=True)
    res = []
    def dfs(node):
        while graph[node]:
            dfs(graph[node].pop())
        res.append(node)
    dfs('JFK')
    return res[::-1]
```

Time: O(E log E) | Space: O(E)

---

## 17. Alien Dictionary (Topological + BFS)

Given sorted dictionary of alien language, derive character order. BFS for Kahn's algorithm. Build graph from adjacent word pairs. Kahn's algorithm (BFS) for topological sort. Queue for indegree-zero nodes.

```python
from collections import deque, defaultdict
def alienOrder(words):
    graph = defaultdict(set)
    indegree = {c: 0 for w in words for c in w}
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        for j in range(min(len(a), len(b))):
            if a[j] != b[j]:
                if b[j] not in graph[a[j]]:
                    graph[a[j]].add(b[j])
                    indegree[b[j]] += 1
                break
        else:
            if len(a) > len(b):
                return ""
    q = deque([c for c in indegree if indegree[c] == 0])
    res = []
    while q:
        c = q.popleft()
        res.append(c)
        for nxt in graph[c]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return ''.join(res) if len(res) == len(indegree) else ""
```

Time: O(n * m) | Space: O(1)

---

## 18. Sequence Reconstruction (BFS)

Check if sequence is the unique shortest supersequence of sequences. BFS for topological order. Build graph. Check if given sequence is the unique topological order. BFS to verify.

```python
from collections import deque, defaultdict
def sequenceReconstruction(org, seqs):
    graph = defaultdict(set)
    indegree = {x: 0 for x in org}
    for seq in seqs:
        for x in seq:
            if x not in indegree:
                indegree[x] = 0
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if b not in graph[a]:
                graph[a].add(b)
                indegree[b] = indegree.get(b, 0) + 1
    q = deque([c for c in indegree if indegree[c] == 0])
    res = []
    while q:
        if len(q) > 1:
            return False
        c = q.popleft()
        res.append(c)
        for nxt in graph[c]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return res == org
```

Time: O(n + e) | Space: O(n)

---

## 19. Course Schedule II (BFS)

Prerequisites for courses. Return valid order to take all courses. Kahn's algorithm. Build graph and indegree. BFS with queue of indegree-zero nodes.

```python
from collections import deque, defaultdict
def findOrder(numCourses, prerequisites):
    graph = defaultdict(list)
    indegree = [0] * numCourses
    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1
    q = deque([i for i in range(numCourses) if indegree[i] == 0])
    res = []
    while q:
        c = q.popleft()
        res.append(c)
        for nxt in graph[c]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return res if len(res) == numCourses else []
```

Time: O(V + E) | Space: O(V)

---

## 20. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix

Flip cells (and neighbors) to convert all to 0. Minimum flips. BFS over state space. State = flattened matrix. Each move flips a cell. Exponential size; use bitmask for small grids.

```python
from collections import deque
def minFlips(mat):
    m, n = len(mat), len(mat[0])
    start = sum(cell << (i * n + j) for i, row in enumerate(mat) for j, cell in enumerate(row))
    if start == 0:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        state, flips = q.popleft()
        if state == 0:
            return flips
        for i in range(m):
            for j in range(n):
                nstate = state
                for di, dj in [(0,0),(0,1),(0,-1),(1,0),(-1,0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        nstate ^= 1 << (ni * n + nj)
                if nstate not in seen:
                    seen.add(nstate)
                    q.append((nstate, flips + 1))
    return -1
```

Time: O(2^(m*n)) | Space: O(2^(m*n))
