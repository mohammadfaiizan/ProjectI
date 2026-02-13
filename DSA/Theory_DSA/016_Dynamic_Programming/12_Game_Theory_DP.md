# Game Theory DP

## Stone Game I (Proof + DP)

```python
def stone_game(piles):
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
    return dp[0][n - 1] > 0
```

## Stone Game II (Variable M)

```python
def stone_game_ii(piles):
    n = len(piles)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + piles[i]
    memo = {}
    
    def dp(i, m):
        if i >= n:
            return 0
        if (i, m) in memo:
            return memo[(i, m)]
        best = 0
        for x in range(1, min(2 * m, n - i) + 1):
            take = suffix[i] - suffix[i + x] - dp(i + x, max(m, x))
            best = max(best, take)
        memo[(i, m)] = best
        return best
    
    return (suffix[0] + dp(0, 1)) // 2
```

## Stone Game III

```python
def stone_game_iii(stone_value):
    n = len(stone_value)
    dp = [float('-inf')] * (n + 1)
    dp[n] = 0
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i, min(i + 3, n)):
            s += stone_value[j]
            dp[i] = max(dp[i], s - dp[j + 1])
    return "Alice" if dp[0] > 0 else ("Bob" if dp[0] < 0 else "Tie")
```

## Stone Game IV (Square Removal)

```python
def winner_square_game(n):
    dp = [False] * (n + 1)
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            if not dp[i - j * j]:
                dp[i] = True
                break
            j += 1
    return dp[n]
```

## Predict the Winner

```python
def predict_the_winner(nums):
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = nums[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0
```

## Nim Game (XOR)

```python
def can_win_nim(n):
    return n % 4 != 0

def nim_sum(piles):
    result = 0
    for p in piles:
        result ^= p
    return result

def nim_game(piles):
    return nim_sum(piles) != 0
```

## Can I Win

```python
def can_i_win(max_choosable, desired_total):
    if desired_total <= 0:
        return True
    if max_choosable * (max_choosable + 1) // 2 < desired_total:
        return False
    memo = {}
    
    def dp(used, remaining):
        if remaining <= 0:
            return False
        if used in memo:
            return memo[used]
        for i in range(1, max_choosable + 1):
            mask = 1 << i
            if not (used & mask):
                if i >= remaining or not dp(used | mask, remaining - i):
                    memo[used] = True
                    return True
        memo[used] = False
        return False
    
    return dp(0, desired_total)
```

## Flip Game II

```python
def can_win_flip(s):
    memo = {}
    
    def dp(state):
        if state in memo:
            return memo[state]
        for i in range(len(state) - 1):
            if state[i] == '+' and state[i + 1] == '+':
                new_state = state[:i] + '--' + state[i + 2:]
                if not dp(new_state):
                    memo[state] = True
                    return True
        memo[state] = False
        return False
    
    return dp(s)
```

## Grundy Numbers / Sprague-Grundy Theorem

```python
def mex(s):
    i = 0
    while i in s:
        i += 1
    return i

def grundy_nim(piles):
    return 0 if not piles else piles[0] if len(piles) == 1 else reduce(lambda x, y: x ^ y, piles)

def calculate_grundy(n, moves):
    grundy = [0] * (n + 1)
    for i in range(1, n + 1):
        reachable = set()
        for m in moves:
            if i >= m:
                reachable.add(grundy[i - m])
        grundy[i] = mex(reachable)
    return grundy
```

## Cat and Mouse Game

```python
def cat_mouse_game(graph):
    n = len(graph)
    DRAW, MOUSE, CAT = 0, 1, 2
    color = [[[0] * n for _ in range(n)] for _ in range(3)]
    degree = [[[0] * n for _ in range(n)] for _ in range(3)]
    for m in range(n):
        for c in range(n):
            degree[MOUSE][m][c] = len(graph[m])
            degree[CAT][m][c] = len(graph[c])
            for node in graph[c]:
                if node == 0:
                    continue
                degree[DRAW][m][c] += 1
    from collections import deque
    q = deque()
    for i in range(1, n):
        for t in range(1, 3):
            color[t][i][i] = CAT
            q.append((t, i, i))
        color[MOUSE][0][i] = MOUSE
        color[CAT][0][i] = MOUSE
        q.append((MOUSE, 0, i))
        q.append((CAT, 0, i))
    while q:
        t, m, c = q.popleft()
        for prev_state in get_prev_states(graph, t, m, c, MOUSE, CAT):
            t2, m2, c2 = prev_state
            if color[t2][m2][c2] != DRAW:
                continue
            if (t2 == MOUSE and t == CAT) or (t2 == CAT and t == MOUSE):
                color[t2][m2][c2] = t
                q.append((t2, m2, c2))
            else:
                degree[t2][m2][c2] -= 1
                if degree[t2][m2][c2] == 0:
                    color[t2][m2][c2] = 3 - t2
                    q.append((t2, m2, c2))
    return color[MOUSE][1][2]

def get_prev_states(graph, t, m, c, MOUSE, CAT):
    res = []
    if t == CAT:
        for m2 in graph[m]:
            res.append((MOUSE, m2, c))
    else:
        for c2 in graph[c]:
            if c2 != 0:
                res.append((CAT, m, c2))
    return res
```
