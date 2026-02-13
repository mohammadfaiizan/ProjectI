# Bitmask DP

## TSP (Visit All Nodes Min Cost)

```python
def tsp(graph):
    n = len(graph)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j) or graph[i][j] == float('inf'):
                    continue
                new_mask = mask | (1 << j)
                dp[new_mask][j] = min(dp[new_mask][j], dp[mask][i] + graph[i][j])
    result = float('inf')
    for i in range(1, n):
        if graph[i][0] != float('inf'):
            result = min(result, dp[(1 << n) - 1][i] + graph[i][0])
    return result
```

## Assignment Problem

```python
def assignment_problem(cost_matrix):
    n = len(cost_matrix)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        j = bin(mask).count('1')
        for i in range(n):
            if mask & (1 << i):
                continue
            new_mask = mask | (1 << i)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost_matrix[j][i])
    return dp[(1 << n) - 1]
```

## Minimum XOR Sum

```python
def minimum_xor_sum(nums1, nums2):
    n = len(nums1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        j = bin(mask).count('1')
        for i in range(n):
            if mask & (1 << i):
                continue
            new_mask = mask | (1 << i)
            dp[new_mask] = min(dp[new_mask], dp[mask] + (nums1[j] ^ nums2[i]))
    return dp[(1 << n) - 1]
```

## Shortest Superstring

```python
def shortest_superstring(words):
    n = len(words)
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(min(len(words[i]), len(words[j])), 0, -1):
                    if words[j].startswith(words[i][-k:]):
                        overlap[i][j] = k
                        break
    dp = [[''] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = words[i]
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                candidate = dp[mask][i] + words[j][overlap[i][j]:]
                if not dp[new_mask][j] or len(candidate) < len(dp[new_mask][j]):
                    dp[new_mask][j] = candidate
    result = ''
    for i in range(n):
        candidate = dp[(1 << n) - 1][i]
        if not result or len(candidate) < len(result):
            result = candidate
    return result
```

## Maximum Students Taking Exam

```python
def max_students(seats):
    m, n = len(seats), len(seats[0])
    
    def count_bits(x):
        return bin(x).count('1')
    
    def valid(row_idx, row_mask, prev_mask):
        for j in range(n):
            if not (row_mask & (1 << j)):
                continue
            if seats[row_idx][j] == '#':
                return False
            if j > 0 and (row_mask & (1 << (j - 1))):
                return False
            if j > 0 and (prev_mask & (1 << (j - 1))):
                return False
            if j < n - 1 and (prev_mask & (1 << (j + 1))):
                return False
        return True
    
    dp = {0: 0}
    for row_idx in range(m):
        new_dp = {}
        for prev_mask in dp:
            for curr_mask in range(1 << n):
                if valid(row_idx, curr_mask, prev_mask):
                    new_dp[curr_mask] = max(new_dp.get(curr_mask, 0), dp[prev_mask] + count_bits(curr_mask))
        dp = new_dp
    return max(dp.values()) if dp else 0
```

## Different Hats

```python
def number_ways(hats):
    n = len(hats)
    MOD = 10**9 + 7
    person_to_hats = [set() for _ in range(n)]
    max_hat = 0
    for i, h_list in enumerate(hats):
        for h in h_list:
            person_to_hats[i].add(h)
            max_hat = max(max_hat, h)
    hat_to_people = [[] for _ in range(max_hat + 1)]
    for i in range(n):
        for h in person_to_hats[i]:
            hat_to_people[h].append(i)
    dp = [0] * (1 << n)
    dp[0] = 1
    for h in range(1, max_hat + 1):
        for mask in range((1 << n) - 1, -1, -1):
            for p in hat_to_people[h]:
                if mask & (1 << p):
                    continue
                new_mask = mask | (1 << p)
                dp[new_mask] = (dp[new_mask] + dp[mask]) % MOD
    return dp[(1 << n) - 1]
```

## Distribute Repeating Integers

```python
def can_distribute(nums, quantity):
    from collections import Counter
    count = list(Counter(nums).values())
    quantity.sort(reverse=True)
    m = len(quantity)
    n = len(count)
    sum_mask = [0] * (1 << m)
    for mask in range(1 << m):
        for i in range(m):
            if mask & (1 << i):
                sum_mask[mask] += quantity[i]
    dp = [[False] * (1 << m) for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(n):
        for mask in range(1 << m):
            if not dp[i][mask]:
                continue
            submask = ((1 << m) - 1) ^ mask
            s = submask
            while s:
                if sum_mask[s] <= count[i]:
                    dp[i + 1][mask | s] = True
                s = (s - 1) & submask
            dp[i + 1][mask] = dp[i + 1][mask] or dp[i][mask]
    return dp[n][(1 << m) - 1]
```

## Maximum Compatibility Score Sum

```python
def max_compatibility_sum(students, mentors):
    m = len(students)
    n = len(students[0])
    score = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            score[i][j] = sum(1 for k in range(n) if students[i][k] == mentors[j][k])
    dp = [0] * (1 << m)
    for mask in range(1 << m):
        j = bin(mask).count('1')
        for i in range(m):
            if mask & (1 << i):
                continue
            new_mask = mask | (1 << i)
            dp[new_mask] = max(dp[new_mask], dp[mask] + score[j][i])
    return dp[(1 << m) - 1]
```

## Parallel Courses II

```python
def min_number_of_semesters(n, dependencies, k):
    prereq = [0] * n
    for u, v in dependencies:
        prereq[v - 1] |= 1 << (u - 1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        available = []
        for i in range(n):
            if (mask & (1 << i)) == 0 and (prereq[i] & mask) == prereq[i]:
                available.append(i)
        for take in range(1, min(k, len(available)) + 1):
            from itertools import combinations
            for combo in combinations(available, take):
                new_mask = mask
                for c in combo:
                    new_mask |= 1 << c
                dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
    return dp[(1 << n) - 1]
```

## Smallest Sufficient Team

```python
def smallest_sufficient_team(req_skills, people):
    n = len(req_skills)
    skill_to_idx = {s: i for i, s in enumerate(req_skills)}
    people_mask = []
    for p in people:
        mask = 0
        for s in p:
            mask |= 1 << skill_to_idx[s]
        people_mask.append(mask)
    dp = [None] * (1 << n)
    dp[0] = []
    for mask in range(1 << n):
        if dp[mask] is None:
            continue
        for i, pm in enumerate(people_mask):
            new_mask = mask | pm
            if new_mask == mask:
                continue
            if dp[new_mask] is None or len(dp[mask]) + 1 < len(dp[new_mask]):
                dp[new_mask] = dp[mask] + [i]
    return dp[(1 << n) - 1]
```

## Maximum AND Sum

```python
def maximum_and_sum(nums, num_slots):
    n = len(nums)
    dp = [0] * (1 << (2 * num_slots))
    for mask in range(1 << (2 * num_slots)):
        idx = bin(mask).count('1')
        if idx >= n:
            continue
        for slot in range(num_slots):
            for pos in range(2):
                bit = 2 * slot + pos
                if mask & (1 << bit):
                    continue
                new_mask = mask | (1 << bit)
                val = (nums[idx] & (slot + 1))
                dp[new_mask] = max(dp[new_mask], dp[mask] + val)
    return max(dp)
```

## Minimum Work Sessions

```python
def min_sessions(tasks, session_time):
    n = len(tasks)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        rem = session_time
        for i in range(n):
            if mask & (1 << i):
                continue
            if tasks[i] <= rem:
                new_mask = mask | (1 << i)
                dp[new_mask] = min(dp[new_mask], dp[mask])
                rem -= tasks[i]
            else:
                new_mask = mask | (1 << i)
                dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
                rem = session_time - tasks[i]
    return dp[(1 << n) - 1]
```

## Stickers to Spell Word

```python
def min_stickers(stickers, target):
    n = len(target)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        for sticker in stickers:
            new_mask = mask
            from collections import Counter
            cnt = Counter(sticker)
            for i, c in enumerate(target):
                if (mask >> i) & 1:
                    continue
                if cnt.get(c, 0) > 0:
                    cnt[c] -= 1
                    new_mask |= 1 << i
            dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
    return dp[(1 << n) - 1] if dp[(1 << n) - 1] != float('inf') else -1
```
