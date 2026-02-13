# Graph - Union Find (Disjoint Set Union)

## Theory

DSU maintains disjoint sets with O(alpha(n)) amortized operations. Supports union (merge two sets) and find (determine which set an element belongs to). Path compression and union by rank/size achieve inverse Ackermann complexity.

## DSU Structure

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
```

## Naive Find/Union O(n)

```python
def find_naive(parent, x):
    if parent[x] != x:
        return find_naive(parent, parent[x])
    return x

def union_naive(parent, x, y):
    px, py = find_naive(parent, x), find_naive(parent, y)
    if px != py:
        parent[px] = py
```

## Path Compression

```python
def find_path_compression(parent, x):
    if parent[x] != x:
        parent[x] = find_path_compression(parent, parent[x])
    return parent[x]
```

## Union by Rank

```python
def union_by_rank(parent, rank, x, y):
    px, py = find_path_compression(parent, x), find_path_compression(parent, y)
    if px == py:
        return
    if rank[px] < rank[py]:
        parent[px] = py
    elif rank[px] > rank[py]:
        parent[py] = px
    else:
        parent[py] = px
        rank[px] += 1
```

## Union by Size

```python
def union_by_size(parent, size, x, y):
    px, py = find_path_compression(parent, x), find_path_compression(parent, y)
    if px == py:
        return
    if size[px] < size[py]:
        px, py = py, px
    parent[py] = px
    size[px] += size[py]
```

## Both Optimizations (Inverse Ackermann)

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

## Connected Components Count

```python
def count_components(n, edges):
    dsu = DSU(n)
    for u, v in edges:
        dsu.union(u, v)
    return len(set(dsu.find(i) for i in range(n)))
```

## Detect Cycle Undirected

```python
def has_cycle_undirected(n, edges):
    dsu = DSU(n)
    for u, v in edges:
        if not dsu.union(u, v):
            return True
    return False
```

## Number of Provinces

```python
def find_circle_num(is_connected):
    n = len(is_connected)
    dsu = DSU(n)
    for i in range(n):
        for j in range(n):
            if is_connected[i][j]:
                dsu.union(i, j)
    return len(set(dsu.find(i) for i in range(n)))
```

## Redundant Connection

```python
def find_redundant_connection(edges):
    n = len(edges)
    dsu = DSU(n + 1)
    for u, v in edges:
        if not dsu.union(u, v):
            return [u, v]
    return []
```

## Accounts Merge

```python
def accounts_merge(accounts):
    from collections import defaultdict
    dsu = DSU(10001)
    email_to_id = {}
    email_to_name = {}
    id_val = 0
    for acc in accounts:
        name = acc[0]
        for i in range(1, len(acc)):
            email = acc[i]
            if email not in email_to_id:
                email_to_id[email] = id_val
                email_to_name[email] = name
                id_val += 1
            dsu.union(email_to_id[acc[1]], email_to_id[email])
    groups = defaultdict(list)
    for email in email_to_id:
        root = dsu.find(email_to_id[email])
        groups[root].append(email)
    return [[email_to_name[emails[0]]] + sorted(emails) for emails in groups.values()]
```

## Satisfiability of Equality Equations

```python
def equations_possible(equations):
    dsu = DSU(26)
    for eq in equations:
        if eq[1] == '=':
            a, b = ord(eq[0]) - ord('a'), ord(eq[3]) - ord('a')
            dsu.union(a, b)
    for eq in equations:
        if eq[1] == '!':
            a, b = ord(eq[0]) - ord('a'), ord(eq[3]) - ord('a')
            if dsu.find(a) == dsu.find(b):
                return False
    return True
```

## Most Stones Removed

```python
def remove_stones(stones):
    dsu = DSU(20002)
    for x, y in stones:
        dsu.union(x, y + 10000)
    return len(stones) - len(set(dsu.find(x) for x, y in stones))
```

## Min Operations to Connect Network

```python
def make_connected(n, connections):
    if len(connections) < n - 1:
        return -1
    dsu = DSU(n)
    for u, v in connections:
        dsu.union(u, v)
    return len(set(dsu.find(i) for i in range(n))) - 1
```

## Smallest String With Swaps

```python
def smallest_string_with_swaps(s, pairs):
    n = len(s)
    dsu = DSU(n)
    for a, b in pairs:
        dsu.union(a, b)
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    result = list(s)
    for indices in groups.values():
        chars = sorted(s[i] for i in indices)
        for i, idx in enumerate(sorted(indices)):
            result[idx] = chars[i]
    return "".join(result)
```

## Regions Cut by Slashes

```python
def regions_by_slashes(grid):
    n = len(grid)
    dsu = DSU(4 * n * n)
    for r in range(n):
        for c in range(n):
            base = 4 * (r * n + c)
            if grid[r][c] == ' ':
                dsu.union(base, base + 1)
                dsu.union(base + 1, base + 2)
                dsu.union(base + 2, base + 3)
            elif grid[r][c] == '/':
                dsu.union(base, base + 3)
                dsu.union(base + 1, base + 2)
            else:
                dsu.union(base, base + 1)
                dsu.union(base + 2, base + 3)
            if r > 0:
                dsu.union(base, base - 4 * n + 2)
            if c > 0:
                dsu.union(base + 3, base - 4 + 1)
    return len(set(dsu.find(i) for i in range(4 * n * n)))
```

## Remove Max Edges Keep Traversable

```python
def max_num_edges_to_remove(n, edges):
    dsu_a = DSU(n + 1)
    dsu_b = DSU(n + 1)
    edges.sort(key=lambda x: -x[0])
    used = 0
    for t, u, v in edges:
        if t == 3:
            if dsu_a.union(u, v) | dsu_b.union(u, v):
                used += 1
        elif t == 1:
            if dsu_a.union(u, v):
                used += 1
        else:
            if dsu_b.union(u, v):
                used += 1
    if len(set(dsu_a.find(i) for i in range(1, n + 1))) > 1 or len(set(dsu_b.find(i) for i in range(1, n + 1))) > 1:
        return -1
    return len(edges) - used
```

## Number of Islands II (Online)

```python
def num_islands_2(m, n, positions):
    dsu = DSU(m * n)
    grid = [[0] * n for _ in range(m)]
    result = []
    count = 0
    for r, c in positions:
        if grid[r][c] == 1:
            result.append(count)
            continue
        grid[r][c] = 1
        count += 1
        idx = r * n + c
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                nidx = nr * n + nc
                if dsu.union(idx, nidx):
                    count -= 1
        result.append(count)
    return result
```

## Longest Consecutive (UF Approach)

```python
def longest_consecutive_uf(nums):
    if not nums:
        return 0
    parent = {}
    size = {}
    def find(x):
        if x not in parent:
            return None
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        if x not in parent or y not in parent:
            return
        px, py = find(x), find(y)
        if px == py:
            return
        if size[px] < size[py]:
            px, py = py, px
        parent[py] = px
        size[px] += size[py]
    for x in nums:
        parent[x] = x
        size[x] = 1
        union(x, x - 1)
        union(x, x + 1)
    return max(size.values())
```

## Making a Large Island

```python
def largest_island(grid):
    n = len(grid)
    dsu = DSU(n * n)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                idx = r * n + c
                for dr, dc in [(1,0),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                        dsu.union(idx, nr * n + nc)
    size = {}
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                root = dsu.find(r * n + c)
                size[root] = size.get(root, 0) + 1
    result = max(size.values()) if size else 0
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                seen = set()
                total = 1
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                        root = dsu.find(nr * n + nc)
                        if root not in seen:
                            seen.add(root)
                            total += size[root]
                result = max(result, total)
    return result
```

## Min Cost Connect All Points (Kruskal's MST)

```python
def min_cost_connect_points(points):
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((d, i, j))
    edges.sort()
    dsu = DSU(n)
    cost = 0
    for d, u, v in edges:
        if dsu.union(u, v):
            cost += d
    return cost
```
