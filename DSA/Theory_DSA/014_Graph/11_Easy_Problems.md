# Graph - Easy Problems

## 1. Find the Town Judge

**Description**: In a town of n people, trust array gives who trusts whom. The town judge trusts nobody, everyone trusts the judge. Find the judge if exists.

**Approach**: Count in-degree (trusted by) and out-degree (trusts) for each person. Judge has in-degree n-1 and out-degree 0.

```python
def findJudge(n, trust):
    in_deg = [0] * (n + 1)
    out_deg = [0] * (n + 1)
    for a, b in trust:
        out_deg[a] += 1
        in_deg[b] += 1
    for i in range(1, n + 1):
        if in_deg[i] == n - 1 and out_deg[i] == 0:
            return i
    return -1
```

Time: O(n + t) | Space: O(n)

---

## 2. Find Center of Star Graph

**Description**: Given edges of a star graph (one center connected to all others), find the center.

**Approach**: The center appears in every edge. Check first two edges; common vertex is center.

```python
def findCenter(edges):
    return edges[0][0] if edges[0][0] in edges[1] else edges[0][1]
```

Time: O(1) | Space: O(1)

---

## 3. Number of Provinces

**Description**: n cities, is_connected matrix. Find number of connected components.

**Approach**: DFS/BFS from each unvisited city. Or Union-Find.

```python
def findCircleNum(isConnected):
    n, visited, count = len(isConnected), set(), 0
    def dfs(i):
        visited.add(i)
        for j in range(n):
            if isConnected[i][j] and j not in visited:
                dfs(j)
    for i in range(n):
        if i not in visited:
            dfs(i)
            count += 1
    return count
```

Time: O(n^2) | Space: O(n)

---

## 4. Find if Path Exists in Graph

**Description**: Undirected graph, check if path exists from source to destination.

**Approach**: BFS or DFS from source, check if destination is reached.

```python
def validPath(n, edges, source, destination):
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited, q = set(), [source]
    while q:
        node = q.pop()
        if node == destination:
            return True
        visited.add(node)
        for nei in adj[node]:
            if nei not in visited:
                q.append(nei)
    return False
```

Time: O(n + e) | Space: O(n)

---

## 5. Keys and Rooms

**Description**: rooms[i] has keys to other rooms. Can you visit all rooms starting from 0?

**Approach**: DFS/BFS from room 0. Check if visited size equals number of rooms.

```python
def canVisitAllRooms(rooms):
    visited, stack = set(), [0]
    while stack:
        r = stack.pop()
        if r in visited:
            continue
        visited.add(r)
        for k in rooms[r]:
            stack.append(k)
    return len(visited) == len(rooms)
```

Time: O(n + k) | Space: O(n)

---

## 6. Flood Fill

**Description**: Given image, start pixel, new color. Flood fill connected same-color pixels.

**Approach**: DFS or BFS from start, replace color for all connected same-color cells.

```python
def floodFill(image, sr, sc, newColor):
    old, m, n = image[sr][sc], len(image), len(image[0])
    if old == newColor:
        return image
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and image[r][c] == old:
            image[r][c] = newColor
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(r+dr, c+dc)
    dfs(sr, sc)
    return image
```

Time: O(m * n) | Space: O(m * n)

---

## 7. Number of Islands

**Description**: 2D grid of '1' and '0'. Count number of connected '1' regions.

**Approach**: DFS/BFS for each unvisited '1', mark entire island, increment count.

```python
def numIslands(grid):
    m, n, count = len(grid), len(grid[0]), 0
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == '1':
            grid[r][c] = '0'
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                dfs(r+dr, c+dc)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count
```

Time: O(m * n) | Space: O(m * n)

---

## 8. Max Area of Island

**Description**: Find maximum area of connected 1s in grid.

**Approach**: DFS for each island, return area (count of cells). Track global max.

```python
def maxAreaOfIsland(grid):
    m, n, res = len(grid), len(grid[0]), 0
    def dfs(r, c):
        if 0 <= r < m and 0 <= c < n and grid[r][c] == 1:
            grid[r][c] = 0
            return 1 + sum(dfs(r+dr, c+dc) for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)])
        return 0
    for i in range(m):
        for j in range(n):
            res = max(res, dfs(i, j))
    return res
```

Time: O(m * n) | Space: O(m * n)

---

## 9. Is Graph Bipartite

**Description**: Can vertices be colored with 2 colors such that no adjacent vertices share color?

**Approach**: BFS/DFS 2-coloring. If conflict, not bipartite.

```python
def isBipartite(graph):
    color = {}
    for i in range(len(graph)):
        if i in color:
            continue
        q = [i]
        color[i] = 0
        while q:
            node = q.pop()
            for nei in graph[node]:
                if nei in color:
                    if color[nei] == color[node]:
                        return False
                else:
                    color[nei] = 1 - color[node]
                    q.append(nei)
    return True
```

Time: O(n + e) | Space: O(n)

---

## 10. Possible Bipartition

**Description**: n people, dislikes pairs. Can we split into two groups with no dislikes within group?

**Approach**: Build graph from dislikes. Check bipartite with BFS/DFS.

```python
def possibleBipartition(n, dislikes):
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in dislikes:
        adj[a].append(b)
        adj[b].append(a)
    color = {}
    for i in range(1, n + 1):
        if i in color:
            continue
        q, color[i] = [i], 0
        while q:
            node = q.pop()
            for nei in adj[node]:
                if nei in color:
                    if color[nei] == color[node]:
                        return False
                else:
                    color[nei] = 1 - color[node]
                    q.append(nei)
    return True
```

Time: O(n + e) | Space: O(n)

---

## 11. Employee Importance

**Description**: Employee tree structure. Get total importance of employee and all subordinates.

**Approach**: Build adjacency from id to employee. DFS/BFS from given id, sum importance.

```python
def getImportance(employees, id):
    emp = {e.id: e for e in employees}
    def dfs(i):
        return emp[i].importance + sum(dfs(s) for s in emp[i].subordinates)
    return dfs(id)
```

Time: O(n) | Space: O(n)

---

## 12. Clone Graph

**Description**: Deep copy a graph with same structure and values.

**Approach**: DFS with mapping dict. For each node, create copy, recurse on neighbors.

```python
def cloneGraph(node):
    if not node:
        return None
    seen = {}
    def dfs(n):
        if n.val in seen:
            return seen[n.val]
        copy = Node(n.val)
        seen[n.val] = copy
        copy.neighbors = [dfs(nei) for nei in n.neighbors]
        return copy
    return dfs(node)
```

Time: O(n) | Space: O(n)

---

## 13. All Paths From Source to Target

**Description**: DAG, find all paths from node 0 to node n-1.

**Approach**: DFS backtracking. Add current to path, recurse on neighbors, backtrack.

```python
def allPathsSourceTarget(graph):
    n, res = len(graph), []
    def dfs(node, path):
        if node == n - 1:
            res.append(path[:])
            return
        for nei in graph[node]:
            path.append(nei)
            dfs(nei, path)
            path.pop()
    dfs(0, [0])
    return res
```

Time: O(2^n) | Space: O(n)

---

## 14. Valid Path

**Description**: n vertices, edges, check if path exists from source to destination.

**Approach**: Build adjacency list, BFS/DFS from source.

```python
def validPath(n, edges, source, destination):
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited, q = set(), [source]
    while q:
        node = q.pop()
        if node == destination:
            return True
        visited.add(node)
        q.extend(nei for nei in adj[node] if nei not in visited)
    return False
```

Time: O(n + e) | Space: O(n)

---

## 15. Minimum Depth of Binary Tree (Graph View)

**Description**: In tree/graph, find minimum distance from root to any leaf.

**Approach**: BFS level by level, return depth when first leaf found.

```python
def minDepth(root):
    if not root:
        return 0
    q, depth = [root], 1
    while q:
        nq = []
        for node in q:
            if not node.left and not node.right:
                return depth
            if node.left:
                nq.append(node.left)
            if node.right:
                nq.append(node.right)
        q, depth = nq, depth + 1
    return depth
```

Time: O(n) | Space: O(n)

---

## 16. Same Tree

**Description**: Check if two trees are identical.

**Approach**: Recursive compare structure and values.

```python
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

Time: O(n) | Space: O(h)

---

## 17. Symmetric Tree

**Description**: Check if tree is mirror of itself.

**Approach**: Helper(left, right): both null true; recurse (left.left, right.right) and (left.right, right.left).

```python
def isSymmetric(root):
    def mirror(l, r):
        if not l and not r:
            return True
        if not l or not r or l.val != r.val:
            return False
        return mirror(l.left, r.right) and mirror(l.right, r.left)
    return mirror(root.left, root.right) if root else True
```

Time: O(n) | Space: O(h)

---

## 18. Invert Binary Tree

**Description**: Swap left and right children for every node.

**Approach**: Recursive swap, then recurse on both children.

```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

Time: O(n) | Space: O(h)

---

## 19. Merge Two Binary Trees

**Description**: Overlay two trees, sum values at overlapping nodes.

**Approach**: Recursive merge. If one null return other. Create new node with sum, merge left and right.

```python
def mergeTrees(t1, t2):
    if not t1:
        return t2
    if not t2:
        return t1
    t1.val += t2.val
    t1.left = mergeTrees(t1.left, t2.left)
    t1.right = mergeTrees(t1.right, t2.right)
    return t1
```

Time: O(n) | Space: O(h)

---

## 20. Average of Levels in Binary Tree

**Description**: Return average value at each level.

**Approach**: BFS level order, compute average per level.

```python
def averageOfLevels(root):
    res, q = [], [root] if root else []
    while q:
        nq, total = [], 0
        for node in q:
            total += node.val
            if node.left:
                nq.append(node.left)
            if node.right:
                nq.append(node.right)
        res.append(total / len(q))
        q = nq
    return res
```

Time: O(n) | Space: O(n)

---

## 21. Second Minimum Node in Binary Tree

**Description**: Find second smallest value in tree (each node has 0, 1, or 2 children).

**Approach**: DFS collect all values, find second min. Or track first and second during traversal.

```python
def findSecondMinimumValue(root):
    vals = set()
    def dfs(node):
        if node:
            vals.add(node.val)
            dfs(node.left)
            dfs(node.right)
    dfs(root)
    vals.discard(min(vals))
    return min(vals) if vals else -1
```

Time: O(n) | Space: O(n)

---

## 22. N-ary Tree Preorder Traversal

**Description**: Preorder traversal of n-ary tree.

**Approach**: Visit root, recurse on children in order.

```python
def preorder(root):
    if not root:
        return []
    res = [root.val]
    for child in root.children:
        res.extend(preorder(child))
    return res
```

Time: O(n) | Space: O(h)

---

## 23. N-ary Tree Postorder Traversal

**Description**: Postorder traversal of n-ary tree.

**Approach**: Recurse on all children, then visit root.

```python
def postorder(root):
    if not root:
        return []
    res = []
    for child in root.children:
        res.extend(postorder(child))
    res.append(root.val)
    return res
```

Time: O(n) | Space: O(h)

---

## 24. N-ary Tree Level Order Traversal

**Description**: Level order traversal of n-ary tree.

**Approach**: BFS with queue.

```python
def levelOrder(root):
    if not root:
        return []
    res, q = [], [root]
    while q:
        res.append([n.val for n in q])
        q = [c for n in q for c in n.children]
    return res
```

Time: O(n) | Space: O(n)

---

## 25. Maximum Depth of N-ary Tree

**Description**: Return maximum depth of n-ary tree.

**Approach**: 1 + max(child depths) for each node, or BFS count levels.

```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max((maxDepth(c) for c in root.children), default=0)
```

Time: O(n) | Space: O(h)
