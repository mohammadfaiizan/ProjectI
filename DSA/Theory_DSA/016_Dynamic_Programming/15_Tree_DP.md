# Tree DP

## House Robber III

```python
def rob_tree(root):
    def dp(node):
        if not node:
            return (0, 0)
        left_rob, left_skip = dp(node.left)
        right_rob, right_skip = dp(node.right)
        rob = node.val + left_skip + right_skip
        skip = max(left_rob, left_skip) + max(right_rob, right_skip)
        return (rob, skip)
    return max(dp(root))
```

## Binary Tree Cameras

```python
def min_camera_cover(root):
    def dp(node):
        if not node:
            return (0, 0, float('inf'))
        l = dp(node.left)
        r = dp(node.right)
        cover_children = min(l[1], l[2]) + min(r[1], r[2])
        cover_self = 1 + min(l) + min(r)
        cover_parent = min(l[1], l[2]) + min(r[1], r[2])
        return (cover_children, cover_self, cover_parent)
    result = dp(root)
    return min(result[0], result[1])
```

## Max Path Sum

```python
def max_path_sum(root):
    result = [float('-inf')]
    
    def dfs(node):
        if not node:
            return 0
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        result[0] = max(result[0], node.val + left + right)
        return node.val + max(left, right)
    
    dfs(root)
    return result[0]
```

## Diameter via DP

```python
def diameter_of_binary_tree(root):
    result = [0]
    
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        result[0] = max(result[0], left + right)
        return 1 + max(left, right)
    
    height(root)
    return result[0]
```

## Longest Univalue Path

```python
def longest_univalue_path(root):
    result = [0]
    
    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        left_arrow = right_arrow = 0
        if node.left and node.left.val == node.val:
            left_arrow = left + 1
        if node.right and node.right.val == node.val:
            right_arrow = right + 1
        result[0] = max(result[0], left_arrow + right_arrow)
        return max(left_arrow, right_arrow)
    
    dfs(root)
    return result[0]
```

## Distribute Coins

```python
def distribute_coins(root):
    moves = [0]
    
    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        moves[0] += abs(left) + abs(right)
        return node.val - 1 + left + right
    
    dfs(root)
    return moves[0]
```

## Sum of Distances in Tree (Re-rooting Two DFS)

```python
def sum_of_distances_in_tree(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    count = [1] * n
    ans = [0] * n
    
    def dfs1(node, parent):
        for child in graph[node]:
            if child != parent:
                dfs1(child, node)
                count[node] += count[child]
                ans[node] += ans[child] + count[child]
    
    def dfs2(node, parent):
        for child in graph[node]:
            if child != parent:
                ans[child] = ans[node] - count[child] + (n - count[child])
                dfs2(child, node)
    
    dfs1(0, -1)
    dfs2(0, -1)
    return ans
```

## Number of Good Leaf Node Pairs

```python
def count_pairs(root, distance):
    result = [0]
    
    def dfs(node):
        if not node:
            return []
        if not node.left and not node.right:
            return [1]
        left = dfs(node.left)
        right = dfs(node.right)
        for l in left:
            for r in right:
                if l + r <= distance:
                    result[0] += 1
        return [d + 1 for d in left + right if d + 1 < distance]
    
    dfs(root)
    return result[0]
```

## Minimum Cost Tree from Leaf Values

```python
def mct_from_leaf_values(arr):
    n = len(arr)
    dp = [[float('inf')] * n for _ in range(n)]
    max_val = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
        max_val[i][i] = arr[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            max_val[i][j] = max(max_val[i][j - 1], arr[j])
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + max_val[i][k] * max_val[k + 1][j])
    return dp[0][n - 1]
```

## Longest Zigzag Path

```python
def longest_zigzag(root):
    result = [0]
    
    def dfs(node, left_len, right_len):
        if not node:
            return
        result[0] = max(result[0], left_len, right_len)
        if node.left:
            dfs(node.left, right_len + 1, 0)
        if node.right:
            dfs(node.right, 0, left_len + 1)
    
    dfs(root, 0, 0)
    return result[0]
```

## Count Nodes Equal to Average of Subtree

```python
def average_of_subtree(root):
    result = [0]
    
    def dfs(node):
        if not node:
            return (0, 0)
        left_sum, left_count = dfs(node.left)
        right_sum, right_count = dfs(node.right)
        total_sum = node.val + left_sum + right_sum
        total_count = 1 + left_count + right_count
        if total_sum // total_count == node.val:
            result[0] += 1
        return (total_sum, total_count)
    
    dfs(root)
    return result[0]
```
