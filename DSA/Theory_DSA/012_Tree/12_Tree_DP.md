# Tree Dynamic Programming

## DP on Trees Concept

Solve subproblems at each node, typically in DFS post-order (process children before parent). State often includes: include/exclude node, values from subtrees. Combine children results to compute parent.

## Diameter via DP

For each node, diameter through node = max left depth + max right depth. Track global max.

```python
def diameter_of_binary_tree(root):
    result = 0
    def height(node):
        nonlocal result
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        result = max(result, left + right)
        return 1 + max(left, right)
    height(root)
    return result
```

## Max Path Sum

At each node: max path through node = node.val + max(0, left_gain) + max(0, right_gain). Return to parent: node.val + max(left_gain, right_gain).

```python
def max_path_sum(root):
    result = float('-inf')
    def gain(node):
        nonlocal result
        if not node:
            return 0
        left = max(0, gain(node.left))
        right = max(0, gain(node.right))
        result = max(result, node.val + left + right)
        return node.val + max(left, right)
    gain(root)
    return result
```

## House Robber III

Each node: (rob_this, skip_this). Rob = val + left_skip + right_skip. Skip = max(left) + max(right).

```python
def rob(root):
    def dfs(node):
        if not node:
            return (0, 0)
        left = dfs(node.left)
        right = dfs(node.right)
        rob_this = node.val + left[1] + right[1]
        skip_this = max(left) + max(right)
        return (rob_this, skip_this)
    return max(dfs(root))
```

## Binary Tree Cameras

State: 0=covered, 1=covered by child (camera), 2=needs cover. Leaf returns 2. If any child is 2, place camera (return 1). If any child is 1, current covered (return 0). Else return 2.

```python
def min_camera_cover(root):
    cameras = 0
    COVERED = 0
    COVERED_BY_CHILD = 1
    NEEDS_COVER = 2
    def dfs(node):
        nonlocal cameras
        if not node:
            return COVERED
        left = dfs(node.left)
        right = dfs(node.right)
        if left == NEEDS_COVER or right == NEEDS_COVER:
            cameras += 1
            return COVERED_BY_CHILD
        if left == COVERED_BY_CHILD or right == COVERED_BY_CHILD:
            return COVERED
        return NEEDS_COVER
    if dfs(root) == NEEDS_COVER:
        cameras += 1
    return cameras
```

## Distribute Coins

Excess = coins - 1. Moves = sum of absolute excess passed across edges. DFS returns excess of subtree (positive = send up, negative = need down).

```python
def distribute_coins(root):
    moves = 0
    def dfs(node):
        nonlocal moves
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        moves += abs(left) + abs(right)
        return node.val + left + right - 1
    dfs(root)
    return moves
```

## Longest Univalue Path

At each node: longest univalue path through node = left_same + right_same (if children match). Return to parent: 1 + max(left_same, right_same) if child matches.

```python
def longest_univalue_path(root):
    result = 0
    def dfs(node):
        nonlocal result
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        left_len = 1 + left if node.left and node.left.val == node.val else 0
        right_len = 1 + right if node.right and node.right.val == node.val else 0
        result = max(result, left_len + right_len)
        return max(left_len, right_len)
    dfs(root)
    return result
```

## Count Good Nodes

```python
def good_nodes(root):
    def dfs(node, max_so_far):
        if not node:
            return 0
        count = 1 if node.val >= max_so_far else 0
        max_so_far = max(max_so_far, node.val)
        return count + dfs(node.left, max_so_far) + dfs(node.right, max_so_far)
    return dfs(root, float('-inf'))
```

## Sum of Distances in Tree (Re-rooting)

Two DFS. First: compute subtree sizes and sum of distances from root. Second: when moving root from parent to child, new_sum = old_sum - size[child] + (n - size[child]).

```python
def sum_of_distances_in_tree(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    size = [0] * n
    dist_sum = [0] * n
    def dfs1(node, parent):
        size[node] = 1
        for child in graph[node]:
            if child != parent:
                dfs1(child, node)
                size[node] += size[child]
                dist_sum[node] += dist_sum[child] + size[child]
    def dfs2(node, parent):
        for child in graph[node]:
            if child != parent:
                dist_sum[child] = dist_sum[node] - size[child] + (n - size[child])
                dfs2(child, node)
    dfs1(0, -1)
    dfs2(0, -1)
    return dist_sum
```

## Minimum Cost Tree from Leaf Values

For each segment [i,j], try all splits k. Cost = max(left[i:k]) * max(right[k:j]) + dp[i][k] + dp[k][j]. Use DP or MCM style.

```python
def mct_from_leaf_values(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                dp[i][j] = min(dp[i][j],
                    max(arr[i:k+1]) * max(arr[k+1:j+1]) + dp[i][k] + dp[k+1][j])
    return dp[0][n-1]
```

## Number of Good Leaf Node Pairs

Count pairs of leaves with distance <= distance. For each node, get list of leaf distances in subtree. Combine: for each left_dist, right_dist, if left_dist + right_dist <= distance, add count. Return leaf distances + 1 to parent.

```python
def count_pairs(root, distance):
    result = 0
    def dfs(node):
        nonlocal result
        if not node:
            return []
        if not node.left and not node.right:
            return [1]
        left = dfs(node.left)
        right = dfs(node.right)
        for l in left:
            for r in right:
                if l + r <= distance:
                    result += 1
        return [d + 1 for d in left + right if d + 1 < distance]
    dfs(root)
    return result
```

## Longest Zigzag Path

State: (longest_ending_left, longest_ending_right). From left child: right_len = 1 + left_right. From right child: left_len = 1 + right_left.

```python
def longest_zigzag(root):
    result = 0
    def dfs(node):
        nonlocal result
        if not node:
            return (-1, -1)
        left = dfs(node.left)
        right = dfs(node.right)
        left_len = 1 + right[0]
        right_len = 1 + left[1]
        result = max(result, left_len, right_len)
        return (left_len, right_len)
    dfs(root)
    return result
```
