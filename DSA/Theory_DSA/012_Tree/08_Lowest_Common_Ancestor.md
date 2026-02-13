# Lowest Common Ancestor

## LCA in Binary Tree (Recursive)

Return node if it equals p or q. If both subtrees return non-null, current is LCA. Else return non-null subtree result.

```python
def lca_binary_tree(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca_binary_tree(root.left, p, q)
    right = lca_binary_tree(root.right, p, q)
    if left and right:
        return root
    return left or right
```

## LCA in BST (Compare with Root) O(h)

Use BST property. If both p and q are smaller than root, LCA in left. If both larger, LCA in right. Else root is LCA.

```python
def lca_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

## LCA with Parent Pointers (Linked List Intersection)

Find depth of both nodes. Move deeper node up by depth difference. Move both up until they meet.

```python
def lca_with_parent(p, q):
    def depth(node):
        d = 0
        while node:
            node = node.parent
            d += 1
        return d
    dp = depth(p)
    dq = depth(q)
    while dp > dq:
        p = p.parent
        dp -= 1
    while dq > dp:
        q = q.parent
        dq -= 1
    while p != q:
        p = p.parent
        q = q.parent
    return p
```

## LCA for N Nodes

Extend binary LCA. LCA of n nodes = LCA of (LCA of first n/2, LCA of last n/2).

```python
def lca_n_nodes(root, nodes):
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    mid = len(nodes) // 2
    left_lca = lca_n_nodes(root, nodes[:mid])
    right_lca = lca_n_nodes(root, nodes[mid:])
    return lca_binary_tree(root, left_lca, right_lca)
```

## Distance Between Two Nodes via LCA

Distance = depth(p) + depth(q) - 2 * depth(lca).

```python
def distance_between_nodes(root, p, q):
    def depth(node, target, d):
        if not node:
            return -1
        if node == target:
            return d
        left = depth(node.left, target, d + 1)
        if left != -1:
            return left
        return depth(node.right, target, d + 1)
    lca = lca_binary_tree(root, p, q)
    return depth(lca, p, 0) + depth(lca, q, 0)
```

## Check if Node is Ancestor

```python
def is_ancestor(ancestor, node):
    if not ancestor:
        return False
    if ancestor == node:
        return True
    return is_ancestor(ancestor.left, node) or is_ancestor(ancestor.right, node)
```

## All Ancestors of a Node

```python
def all_ancestors(root, target):
    result = []
    def dfs(node, path):
        if not node:
            return False
        path.append(node)
        if node == target:
            result.extend(path[:-1])
            return True
        if dfs(node.left, path) or dfs(node.right, path):
            return True
        path.pop()
        return False
    dfs(root, [])
    return result
```

## Kth Ancestor (Binary Lifting) O(log n) per Query

Precompute ancestor table: anc[node][0] = parent, anc[node][i] = 2^i-th ancestor. For kth ancestor, use binary representation of k.

```python
def kth_ancestor_binary_lifting(parent, node, k):
    LOG = 20
    n = len(parent)
    anc = [[-1] * LOG for _ in range(n)]
    for i in range(n):
        anc[i][0] = parent[i]
    for j in range(1, LOG):
        for i in range(n):
            if anc[i][j-1] != -1:
                anc[i][j] = anc[anc[i][j-1]][j-1]
    curr = node
    for j in range(LOG):
        if (k >> j) & 1:
            curr = anc[curr][j]
            if curr == -1:
                return -1
    return curr
```

For tree with root and node pointers:

```python
def kth_ancestor(root, node, k):
    ancestors = []
    def find_path(curr, target, path):
        if not curr:
            return False
        path.append(curr)
        if curr == target:
            return True
        if find_path(curr.left, target, path) or find_path(curr.right, target, path):
            return True
        path.pop()
        return False
    find_path(root, node, ancestors)
    if k >= len(ancestors):
        return None
    return ancestors[-(k+1)]
```

## LCA Using Euler Tour + RMQ Overview

Euler tour: DFS visiting each edge twice. Store nodes in visit order. LCA(u,v) = node with minimum depth in Euler tour between first occurrence of u and first occurrence of v. Use sparse table for RMQ. Preprocessing O(n log n), query O(1).

```python
def euler_tour_lca(root, n):
    euler = []
    depth = []
    first = [-1] * (n + 1)
    def dfs(node, d):
        if not node:
            return
        first[node.val] = len(euler)
        euler.append(node.val)
        depth.append(d)
        dfs(node.left, d + 1)
        if node.left:
            euler.append(node.val)
            depth.append(d)
        dfs(node.right, d + 1)
        if node.right:
            euler.append(node.val)
            depth.append(d)
    dfs(root, 0)
    return euler, depth, first
```
