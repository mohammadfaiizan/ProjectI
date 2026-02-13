# Tree - Medium Problems

## 1. Binary Tree Right Side View

Return values of rightmost nodes at each level. BFS, append last node of each level. Or DFS with level, overwrite result[level] (right-first DFS).

```python
def rightSideView(root):
    if not root:
        return []
    from collections import deque
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

## 2. Binary Tree Level Order Traversal II

Level order from bottom to top. Same as level order, reverse result. Or use deque appendleft for each level.

```python
def levelOrderBottom(root):
    if not root:
        return []
    from collections import deque
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res[::-1]
```

Time: O(n) | Space: O(n)

---

## 3. Binary Tree Zigzag Level Order Traversal

Level order but alternate left-to-right and right-to-left per level. BFS. Reverse every other level. Or use deque and alternate popleft/popright.

```python
def zigzagLevelOrder(root):
    if not root:
        return []
    from collections import deque
    q = deque([root])
    res = []
    rev = False
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level[::-1] if rev else level)
        rev = not rev
    return res
```

Time: O(n) | Space: O(n)

---

## 4. Construct Binary Tree from Preorder and Inorder

Build tree from preorder and inorder traversal. Preorder[0] is root. Find in inorder to split. Left subtree: inorder[:idx], preorder[1:idx+1]. Right: inorder[idx+1:], preorder[idx+1:].

```python
def buildTree(preorder, inorder):
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    idx = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:idx+1], inorder[:idx])
    root.right = buildTree(preorder[idx+1:], inorder[idx+1:])
    return root
```

Time: O(n) | Space: O(n)

---

## 5. Construct Binary Tree from Inorder and Postorder

Build tree from inorder and postorder. Postorder[-1] is root. Find in inorder. Left: inorder[:idx], postorder[:idx]. Right: inorder[idx+1:], postorder[idx:-1].

```python
def buildTree(inorder, postorder):
    if not inorder:
        return None
    root = TreeNode(postorder[-1])
    idx = inorder.index(postorder[-1])
    root.left = buildTree(inorder[:idx], postorder[:idx])
    root.right = buildTree(inorder[idx+1:], postorder[idx:-1])
    return root
```

Time: O(n) | Space: O(n)

---

## 6. Binary Tree Maximum Path Sum

Find maximum path sum (any node to any node, path = parent-child). At each node, path through node = val + max(0, left_gain) + max(0, right_gain). Return to parent: val + max(left, right). Track global max.

```python
def maxPathSum(root):
    best = float('-inf')
    def gain(node):
        nonlocal best
        if not node:
            return 0
        l = max(0, gain(node.left))
        r = max(0, gain(node.right))
        best = max(best, node.val + l + r)
        return node.val + max(l, r)
    gain(root)
    return best
```

Time: O(n) | Space: O(h)

---

## 7. Validate Binary Search Tree

Check if tree is valid BST. DFS with (low, high) range. Node must be in (low, high). Recurse left with (low, val), right with (val, high).

```python
def isValidBST(root):
    def check(node, lo, hi):
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return check(node.left, lo, node.val) and check(node.right, node.val, hi)
    return check(root, float('-inf'), float('inf'))
```

Time: O(n) | Space: O(h)

---

## 8. Kth Smallest Element in BST

Find kth smallest element. Inorder traversal, return when count reaches k. Iterative stack or recursive with counter.

```python
def kthSmallest(root, k):
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.right
```

Time: O(h + k) | Space: O(h)

---

## 9. Lowest Common Ancestor of Binary Tree

Find LCA of two nodes. If root is p or q or null, return root. Recurse left and right. If both return non-null, root is LCA. Else return non-null.

```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

Time: O(n) | Space: O(h)

---

## 10. Lowest Common Ancestor of BST

Find LCA in BST. If both p, q < root, go left. If both > root, go right. Else root is LCA.

```python
def lowestCommonAncestor(root, p, q):
    if p.val < root.val and q.val < root.val:
        return lowestCommonAncestor(root.left, p, q)
    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor(root.right, p, q)
    return root
```

Time: O(h) | Space: O(h)

---

## 11. Binary Tree from Preorder and Postorder (Full Binary)

Construct full binary tree from preorder and postorder. Preorder[0] is root. Preorder[1] is left root. Find in postorder to get left subtree size. Split and recurse.

```python
def constructFromPrePost(preorder, postorder):
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    if len(preorder) == 1:
        return root
    idx = postorder.index(preorder[1])
    n = idx + 1
    root.left = constructFromPrePost(preorder[1:1+n], postorder[:n])
    root.right = constructFromPrePost(preorder[1+n:], postorder[n:-1])
    return root
```

Time: O(n) | Space: O(n)

---

## 12. Flatten Binary Tree to Linked List

Flatten to right-only linked list in preorder. Recursive. Flatten left and right. Set root.right = flattened left, find tail, tail.right = flattened right, root.left = None.

```python
def flatten(root):
    if not root:
        return
    flatten(root.left)
    flatten(root.right)
    right = root.right
    root.right = root.left
    root.left = None
    cur = root
    while cur.right:
        cur = cur.right
    cur.right = right
```

Time: O(n) | Space: O(h)

---

## 13. Populating Next Right Pointers in Each Node

Connect each node to its next right at same level. BFS with level. Or use next pointers: for each level, link children. Parent has next, so parent.next.left is right sibling's left.

```python
def connect(root):
    if not root:
        return None
    leftmost = root
    while leftmost.left:
        head = leftmost
        while head:
            head.left.next = head.right
            if head.next:
                head.right.next = head.next.left
            head = head.next
        leftmost = leftmost.left
    return root
```

Time: O(n) | Space: O(1)

---

## 14. Path Sum II

Find all root-to-leaf paths with given sum. DFS with path list. At leaf, if sum matches, append path copy to result. Backtrack (pop) after recurse.

```python
def pathSum(root, targetSum):
    res = []
    def dfs(node, path, rem):
        if not node:
            return
        path.append(node.val)
        if not node.left and not node.right and rem == node.val:
            res.append(path[:])
        dfs(node.left, path, rem - node.val)
        dfs(node.right, path, rem - node.val)
        path.pop()
    dfs(root, [], targetSum)
    return res
```

Time: O(n) | Space: O(h)

---

## 15. Path Sum III

Count paths (any downward) that sum to target. Prefix sum + hash map. At each node, count how many prefix_sum - target exist. Update map, recurse, backtrack.

```python
def pathSum(root, targetSum):
    from collections import defaultdict
    pre = defaultdict(int)
    pre[0] = 1
    count = 0

    def dfs(node, cur):
        nonlocal count
        if not node:
            return
        cur += node.val
        count += pre.get(cur - targetSum, 0)
        pre[cur] += 1
        dfs(node.left, cur)
        dfs(node.right, cur)
        pre[cur] -= 1

    dfs(root, 0)
    return count
```

Time: O(n) | Space: O(n)

---

## 16. Binary Search Tree Iterator

Iterator for inorder traversal with next() and hasNext(). Stack. Push left spine. next() pops, pushes left spine of right child. hasNext() = stack non-empty.

```python
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self):
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val

    def hasNext(self):
        return len(self.stack) > 0
```

Time: O(1) amortized | Space: O(h)

---

## 17. Count Good Nodes in Binary Tree

Count nodes where path from root has no value greater than node. DFS with max_so_far. If node.val >= max, count++. Recurse with updated max.

```python
def goodNodes(root):
    def dfs(node, mx):
        if not node:
            return 0
        cnt = 1 if node.val >= mx else 0
        mx = max(mx, node.val)
        return cnt + dfs(node.left, mx) + dfs(node.right, mx)
    return dfs(root, float('-inf'))
```

Time: O(n) | Space: O(h)

---

## 18. Delete Node in BST

Delete node with given value from BST. Find node. If leaf, remove. If one child, replace with child. If two children, replace with inorder successor, delete successor.

```python
def deleteNode(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        succ = root.right
        while succ.left:
            succ = succ.left
        root.val = succ.val
        root.right = deleteNode(root.right, succ.val)
    return root
```

Time: O(h) | Space: O(h)

---

## 19. Trim BST

Remove nodes outside [low, high] range. If root < low, return trim(root.right). If root > high, return trim(root.left). Else root.left = trim(left), root.right = trim(right).

```python
def trimBST(root, low, high):
    if not root:
        return None
    if root.val < low:
        return trimBST(root.right, low, high)
    if root.val > high:
        return trimBST(root.left, low, high)
    root.left = trimBST(root.left, low, high)
    root.right = trimBST(root.right, low, high)
    return root
```

Time: O(n) | Space: O(h)

---

## 20. Unique Binary Search Trees II

Generate all structurally unique BSTs with n nodes (values 1 to n). For each i as root, left = generate(1, i-1), right = generate(i+1, n). Combine all pairs.

```python
def generateTrees(n):
    def gen(lo, hi):
        if lo > hi:
            return [None]
        res = []
        for i in range(lo, hi + 1):
            for left in gen(lo, i - 1):
                for right in gen(i + 1, hi):
                    root = TreeNode(i)
                    root.left = left
                    root.right = right
                    res.append(root)
        return res
    return gen(1, n) if n else []
```

Time: O(Catalan) | Space: O(Catalan)

---

## 21. Binary Tree Vertical Order Traversal

Traverse by vertical columns (left to right). BFS/DFS with column index. Map column to list of (row, val). Sort by row for same column. Return sorted by column.

```python
def verticalOrder(root):
    if not root:
        return []
    from collections import defaultdict, deque
    col_map = defaultdict(list)
    q = deque([(root, 0)])
    while q:
        node, col = q.popleft()
        col_map[col].append(node.val)
        if node.left:
            q.append((node.left, col - 1))
        if node.right:
            q.append((node.right, col + 1))
    return [col_map[c] for c in sorted(col_map)]
```

Time: O(n log n) | Space: O(n)

---

## 22. House Robber III

Max money from nodes, no two adjacent nodes. DP. (rob_this, skip_this). Rob = val + left_skip + right_skip. Skip = max(left) + max(right).

```python
def rob(root):
    def dfs(node):
        if not node:
            return (0, 0)
        left_rob, left_skip = dfs(node.left)
        right_rob, right_skip = dfs(node.right)
        rob_this = node.val + left_skip + right_skip
        skip_this = max(left_rob, left_skip) + max(right_rob, right_skip)
        return (rob_this, skip_this)
    return max(dfs(root))
```

Time: O(n) | Space: O(h)

---

## 23. Binary Tree Cameras

Minimum cameras to cover all nodes (camera covers parent, self, children). Greedy/DP. State: 0=covered, 1=has camera, 2=needs cover. Leaf returns 2. If child 2, place camera. If child 1, current covered.

```python
def minCameraCover(root):
    cameras = 0
    def dfs(node):
        nonlocal cameras
        if not node:
            return 0
        l, r = dfs(node.left), dfs(node.right)
        if l == 2 or r == 2:
            cameras += 1
            return 1
        if l == 1 or r == 1:
            return 0
        return 2
    return (cameras + 1) if dfs(root) == 2 else cameras
```

Time: O(n) | Space: O(h)

---

## 24. Distribute Coins in Binary Tree

Minimum moves so every node has exactly 1 coin. DFS returns excess (coins - 1). Moves += |left_excess| + |right_excess|. Return node.val + left + right - 1.

```python
def distributeCoins(root):
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

Time: O(n) | Space: O(h)

---

## 25. All Nodes Distance K in Binary Tree

Return values of nodes at distance k from target. Build parent map. BFS from target to k levels (include parent, left, right). Visited set.

```python
def distanceK(root, target, k):
    parent = {}
    def build_parent(node, p):
        if not node:
            return
        parent[node] = p
        build_parent(node.left, node)
        build_parent(node.right, node)
    build_parent(root, None)

    from collections import deque
    q = deque([(target, 0)])
    vis = {target}
    res = []
    while q:
        node, d = q.popleft()
        if d == k:
            res.append(node.val)
        elif d < k:
            for nxt in [node.left, node.right, parent.get(node)]:
                if nxt and nxt not in vis:
                    vis.add(nxt)
                    q.append((nxt, d + 1))
    return res
```

Time: O(n) | Space: O(n)

---

## Hard Problems

## 1. Serialize and Deserialize Binary Tree

Convert tree to string and back. Preorder with "null" for missing. Comma-separated. Deserialize by consuming tokens.

```python
class Codec:
    def serialize(self, root):
        def enc(node):
            if not node:
                return "null"
            return str(node.val) + "," + enc(node.left) + "," + enc(node.right)
        return enc(root)

    def deserialize(self, data):
        vals = iter(data.split(","))
        def dec():
            v = next(vals)
            if v == "null":
                return None
            node = TreeNode(int(v))
            node.left = dec()
            node.right = dec()
            return node
        return dec()
```

Time: O(n) | Space: O(n)

---

## 2. Binary Tree Maximum Path Sum (Hard)

Same as medium but often classified hard in some platforms. See medium #6.

```python
def maxPathSum(root):
    best = float('-inf')
    def gain(node):
        nonlocal best
        if not node:
            return 0
        l = max(0, gain(node.left))
        r = max(0, gain(node.right))
        best = max(best, node.val + l + r)
        return node.val + max(l, r)
    gain(root)
    return best
```

Time: O(n) | Space: O(h)

---

## 3. Count Complete Tree Nodes

Count nodes in complete binary tree in O(log^2 n) or better. Compute left and right heights from root. If equal, full tree 2^h - 1. Else recurse on left and right subtrees.

```python
def countNodes(root):
    def height(node, go_left):
        h = 0
        while node:
            h += 1
            node = node.left if go_left else node.right
        return h

    if not root:
        return 0
    lh = height(root.left, True)
    rh = height(root.right, False)
    if lh == rh:
        return (1 << lh) - 1
    return 1 + countNodes(root.left) + countNodes(root.right)
```

Time: O(log^2 n) | Space: O(log n)

---

## 4. Recover Binary Search Tree

Two nodes are swapped. Recover without changing structure. Inorder find two inversions. First: prev > curr. Second: next prev > curr. Swap first's prev with second's curr.

```python
def recoverTree(root):
    first = second = prev = None

    def inorder(node):
        nonlocal first, second, prev
        if not node:
            return
        inorder(node.left)
        if prev and prev.val > node.val:
            if not first:
                first, second = prev, node
            else:
                second = node
        prev = node
        inorder(node.right)

    inorder(root)
    first.val, second.val = second.val, first.val
```

Time: O(n) | Space: O(h)

---

## 5. Binary Tree Postorder Traversal (Iterative One Stack)

Postorder with single stack. Reverse preorder (root-right-left) gives postorder reversed. Or use stack with peek to detect when to process.

```python
def postorderTraversal(root):
    if not root:
        return []
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return res[::-1]
```

Time: O(n) | Space: O(n)

---

## 6. Sum of Distances in Tree

For each node, sum of distances to all other nodes. Re-rooting. First DFS: subtree sizes, sum of distances from root. Second DFS: when moving root to child, new_sum = old - size[child] + (n - size[child]).

```python
def sumOfDistancesInTree(n, edges):
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    size = [1] * n
    res = [0] * n

    def dfs1(node, parent):
        for child in graph[node]:
            if child != parent:
                dfs1(child, node)
                size[node] += size[child]
                res[node] += res[child] + size[child]

    def dfs2(node, parent):
        for child in graph[node]:
            if child != parent:
                res[child] = res[node] - size[child] + (n - size[child])
                dfs2(child, node)

    dfs1(0, -1)
    dfs2(0, -1)
    return res
```

Time: O(n) | Space: O(n)

---

## 7. Number of Good Leaf Node Pairs

Count pairs of leaves with distance <= d. At each node, get list of leaf distances. For each (left_d, right_d) with left_d + right_d <= d, add to count. Return distances + 1.

```python
def countPairs(root, distance):
    count = 0
    def dfs(node):
        nonlocal count
        if not node:
            return []
        if not node.left and not node.right:
            return [1]
        left = [d + 1 for d in dfs(node.left)]
        right = [d + 1 for d in dfs(node.right)]
        for ld in left:
            for rd in right:
                if ld + rd <= distance:
                    count += 1
        return left + right
    dfs(root)
    return count
```

Time: O(n * leaves^2) | Space: O(h)

---

## 8. Minimum Cost Tree from Leaf Values

Build tree from leaf array. Cost of node = max(left_leaves) * max(right_leaves). Minimize total cost. DP. dp[i][j] = min cost for leaves i..j. Try all splits. Or greedy: repeatedly pick min and merge with smaller neighbor (stack).

```python
def mctFromLeafValues(arr):
    stack = [float('inf')]
    res = 0
    for x in arr:
        while stack[-1] <= x:
            mid = stack.pop()
            res += mid * min(stack[-1], x)
        stack.append(x)
    while len(stack) > 2:
        res += stack.pop() * stack[-1]
    return res
```

Time: O(n) | Space: O(n)

---

## 9. Count of Smaller Numbers After Self

For each element, count elements to the right that are smaller. Merge sort (inversion count) or coordinate compression + BIT/Segment tree. Process from right, query count in [min, num-1], update at num.

```python
def countSmaller(nums):
    import bisect
    sorted_nums = []
    res = []
    for x in reversed(nums):
        i = bisect.bisect_left(sorted_nums, x)
        res.append(i)
        bisect.insort(sorted_nums, x)
    return res[::-1]
```

Time: O(n log n) | Space: O(n)

---

## 10. Longest Increasing Path in Matrix

Not tree but similar DFS/DP structure. DFS with memo. At each cell, try 4 directions. Memo[i][j] = 1 + max(valid neighbors).

```python
def longestIncreasingPath(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]
        best = 1
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        memo[(r, c)] = best
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))
```

Time: O(m * n) | Space: O(m * n)

---

## 11. Binary Tree Cameras (Alternative Formulations)

Variants of camera placement. See medium #23.

```python
def minCameraCover(root):
    cameras = 0
    def dfs(node):
        nonlocal cameras
        if not node:
            return 0
        l, r = dfs(node.left), dfs(node.right)
        if l == 2 or r == 2:
            cameras += 1
            return 1
        if l == 1 or r == 1:
            return 0
        return 2
    return (cameras + 1) if dfs(root) == 2 else cameras
```

Time: O(n) | Space: O(h)

---

## 12. Redundant Connection II

Directed graph from tree + one edge. Find edge to remove to get valid rooted tree. Union-Find. Case 1: node has two parents. Case 2: cycle. Handle both.

```python
def findRedundantDirectedConnection(edges):
    n = len(edges)
    in_deg = {}
    cand1 = cand2 = None
    for u, v in edges:
        if v in in_deg:
            cand1 = [in_deg[v], v]
            cand2 = [u, v]
        in_deg[v] = u

    def find(x, p):
        if p[x] != x:
            p[x] = find(p[x], p)
        return p[x]

    def has_cycle(skip):
        p = list(range(n + 1))
        for i, (u, v) in enumerate(edges):
            if i == skip:
                continue
            if find(u, p) == find(v, p):
                return True
            p[find(u, p)] = find(v, p)
        return False

    if cand2 is not None:
        skip = next(i for i, e in enumerate(edges) if e == cand2)
        return cand2 if not has_cycle(skip) else cand1

    p = list(range(n + 1))
    for u, v in edges:
        if find(u, p) == find(v, p):
            return [u, v]
        p[find(u, p)] = find(v, p)
    return []
```

Time: O(n) | Space: O(n)

---

## 13. Binary Tree Maximum Path Sum (Follow-up)

Return the path itself, not just sum. Track path in DFS. Return (max_sum, path). Combine paths at each node.

```python
def maxPathSumWithPath(root):
    best_sum = float('-inf')
    best_path = []

    def dfs(node):
        nonlocal best_sum, best_path
        if not node:
            return (0, [])
        l_sum, l_path = dfs(node.left)
        r_sum, r_path = dfs(node.right)
        l_gain = max(0, l_sum)
        r_gain = max(0, r_sum)
        path_sum = node.val + l_gain + r_gain
        if path_sum > best_sum:
            best_sum = path_sum
            left_part = l_path if l_gain > 0 else []
            right_part = r_path if r_gain > 0 else []
            best_path = left_part[::-1] + [node.val] + right_part
        ret_sum = node.val + max(l_gain, r_gain)
        ret_path = ([node.val] + l_path) if l_gain >= r_gain else ([node.val] + r_path)
        return (ret_sum, ret_path)

    dfs(root)
    return best_path
```

Time: O(n) | Space: O(h)

---

## 14. Count Univalue Subtrees

Count subtrees where all values are same. DFS returns (is_univalue, value). Univalue if left and right univalue and match node value.

```python
def countUnivalSubtrees(root):
    count = 0
    def dfs(node):
        nonlocal count
        if not node:
            return True, None
        l_uni, l_val = dfs(node.left)
        r_uni, r_val = dfs(node.right)
        ok = l_uni and r_uni
        if node.left and l_val is not None and l_val != node.val:
            ok = False
        if node.right and r_val is not None and r_val != node.val:
            ok = False
        if ok:
            count += 1
        return ok, node.val
    dfs(root)
    return count
```

Time: O(n) | Space: O(h)

---

## 15. Closest Leaf in Binary Tree

Find nearest leaf to given node k. Build graph (node to neighbors). BFS from k to find nearest leaf.

```python
def findClosestLeaf(root, k):
    from collections import defaultdict, deque
    graph = defaultdict(list)
    leaves = set()

    def build(node, parent):
        if not node:
            return
        if parent is not None:
            graph[node.val].append(parent.val)
        if node.left:
            graph[node.val].append(node.left.val)
            build(node.left, node)
        if node.right:
            graph[node.val].append(node.right.val)
            build(node.right, node)
        if not node.left and not node.right:
            leaves.add(node.val)

    build(root, None)
    q = deque([k])
    vis = {k}
    while q:
        cur = q.popleft()
        if cur in leaves:
            return cur
        for nxt in graph[cur]:
            if nxt not in vis:
                vis.add(nxt)
                q.append(nxt)
    return -1
```

Time: O(n) | Space: O(n)
