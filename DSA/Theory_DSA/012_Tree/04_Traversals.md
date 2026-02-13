# Tree - Traversals

## Inorder Recursive

Left - Root - Right. For BST yields sorted order.

```python
def inorder_recursive(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)
    dfs(root)
    return result
```

## Inorder Iterative (Stack)

```python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result
```

## Morris Inorder (O(1) Space)

Uses threaded binary tree. For current node, find its inorder predecessor. If predecessor's right is null, link it to current and go left. Else unlink and process current, go right.

```python
def morris_inorder(root):
    result = []
    curr = root
    while curr:
        if not curr.left:
            result.append(curr.val)
            curr = curr.right
        else:
            pred = curr.left
            while pred.right and pred.right != curr:
                pred = pred.right
            if not pred.right:
                pred.right = curr
                curr = curr.left
            else:
                pred.right = None
                result.append(curr.val)
                curr = curr.right
    return result
```

## Preorder Recursive

Root - Left - Right.

```python
def preorder_recursive(root):
    result = []
    def dfs(node):
        if not node:
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return result
```

## Preorder Iterative (Stack)

```python
def preorder_iterative(root):
    if not root:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

## Morris Preorder

```python
def morris_preorder(root):
    result = []
    curr = root
    while curr:
        if not curr.left:
            result.append(curr.val)
            curr = curr.right
        else:
            pred = curr.left
            while pred.right and pred.right != curr:
                pred = pred.right
            if not pred.right:
                result.append(curr.val)
                pred.right = curr
                curr = curr.left
            else:
                pred.right = None
                curr = curr.right
    return result
```

## Postorder Recursive

Left - Right - Root.

```python
def postorder_recursive(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        dfs(node.right)
        result.append(node.val)
    dfs(root)
    return result
```

## Postorder Iterative (Two Stacks)

```python
def postorder_two_stacks(root):
    if not root:
        return []
    s1 = [root]
    s2 = []
    while s1:
        node = s1.pop()
        s2.append(node)
        if node.left:
            s1.append(node.left)
        if node.right:
            s1.append(node.right)
    return [n.val for n in reversed(s2)]
```

## Postorder Iterative (One Stack)

```python
def postorder_one_stack(root):
    if not root:
        return []
    result = []
    stack = []
    curr = root
    while curr or stack:
        while curr:
            if curr.right:
                stack.append(curr.right)
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        if stack and curr.right == stack[-1]:
            stack.pop()
            stack.append(curr)
            curr = curr.right
        else:
            result.append(curr.val)
            curr = None
    return result
```

## Level-Order (BFS)

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        node = q.popleft()
        result.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return result
```

## Level-Order Bottom-Up

```python
from collections import deque

def level_order_bottom_up(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result[::-1]
```

## Zigzag Level-Order

```python
from collections import deque

def zigzag_level_order(root):
    if not root:
        return []
    result = []
    q = deque([root])
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
        result.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return result
```

## Reverse Level-Order

```python
from collections import deque

def reverse_level_order(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        node = q.popleft()
        result.append(node.val)
        if node.right:
            q.append(node.right)
        if node.left:
            q.append(node.left)
    return result[::-1]
```

## Vertical Order

```python
from collections import defaultdict, deque

def vertical_order(root):
    if not root:
        return []
    cols = defaultdict(list)
    q = deque([(root, 0)])
    while q:
        node, col = q.popleft()
        cols[col].append(node.val)
        if node.left:
            q.append((node.left, col - 1))
        if node.right:
            q.append((node.right, col + 1))
    return [cols[c] for c in sorted(cols.keys())]
```

## Diagonal Order

```python
from collections import defaultdict

def diagonal_order(root):
    if not root:
        return []
    diag = defaultdict(list)
    def dfs(node, d):
        if not node:
            return
        diag[d].append(node.val)
        dfs(node.left, d + 1)
        dfs(node.right, d)
    dfs(root, 0)
    return [diag[i] for i in sorted(diag.keys())]
```

## Boundary Traversal

Left boundary (top-down) + leaves (left to right) + right boundary (bottom-up).

```python
def boundary_traversal(root):
    if not root:
        return []
    result = [root.val]
    def left_boundary(node):
        if not node or (not node.left and not node.right):
            return
        result.append(node.val)
        left_boundary(node.left if node.left else node.right)
    def leaves(node):
        if not node:
            return
        if not node.left and not node.right:
            result.append(node.val)
            return
        leaves(node.left)
        leaves(node.right)
    def right_boundary(node):
        if not node or (not node.left and not node.right):
            return
        right_boundary(node.right if node.right else node.left)
        result.append(node.val)
    left_boundary(root.left)
    leaves(root.left)
    leaves(root.right)
    right_boundary(root.right)
    return result
```

## Right Side View

```python
from collections import deque

def right_side_view(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        result.append(q[-1].val)
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return result
```

## Left Side View

```python
from collections import deque

def left_side_view(root):
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        result.append(q[0].val)
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return result
```

## Top View

```python
from collections import defaultdict, deque

def top_view(root):
    if not root:
        return []
    cols = defaultdict(lambda: (float('inf'), 0))
    q = deque([(root, 0, 0)])
    while q:
        node, col, row = q.popleft()
        if row < cols[col][0]:
            cols[col] = (row, node.val)
        if node.left:
            q.append((node.left, col - 1, row + 1))
        if node.right:
            q.append((node.right, col + 1, row + 1))
    return [cols[c][1] for c in sorted(cols.keys())]
```

## Bottom View

```python
from collections import defaultdict, deque

def bottom_view(root):
    if not root:
        return []
    cols = defaultdict(lambda: (float('-inf'), 0))
    q = deque([(root, 0, 0)])
    while q:
        node, col, row = q.popleft()
        if row >= cols[col][0]:
            cols[col] = (row, node.val)
        if node.left:
            q.append((node.left, col - 1, row + 1))
        if node.right:
            q.append((node.right, col + 1, row + 1))
    return [cols[c][1] for c in sorted(cols.keys())]
```

## Nodes Between Two Levels

```python
from collections import deque

def nodes_between_levels(root, low, high):
    if not root:
        return []
    result = []
    q = deque([root])
    level = 0
    while q and level <= high:
        for _ in range(len(q)):
            node = q.popleft()
            if low <= level <= high:
                result.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level += 1
    return result
```

## All Nodes at Given Level

```python
from collections import deque

def nodes_at_level(root, k):
    if not root:
        return []
    q = deque([root])
    level = 0
    while q:
        if level == k:
            return [n.val for n in q]
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level += 1
    return []
```

## Ancestors of a Node

```python
def ancestors_of_node(root, target):
    result = []
    def dfs(node, path):
        if not node:
            return False
        path.append(node.val)
        if node.val == target:
            result.extend(path[:-1])
            return True
        if dfs(node.left, path) or dfs(node.right, path):
            return True
        path.pop()
        return False
    dfs(root, [])
    return result
```

## Cousins of a Node

Nodes at same level with different parents.

```python
from collections import deque

def cousins_of_node(root, target_val):
    if not root or root.val == target_val:
        return []
    target_level = None
    target_parent = None
    q = deque([(root, 0, None)])
    while q:
        node, level, parent = q.popleft()
        if node.val == target_val:
            target_level = level
            target_parent = parent
            break
        if node.left:
            q.append((node.left, level + 1, node))
        if node.right:
            q.append((node.right, level + 1, node))
    if target_level is None:
        return []
    result = []
    q = deque([(root, 0, None)])
    while q:
        node, level, parent = q.popleft()
        if level == target_level and parent != target_parent:
            result.append(node.val)
        if node.left:
            q.append((node.left, level + 1, node))
        if node.right:
            q.append((node.right, level + 1, node))
    return result
```

## Serialize Tree (Preorder with Nulls)

```python
def serialize(root):
    result = []
    def preorder(node):
        if not node:
            result.append('null')
            return
        result.append(str(node.val))
        preorder(node.left)
        preorder(node.right)
    preorder(root)
    return ','.join(result)
```

## Deserialize Tree

```python
def deserialize(data):
    if not data:
        return None
    vals = iter(data.split(','))
    def build():
        val = next(vals)
        if val == 'null':
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    return build()
```
