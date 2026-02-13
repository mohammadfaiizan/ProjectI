# Tree Construction

## Build from Inorder + Preorder

Preorder gives root. Find root in inorder to split left/right subtrees.

```python
def build_from_inorder_preorder(inorder, preorder):
    if not inorder or not preorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    idx = inorder.index(root_val)
    root.left = build_from_inorder_preorder(inorder[:idx], preorder[1:idx+1])
    root.right = build_from_inorder_preorder(inorder[idx+1:], preorder[idx+1:])
    return root
```

Optimized with hash map for O(n):

```python
def build_from_inorder_preorder_optimized(inorder, preorder):
    idx_map = {v: i for i, v in enumerate(inorder)}
    def build(io_left, io_right, po_left, po_right):
        if io_left > io_right:
            return None
        root_val = preorder[po_left]
        root = TreeNode(root_val)
        idx = idx_map[root_val]
        left_size = idx - io_left
        root.left = build(io_left, idx - 1, po_left + 1, po_left + left_size)
        root.right = build(idx + 1, io_right, po_left + left_size + 1, po_right)
        return root
    return build(0, len(inorder) - 1, 0, len(preorder) - 1)
```

## Build from Inorder + Postorder

Postorder gives root (last element). Find root in inorder to split.

```python
def build_from_inorder_postorder(inorder, postorder):
    if not inorder or not postorder:
        return None
    root_val = postorder[-1]
    root = TreeNode(root_val)
    idx = inorder.index(root_val)
    root.left = build_from_inorder_postorder(inorder[:idx], postorder[:idx])
    root.right = build_from_inorder_postorder(inorder[idx+1:], postorder[idx:-1])
    return root
```

## Build from Preorder + Postorder (Full Binary Tree)

For full binary tree, preorder[0] is root, preorder[1] is left subtree root. Find it in postorder to split.

```python
def build_from_preorder_postorder(preorder, postorder):
    if not preorder or not postorder:
        return None
    root = TreeNode(preorder[0])
    if len(preorder) == 1:
        return root
    left_root_val = preorder[1]
    left_size = postorder.index(left_root_val) + 1
    root.left = build_from_preorder_postorder(preorder[1:1+left_size], postorder[:left_size])
    root.right = build_from_preorder_postorder(preorder[1+left_size:], postorder[left_size:-1])
    return root
```

## Build BST from Preorder

First element is root. Split remaining into smaller and larger.

```python
def bst_from_preorder(preorder):
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    i = 1
    while i < len(preorder) and preorder[i] < root.val:
        i += 1
    root.left = bst_from_preorder(preorder[1:i])
    root.right = bst_from_preorder(preorder[i:])
    return root
```

Using upper bound (O(n)):

```python
def bst_from_preorder_optimized(preorder):
    idx = [0]
    def build(upper):
        if idx[0] >= len(preorder) or preorder[idx[0]] > upper:
            return None
        root = TreeNode(preorder[idx[0]])
        idx[0] += 1
        root.left = build(root.val)
        root.right = build(upper)
        return root
    return build(float('inf'))
```

## Build BST from Postorder

Last element is root. Split from right.

```python
def bst_from_postorder(postorder):
    if not postorder:
        return None
    root_val = postorder[-1]
    root = TreeNode(root_val)
    i = len(postorder) - 2
    while i >= 0 and postorder[i] > root_val:
        i -= 1
    root.left = bst_from_postorder(postorder[:i+1])
    root.right = bst_from_postorder(postorder[i+1:-1])
    return root
```

## Build BST from Level-Order

Insert elements one by one in level order.

```python
def bst_from_level_order(level_order):
    if not level_order:
        return None
    root = TreeNode(level_order[0])
    for val in level_order[1:]:
        curr = root
        while True:
            if val < curr.val:
                if curr.left is None:
                    curr.left = TreeNode(val)
                    break
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = TreeNode(val)
                    break
                curr = curr.right
    return root
```

## Build Complete Binary Tree from Level-Order

For complete tree, parent of index i is (i-1)//2, left child 2*i+1, right 2*i+2.

```python
def complete_tree_from_level_order(arr):
    if not arr:
        return None
    n = len(arr)
    nodes = [TreeNode(arr[i]) for i in range(n)]
    for i in range(n):
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2
        if left_idx < n:
            nodes[i].left = nodes[left_idx]
        if right_idx < n:
            nodes[i].right = nodes[right_idx]
    return nodes[0]
```

## Serialize and Deserialize (Preorder)

```python
def serialize_preorder(root):
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

def deserialize_preorder(data):
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

## Serialize and Deserialize (Level-Order)

```python
from collections import deque

def serialize_level_order(root):
    if not root:
        return ''
    result = []
    q = deque([root])
    while q:
        node = q.popleft()
        result.append(str(node.val) if node else 'null')
        if node:
            q.append(node.left)
            q.append(node.right)
    while result and result[-1] == 'null':
        result.pop()
    return ','.join(result)

def deserialize_level_order(data):
    if not data:
        return None
    vals = data.split(',')
    root = TreeNode(int(vals[0]))
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i]))
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i]))
            q.append(node.right)
        i += 1
    return root
```

## Construct Maximum Binary Tree

Root is max element. Left subtree from elements before max, right from elements after.

```python
def construct_maximum_binary_tree(nums):
    if not nums:
        return None
    max_idx = nums.index(max(nums))
    root = TreeNode(nums[max_idx])
    root.left = construct_maximum_binary_tree(nums[:max_idx])
    root.right = construct_maximum_binary_tree(nums[max_idx+1:])
    return root
```

## Construct from String with Parenthesis

Format: "4(2(3)(1))(6(5))" - value followed by (left)(right).

```python
def str2tree(s):
    if not s:
        return None
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == '-'):
        i += 1
    val = int(s[:i])
    root = TreeNode(val)
    if i >= len(s):
        return root
    start = i
    count = 0
    for j in range(i, len(s)):
        if s[j] == '(':
            count += 1
        elif s[j] == ')':
            count -= 1
        if count == 0:
            break
    root.left = str2tree(s[i+1:j]) if j > i else None
    root.right = str2tree(s[j+2:-1]) if j + 2 < len(s) - 1 else None
    return root
```

Improved version:

```python
def str2tree(s):
    def parse(i):
        if i >= len(s):
            return None, i
        neg = 1
        if s[i] == '-':
            neg = -1
            i += 1
        num = 0
        while i < len(s) and s[i].isdigit():
            num = num * 10 + int(s[i])
            i += 1
        root = TreeNode(neg * num)
        if i < len(s) and s[i] == '(':
            i += 1
            root.left, i = parse(i)
            i += 1
        if i < len(s) and s[i] == '(':
            i += 1
            root.right, i = parse(i)
            i += 1
        return root, i
    root, _ = parse(0)
    return root
```
