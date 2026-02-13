# Tree - Basic Operations

## TreeNode Class

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## Insert into BST

### Iterative

```python
def insert_bst_iterative(root, val):
    node = TreeNode(val)
    if not root:
        return node
    curr = root
    while True:
        if val < curr.val:
            if curr.left is None:
                curr.left = node
                break
            curr = curr.left
        else:
            if curr.right is None:
                curr.right = node
                break
            curr = curr.right
    return root
```

### Recursive

```python
def insert_bst_recursive(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst_recursive(root.left, val)
    else:
        root.right = insert_bst_recursive(root.right, val)
    return root
```

## Search in BST

### Iterative

```python
def search_bst_iterative(root, val):
    curr = root
    while curr:
        if curr.val == val:
            return curr
        if val < curr.val:
            curr = curr.left
        else:
            curr = curr.right
    return None
```

### Recursive

```python
def search_bst_recursive(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search_bst_recursive(root.left, val)
    return search_bst_recursive(root.right, val)
```

## Delete from BST

Three cases: leaf node, one child, two children (replace with inorder successor).

```python
def delete_bst(root, val):
    if not root:
        return None
    if val < root.val:
        root.left = delete_bst(root.left, val)
    elif val > root.val:
        root.right = delete_bst(root.right, val)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        succ = find_min(root.right)
        root.val = succ.val
        root.right = delete_bst(root.right, succ.val)
    return root
```

## Find Minimum (Leftmost)

```python
def find_min(root):
    if not root:
        return None
    while root.left:
        root = root.left
    return root
```

## Find Maximum

```python
def find_max(root):
    if not root:
        return None
    while root.right:
        root = root.right
    return root
```

## Find Height (Recursive)

```python
def find_height(root):
    if not root:
        return -1
    return 1 + max(find_height(root.left), find_height(root.right))
```

## Count Total Nodes

```python
def count_nodes(root):
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)
```

## Count Leaf Nodes

```python
def count_leaf_nodes(root):
    if not root:
        return 0
    if not root.left and not root.right:
        return 1
    return count_leaf_nodes(root.left) + count_leaf_nodes(root.right)
```

## Check Empty

```python
def is_empty(root):
    return root is None
```

## Inorder Traversal (Recursive)

Left - Root - Right. For BST, yields sorted order.

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

## Preorder Traversal (Recursive)

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

## Postorder Traversal (Recursive)

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

## Level-Order Traversal (BFS with Queue)

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
