# Balanced Trees

## AVL Tree

### Balance Factor

BF(node) = height(left) - height(right). Valid range: -1, 0, 1.

### Rotations

- **LL (Right rotation)**: Left-left case. Single right rotation on unbalanced node.
- **RR (Left rotation)**: Right-right case. Single left rotation on unbalanced node.
- **LR (Left-Right)**: Left subtree's right child. Left rotate on left child, then right rotate on node.
- **RL (Right-Left)**: Right subtree's left child. Right rotate on right child, then left rotate on node.

### Insert with Rebalancing

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

def get_height(node):
    return node.height if node else 0

def get_balance(node):
    return get_height(node.left) - get_height(node.right) if node else 0

def right_rotate(z):
    y = z.left
    T3 = y.right
    y.right = z
    z.left = T3
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y

def left_rotate(z):
    y = z.right
    T2 = y.left
    y.left = z
    z.right = T2
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y

def avl_insert(node, val):
    if not node:
        return AVLNode(val)
    if val < node.val:
        node.left = avl_insert(node.left, val)
    elif val > node.val:
        node.right = avl_insert(node.right, val)
    else:
        return node
    node.height = 1 + max(get_height(node.left), get_height(node.right))
    balance = get_balance(node)
    if balance > 1 and val < node.left.val:
        return right_rotate(node)
    if balance < -1 and val > node.right.val:
        return left_rotate(node)
    if balance > 1 and val > node.left.val:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    if balance < -1 and val < node.right.val:
        node.right = right_rotate(node.right)
        return left_rotate(node)
    return node
```

### Delete with Rebalancing

```python
def avl_min_node(node):
    while node.left:
        node = node.left
    return node

def avl_delete(node, val):
    if not node:
        return node
    if val < node.val:
        node.left = avl_delete(node.left, val)
    elif val > node.val:
        node.right = avl_delete(node.right, val)
    else:
        if not node.left:
            return node.right
        if not node.right:
            return node.left
        temp = avl_min_node(node.right)
        node.val = temp.val
        node.right = avl_delete(node.right, temp.val)
    node.height = 1 + max(get_height(node.left), get_height(node.right))
    balance = get_balance(node)
    if balance > 1 and get_balance(node.left) >= 0:
        return right_rotate(node)
    if balance > 1 and get_balance(node.left) < 0:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    if balance < -1 and get_balance(node.right) <= 0:
        return left_rotate(node)
    if balance < -1 and get_balance(node.right) > 0:
        node.right = right_rotate(node.right)
        return left_rotate(node)
    return node
```

### Height Guarantee

AVL tree height is O(log n). At most 1.44 * log2(n+2) - 1.29.

## Red-Black Tree

### Five Properties

1. Every node is red or black
2. Root is black
3. All leaves (NIL) are black
4. Red node has black children
5. Every path from node to descendant leaf has same number of black nodes

### Insert Rules

New node is red. Recolor and rotate to fix violations. Cases: uncle red (recolor), uncle black (rotate).

```python
RED = True
BLACK = False

class RBNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
        self.color = RED

def rb_insert(root, val):
    node = RBNode(val)
    if not root:
        node.color = BLACK
        return node
    curr = root
    parent = None
    while curr:
        parent = curr
        curr = curr.left if val < curr.val else curr.right
    node.parent = parent
    if val < parent.val:
        parent.left = node
    else:
        parent.right = node
    return rb_insert_fix(root, node)

def rb_insert_fix(root, node):
    while node.parent and node.parent.color == RED:
        if node.parent == node.parent.parent.left:
            uncle = node.parent.parent.right
            if uncle and uncle.color == RED:
                node.parent.color = BLACK
                uncle.color = BLACK
                node.parent.parent.color = RED
                node = node.parent.parent
            else:
                if node == node.parent.right:
                    node = node.parent
                    root = rb_left_rotate(root, node)
                node.parent.color = BLACK
                node.parent.parent.color = RED
                root = rb_right_rotate(root, node.parent.parent)
        else:
            uncle = node.parent.parent.left
            if uncle and uncle.color == RED:
                node.parent.color = BLACK
                uncle.color = BLACK
                node.parent.parent.color = RED
                node = node.parent.parent
            else:
                if node == node.parent.left:
                    node = node.parent
                    root = rb_right_rotate(root, node)
                node.parent.color = BLACK
                node.parent.parent.color = RED
                root = rb_left_rotate(root, node.parent.parent)
    root.color = BLACK
    return root

def rb_left_rotate(root, x):
    y = x.right
    x.right = y.left
    if y.left:
        y.left.parent = x
    y.parent = x.parent
    if not x.parent:
        root = y
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y
    y.left = x
    x.parent = y
    return root

def rb_right_rotate(root, x):
    y = x.left
    x.left = y.right
    if y.right:
        y.right.parent = x
    y.parent = x.parent
    if not x.parent:
        root = y
    elif x == x.parent.right:
        x.parent.right = y
    else:
        x.parent.left = y
    y.right = x
    x.parent = y
    return root
```

### Delete Overview

Replace node with successor, fix double-black. Cases: sibling red, sibling black with black children, sibling black with red child.

### O(log n) Guarantee

Black height is at least h/2. Path length at most 2*log2(n+1).

## Comparison: AVL vs Red-Black

| Aspect | AVL | Red-Black |
|--------|-----|-----------|
| Balance | Stricter (|BF| <= 1) | Looser |
| Lookup | Faster (more balanced) | Slightly slower |
| Insert/Delete | More rotations | Fewer rotations |
| Use case | Lookup-heavy | Insert/delete-heavy |
| Height | ~1.44 log n | ~2 log n |

## Splay Tree Overview

Self-adjusting BST. Recently accessed elements move to root (move-to-root). Amortized O(log n).

### Operations

- **Zig**: Node is child of root. Single rotation.
- **Zag**: Same as zig for right child.
- **Zig-Zig**: Node and parent both left (or both right) children. Rotate parent then node.
- **Zig-Zag**: Node is left of right child (or right of left). Rotate node twice.

Access, insert, delete all splay the node (or its predecessor/successor) to root.

## Treap Overview

BST by key + heap by random priority. Each node has (key, priority). BST property on keys, heap property on priorities. Expected height O(log n). Insert: add as leaf, rotate up to fix heap. Delete: rotate down to leaf, remove.

```python
import random

class TreapNode:
    def __init__(self, val):
        self.val = val
        self.priority = random.random()
        self.left = None
        self.right = None

def treap_insert(root, val):
    if not root:
        return TreapNode(val)
    if val < root.val:
        root.left = treap_insert(root.left, val)
        if root.left.priority > root.priority:
            root = right_rotate(root)
    else:
        root.right = treap_insert(root.right, val)
        if root.right.priority > root.priority:
            root = left_rotate(root)
    return root
```

## Skip List Overview

Probabilistic alternative to balanced trees. Multiple linked lists at different levels. Level 0 has all elements. Higher levels have fewer elements. Search: start at top, go right while next < target, else go down. Insert: random level. Expected O(log n) search, insert, delete. Simpler than balanced trees, good for concurrent access.
