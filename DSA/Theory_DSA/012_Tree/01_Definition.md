# Tree - Definition and Terminology

## Tree Terminology

### Basic Terms

- **Root**: The topmost node of the tree. There is exactly one root in a tree. It has no parent.
- **Parent**: A node that has child nodes. Every node except the root has exactly one parent.
- **Child**: A node directly connected below another node when moving away from the root.
- **Sibling**: Nodes that share the same parent.
- **Leaf (External node)**: A node with no children. Terminal nodes of the tree.
- **Internal node**: A node that has at least one child. All nodes except leaves.
- **Edge**: Connection between two nodes. A tree with n nodes has exactly n-1 edges.
- **Path**: Sequence of nodes connected by edges. Path length = number of edges.
- **Height**: For a node, the number of edges on the longest path from that node to a leaf. Height of a leaf is 0. Height of tree = height of root.
- **Depth**: Number of edges from root to the node. Depth of root is 0.
- **Level**: Same as depth. Level 0 = root, level 1 = root's children, etc.
- **Degree**: Number of children of a node. In a binary tree, degree is at most 2.
- **Subtree**: A node and all its descendants. Each node is root of its own subtree.

### Node Structure

A typical binary tree node:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## Binary Tree

A binary tree is a tree where each node has at most two children: left and right.

## Types of Binary Trees

### Full Binary Tree
Every node has either 0 or 2 children. No node has exactly one child.

### Complete Binary Tree
All levels are fully filled except possibly the last, which is filled from left to right. Used in heap implementation.

### Perfect Binary Tree
All internal nodes have exactly two children and all leaves are at the same level. A perfect tree of height h has 2^(h+1) - 1 nodes.

### Balanced Binary Tree
For every node, the height difference between left and right subtrees is at most 1. AVL and Red-Black trees are balanced. Height is O(log n).

### Degenerate (Skewed) Tree
Every internal node has exactly one child. Essentially a linked list. Height is O(n).

## Binary Search Tree (BST) Property

For every node:
- All values in left subtree are strictly less than node value
- All values in right subtree are strictly greater than node value
- No duplicate values (or define left <= or right >= consistently)

Inorder traversal of BST yields sorted sequence.

## When to Use Trees

- Hierarchical data (file systems, org charts, DOM)
- Search operations with O(log n) average (BST)
- Priority queues (heaps)
- Expression parsing (expression trees)
- Trie for prefix matching
- Segment trees for range queries

## Time Complexity Summary

### Unbalanced BST (worst case - skewed)
| Operation | Time |
|-----------|------|
| Search | O(n) |
| Insert | O(n) |
| Delete | O(n) |
| Min/Max | O(n) |
| Inorder | O(n) |

### Balanced BST (AVL, Red-Black)
| Operation | Time |
|-----------|------|
| Search | O(log n) |
| Insert | O(log n) |
| Delete | O(log n) |
| Min/Max | O(log n) |
| Inorder | O(n) |

### Space Complexity
- Recursion stack: O(h) where h is height
- Balanced: O(log n), Unbalanced: O(n)
