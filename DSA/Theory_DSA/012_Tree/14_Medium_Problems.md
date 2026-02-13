# Tree - Medium Problems

## 1. Binary Tree Right Side View

**Description**: Return values of rightmost nodes at each level.

**Approach**: BFS, append last node of each level. Or DFS with level, overwrite result[level] (right-first DFS).

---

## 2. Binary Tree Level Order Traversal II

**Description**: Level order from bottom to top.

**Approach**: Same as level order, reverse result. Or use deque appendleft for each level.

---

## 3. Binary Tree Zigzag Level Order Traversal

**Description**: Level order but alternate left-to-right and right-to-left per level.

**Approach**: BFS. Reverse every other level. Or use deque and alternate popleft/popright.

---

## 4. Construct Binary Tree from Preorder and Inorder

**Description**: Build tree from preorder and inorder traversal.

**Approach**: Preorder[0] is root. Find in inorder to split. Left subtree: inorder[:idx], preorder[1:idx+1]. Right: inorder[idx+1:], preorder[idx+1:].

---

## 5. Construct Binary Tree from Inorder and Postorder

**Description**: Build tree from inorder and postorder.

**Approach**: Postorder[-1] is root. Find in inorder. Left: inorder[:idx], postorder[:idx]. Right: inorder[idx+1:], postorder[idx:-1].

---

## 6. Binary Tree Maximum Path Sum

**Description**: Find maximum path sum (any node to any node, path = parent-child).

**Approach**: At each node, path through node = val + max(0, left_gain) + max(0, right_gain). Return to parent: val + max(left, right). Track global max.

---

## 7. Validate Binary Search Tree

**Description**: Check if tree is valid BST.

**Approach**: DFS with (low, high) range. Node must be in (low, high). Recurse left with (low, val), right with (val, high).

---

## 8. Kth Smallest Element in BST

**Description**: Find kth smallest element.

**Approach**: Inorder traversal, return when count reaches k. Iterative stack or recursive with counter.

---

## 9. Lowest Common Ancestor of Binary Tree

**Description**: Find LCA of two nodes.

**Approach**: If root is p or q or null, return root. Recurse left and right. If both return non-null, root is LCA. Else return non-null.

---

## 10. Lowest Common Ancestor of BST

**Description**: Find LCA in BST.

**Approach**: If both p, q < root, go left. If both > root, go right. Else root is LCA.

---

## 11. Binary Tree from Preorder and Postorder (Full Binary)

**Description**: Construct full binary tree from preorder and postorder.

**Approach**: Preorder[0] is root. Preorder[1] is left root. Find in postorder to get left subtree size. Split and recurse.

---

## 12. Flatten Binary Tree to Linked List

**Description**: Flatten to right-only linked list in preorder.

**Approach**: Recursive. Flatten left and right. Set root.right = flattened left, find tail, tail.right = flattened right, root.left = None.

---

## 13. Populating Next Right Pointers in Each Node

**Description**: Connect each node to its next right at same level.

**Approach**: BFS with level. Or use next pointers: for each level, link children. Parent has next, so parent.next.left is right sibling's left.

---

## 14. Path Sum II

**Description**: Find all root-to-leaf paths with given sum.

**Approach**: DFS with path list. At leaf, if sum matches, append path copy to result. Backtrack (pop) after recurse.

---

## 15. Path Sum III

**Description**: Count paths (any downward) that sum to target.

**Approach**: Prefix sum + hash map. At each node, count how many prefix_sum - target exist. Update map, recurse, backtrack.

---

## 16. Binary Search Tree Iterator

**Description**: Iterator for inorder traversal with next() and hasNext().

**Approach**: Stack. Push left spine. next() pops, pushes left spine of right child. hasNext() = stack non-empty.

---

## 17. Count Good Nodes in Binary Tree

**Description**: Count nodes where path from root has no value greater than node.

**Approach**: DFS with max_so_far. If node.val >= max, count++. Recurse with updated max.

---

## 18. Delete Node in BST

**Description**: Delete node with given value from BST.

**Approach**: Find node. If leaf, remove. If one child, replace with child. If two children, replace with inorder successor, delete successor.

---

## 19. Trim BST

**Description**: Remove nodes outside [low, high] range.

**Approach**: If root < low, return trim(root.right). If root > high, return trim(root.left). Else root.left = trim(left), root.right = trim(right).

---

## 20. Unique Binary Search Trees II

**Description**: Generate all structurally unique BSTs with n nodes (values 1 to n).

**Approach**: For each i as root, left = generate(1, i-1), right = generate(i+1, n). Combine all pairs.

---

## 21. Binary Tree Vertical Order Traversal

**Description**: Traverse by vertical columns (left to right).

**Approach**: BFS/DFS with column index. Map column to list of (row, val). Sort by row for same column. Return sorted by column.

---

## 22. House Robber III

**Description**: Max money from nodes, no two adjacent nodes.

**Approach**: DP. (rob_this, skip_this). Rob = val + left_skip + right_skip. Skip = max(left) + max(right).

---

## 23. Binary Tree Cameras

**Description**: Minimum cameras to cover all nodes (camera covers parent, self, children).

**Approach**: Greedy/DP. State: 0=covered, 1=has camera, 2=needs cover. Leaf returns 2. If child 2, place camera. If child 1, current covered.

---

## 24. Distribute Coins in Binary Tree

**Description**: Minimum moves so every node has exactly 1 coin.

**Approach**: DFS returns excess (coins - 1). Moves += |left_excess| + |right_excess|. Return node.val + left + right - 1.

---

## 25. All Nodes Distance K in Binary Tree

**Description**: Return values of nodes at distance k from target.

**Approach**: Build parent map. BFS from target to k levels (include parent, left, right). Visited set.

---

## Hard Problems

### 1. Serialize and Deserialize Binary Tree

**Description**: Convert tree to string and back.

**Approach**: Preorder with "null" for missing. Comma-separated. Deserialize by consuming tokens.

---

### 2. Binary Tree Maximum Path Sum

**Description**: Same as medium but often classified hard in some platforms.

**Approach**: See medium #6.

---

### 3. Count Complete Tree Nodes

**Description**: Count nodes in complete binary tree in O(log^2 n) or better.

**Approach**: Compute left and right heights from root. If equal, full tree 2^h - 1. Else recurse on left and right subtrees.

---

### 4. Recover Binary Search Tree

**Description**: Two nodes are swapped. Recover without changing structure.

**Approach**: Inorder find two inversions. First: prev > curr. Second: next prev > curr. Swap first's prev with second's curr.

---

### 5. Binary Tree Postorder Traversal (Iterative One Stack)

**Description**: Postorder with single stack.

**Approach**: Reverse preorder (root-right-left) gives postorder reversed. Or use stack with peek to detect when to process.

---

### 6. Sum of Distances in Tree

**Description**: For each node, sum of distances to all other nodes.

**Approach**: Re-rooting. First DFS: subtree sizes, sum of distances from root. Second DFS: when moving root to child, new_sum = old - size[child] + (n - size[child]).

---

### 7. Number of Good Leaf Node Pairs

**Description**: Count pairs of leaves with distance <= d.

**Approach**: At each node, get list of leaf distances. For each (left_d, right_d) with left_d + right_d <= d, add to count. Return distances + 1.

---

### 8. Minimum Cost Tree from Leaf Values

**Description**: Build tree from leaf array. Cost of node = max(left_leaves) * max(right_leaves). Minimize total cost.

**Approach**: DP. dp[i][j] = min cost for leaves i..j. Try all splits. Or greedy: repeatedly pick min and merge with smaller neighbor (stack).

---

### 9. Count of Smaller Numbers After Self

**Description**: For each element, count elements to the right that are smaller.

**Approach**: Merge sort (inversion count) or coordinate compression + BIT/Segment tree. Process from right, query count in [min, num-1], update at num.

---

### 10. Longest Increasing Path in Matrix

**Description**: Not tree but similar DFS/DP structure.

**Approach**: DFS with memo. At each cell, try 4 directions. Memo[i][j] = 1 + max(valid neighbors).

---

### 11. Binary Tree Cameras (Alternative Formulations)

**Description**: Variants of camera placement.

**Approach**: See medium #23.

---

### 12. Redundant Connection II

**Description**: Directed graph from tree + one edge. Find edge to remove to get valid rooted tree.

**Approach**: Union-Find. Case 1: node has two parents. Case 2: cycle. Handle both.

---

### 13. Binary Tree Maximum Path Sum (Follow-up)

**Description**: Return the path itself, not just sum.

**Approach**: Track path in DFS. Return (max_sum, path). Combine paths at each node.

---

### 14. Count Univalue Subtrees

**Description**: Count subtrees where all values are same.

**Approach**: DFS returns (is_univalue, value). Univalue if left and right univalue and match node value.

---

### 15. Closest Leaf in Binary Tree

**Description**: Find nearest leaf to given node k.

**Approach**: Build graph (node to neighbors). BFS from k to find nearest leaf.
