# Tree - Easy Problems

## 1. Maximum Depth of Binary Tree

**Description**: Return the maximum depth (number of nodes on longest path from root to leaf).

**Approach**: Recursive: depth = 1 + max(left_depth, right_depth). Base: null returns 0.

---

## 2. Same Tree

**Description**: Check if two binary trees are structurally identical with same values.

**Approach**: Recursive compare. Both null: true. One null or val mismatch: false. Else recurse on left and right.

---

## 3. Invert Binary Tree

**Description**: Mirror the tree (swap left and right children for every node).

**Approach**: Recursive swap. Swap left/right, recurse on both children.

---

## 4. Symmetric Tree

**Description**: Check if tree is mirror of itself around center.

**Approach**: Helper(left, right): both null true; one null or val mismatch false; recurse (left.left, right.right) and (left.right, right.left).

---

## 5. Binary Tree Inorder Traversal

**Description**: Return inorder (left-root-right) traversal.

**Approach**: Recursive DFS or iterative stack. For BST yields sorted order.

---

## 6. Binary Tree Preorder Traversal

**Description**: Return preorder (root-left-right) traversal.

**Approach**: Recursive or iterative with stack (push right before left).

---

## 7. Binary Tree Postorder Traversal

**Description**: Return postorder (left-right-root) traversal.

**Approach**: Recursive or iterative with two stacks or one stack.

---

## 8. Binary Tree Level Order Traversal

**Description**: Return level-by-level traversal (BFS).

**Approach**: Queue. Process level by level, collect values.

---

## 9. Convert Sorted Array to BST

**Description**: Build height-balanced BST from sorted array.

**Approach**: Mid element is root. Recursively build left from left half, right from right half.

---

## 10. Balanced Binary Tree

**Description**: Check if for every node |height(left) - height(right)| <= 1.

**Approach**: DFS returns height. If any subtree returns -1 (unbalanced) or |L-R| > 1, propagate -1. Else return 1 + max(L, R).

---

## 11. Minimum Depth of Binary Tree

**Description**: Minimum number of nodes from root to nearest leaf.

**Approach**: If null return 0. If no left, return 1 + min_depth(right). If no right, return 1 + min_depth(left). Else 1 + min(both).

---

## 12. Path Sum

**Description**: Check if there exists root-to-leaf path with given sum.

**Approach**: DFS. At leaf, check if remaining sum equals node val. Else recurse with sum - node.val.

---

## 13. Merge Two Binary Trees

**Description**: Merge two trees by summing overlapping nodes.

**Approach**: If one null return other. Create new node with sum. Recurse on left and right.

---

## 14. Diameter of Binary Tree

**Description**: Longest path between any two nodes (edges).

**Approach**: At each node, diameter through node = left_height + right_height. Track global max. Return height = 1 + max(left, right).

---

## 15. Subtree of Another Tree

**Description**: Check if subRoot is subtree of root.

**Approach**: For each node, check if tree rooted there equals subRoot (same structure and values). Recurse on left and right.

---

## 16. Search in a BST

**Description**: Find node with value equal to target in BST.

**Approach**: Iterative or recursive. If target < val go left, else go right. Return node or null.

---

## 17. Insert into a BST

**Description**: Insert value into BST maintaining property.

**Approach**: Find leaf position (go left if smaller, right if larger), attach new node.

---

## 18. Leaf-Similar Trees

**Description**: Check if two trees have same leaf value sequence.

**Approach**: DFS collect leaves for both trees. Compare sequences.

---

## 19. Range Sum of BST

**Description**: Sum all values in BST in range [low, high].

**Approach**: Inorder or recursive. If node < low, go right only. If node > high, go left only. Else add node and recurse both.

---

## 20. Increasing Order Search Tree

**Description**: Reorder BST so leftmost node is root, no left children, right chain is inorder.

**Approach**: Inorder traversal, build new tree with only right children. Or in-place relink during inorder.

---

## 21. Univalued Binary Tree

**Description**: Check if all node values are same.

**Approach**: DFS. Compare each node with root value. Return false on mismatch.

---

## 22. Sum of Root to Leaf Binary Numbers

**Description**: Root-to-leaf paths represent binary numbers. Sum all.

**Approach**: DFS with current number. At leaf, add to sum. Pass (num * 2 + val) to children.

---

## 23. Second Minimum Node in Binary Tree

**Description**: Tree where root = min(root.left, root.right). Find second minimum value.

**Approach**: Root is minimum. Second min is smallest value greater than root. DFS find min value > root.

---

## 24. Find Mode in BST

**Description**: Find most frequently occurring value(s) in BST.

**Approach**: Inorder gives sorted order. Count consecutive same values, track max count and modes.

---

## 25. Minimum Absolute Difference in BST

**Description**: Minimum absolute difference between any two node values in BST.

**Approach**: Inorder traversal. Track previous value. Update min diff with current - prev.
