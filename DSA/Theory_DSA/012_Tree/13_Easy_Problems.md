# Tree - Easy Problems

## 1. Maximum Depth of Binary Tree

Return the maximum depth (number of nodes on longest path from root to leaf). Recursive: depth = 1 + max(left_depth, right_depth). Base: null returns 0.

```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

Time: O(n) | Space: O(h)

---

## 2. Same Tree

Check if two binary trees are structurally identical with same values. Recursive compare. Both null: true. One null or val mismatch: false. Else recurse on left and right.

```python
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

Time: O(n) | Space: O(h)

---

## 3. Invert Binary Tree

Mirror the tree (swap left and right children for every node). Recursive swap. Swap left/right, recurse on both children.

```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

Time: O(n) | Space: O(h)

---

## 4. Symmetric Tree

Check if tree is mirror of itself around center. Helper(left, right): both null true; one null or val mismatch false; recurse (left.left, right.right) and (left.right, right.left).

```python
def isSymmetric(root):
    def mirror(l, r):
        if not l and not r:
            return True
        if not l or not r or l.val != r.val:
            return False
        return mirror(l.left, r.right) and mirror(l.right, r.left)
    return mirror(root.left, root.right) if root else True
```

Time: O(n) | Space: O(h)

---

## 5. Binary Tree Inorder Traversal

Return inorder (left-root-right) traversal. Recursive DFS or iterative stack. For BST yields sorted order.

```python
def inorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res
```

Time: O(n) | Space: O(h)

---

## 6. Binary Tree Preorder Traversal

Return preorder (root-left-right) traversal. Recursive or iterative with stack (push right before left).

```python
def preorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        res.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return res
```

Time: O(n) | Space: O(h)

---

## 7. Binary Tree Postorder Traversal

Return postorder (left-right-root) traversal. Recursive or iterative with two stacks or one stack.

```python
def postorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        dfs(node.right)
        res.append(node.val)
    dfs(root)
    return res
```

Time: O(n) | Space: O(h)

---

## 8. Binary Tree Level Order Traversal

Return level-by-level traversal (BFS). Queue. Process level by level, collect values.

```python
def levelOrder(root):
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
    return res
```

Time: O(n) | Space: O(n)

---

## 9. Convert Sorted Array to BST

Build height-balanced BST from sorted array. Mid element is root. Recursively build left from left half, right from right half.

```python
def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    return root
```

Time: O(n) | Space: O(log n)

---

## 10. Balanced Binary Tree

Check if for every node |height(left) - height(right)| <= 1. DFS returns height. If any subtree returns -1 (unbalanced) or |L-R| > 1, propagate -1. Else return 1 + max(L, R).

```python
def isBalanced(root):
    def height(node):
        if not node:
            return 0
        lh = height(node.left)
        rh = height(node.right)
        if lh == -1 or rh == -1 or abs(lh - rh) > 1:
            return -1
        return 1 + max(lh, rh)
    return height(root) != -1
```

Time: O(n) | Space: O(h)

---

## 11. Minimum Depth of Binary Tree

Minimum number of nodes from root to nearest leaf. If null return 0. If no left, return 1 + min_depth(right). If no right, return 1 + min_depth(left). Else 1 + min(both).

```python
def minDepth(root):
    if not root:
        return 0
    if not root.left:
        return 1 + minDepth(root.right)
    if not root.right:
        return 1 + minDepth(root.left)
    return 1 + min(minDepth(root.left), minDepth(root.right))
```

Time: O(n) | Space: O(h)

---

## 12. Path Sum

Check if there exists root-to-leaf path with given sum. DFS. At leaf, check if remaining sum equals node val. Else recurse with sum - node.val.

```python
def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    rem = targetSum - root.val
    return hasPathSum(root.left, rem) or hasPathSum(root.right, rem)
```

Time: O(n) | Space: O(h)

---

## 13. Merge Two Binary Trees

Merge two trees by summing overlapping nodes. If one null return other. Create new node with sum. Recurse on left and right.

```python
def mergeTrees(root1, root2):
    if not root1:
        return root2
    if not root2:
        return root1
    root = TreeNode(root1.val + root2.val)
    root.left = mergeTrees(root1.left, root2.left)
    root.right = mergeTrees(root1.right, root2.right)
    return root
```

Time: O(n) | Space: O(h)

---

## 14. Diameter of Binary Tree

Longest path between any two nodes (edges). At each node, diameter through node = left_height + right_height. Track global max. Return height = 1 + max(left, right).

```python
def diameterOfBinaryTree(root):
    best = 0
    def height(node):
        nonlocal best
        if not node:
            return 0
        lh = height(node.left)
        rh = height(node.right)
        best = max(best, lh + rh)
        return 1 + max(lh, rh)
    height(root)
    return best
```

Time: O(n) | Space: O(h)

---

## 15. Subtree of Another Tree

Check if subRoot is subtree of root. For each node, check if tree rooted there equals subRoot (same structure and values). Recurse on left and right.

```python
def isSubtree(root, subRoot):
    def same(p, q):
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return same(p.left, q.left) and same(p.right, q.right)

    if not root:
        return False
    if same(root, subRoot):
        return True
    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)
```

Time: O(n * m) | Space: O(h)

---

## 16. Search in a BST

Find node with value equal to target in BST. Iterative or recursive. If target < val go left, else go right. Return node or null.

```python
def searchBST(root, val):
    if not root or root.val == val:
        return root
    return searchBST(root.left, val) if val < root.val else searchBST(root.right, val)
```

Time: O(h) | Space: O(h)

---

## 17. Insert into a BST

Insert value into BST maintaining property. Find leaf position (go left if smaller, right if larger), attach new node.

```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    return root
```

Time: O(h) | Space: O(h)

---

## 18. Leaf-Similar Trees

Check if two trees have same leaf value sequence. DFS collect leaves for both trees. Compare sequences.

```python
def leafSimilar(root1, root2):
    def leaves(node):
        if not node:
            return []
        if not node.left and not node.right:
            return [node.val]
        return leaves(node.left) + leaves(node.right)
    return leaves(root1) == leaves(root2)
```

Time: O(n) | Space: O(h)

---

## 19. Range Sum of BST

Sum all values in BST in range [low, high]. Inorder or recursive. If node < low, go right only. If node > high, go left only. Else add node and recurse both.

```python
def rangeSumBST(root, low, high):
    if not root:
        return 0
    if root.val < low:
        return rangeSumBST(root.right, low, high)
    if root.val > high:
        return rangeSumBST(root.left, low, high)
    return root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high)
```

Time: O(n) | Space: O(h)

---

## 20. Increasing Order Search Tree

Reorder BST so leftmost node is root, no left children, right chain is inorder. Inorder traversal, build new tree with only right children. Or in-place relink during inorder.

```python
def increasingBST(root):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    vals = inorder(root)
    dummy = cur = TreeNode()
    for v in vals:
        cur.right = TreeNode(v)
        cur = cur.right
    return dummy.right
```

Time: O(n) | Space: O(n)

---

## 21. Univalued Binary Tree

Check if all node values are same. DFS. Compare each node with root value. Return false on mismatch.

```python
def isUnivalTree(root):
    val = root.val
    def dfs(node):
        if not node:
            return True
        if node.val != val:
            return False
        return dfs(node.left) and dfs(node.right)
    return dfs(root)
```

Time: O(n) | Space: O(h)

---

## 22. Sum of Root to Leaf Binary Numbers

Root-to-leaf paths represent binary numbers. Sum all. DFS with current number. At leaf, add to sum. Pass (num * 2 + val) to children.

```python
def sumRootToLeaf(root):
    total = 0
    def dfs(node, num):
        nonlocal total
        if not node:
            return
        num = num * 2 + node.val
        if not node.left and not node.right:
            total += num
            return
        dfs(node.left, num)
        dfs(node.right, num)
    dfs(root, 0)
    return total
```

Time: O(n) | Space: O(h)

---

## 23. Second Minimum Node in Binary Tree

Tree where root = min(root.left, root.right). Find second minimum value. Root is minimum. Second min is smallest value greater than root. DFS find min value > root.

```python
def findSecondMinimumValue(root):
    def dfs(node, first):
        if not node:
            return float('inf')
        if node.val > first:
            return node.val
        return min(dfs(node.left, first), dfs(node.right, first))
    res = dfs(root, root.val)
    return res if res != float('inf') else -1
```

Time: O(n) | Space: O(h)

---

## 24. Find Mode in BST

Find most frequently occurring value(s) in BST. Inorder gives sorted order. Count consecutive same values, track max count and modes.

```python
def findMode(root):
    from collections import Counter
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    if not root:
        return []
    vals = inorder(root)
    c = Counter(vals)
    mx = max(c.values())
    return [v for v, cnt in c.items() if cnt == mx]
```

Time: O(n) | Space: O(n)

---

## 25. Minimum Absolute Difference in BST

Minimum absolute difference between any two node values in BST. Inorder traversal. Track previous value. Update min diff with current - prev.

```python
def getMinimumDifference(root):
    prev, best = None, float('inf')
    def inorder(node):
        nonlocal prev, best
        if not node:
            return
        inorder(node.left)
        if prev is not None:
            best = min(best, node.val - prev)
        prev = node.val
        inorder(node.right)
    inorder(root)
    return best
```

Time: O(n) | Space: O(h)
