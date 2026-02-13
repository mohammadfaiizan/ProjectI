# Binary Search Tree

## BST Property/Invariant

For every node with value v:
- All values in left subtree < v
- All values in right subtree > v
- Inorder traversal yields sorted sequence

## Insert

### Iterative

```python
def bst_insert_iterative(root, val):
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
def bst_insert_recursive(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_insert_recursive(root.left, val)
    else:
        root.right = bst_insert_recursive(root.right, val)
    return root
```

## Search

```python
def bst_search(root, val):
    while root:
        if root.val == val:
            return root
        root = root.left if val < root.val else root.right
    return None
```

## Delete (3 Cases)

Leaf: remove. One child: replace with child. Two children: replace with inorder successor.

```python
def bst_delete(root, val):
    if not root:
        return None
    if val < root.val:
        root.left = bst_delete(root.left, val)
    elif val > root.val:
        root.right = bst_delete(root.right, val)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        succ = bst_min(root.right)
        root.val = succ.val
        root.right = bst_delete(root.right, succ.val)
    return root

def bst_min(root):
    while root.left:
        root = root.left
    return root
```

## Find Min/Max

```python
def bst_min(root):
    if not root:
        return None
    while root.left:
        root = root.left
    return root

def bst_max(root):
    if not root:
        return None
    while root.right:
        root = root.right
    return root
```

## Inorder Successor (With Parent Pointer)

```python
def inorder_successor_with_parent(node):
    if node.right:
        curr = node.right
        while curr.left:
            curr = curr.left
        return curr
    curr = node
    while curr.parent and curr.parent.right == curr:
        curr = curr.parent
    return curr.parent
```

## Inorder Successor (Without Parent)

```python
def inorder_successor_bst(root, p):
    succ = None
    while root:
        if p.val < root.val:
            succ = root
            root = root.left
        else:
            root = root.right
    return succ
```

## Inorder Predecessor

```python
def inorder_predecessor_bst(root, p):
    pred = None
    while root:
        if p.val > root.val:
            pred = root
            root = root.right
        else:
            root = root.left
    return pred
```

## Validate BST (In-Range)

```python
def validate_bst(root):
    def check(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return check(node.left, low, node.val) and check(node.right, node.val, high)
    return check(root, float('-inf'), float('inf'))
```

## Validate BST (Inorder)

```python
def validate_bst_inorder(root):
    prev = None
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        if not inorder(node.left):
            return False
        if prev is not None and node.val <= prev:
            return False
        prev = node.val
        return inorder(node.right)
    return inorder(root)
```

## Sorted Array to Balanced BST

```python
def sorted_array_to_bst(nums):
    def build(left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        node = TreeNode(nums[mid])
        node.left = build(left, mid - 1)
        node.right = build(mid + 1, right)
        return node
    return build(0, len(nums) - 1)
```

## Sorted Linked List to BST

```python
def sorted_list_to_bst(head):
    def find_mid(start, end):
        slow = fast = start
        while fast != end and fast.next != end:
            slow = slow.next
            fast = fast.next.next
        return slow
    def build(start, end):
        if start == end:
            return None
        mid = find_mid(start, end)
        node = TreeNode(mid.val)
        node.left = build(start, mid)
        node.right = build(mid.next, end)
        return node
    return build(head, None)
```

## BST to Sorted Doubly Linked List (In-Place)

```python
def bst_to_dll(root):
    head = prev = None
    def inorder(node):
        nonlocal head, prev
        if not node:
            return
        inorder(node.left)
        if prev:
            prev.right = node
            node.left = prev
        else:
            head = node
        prev = node
        inorder(node.right)
    inorder(root)
    if head and prev:
        head.left = prev
        prev.right = head
    return head
```

## Kth Smallest

```python
def kth_smallest_bst(root, k):
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.right
```

## Kth Largest

```python
def kth_largest_bst(root, k):
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.right
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.left
```

## Floor (Largest <= Target)

```python
def floor_bst(root, target):
    floor_val = None
    while root:
        if root.val == target:
            return root.val
        if root.val < target:
            floor_val = root.val
            root = root.right
        else:
            root = root.left
    return floor_val
```

## Ceil (Smallest >= Target)

```python
def ceil_bst(root, target):
    ceil_val = None
    while root:
        if root.val == target:
            return root.val
        if root.val > target:
            ceil_val = root.val
            root = root.left
        else:
            root = root.right
    return ceil_val
```

## Count Nodes in Range

```python
def count_in_range(root, low, high):
    if not root:
        return 0
    if root.val < low:
        return count_in_range(root.right, low, high)
    if root.val > high:
        return count_in_range(root.left, low, high)
    return 1 + count_in_range(root.left, low, high) + count_in_range(root.right, low, high)
```

## Pair with Sum in BST (Two Pointer Inorder)

```python
def find_target(root, k):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    arr = inorder(root)
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == k:
            return True
        if s < k:
            left += 1
        else:
            right -= 1
    return False
```

## Recover BST (Two Swapped Nodes)

```python
def recover_bst(root):
    first = second = prev = None
    def inorder(node):
        nonlocal first, second, prev
        if not node:
            return
        inorder(node.left)
        if prev and prev.val > node.val:
            if first is None:
                first, second = prev, node
            else:
                second = node
        prev = node
        inorder(node.right)
    inorder(root)
    first.val, second.val = second.val, first.val
```

## Trim BST

```python
def trim_bst(root, low, high):
    if not root:
        return None
    if root.val < low:
        return trim_bst(root.right, low, high)
    if root.val > high:
        return trim_bst(root.left, low, high)
    root.left = trim_bst(root.left, low, high)
    root.right = trim_bst(root.right, low, high)
    return root
```

## Closest BST Value

```python
def closest_value(root, target):
    closest = root.val
    while root:
        if abs(root.val - target) < abs(closest - target):
            closest = root.val
        root = root.left if target < root.val else root.right
    return closest
```

## Closest BST Value II (K Closest)

```python
def closest_k_values(root, target, k):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    arr = inorder(root)
    arr.sort(key=lambda x: abs(x - target))
    return arr[:k]
```

## Unique BST Count (Catalan)

C(n) = (2n)! / ((n+1)! * n!) = C(2n,n) / (n+1)

```python
def num_trees(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    return dp[n]
```

## Unique BST Generation

```python
def generate_trees(n):
    def build(left, right):
        if left > right:
            return [None]
        trees = []
        for i in range(left, right + 1):
            left_trees = build(left, i - 1)
            right_trees = build(i + 1, right)
            for l in left_trees:
                for r in right_trees:
                    node = TreeNode(i)
                    node.left = l
                    node.right = r
                    trees.append(node)
        return trees
    return build(1, n) if n else []
```

## Largest BST Subtree

```python
def largest_bst_subtree(root):
    def dfs(node):
        if not node:
            return float('inf'), float('-inf'), 0, True
        lmin, lmax, lsize, lvalid = dfs(node.left)
        rmin, rmax, rsize, rvalid = dfs(node.right)
        valid = lvalid and rvalid and lmax < node.val < rmin
        size = lsize + rsize + 1 if valid else max(lsize, rsize)
        return min(lmin, node.val), max(rmax, node.val), size, valid
    return dfs(root)[2]
```

## BST to Greater Tree

```python
def bst_to_gst(root):
    total = 0
    def reverse_inorder(node):
        nonlocal total
        if not node:
            return
        reverse_inorder(node.right)
        total += node.val
        node.val = total
        reverse_inorder(node.left)
    reverse_inorder(root)
    return root
```

## Balance a BST

```python
def balance_bst(root):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    def build(arr, left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        node = TreeNode(arr[mid])
        node.left = build(arr, left, mid - 1)
        node.right = build(arr, mid + 1, right)
        return node
    arr = inorder(root)
    return build(arr, 0, len(arr) - 1)
```

## Merge Two BSTs Balanced

```python
def merge_two_bsts(root1, root2):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    arr = sorted(inorder(root1) + inorder(root2))
    def build(left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        node = TreeNode(arr[mid])
        node.left = build(left, mid - 1)
        node.right = build(mid + 1, right)
        return node
    return build(0, len(arr) - 1)
```

## Split BST

```python
def split_bst(root, v):
    if not root:
        return None, None
    if root.val <= v:
        small, large = split_bst(root.right, v)
        root.right = small
        return root, large
    else:
        small, large = split_bst(root.left, v)
        root.left = large
        return small, root
```
