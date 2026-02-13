# Tree - Advanced Operations

## Check if BST (In-Range)

```python
def is_valid_bst(root):
    def validate(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return validate(node.left, low, node.val) and validate(node.right, node.val, high)
    return validate(root, float('-inf'), float('inf'))
```

## Check Balanced

```python
def is_balanced(root):
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return height(root) != -1
```

## Check Identical Trees

```python
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
```

## Check Symmetric/Mirror

```python
def is_symmetric(root):
    def mirror(left, right):
        if not left and not right:
            return True
        if not left or not right or left.val != right.val:
            return False
        return mirror(left.left, right.right) and mirror(left.right, right.left)
    return not root or mirror(root.left, root.right)
```

## Mirror/Invert Tree

```python
def invert_tree(root):
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

## Diameter

```python
def diameter_of_binary_tree(root):
    result = 0
    def height(node):
        nonlocal result
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        result = max(result, left + right)
        return 1 + max(left, right)
    height(root)
    return result
```

## Max Depth

```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

## Min Depth

```python
def min_depth(root):
    if not root:
        return 0
    if not root.left:
        return 1 + min_depth(root.right)
    if not root.right:
        return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))
```

## Max Path Sum (Any to Any)

```python
def max_path_sum(root):
    result = float('-inf')
    def gain(node):
        nonlocal result
        if not node:
            return 0
        left = max(0, gain(node.left))
        right = max(0, gain(node.right))
        result = max(result, node.val + left + right)
        return node.val + max(left, right)
    gain(root)
    return result
```

## Max Width

```python
from collections import deque

def width_of_binary_tree(root):
    if not root:
        return 0
    q = deque([(root, 0)])
    max_width = 0
    while q:
        n = len(q)
        left = q[0][1]
        for _ in range(n):
            node, idx = q.popleft()
            if node.left:
                q.append((node.left, 2 * idx))
            if node.right:
                q.append((node.right, 2 * idx + 1))
        max_width = max(max_width, idx - left + 1)
    return max_width
```

## All Root-to-Leaf Paths

```python
def binary_tree_paths(root):
    result = []
    def dfs(node, path):
        if not node:
            return
        path.append(str(node.val))
        if not node.left and not node.right:
            result.append('->'.join(path))
        else:
            dfs(node.left, path)
            dfs(node.right, path)
        path.pop()
    dfs(root, [])
    return result
```

## Path Sum (Root to Leaf Exists)

```python
def has_path_sum(root, target_sum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target_sum
    remaining = target_sum - root.val
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)
```

## Path Sum II (All Paths)

```python
def path_sum_ii(root, target_sum):
    result = []
    def dfs(node, path, remaining):
        if not node:
            return
        path.append(node.val)
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])
        else:
            dfs(node.left, path, remaining - node.val)
            dfs(node.right, path, remaining - node.val)
        path.pop()
    dfs(root, [], target_sum)
    return result
```

## Path Sum III (Any Downward Path)

```python
from collections import defaultdict

def path_sum_iii(root, target_sum):
    count = 0
    prefix = defaultdict(int)
    prefix[0] = 1
    def dfs(node, curr_sum):
        nonlocal count
        if not node:
            return
        curr_sum += node.val
        count += prefix.get(curr_sum - target_sum, 0)
        prefix[curr_sum] += 1
        dfs(node.left, curr_sum)
        dfs(node.right, curr_sum)
        prefix[curr_sum] -= 1
    dfs(root, 0)
    return count
```

## Nodes at Distance K from Target

```python
from collections import deque

def distance_k(root, target, k):
    parent = {}
    def build_parent(node, p):
        if not node:
            return
        parent[node] = p
        build_parent(node.left, node)
        build_parent(node.right, node)
    build_parent(root, None)
    visited = {target}
    q = deque([target])
    for _ in range(k):
        for _ in range(len(q)):
            node = q.popleft()
            for neighbor in [node.left, node.right, parent.get(node)]:
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
    return [n.val for n in q]
```

## Count Complete Tree Nodes O(log^2 n)

```python
def count_complete_tree_nodes(root):
    def left_height(node):
        h = 0
        while node:
            h += 1
            node = node.left
        return h
    def right_height(node):
        h = 0
        while node:
            h += 1
            node = node.right
        return h
    if not root:
        return 0
    lh = left_height(root)
    rh = right_height(root)
    if lh == rh:
        return (1 << lh) - 1
    return 1 + count_complete_tree_nodes(root.left) + count_complete_tree_nodes(root.right)
```

## Flatten to Linked List

```python
def flatten(root):
    def dfs(node):
        if not node:
            return None
        if not node.left and not node.right:
            return node
        left_tail = dfs(node.left)
        right_tail = dfs(node.right)
        if left_tail:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail if right_tail else left_tail
    dfs(root)
```

## Inorder Successor in BST

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

## Inorder Predecessor in BST

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

## Floor and Ceil in BST

```python
def floor_bst(root, key):
    floor_val = None
    while root:
        if root.val == key:
            return root.val
        if root.val < key:
            floor_val = root.val
            root = root.right
        else:
            root = root.left
    return floor_val

def ceil_bst(root, key):
    ceil_val = None
    while root:
        if root.val == key:
            return root.val
        if root.val > key:
            ceil_val = root.val
            root = root.left
        else:
            root = root.right
    return ceil_val
```

## Kth Smallest in BST

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

## Kth Largest in BST

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

## BST to Sorted Doubly Linked List

```python
def bst_to_doubly_linked_list(root):
    head = None
    prev = None
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

## Merge Two BSTs

```python
def merge_two_bsts(root1, root2):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    arr = inorder(root1) + inorder(root2)
    arr.sort()
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

## LCA in BST

```python
def lca_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

## LCA in Binary Tree

```python
def lca_binary_tree(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca_binary_tree(root.left, p, q)
    right = lca_binary_tree(root.right, p, q)
    if left and right:
        return root
    return left or right
```

## Sum of Left Leaves

```python
def sum_of_left_leaves(root):
    def dfs(node, is_left):
        if not node:
            return 0
        if not node.left and not node.right and is_left:
            return node.val
        return dfs(node.left, True) + dfs(node.right, False)
    return dfs(root, False)
```

## Vertical Order Traversal

```python
from collections import defaultdict

def vertical_order_traversal(root):
    cols = defaultdict(list)
    def dfs(node, row, col):
        if not node:
            return
        cols[col].append((row, node.val))
        dfs(node.left, row + 1, col - 1)
        dfs(node.right, row + 1, col + 1)
    dfs(root, 0, 0)
    result = []
    for col in sorted(cols.keys()):
        result.append([v for _, v in sorted(cols[col])])
    return result
```

## Boundary Traversal

```python
def boundary_traversal(root):
    if not root:
        return []
    result = [root.val]
    def left_boundary(node):
        if not node or (not node.left and not node.right):
            return
        result.append(node.val)
        if node.left:
            left_boundary(node.left)
        else:
            left_boundary(node.right)
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
        if node.right:
            right_boundary(node.right)
        else:
            right_boundary(node.left)
        result.append(node.val)
    left_boundary(root.left)
    leaves(root.left)
    leaves(root.right)
    right_boundary(root.right)
    return result
```

## Subtree of Another Tree

```python
def is_subtree(root, sub_root):
    def same(s, t):
        if not s and not t:
            return True
        if not s or not t or s.val != t.val:
            return False
        return same(s.left, t.left) and same(s.right, t.right)
    if not sub_root:
        return True
    if not root:
        return False
    if same(root, sub_root):
        return True
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)
```

## Duplicate Subtrees

```python
from collections import defaultdict

def find_duplicate_subtrees(root):
    count = defaultdict(int)
    result = []
    def serialize(node):
        if not node:
            return '#'
        s = f'{node.val},{serialize(node.left)},{serialize(node.right)}'
        count[s] += 1
        if count[s] == 2:
            result.append(node)
        return s
    serialize(root)
    return result
```

## Count Good Nodes

```python
def good_nodes(root):
    def dfs(node, max_so_far):
        if not node:
            return 0
        count = 1 if node.val >= max_so_far else 0
        max_so_far = max(max_so_far, node.val)
        return count + dfs(node.left, max_so_far) + dfs(node.right, max_so_far)
    return dfs(root, float('-inf'))
```

## Binary Tree Cameras

```python
def min_camera_cover(root):
    cameras = 0
    COVERED = 0
    COVERED_BY_CHILD = 1
    NEEDS_COVER = 2
    def dfs(node):
        nonlocal cameras
        if not node:
            return COVERED
        left = dfs(node.left)
        right = dfs(node.right)
        if left == NEEDS_COVER or right == NEEDS_COVER:
            cameras += 1
            return COVERED_BY_CHILD
        if left == COVERED_BY_CHILD or right == COVERED_BY_CHILD:
            return COVERED
        return NEEDS_COVER
    if dfs(root) == NEEDS_COVER:
        cameras += 1
    return cameras
```

## House Robber III

```python
def rob(root):
    def dfs(node):
        if not node:
            return (0, 0)
        left = dfs(node.left)
        right = dfs(node.right)
        rob_this = node.val + left[1] + right[1]
        skip_this = max(left) + max(right)
        return (rob_this, skip_this)
    return max(dfs(root))
```

## Distribute Coins

```python
def distribute_coins(root):
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
