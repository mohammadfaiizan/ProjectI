# B-Trees

## B-Tree Properties (Order m)

- Every node has at most m children
- Root has at least 2 children (unless it is a leaf)
- Internal nodes (except root) have at least ceil(m/2) children
- All leaves at same level
- Keys in node are sorted
- Node with k keys has k+1 children (or is leaf)

Min keys in internal node (except root): ceil(m/2) - 1
Max keys: m - 1

## Search

Traverse from root. At each node, find key or child pointer. If key found, return. Else follow child pointer. O(log n) disk accesses.

```
BTreeSearch(node, key):
    i = 0
    while i < node.key_count and key > node.keys[i]:
        i++
    if i < node.key_count and key == node.keys[i]:
        return (node, i)
    if node.leaf:
        return NIL
    return BTreeSearch(node.children[i], key)
```

## Insert (Split if Overflow)

Insert in leaf. If leaf overflows (keys >= m), split: promote middle key to parent, create two siblings. May cascade splits up to root. If root splits, new root with one key.

```
BTreeInsert(T, key):
    r = T.root
    if r is full (m-1 keys):
        s = new node
        T.root = s
        s.leaf = false
        s.children[0] = r
        BTreeSplitChild(s, 0)
        BTreeInsertNonFull(s, key)
    else:
        BTreeInsertNonFull(r, key)

BTreeSplitChild(node, i):
    y = node.children[i]
    z = new node
    z.leaf = y.leaf
    mid = ceil(m/2) - 1
    for j = 0 to mid-1:
        z.keys[j] = y.keys[mid+1+j]
    if not y.leaf:
        for j = 0 to mid:
            z.children[j] = y.children[mid+1+j]
    y.key_count = mid
    for j = node.key_count down to i+1:
        node.children[j+1] = node.children[j]
    node.children[i+1] = z
    for j = node.key_count-1 down to i:
        node.keys[j+1] = node.keys[j]
    node.keys[i] = y.keys[mid]
    node.key_count++

BTreeInsertNonFull(node, key):
    i = node.key_count - 1
    if node.leaf:
        while i >= 0 and key < node.keys[i]:
            node.keys[i+1] = node.keys[i]
            i--
        node.keys[i+1] = key
        node.key_count++
    else:
        while i >= 0 and key < node.keys[i]:
            i--
        i++
        if node.children[i].key_count == m-1:
            BTreeSplitChild(node, i)
            if key > node.keys[i]:
                i++
        BTreeInsertNonFull(node.children[i], key)
```

## Delete (Merge/Borrow if Underflow)

If key in leaf: remove. If key in internal node: replace with predecessor (max in left subtree) or successor (min in right subtree), delete from subtree. If node underflows after delete (keys < ceil(m/2)-1): borrow from sibling or merge with sibling.

Delete cases:
1. Key in leaf, leaf has enough keys: simple delete
2. Key in leaf, leaf underflows: borrow from sibling or merge
3. Key in internal node: replace with predecessor/successor, recurse
4. Child underflows: borrow or merge

```
BTreeDelete(node, key):
    Find key or child index i
    if key in node:
        if node.leaf:
            remove key
        else:
            pred = predecessor in left subtree
            node.keys[i] = pred
            BTreeDelete(node.children[i], pred)
    else:
        if node.children[i] has minimum keys:
            if sibling has extra: borrow
            else: merge with sibling
        BTreeDelete(node.children[i], key)
```

## B+ Tree

- All data in leaves; internal nodes store only keys for routing
- Leaves linked (singly or doubly) for range scans
- Internal node keys duplicated in leaves or used as separators
- Better for range queries and sequential access
- Used in database indexes (MySQL InnoDB, PostgreSQL)

## B* Tree

- Minimum 2/3 full instead of 1/2
- Redistribute keys between siblings before split (when both full, split into three nodes)
- Fewer splits, better space utilization

## Use in Databases and File Systems

- Disk block size ~4KB-16KB; each node fits in one block
- Height 3-4 for millions of keys (m=100-200)
- Reduces disk I/O vs BST (each level = disk read)
- B+ tree: index + data separation, efficient range scans

## Comparison with BST/AVL/RB

| Structure | Disk optimization | Height | Use case |
|-----------|-------------------|--------|----------|
| BST/AVL/RB | No | O(log n) | In-memory |
| B-tree | Yes (node = block) | O(log_m n) | Disk, DB |
| B+ tree | Yes | O(log_m n) | DB index |

## 2-3 Tree

- Order 3: 2 or 3 children per node
- 1 or 2 keys per node
- All leaves same level
- Insert: add to leaf, split 3-node into 2 nodes, promote middle
- Delete: merge or redistribute

## 2-3-4 Tree

- Order 4: 2, 3, or 4 children
- 1, 2, or 3 keys per node
- Same level leaves
- Split on descent to avoid cascading splits (preemptive split)
- Red-black tree is isomorphic to 2-3-4 tree (red = 3/4-node part)
