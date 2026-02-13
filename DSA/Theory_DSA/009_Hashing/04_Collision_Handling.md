# Collision Handling

A collision occurs when two different keys hash to the same bucket index. Collision handling strategies determine how to store multiple entries in the same bucket.

## Separate Chaining (Linked List per Bucket)

Each bucket holds a linked list of (key, value) pairs. On collision, append to the list.

```python
class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None

class ChainedHashTable:
    def __init__(self, capacity=16):
        self.cap = capacity
        self.buckets = [None] * capacity
        self.size = 0

    def _hash(self, key):
        return hash(key) % self.cap

    def put(self, key, value):
        idx = self._hash(key)
        node = self.buckets[idx]
        while node:
            if node.key == key:
                node.val = value
                return
            node = node.next
        new = ListNode(key, value)
        new.next = self.buckets[idx]
        self.buckets[idx] = new
        self.size += 1

    def get(self, key):
        idx = self._hash(key)
        node = self.buckets[idx]
        while node:
            if node.key == key:
                return node.val
            node = node.next
        return None

    def remove(self, key):
        idx = self._hash(key)
        prev, node = None, self.buckets[idx]
        while node:
            if node.key == key:
                if prev:
                    prev.next = node.next
                else:
                    self.buckets[idx] = node.next
                self.size -= 1
                return True
            prev, node = node, node.next
        return False
```

## Separate Chaining with BST

Replace linked list with balanced BST for O(log n) operations per bucket when chains are long.

```python
class BSTNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

def bst_put(root, key, value):
    if not root:
        return BSTNode(key, value)
    if key < root.key:
        root.left = bst_put(root.left, key, value)
    elif key > root.key:
        root.right = bst_put(root.right, key, value)
    else:
        root.val = value
    return root

def bst_get(root, key):
    if not root:
        return None
    if key < root.key:
        return bst_get(root.left, key)
    if key > root.key:
        return bst_get(root.right, key)
    return root.val

def bst_remove(root, key):
    if not root:
        return None
    if key < root.key:
        root.left = bst_remove(root.left, key)
        return root
    if key > root.key:
        root.right = bst_remove(root.right, key)
        return root
    if not root.left:
        return root.right
    if not root.right:
        return root.left
    succ = root.right
    while succ.left:
        succ = succ.left
    root.key, root.val = succ.key, succ.val
    root.right = bst_remove(root.right, succ.key)
    return root
```

## Open Addressing: Linear Probing

If slot h(k) is occupied, try h(k)+1, h(k)+2, ... until empty slot. Step size = 1.

Clustering: consecutive occupied slots form clusters that grow and slow down probes.

```python
class LinearProbingHashTable:
    TOMBSTONE = object()

    def __init__(self, capacity=16):
        self.cap = capacity
        self.buckets = [None] * capacity
        self.size = 0

    def _hash(self, key):
        return hash(key) % self.cap

    def _probe(self, key, for_insert=False):
        idx = self._hash(key)
        for _ in range(self.cap):
            cell = self.buckets[idx]
            if cell is None:
                return idx
            if cell is self.TOMBSTONE:
                if for_insert:
                    return idx
            elif cell[0] == key:
                return idx
            idx = (idx + 1) % self.cap
        return -1

    def put(self, key, value):
        idx = self._probe(key, for_insert=True)
        if idx < 0:
            return False
        if self.buckets[idx] is None or self.buckets[idx] is self.TOMBSTONE:
            self.size += 1
        self.buckets[idx] = (key, value)
        return True

    def get(self, key):
        idx = self._probe(key)
        if idx < 0 or self.buckets[idx] is None or self.buckets[idx] is self.TOMBSTONE:
            return None
        return self.buckets[idx][1]

    def remove(self, key):
        idx = self._probe(key)
        if idx < 0 or self.buckets[idx] is None or self.buckets[idx] is self.TOMBSTONE:
            return False
        self.buckets[idx] = self.TOMBSTONE
        self.size -= 1
        return True
```

## Open Addressing: Quadratic Probing

Probe sequence: h(k), h(k)+1^2, h(k)+2^2, h(k)+3^2, ... Step = i^2 for i = 0, 1, 2, ...

Reduces primary clustering but can cause secondary clustering. Table size should be prime and load factor < 0.5 for guaranteed empty slot.

```python
class QuadraticProbingHashTable:
    def __init__(self, capacity=16):
        self.cap = capacity
        self.buckets = [None] * capacity
        self.size = 0

    def _hash(self, key):
        return hash(key) % self.cap

    def _probe(self, key):
        base = self._hash(key)
        for i in range(self.cap):
            idx = (base + i * i) % self.cap
            if self.buckets[idx] is None:
                return idx
            if self.buckets[idx][0] == key:
                return idx
        return -1

    def put(self, key, value):
        idx = self._probe(key)
        if idx < 0:
            return False
        if self.buckets[idx] is None:
            self.size += 1
        self.buckets[idx] = (key, value)
        return True

    def get(self, key):
        idx = self._probe(key)
        if idx < 0 or self.buckets[idx] is None:
            return None
        return self.buckets[idx][1]
```

## Open Addressing: Double Hashing

Probe sequence: h1(k), h1(k)+h2(k), h1(k)+2*h2(k), ... Step = h2(key). h2 must never be 0 and should be coprime to m.

```python
class DoubleHashingHashTable:
    def __init__(self, capacity=16):
        self.cap = capacity
        self.buckets = [None] * capacity
        self.size = 0

    def _hash1(self, key):
        return hash(key) % self.cap

    def _hash2(self, key):
        h = hash(key) % self.cap
        return h if h else 1

    def _probe(self, key):
        h1 = self._hash1(key)
        h2 = self._hash2(key)
        for i in range(self.cap):
            idx = (h1 + i * h2) % self.cap
            if self.buckets[idx] is None:
                return idx
            if self.buckets[idx][0] == key:
                return idx
        return -1

    def put(self, key, value):
        idx = self._probe(key)
        if idx < 0:
            return False
        if self.buckets[idx] is None:
            self.size += 1
        self.buckets[idx] = (key, value)
        return True

    def get(self, key):
        idx = self._probe(key)
        if idx < 0 or self.buckets[idx] is None:
            return None
        return self.buckets[idx][1]
```

## Robin Hood Hashing Concept

Variant of linear probing. When probing, if current element has traveled less than the new element, swap them and continue. New element "steals" from the "rich" (those that traveled far). Reduces variance in probe length.

## Cuckoo Hashing Concept

Uses two hash tables and two hash functions. On insert, if slot is occupied, evict existing and re-insert evicted using its alternate hash. May need rehashing if cycle occurs.

```python
class CuckooHashTable:
    def __init__(self, capacity=16):
        self.cap = capacity
        self.t1 = [None] * capacity
        self.t2 = [None] * capacity
        self.max_kicks = capacity

    def _h1(self, key):
        return hash(key) % self.cap

    def _h2(self, key):
        return (hash(key) * 31 + 1) % self.cap

    def _get(self, key):
        i1, i2 = self._h1(key), self._h2(key)
        if self.t1[i1] and self.t1[i1][0] == key:
            return self.t1[i1][1]
        if self.t2[i2] and self.t2[i2][0] == key:
            return self.t2[i2][1]
        return None

    def put(self, key, value):
        i1, i2 = self._h1(key), self._h2(key)
        if self.t1[i1] and self.t1[i1][0] == key:
            self.t1[i1] = (key, value)
            return True
        if self.t2[i2] and self.t2[i2][0] == key:
            self.t2[i2] = (key, value)
            return True
        k, v = key, value
        for _ in range(self.max_kicks):
            i1 = self._h1(k)
            if self.t1[i1] is None:
                self.t1[i1] = (k, v)
                return True
            self.t1[i1], k, v = (k, v), self.t1[i1][0], self.t1[i1][1]
            i2 = self._h2(k)
            if self.t2[i2] is None:
                self.t2[i2] = (k, v)
                return True
            self.t2[i2], k, v = (k, v), self.t2[i2][0], self.t2[i2][1]
        return False
```

## Comparison Table

| Method | Load Factor Tolerance | Cache Performance | Deletion Handling |
|--------|----------------------|-------------------|-------------------|
| Separate chaining | High (0.75-1.0) | Poor (pointer chasing) | Simple (remove from list) |
| Chaining + BST | High | Poor | Standard BST delete |
| Linear probing | Medium (0.5-0.75) | Good (contiguous) | Tombstone or shift |
| Quadratic probing | Medium (0.5) | Good | Tombstone |
| Double hashing | Medium (0.5-0.75) | Good | Tombstone |
| Robin Hood | Higher | Good | Swap and continue |
| Cuckoo | 0.5 typically | Good | Simple |

## Tombstone/Lazy Deletion in Open Addressing

When deleting, mark slot as TOMBSTONE instead of making it empty. Insert can reuse tombstone. Lookup continues past tombstone. Rehashing clears tombstones.

## Rehashing Trigger and Process

**Trigger**: When load factor alpha = n/m exceeds threshold (e.g., 0.75 for chaining, 0.5 for open addressing).

**Process**:
1. Allocate new table with m' = 2*m (or next prime)
2. For each entry in old table, compute new index h(key) mod m'
3. Insert into new table
4. Replace old table with new table

```python
def rehash(old_buckets, new_cap):
    new_buckets = [None] * new_cap
    for entry in old_buckets:
        if entry is None or entry == TOMBSTONE:
            continue
        key, value = entry
        idx = hash(key) % new_cap
        while new_buckets[idx] is not None:
            idx = (idx + 1) % new_cap
        new_buckets[idx] = (key, value)
    return new_buckets
```
