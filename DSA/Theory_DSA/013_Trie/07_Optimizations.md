# Trie - Optimizations

## Compressed Trie / Radix Tree (Merge Single-Child Chains)

Merge chains of nodes that have only one child into a single node with an edge labeled by the concatenated string. Reduces node count and memory.

```python
class RadixNode:
    def __init__(self, label: str = ""):
        self.label = label
        self.children = {}
        self.is_end = False

class RadixTrie:
    def __init__(self):
        self.root = RadixNode()

    def insert(self, word: str) -> None:
        if not word:
            self.root.is_end = True
            return
        node = self.root
        i = 0
        while i < len(word):
            matched = False
            for edge, child in list(node.children.items()):
                j = 0
                while j < len(edge) and i + j < len(word) and edge[j] == word[i + j]:
                    j += 1
                if j > 0:
                    if j == len(edge):
                        node = child
                        i += j
                        matched = True
                    else:
                        mid = RadixNode(edge[:j])
                        mid.children[edge[j:]] = child
                        if i + j < len(word):
                            mid.children[word[i+j:]] = RadixNode()
                            mid.children[word[i+j:]].is_end = True
                        else:
                            mid.is_end = True
                        del node.children[edge]
                        node.children[edge[:j]] = mid
                        return
                    break
            if not matched:
                node.children[word[i:]] = RadixNode()
                node.children[word[i:]].is_end = True
                return
        node.is_end = True
```

## Patricia Trie

Patricia (Practical Algorithm To Retrieve Information Coded In Alphanumeric) is a space-optimized trie where each node with only one child is merged with its parent. Same concept as radix tree.

## Ternary Search Trie (Space-Efficient)

Uses three pointers per node (left, mid, right) instead of 26 or a map. Good for sparse data.

```python
class TSTNode:
    def __init__(self, char: str):
        self.char = char
        self.left = self.mid = self.right = None
        self.value = None

class TernaryTrie:
    def __init__(self):
        self.root = None

    def _put(self, node, key: str, i: int, val) -> TSTNode:
        c = key[i]
        if node is None:
            node = TSTNode(c)
        if c < node.char:
            node.left = self._put(node.left, key, i, val)
        elif c > node.char:
            node.right = self._put(node.right, key, i, val)
        elif i + 1 < len(key):
            node.mid = self._put(node.mid, key, i + 1, val)
        else:
            node.value = val
        return node

    def put(self, key: str, val) -> None:
        self.root = self._put(self.root, key, 0, val)
```

## Array vs Map Benchmarking

| Operation | Array (26) | HashMap |
|-----------|------------|---------|
| Insert 10K words | Faster (no hash) | Slightly slower |
| Search | O(1) index | O(1) avg |
| Memory (sparse) | Higher (26 ptrs) | Lower |
| Memory (dense) | Lower | Slightly higher |

Use array when: lowercase a-z only, dense trie. Use map when: Unicode, variable alphabet, sparse.

## Lazy vs Eager Deletion

**Eager**: Delete nodes immediately when word is removed. May leave orphan chains.

**Lazy**: Mark node as not end; optionally prune in background. Simpler, defers work.

```python
def delete_lazy(self, word: str) -> bool:
    node = self.root
    for c in word:
        if c not in node.children:
            return False
        node = node.children[c]
    if not node.is_end:
        return False
    node.is_end = False
    return True
```

## Bitwise Trie for XOR Problems

### Max XOR Queries

```python
class XORTrie:
    def __init__(self, bits=32):
        self.root = {}
        self.bits = bits

    def insert(self, x: int) -> None:
        node = self.root
        for i in range(self.bits - 1, -1, -1):
            b = (x >> i) & 1
            if b not in node:
                node[b] = {}
            node = node[b]

    def max_xor(self, x: int) -> int:
        node = self.root
        res = 0
        for i in range(self.bits - 1, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            if want in node:
                res |= (1 << i)
                node = node[want]
            else:
                node = node[b]
        return res
```

### Count Pairs with XOR in Range

```python
def count_pairs_xor_in_range(nums: list[int], low: int, high: int) -> int:
    trie = XORTrie()
    count = 0
    for x in nums:
        count += trie.count_xor_in_range(x, high) - trie.count_xor_in_range(x, low - 1)
        trie.insert(x)
    return count
```

## Persistent Trie Overview

Persistent trie keeps history of all versions. Each update creates new nodes along the path, sharing unchanged subtrees. Used for range queries (e.g., XOR in range [L, R]).

```python
class PersistentTrieNode:
    def __init__(self):
        self.children = [None, None]
        self.count = 0

def persistent_trie_insert(root, x, bits=32):
    new_root = PersistentTrieNode()
    new_root.children = root.children[:] if root else [None, None]
    node = new_root
    for i in range(bits - 1, -1, -1):
        b = (x >> i) & 1
        new_node = PersistentTrieNode()
        if root and root.children[b]:
            new_node.children = root.children[b].children[:]
            new_node.count = root.children[b].count
        new_node.count += 1
        node.children[b] = new_node
        node = new_node
        root = root.children[b] if root else None
    return new_root
```

## Double-Array Trie Overview

Double-array trie uses two arrays: BASE and CHECK. State transition: next_state = BASE[current] + char_code. CHECK[next_state] = current. Enables compact storage and fast lookup. Used in Japanese morphological analyzers (MeCab).

```python
class DoubleArrayTrie:
    def __init__(self):
        self.base = [0]
        self.check = [-1]

    def build(self, words: list[str]) -> None:
        for word in sorted(words):
            state = 0
            for c in word:
                code = ord(c) - ord('a')
                next_s = self.base[state] + code
                if next_s >= len(self.base):
                    self.base.extend([0] * (next_s - len(self.base) + 1))
                    self.check.extend([-1] * (next_s - len(self.check) + 1))
                self.check[next_s] = state
                state = next_s
```
