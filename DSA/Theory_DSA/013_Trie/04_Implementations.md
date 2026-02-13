# Trie - Implementations

## Array-Based Trie (children[26] for Lowercase)

Each node has a fixed array of 26 slots for lowercase English letters. Index 0 = 'a', index 25 = 'z'.

**Theory**: O(1) child access by index. Fixed memory per node (26 pointers). Best when alphabet is small and known.

```python
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

class ArrayTrie:
    def __init__(self):
        self.root = TrieNode()

    def _idx(self, c: str) -> int:
        return ord(c) - ord('a')

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            i = self._idx(c)
            if node.children[i] is None:
                node.children[i] = TrieNode()
            node = node.children[i]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            i = self._idx(c)
            if node.children[i] is None:
                return False
            node = node.children[i]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            i = self._idx(c)
            if node.children[i] is None:
                return False
            node = node.children[i]
        return True
```

## HashMap-Based Trie (children as dict)

Each node stores children in a dictionary. Supports any character set.

**Theory**: O(1) average child access. Dynamic memory - only allocates for characters that appear. Flexible for Unicode, digits, etc.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class MapTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True
```

## Comparison Table

| Aspect | Array-Based | HashMap-Based |
|--------|-------------|---------------|
| Child access | O(1) index | O(1) average |
| Memory per node | 26 pointers (fixed) | Only used chars |
| Alphabet support | Lowercase a-z only | Any characters |
| Sparse data | Wastes space | Efficient |
| Dense data | Efficient | Slight overhead |

## Memory Analysis

- **Array**: 26 * 8 bytes = 208 bytes per node (64-bit pointers). Many slots null for sparse tries.
- **HashMap**: Overhead of dict plus one entry per used character. Better for sparse tries (few words, long prefixes).
- **Rule of thumb**: Use array when alphabet is small and fixed; use map when alphabet is large or variable.

## Bitwise Trie for Integers (Binary Trie)

Each level represents one bit. Left child = 0, right child = 1. Used for XOR problems, IP routing.

```python
class BinaryTrieNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.count = 0

class BinaryTrie:
    def __init__(self, bits: int = 32):
        self.root = BinaryTrieNode()
        self.bits = bits

    def insert(self, num: int) -> None:
        node = self.root
        for i in range(self.bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit:
                if node.right is None:
                    node.right = BinaryTrieNode()
                node = node.right
            else:
                if node.left is None:
                    node.left = BinaryTrieNode()
                node = node.left
            node.count += 1

    def max_xor(self, num: int) -> int:
        node = self.root
        result = 0
        for i in range(self.bits - 1, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit
            if want and node.right and node.right.count > 0:
                result |= (1 << i)
                node = node.right
            elif not want and node.left and node.left.count > 0:
                node = node.left
            else:
                child = node.right if node.right else node.left
                if bit and child:
                    result |= (1 << i)
                node = child
        return result
```

## Ternary Search Trie (Three Children: Less, Equal, Greater)

Each node has three children: one for characters less than current, one for equal, one for greater. Space-efficient for sparse data.

```python
class TSTNode:
    def __init__(self, char: str):
        self.char = char
        self.left = None
        self.mid = None
        self.right = None
        self.is_end = False
        self.value = None

class TernarySearchTrie:
    def __init__(self):
        self.root = None

    def _insert(self, node, word: str, i: int) -> 'TSTNode':
        c = word[i]
        if node is None:
            node = TSTNode(c)
        if c < node.char:
            node.left = self._insert(node.left, word, i)
        elif c > node.char:
            node.right = self._insert(node.right, word, i)
        else:
            if i + 1 < len(word):
                node.mid = self._insert(node.mid, word, i + 1)
            else:
                node.is_end = True
        return node

    def insert(self, word: str) -> None:
        if word:
            self.root = self._insert(self.root, word, 0)

    def _search(self, node, word: str, i: int) -> bool:
        if node is None or i >= len(word):
            return False
        c = word[i]
        if c < node.char:
            return self._search(node.left, word, i)
        if c > node.char:
            return self._search(node.right, word, i)
        if i + 1 == len(word):
            return node.is_end
        return self._search(node.mid, word, i + 1)

    def search(self, word: str) -> bool:
        return self._search(self.root, word, 0) if word else False
```
