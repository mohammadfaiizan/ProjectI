# Trie - Basic Operations

## Node and Trie Structure

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
```

## Insert Word

```python
def insert(self, word: str) -> None:
    node = self.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end = True
```

## Search Exact Word

```python
def search(self, word: str) -> bool:
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end
```

## Search Prefix (startsWith)

```python
def starts_with(self, prefix: str) -> bool:
    node = self.root
    for char in prefix:
        if char not in node.children:
            return False
        node = node.children[char]
    return True
```

## Delete Word

Handles shared prefixes by only removing nodes that are not part of other words.

```python
def delete(self, word: str) -> bool:
    def _delete(node, word, index):
        if index == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            return len(node.children) == 0
        char = word[index]
        if char not in node.children:
            return False
        child = node.children[char]
        should_delete = _delete(child, word, index + 1)
        if should_delete:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end
        return False
    return _delete(self.root, word, 0)
```

## Count Words with Given Prefix

```python
def count_words_with_prefix(self, prefix: str) -> int:
    node = self.root
    for char in prefix:
        if char not in node.children:
            return 0
        node = node.children[char]
    return self._count_words(node)

def _count_words(self, node: TrieNode) -> int:
    count = 1 if node.is_end else 0
    for child in node.children.values():
        count += self._count_words(child)
    return count
```

## Count Distinct Words

```python
def count_distinct_words(self) -> int:
    return self._count_words(self.root)
```

## Get All Words in Trie (DFS)

```python
def get_all_words(self) -> list[str]:
    result = []
    def dfs(node, path):
        if node.is_end:
            result.append("".join(path))
        for char, child in sorted(node.children.items()):
            path.append(char)
            dfs(child, path)
            path.pop()
    dfs(self.root, [])
    return result
```

## Get All Words with Given Prefix

```python
def get_words_with_prefix(self, prefix: str) -> list[str]:
    node = self.root
    for char in prefix:
        if char not in node.children:
            return []
        node = node.children[char]
    result = []
    def dfs(n, path):
        if n.is_end:
            result.append(prefix + "".join(path))
        for char, child in sorted(n.children.items()):
            path.append(char)
            dfs(child, path)
            path.pop()
    dfs(node, [])
    return result
```

## Check If Any Word Starts With Prefix

Same as `startsWith` - returns True if at least one word has the given prefix.

```python
def has_prefix(self, prefix: str) -> bool:
    return self.starts_with(prefix)
```

## Longest Word in Trie

For "longest word where each prefix is also a word" (e.g. LeetCode 720): only extend to children that are word ends.

```python
def longest_word(self) -> str:
    result = [""]
    def dfs(node, path):
        if node.is_end and len(path) > len(result[0]):
            result[0] = "".join(path)
        for char, child in sorted(node.children.items()):
            if child.is_end:
                path.append(char)
                dfs(child, path)
                path.pop()
    dfs(self.root, [])
    return result[0]
```

## Shortest Unique Prefix for Each Word

For each word, find the minimum prefix that uniquely identifies it among all words in the trie.

```python
def shortest_unique_prefixes(self, words: list[str]) -> dict[str, str]:
    trie = Trie()
    for word in words:
        trie.insert(word)
    result = {}
    for word in words:
        node = trie.root
        prefix = []
        for i, char in enumerate(word):
            prefix.append(char)
            node = node.children[char]
            count = trie._count_words(node)
            if count == 1 or node.is_end:
                result[word] = "".join(prefix)
                break
    return result
```

## isEmpty

```python
def is_empty(self) -> bool:
    return len(self.root.children) == 0
```

## Size (Number of Words)

```python
def size(self) -> int:
    return self.count_distinct_words()
```
