# Trie - Pattern Matching

## Wildcard Search (. Matches Any Single Char)

Use DFS with branching when encountering '.'. Each '.' can match any character, so try all children.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            c = word[i]
            if c == '.':
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            if c not in node.children:
                return False
            return dfs(node.children[c], i + 1)
        return dfs(self.root, 0)
```

## Prefix Matching (Find All Words with Prefix)

```python
def prefix_match(self, prefix: str) -> list[str]:
    node = self.root
    for c in prefix:
        if c not in node.children:
            return []
        node = node.children[c]
    result = []
    def dfs(n, path):
        if n.is_end:
            result.append(prefix + "".join(path))
        for c, child in n.children.items():
            path.append(c)
            dfs(child, path)
            path.pop()
    dfs(node, [])
    return result
```

## Suffix Trie (Insert All Suffixes, Substring Queries)

Insert every suffix of the string. Enables O(m) substring search for pattern of length m.

```python
class SuffixTrie:
    def __init__(self, text: str):
        self.root = {}
        for i in range(len(text)):
            self._insert_suffix(text[i:], i)

    def _insert_suffix(self, suffix: str, start: int) -> None:
        node = self.root
        for c in suffix:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = start

    def search_substring(self, pattern: str) -> list[int]:
        node = self.root
        for c in pattern:
            if c not in node:
                return []
            node = node[c]
        starts = []
        def collect(n):
            if '$' in n:
                starts.append(n['$'])
            for k, v in n.items():
                if k != '$':
                    collect(v)
        collect(node)
        return starts
```

## Longest Common Prefix Using Trie

```python
def longest_common_prefix(self, words: list[str]) -> str:
    if not words:
        return ""
    trie = Trie()
    for w in words:
        trie.insert(w)
    node = trie.root
    prefix = []
    while len(node.children) == 1 and not node.is_end:
        c = next(iter(node.children))
        prefix.append(c)
        node = node.children[c]
    return "".join(prefix)
```

## Search for Word with One Character Difference

```python
def search_one_diff(self, word: str) -> bool:
    def dfs(node, i, diff_count):
        if i == len(word):
            return node.is_end and diff_count == 1
        if diff_count > 1:
            return False
        c = word[i]
        for ch, child in node.children.items():
            new_diff = diff_count + (1 if ch != c else 0)
            if dfs(child, i + 1, new_diff):
                return True
        return False
    return dfs(self.root, 0, 0)
```

## Word Search II (Grid + Multiple Patterns - Trie + Backtracking)

```python
def find_words(self, board: list[list[str]], words: list[str]) -> list[str]:
    trie = Trie()
    for w in words:
        trie.insert(w)
    rows, cols = len(board), len(board[0])
    result = set()

    def dfs(r, c, node, path):
        if node.is_end:
            result.add("".join(path))
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] == '#':
            return
        ch = board[r][c]
        if ch not in node.children:
            return
        child = node.children[ch]
        path.append(ch)
        board[r][c] = '#'
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(r+dr, c+dc, child, path)
        board[r][c] = ch
        path.pop()

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, [])
    return list(result)
```

## Stream of Characters (Check Suffixes - Reverse Trie)

Store words in reverse. For each new character, maintain current suffix and check if any reversed word matches.

```python
class StreamChecker:
    def __init__(self, words: list[str]):
        self.trie = {}
        for w in words:
            node = self.trie
            for c in reversed(w):
                if c not in node:
                    node[c] = {}
                node = node[c]
            node['$'] = True
        self.stream = []

    def query(self, letter: str) -> bool:
        self.stream.append(letter)
        node = self.trie
        for i in range(len(self.stream) - 1, -1, -1):
            c = self.stream[i]
            if c not in node:
                return False
            node = node[c]
            if '$' in node:
                return True
        return False
```

## CamelCase Matching

Match pattern where pattern letters must appear in order with same case; lowercase in pattern can match multiple lowercase chars.

```python
def camel_match(self, queries: list[str], pattern: str) -> list[bool]:
    def match(query: str, pattern: str) -> bool:
        j = 0
        for c in query:
            if j < len(pattern) and c == pattern[j]:
                j += 1
            elif c.isupper():
                return False
        return j == len(pattern)
    return [match(q, pattern) for q in queries]
```

## Map Sum Pairs (Trie with Values)

```python
class MapSum:
    def __init__(self):
        self.root = {}
        self.vals = {}

    def insert(self, key: str, val: int) -> None:
        delta = val - self.vals.get(key, 0)
        self.vals[key] = val
        node = self.root
        for c in key:
            if c not in node:
                node[c] = {}
            node = node[c]
            node['#'] = node.get('#', 0) + delta
        node['#'] = node.get('#', 0) + delta

    def sum(self, prefix: str) -> int:
        node = self.root
        for c in prefix:
            if c not in node:
                return 0
            node = node[c]
        return node.get('#', 0)
```

## Prefix and Suffix Search

Design a structure that finds word with given prefix and suffix. Store for each word, for each suffix s: key = s + '#' + word. Search key = suffix + '#' + prefix.

```python
class WordFilter:
    def __init__(self, words: list[str]):
        self.trie = {}
        for idx, word in enumerate(words):
            for i in range(len(word) + 1):
                key = word[i:] + '#' + word
                node = self.trie
                for c in key:
                    if c not in node:
                        node[c] = {}
                    node = node[c]
                    node['$'] = idx

    def f(self, prefix: str, suffix: str) -> int:
        key = suffix + '#' + prefix
        node = self.trie
        for c in key:
            if c not in node:
                return -1
            node = node[c]
        return node.get('$', -1)
```
