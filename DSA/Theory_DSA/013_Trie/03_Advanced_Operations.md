# Trie - Advanced Operations

## Implement Trie Class (Insert, Search, startsWith, Delete)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def delete(self, word: str) -> bool:
        def _delete(node, word, i):
            if i == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0
            char = word[i]
            if char not in node.children:
                return False
            if _delete(node.children[char], word, i + 1):
                del node.children[char]
                return len(node.children) == 0 and not node.is_end
            return False
        return _delete(self.root, word, 0)
```

## Auto-Complete (Return All Words with Prefix Sorted)

```python
def autocomplete_sorted(self, prefix: str) -> list[str]:
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

## Auto-Complete Top-K by Frequency

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.freq = 0

class AutocompleteTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, freq: int = 1) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.freq += freq

    def autocomplete_top_k(self, prefix: str, k: int) -> list[str]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        candidates = []
        def dfs(n, path):
            if n.is_end:
                candidates.append((n.freq, prefix + "".join(path)))
            for char, child in n.children.items():
                path.append(char)
                dfs(child, path)
                path.pop()
        dfs(node, [])
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return [w for _, w in candidates[:k]]
```

## Spell Checker (Suggest Corrections Edit Distance 1-2)

```python
def spell_check_suggestions(self, word: str, max_edit: int = 2) -> list[str]:
    def edit_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    all_words = self.get_all_words()
    suggestions = []
    for w in all_words:
        if abs(len(w) - len(word)) <= max_edit:
            d = edit_distance(word, w)
            if d <= max_edit and d > 0:
                suggestions.append((d, w))
    suggestions.sort(key=lambda x: (x[0], x[1]))
    return [w for _, w in suggestions[:10]]
```

## Longest Common Prefix of All Words Using Trie

```python
def longest_common_prefix_trie(self, words: list[str]) -> str:
    if not words:
        return ""
    trie = Trie()
    for w in words:
        trie.insert(w)
    node = trie.root
    prefix = []
    while len(node.children) == 1 and not node.is_end:
        char = next(iter(node.children))
        prefix.append(char)
        node = node.children[char]
    return "".join(prefix)
```

## Word Frequency Counter Using Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.freq = 0

def word_frequency_trie(self, words: list[str]) -> dict[str, int]:
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.freq += 1
    result = {}
    def dfs(n, path):
        if n.freq > 0:
            result["".join(path)] = n.freq
        for char, child in n.children.items():
            path.append(char)
            dfs(child, path)
            path.pop()
    dfs(root, [])
    return result
```

## Trie with Frequency Counts at Nodes

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.freq = 0

class FreqTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, count: int = 1) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.freq += count
```

## Trie-Based Sorting (Insert All, DFS in Order)

```python
def trie_sort(self, words: list[str]) -> list[str]:
    trie = Trie()
    for w in words:
        trie.insert(w)
    return trie.get_all_words()
```

## Trie with Wildcard Search (. Matches Any)

```python
def search_wildcard(self, word: str) -> bool:
    def dfs(node, i):
        if i == len(word):
            return node.is_end
        char = word[i]
        if char == '.':
            for child in node.children.values():
                if dfs(child, i + 1):
                    return True
            return False
        if char not in node.children:
            return False
        return dfs(node.children[char], i + 1)
    return dfs(self.root, 0)
```

## Lexicographic Ordering

```python
def get_sorted_words(self) -> list[str]:
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

## Print All Words Sorted

```python
def print_sorted(self) -> None:
    for word in self.get_sorted_words():
        print(word)
```

## Count Distinct Substrings of a String Using Trie

```python
def count_distinct_substrings(self, s: str) -> int:
    root = {}
    count = 0
    for i in range(len(s)):
        node = root
        for j in range(i, len(s)):
            char = s[j]
            if char not in node:
                node[char] = {}
                count += 1
            node = node[char]
    return count
```

## Palindrome Pairs Using Trie

Store reversed words in trie. For each word, traverse trie; when we reach a node that is end of a reversed word, check if remainder of current word is palindrome. When word exhausted, collect all words in subtree whose path from current node is palindrome.

```python
def palindrome_pairs(words: list[str]) -> list[tuple[int, int]]:
    trie = {}
    for i, w in enumerate(words):
        node = trie
        for c in reversed(w):
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = i

    def is_pal(s):
        return s == s[::-1]

    result = []
    for i, w in enumerate(words):
        node = trie
        for j, c in enumerate(w):
            if '$' in node and is_pal(w[j:]) and node['$'] != i:
                result.append((i, node['$']))
            if c not in node:
                break
            node = node[c]
        else:
            def collect(n, path):
                if '$' in n and n['$'] != i and is_pal(path):
                    result.append((i, n['$']))
                for k, v in n.items():
                    if k != '$':
                        collect(v, path + k)
            collect(node, "")
    return result
```

## Compressed Trie Operations

Compressed trie merges chains of single-child nodes into edges that store substrings.

```python
class CompressedTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.edge_label = ""

class CompressedTrie:
    def __init__(self):
        self.root = CompressedTrieNode()

    def insert(self, word: str) -> None:
        if not word:
            self.root.is_end = True
            return
        node = self.root
        i = 0
        while i < len(word):
            matched = False
            for edge, child in node.children.items():
                j = 0
                while j < len(edge) and i + j < len(word) and edge[j] == word[i + j]:
                    j += 1
                if j > 0:
                    if j == len(edge):
                        node = child
                        i += j
                        matched = True
                    else:
                        mid = CompressedTrieNode()
                        mid.children[edge[j:]] = child
                        mid.children[word[i+j:]] = CompressedTrieNode()
                        mid.children[word[i+j:]].is_end = True
                        del node.children[edge]
                        node.children[edge[:j]] = mid
                        return
                    break
            if not matched:
                node.children[word[i:]] = CompressedTrieNode()
                node.children[word[i:]].is_end = True
                return
        node.is_end = True
```
