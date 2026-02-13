# Trie - Applications

## Autocomplete System (Design Search Autocomplete Top-K by Frequency)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.freq = 0

class AutocompleteSystem:
    def __init__(self, sentences: list[str], times: list[int]):
        self.root = TrieNode()
        self.cur_input = ""
        for s, t in zip(sentences, times):
            self._insert(s, t)

    def _insert(self, s: str, delta: int = 1) -> None:
        node = self.root
        for c in s:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True
        node.freq += delta

    def input(self, c: str) -> list[str]:
        if c == '#':
            self._insert(self.cur_input)
            self.cur_input = ""
            return []
        self.cur_input += c
        node = self.root
        for ch in self.cur_input:
            if ch not in node.children:
                return []
            node = node.children[ch]
        candidates = []
        def dfs(n, path):
            if n.is_end:
                candidates.append((-n.freq, self.cur_input + "".join(path)))
            for ch, child in n.children.items():
                path.append(ch)
                dfs(child, path)
                path.pop()
        dfs(node, [])
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [w for _, w in candidates[:3]]
```

## Spell Checker (Words Within Edit Distance)

```python
def spell_check(self, word: str, dictionary: list[str], max_edit: int = 2) -> list[str]:
    trie = Trie()
    for w in dictionary:
        trie.insert(w)
    result = []
    def dfs(node, remain, path, edits):
        if edits > max_edit:
            return
        if not remain:
            if node.is_end:
                result.append("".join(path))
            return
        c = remain[0]
        if c in node.children:
            path.append(c)
            dfs(node.children[c], remain[1:], path, edits)
            path.pop()
        for ch, child in node.children.items():
            path.append(ch)
            dfs(child, remain[1:], path, edits + 1)
            path.pop()
            path.append(ch)
            dfs(child, remain, path, edits + 1)
            path.pop()
    dfs(trie.root, word, [], 0)
    return result[:10]
```

## IP Routing / Longest Prefix Match

```python
class IPRouter:
    def __init__(self):
        self.trie = {}
        self.routes = {}

    def add_route(self, prefix: str, next_hop: str) -> None:
        node = self.trie
        for c in prefix:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = next_hop

    def lookup(self, ip: str) -> str:
        node = self.trie
        result = None
        for c in ip:
            if '$' in node:
                result = node['$']
            if c not in node:
                break
            node = node[c]
        if '$' in node:
            result = node['$']
        return result or "default"
```

## Word Search II (Find All Dictionary Words in Grid)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    root = TrieNode()
    for w in words:
        node = root
        for c in w:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

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
            dfs(r, c, root, [])
    return list(result)
```

## Search Suggestions System

```python
import bisect

def suggested_products(products: list[str], search_word: str) -> list[list[str]]:
    products.sort()
    result = []
    prefix = ""
    for c in search_word:
        prefix += c
        start = bisect.bisect_left(products, prefix)
        suggestions = []
        for i in range(start, min(start + 3, len(products))):
            if products[i].startswith(prefix):
                suggestions.append(products[i])
            else:
                break
        result.append(suggestions)
    return result
```

## Replace Words (Replace with Shortest Root)

```python
def replace_words(dictionary: list[str], sentence: str) -> str:
    trie = {}
    for word in dictionary:
        node = trie
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = word

    def get_root(word):
        node = trie
        for i, c in enumerate(word):
            if '$' in node:
                return node['$']
            if c not in node:
                return word
            node = node[c]
        return node.get('$', word)

    return " ".join(get_root(w) for w in sentence.split())
```

## Implement Magic Dictionary (One Char Different)

```python
class MagicDictionary:
    def __init__(self):
        self.words = set()
        self.trie = {}

    def build_dict(self, dictionary: list[str]) -> None:
        for w in dictionary:
            self.words.add(w)
            node = self.trie
            for c in w:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node['$'] = True

    def search(self, word: str) -> bool:
        def dfs(node, i, changed):
            if i == len(word):
                return '$' in node and changed
            c = word[i]
            if changed:
                if c not in node:
                    return False
                return dfs(node[c], i + 1, True)
            for ch, child in node.items():
                if ch == '$':
                    continue
                if ch == c:
                    if dfs(child, i + 1, False):
                        return True
                else:
                    if dfs(child, i + 1, True):
                        return True
            return False
        return dfs(self.trie, 0, False)
```

## Word Squares

```python
def word_squares(words: list[str]) -> list[list[str]]:
    n = len(words[0]) if words else 0
    trie = {}
    for w in words:
        node = trie
        for c in w:
            if c not in node:
                node[c] = {}
            node = node[c]
            if '#' not in node:
                node['#'] = []
            node['#'].append(w)

    def get_prefix_words(prefix):
        node = trie
        for c in prefix:
            if c not in node:
                return []
            node = node[c]
        return node.get('#', [])

    result = []
    def backtrack(square):
        if len(square) == n:
            result.append(square[:])
            return
        prefix = "".join(s[len(square)] for s in square)
        for w in get_prefix_words(prefix):
            square.append(w)
            backtrack(square)
            square.pop()

    for w in words:
        backtrack([w])
    return result
```

## Palindrome Pairs (Trie Approach)

```python
def palindrome_pairs(words: list[str]) -> list[list[int]]:
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
            if '$' in node and is_pal(w[j:]):
                if i != node['$']:
                    result.append([i, node['$']])
            if c not in node:
                break
            node = node[c]
        else:
            def collect(n, path):
                if '$' in n and n['$'] != i:
                    result.append([i, n['$']])
                for k, v in n.items():
                    if k != '$':
                        collect(v, path + k)
            collect(node, "")
    return result
```

## Maximum XOR of Two Numbers (Bitwise Trie)

```python
class BinaryTrie:
    def __init__(self):
        self.root = {}

    def insert(self, num: int) -> None:
        node = self.root
        for i in range(31, -1, -1):
            b = (num >> i) & 1
            if b not in node:
                node[b] = {}
            node = node[b]

    def max_xor(self, num: int) -> int:
        node = self.root
        res = 0
        for i in range(31, -1, -1):
            b = (num >> i) & 1
            want = 1 - b
            if want in node:
                res |= (1 << i)
                node = node[want]
            else:
                node = node[b]
        return res

def find_maximum_xor(nums: list[int]) -> int:
    trie = BinaryTrie()
    for n in nums:
        trie.insert(n)
    return max(trie.max_xor(n) for n in nums)
```

## Maximum XOR with Element from Array

```python
def maximize_xor(nums: list[int], queries: list[list[int]]) -> list[int]:
    trie = BinaryTrie()
    nums.sort()
    qs = sorted(enumerate(queries), key=lambda x: x[1][1])
    result = [0] * len(queries)
    idx = 0
    for i, (xi, mi) in qs:
        while idx < len(nums) and nums[idx] <= mi:
            trie.insert(nums[idx])
            idx += 1
        result[i] = trie.max_xor(xi) if idx > 0 else -1
    return result
```

## Concatenated Words

```python
def find_all_concatenated_words_in_a_dict(words: list[str]) -> list[str]:
    trie = {}
    for w in words:
        node = trie
        for c in w:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = True

    def can_form(word, count=0):
        if not word and count >= 2:
            return True
        node = trie
        for i, c in enumerate(word):
            if c not in node:
                return False
            node = node[c]
            if '$' in node and can_form(word[i+1:], count + 1):
                return True
        return False

    return [w for w in words if can_form(w)]
```

## Word Break (Trie Approach)

```python
def word_break(s: str, word_dict: list[str]) -> bool:
    trie = {}
    for w in word_dict:
        node = trie
        for c in w:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['$'] = True

    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(n):
        if not dp[i]:
            continue
        node = trie
        for j in range(i, n):
            if s[j] not in node:
                break
            node = node[s[j]]
            if '$' in node:
                dp[j + 1] = True
    return dp[n]
```
