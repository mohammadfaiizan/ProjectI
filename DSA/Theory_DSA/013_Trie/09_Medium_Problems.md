# Trie - Medium Problems

## 01. Word Search II

**Description**: Given 2D board and list of words, find all words that exist in board (adjacent cells, no reuse).

**Approach**: Build trie from words. Backtrack on grid: from each cell, traverse trie; when is_end, add word; prune when no child for current char.

```python
def findWords(board, words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = w
    res, m, n = set(), len(board), len(board[0])
    def dfs(r, c, node):
        if '#' in node:
            res.add(node['#'])
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] not in node:
            return
        ch, board[r][c] = board[r][c], '#'
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(r+dr, c+dc, node[ch])
        board[r][c] = ch
    for i in range(m):
        for j in range(n):
            dfs(i, j, trie)
    return list(res)
```

Time: O(m*n*4^L) | Space: O(W)

---

## 02. Design Add and Search Words Data Structure

**Description**: Add words and search with '.' wildcard.

**Approach**: Trie + DFS for wildcard. When '.', try all children.

```python
def search_wildcard(node, word, i):
    if i == len(word):
        return node.get('is_end', False)
    c = word[i]
    if c == '.':
        return any(search_wildcard(child, word, i+1) for k, child in node.items() if k != 'is_end')
    if c not in node:
        return False
    return search_wildcard(node[c], word, i+1)
```

Time: O(m) insert, O(26^m) search | Space: O(n)

---

## 03. Prefix and Suffix Search

**Description**: Design structure to find word with given prefix and suffix. Return largest index.

**Approach**: For each word, store suffix + '#' + word for every suffix. Query: suffix + '#' + prefix. Store max index at each path.

```python
def build_prefix_suffix(words):
    trie = {}
    for idx, w in enumerate(words):
        for i in range(len(w), -1, -1):
            key = w[i:] + '#' + w
            node = trie
            for c in key:
                node = node.setdefault(c, {})
                node['$'] = idx
    return trie

def search_prefix_suffix(trie, prefix, suffix):
    key = suffix + '#' + prefix
    node = trie
    for c in key:
        if c not in node:
            return -1
        node = node[c]
    return node.get('$', -1)
```

Time: O(W*L^2) build, O(P+S) query | Space: O(W*L^2)

---

## 04. Implement Magic Dictionary

**Description**: Build dict, search if any word matches with exactly one character different.

**Approach**: Trie. Search with one allowed mismatch: when chars match continue; when different, recurse with changed=True.

```python
def magic_search(node, word, i, changed):
    if i == len(word):
        return changed and '#' in node
    c = word[i]
    if c in node and magic_search(node[c], word, i+1, changed):
        return True
    if not changed:
        for k in node:
            if k != '#' and k != c and magic_search(node[k], word, i+1, True):
                return True
    return False
```

Time: O(n*26^m) | Space: O(n)

---

## 05. Replace Words

**Description**: Replace words in sentence with shortest root from dictionary.

**Approach**: Trie from roots. For each word, traverse until is_end, replace with path.

```python
def replaceWords(dictionary, sentence):
    trie = {}
    for root in dictionary:
        node = trie
        for c in root:
            node = node.setdefault(c, {})
        node['#'] = root
    def replace(word):
        node = trie
        for c in word:
            if '#' in node:
                return node['#']
            if c not in node:
                return word
            node = node[c]
        return word
    return ' '.join(replace(w) for w in sentence.split())
```

Time: O(d + s) | Space: O(d)

---

## 06. Map Sum Pairs

**Description**: Insert (key, val), sum(prefix) returns sum of values for keys with prefix.

**Approach**: Trie with value at node. Store delta on insert. Sum = sum of values in prefix subtree.

```python
class MapSum:
    def __init__(self):
        self.trie = {}
        self.vals = {}
    def insert(self, key, val):
        delta = val - self.vals.get(key, 0)
        self.vals[key] = val
        node = self.trie
        for c in key:
            node = node.setdefault(c, {})
            node['$'] = node.get('$', 0) + delta
    def sum(self, prefix):
        node = self.trie
        for c in prefix:
            if c not in node:
                return 0
            node = node[c]
        return node.get('$', 0)
```

Time: O(k) per op | Space: O(n)

---

## 07. Word Squares

**Description**: Form word squares where k-th row and column read same string.

**Approach**: Trie stores words by prefix. Backtrack: build square row by row; for row i, prefix = column i of current rows; get words with that prefix from trie.

```python
def wordSquares(words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node.setdefault('#', []).append(w)
    def get_words(prefix):
        node = trie
        for c in prefix:
            if c not in node:
                return []
            node = node[c]
        return node.get('#', [])
    def backtrack(square):
        if len(square) == len(square[0]) if square else 0:
            res.append(square[:])
            return
        prefix = ''.join(row[len(square)] for row in square)
        for w in get_words(prefix):
            backtrack(square + [w])
    res = []
    for w in words:
        backtrack([w])
    return res
```

Time: O(n * 26^L) | Space: O(n)

---

## 08. Palindrome Pairs

**Description**: Find all pairs (i, j) where words[i] + words[j] is palindrome.

**Approach**: Trie of reversed words. For each word, traverse trie; when at is_end, check if remainder of word is palindrome. Also check words in subtree when current word exhausted.

```python
def palindromePairs(words):
    trie = {}
    for i, w in enumerate(words):
        node = trie
        for c in reversed(w):
            node = node.setdefault(c, {})
        node.setdefault('#', []).append((i, len(w)))
    def is_pal(s):
        return s == s[::-1]
    res = []
    for i, w in enumerate(words):
        node = trie
        for j, c in enumerate(w):
            if '#' in node and j > 0 and is_pal(w[j:]):
                for k, _ in node['#']:
                    if k != i:
                        res.append([i, k])
            if c not in node:
                break
            node = node[c]
        else:
            if '#' in node:
                for k, _ in node['#']:
                    if k != i and is_pal(w + words[k]):
                        res.append([i, k])
    return res
```

Time: O(n * L^2) | Space: O(n*L)

---

## 09. Maximum XOR of Two Numbers in an Array

**Description**: Find maximum XOR of any two numbers in array.

**Approach**: Bitwise trie. Insert numbers as 32-bit binary. For each number, traverse trie choosing opposite bit when available to maximize XOR.

```python
def findMaximumXOR(nums):
    trie = {}
    for x in nums:
        node = trie
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            node = node.setdefault(b, {})
    res = 0
    for x in nums:
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur <<= 1
            if want in node:
                node, cur = node[want], cur + 1
            else:
                node = node[b]
        res = max(res, cur)
    return res
```

Time: O(n * 32) | Space: O(n * 32)

---

## 10. Maximum XOR With an Element From Array

**Description**: For each query (x, m), find max XOR of x with any array element <= m.

**Approach**: Sort queries by m. Process in order, inserting numbers into trie as m increases. Query max XOR for x.

```python
def maximizeXor(nums, queries):
    nums.sort()
    qs = sorted(enumerate(queries), key=lambda x: x[1][1])
    trie, res, j = {}, [0]*len(queries), 0
    for idx, (x, m) in qs:
        while j < len(nums) and nums[j] <= m:
            node = trie
            for i in range(31, -1, -1):
                b = (nums[j] >> i) & 1
                node = node.setdefault(b, {})
            j += 1
        if not trie:
            res[idx] = -1
            continue
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur <<= 1
            if want in node:
                node, cur = node[want], cur + 1
            else:
                node = node[b]
        res[idx] = cur
    return res
```

Time: O((n+q)*32) | Space: O(n*32)

---

## 11. Search Suggestions System

**Description**: As user types, suggest top 3 products (by prefix match).

**Approach**: Sort products. Binary search for prefix. Or trie with DFS to collect, sort by relevance.

```python
def suggestedProducts(products, searchWord):
    import bisect
    products.sort()
    res, prefix, i = [], "", 0
    for c in searchWord:
        prefix += c
        i = bisect.bisect_left(products, prefix, i)
        res.append([p for p in products[i:i+3] if p.startswith(prefix)])
    return res
```

Time: O(n log n + m) | Space: O(1)

---

## 12. Stream of Characters

**Description**: Check if any suffix of stream is in words.

**Approach**: Trie of reversed words. Maintain stream. For each new char, check reversed suffixes.

```python
class StreamChecker:
    def __init__(self, words):
        self.trie = {}
        for w in words:
            node = self.trie
            for c in reversed(w):
                node = node.setdefault(c, {})
            node['#'] = True
        self.buf = []
    def query(self, letter):
        self.buf.append(letter)
        node = self.trie
        for c in reversed(self.buf):
            if c not in node:
                return False
            node = node[c]
            if '#' in node:
                return True
        return False
```

Time: O(m) per query | Space: O(W)

---

## 13. Word Break

**Description**: Determine if string can be segmented into dictionary words.

**Approach**: Trie + DP. Trie for fast prefix lookup. dp[i] = can segment s[:i]. For each i where dp[i], try all prefixes starting at i via trie.

```python
def wordBreak(s, wordDict):
    trie = {}
    for w in wordDict:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = True
    dp = [False] * (len(s)+1)
    dp[0] = True
    for i in range(len(s)):
        if not dp[i]:
            continue
        node = trie
        for j in range(i, len(s)):
            if s[j] not in node:
                break
            node = node[s[j]]
            if '#' in node:
                dp[j+1] = True
    return dp[-1]
```

Time: O(n^2) | Space: O(n + W)

---

## 14. Concatenated Words

**Description**: Find all words that can be formed by concatenating two or more shorter words from same list.

**Approach**: Trie of all words. For each word, DFS: at each position check if prefix is word and remainder can be formed.

```python
def findAllConcatenatedWordsInADict(words):
    words_set = set(words)
    def can_form(w, count=0):
        if not w and count >= 2:
            return True
        for i in range(1, len(w)+1):
            if w[:i] in words_set and can_form(w[i:], count+1):
                return True
        return False
    return [w for w in words if can_form(w)]
```

Time: O(n * L^2) | Space: O(n)

---

## 15. Add and Search Word - Data Structure Design

**Description**: Same as Design Add and Search Words. Add word, search with '.' wildcard.

**Approach**: Trie with DFS for wildcard search.

```python
def search_word(node, word, i):
    if i == len(word):
        return node.get('is_end', False)
    c = word[i]
    if c == '.':
        return any(search_word(child, word, i+1) for k, child in node.items() if k != 'is_end')
    return c in node and search_word(node[c], word, i+1)
```

Time: O(26^m) worst search | Space: O(n)

---

## 16. Shortest Word Distance II

**Description**: Design class with list of words. Query shortest distance between two words.

**Approach**: HashMap word to sorted list of indices. Two pointers or binary search for min |i - j|.

```python
def shortestDistance2(words, word1, word2):
    idx = {}
    for i, w in enumerate(words):
        idx.setdefault(w, []).append(i)
    l1, l2 = idx[word1], idx[word2]
    i, j, res = 0, 0, float('inf')
    while i < len(l1) and j < len(l2):
        res = min(res, abs(l1[i] - l2[j]))
        if l1[i] < l2[j]:
            i += 1
        else:
            j += 1
    return res
```

Time: O(n) build, O(a+b) query | Space: O(n)

---

## 17. Shortest Word Distance III

**Description**: Shortest distance between word1 and word2 when they can be same.

**Approach**: Track last indices of both. When same word, use prev and current.

```python
def shortestDistance3(words, word1, word2):
    i1, i2, res = -1, -1, float('inf')
    for i, w in enumerate(words):
        if w == word1:
            if word1 == word2:
                i1, i2 = i2, i
            else:
                i1 = i
        elif w == word2:
            i2 = i
        if i1 != -1 and i2 != -1:
            res = min(res, abs(i1 - i2))
    return res
```

Time: O(n) | Space: O(1)

---

## 18. CamelCase Matching

**Description**: For each query, check if it matches pattern (uppercase must match, lowercase can match multiple).

**Approach**: For each query, two pointers: pattern pointer advances when match; if query has uppercase not matching, return false.

```python
def camelMatch(queries, pattern):
    def match(q):
        j = 0
        for c in q:
            if j < len(pattern) and c == pattern[j]:
                j += 1
            elif c.isupper():
                return False
        return j == len(pattern)
    return [match(q) for q in queries]
```

Time: O(n * m) | Space: O(1)

---

## 19. Count Pairs With Given XOR

**Description**: Count pairs (i, j) with nums[i] XOR nums[j] == target.

**Approach**: For each x, need y = x XOR target. Use HashMap to count. Trie for range XOR variants.

```python
def countPairs(nums, target):
    from collections import Counter
    c = Counter(nums)
    return sum(c.get(x ^ target, 0) for x in nums) // 2
```

Time: O(n) | Space: O(n)

---

## 20. Substring With Concatenation of All Words

**Description**: Find all starting indices where substring is concatenation of all words (each exactly once).

**Approach**: Sliding window with word length. HashMap word counts. Trie can store words for matching.

```python
def findSubstring(s, words):
    from collections import Counter
    n, m, k = len(s), len(words[0]), len(words)
    target = Counter(words)
    res = []
    for i in range(n - m * k + 1):
        seen = Counter()
        for j in range(k):
            w = s[i + j*m : i + (j+1)*m]
            if w not in target:
                break
            seen[w] += 1
            if seen[w] > target[w]:
                break
        else:
            res.append(i)
    return res
```

Time: O(n * m) | Space: O(k)

---

## 21. Word Ladder

**Description**: Shortest transformation from begin to end word, changing one letter at a time, each step must be in word list.

**Approach**: BFS. Trie can optimize neighbor finding: for each position, try all 26 letters, check if in trie.

```python
def ladderLength(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return 0
    q, dist = [beginWord], 1
    while q:
        nq = []
        for w in q:
            if w == endWord:
                return dist
            for i in range(len(w)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = w[:i] + c + w[i+1:]
                    if nw in words:
                        words.discard(nw)
                        nq.append(nw)
        q, dist = nq, dist + 1
    return 0
```

Time: O(n * m * 26) | Space: O(n)

---

## 22. Word Ladder II

**Description**: Find all shortest transformation sequences from begin to end.

**Approach**: BFS to find distance, then DFS to reconstruct paths. Trie for neighbor lookup.

```python
def findLadders(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words:
        return []
    layer = {beginWord: [[beginWord]]}
    while layer:
        nlayer = {}
        for w in layer:
            if w == endWord:
                return layer[w]
            for i in range(len(w)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = w[:i] + c + w[i+1:]
                    if nw in words:
                        nlayer.setdefault(nw, []).extend(path + [nw] for path in layer[w])
        words -= set(nlayer.keys())
        layer = nlayer
    return []
```

Time: O(n * m * 26) | Space: O(n * paths)

---

## 23. Group Anagrams

**Description**: Group words that are anagrams.

**Approach**: Sort each word as key, group. Trie can store by sorted form.

```python
def groupAnagrams(strs):
    from collections import defaultdict
    d = defaultdict(list)
    for s in strs:
        d[tuple(sorted(s))].append(s)
    return list(d.values())
```

Time: O(n * k log k) | Space: O(n)

---

## 24. Longest Word in Dictionary Through Deleting

**Description**: Find longest word in dictionary that is subsequence of string s.

**Approach**: For each dict word, check if subsequence of s. Sort by length desc and lexicographic. Trie: build from s for subsequence matching.

```python
def findLongestWord(s, dictionary):
    def is_subseq(w):
        j = 0
        for c in s:
            if j < len(w) and c == w[j]:
                j += 1
        return j == len(w)
    dictionary.sort(key=lambda x: (-len(x), x))
    for w in dictionary:
        if is_subseq(w):
            return w
    return ""
```

Time: O(n * L) | Space: O(1)

---

## 25. Number of Matching Subsequences

**Description**: Count how many words in words are subsequences of string s.

**Approach**: Preprocess s: for each char, list indices. For each word, binary search next occurrence. Trie of subsequences possible.

```python
def numMatchingSubseq(s, words):
    import bisect
    from collections import defaultdict
    idx = defaultdict(list)
    for i, c in enumerate(s):
        idx[c].append(i)
    def is_subseq(w):
        cur = -1
        for c in w:
            pos = idx[c]
            lo = bisect.bisect_right(pos, cur)
            if lo >= len(pos):
                return False
            cur = pos[lo]
        return True
    return sum(is_subseq(w) for w in words)
```

Time: O(s + n * L log s) | Space: O(s)

---

## Hard Problems

## H01. Word Search II

**Description**: Find all words from dictionary in 2D grid. Same as medium but often classified hard.

**Approach**: Trie + backtracking. Build trie, prune during backtrack when no words with current prefix.

```python
def findWords(board, words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = w
    res, m, n = set(), len(board), len(board[0])
    def dfs(r, c, node):
        if '#' in node:
            res.add(node['#'])
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] not in node:
            return
        ch, board[r][c] = board[r][c], '#'
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(r+dr, c+dc, node[ch])
        board[r][c] = ch
    for i in range(m):
        for j in range(n):
            dfs(i, j, trie)
    return list(res)
```

Time: O(m*n*4^L) | Space: O(W)

---

## H02. Palindrome Pairs

**Description**: Find all (i, j) where words[i] + words[j] is palindrome.

**Approach**: Trie of reversed words. For each word, traverse trie; at each is_end check remainder palindrome; when word exhausted, collect all words in subtree with palindrome suffix.

```python
def palindromePairs(words):
    d = {w: i for i, w in enumerate(words)}
    res = []
    for i, w in enumerate(words):
        for j in range(len(w)+1):
            pre, suf = w[:j], w[j:]
            if pre[::-1] in d and d[pre[::-1]] != i and suf == suf[::-1]:
                res.append([i, d[pre[::-1]]])
            if j > 0 and suf[::-1] in d and d[suf[::-1]] != i and pre == pre[::-1]:
                res.append([d[suf[::-1]], i])
    return res
```

Time: O(n * L^2) | Space: O(n)

---

## H03. Maximum XOR of Two Numbers in an Array

**Description**: Max XOR of any two numbers.

**Approach**: Binary trie. Insert bits MSB first. For each number, greedily choose opposite bit.

```python
def findMaximumXOR(nums):
    trie = {}
    for x in nums:
        node = trie
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            node = node.setdefault(b, {})
    res = 0
    for x in nums:
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur = (cur << 1) + (1 if want in node else 0)
            node = node[want] if want in node else node[b]
        res = max(res, cur)
    return res
```

Time: O(n * 32) | Space: O(n * 32)

---

## H04. Word Squares

**Description**: Form NxN grid where each row and column is a valid word.

**Approach**: Trie by prefix. Backtrack row by row. Prefix for row i = column i of current grid. Get candidates from trie.

```python
def wordSquares(words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node.setdefault('#', []).append(w)
    def get(prefix):
        node = trie
        for c in prefix:
            if c not in node:
                return []
            node = node[c]
        return node.get('#', [])
    def bt(sq):
        if len(sq) == len(sq[0]) if sq else 0:
            res.append(sq[:])
            return
        prefix = ''.join(row[len(sq)] for row in sq)
        for w in get(prefix):
            bt(sq + [w])
    res = []
    for w in words:
        bt([w])
    return res
```

Time: O(n * 26^L) | Space: O(n)

---

## H05. Prefix and Suffix Search

**Description**: Find word with prefix and suffix, return max index.

**Approach**: Store suffix + '#' + word for each suffix of each word. Query suffix + '#' + prefix.

```python
class WordFilter:
    def __init__(self, words):
        self.trie = {}
        for idx, w in enumerate(words):
            for i in range(len(w)+1):
                key = w[i:] + '#' + w
                node = self.trie
                for c in key:
                    node = node.setdefault(c, {})
                node['$'] = idx
    def f(self, prefix, suffix):
        key = suffix + '#' + prefix
        node = self.trie
        for c in key:
            if c not in node:
                return -1
            node = node[c]
        return node.get('$', -1)
```

Time: O(W*L^2) | Space: O(W*L^2)

---

## H06. Count Pairs With XOR in a Range

**Description**: Count pairs (i, j) where low <= nums[i] XOR nums[j] <= high.

**Approach**: Count pairs with XOR < high+1 minus count with XOR < low. Use binary trie with count at nodes.

```python
def countPairs(nums, low, high):
    res = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            xor_val = nums[i] ^ nums[j]
            if low <= xor_val <= high:
                res += 1
    return res
```

Time: O(n * 15) | Space: O(n * 15)

---

## H07. Maximum XOR With an Element From Array

**Description**: Queries (x, m): max XOR of x with array element <= m.

**Approach**: Sort queries by m. Incrementally build trie. For each query, find max XOR in trie.

```python
def maximizeXor(nums, queries):
    nums.sort()
    qs = sorted(enumerate(queries), key=lambda x: x[1][1])
    trie, res, j = {}, [0]*len(queries), 0
    for idx, (x, m) in qs:
        while j < len(nums) and nums[j] <= m:
            node = trie
            for i in range(31, -1, -1):
                b = (nums[j] >> i) & 1
                node = node.setdefault(b, {})
            j += 1
        if not trie:
            res[idx] = -1
            continue
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur = (cur << 1) + (1 if want in node else 0)
            node = node[want] if want in node else node[b]
        res[idx] = cur
    return res
```

Time: O((n+q)*32) | Space: O(n*32)

---

## H08. Word Search II (Optimized)

**Description**: Same as medium but with large dictionary. Need pruning.

**Approach**: Trie with remove during backtrack to avoid duplicate finds. Or mark node as visited.

```python
def findWords(board, words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = w
    res, m, n = set(), len(board), len(board[0])
    def dfs(r, c, node):
        if '#' in node:
            res.add(node['#'])
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] not in node:
            return
        ch, board[r][c] = board[r][c], '#'
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(r+dr, c+dc, node[ch])
        board[r][c] = ch
    for i in range(m):
        for j in range(n):
            dfs(i, j, trie)
    return list(res)
```

Time: O(m*n*4^L) | Space: O(W)

---

## H09. Count Distinct Substrings

**Description**: Count distinct substrings of string.

**Approach**: Suffix trie. Each path from root is unique substring. Count nodes (excluding root) or use suffix array.

```python
def countDistinctSubstrings(s):
    trie = {}
    count = 0
    for i in range(len(s)):
        node = trie
        for j in range(i, len(s)):
            if s[j] not in node:
                node[s[j]] = {}
                count += 1
            node = node[s[j]]
    return count
```

Time: O(n^2) | Space: O(n^2)

---

## H10. Longest Duplicate Substring

**Description**: Find longest substring that appears at least twice.

**Approach**: Binary search on length + rolling hash or suffix array. Trie of suffixes for each length.

```python
def longestDupSubstring(s):
    n = len(s)
    lo, hi, res = 1, n, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        seen = set()
        found = ""
        for i in range(n - mid + 1):
            sub = s[i:i+mid]
            if sub in seen:
                found = sub
                break
            seen.add(sub)
        if found:
            res, lo = found, mid + 1
        else:
            hi = mid - 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## H11. Multi-Search

**Description**: Given string b and array of small strings T, find all occurrences of each T[i] in b.

**Approach**: Build trie from T. For each starting position in b, traverse trie and record matches.

```python
def multiSearch(big, small):
    trie = {}
    for i, t in enumerate(small):
        node = trie
        for c in t:
            node = node.setdefault(c, {})
        node.setdefault('#', []).append(i)
    res = [[] for _ in small]
    for i in range(len(big)):
        node = trie
        for j in range(i, len(big)):
            if big[j] not in node:
                break
            node = node[big[j]]
            if '#' in node:
                for idx in node['#']:
                    res[idx].append(i)
    return res
```

Time: O(b * T + sum(small)) | Space: O(sum(small))

---

## H12. Word Rectangle

**Description**: Find largest rectangle of letters such that each row and column is a word.

**Approach**: Trie for rows and columns. Try dimensions, backtrack with trie validation.

```python
def maxRectangle(words):
    by_len = {}
    for w in words:
        by_len.setdefault(len(w), set()).add(w)
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = True
    def get_prefix(prefix):
        node = trie
        for c in prefix:
            if c not in node:
                return []
            node = node[c]
        return [w for w in by_len.get(len(prefix), []) if w.startswith(prefix)]
    best = []
    for cols in sorted(by_len.keys(), reverse=True):
        for rows in sorted(by_len.keys(), reverse=True):
            if cols * rows <= len(best) * len(best[0]) if best else 0:
                continue
            def bt(rect):
                nonlocal best
                if len(rect) == rows:
                    best = rect[:]
                    return
                prefix = ''.join(rect[i][len(rect)] for i in range(len(rect))) if rect else ''
                for w in by_len.get(rows, []):
                    if all((rect + [w])[i][:len(rect)+1] in get_prefix('') for i in range(len(rect)+1)):
                        bt(rect + [w])
            bt([])
    return best
```

Time: O(W * L^2) | Space: O(W)

---

## H13. Maximum XOR of Two Numbers in a Tree

**Description**: Tree with values on nodes. Find max XOR of any two node values.

**Approach**: DFS from root. At each node, insert path XOR into trie. Query max XOR with current path XOR.

```python
def maxXorTree(root):
    trie, res = {}, [0]
    def query(x):
        node, cur = trie, 0
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            want = 1 - b
            cur = (cur << 1) + (1 if want in node else 0)
            node = node[want] if want in node else node[b]
        return cur
    def insert(x):
        node = trie
        for i in range(31, -1, -1):
            b = (x >> i) & 1
            node = node.setdefault(b, {})
    def dfs(n, xor_val):
        res[0] = max(res[0], query(xor_val))
        insert(xor_val)
        if n.left:
            dfs(n.left, xor_val ^ n.left.val)
        if n.right:
            dfs(n.right, xor_val ^ n.right.val)
    dfs(root, root.val)
    return res[0]
```

Time: O(n * 32) | Space: O(n * 32)

---

## H14. Count Substrings With One Distinct Letter

**Description**: Count substrings with exactly one distinct character.

**Approach**: Group consecutive same chars. Each group of length n contributes n*(n+1)/2. Trie not typical.

```python
def countLetters(s):
    res, i = 0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        n = j - i
        res += n * (n + 1) // 2
        i = j
    return res
```

Time: O(n) | Space: O(1)

---

## H15. Substring With Largest Variance

**Description**: Find substring with largest difference between max and min frequency of two chars.

**Approach**: Kadane variant. For each pair of chars (a, b), treat as +1 and -1, find max subarray sum. Trie not typical.

```python
def largestVariance(s):
    chars = list(set(s))
    res = 0
    for a in chars:
        for b in chars:
            if a == b:
                continue
            cur, has_b, first_b = 0, False, False
            for c in s:
                if c == a:
                    cur += 1
                elif c == b:
                    has_b = True
                    cur -= 1
                    if cur < 0:
                        cur = 0
                        first_b = False
                    else:
                        first_b = True
                if has_b and first_b:
                    res = max(res, cur)
    return res
```

Time: O(26^2 * n) | Space: O(1)
