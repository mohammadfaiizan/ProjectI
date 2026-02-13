# Pattern Matching Theory and Implementations

## Theory

Pattern matching finds occurrences of a pattern string P in a text string T. Naive approach checks every position; advanced algorithms use preprocessing to achieve better time complexity.

## Brute Force O(n*m)

```python
def brute_force_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            return i
    return -1

def brute_force_all(text, pattern):
    n, m = len(text), len(pattern)
    result = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            result.append(i)
    return result
```

## KMP - Prefix/Failure Function

```python
def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps
```

## KMP - Search

```python
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    lps = build_lps(pattern)
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            return i - j
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

def kmp_search_all(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return [0]
    lps = build_lps(pattern)
    result = []
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return result
```

## Rabin-Karp (Rolling Hash)

```python
def rabin_karp(text, pattern, base=256, mod=10**9 + 7):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    pattern_hash = 0
    text_hash = 0
    h = pow(base, m - 1, mod)
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod
    for i in range(n - m + 1):
        if pattern_hash == text_hash and text[i:i + m] == pattern:
            return i
        if i < n - m:
            text_hash = (text_hash - ord(text[i]) * h) % mod
            text_hash = (text_hash * base + ord(text[i + m])) % mod
            text_hash = (text_hash + mod) % mod
    return -1
```

## Z-Algorithm - Z-Array

```python
def build_z_array(s):
    n = len(s)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z
```

## Z-Algorithm - Search

```python
def z_search(text, pattern):
    concat = pattern + "$" + text
    z = build_z_array(concat)
    m = len(pattern)
    result = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            result.append(i - m - 1)
    return result
```

## Boyer-Moore (Bad Char Heuristic)

```python
def build_bad_char_table(pattern):
    table = {}
    m = len(pattern)
    for i in range(m - 1):
        table[pattern[i]] = m - 1 - i
    return table

def boyer_moore_bad_char(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    bad_char = build_bad_char_table(pattern)
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            return s
        s += bad_char.get(text[s + j], m)
    return -1
```

## Aho-Corasick Overview

Aho-Corasick builds a trie of all patterns and adds failure links (similar to KMP) for multiple pattern search. Time: O(n + m + z) where z is number of matches. Use when searching for multiple patterns simultaneously in text.

```python
from collections import defaultdict, deque

class AhoCorasickNode:
    def __init__(self):
        self.children = defaultdict(AhoCorasickNode)
        self.fail = None
        self.output = []

def build_aho_corasick(patterns):
    root = AhoCorasickNode()
    for pattern in patterns:
        node = root
        for c in pattern:
            node = node.children[c]
        node.output.append(pattern)
    queue = deque()
    for c, child in root.children.items():
        child.fail = root
        queue.append(child)
    while queue:
        current = queue.popleft()
        for c, child in current.children.items():
            queue.append(child)
            fail = current.fail
            while fail and c not in fail.children:
                fail = fail.fail
            child.fail = fail.children[c] if fail else root
            child.output.extend(child.fail.output)
    return root

def aho_corasick_search(text, root):
    result = []
    node = root
    for i, c in enumerate(text):
        while node and c not in node.children:
            node = node.fail
        if not node:
            node = root
            continue
        node = node.children[c]
        for pattern in node.output:
            result.append((i - len(pattern) + 1, pattern))
    return result
```

## Suffix Array Overview

A suffix array is a sorted array of all suffixes of a string. Used for full-text search, longest repeated substring, and LCP computation. Build time O(n log n) with doubling algorithm.

```python
def build_suffix_array(s):
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    return [i for _, i in suffixes]

def search_suffix_array(text, pattern, suffix_arr):
    n, m = len(text), len(pattern)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        suffix = text[suffix_arr[mid]:]
        if pattern == suffix[:m]:
            return suffix_arr[mid]
        if pattern < suffix[:m]:
            right = mid - 1
        else:
            left = mid + 1
    return -1
```

## Comparison Table

| Algorithm | Preprocessing | Search | Best For |
|-----------|---------------|--------|----------|
| Brute Force | O(1) | O(n*m) | Simple, short patterns |
| KMP | O(m) | O(n) | Single pattern, worst-case guarantee |
| Rabin-Karp | O(m) | O(n) average | Multiple patterns, plagiarism detection |
| Z-Algorithm | O(n+m) | O(n+m) | Pattern+text combined, simple implementation |
| Boyer-Moore | O(m + sigma) | O(n/m) best | Long patterns, natural language |
| Aho-Corasick | O(sum of pattern lengths) | O(n + z) | Multiple patterns simultaneously |
| Suffix Array | O(n log n) | O(m log n) | Full-text index, repeated queries |
