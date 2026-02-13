# String Searching

## Naive String Search

**Idea:** For each starting position in text, check if pattern matches. O(n * m) where n = text length, m = pattern length.

```python
def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    result = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            result.append(i)
    return result
```

---

## KMP (Knuth-Morris-Pratt)

**Idea:** Build failure/lps (longest proper prefix which is also suffix) array for pattern. When mismatch, shift by lps value instead of 1. O(m) preprocessing, O(n) search, total O(n + m).

**Failure function:** lps[i] = length of longest proper prefix of pattern[0:i+1] that is also a suffix.

```python
def compute_lps(pattern):
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

def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    lps = compute_lps(pattern)
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

---

## Rabin-Karp

**Idea:** Rolling hash. Compute hash of pattern. For each window in text, compute hash. If match, verify (collision). Average O(n + m), worst O(n * m) with many collisions.

```python
def rabin_karp(text, pattern, base=256, mod=10**9+7):
    n, m = len(text), len(pattern)
    if m > n:
        return []
    pattern_hash = 0
    window_hash = 0
    h = pow(base, m - 1, mod)
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        window_hash = (window_hash * base + ord(text[i])) % mod
    result = []
    for i in range(n - m + 1):
        if pattern_hash == window_hash and text[i:i + m] == pattern:
            result.append(i)
        if i < n - m:
            window_hash = (window_hash - ord(text[i]) * h) % mod
            window_hash = (window_hash * base + ord(text[i + m])) % mod
            window_hash = (window_hash + mod) % mod
    return result
```

---

## Z-Algorithm

**Idea:** Z-array: Z[i] = length of longest substring starting at i that is also prefix. Use Z-array for pattern matching: pattern + "$" + text, find Z[i] = len(pattern).

**Z-array construction:** Use Z-box [L, R]. If i > R, compute Z[i] from scratch. Else use Z[i-L] if i + Z[i-L] <= R, else extend from R.

```python
def compute_z_array(s):
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

def z_search(text, pattern):
    concat = pattern + "$" + text
    z = compute_z_array(concat)
    m = len(pattern)
    return [i - m - 1 for i in range(len(concat)) if z[i] == m]
```

---

## Boyer-Moore (Bad Character Heuristic)

**Idea:** Compare from right of pattern. On mismatch, shift by max(1, j - last_occurrence[mismatch_char]). Can skip many positions.

```python
def bad_char_table(pattern):
    table = {}
    for i, c in enumerate(pattern):
        table[c] = i
    return table

def boyer_moore(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    bc = bad_char_table(pattern)
    result = []
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and text[s + j] == pattern[j]:
            j -= 1
        if j < 0:
            result.append(s)
            s += m
        else:
            shift = j - bc.get(text[s + j], -1)
            s += max(1, shift)
    return result
```

---

## Aho-Corasick Overview

**Idea:** Multiple pattern matching. Build trie of all patterns. Add failure links (like KMP). Scan text once, follow trie and failure links. O(m + n + z) where m = total pattern length, n = text length, z = matches.

**Mechanism:** Trie + failure/suffix links. On mismatch, follow failure link. Output all patterns ending at current node.

---

## Suffix Array + LCP Overview

**Suffix array:** Sorted array of all suffixes. SA[i] = starting index of i-th smallest suffix.

**LCP (Longest Common Prefix):** LCP[i] = common prefix length of SA[i] and SA[i-1].

**Use:** Pattern search (binary search on suffix array), longest repeated substring, etc.

---

## When to Use Each (Comparison Table)

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| Naive | O(n*m) | O(1) | Simple, short patterns |
| KMP | O(n+m) | O(m) | Single pattern, no backtrack |
| Rabin-Karp | O(n+m) avg | O(1) | Multiple patterns same length, plagiarism |
| Z-algorithm | O(n+m) | O(n+m) | Single pattern, Z-array useful |
| Boyer-Moore | O(n/m) best | O(m) | Long patterns, natural language |
| Aho-Corasick | O(n+m+z) | O(m) | Multiple patterns |
| Suffix array | O(n log n) build | O(n) | Complex queries, repeated substrings |
