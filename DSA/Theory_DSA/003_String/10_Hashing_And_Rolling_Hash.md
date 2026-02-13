# Hashing and Rolling Hash Theory and Implementations

## Theory

Hash functions map strings to integers. A good hash is uniform and minimizes collisions. Rolling hash allows O(1) update when sliding a window: subtract contribution of leaving char, add contribution of entering char.

## Polynomial Hash Function

```python
def polynomial_hash(s, base=31, mod=10**9 + 7):
    h = 0
    for c in s:
        h = (h * base + ord(c)) % mod
    return h

def polynomial_hash_alternative(s, base=256, mod=10**9 + 7):
    h = 0
    for c in s:
        h = (h * base + ord(c)) % mod
    return h
```

## Rolling Hash Concept

When window slides: new_hash = (old_hash - old_char * base^(m-1)) * base + new_char. Precompute base^(m-1) for O(1) updates.

## Rabin-Karp Using Rolling Hash

```python
def rabin_karp_rolling(text, pattern, base=256, mod=10**9 + 7):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    h = pow(base, m - 1, mod)
    pattern_hash = 0
    text_hash = 0
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

## Handling Collisions (Double Hash)

```python
def double_hash(s, base1=31, mod1=10**9+7, base2=37, mod2=10**9+9):
    h1 = h2 = 0
    for c in s:
        h1 = (h1 * base1 + ord(c)) % mod1
        h2 = (h2 * base2 + ord(c)) % mod2
    return (h1, h2)

def rabin_karp_double_hash(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    base1, mod1 = 31, 10**9 + 7
    base2, mod2 = 37, 10**9 + 9
    h1 = pow(base1, m - 1, mod1)
    h2 = pow(base2, m - 1, mod2)
    ph1 = ph2 = th1 = th2 = 0
    for i in range(m):
        ph1 = (ph1 * base1 + ord(pattern[i])) % mod1
        ph2 = (ph2 * base2 + ord(pattern[i])) % mod2
        th1 = (th1 * base1 + ord(text[i])) % mod1
        th2 = (th2 * base2 + ord(text[i])) % mod2
    for i in range(n - m + 1):
        if (ph1, ph2) == (th1, th2) and text[i:i + m] == pattern:
            return i
        if i < n - m:
            th1 = (th1 - ord(text[i]) * h1) % mod1
            th1 = (th1 * base1 + ord(text[i + m])) % mod1
            th1 = (th1 + mod1) % mod1
            th2 = (th2 - ord(text[i]) * h2) % mod2
            th2 = (th2 * base2 + ord(text[i + m])) % mod2
            th2 = (th2 + mod2) % mod2
    return -1
```

## Repeated DNA Sequences

```python
def find_repeated_dna(s):
    if len(s) < 10:
        return []
    from collections import defaultdict
    seen = defaultdict(int)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    hash_val = 0
    for i in range(10):
        hash_val = (hash_val << 2) | mapping[s[i]]
    seen[hash_val] = 1
    result = []
    mask = (1 << 20) - 1
    for i in range(10, len(s)):
        hash_val = ((hash_val << 2) | mapping[s[i]]) & mask
        seen[hash_val] += 1
        if seen[hash_val] == 2:
            result.append(s[i - 9:i + 1])
    return result
```

## Longest Duplicate Substring

```python
def longest_duplicate_substring(s):
    def has_duplicate(length):
        seen = set()
        h = 0
        base = 26
        mod = 10**9 + 7
        power = pow(base, length - 1, mod)
        for i in range(length):
            h = (h * base + ord(s[i]) - ord("a")) % mod
        seen.add(h)
        for i in range(length, len(s)):
            h = (h - (ord(s[i - length]) - ord("a")) * power) % mod
            h = (h * base + ord(s[i]) - ord("a")) % mod
            h = (h + mod) % mod
            if h in seen:
                return s[i - length + 1:i + 1]
            seen.add(h)
        return None

    left, right = 1, len(s) - 1
    result = ""
    while left <= right:
        mid = (left + right) // 2
        dup = has_duplicate(mid)
        if dup:
            result = dup
            left = mid + 1
        else:
            right = mid - 1
    return result
```

## Group Shifted Strings

```python
def group_shifted_strings(strings):
    def get_key(s):
        if not s:
            return ()
        key = []
        for i in range(1, len(s)):
            diff = (ord(s[i]) - ord(s[i - 1])) % 26
            key.append(diff)
        return tuple(key)

    from collections import defaultdict
    groups = defaultdict(list)
    for s in strings:
        groups[get_key(s)].append(s)
    return list(groups.values())
```

## Isomorphic Strings

```python
def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    s_to_t = {}
    t_to_s = {}
    for a, b in zip(s, t):
        if a in s_to_t:
            if s_to_t[a] != b:
                return False
        else:
            s_to_t[a] = b
        if b in t_to_s:
            if t_to_s[b] != a:
                return False
        else:
            t_to_s[b] = a
    return True
```

## Word Pattern

```python
def word_pattern(pattern, s):
    words = s.split()
    if len(pattern) != len(words):
        return False
    p_to_w = {}
    w_to_p = {}
    for p, w in zip(pattern, words):
        if p in p_to_w:
            if p_to_w[p] != w:
                return False
        else:
            p_to_w[p] = w
        if w in w_to_p:
            if w_to_p[w] != p:
                return False
        else:
            w_to_p[w] = p
    return True
```

## Count Distinct Substrings Using Rolling Hash

```python
def count_distinct_substrings(s):
    n = len(s)
    base = 31
    mod = 10**9 + 7
    seen = set()
    for i in range(n):
        h = 0
        for j in range(i, n):
            h = (h * base + ord(s[j]) - ord("a") + 1) % mod
            seen.add(h)
    return len(seen)
```
