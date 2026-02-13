# Hard String Problems

## 01. Regular Expression Matching

**Description**: Match string with pattern containing . and *.
**Approach**: DP: dp[i][j] = match s[:i] with p[:j], handle * (zero or more of preceding).

```python
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.'))
            elif p[j - 1] == '.' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 02. Wildcard Matching

**Description**: Match string with pattern containing ? and *.
**Approach**: DP similar to regex, * matches any sequence (greedy or DP).

```python
def is_match_wildcard(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 03. Minimum Window Substring

**Description**: Smallest substring containing all chars of t.
**Approach**: Sliding window with frequency maps, expand then shrink.

```python
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    have = 0
    need_cnt = len(need)
    start, length = 0, float('inf')
    l = 0
    for r, c in enumerate(s):
        if c in need:
            need[c] -= 1
            if need[c] == 0:
                have += 1
        while have == need_cnt:
            if r - l + 1 < length:
                start, length = l, r - l + 1
            if s[l] in need:
                need[s[l]] += 1
                if need[s[l]] > 0:
                    have -= 1
            l += 1
    return s[start:start + length] if length != float('inf') else ""
```

Time: O(n + m) | Space: O(m)

---

## 04. Substring with Concatenation of All Words

**Description**: Find all starting indices of concatenation of words.
**Approach**: Sliding window over starts, check word-by-word with frequency.

```python
def find_substring(s, words):
    from collections import Counter
    if not words:
        return []
    wlen, nw = len(words[0]), len(words)
    target = Counter(words)
    out = []
    for start in range(wlen):
        seen = Counter()
        cnt = 0
        for i in range(start, len(s) - wlen + 1, wlen):
            w = s[i:i + wlen]
            if w in target:
                seen[w] += 1
                cnt += 1
                while seen[w] > target[w]:
                    first = s[start:start + wlen]
                    seen[first] -= 1
                    cnt -= 1
                    start += wlen
                if cnt == nw:
                    out.append(start)
            else:
                seen.clear()
                cnt = 0
                start = i + wlen
    return out
```

Time: O(n * wlen) | Space: O(nw)

---

## 05. Longest Valid Parentheses

**Description**: Length of longest valid parentheses substring.
**Approach**: Stack (store indices) or DP: dp[i] = longest valid ending at i.

```python
def longest_valid_parentheses(s):
    st = [-1]
    best = 0
    for i, c in enumerate(s):
        if c == '(':
            st.append(i)
        else:
            st.pop()
            if not st:
                st.append(i)
            else:
                best = max(best, i - st[-1])
    return best
```

Time: O(n) | Space: O(n)

---

## 06. Edit Distance (Levenshtein)

**Description**: Min insert/delete/replace to transform s1 to s2.
**Approach**: DP: dp[i][j] = min edits for s1[:i] to s2[:j].

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 07. Scramble String

**Description**: Can s1 be scrambled to form s2 (binary tree swap children)?
**Approach**: Recursion with memo: try all split points, check (a1,b1)+(a2,b2) or (a1,b2)+(a2,b1).

```python
def is_scramble(s1, s2):
    if s1 == s2:
        return True
    if sorted(s1) != sorted(s2):
        return False
    n = len(s1)
    for i in range(1, n):
        if (is_scramble(s1[:i], s2[:i]) and is_scramble(s1[i:], s2[i:])) or \
           (is_scramble(s1[:i], s2[-i:]) and is_scramble(s1[i:], s2[:-i])):
            return True
    return False
```

Time: O(n^4) | Space: O(n^2)

---

## 08. Distinct Subsequences

**Description**: Number of times t appears as subsequence of s.
**Approach**: DP: dp[i][j] = count for s[:i] and t[:j], match or skip.

```python
def num_distinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[m][n]
```

Time: O(m * n) | Space: O(m * n)

---

## 09. Minimum Insertions to Make Palindrome

**Description**: Min chars to insert for palindrome.
**Approach**: n - longest palindromic subsequence.

```python
def min_insertions(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

Time: O(n^2) | Space: O(n^2)

---

## 10. Palindrome Partitioning II

**Description**: Min cuts so each part is palindrome.
**Approach**: DP: is_pal[i][j], then cuts[j] = min cuts for s[:j].

```python
def min_cut(s):
    n = len(s)
    pal = [[False] * n for _ in range(n)]
    for L in range(1, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            pal[i][j] = (s[i] == s[j]) and (L <= 2 or pal[i + 1][j - 1])
    cuts = list(range(n + 1))
    for j in range(1, n + 1):
        for i in range(j):
            if pal[i][j - 1]:
                cuts[j] = min(cuts[j], cuts[i] + 1)
    return cuts[n] - 1
```

Time: O(n^2) | Space: O(n^2)

---

## 11. Word Ladder II

**Description**: All shortest transformation sequences from begin to end word.
**Approach**: BFS to find distance, DFS to reconstruct all paths.

```python
def find_ladders(begin, end, word_list):
    from collections import defaultdict, deque
    words = set(word_list)
    if end not in words:
        return []
    layer = {begin: [begin]}
    while layer:
        next_layer = defaultdict(list)
        for w in layer:
            if w == end:
                return layer[w]
            for i in range(len(w)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    nw = w[:i] + c + w[i + 1:]
                    if nw in words:
                        next_layer[nw].append(w)
        words -= set(next_layer.keys())
        layer = next_layer
    return []
```

Time: O(n * L^2) | Space: O(n)

---

## 12. Word Ladder

**Description**: Shortest transformation from begin to end (one char change).
**Approach**: BFS with queue, try all one-char variations.

```python
def ladder_length(begin, end, word_list):
    from collections import deque
    words = set(word_list)
    if end not in words:
        return 0
    q = deque([(begin, 1)])
    while q:
        w, d = q.popleft()
        if w == end:
            return d
        for i in range(len(w)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                nw = w[:i] + c + w[i + 1:]
                if nw in words:
                    words.discard(nw)
                    q.append((nw, d + 1))
    return 0
```

Time: O(n * L^2) | Space: O(n)

---

## 13. Alien Dictionary

**Description**: Order of letters from sorted dictionary of alien language.
**Approach**: Build graph from adjacent word pairs, topological sort.

```python
def alien_order(words):
    from collections import defaultdict, deque
    adj = defaultdict(set)
    in_deg = {c: 0 for w in words for c in w}
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        for j in range(min(len(a), len(b))):
            if a[j] != b[j]:
                if b[j] not in adj[a[j]]:
                    adj[a[j]].add(b[j])
                    in_deg[b[j]] += 1
                break
        else:
            if len(a) > len(b):
                return ""
    q = deque([c for c in in_deg if in_deg[c] == 0])
    out = []
    while q:
        c = q.popleft()
        out.append(c)
        for nxt in adj[c]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    return "".join(out) if len(out) == len(in_deg) else ""
```

Time: O(n * L) | Space: O(1)

---

## 14. Longest Duplicate Substring

**Description**: Longest substring that appears at least twice.
**Approach**: Binary search on length + rolling hash (Rabin-Karp) to check.

```python
def longest_dup_substring(s):
    n = len(s)
    base, mod = 26, 2**63 - 1
    def check(L):
        h = 0
        for i in range(L):
            h = (h * base + ord(s[i]) - ord('a')) % mod
        seen = {h}
        p = pow(base, L - 1, mod)
        for i in range(L, n):
            h = (h - (ord(s[i - L]) - ord('a')) * p) % mod
            h = (h * base + ord(s[i]) - ord('a')) % mod
            if h in seen:
                return s[i - L + 1:i + 1]
            seen.add(h)
        return ""
    lo, hi = 1, n
    res = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cur = check(mid)
        if cur:
            res = cur
            lo = mid + 1
        else:
            hi = mid - 1
    return res
```

Time: O(n log n) | Space: O(n)

---

## 15. Count Unique Characters of All Substrings

**Description**: Sum of unique char count over all substrings.
**Approach**: For each char, count substrings where it is unique (contribution = (i-prev)*(next-i)).

```python
def unique_letter_string(s):
    idx = {c: [-1, -1] for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    res = 0
    for i, c in enumerate(s):
        prev, curr = idx[c]
        res += (curr - prev) * (i - curr)
        idx[c] = [curr, i]
    for c in idx:
        prev, curr = idx[c]
        res += (curr - prev) * (len(s) - curr)
    return res
```

Time: O(n) | Space: O(1)

---

## 16. Palindrome Pairs

**Description**: Pairs (i,j) where words[i] + words[j] is palindrome.
**Approach**: For each word, check if reverse of prefix/suffix exists and remainder is palindrome.

```python
def palindrome_pairs(words):
    d = {w: i for i, w in enumerate(words)}
    out = []
    for i, w in enumerate(words):
        for j in range(len(w) + 1):
            pre, suf = w[:j], w[j:]
            rev_pre, rev_suf = pre[::-1], suf[::-1]
            if rev_pre in d and d[rev_pre] != i and suf == suf[::-1]:
                out.append([i, d[rev_pre]])
            if j and rev_suf in d and d[rev_suf] != i and pre == pre[::-1]:
                out.append([d[rev_suf], i])
    return out
```

Time: O(n * L^2) | Space: O(n)

---

## 17. Shortest Palindrome

**Description**: Prepend min chars to make palindrome.
**Approach**: Find longest palindromic prefix, prepend reverse of rest. Use KMP on s + "#" + reverse(s).

```python
def shortest_palindrome(s):
    if not s:
        return ""
    rev = s[::-1]
    combined = s + "#" + rev
    n = len(combined)
    lps = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and combined[i] != combined[j]:
            j = lps[j - 1]
        if combined[i] == combined[j]:
            j += 1
        lps[i] = j
    return rev[:len(s) - lps[-1]] + s
```

Time: O(n) | Space: O(n)
