# Regex and Wildcard Matching Theory and Implementations

## Theory

Pattern matching with special characters: `.` matches any single char, `*` matches zero or more of preceding element. Wildcard `?` matches any single char, `*` matches any sequence.

## Regex Matching (DP)

```python
def is_match_regex(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 2]
                if p[j - 2] == "." or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif p[j - 1] == "." or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

## Wildcard Matching (DP)

```python
def is_match_wildcard(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == "?" or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

## Implement strStr

```python
def str_str(haystack, needle):
    if not needle:
        return 0
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1

def str_str_kmp(haystack, needle):
    if not needle:
        return 0
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if needle[i] == needle[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    i = j = 0
    while i < n:
        if haystack[i] == needle[j]:
            i += 1
            j += 1
        if j == m:
            return i - j
        elif i < n and haystack[i] != needle[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1
```

## Finite Automaton Overview

A finite automaton (DFA) for pattern matching: states represent "how much of pattern we've matched." Each transition consumes one text char. Build transition table from pattern; run text through automaton. KMP is a simplified version. Full DFA for pattern of length m has m+1 states and sigma (alphabet size) transitions per state. Build time O(m*sigma), search O(n).

```python
def build_dfa(pattern, sigma=256):
    m = len(pattern)
    dfa = [[0] * sigma for _ in range(m + 1)]
    dfa[0][ord(pattern[0])] = 1
    x = 0
    for j in range(1, m):
        for c in range(sigma):
            dfa[j][c] = dfa[x][c]
        dfa[j][ord(pattern[j])] = j + 1
        x = dfa[x][ord(pattern[j])]
    return dfa

def dfa_search(text, pattern, dfa):
    m = len(pattern)
    j = 0
    for i, c in enumerate(text):
        j = dfa[j][ord(c)] if ord(c) < 256 else 0
        if j == m:
            return i - m + 1
    return -1
```
