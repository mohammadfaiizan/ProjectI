# String DP

## Longest Common Subsequence

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

## Longest Common Substring

```python
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                max_len = max(max_len, dp[i][j])
    return max_len
```

## Edit Distance (Levenshtein)

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

## One Edit Distance

```python
def is_one_edit_distance(s, t):
    if abs(len(s) - len(t)) > 1:
        return False
    if len(s) > len(t):
        s, t = t, s
    for i in range(len(s)):
        if s[i] != t[i]:
            if len(s) == len(t):
                return s[i + 1:] == t[i + 1:]
            return s[i:] == t[i + 1:]
    return len(t) - len(s) == 1
```

## Longest Palindromic Subsequence

```python
def longest_palindrome_subseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i + 1][j - 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

## Longest Palindromic Substring (DP)

```python
def longest_palindrome_substring(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    for i in range(n):
        dp[i][i] = True
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start, max_len = i, 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start, max_len = i, length
    return s[start:start + max_len]
```

## Word Break I

```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
```

## Word Break II

```python
def word_break_ii(s, word_dict):
    word_set = set(word_dict)
    memo = {}
    
    def dp(start):
        if start == len(s):
            return [""]
        if start in memo:
            return memo[start]
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for suffix in dp(end):
                    result.append(word + (" " + suffix if suffix else ""))
        memo[start] = result
        return result
    
    return dp(0)
```

## Interleaving String

```python
def is_interleave(s1, s2, s3):
    if len(s1) + len(s2) != len(s3):
        return False
    m, n = len(s1), len(s2)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(m + 1):
        for j in range(n + 1):
            if i > 0 and s1[i - 1] == s3[i + j - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j]
            if j > 0 and s2[j - 1] == s3[i + j - 1]:
                dp[i][j] = dp[i][j] or dp[i][j - 1]
    return dp[m][n]
```

## Distinct Subsequences

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

## Shortest Common Supersequence

```python
def shortest_common_supersequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            result.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            result.append(str1[i - 1])
            i -= 1
        else:
            result.append(str2[j - 1])
            j -= 1
    result.extend(str1[:i][::-1])
    result.extend(str2[:j][::-1])
    return ''.join(reversed(result))
```

## Min Insertion to Make Palindrome

```python
def min_insertions(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

## Min Deletion to Make Palindrome

```python
def min_deletions(s):
    n = len(s)
    lps = longest_palindrome_subseq(s)
    return n - lps
```

## Longest Repeating Subsequence

```python
def longest_repeating_subsequence(s):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == s[j - 1] and i != j:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][n]
```

## Min ASCII Delete Sum

```python
def minimum_delete_sum(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + ord(s1[i - 1]),
                    dp[i][j - 1] + ord(s2[j - 1])
                )
    return dp[m][n]
```

## Max Length Repeated Subarray

```python
def find_length(nums1, nums2):
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                max_len = max(max_len, dp[i][j])
    return max_len
```

## Wildcard Matching

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

## Regex Matching

```python
def is_match_regex(s, p):
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

## Scramble String

```python
def is_scramble(s1, s2):
    if len(s1) != len(s2):
        return False
    if s1 == s2:
        return True
    if sorted(s1) != sorted(s2):
        return False
    n = len(s1)
    for i in range(1, n):
        if (is_scramble(s1[:i], s2[:i]) and is_scramble(s1[i:], s2[i:])) or \
           (is_scramble(s1[:i], s2[n-i:]) and is_scramble(s1[i:], s2[:n-i])):
            return True
    return False
```

## Count Different Palindromic Subsequences

```python
def count_palindromic_subsequences(s):
    n = len(s)
    MOD = 10**9 + 7
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                left, right = i + 1, j - 1
                while left <= right and s[left] != s[i]:
                    left += 1
                while left <= right and s[right] != s[i]:
                    right -= 1
                if left > right:
                    dp[i][j] = 2 * dp[i + 1][j - 1] + 2
                elif left == right:
                    dp[i][j] = 2 * dp[i + 1][j - 1] + 1
                else:
                    dp[i][j] = 2 * dp[i + 1][j - 1] - dp[left + 1][right - 1]
            else:
                dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]
            dp[i][j] = (dp[i][j] + MOD) % MOD
    return dp[0][n - 1]
```

## Longest String Chain

```python
def longest_str_chain(words):
    words.sort(key=len)
    dp = {}
    for w in words:
        dp[w] = 1
        for i in range(len(w)):
            pred = w[:i] + w[i + 1:]
            if pred in dp:
                dp[w] = max(dp[w], dp[pred] + 1)
    return max(dp.values())
```
