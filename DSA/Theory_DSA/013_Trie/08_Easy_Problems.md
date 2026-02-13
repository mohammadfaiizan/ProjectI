# Trie - Easy Problems

## 01. Implement Trie (Prefix Tree)

**Description**: Implement a trie with insert, search, and startsWith methods.

**Approach**: Standard trie with HashMap or array children. Insert character by character, set is_end at last node. Search returns true only if path exists and is_end. startsWith returns true if path exists.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

def insert(root, word):
    node = root
    for c in word:
        if c not in node.children:
            node.children[c] = TrieNode()
        node = node.children[c]
    node.is_end = True

def search(root, word):
    node = root
    for c in word:
        if c not in node.children:
            return False
        node = node.children[c]
    return node.is_end

def startsWith(root, prefix):
    node = root
    for c in prefix:
        if c not in node.children:
            return False
        node = node.children[c]
    return True
```

Time: O(m) per op | Space: O(m) total

---

## 02. Design Add and Search Words Data Structure

**Description**: Design a structure that supports adding words and searching with '.' as wildcard matching any single character.

**Approach**: Trie structure. For search with '.', use DFS: when char is '.', recurse on all children; otherwise follow the specific child.

```python
def search_word(node, word, i):
    if i == len(word):
        return node.is_end
    c = word[i]
    if c == '.':
        return any(search_word(child, word, i+1) for child in node.children.values())
    if c not in node.children:
        return False
    return search_word(node.children[c], word, i+1)
```

Time: O(m) insert, O(26^m) worst search | Space: O(n)

---

## 03. Longest Common Prefix

**Description**: Find the longest common prefix string amongst an array of strings.

**Approach**: Build trie from all words. Traverse from root while node has exactly one child and is not a word end. The path gives LCP.

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
```

Time: O(S) | Space: O(1)

---

## 04. Replace Words

**Description**: In a sentence, replace each word with its shortest root from dictionary if it has one.

**Approach**: Build trie from dictionary roots. For each word in sentence, traverse trie; when is_end is hit, that prefix is the replacement.

```python
def replaceWords(dictionary, sentence):
    trie = {}
    for root in dictionary:
        node = trie
        for c in root:
            if c not in node:
                node[c] = {}
            node = node[c]
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

## 05. Map Sum Pairs

**Description**: Design a map that supports insert(key, val) and sum(prefix) returning sum of all values for keys with that prefix.

**Approach**: Trie with value/count at each node. On insert, compute delta from old value. Sum = sum of values in subtree under prefix.

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

## 06. Index Pairs of a String

**Description**: Given text and list of words, find all [start, end] pairs where text[start:end] is in words.

**Approach**: Build trie from words. For each starting index i, traverse trie with text[i:], record (i, j) when is_end at position j.

```python
def indexPairs(text, words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = True
    res = []
    for i in range(len(text)):
        node = trie
        for j in range(i, len(text)):
            if text[j] not in node:
                break
            node = node[text[j]]
            if '#' in node:
                res.append([i, j])
    return res
```

Time: O(n * m + W) | Space: O(W)

---

## 07. Count Prefixes of a Given String

**Description**: Count how many words in array are prefix of given string.

**Approach**: Build trie from words. Traverse string in trie; at each step count words (is_end) under current node. Or check each word with string.startswith(word).

```python
def countPrefixes(words, s):
    return sum(1 for w in words if s.startswith(w))
```

Time: O(n * m) | Space: O(1)

---

## 08. Stream of Characters

**Description**: Design StreamChecker that receives stream of chars and checks if any suffix of stream so far is in words.

**Approach**: Build trie from reversed words. Maintain recent chars. For each new char, check reversed suffixes against trie.

```python
class StreamChecker:
    def __init__(self, words):
        self.trie = {}
        for w in words:
            node = self.trie
            for c in reversed(w):
                node = node.setdefault(c, {})
            node['#'] = True
        self.stream = []

    def query(self, letter):
        self.stream.append(letter)
        node = self.trie
        for c in reversed(self.stream):
            if c not in node:
                return False
            node = node[c]
            if '#' in node:
                return True
        return False
```

Time: O(m) per query | Space: O(W)

---

## 09. Search Suggestions System

**Description**: As user types, suggest top 3 products with matching prefix.

**Approach**: Sort products, use binary search for prefix. Or build trie, DFS to collect words, sort by frequency and take top 3.

```python
def suggestedProducts(products, searchWord):
    products.sort()
    res, prefix = [], ""
    for c in searchWord:
        prefix += c
        res.append([p for p in products if p.startswith(prefix)][:3])
    return res
```

Time: O(n log n + m) | Space: O(1)

---

## 10. Shortest Word Distance

**Description**: Given list of words and two words, find shortest distance between their occurrences.

**Approach**: Store indices in list per word. Two pointers to find min difference. Trie can store word positions.

```python
def shortestDistance(words, word1, word2):
    i1, i2, res = -1, -1, float('inf')
    for i, w in enumerate(words):
        if w == word1:
            i1 = i
        elif w == word2:
            i2 = i
        if i1 != -1 and i2 != -1:
            res = min(res, abs(i1 - i2))
    return res
```

Time: O(n) | Space: O(1)

---

## 11. Word Pattern

**Description**: Check if pattern matches string (bijection between pattern chars and words).

**Approach**: Two HashMaps for mapping. Trie not typical.

```python
def wordPattern(pattern, s):
    words = s.split()
    if len(pattern) != len(words):
        return False
    p2w, w2p = {}, {}
    for p, w in zip(pattern, words):
        if (p in p2w and p2w[p] != w) or (w in w2p and w2p[w] != p):
            return False
        p2w[p], w2p[w] = w, p
    return True
```

Time: O(n) | Space: O(n)

---

## 12. Isomorphic Strings

**Description**: Check if two strings are isomorphic (character mapping).

**Approach**: Build mapping both ways. Trie not needed.

```python
def isIsomorphic(s, t):
    if len(s) != len(t):
        return False
    m1, m2 = {}, {}
    for a, b in zip(s, t):
        if (a in m1 and m1[a] != b) or (b in m2 and m2[b] != a):
            return False
        m1[a], m2[b] = b, a
    return True
```

Time: O(n) | Space: O(1)

---

## 13. Valid Anagram

**Description**: Check if two strings are anagrams.

**Approach**: Count array or sort. Trie can store sorted anagram groups.

```python
def isAnagram(s, t):
    return sorted(s) == sorted(t) if len(s) == len(t) else False
```

Time: O(n log n) | Space: O(n)

---

## 14. First Unique Character in a String

**Description**: Find index of first non-repeating character.

**Approach**: Count frequency, scan for first with count 1. Trie can track first occurrence.

```python
def firstUniqChar(s):
    from collections import Counter
    c = Counter(s)
    for i, ch in enumerate(s):
        if c[ch] == 1:
            return i
    return -1
```

Time: O(n) | Space: O(1)

---

## 15. Find the Difference

**Description**: String t is s with one extra letter. Find the extra letter.

**Approach**: XOR or count array. Trie not needed.

```python
def findTheDifference(s, t):
    res = 0
    for c in s + t:
        res ^= ord(c)
    return chr(res)
```

Time: O(n) | Space: O(1)

---

## 16. Reverse Words in a String III

**Description**: Reverse each word in string, keep word order.

**Approach**: Split, reverse each, join. Trie not needed.

```python
def reverseWords(s):
    return ' '.join(w[::-1] for w in s.split())
```

Time: O(n) | Space: O(n)

---

## 17. Count Binary Substrings

**Description**: Count contiguous substrings with same number of 0s and 1s.

**Approach**: Group consecutive same chars, adjacent groups contribute min(count1, count2). Trie not needed.

```python
def countBinarySubstrings(s):
    prev, curr, res = 0, 1, 0
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            curr += 1
        else:
            prev, curr = curr, 1
        if prev >= curr:
            res += 1
    return res
```

Time: O(n) | Space: O(n)

---

## 18. To Lower Case

**Description**: Convert string to lowercase.

**Approach**: Built-in or char-by-char. Trie not needed.

```python
def toLowerCase(s):
    return s.lower()
```

Time: O(n) | Space: O(n)

---

## 19. Robot Return to Origin

**Description**: Check if moves return robot to origin.

**Approach**: Count U-D and L-R. Trie not needed.

```python
def judgeCircle(moves):
    return moves.count('U') == moves.count('D') and moves.count('L') == moves.count('R')
```

Time: O(n) | Space: O(1)

---

## 20. Defanging an IP Address

**Description**: Replace '.' with '[.]' in IP string.

**Approach**: String replace. Trie not needed.

```python
def defangIPaddr(address):
    return address.replace('.', '[.]')
```

Time: O(n) | Space: O(n)

---

## 21. Jewels and Stones

**Description**: Count how many chars of stones are in jewels.

**Approach**: Set of jewels, count stones in set. Trie can store jewels for prefix matching.

```python
def numJewelsInStones(jewels, stones):
    j = set(jewels)
    return sum(1 for s in stones if s in j)
```

Time: O(n) | Space: O(j)

---

## 22. Unique Morse Code Words

**Description**: Count unique morse code representations of words.

**Approach**: Convert each word to morse, add to set, return size. Trie can store morse strings.

```python
def uniqueMorseRepresentations(words):
    morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
    return len(set(''.join(morse[ord(c)-97] for c in w) for w in words))
```

Time: O(n * m) | Space: O(n)

---

## 23. Goat Latin

**Description**: Apply goat latin rules to sentence.

**Approach**: Split, transform each word by rules, join. Trie not needed.

```python
def toGoatLatin(sentence):
    vowels = set('aeiouAEIOU')
    words = sentence.split()
    return ' '.join((w if w[0] in vowels else w[1:]+w[0]) + 'ma' + 'a'*(i+1) for i, w in enumerate(words))
```

Time: O(n) | Space: O(n)

---

## 24. Buddy Strings

**Description**: Check if swap of two chars in A can make B.

**Approach**: If A == B, need duplicate char. Else exactly two positions differ and A[i]=B[j], A[j]=B[i]. Trie not needed.

```python
def buddyStrings(s, goal):
    if len(s) != len(goal):
        return False
    if s == goal:
        return len(set(s)) < len(s)
    diff = [(a, b) for a, b in zip(s, goal) if a != b]
    return len(diff) == 2 and diff[0] == (diff[1][1], diff[1][0])
```

Time: O(n) | Space: O(1)

---

## 25. Longest Uncommon Subsequence I

**Description**: Find longest uncommon subsequence of two strings.

**Approach**: If A == B return -1. Else return max(len(A), len(B)). Trie not needed.

```python
def findLUSlength(a, b):
    return -1 if a == b else max(len(a), len(b))
```

Time: O(n) | Space: O(1)
