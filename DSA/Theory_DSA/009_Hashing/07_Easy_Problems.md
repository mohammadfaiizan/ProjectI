# Easy Hashing Problems

## 1. Two Sum

Given an array and target, return indices of two numbers that add up to target. Hash map storing value to index. For each element, check if complement (target - value) exists.

```python
def twoSum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```

Time: O(n) | Space: O(n)

---

## 2. Contains Duplicate

Return true if array has any duplicate element. Compare len(nums) with len(set(nums)) or use set to track seen elements.

```python
def containsDuplicate(nums):
    return len(nums) != len(set(nums))
```

Time: O(n) | Space: O(n)

---

## 3. Valid Anagram

Check if two strings are anagrams. Count characters in both strings; compare counts or use sorted strings as key.

```python
def isAnagram(s, t):
    from collections import Counter
    return Counter(s) == Counter(t)
```

Time: O(n) | Space: O(1) for fixed alphabet

---

## 4. First Unique Character in a String

Find index of first non-repeating character. Count frequency with Counter, scan for first char with count 1.

```python
def firstUniqChar(s):
    from collections import Counter
    cnt = Counter(s)
    for i, c in enumerate(s):
        if cnt[c] == 1:
            return i
    return -1
```

Time: O(n) | Space: O(1)

---

## 5. Intersection of Two Arrays

Return unique elements common to both arrays. Convert both to sets, return intersection.

```python
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))
```

Time: O(n + m) | Space: O(n + m)

---

## 6. Intersection of Two Arrays II

Return intersection with duplicates preserved (min count). Count one array, iterate second and decrement count when found.

```python
def intersect(nums1, nums2):
    from collections import Counter
    c = Counter(nums1)
    out = []
    for x in nums2:
        if c[x] > 0:
            out.append(x)
            c[x] -= 1
    return out
```

Time: O(n + m) | Space: O(min(n, m))

---

## 7. Happy Number

Determine if number reaches 1 when repeatedly summing squares of digits. Set to detect cycle; if we see a number again, not happy.

```python
def isHappy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d)**2 for d in str(n))
    return n == 1
```

Time: O(log n) | Space: O(log n)

---

## 8. Isomorphic Strings

Check if two strings have one-to-one character mapping. Two hash maps: s->t and t->s. Verify consistency for each pair.

```python
def isIsomorphic(s, t):
    st, ts = {}, {}
    for a, b in zip(s, t):
        if (a in st and st[a] != b) or (b in ts and ts[b] != a):
            return False
        st[a], ts[b] = b, a
    return True
```

Time: O(n) | Space: O(1)

---

## 9. Word Pattern

Check if pattern matches space-separated words bijectively. Same as isomorphic: pattern char to word and word to pattern char.

```python
def wordPattern(pattern, s):
    words = s.split()
    if len(pattern) != len(words):
        return False
    pw, wp = {}, {}
    for p, w in zip(pattern, words):
        if (p in pw and pw[p] != w) or (w in wp and wp[w] != p):
            return False
        pw[p], wp[w] = w, p
    return True
```

Time: O(n) | Space: O(n)

---

## 10. Contains Duplicate II

Check if duplicate exists within distance k. Sliding window with set or dict storing last index per value.

```python
def containsNearbyDuplicate(nums, k):
    seen = {}
    for i, x in enumerate(nums):
        if x in seen and i - seen[x] <= k:
            return True
        seen[x] = i
    return False
```

Time: O(n) | Space: O(min(n, k))

---

## 11. Ransom Note

Check if magazine has enough letters to form ransom note. Count magazine chars, decrement for each ransom char.

```python
def canConstruct(ransomNote, magazine):
    from collections import Counter
    c = Counter(magazine)
    for ch in ransomNote:
        if c[ch] <= 0:
            return False
        c[ch] -= 1
    return True
```

Time: O(n + m) | Space: O(1)

---

## 12. Jewels and Stones

Count how many stones are jewels. Set of jewels, count stones in set.

```python
def numJewelsInStones(jewels, stones):
    j = set(jewels)
    return sum(1 for s in stones if s in j)
```

Time: O(n + m) | Space: O(j)

---

## 13. Find the Difference

Find the one extra character in string t compared to s. Count chars in both, find the one with different count.

```python
def findTheDifference(s, t):
    from collections import Counter
    return list((Counter(t) - Counter(s)).keys())[0]
```

Time: O(n) | Space: O(1)

---

## 14. Single Number

Find the single non-duplicate in array where others appear twice. XOR all elements (no hash needed) or use Counter.

```python
def singleNumber(nums):
    res = 0
    for x in nums:
        res ^= x
    return res
```

Time: O(n) | Space: O(1)

---

## 15. Majority Element

Find element appearing more than n/2 times. Boyer-Moore voting or Counter.most_common(1).

```python
def majorityElement(nums):
    cand, cnt = None, 0
    for x in nums:
        if cnt == 0:
            cand = x
        cnt += 1 if x == cand else -1
    return cand
```

Time: O(n) | Space: O(1)

---

## 16. Find All Numbers Disappeared in an Array

Array 1..n, some missing; return missing numbers. Mark indices by negating; unmarked indices are missing.

```python
def findDisappearedNumbers(nums):
    for x in nums:
        i = abs(x) - 1
        if nums[i] > 0:
            nums[i] *= -1
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]
```

Time: O(n) | Space: O(1)

---

## 17. Find All Duplicates in an Array

Array 1..n, some appear twice; return duplicates. Same marking; when we try to mark already negative, it is duplicate.

```python
def findDuplicates(nums):
    out = []
    for x in nums:
        i = abs(x) - 1
        if nums[i] < 0:
            out.append(abs(x))
        else:
            nums[i] *= -1
    return out
```

Time: O(n) | Space: O(1)

---

## 18. Keyboard Row

Return words that can be typed using one keyboard row. Map each letter to row number; check all chars in word same row.

```python
def findWords(words):
    rows = [set("qwertyuiop"), set("asdfghjkl"), set("zxcvbnm")]
    out = []
    for w in words:
        r = next(r for r in rows if w[0].lower() in r)
        if all(c.lower() in r for c in w):
            out.append(w)
    return out
```

Time: O(n * m) | Space: O(1)

---

## 19. Distribute Candies

Max distinct candy types sister can get (n/2 max). min(len(set(candyType)), n//2).

```python
def distributeCandies(candyType):
    return min(len(set(candyType)), len(candyType) // 2)
```

Time: O(n) | Space: O(n)

---

## 20. Set Mismatch

Array 1..n with one duplicate and one missing; return [duplicate, missing]. Find duplicate via marking; missing = expected sum - actual sum + duplicate.

```python
def findErrorNums(nums):
    n = len(nums)
    total = n * (n + 1) // 2
    actual = sum(nums)
    dup = None
    for x in nums:
        i = abs(x) - 1
        if nums[i] < 0:
            dup = abs(x)
        else:
            nums[i] *= -1
    missing = total - actual + dup
    return [dup, missing]
```

Time: O(n) | Space: O(1)

---

## 21. Number of Good Pairs

Count pairs (i,j) with i<j and nums[i]==nums[j]. Count frequencies; each count c contributes c*(c-1)//2.

```python
def numIdenticalPairs(nums):
    from collections import Counter
    return sum(c * (c - 1) // 2 for c in Counter(nums).values())
```

Time: O(n) | Space: O(n)

---

## 22. Count Pairs with Given Difference K

Count pairs with absolute difference k. Counter; for each x, add count of x+k and x-k (handle k=0 separately).

```python
def countPairs(nums, k):
    from collections import Counter
    c = Counter(nums)
    if k == 0:
        return sum(v * (v - 1) // 2 for v in c.values())
    return sum(c.get(x + k, 0) for x in c)
```

Time: O(n) | Space: O(n)

---

## 23. Find Common Characters

Common characters across all strings with multiplicity. Intersect Counter of each string (use & operator).

```python
def commonChars(words):
    from collections import Counter
    res = Counter(words[0])
    for w in words[1:]:
        res &= Counter(w)
    return list(res.elements())
```

Time: O(n * m) | Space: O(m)

---

## 24. Subdomain Visit Count

Parse "cnt domain" and aggregate by domain and subdomains. Split domain by dots, add count to each suffix subdomain.

```python
def subdomainVisits(cpdomains):
    from collections import Counter
    cnt = Counter()
    for s in cpdomains:
        n, dom = s.split()
        n = int(n)
        parts = dom.split('.')
        for i in range(len(parts)):
            cnt['.'.join(parts[i:])] += n
    return [f"{v} {k}" for k, v in cnt.items()]
```

Time: O(n * m) | Space: O(n)

---

## 25. Most Common Word

Most frequent word not in banned list. Regex to extract words, Counter, filter banned.

```python
def mostCommonWord(paragraph, banned):
    import re
    from collections import Counter
    words = re.findall(r'\w+', paragraph.lower())
    banned = set(banned)
    return Counter(w for w in words if w not in banned).most_common(1)[0][0]
```

Time: O(n) | Space: O(n)

---

## 26. Unique Morse Code Words

Count unique morse representations of words. Map each word to morse string, add to set, return len(set).

```python
def uniqueMorseRepresentations(words):
    morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
    return len(set(''.join(morse[ord(c)-97] for c in w) for w in words))
```

Time: O(n * m) | Space: O(n)

---

## 27. Buddy Strings

Can we swap two chars in A to get B? If A==B, need at least one duplicate. Else need exactly two mismatches that swap correctly.

```python
def buddyStrings(s, goal):
    if len(s) != len(goal):
        return False
    if s == goal:
        return len(set(s)) < len(s)
    diffs = [(a, b) for a, b in zip(s, goal) if a != b]
    return len(diffs) == 2 and diffs[0] == (diffs[1][1], diffs[1][0])
```

Time: O(n) | Space: O(1)

---

## 28. Uncommon Words from Two Sentences

Words that appear exactly once across both sentences. Count all words; return those with count 1.

```python
def uncommonFromSentences(s1, s2):
    from collections import Counter
    c = Counter(s1.split() + s2.split())
    return [w for w, cnt in c.items() if cnt == 1]
```

Time: O(n + m) | Space: O(n + m)
