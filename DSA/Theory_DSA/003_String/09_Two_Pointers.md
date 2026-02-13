# Two Pointers Theory and Implementations

## Theory

Two pointers use two indices moving through a sequence, often from opposite ends or at different speeds. Used for palindrome checks, in-place reversals, and window problems.

## Valid Palindrome

```python
def valid_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

## Valid Palindrome II

```python
def valid_palindrome_ii(s):
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    return True
```

## Reverse String In-Place

```python
def reverse_string_inplace(s):
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return "".join(chars)
```

## Reverse Vowels

```python
def reverse_vowels(s):
    vowels = set("aeiouAEIOU")
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        while left < right and chars[left] not in vowels:
            left += 1
        while left < right and chars[right] not in vowels:
            right -= 1
        if left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
    return "".join(chars)
```

## Reverse Words in String

```python
def reverse_words(s):
    return " ".join(s.split()[::-1])

def reverse_words_inplace_concept(s):
    words = s.split()
    left, right = 0, len(words) - 1
    while left < right:
        words[left], words[right] = words[right], words[left]
        left += 1
        right -= 1
    return " ".join(words)
```

## Remove Specified Chars

```python
def remove_chars(s, to_remove):
    to_remove_set = set(to_remove)
    result = []
    for c in s:
        if c not in to_remove_set:
            result.append(c)
    return "".join(result)

def remove_chars_inplace_concept(s, to_remove):
    to_remove_set = set(to_remove)
    chars = list(s)
    write = 0
    for read in range(len(chars)):
        if chars[read] not in to_remove_set:
            chars[write] = chars[read]
            write += 1
    return "".join(chars[:write])
```

## Partition Labels

```python
def partition_labels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    return result
```

## Backspace String Compare

```python
def backspace_compare(s, t):
    def process(s):
        stack = []
        for c in s:
            if c == "#":
                if stack:
                    stack.pop()
            else:
                stack.append(c)
        return "".join(stack)
    return process(s) == process(t)

def backspace_compare_two_pointers(s, t):
    def next_valid(s, i):
        backspaces = 0
        while i >= 0:
            if s[i] == "#":
                backspaces += 1
            elif backspaces > 0:
                backspaces -= 1
            else:
                return i - 1, s[i]
            i -= 1
        return -1, ""

    i, j = len(s) - 1, len(t) - 1
    while True:
        i, ci = next_valid(s, i)
        j, cj = next_valid(t, j)
        if ci != cj:
            return False
        if ci == "" and cj == "":
            return True
```

## Long Pressed Name

```python
def is_long_pressed_name(name, typed):
    i = j = 0
    while j < len(typed):
        if i < len(name) and name[i] == typed[j]:
            i += 1
            j += 1
        elif j > 0 and typed[j] == typed[j - 1]:
            j += 1
        else:
            return False
    return i == len(name)
```

## Sentence Reversal

```python
def reverse_sentence(s):
    return " ".join(reversed(s.split()))
```

## Word Reversal Within Sentence

```python
def reverse_words_in_sentence(s):
    return " ".join(word[::-1] for word in s.split())

def reverse_words_preserve_spaces(s):
    result = []
    word = []
    for c in s:
        if c == " ":
            if word:
                result.append("".join(reversed(word)))
                word = []
            result.append(" ")
        else:
            word.append(c)
    if word:
        result.append("".join(reversed(word)))
    return "".join(result)
```
