# String Definition and Fundamentals

## String as Character Sequence

A string is an ordered sequence of characters. Each character occupies a fixed number of bytes depending on the encoding. In Python, strings are sequences of Unicode code points, not raw bytes.

## Immutability in Python

Python strings are immutable. Once created, a string cannot be modified. Operations that appear to modify a string (concatenation, replacement) create new string objects.

```python
s = "hello"
s[0] = "H"
```

The above raises TypeError. To "modify" a string, you must create a new one:

```python
s = "h" + s[1:]
s = s.replace("h", "H")
```

Immutability enables:
- Safe sharing and caching (string interning)
- Use as dictionary keys and set elements
- Thread safety without locking

## Encoding: ASCII, UTF-8

**ASCII**: 7-bit encoding for 128 characters (0-127). Covers English letters, digits, basic symbols. Each character = 1 byte.

**UTF-8**: Variable-length encoding for Unicode. ASCII characters (0-127) use 1 byte. Other characters use 2-4 bytes. Backward compatible with ASCII.

```python
s = "hello"
s.encode("ascii")
s.encode("utf-8")
ord("A")
chr(65)
```

## Char Array vs String

| Aspect | Char Array (C/C++) | String (Python) |
|--------|-------------------|-----------------|
| Mutability | Mutable | Immutable |
| Length | Fixed or null-terminated | Dynamic, stored with object |
| Memory | Contiguous bytes | Object with length + data |
| Bounds | Manual management | Automatic |
| Encoding | Usually ASCII/bytes | Unicode by default |

In C, a string is a char array ending with null (`\0`). In Python, strings are first-class objects with length metadata.

## Time Complexity of Operations

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Access by index | O(1) | Direct indexing |
| Concatenate | O(n + m) | Creates new string, copies both |
| Substring (slicing) | O(k) | k = length of slice, copies k chars |
| Compare (==, <, >) | O(n) | Worst case scans full length |
| Find substring | O(n * m) | Naive; KMP gives O(n + m) |
| Length | O(1) | Stored in object |

## String Interning

Python interns some string literals at compile/load time. Identical literals may share the same object in memory.

```python
a = "hello"
b = "hello"
a is b
```

For short strings and literals, `is` may be True. Do not rely on interning for correctness; use `==` for equality. `sys.intern()` explicitly interns a string for optimization in symbol tables.

## When to Use String vs List of Chars

**Use String when**:
- Representing text, identifiers, paths
- Need immutability (keys, set elements)
- Pattern matching, parsing
- Output/display purposes

**Use List of Chars when**:
- Frequent in-place modifications
- Building string character by character (then join at end)
- Need mutable sequence for algorithm (e.g., two-pointer swap)

```python
chars = list("hello")
chars[0] = "H"
result = "".join(chars)
```

For many concatenations, building a list and joining is O(n) total; repeated string concatenation is O(n^2) in the worst case.
