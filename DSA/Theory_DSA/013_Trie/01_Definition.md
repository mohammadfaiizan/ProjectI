# Trie - Definition and Concepts

## Trie (Prefix Tree) Concept

A trie, also called a prefix tree or digital tree, is a tree-like data structure used to store a dynamic set of strings where keys are usually strings. Unlike a binary search tree, no node in the tree stores the key associated with that node. Instead, the node's position in the tree defines the key with which it is associated. All descendants of a node share a common prefix of the string associated with that node, and the root is associated with the empty string.

## Node Structure

### Children Map or Array

Each trie node typically contains:

- **children**: A mapping from characters to child nodes. Implemented as either:
  - Array of size 26 (for lowercase English letters): `children[0]` for 'a', `children[25]` for 'z'
  - HashMap/dictionary: maps character to child node, supports any character set
- **isEnd**: Boolean flag indicating whether a complete word ends at this node

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

### Array-Based Node (Lowercase Letters Only)

```python
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False
```

## Properties

- **Common prefixes share nodes**: Words like "cat", "car", "card" share the path for "ca"
- **Depth equals word length**: The depth of a node from root equals the length of the prefix it represents
- **Root represents empty string**: The root node corresponds to the empty prefix
- **Path from root to any node**: Uniquely identifies a prefix; if `is_end` is true, that path is a complete word

## Trie vs Hash Map

| Aspect | Trie | Hash Map |
|--------|------|----------|
| Prefix queries | O(L) to find all words with prefix | O(n) must scan all keys |
| Sorted order | DFS yields lexicographic order | No inherent order |
| Memory | Shares storage for common prefixes | Each key stored independently |
| Exact search | O(L) | O(L) average |
| Prefix search | Native, efficient | Not supported directly |

## Trie vs BST

| Aspect | Trie | BST |
|--------|------|-----|
| Key structure | String, character by character | Comparable key as whole |
| Prefix queries | Natural, O(L) | Requires range query O(k + log n) |
| Order | Lexicographic by DFS | Inorder for sorted |
| Space | Shares prefixes | Each key stored |

## Time Complexity

Let L = length of the word/prefix, n = number of words, k = alphabet size.

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(L) | O(L) for new nodes |
| Search (exact) | O(L) | O(1) |
| Search (prefix) | O(L) | O(1) |
| Delete | O(L) | O(1) |
| Find all with prefix | O(L + m) | O(m) where m = output size |

## When to Use Trie

- **Autocomplete**: Suggest completions as user types
- **Spell check**: Find words within edit distance
- **IP routing**: Longest prefix match for routing tables
- **Word games**: Scrabble, Boggle, word search
- **Text search**: Substring, prefix, suffix queries
- **XOR problems**: Bitwise trie for maximum XOR queries
