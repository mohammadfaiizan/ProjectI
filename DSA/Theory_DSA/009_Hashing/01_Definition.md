# Hashing - Definition and Fundamentals

## Hash Table Concept

A hash table (hash map) is a data structure that implements an associative array abstract data type. It maps keys to values using a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.

Core idea: given a key, a hash function h(k) produces an index in the range [0, m-1] where m is the number of buckets. The key-value pair is stored at that index (or in a chain/list at that index when collisions occur).

## Key-Value Mapping

- **Key**: The identifier used to look up a value. Keys must be hashable (immutable and comparable in most implementations).
- **Value**: The data associated with the key. Can be any type.
- **Mapping**: Each key maps to at most one value. Inserting the same key again overwrites the previous value.

## Hash Map vs Hash Set

| Aspect | Hash Map | Hash Set |
|--------|----------|----------|
| Stores | Key-value pairs | Keys only (no value) |
| Use case | Lookup value by key | Membership test, deduplication |
| Duplicate keys | Not allowed (overwrites) | Not allowed |
| Operations | put(k,v), get(k), remove(k) | add(k), contains(k), remove(k) |
| Python type | dict | set |
| Typical implementation | Array of (key, value) pairs per bucket | Array of keys per bucket |

## Hash Function

A hash function h: U -> [0, m-1] maps keys from universe U to bucket indices.

Properties of a good hash function:
1. **Deterministic**: Same key always produces same index
2. **Uniform distribution**: Keys should spread evenly across buckets to minimize collisions
3. **Fast to compute**: O(1) or O(len(key)) for strings
4. **Minimal collisions**: Different keys should rarely map to same index

## Load Factor

Load factor alpha = n / m, where:
- n = number of key-value pairs stored
- m = number of buckets

Higher load factor means more collisions and longer chains. Typical thresholds:
- Separate chaining: 0.75 to 1.0 before resize
- Open addressing: 0.5 to 0.75 before resize

## Rehashing

When load factor exceeds a threshold, the table is resized (typically doubled) and all entries are reinserted using the hash function with the new m.

**When**: After insert when alpha exceeds threshold (e.g., 0.75).

**How**:
1. Allocate new array of size 2*m (or next prime)
2. For each entry in old table, compute new index with h(key) mod new_m
3. Insert into new table
4. Discard old table

Rehashing is O(n) but amortized over many inserts, so average insert remains O(1).

## Time Complexity

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Insert | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Lookup | O(1) | O(n) |
| Get keys/values | O(n) | O(n) |

Worst case occurs when all keys hash to the same bucket (bad hash function or adversarial input), degenerating to a linked list.

## Python dict and set Internals

- **dict**: Hash table with open addressing (CPython uses a variant with indices). Keys must be hashable. Resizes when 2/3 full. Average O(1) for get/set/del.
- **set**: Same underlying structure as dict but stores only keys (no values). Used for O(1) membership and deduplication.
- **Hashable**: An object is hashable if it has a __hash__ method and __eq__. Immutable types (int, str, tuple of hashables) are hashable. Mutable types (list, dict, set) are not.

## When to Use Hashing

| Use Case | Description |
|----------|-------------|
| Fast lookup | O(1) find by key instead of O(n) linear search |
| Counting | Frequency map: key -> count |
| Deduplication | Use set to remove duplicates in O(n) |
| Grouping | Group items by some key (e.g., anagrams by sorted string) |
| Caching | Memoization, LRU cache |
| Two-sum style | Store complements for O(n) pair finding |
| Subarray sum | Prefix sum with hash for O(n) subarray queries |

## Time Complexity Table

| Operation | Hash Map | Hash Set |
|-----------|----------|----------|
| Insert/Add | O(1) avg | O(1) avg |
| Delete/Remove | O(1) avg | O(1) avg |
| Lookup/Contains | O(1) avg | O(1) avg |
| Iterate all | O(n) | O(n) |
| Get size | O(1) | O(1) |
| Clear | O(n) | O(n) |
| Space | O(n) | O(n) |
