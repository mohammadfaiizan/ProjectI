# Basic Hashing Operations

## Insert Key-Value Pair

```python
def insert_key_value(d, key, value):
    d[key] = value

d = {}
insert_key_value(d, "a", 1)
insert_key_value(d, "b", 2)
```

## Delete by Key

```python
def delete_by_key(d, key):
    if key in d:
        del d[key]
        return True
    return False

def delete_by_key_pop(d, key):
    return d.pop(key, None)
```

## Lookup by Key

```python
def lookup(d, key):
    return d.get(key)

def lookup_with_default(d, key, default=None):
    return d.get(key, default)
```

## Check if Key Exists

```python
def key_exists(d, key):
    return key in d
```

## Get All Keys

```python
def get_all_keys(d):
    return list(d.keys())
```

## Get All Values

```python
def get_all_values(d):
    return list(d.values())
```

## Iterate Over Entries

```python
def iterate_entries(d):
    for key, value in d.items():
        yield key, value

def iterate_keys(d):
    for key in d:
        yield key

def iterate_values(d):
    for value in d.values():
        yield value
```

## Get Size

```python
def get_size(d):
    return len(d)
```

## Check isEmpty

```python
def is_empty(d):
    return len(d) == 0
```

## Clear All Entries

```python
def clear_entries(d):
    d.clear()
```

## Hash Set from Scratch (Array + Chaining)

```python
class HashSet:
    def __init__(self, capacity=16, load_factor=0.75):
        self.capacity = capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for key in bucket:
                self.add(key)

    def add(self, key):
        if self.size >= self.capacity * self.load_factor:
            self._resize()
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, k in enumerate(bucket):
            if k == key:
                return
        bucket.append(key)
        self.size += 1

    def remove(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, k in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def contains(self, key):
        idx = self._hash(key)
        return key in self.buckets[idx]

    def __len__(self):
        return self.size
```

## Hash Map from Scratch (Array of Buckets with Chaining)

```python
class HashMap:
    def __init__(self, capacity=16, load_factor=0.75):
        self.capacity = capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.put(k, v)

    def put(self, key, value):
        if self.size >= self.capacity * self.load_factor:
            self._resize()
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1

    def get(self, key, default=None):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return default

    def remove(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def contains(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return True
        return False

    def keys(self):
        return [k for b in self.buckets for k, _ in b]

    def values(self):
        return [v for b in self.buckets for _, v in b]

    def __len__(self):
        return self.size
```

## Handle Resize and Rehash

```python
def rehash(old_buckets, new_capacity):
    new_buckets = [[] for _ in range(new_capacity)]
    for bucket in old_buckets:
        for key, value in bucket:
            idx = hash(key) % new_capacity
            new_buckets[idx].append((key, value))
    return new_buckets
```

## defaultdict Usage

```python
from collections import defaultdict

def defaultdict_example():
    d = defaultdict(int)
    for x in [1, 2, 1, 3, 2, 1]:
        d[x] += 1

    d_list = defaultdict(list)
    d_list["a"].append(1)
    d_list["a"].append(2)

    d_set = defaultdict(set)
    d_set["x"].add(1)
    d_set["x"].add(1)
```

## Counter (Frequency Map)

```python
from collections import Counter

def counter_example():
    arr = [1, 2, 2, 3, 1, 2, 3, 3, 3]
    cnt = Counter(arr)
    most_common = cnt.most_common(2)
    total = sum(cnt.values())
    elements = list(cnt.elements())
```
