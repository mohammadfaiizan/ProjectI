# Hash Functions

A hash function maps keys from a large universe to a small set of indices. Good hash functions minimize collisions and distribute keys uniformly.

## Division Method

h(k) = k mod m

Choose m as a prime not close to a power of 2 to avoid patterns in low-order bits. Avoid m = 2^p if keys have low-order bit patterns.

```python
def division_hash(key, m):
    return key % m

def division_hash_prime(key, m):
    return abs(hash(key)) % m
```

Choosing m: Use prime numbers (e.g., 1009, 10007) for better distribution. For power-of-2 sizes, use high-order bits: k mod 2^p uses p low-order bits which may be non-uniform.

## Multiplication Method

h(k) = floor(m * (k * A mod 1)) where 0 < A < 1.

(k * A mod 1) extracts fractional part. Good choice: A = (sqrt(5)-1)/2 approximately 0.618.

```python
def multiplication_hash(key, m, A=0.618033988749895):
    frac = (key * A) % 1
    return int(m * frac) % m

def multiplication_hash_signed(key, m):
    key = key if key >= 0 else -key
    A = 0.618033988749895
    return int(m * ((key * A) % 1)) % m
```

## Mid-Square Method

Square the key and extract middle digits as the hash. Good for numeric keys.

```python
def mid_square_hash(key, m):
    squared = key * key
    s = str(squared)
    mid = len(s) // 2
    half = (len(s) - 1) // 2
    extracted = s[mid - half:mid + half + 1]
    return int(extracted or "0") % m
```

## Folding Method

Split key into parts, add (or XOR) them, then mod m.

```python
def folding_hash(key, m, part_size=4):
    s = str(abs(key))
    total = 0
    for i in range(0, len(s), part_size):
        part = s[i:i + part_size]
        total += int(part)
    return total % m
```

## Polynomial Rolling Hash for Strings

h(s) = (s[0] * p^0 + s[1] * p^1 + ... + s[n-1] * p^(n-1)) mod m

Used in Rabin-Karp. p is a prime (31, 37), m is large prime to avoid overflow.

```python
def polynomial_rolling_hash(s, p=31, m=10**9 + 7):
    h = 0
    pow_p = 1
    for c in s:
        h = (h + (ord(c) - ord('a') + 1) * pow_p) % m
        pow_p = (pow_p * p) % m
    return h

def polynomial_hash_prefix(s, p=31, m=10**9 + 7):
    n = len(s)
    h = [0] * (n + 1)
    pow_p = [1] * (n + 1)
    for i in range(n):
        h[i + 1] = (h[i] + (ord(s[i]) - ord('a') + 1) * pow_p[i]) % m
        pow_p[i + 1] = (pow_p[i] * p) % m
    return h, pow_p

def substring_hash(h, pow_p, l, r, m=10**9 + 7):
    inv = pow(pow_p[l], m - 2, m)
    return (h[r + 1] - h[l] + m) * inv % m
```

## Universal Hashing

Family H of hash functions. For any two distinct keys x, y, probability that h(x) = h(y) when h is chosen randomly from H is at most 1/m.

```python
import random

def universal_hash_family(m, p=None):
    if p is None:
        p = 2**61 - 1
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)
    def h(key):
        return ((a * key + b) % p) % m
    return h
```

## Perfect Hashing

Two-level scheme: first level hashes to buckets, second level uses a separate hash function per bucket chosen so no collisions occur within bucket. Used when key set is static and known in advance.

Overview: Build first-level table. For each bucket with more than one key, build a secondary table with a hash function that causes no collisions (try random functions until collision-free).

## Hash Function Properties

| Property | Description |
|----------|-------------|
| Deterministic | Same key always produces same hash |
| Uniform distribution | Keys spread evenly across buckets |
| Efficient | O(1) for integers, O(len(key)) for strings |
| Avalanche | Small change in input causes large change in output |
| Non-invertible | Cannot recover key from hash (for crypto hashes) |

## Comparison Table

| Method | Best For | Collision Resistance | Speed |
|--------|----------|----------------------|-------|
| Division | Integers, general | Depends on m | O(1) |
| Multiplication | Integers | Good with proper A | O(1) |
| Mid-square | Numeric keys | Moderate | O(1) |
| Folding | Long numeric keys | Moderate | O(k) |
| Polynomial rolling | Strings | Good | O(n) |
| Universal | Adversarial input | Theoretically 1/m | O(1) |
| Perfect | Static sets | Zero collisions | O(1) |
