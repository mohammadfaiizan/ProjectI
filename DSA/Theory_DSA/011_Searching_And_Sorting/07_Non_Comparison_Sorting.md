# Non-Comparison Sorting

## Counting Sort

**Idea:** Count frequency of each value. Compute cumulative positions. Place each element in output using count as index, then decrement count. Stable.

**Steps:**
1. Find range [min_val, max_val]
2. Count frequency of each value
3. Compute cumulative count (positions)
4. Traverse input backwards, place each at count[val]-1, decrement count[val]

**Complexity:** O(n + k) where k = range. Stable.

```python
def counting_sort(arr):
    if not arr:
        return []
    min_val, max_val = min(arr), max(arr)
    k = max_val - min_val + 1
    count = [0] * k
    for x in arr:
        count[x - min_val] += 1
    for i in range(1, k):
        count[i] += count[i - 1]
    output = [0] * len(arr)
    for x in reversed(arr):
        idx = x - min_val
        output[count[idx] - 1] = x
        count[idx] -= 1
    return output
```

---

## Radix Sort LSD (Least Significant Digit)

**Idea:** Sort by each digit from least to most significant. Use stable sort (counting sort) per digit. O(d * (n + k)) where d = digits, k = 10 for decimal.

```python
def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        idx = (arr[i] // exp) % 10
        count[idx] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        idx = (arr[i] // exp) % 10
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1
    for i in range(n):
        arr[i] = output[i]

def radix_sort_lsd(arr):
    if not arr:
        return
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
```

---

## Radix Sort MSD (Most Significant Digit)

**Idea:** Sort by most significant digit first. Recursively sort each bucket by next digit. Can terminate early for sparse data.

```python
def radix_sort_msd(arr, exp=None):
    if not arr:
        return []
    if exp is None:
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            exp *= 10
        exp //= 10
    if exp == 0:
        return arr
    buckets = [[] for _ in range(10)]
    for x in arr:
        idx = (x // exp) % 10
        buckets[idx].append(x)
    result = []
    for bucket in buckets:
        if bucket:
            result.extend(radix_sort_msd(bucket, exp // 10))
    return result
```

---

## Bucket Sort

**Idea:** Distribute elements into buckets by range. Sort each bucket (insertion sort). Concatenate. Works well for uniform distribution.

**Complexity:** O(n + k) average. O(n^2) worst if all in one bucket.

```python
def bucket_sort(arr, bucket_count=10):
    if not arr:
        return []
    min_val, max_val = min(arr), max(arr)
    if min_val == max_val:
        return arr[:]
    bucket_range = (max_val - min_val) / bucket_count + 1e-9
    buckets = [[] for _ in range(bucket_count)]
    for x in arr:
        idx = min(int((x - min_val) / bucket_range), bucket_count - 1)
        buckets[idx].append(x)
    for bucket in buckets:
        bucket.sort()
    return [x for bucket in buckets for x in bucket]
```

---

## Pigeonhole Sort

**Idea:** When range of values (max - min + 1) is approximately n. Create holes for each possible value. Place each element in its hole. Scan holes in order for output.

**Use case:** Integer array where range is O(n).

```python
def pigeonhole_sort(arr):
    if not arr:
        return []
    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1
    holes = [0] * size
    for x in arr:
        holes[x - min_val] += 1
    i = 0
    for count in range(size):
        while holes[count] > 0:
            arr[i] = count + min_val
            holes[count] -= 1
            i += 1
    return arr
```

---

## When Each Beats Comparison Sorts

| Sort | Beats O(n log n) when |
|------|------------------------|
| Counting | Integer keys, range k = O(n) or small |
| Radix | Integer/string keys, fixed digit length d, d * n < n log n |
| Bucket | Uniform distribution, keys in [0,1) or known range |
| Pigeonhole | Range of values approx n, integer keys |

**Rule of thumb:** Use non-comparison when key structure allows (integers, bounded range, uniform). Otherwise use comparison sort.
