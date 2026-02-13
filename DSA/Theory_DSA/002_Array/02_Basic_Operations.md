# Basic Array Operations

## Create and Initialize

```python
def create_empty():
    return []

def create_with_size(n, default=0):
    return [default] * n

def create_from_values(*values):
    return list(values)

def create_range(n):
    return list(range(n))

def create_range_with_step(start, stop, step):
    return list(range(start, stop, step))
```

## Access by Index

```python
def access_by_index(arr, index):
    return arr[index]

def safe_access(arr, index, default=None):
    try:
        return arr[index]
    except IndexError:
        return default
```

## Traverse Forward

```python
def traverse_forward(arr):
    for i in range(len(arr)):
        print(arr[i])

def traverse_forward_enumerate(arr):
    for i, val in enumerate(arr):
        print(i, val)
```

## Traverse Backward

```python
def traverse_backward(arr):
    for i in range(len(arr) - 1, -1, -1):
        print(arr[i])

def traverse_backward_reversed(arr):
    for val in reversed(arr):
        print(val)
```

## Linear Search

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def linear_search_all(arr, target):
    return [i for i in range(len(arr)) if arr[i] == target]
```

## Insert at End (Append)

```python
def insert_at_end(arr, value):
    arr.append(value)
    return arr
```

## Insert at Beginning

```python
def insert_at_beginning(arr, value):
    arr.insert(0, value)
    return arr

def insert_at_beginning_manual(arr, value):
    return [value] + arr
```

## Insert at Index

```python
def insert_at_index(arr, index, value):
    arr.insert(index, value)
    return arr

def insert_at_index_manual(arr, index, value):
    return arr[:index] + [value] + arr[index:]
```

## Delete from End

```python
def delete_from_end(arr):
    if arr:
        arr.pop()
    return arr
```

## Delete from Beginning

```python
def delete_from_beginning(arr):
    if arr:
        arr.pop(0)
    return arr

def delete_from_beginning_slice(arr):
    return arr[1:] if arr else []
```

## Delete at Index

```python
def delete_at_index(arr, index):
    if 0 <= index < len(arr):
        arr.pop(index)
    return arr

def delete_at_index_slice(arr, index):
    return arr[:index] + arr[index + 1:] if 0 <= index < len(arr) else arr
```

## Delete by Value

```python
def delete_by_value(arr, value):
    arr.remove(value)
    return arr

def delete_by_value_first(arr, value):
    for i in range(len(arr)):
        if arr[i] == value:
            return arr[:i] + arr[i + 1:]
    return arr

def delete_all_occurrences(arr, value):
    return [x for x in arr if x != value]
```

## Find Length

```python
def find_length(arr):
    return len(arr)

def find_length_manual(arr):
    count = 0
    for _ in arr:
        count += 1
    return count
```

## Check Empty

```python
def check_empty(arr):
    return len(arr) == 0

def check_empty_bool(arr):
    return not arr
```

## Find Min

```python
def find_min(arr):
    if not arr:
        return None
    return min(arr)

def find_min_manual(arr):
    if not arr:
        return None
    m = arr[0]
    for x in arr[1:]:
        if x < m:
            m = x
    return m
```

## Find Max

```python
def find_max(arr):
    if not arr:
        return None
    return max(arr)

def find_max_manual(arr):
    if not arr:
        return None
    m = arr[0]
    for x in arr[1:]:
        if x > m:
            m = x
    return m
```

## Find Sum

```python
def find_sum(arr):
    return sum(arr)

def find_sum_manual(arr):
    total = 0
    for x in arr:
        total += x
    return total
```

## Find Average

```python
def find_average(arr):
    if not arr:
        return None
    return sum(arr) / len(arr)
```

## Copy (Shallow vs Deep)

```python
def copy_shallow(arr):
    return arr.copy()

def copy_shallow_slice(arr):
    return arr[:]

def copy_shallow_list(arr):
    return list(arr)

def copy_deep(arr):
    import copy
    return copy.deepcopy(arr)

def copy_deep_nested_manual(arr):
    return [x.copy() if isinstance(x, list) else x for x in arr]
```

## Compare Two Arrays

```python
def compare_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    for a, b in zip(arr1, arr2):
        if a != b:
            return False
    return True

def compare_arrays_direct(arr1, arr2):
    return arr1 == arr2
```

## Print Array

```python
def print_array(arr):
    print(arr)

def print_array_formatted(arr):
    print("[" + ", ".join(map(str, arr)) + "]")

def print_array_indexed(arr):
    for i, val in enumerate(arr):
        print(f"arr[{i}] = {val}")
```
