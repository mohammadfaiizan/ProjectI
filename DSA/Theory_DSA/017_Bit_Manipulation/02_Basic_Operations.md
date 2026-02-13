# Basic Bit Operations

## Check if ith Bit is Set

```python
def is_bit_set(n: int, i: int) -> bool:
    return bool(n & (1 << i))
```

## Set ith Bit

```python
def set_bit(n: int, i: int) -> int:
    return n | (1 << i)
```

## Clear ith Bit

```python
def clear_bit(n: int, i: int) -> int:
    return n & ~(1 << i)
```

## Toggle ith Bit

```python
def toggle_bit(n: int, i: int) -> int:
    return n ^ (1 << i)
```

## Check Even/Odd

```python
def is_even(n: int) -> bool:
    return (n & 1) == 0

def is_odd(n: int) -> bool:
    return (n & 1) == 1
```

## Count Set Bits - Loop

```python
def count_set_bits_loop(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
```

## Count Set Bits - Brian Kernighan's Algorithm (O(set bits))

```python
def count_set_bits_kernighan(n: int) -> int:
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count
```

## Count Set Bits - Built-in

```python
def count_set_bits_builtin(n: int) -> int:
    return bin(n).count('1')

def count_set_bits_builtin2(n: int) -> int:
    return n.bit_count()
```

## Check Power of Two

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

## Find Rightmost Set Bit Position

```python
def rightmost_set_bit_position(n: int) -> int:
    if n == 0:
        return -1
    pos = 0
    while (n & 1) == 0:
        n >>= 1
        pos += 1
    return pos

def rightmost_set_bit_position_v2(n: int) -> int:
    if n == 0:
        return -1
    return (n & -n).bit_length() - 1
```

## Turn Off Rightmost Set Bit

```python
def turn_off_rightmost_set_bit(n: int) -> int:
    return n & (n - 1)
```

## Isolate Rightmost Set Bit

```python
def isolate_rightmost_set_bit(n: int) -> int:
    return n & -n
```

## Find Position of Only Set Bit

For numbers that are powers of two, find the 0-indexed position.

```python
def position_of_only_set_bit(n: int) -> int:
    if n == 0 or (n & (n - 1)) != 0:
        return -1
    pos = 0
    while n > 1:
        n >>= 1
        pos += 1
    return pos

def position_of_only_set_bit_v2(n: int) -> int:
    if n == 0 or (n & (n - 1)) != 0:
        return -1
    return n.bit_length() - 1
```

## Check Alternating Bits

```python
def has_alternating_bits(n: int) -> bool:
    x = n ^ (n >> 1)
    return (x & (x + 1)) == 0
```

## Count Trailing Zeros

```python
def count_trailing_zeros(n: int) -> int:
    if n == 0:
        return 32
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count

def count_trailing_zeros_v2(n: int) -> int:
    if n == 0:
        return 32
    return (n & -n).bit_length() - 1
```

## Count Leading Zeros

```python
def count_leading_zeros(n: int, bits: int = 32) -> int:
    if n == 0:
        return bits
    return bits - n.bit_length()
```

## Find MSB Position

```python
def find_msb_position(n: int) -> int:
    if n == 0:
        return -1
    return n.bit_length() - 1
```

## Find LSB Position

```python
def find_lsb_position(n: int) -> int:
    if n == 0:
        return -1
    return (n & -n).bit_length() - 1
```
