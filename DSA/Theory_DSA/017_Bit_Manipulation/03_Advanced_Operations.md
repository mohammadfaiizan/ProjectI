# Advanced Bit Operations

## Reverse Bits of Integer

```python
def reverse_bits(n: int, bits: int = 32) -> int:
    result = 0
    for i in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def reverse_bits_v2(n: int, bits: int = 32) -> int:
    result = 0
    for i in range(bits):
        if n & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result
```

## Rotate Bits Left

```python
def rotate_left(n: int, d: int, bits: int = 32) -> int:
    d = d % bits
    mask = (1 << bits) - 1
    n &= mask
    return ((n << d) | (n >> (bits - d))) & mask
```

## Rotate Bits Right

```python
def rotate_right(n: int, d: int, bits: int = 32) -> int:
    d = d % bits
    mask = (1 << bits) - 1
    n &= mask
    return ((n >> d) | (n << (bits - d))) & mask
```

## Swap Two Numbers Without Temp (XOR)

```python
def swap_xor(a: int, b: int) -> tuple[int, int]:
    a ^= b
    b ^= a
    a ^= b
    return a, b
```

## Swap All Even and Odd Bits

```python
def swap_even_odd_bits(n: int) -> int:
    even_bits = n & 0xAAAAAAAA
    odd_bits = n & 0x55555555
    return (even_bits >> 1) | (odd_bits << 1)
```

## Next Higher Number with Same Set Bits

```python
def next_higher_same_set_bits(n: int) -> int:
    if n == 0:
        return 0
    right_one = n & -n
    next_higher_one = n + right_one
    right_ones_pattern = n ^ next_higher_one
    right_ones_pattern = (right_ones_pattern // right_one) >> 2
    return next_higher_one | right_ones_pattern
```

## Add Two Numbers Without Arithmetic (XOR sum, AND+shift carry)

```python
def add_without_arithmetic(a: int, b: int) -> int:
    while b != 0:
        carry = a & b
        a = a ^ b
        b = carry << 1
    return a
```

## Subtract Without Arithmetic

```python
def subtract_without_arithmetic(a: int, b: int) -> int:
    while b != 0:
        borrow = (~a) & b
        a = a ^ b
        b = borrow << 1
    return a

def subtract_v2(a: int, b: int) -> int:
    return add_without_arithmetic(a, add_without_arithmetic(~b, 1))
```

## Multiply Without Arithmetic (Shift and Add)

```python
def multiply_without_arithmetic(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result += a
        a <<= 1
        b >>= 1
    return result
```

## Divide Without Arithmetic (Shift and Subtract)

```python
def divide_without_arithmetic(dividend: int, divisor: int) -> int:
    if divisor == 0:
        raise ValueError("Division by zero")
    sign = -1 if (dividend < 0) ^ (divisor < 0) else 1
    dividend = abs(dividend)
    divisor = abs(divisor)
    quotient = 0
    while dividend >= divisor:
        temp = divisor
        multiple = 1
        while dividend >= (temp << 1):
            temp <<= 1
            multiple <<= 1
        dividend -= temp
        quotient += multiple
    return sign * quotient
```

## Absolute Value Without Branching

```python
def abs_without_branching(n: int, bits: int = 32) -> int:
    mask = n >> (bits - 1)
    return (n + mask) ^ mask

def abs_without_branching_v2(n: int) -> int:
    mask = n >> 63
    return (n ^ mask) - mask
```

## Compute Sign

```python
def compute_sign(n: int) -> int:
    return (n >> 31) | (-n >> 31) if n != 0 else 0

def sign(n: int) -> int:
    return 1 if n > 0 else (-1 if n < 0 else 0)
```

## Min/Max Without Branching

```python
def min_without_branching(a: int, b: int) -> int:
    return b ^ ((a ^ b) & -(a < b))

def max_without_branching(a: int, b: int) -> int:
    return a ^ ((a ^ b) & -(a < b))
```

## Check Opposite Signs

```python
def opposite_signs(a: int, b: int) -> bool:
    return (a ^ b) < 0
```

## Compute Parity

```python
def compute_parity(n: int) -> int:
    parity = 0
    while n:
        parity ^= 1
        n &= n - 1
    return parity
```

## Count Bits to Flip A to B (XOR and count)

```python
def count_bits_to_flip(a: int, b: int) -> int:
    xor_val = a ^ b
    count = 0
    while xor_val:
        count += 1
        xor_val &= xor_val - 1
    return count
```

## Binary Representation of Float

```python
import struct

def float_to_binary(f: float) -> str:
    packed = struct.pack('>f', f)
    bits = struct.unpack('>I', packed)[0]
    return format(bits, '032b')

def double_to_binary(d: float) -> str:
    packed = struct.pack('>d', d)
    bits = struct.unpack('>Q', packed)[0]
    return format(bits, '064b')
```

## Longest Sequence of 1s by Flipping One 0

```python
def longest_ones_flip_one_zero(n: int) -> int:
    if n == 0 or n == -1:
        return 32
    prev_len = 0
    curr_len = 0
    max_len = 1
    while n != 0:
        if n & 1:
            curr_len += 1
        else:
            prev_len = curr_len if (n & 2) else 0
            curr_len = 0
        max_len = max(max_len, prev_len + curr_len + 1)
        n >>= 1
    return max_len

def longest_ones_flip_one_zero_array(arr: list[int]) -> int:
    prev_cnt = 0
    curr_cnt = 0
    max_cnt = 0
    for num in arr:
        if num == 1:
            curr_cnt += 1
        else:
            prev_cnt = curr_cnt
            curr_cnt = 0
        max_cnt = max(max_cnt, prev_cnt + curr_cnt + 1)
    return min(max_cnt, len(arr))
```
