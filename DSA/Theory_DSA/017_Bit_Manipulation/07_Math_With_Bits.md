# Math with Bits

## Multiply Using Shift and Add

```python
def multiply_shift_add(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result += a
        a <<= 1
        b >>= 1
    return result
```

## Divide Using Shift and Subtract

```python
def divide_shift_subtract(dividend: int, divisor: int) -> int:
    if divisor == 0:
        raise ValueError("Division by zero")
    sign = -1 if (dividend < 0) ^ (divisor < 0) else 1
    dividend = abs(dividend)
    divisor = abs(divisor)
    quotient = 0
    while dividend >= divisor:
        temp = divisor
        count = 0
        while dividend >= (temp << 1):
            temp <<= 1
            count += 1
        dividend -= temp
        quotient += 1 << count
    return sign * quotient
```

## Modulo by Power of 2

```python
def modulo_power_of_two(n: int, k: int) -> int:
    return n & ((1 << k) - 1)
```

## Fast Exponentiation / Binary Exponentiation (a^b in O(log b))

```python
def binary_exponentiation(a: int, b: int) -> int:
    result = 1
    base = a
    while b:
        if b & 1:
            result *= base
        base *= base
        b >>= 1
    return result
```

## Modular Exponentation

```python
def modular_exponentiation(a: int, b: int, mod: int) -> int:
    result = 1
    a %= mod
    while b:
        if b & 1:
            result = (result * a) % mod
        a = (a * a) % mod
        b >>= 1
    return result
```

## GCD Using Binary GCD (Stein's Algorithm)

```python
def binary_gcd(a: int, b: int) -> int:
    if a == 0:
        return b
    if b == 0:
        return a
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1
    while (a & 1) == 0:
        a >>= 1
    while b:
        while (b & 1) == 0:
            b >>= 1
        if a > b:
            a, b = b, a
        b -= a
    return a << shift
```

## Count Set Bits for All 0 to n (Pattern-Based)

```python
def count_bits_all(n: int) -> list[int]:
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        result[i] = result[i >> 1] + (i & 1)
    return result
```

## Reverse Bits of 32-bit Integer

```python
def reverse_bits_32(n: int) -> int:
    n = (n >> 16) | (n << 16)
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)
    return n & 0xFFFFFFFF
```

## Popcount for All Numbers DP

```python
def popcount_dp(n: int) -> list[int]:
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        result[i] = result[i & (i - 1)] + 1
    return result
```

## Power of Two/Four/Eight Checks

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def is_power_of_four(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0 and (n & 0xAAAAAAAA) == 0

def is_power_of_eight(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0 and (n.bit_length() - 1) % 3 == 0
```

## Sum of Two Integers Without +

```python
def get_sum(a: int, b: int) -> int:
    mask = 0xFFFFFFFF
    while b:
        carry = (a & b) << 1
        a = (a ^ b) & mask
        b = carry & mask
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)
```

## Gray Code Generation

```python
def gray_code(n: int) -> list[int]:
    return [i ^ (i >> 1) for i in range(1 << n)]
```

## Circular Permutation in Binary

```python
def circular_permutation(n: int, start: int) -> list[int]:
    gray = [i ^ (i >> 1) for i in range(1 << n)]
    idx = gray.index(start)
    return gray[idx:] + gray[:idx]
```
