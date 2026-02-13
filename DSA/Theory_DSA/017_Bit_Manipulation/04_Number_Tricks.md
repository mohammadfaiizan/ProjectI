# Number Tricks with Bits

## Check Even/Odd (& 1)

```python
def is_even(n: int) -> bool:
    return (n & 1) == 0
```

## Multiply by 2 (<<)

```python
def multiply_by_two(n: int) -> int:
    return n << 1
```

## Divide by 2 (>>)

```python
def divide_by_two(n: int) -> int:
    return n >> 1
```

## Swap Without Temp (XOR)

```python
def swap(a: int, b: int) -> tuple[int, int]:
    a ^= b
    b ^= a
    a ^= b
    return a, b
```

## Absolute Value Without Branching

```python
def abs_no_branch(n: int) -> int:
    mask = n >> 63
    return (n ^ mask) - mask
```

## Find Sign

```python
def find_sign(n: int) -> int:
    return (n >> 31) | (1 if n > 0 else 0) if n != 0 else 0
```

## Check Power of Two

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

## Check Power of Four

```python
def is_power_of_four(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0 and (n & 0xAAAAAAAA) == 0
```

## Next Power of Two

```python
def next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1
```

## Round Up to Nearest Power of Two

```python
def round_up_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()
```

## Average Without Overflow ((a&b)+((a^b)>>1))

```python
def average_no_overflow(a: int, b: int) -> int:
    return (a & b) + ((a ^ b) >> 1)
```

## Check Same Sign

```python
def same_sign(a: int, b: int) -> bool:
    return (a ^ b) >= 0
```

## Turn Off Rightmost Set Bit

```python
def turn_off_rightmost(n: int) -> int:
    return n & (n - 1)
```

## Rightmost Set Bit Position

```python
def rightmost_set_bit_pos(n: int) -> int:
    if n == 0:
        return -1
    return (n & -n).bit_length() - 1
```

## Uppercase to Lowercase (| 32)

```python
def to_lowercase(c: str) -> str:
    return chr(ord(c) | 32)
```

## Lowercase to Uppercase (& ~32)

```python
def to_uppercase(c: str) -> str:
    return chr(ord(c) & ~32)
```

## Toggle Case (^ 32)

```python
def toggle_case(c: str) -> str:
    return chr(ord(c) ^ 32)
```

## Add 1 Using Bits

```python
def add_one(n: int) -> int:
    return -~n
```

## Modulo with Power of 2 (n & (2^k - 1))

```python
def modulo_power_of_two(n: int, k: int) -> int:
    return n & ((1 << k) - 1)
```

## Check Binary Palindrome

```python
def is_binary_palindrome(n: int) -> bool:
    rev = 0
    temp = n
    while temp:
        rev = (rev << 1) | (temp & 1)
        temp >>= 1
    return rev == n
```

## XOR 1 to n (Pattern of 4)

```python
def xor_1_to_n(n: int) -> int:
    if n % 4 == 0:
        return n
    if n % 4 == 1:
        return 1
    if n % 4 == 2:
        return n + 1
    return 0
```

## Floor Log Base 2

```python
def floor_log2(n: int) -> int:
    if n <= 0:
        return -1
    return n.bit_length() - 1
```
