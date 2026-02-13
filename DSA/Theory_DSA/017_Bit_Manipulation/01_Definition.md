# Bit Manipulation - Definition and Fundamentals

## Binary Number System (Base 2)

The binary number system uses only two digits: 0 and 1. Each digit is called a bit. Unlike decimal (base 10) where each position represents a power of 10, in binary each position represents a power of 2.

Example: 1011 in binary = 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 8 + 0 + 2 + 1 = 11 in decimal.

## Bit

A bit is the smallest unit of data in computing. It can hold only one of two values: 0 or 1. Bits are the fundamental building blocks of all digital data.

## Byte

A byte consists of 8 bits. One byte can represent 256 different values (2^8 = 256), typically 0 to 255 for unsigned integers.

## Word Size (32/64 bits)

- **32-bit**: A word is 4 bytes (32 bits). Can represent integers from -2^31 to 2^31-1 for signed, or 0 to 2^32-1 for unsigned.
- **64-bit**: A word is 8 bytes (64 bits). Can represent integers from -2^63 to 2^63-1 for signed, or 0 to 2^64-1 for unsigned.

## Bitwise Operators

### AND (&)
Returns 1 only when both bits are 1. Otherwise returns 0.
```
1 & 1 = 1
1 & 0 = 0
0 & 1 = 0
0 & 0 = 0
```

### OR (|)
Returns 1 when at least one bit is 1.
```
1 | 1 = 1
1 | 0 = 1
0 | 1 = 1
0 | 0 = 0
```

### XOR (^)
Returns 1 when bits are different.
```
1 ^ 1 = 0
1 ^ 0 = 1
0 ^ 1 = 1
0 ^ 0 = 0
```

### NOT (~)
Flips all bits. In Python, this gives two's complement representation.
```
~5 = -6  (in 32-bit: 5 = 000...0101, ~5 = 111...1010 = -6)
```

### Left Shift (<<)
Shifts bits left by n positions. Equivalent to multiplying by 2^n.
```
5 << 1 = 10   (5 * 2 = 10)
5 << 2 = 20   (5 * 4 = 20)
```

### Right Shift (>>)
Shifts bits right by n positions. Equivalent to integer division by 2^n.
```
10 >> 1 = 5   (10 // 2 = 5)
10 >> 2 = 2   (10 // 4 = 2)
```

## Signed vs Unsigned

- **Unsigned**: All bits represent magnitude. Range for n bits: 0 to 2^n - 1.
- **Signed**: Most significant bit (MSB) indicates sign. 0 = positive, 1 = negative. Range for n bits: -2^(n-1) to 2^(n-1) - 1.

## Two's Complement

Two's complement is the standard way to represent negative integers in binary.

**How negatives are stored:**
1. For a positive number n, its negative -n is represented as (2^bits - n)
2. Alternatively: invert all bits and add 1
3. The MSB (leftmost bit) is the sign bit: 1 means negative, 0 means non-negative

Example for 8 bits:
- 5 = 00000101
- -5 = 11111011 (flip bits: 11111010, add 1: 11111011)

Key property: n + (-n) = 0 (with overflow discarded)

## Arithmetic vs Logical Shift

- **Arithmetic Right Shift**: Preserves sign. Fills left with sign bit. -8 >> 1 = -4.
- **Logical Right Shift**: Fills left with 0. Treats number as unsigned.
- **Python's >>**: Arithmetic shift for signed integers. Logical behavior depends on implementation.

## When to Use Bit Manipulation

1. **Performance**: Bit operations are faster than arithmetic (multiplication, division)
2. **Space efficiency**: Pack multiple boolean flags into a single integer
3. **Low-level programming**: Hardware interaction, embedded systems
4. **Competitive programming**: Many problems have elegant bit solutions
5. **Cryptography**: XOR for encryption, bit mixing
6. **Graphics**: Color manipulation, alpha blending

## Time Complexity

Each bitwise operation (AND, OR, XOR, NOT, shift) runs in O(1) time at the hardware level for fixed-size integers (32 or 64 bits). For arbitrary-precision integers (Python's int), complexity depends on the number of bits.
