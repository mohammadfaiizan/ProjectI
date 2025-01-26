# Python File: String Methods and Examples

# =======================================
# 1. String Creation and Properties
# =======================================
# Convert value to string
print(str(123))  # '123'

# Get the length of a string
print(len("hello"))  # 5

# =======================================
# 2. Accessing and Slicing
# =======================================
s = "hello"

# Access specific characters
print(s[0])  # 'h'
print(s[-1])  # 'o'

# Extract substrings
print(s[1:4])  # 'ell'
print(s[::-1])  # Reverse string: 'olleh'

# =======================================
# 3. Searching and Checking
# =======================================
# Check if a substring exists
print("he" in "hello")  # True

# Find first occurrence of a substring
print("hello".find("l"))  # 2

# Find index of a substring (raises exception if not found)
print("hello".index("l"))  # 2

# Find the last occurrence of a substring
print("hello".rfind("l"))  # 3

# Check prefix/suffix
print("hello".startswith("he"))  # True
print("hello".endswith("lo"))  # True

# =======================================
# 4. Modifying and Replacing
# =======================================
# Replace all occurrences of a substring
print("hello world".replace("world", "Python"))  # 'hello Python'

# Join an iterable with a string separator
print(", ".join(["a", "b", "c"]))  # 'a, b, c'

# Split a string into a list by delimiter
print("a,b,c".split(","))  # ['a', 'b', 'c']

# Split a string into three parts
print("a=b=c".partition("="))  # ('a', '=', 'b=c')

# Remove leading and trailing spaces
print("  hello  ".strip())  # 'hello'

# Remove spaces from left/right
print("  hello".lstrip())  # 'hello'
print("hello  ".rstrip())  # 'hello'

# =======================================
# 5. Case Transformations
# =======================================
# Convert to lowercase/uppercase
print("Hello".lower())  # 'hello'
print("Hello".upper())  # 'HELLO'

# Capitalize first letter of each word
print("hello world".title())  # 'Hello World'

# Capitalize first letter of the string
print("hello".capitalize())  # 'Hello'

# Swap the case of all letters
print("Hello".swapcase())  # 'hELLO'

# =======================================
# 6. Alignment
# =======================================
# Center the string
print("hello".center(10, "-"))  # '--hello---'

# Left-align or right-align the string
print("hello".ljust(10, "-"))  # 'hello-----'
print("hello".rjust(10, "-"))  # '-----hello'

# Pad a string with zeros
print("42".zfill(5))  # '00042'

# =======================================
# 7. Counting and Statistics
# =======================================
# Count occurrences of a substring
print("hello".count("l"))  # 2

# Check if all characters are alphabetic
print("abc".isalpha())  # True

# Check if all characters are digits
print("123".isdigit())  # True

# Check if all characters are alphanumeric
print("abc123".isalnum())  # True

# Check if all characters are whitespace
print("   ".isspace())  # True

# =======================================
# 8. Encoding and Decoding
# =======================================
# Encode a string to bytes
print("hello".encode("utf-8"))  # b'hello'

# Decode bytes back to string
print(b'hello'.decode("utf-8"))  # 'hello'

# =======================================
# 9. Special Use Cases
# =======================================
# Get Unicode code point or character
print(ord("a"))  # 97
print(chr(97))  # 'a'

# String formatting
print("{} is {}".format("Python", "fun"))  # 'Python is fun'
print(f"{5 + 5}")  # '10'

# Character mapping
table = str.maketrans("abc", "123")
print("apple".translate(table))  # '1pple'

# =======================================
# Examples for DSA Use Cases
# =======================================
# Palindrome Check
s = "racecar"
print(s == s[::-1])  # True

# Anagram Check
s1, s2 = "listen", "silent"
print(sorted(s1) == sorted(s2))  # True

# Frequency Count
from collections import Counter
freq = Counter("hello")
print(freq)  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})