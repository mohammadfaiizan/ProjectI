"""
Number Theory String Problems - Multiple Approaches
Difficulty: Hard

Number theory problems involving strings and trie structures
for competitive programming.

Problems:
1. String Modular Arithmetic
2. Prime Pattern Matching
3. GCD of String Values
4. Fibonacci String Sequences
5. Digit DP with Trie
6. Modular String Operations
"""

from typing import List, Dict, Tuple, Optional
import math

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = 0
        self.count = 0

class NumberTheoryStrings:
    
    def __init__(self):
        self.root = TrieNode()
        self.MOD = 10**9 + 7
    
    def string_modular_arithmetic(self, strings: List[str], base: int, mod: int) -> Dict[str, int]:
        """
        Convert strings to numbers in given base and perform modular arithmetic
        Time: O(n * m) where n=strings, m=avg_length
        Space: O(n)
        """
        results = {}
        
        for s in strings:
            value = 0
            for char in s:
                if char.isdigit():
                    digit = int(char)
                else:
                    digit = ord(char.lower()) - ord('a') + 10
                
                if digit >= base:
                    value = -1  # Invalid
                    break
                
                value = (value * base + digit) % mod
            
            results[s] = value
        
        return results
    
    def prime_pattern_matching(self, text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Find patterns where string hash values are prime
        Time: O(|text| + sum(|patterns|))
        Space: O(sum(|patterns|))
        """
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            
            for i in range(3, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        def string_hash(s: str) -> int:
            hash_val = 0
            for char in s:
                hash_val = hash_val * 31 + ord(char)
            return hash_val
        
        # Filter patterns with prime hashes
        prime_patterns = []
        for pattern in patterns:
            hash_val = string_hash(pattern)
            if is_prime(hash_val):
                prime_patterns.append(pattern)
        
        # Search for prime patterns
        results = {}
        for pattern in prime_patterns:
            positions = []
            pattern_len = len(pattern)
            
            for i in range(len(text) - pattern_len + 1):
                if text[i:i + pattern_len] == pattern:
                    positions.append(i)
            
            results[pattern] = positions
        
        return results
    
    def gcd_string_values(self, strings: List[str]) -> int:
        """
        Calculate GCD of string values (treating strings as numbers)
        Time: O(n * m + log(max_value))
        Space: O(1)
        """
        def string_to_number(s: str) -> int:
            result = 0
            for char in s:
                if char.isdigit():
                    result = result * 10 + int(char)
                else:
                    # Use ASCII value for non-digits
                    result = result * 256 + ord(char)
            return result
        
        if not strings:
            return 0
        
        gcd_result = string_to_number(strings[0])
        
        for i in range(1, len(strings)):
            value = string_to_number(strings[i])
            gcd_result = math.gcd(gcd_result, value)
            
            if gcd_result == 1:
                break  # Optimization: GCD can't get smaller than 1
        
        return gcd_result
    
    def fibonacci_string_sequences(self, n: int) -> List[str]:
        """
        Generate Fibonacci-like string sequences
        Time: O(n * fib(n))
        Space: O(fib(n))
        """
        if n <= 0:
            return []
        if n == 1:
            return ["A"]
        if n == 2:
            return ["A", "B"]
        
        # Generate Fibonacci strings: F(n) = F(n-1) + F(n-2)
        fib_strings = ["A", "B"]
        
        for i in range(2, n):
            next_string = fib_strings[i-1] + fib_strings[i-2]
            fib_strings.append(next_string)
        
        return fib_strings
    
    def digit_dp_with_trie(self, upper_bound: str, forbidden_patterns: List[str]) -> int:
        """
        Count numbers up to upper_bound without forbidden digit patterns
        Time: O(|upper_bound| * states * 10)
        Space: O(states)
        """
        # Build trie of forbidden patterns
        forbidden_trie = TrieNode()
        
        for pattern in forbidden_patterns:
            node = forbidden_trie
            for digit in pattern:
                if digit not in node.children:
                    node.children[digit] = TrieNode()
                node = node.children[digit]
            node.is_end = True
        
        n = len(upper_bound)
        memo = {}
        
        def dp(pos: int, tight: bool, started: bool, trie_node: TrieNode) -> int:
            """
            pos: current position in number
            tight: whether we're still bounded by upper_bound
            started: whether we've placed a non-zero digit
            trie_node: current position in forbidden pattern trie
            """
            if trie_node.is_end:
                return 0  # Found forbidden pattern
            
            if pos == n:
                return 1 if started else 0
            
            state = (pos, tight, started, id(trie_node))
            if state in memo:
                return memo[state]
            
            limit = int(upper_bound[pos]) if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                # Navigate trie
                digit_str = str(digit)
                if digit_str in trie_node.children:
                    new_trie_node = trie_node.children[digit_str]
                else:
                    new_trie_node = forbidden_trie  # Reset to root
                
                if not new_trie_node.is_end:  # Don't continue if forbidden pattern found
                    result = (result + dp(pos + 1, new_tight, new_started, new_trie_node)) % self.MOD
            
            memo[state] = result
            return result
        
        return dp(0, True, False, forbidden_trie)
    
    def modular_string_operations(self, operations: List[Tuple[str, str, str]]) -> List[int]:
        """
        Perform modular operations on strings treated as numbers
        Time: O(n * m) where n=operations, m=avg_string_length
        Space: O(1)
        """
        def string_to_mod(s: str, mod: int) -> int:
            result = 0
            for char in s:
                digit = ord(char) - ord('0') if char.isdigit() else ord(char) - ord('a') + 10
                result = (result * 36 + digit) % mod  # Base 36
            return result
        
        results = []
        
        for op, a, b in operations:
            mod_a = string_to_mod(a, self.MOD)
            mod_b = string_to_mod(b, self.MOD)
            
            if op == "add":
                result = (mod_a + mod_b) % self.MOD
            elif op == "mul":
                result = (mod_a * mod_b) % self.MOD
            elif op == "sub":
                result = (mod_a - mod_b + self.MOD) % self.MOD
            elif op == "pow":
                result = pow(mod_a, mod_b, self.MOD)
            else:
                result = 0
            
            results.append(result)
        
        return results


def test_string_modular_arithmetic():
    """Test string modular arithmetic"""
    print("=== Testing String Modular Arithmetic ===")
    
    solver = NumberTheoryStrings()
    
    strings = ["123", "abc", "456", "def"]
    base = 16
    mod = 1000
    
    print(f"Strings: {strings}")
    print(f"Base: {base}, Mod: {mod}")
    
    results = solver.string_modular_arithmetic(strings, base, mod)
    
    for string, value in results.items():
        print(f"'{string}' -> {value}")

def test_prime_pattern_matching():
    """Test prime pattern matching"""
    print("\n=== Testing Prime Pattern Matching ===")
    
    solver = NumberTheoryStrings()
    
    text = "abcdefghijk"
    patterns = ["abc", "def", "ghi", "xyz"]
    
    print(f"Text: '{text}'")
    print(f"Patterns: {patterns}")
    
    matches = solver.prime_pattern_matching(text, patterns)
    
    print("Prime patterns found:")
    for pattern, positions in matches.items():
        print(f"'{pattern}': {positions}")

def test_fibonacci_strings():
    """Test Fibonacci string sequences"""
    print("\n=== Testing Fibonacci String Sequences ===")
    
    solver = NumberTheoryStrings()
    
    n = 6
    print(f"Generating first {n} Fibonacci strings:")
    
    fib_strings = solver.fibonacci_string_sequences(n)
    
    for i, s in enumerate(fib_strings):
        print(f"F({i+1}): '{s}' (length: {len(s)})")

def test_digit_dp():
    """Test digit DP with trie"""
    print("\n=== Testing Digit DP with Trie ===")
    
    solver = NumberTheoryStrings()
    
    upper_bound = "1000"
    forbidden_patterns = ["13", "666"]
    
    print(f"Upper bound: {upper_bound}")
    print(f"Forbidden patterns: {forbidden_patterns}")
    
    count = solver.digit_dp_with_trie(upper_bound, forbidden_patterns)
    print(f"Count of valid numbers: {count}")

def test_modular_operations():
    """Test modular string operations"""
    print("\n=== Testing Modular String Operations ===")
    
    solver = NumberTheoryStrings()
    
    operations = [
        ("add", "123", "456"),
        ("mul", "abc", "def"),
        ("sub", "999", "111"),
        ("pow", "10", "3")
    ]
    
    print("Operations:")
    for op, a, b in operations:
        print(f"  {op}('{a}', '{b}')")
    
    results = solver.modular_string_operations(operations)
    
    print("\nResults:")
    for i, result in enumerate(results):
        op, a, b = operations[i]
        print(f"  {op}('{a}', '{b}') = {result}")

if __name__ == "__main__":
    test_string_modular_arithmetic()
    test_prime_pattern_matching()
    test_fibonacci_strings()
    test_digit_dp()
    test_modular_operations()
