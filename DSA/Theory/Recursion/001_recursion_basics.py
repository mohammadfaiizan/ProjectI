"""
Recursion Basics - Fundamental Concepts
=======================================

Topics: Definition, base cases, recursive cases, call stack, basic examples
Companies: All major tech companies test recursion fundamentals
Difficulty: Easy to Medium
Time Complexity: Varies by problem
Space Complexity: O(depth) for call stack
"""

from typing import List, Optional, Any
import sys

class RecursionBasics:
    
    def __init__(self):
        """Initialize with call tracking for analysis"""
        self.call_count = 0
        self.max_depth = 0
        self.current_depth = 0
    
    # ==========================================
    # 1. WHAT IS RECURSION?
    # ==========================================
    
    def demonstrate_recursion_concept(self) -> None:
        """
        Demonstrate the fundamental concept of recursion
        
        Recursion is a programming technique where a function calls itself
        to solve a problem by breaking it into smaller, similar subproblems.
        """
        print("=== What is Recursion? ===")
        print("Definition: A function that calls itself to solve a problem")
        print("Key Components:")
        print("1. Base Case - Condition that stops recursion")
        print("2. Recursive Case - Function calls itself with different input")
        print("3. Progress - Each call moves closer to base case")
        
        print("\nAnalogy: Nested Russian Dolls (Matryoshka)")
        print("- Each doll contains a smaller doll (recursive case)")
        print("- Smallest doll is empty (base case)")
        print("- To count dolls: 1 + count_dolls_inside()")
        
        # Simple example
        def countdown(n):
            print(f"Counting: {n}")
            if n <= 0:  # Base case
                print("Done!")
                return
            countdown(n - 1)  # Recursive case
        
        print("\nExample - Countdown from 3:")
        countdown(3)
    
    # ==========================================
    # 2. FACTORIAL - CLASSIC EXAMPLE
    # ==========================================
    
    def factorial_recursive(self, n: int) -> int:
        """
        Calculate factorial using recursion
        
        Definition: n! = n × (n-1) × (n-2) × ... × 1
        Base Case: 0! = 1, 1! = 1
        Recursive Case: n! = n × (n-1)!
        
        Time: O(n), Space: O(n)
        """
        self.call_count += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        print(f"{'  ' * (self.current_depth-1)}factorial({n}) called")
        
        # Base case
        if n <= 1:
            result = 1
            print(f"{'  ' * (self.current_depth-1)}Base case: factorial({n}) = {result}")
        else:
            # Recursive case
            result = n * self.factorial_recursive(n - 1)
            print(f"{'  ' * (self.current_depth-1)}factorial({n}) = {n} × factorial({n-1}) = {result}")
        
        self.current_depth -= 1
        return result
    
    def factorial_iterative(self, n: int) -> int:
        """
        Calculate factorial using iteration for comparison
        Time: O(n), Space: O(1)
        """
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    # ==========================================
    # 3. FIBONACCI - MULTIPLE RECURSIVE CALLS
    # ==========================================
    
    def fibonacci_naive(self, n: int) -> int:
        """
        Calculate Fibonacci number using naive recursion
        
        Definition: F(n) = F(n-1) + F(n-2)
        Base Cases: F(0) = 0, F(1) = 1
        
        Time: O(2^n), Space: O(n)
        """
        self.call_count += 1
        
        # Base cases
        if n <= 1:
            return n
        
        # Recursive case - two recursive calls
        return self.fibonacci_naive(n - 1) + self.fibonacci_naive(n - 2)
    
    def fibonacci_trace(self, n: int, depth: int = 0) -> int:
        """
        Fibonacci with call tracing to show recursion tree
        """
        indent = "  " * depth
        print(f"{indent}fib({n})")
        
        if n <= 1:
            print(f"{indent}→ {n}")
            return n
        
        left = self.fibonacci_trace(n - 1, depth + 1)
        right = self.fibonacci_trace(n - 2, depth + 1)
        result = left + right
        print(f"{indent}→ {result}")
        return result
    
    # ==========================================
    # 4. POWER FUNCTION
    # ==========================================
    
    def power_recursive(self, base: float, exponent: int) -> float:
        """
        Calculate base^exponent using recursion
        
        Base Case: base^0 = 1
        Recursive Case: base^n = base × base^(n-1)
        
        Time: O(n), Space: O(n)
        """
        self.call_count += 1
        
        # Base case
        if exponent == 0:
            return 1
        
        # Handle negative exponents
        if exponent < 0:
            return 1 / self.power_recursive(base, -exponent)
        
        # Recursive case
        return base * self.power_recursive(base, exponent - 1)
    
    def power_optimized(self, base: float, exponent: int) -> float:
        """
        Optimized power using divide and conquer
        
        Idea: base^n = (base^(n/2))^2
        
        Time: O(log n), Space: O(log n)
        """
        self.call_count += 1
        
        # Base case
        if exponent == 0:
            return 1
        
        # Handle negative exponents
        if exponent < 0:
            return 1 / self.power_optimized(base, -exponent)
        
        # Divide and conquer
        half_power = self.power_optimized(base, exponent // 2)
        
        if exponent % 2 == 0:
            return half_power * half_power
        else:
            return base * half_power * half_power
    
    # ==========================================
    # 5. BASIC ARRAY OPERATIONS
    # ==========================================
    
    def array_sum(self, arr: List[int], index: int = 0) -> int:
        """
        Calculate sum of array using recursion
        
        Base Case: index >= len(arr), return 0
        Recursive Case: arr[index] + sum(rest of array)
        
        Time: O(n), Space: O(n)
        """
        # Base case
        if index >= len(arr):
            return 0
        
        # Recursive case
        return arr[index] + self.array_sum(arr, index + 1)
    
    def array_max(self, arr: List[int], index: int = 0) -> int:
        """
        Find maximum element using recursion
        """
        # Base case
        if index == len(arr) - 1:
            return arr[index]
        
        # Recursive case
        rest_max = self.array_max(arr, index + 1)
        return max(arr[index], rest_max)
    
    def count_elements(self, arr: List[int], target: int, index: int = 0) -> int:
        """
        Count occurrences of target element using recursion
        """
        # Base case
        if index >= len(arr):
            return 0
        
        # Recursive case
        count = 1 if arr[index] == target else 0
        return count + self.count_elements(arr, target, index + 1)
    
    # ==========================================
    # 6. STRING OPERATIONS
    # ==========================================
    
    def string_reverse(self, s: str) -> str:
        """
        Reverse string using recursion
        
        Base Case: empty or single character
        Recursive Case: last_char + reverse(rest)
        
        Time: O(n), Space: O(n)
        """
        # Base case
        if len(s) <= 1:
            return s
        
        # Recursive case
        return s[-1] + self.string_reverse(s[:-1])
    
    def is_palindrome(self, s: str, start: int = 0, end: int = None) -> bool:
        """
        Check if string is palindrome using recursion
        """
        if end is None:
            end = len(s) - 1
        
        # Base case
        if start >= end:
            return True
        
        # Check current characters and recurse
        if s[start] != s[end]:
            return False
        
        return self.is_palindrome(s, start + 1, end - 1)
    
    def count_vowels(self, s: str, index: int = 0) -> int:
        """
        Count vowels in string using recursion
        """
        # Base case
        if index >= len(s):
            return 0
        
        # Check if current character is vowel
        vowels = "aeiouAEIOU"
        count = 1 if s[index] in vowels else 0
        
        # Recursive case
        return count + self.count_vowels(s, index + 1)
    
    # ==========================================
    # 7. MATHEMATICAL FUNCTIONS
    # ==========================================
    
    def gcd_recursive(self, a: int, b: int) -> int:
        """
        Calculate GCD using Euclidean algorithm with recursion
        
        Base Case: b = 0, return a
        Recursive Case: gcd(b, a % b)
        
        Time: O(log min(a,b)), Space: O(log min(a,b))
        """
        # Base case
        if b == 0:
            return a
        
        # Recursive case
        return self.gcd_recursive(b, a % b)
    
    def count_digits(self, n: int) -> int:
        """
        Count digits in number using recursion
        """
        # Base case
        if n < 10:
            return 1
        
        # Recursive case
        return 1 + self.count_digits(n // 10)
    
    def sum_of_digits(self, n: int) -> int:
        """
        Calculate sum of digits using recursion
        """
        # Base case
        if n < 10:
            return n
        
        # Recursive case
        return (n % 10) + self.sum_of_digits(n // 10)
    
    # ==========================================
    # 8. PERFORMANCE ANALYSIS
    # ==========================================
    
    def analyze_performance(self, n: int = 5) -> None:
        """
        Analyze performance of different recursive approaches
        """
        print(f"=== Performance Analysis (n={n}) ===")
        
        # Reset counters
        self.call_count = 0
        self.max_depth = 0
        self.current_depth = 0
        
        # Test factorial
        print(f"\n1. Factorial({n}):")
        factorial_result = self.factorial_recursive(n)
        print(f"   Result: {factorial_result}")
        print(f"   Calls: {self.call_count}")
        print(f"   Max depth: {self.max_depth}")
        
        # Reset and test power
        self.call_count = 0
        print(f"\n2. Power(2, {n}):")
        
        # Basic power
        basic_calls = 0
        self.call_count = 0
        power_basic = self.power_recursive(2, n)
        basic_calls = self.call_count
        
        # Optimized power
        self.call_count = 0
        power_opt = self.power_optimized(2, n)
        opt_calls = self.call_count
        
        print(f"   Basic: {power_basic} ({basic_calls} calls)")
        print(f"   Optimized: {power_opt} ({opt_calls} calls)")
        print(f"   Improvement: {basic_calls/opt_calls:.1f}x fewer calls")
        
        # Test with array
        test_array = list(range(1, n+1))
        print(f"\n3. Array Operations on {test_array}:")
        print(f"   Sum: {self.array_sum(test_array)}")
        print(f"   Max: {self.array_max(test_array)}")
        print(f"   Count 3's: {self.count_elements(test_array, 3)}")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_recursion_basics():
    """Demonstrate all basic recursion concepts"""
    print("=== RECURSION BASICS DEMONSTRATION ===\n")
    
    recursion = RecursionBasics()
    
    # 1. Concept explanation
    recursion.demonstrate_recursion_concept()
    print("\n" + "="*50 + "\n")
    
    # 2. Factorial with tracing
    print("=== FACTORIAL WITH CALL TRACING ===")
    recursion.call_count = 0
    recursion.max_depth = 0
    recursion.current_depth = 0
    
    result = recursion.factorial_recursive(4)
    print(f"\nFinal result: {result}")
    print(f"Total calls: {recursion.call_count}")
    print("\n" + "="*50 + "\n")
    
    # 3. Fibonacci tree visualization
    print("=== FIBONACCI RECURSION TREE ===")
    print("fib(4) call tree:")
    recursion.fibonacci_trace(4)
    print("\n" + "="*50 + "\n")
    
    # 4. Power comparison
    print("=== POWER FUNCTION COMPARISON ===")
    recursion.call_count = 0
    basic_power = recursion.power_recursive(2, 5)
    basic_calls = recursion.call_count
    
    recursion.call_count = 0
    opt_power = recursion.power_optimized(2, 5)
    opt_calls = recursion.call_count
    
    print(f"Power(2, 5):")
    print(f"Basic recursion: {basic_power} ({basic_calls} calls)")
    print(f"Optimized: {opt_power} ({opt_calls} calls)")
    print("\n" + "="*50 + "\n")
    
    # 5. Array and string operations
    print("=== ARRAY AND STRING OPERATIONS ===")
    test_array = [1, 2, 3, 4, 5]
    print(f"Array: {test_array}")
    print(f"Sum: {recursion.array_sum(test_array)}")
    print(f"Max: {recursion.array_max(test_array)}")
    
    test_string = "racecar"
    print(f"\nString: '{test_string}'")
    print(f"Reversed: '{recursion.string_reverse(test_string)}'")
    print(f"Is palindrome: {recursion.is_palindrome(test_string)}")
    print(f"Vowel count: {recursion.count_vowels(test_string)}")
    print("\n" + "="*50 + "\n")
    
    # 6. Mathematical functions
    print("=== MATHEMATICAL FUNCTIONS ===")
    print(f"GCD(48, 18): {recursion.gcd_recursive(48, 18)}")
    print(f"Digits in 12345: {recursion.count_digits(12345)}")
    print(f"Sum of digits in 12345: {recursion.sum_of_digits(12345)}")
    
    # 7. Performance analysis
    recursion.analyze_performance(6)

if __name__ == "__main__":
    demonstrate_recursion_basics()
    
    print("\n=== KEY TAKEAWAYS ===")
    print("1. Always define clear base cases")
    print("2. Ensure recursive calls make progress toward base case")
    print("3. Consider space complexity due to call stack")
    print("4. Some problems have more efficient iterative solutions")
    print("5. Optimization techniques like memoization can help")
    print("6. Trace through small examples to understand behavior")
