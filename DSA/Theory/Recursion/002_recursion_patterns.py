"""
Recursion Patterns - Linear, Binary, and Tail Recursion
======================================================

Topics: Linear recursion, binary recursion, tail recursion, multiple recursion
Companies: Google, Amazon, Microsoft, Facebook, Apple
Difficulty: Medium
Time Complexity: Varies by pattern
Space Complexity: O(depth) to O(2^depth)
"""

from typing import List, Optional, Tuple, Any
import time

class RecursionPatterns:
    
    def __init__(self):
        """Initialize with performance tracking"""
        self.call_count = 0
        self.max_depth = 0
        self.current_depth = 0
    
    # ==========================================
    # 1. LINEAR RECURSION (Single recursive call)
    # ==========================================
    
    def linear_recursion_examples(self) -> None:
        """
        Linear Recursion: Function makes at most one recursive call
        
        Characteristics:
        - Easy to understand and implement
        - Can often be converted to iteration
        - Space complexity: O(depth)
        - Time complexity: usually O(n)
        """
        print("=== LINEAR RECURSION PATTERNS ===")
        
        # Example 1: Sum of array elements
        def array_sum(arr: List[int], index: int = 0) -> int:
            """Sum array elements using linear recursion"""
            if index >= len(arr):  # Base case
                return 0
            return arr[index] + array_sum(arr, index + 1)  # One recursive call
        
        # Example 2: Reverse a string
        def string_reverse(s: str) -> str:
            """Reverse string using linear recursion"""
            if len(s) <= 1:  # Base case
                return s
            return s[-1] + string_reverse(s[:-1])  # One recursive call
        
        # Example 3: Count down
        def countdown(n: int) -> None:
            """Count down from n to 0"""
            if n <= 0:  # Base case
                print("Blast off!")
                return
            print(n)
            countdown(n - 1)  # One recursive call
        
        # Example 4: Calculate factorial
        def factorial(n: int) -> int:
            """Calculate factorial using linear recursion"""
            if n <= 1:  # Base case
                return 1
            return n * factorial(n - 1)  # One recursive call
        
        # Test examples
        test_array = [1, 2, 3, 4, 5]
        print(f"Array sum of {test_array}: {array_sum(test_array)}")
        
        test_string = "recursion"
        print(f"Reverse of '{test_string}': '{string_reverse(test_string)}'")
        
        print("Countdown from 3:")
        countdown(3)
        
        print(f"Factorial of 5: {factorial(5)}")
    
    def linear_search_recursive(self, arr: List[int], target: int, index: int = 0) -> int:
        """
        Linear search using recursion
        
        Time: O(n), Space: O(n)
        """
        # Base case: not found
        if index >= len(arr):
            return -1
        
        # Base case: found
        if arr[index] == target:
            return index
        
        # Recursive case: search in rest of array
        return self.linear_search_recursive(arr, target, index + 1)
    
    def print_array_recursive(self, arr: List[int], index: int = 0) -> None:
        """Print array elements using linear recursion"""
        if index >= len(arr):  # Base case
            return
        
        print(arr[index], end=" ")
        self.print_array_recursive(arr, index + 1)  # One recursive call
    
    # ==========================================
    # 2. BINARY RECURSION (Two recursive calls)
    # ==========================================
    
    def binary_recursion_examples(self) -> None:
        """
        Binary Recursion: Function makes two recursive calls
        
        Characteristics:
        - More complex than linear recursion
        - Often leads to exponential time complexity
        - Creates recursion tree structure
        - Common in divide-and-conquer algorithms
        """
        print("\n=== BINARY RECURSION PATTERNS ===")
        
        # Example 1: Fibonacci sequence
        def fibonacci(n: int) -> int:
            """Calculate Fibonacci using binary recursion"""
            if n <= 1:  # Base cases
                return n
            # Two recursive calls
            return fibonacci(n - 1) + fibonacci(n - 2)
        
        # Example 2: Count binary strings without consecutive 1s
        def count_binary_strings(n: int) -> int:
            """Count n-bit binary strings without consecutive 1s"""
            if n <= 0:
                return 1
            if n == 1:
                return 2
            
            # Either end with 0 or end with 1 (but previous must be 0)
            return count_binary_strings(n - 1) + count_binary_strings(n - 2)
        
        # Example 3: Tower of Hanoi moves
        def hanoi_moves(n: int) -> int:
            """Calculate minimum moves for Tower of Hanoi"""
            if n <= 0:  # Base case
                return 0
            # Move n-1 disks, move largest, move n-1 disks again
            return 2 * hanoi_moves(n - 1) + 1
        
        # Example 4: Check if number is power of 2
        def is_power_of_two(n: int) -> bool:
            """Check if number is power of 2 using recursion"""
            if n <= 0:
                return False
            if n == 1:  # 2^0 = 1
                return True
            if n % 2 != 0:  # Odd numbers can't be powers of 2
                return False
            return is_power_of_two(n // 2)
        
        # Test examples (use small values due to exponential complexity)
        print(f"Fibonacci(8): {fibonacci(8)}")
        print(f"Binary strings of length 5: {count_binary_strings(5)}")
        print(f"Hanoi moves for 4 disks: {hanoi_moves(4)}")
        print(f"Is 16 power of 2: {is_power_of_two(16)}")
        print(f"Is 15 power of 2: {is_power_of_two(15)}")
        
        print("⚠️  Warning: Binary recursion often has exponential time complexity!")
    
    def fibonacci_with_trace(self, n: int, depth: int = 0) -> int:
        """
        Fibonacci with call tracing to visualize binary recursion tree
        """
        indent = "  " * depth
        print(f"{indent}fib({n})")
        
        if n <= 1:
            print(f"{indent}→ {n}")
            return n
        
        left = self.fibonacci_with_trace(n - 1, depth + 1)
        right = self.fibonacci_with_trace(n - 2, depth + 1)
        result = left + right
        print(f"{indent}→ {result}")
        return result
    
    # ==========================================
    # 3. TAIL RECURSION (Optimizable)
    # ==========================================
    
    def tail_recursion_examples(self) -> None:
        """
        Tail Recursion: Recursive call is the last operation
        
        Characteristics:
        - Can be optimized to iteration by compiler
        - More memory efficient than regular recursion
        - Python doesn't optimize tail recursion automatically
        - Uses accumulator pattern
        """
        print("\n=== TAIL RECURSION PATTERNS ===")
        
        # Example 1: Tail recursive factorial
        def factorial_tail(n: int, accumulator: int = 1) -> int:
            """Factorial using tail recursion with accumulator"""
            if n <= 1:  # Base case
                return accumulator
            # Tail recursive call (last operation)
            return factorial_tail(n - 1, n * accumulator)
        
        # Example 2: Tail recursive Fibonacci
        def fibonacci_tail(n: int, a: int = 0, b: int = 1) -> int:
            """Fibonacci using tail recursion"""
            if n == 0:
                return a
            # Tail recursive call
            return fibonacci_tail(n - 1, b, a + b)
        
        # Example 3: Tail recursive sum
        def sum_tail(arr: List[int], index: int = 0, accumulator: int = 0) -> int:
            """Array sum using tail recursion"""
            if index >= len(arr):
                return accumulator
            # Tail recursive call
            return sum_tail(arr, index + 1, accumulator + arr[index])
        
        # Example 4: Tail recursive reverse
        def reverse_tail(s: str, accumulator: str = "") -> str:
            """String reverse using tail recursion"""
            if not s:
                return accumulator
            # Tail recursive call
            return reverse_tail(s[1:], s[0] + accumulator)
        
        # Test examples
        print(f"Tail factorial(5): {factorial_tail(5)}")
        print(f"Tail fibonacci(10): {fibonacci_tail(10)}")
        print(f"Tail sum([1,2,3,4,5]): {sum_tail([1, 2, 3, 4, 5])}")
        print(f"Tail reverse('hello'): '{reverse_tail('hello')}'")
        
        print("✅ Tail recursion can be easily converted to iteration")
    
    def convert_tail_to_iteration(self) -> None:
        """
        Demonstrate how tail recursion converts to iteration
        """
        print("\n=== TAIL RECURSION → ITERATION CONVERSION ===")
        
        # Tail recursive version
        def factorial_tail_recursive(n: int, acc: int = 1) -> int:
            if n <= 1:
                return acc
            return factorial_tail_recursive(n - 1, n * acc)
        
        # Equivalent iterative version
        def factorial_iterative(n: int) -> int:
            acc = 1
            while n > 1:
                acc = n * acc
                n = n - 1
            return acc
        
        # Test both
        test_n = 5
        tail_result = factorial_tail_recursive(test_n)
        iter_result = factorial_iterative(test_n)
        
        print(f"Factorial({test_n}):")
        print(f"  Tail recursive: {tail_result}")
        print(f"  Iterative: {iter_result}")
        print(f"  Results match: {tail_result == iter_result}")
        
        print("\nConversion pattern:")
        print("1. Parameters become local variables")
        print("2. Base case becomes loop termination condition")
        print("3. Recursive call becomes variable updates + continue")
    
    # ==========================================
    # 4. MULTIPLE RECURSION (More than 2 calls)
    # ==========================================
    
    def multiple_recursion_examples(self) -> None:
        """
        Multiple Recursion: Function makes more than two recursive calls
        
        Examples: Tree traversals, combinatorial problems, game algorithms
        """
        print("\n=== MULTIPLE RECURSION PATTERNS ===")
        
        # Example 1: Sum of digits in all numbers from 1 to n
        def sum_all_digits(n: int) -> int:
            """Sum of digits in all numbers from 1 to n"""
            if n <= 0:
                return 0
            
            def digit_sum(num):
                if num < 10:
                    return num
                return (num % 10) + digit_sum(num // 10)
            
            return digit_sum(n) + sum_all_digits(n - 1)
        
        # Example 2: Generate all subsets (2^n recursive calls)
        def generate_subsets(arr: List[int], index: int = 0, current: List[int] = None) -> List[List[int]]:
            """Generate all subsets using multiple recursion"""
            if current is None:
                current = []
            
            if index >= len(arr):
                return [current[:]]  # Return copy of current subset
            
            # Two choices for each element: include or exclude
            result = []
            
            # Exclude current element
            result.extend(generate_subsets(arr, index + 1, current))
            
            # Include current element
            current.append(arr[index])
            result.extend(generate_subsets(arr, index + 1, current))
            current.pop()  # Backtrack
            
            return result
        
        # Example 3: Tribonacci (three recursive calls)
        def tribonacci(n: int) -> int:
            """Tribonacci: sum of previous three terms"""
            if n <= 0:
                return 0
            if n <= 2:
                return 1
            
            # Three recursive calls
            return tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3)
        
        # Test examples
        print(f"Sum of digits 1 to 5: {sum_all_digits(5)}")
        
        subsets = generate_subsets([1, 2, 3])
        print(f"Subsets of [1,2,3]: {subsets}")
        
        print(f"Tribonacci(6): {tribonacci(6)}")
        
        print("⚠️  Multiple recursion can lead to very high time complexity!")
    
    # ==========================================
    # 5. PERFORMANCE COMPARISON
    # ==========================================
    
    def compare_recursion_patterns(self, n: int = 20) -> None:
        """
        Compare performance of different recursion patterns
        """
        print(f"\n=== RECURSION PATTERNS PERFORMANCE (n={n}) ===")
        
        # Linear recursion - Factorial
        def factorial_linear(num):
            if num <= 1:
                return 1
            return num * factorial_linear(num - 1)
        
        # Binary recursion - Fibonacci (naive)
        def fibonacci_binary(num):
            if num <= 1:
                return num
            return fibonacci_binary(num - 1) + fibonacci_binary(num - 2)
        
        # Tail recursion - Factorial
        def factorial_tail(num, acc=1):
            if num <= 1:
                return acc
            return factorial_tail(num - 1, num * acc)
        
        # Time measurements
        start_time = time.time()
        linear_result = factorial_linear(n)
        linear_time = time.time() - start_time
        
        # For Fibonacci, use smaller n to avoid timeout
        fib_n = min(n, 30)
        start_time = time.time()
        binary_result = fibonacci_binary(fib_n)
        binary_time = time.time() - start_time
        
        start_time = time.time()
        tail_result = factorial_tail(n)
        tail_time = time.time() - start_time
        
        print(f"Linear recursion (factorial({n})): {linear_result}")
        print(f"  Time: {linear_time:.6f} seconds")
        
        print(f"Binary recursion (fibonacci({fib_n})): {binary_result}")
        print(f"  Time: {binary_time:.6f} seconds")
        
        print(f"Tail recursion (factorial({n})): {tail_result}")
        print(f"  Time: {tail_time:.6f} seconds")
        
        print(f"\nComparison:")
        if binary_time > 0:
            print(f"  Linear is {binary_time/linear_time:.1f}x faster than binary")
        print(f"  Tail is similar to linear (both O(n))")
    
    # ==========================================
    # 6. PATTERN IDENTIFICATION
    # ==========================================
    
    def identify_recursion_pattern(self, problem_description: str) -> str:
        """
        Help identify which recursion pattern to use for a problem
        """
        problem = problem_description.lower()
        
        # Linear recursion indicators
        if any(word in problem for word in ['sum', 'count', 'search', 'traverse', 'factorial']):
            return "Linear Recursion - Single recursive call, O(n) complexity"
        
        # Binary recursion indicators
        if any(word in problem for word in ['fibonacci', 'tree traversal', 'divide and conquer']):
            return "Binary Recursion - Two recursive calls, often O(2^n) complexity"
        
        # Tail recursion indicators
        if any(word in problem for word in ['accumulator', 'iterative equivalent', 'optimization']):
            return "Tail Recursion - Optimizable to iteration, O(n) space can become O(1)"
        
        # Multiple recursion indicators
        if any(word in problem for word in ['all combinations', 'all permutations', 'backtracking']):
            return "Multiple Recursion - Many recursive calls, exponential complexity"
        
        return "Pattern unclear - analyze the problem structure"

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_recursion_patterns():
    """Demonstrate all recursion patterns"""
    print("=== RECURSION PATTERNS DEMONSTRATION ===\n")
    
    patterns = RecursionPatterns()
    
    # 1. Linear recursion
    patterns.linear_recursion_examples()
    
    # 2. Binary recursion
    patterns.binary_recursion_examples()
    
    # 3. Fibonacci tree visualization
    print("\n=== FIBONACCI RECURSION TREE (n=4) ===")
    patterns.fibonacci_with_trace(4)
    
    # 4. Tail recursion
    patterns.tail_recursion_examples()
    
    # 5. Tail to iteration conversion
    patterns.convert_tail_to_iteration()
    
    # 6. Multiple recursion
    patterns.multiple_recursion_examples()
    
    # 7. Performance comparison
    patterns.compare_recursion_patterns(15)
    
    # 8. Pattern identification examples
    print("\n=== PATTERN IDENTIFICATION ===")
    test_problems = [
        "Calculate sum of array elements",
        "Generate all subsets of an array",
        "Find nth Fibonacci number",
        "Implement factorial with accumulator"
    ]
    
    for problem in test_problems:
        pattern = patterns.identify_recursion_pattern(problem)
        print(f"Problem: {problem}")
        print(f"  Suggested: {pattern}\n")

if __name__ == "__main__":
    demonstrate_recursion_patterns()
    
    print("=== PATTERN SELECTION GUIDE ===")
    print("📌 Linear Recursion:")
    print("   - Use for: Sequential processing, simple transformations")
    print("   - Examples: Sum, factorial, linear search")
    print("   - Complexity: O(n) time, O(n) space")
    
    print("\n📌 Binary Recursion:")
    print("   - Use for: Problems with two choices, divide-and-conquer")
    print("   - Examples: Fibonacci, tree problems")
    print("   - Complexity: Often O(2^n) time, O(n) space")
    print("   - ⚠️  Consider memoization for optimization")
    
    print("\n📌 Tail Recursion:")
    print("   - Use for: When you can accumulate results")
    print("   - Examples: Factorial with accumulator")
    print("   - Complexity: O(n) time, can be optimized to O(1) space")
    print("   - ✅ Can be converted to iteration")
    
    print("\n📌 Multiple Recursion:")
    print("   - Use for: Combinatorial problems, backtracking")
    print("   - Examples: Generate all permutations/combinations")
    print("   - Complexity: Very high, often exponential")
    print("   - ⚠️  Use pruning and optimization techniques")
