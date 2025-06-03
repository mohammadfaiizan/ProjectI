"""
Dynamic Programming - Digit DP Patterns
This module implements various Digit DP problems including counting numbers
with unique digits, sum of digits in ranges, and constrained digit counting.
"""

from typing import List, Dict, Tuple, Optional, Set
import time
from functools import lru_cache

# ==================== BASIC DIGIT DP FRAMEWORK ====================

class DigitDP:
    """
    Base class for Digit DP problems
    
    Provides common utilities and framework for digit-based dynamic programming.
    """
    
    def __init__(self):
        self.memo = {}
    
    def clear_memo(self):
        """Clear memoization cache"""
        self.memo.clear()
    
    def get_digits(self, n: int) -> List[int]:
        """Convert number to list of digits"""
        if n == 0:
            return [0]
        
        digits = []
        while n > 0:
            digits.append(n % 10)
            n //= 10
        
        return digits[::-1]  # Reverse to get most significant digit first
    
    def digits_to_number(self, digits: List[int]) -> int:
        """Convert list of digits to number"""
        result = 0
        for digit in digits:
            result = result * 10 + digit
        return result

# ==================== UNIQUE DIGITS PROBLEMS ====================

class UniqueDigitsProblems(DigitDP):
    """
    Problems involving counting numbers with unique digits
    """
    
    def count_numbers_with_unique_digits(self, n: int) -> int:
        """
        Count numbers with unique digits from 0 to 10^n - 1
        
        LeetCode 357 - Count Numbers with Unique Digits
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            n: Number of digits (count from 0 to 10^n - 1)
        
        Returns:
            Count of numbers with all unique digits
        """
        if n == 0:
            return 1
        if n == 1:
            return 10
        
        # Mathematical approach
        result = 10  # All single digit numbers
        
        # For k-digit numbers (k >= 2)
        current_count = 9  # First digit can be 1-9 (not 0)
        
        for i in range(2, n + 1):
            current_count *= (11 - i)  # Remaining choices for next digit
            result += current_count
        
        return result
    
    def count_unique_digits_up_to_n(self, n: int) -> int:
        """
        Count numbers with unique digits from 0 to n (inclusive)
        
        Args:
            n: Upper bound (inclusive)
        
        Returns:
            Count of numbers with unique digits
        """
        if n < 0:
            return 0
        
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, mask: int, tight: bool, started: bool) -> int:
            """
            Digit DP function
            
            Args:
                pos: Current position in digits
                mask: Bitmask for used digits
                tight: Whether we're still bounded by n
                started: Whether we've started placing non-zero digits
            """
            if pos == length:
                return 1 if started else 0
            
            state = (pos, mask, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_mask = mask
                new_started = started
                new_tight = tight and (digit == limit)
                
                # Check if digit is already used
                if started and (mask & (1 << digit)):
                    continue
                
                if digit > 0 or started:
                    new_mask |= (1 << digit)
                    new_started = True
                
                result += dp(pos + 1, new_mask, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)
    
    def count_unique_digits_in_range(self, low: int, high: int) -> int:
        """
        Count numbers with unique digits in range [low, high]
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
        
        Returns:
            Count of numbers with unique digits in range
        """
        return (self.count_unique_digits_up_to_n(high) - 
                self.count_unique_digits_up_to_n(low - 1))
    
    def count_unique_digits_with_sum(self, max_digits: int, target_sum: int) -> int:
        """
        Count numbers with unique digits and specific digit sum
        
        Args:
            max_digits: Maximum number of digits
            target_sum: Target sum of digits
        
        Returns:
            Count of numbers with unique digits and target sum
        """
        def dp(pos: int, mask: int, current_sum: int, started: bool) -> int:
            if pos == max_digits:
                return 1 if started and current_sum == target_sum else 0
            
            if current_sum > target_sum:
                return 0
            
            state = (pos, mask, current_sum, started)
            if state in self.memo:
                return self.memo[state]
            
            result = 0
            
            for digit in range(0, 10):
                # Skip if digit already used
                if mask & (1 << digit):
                    continue
                
                new_mask = mask
                new_started = started
                
                if digit > 0 or started:
                    new_mask |= (1 << digit)
                    new_started = True
                
                if digit == 0 and not started:
                    # Leading zero
                    result += dp(pos + 1, new_mask, current_sum, new_started)
                else:
                    result += dp(pos + 1, new_mask, current_sum + digit, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, 0, False)

# ==================== DIGIT SUM PROBLEMS ====================

class DigitSumProblems(DigitDP):
    """
    Problems involving sum of digits in ranges
    """
    
    def sum_of_digits_up_to_n(self, n: int) -> int:
        """
        Calculate sum of all digits in numbers from 1 to n
        
        Args:
            n: Upper bound
        
        Returns:
            Sum of all digits
        """
        if n <= 0:
            return 0
        
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 0
            
            state = (pos, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                # Add current digit if we've started
                digit_contribution = digit if new_started else 0
                
                result += digit_contribution + dp(pos + 1, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, True, False)
    
    def sum_of_digits_in_range(self, low: int, high: int) -> int:
        """
        Sum of all digits in range [low, high]
        
        Args:
            low: Lower bound
            high: Upper bound
        
        Returns:
            Sum of all digits in range
        """
        return (self.sum_of_digits_up_to_n(high) - 
                self.sum_of_digits_up_to_n(low - 1))
    
    def count_numbers_with_digit_sum(self, n: int, target_sum: int) -> int:
        """
        Count numbers up to n with specific digit sum
        
        Args:
            n: Upper bound
            target_sum: Target sum of digits
        
        Returns:
            Count of numbers with target digit sum
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, current_sum: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started and current_sum == target_sum else 0
            
            if current_sum > target_sum:
                return 0
            
            state = (pos, current_sum, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                new_sum = current_sum + (digit if new_started else 0)
                
                result += dp(pos + 1, new_sum, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)
    
    def sum_of_digit_powers(self, n: int, power: int) -> int:
        """
        Sum of k-th powers of all digits in numbers from 1 to n
        
        Args:
            n: Upper bound
            power: Power to raise each digit to
        
        Returns:
            Sum of k-th powers of all digits
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 0
            
            state = (pos, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                digit_contribution = (digit ** power) if new_started else 0
                result += digit_contribution + dp(pos + 1, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, True, False)

# ==================== CONSTRAINED DIGIT PROBLEMS ====================

class ConstrainedDigitProblems(DigitDP):
    """
    Problems with various digit constraints
    """
    
    def count_numbers_without_digit(self, n: int, forbidden_digit: int) -> int:
        """
        Count numbers up to n that don't contain a specific digit
        
        Args:
            n: Upper bound
            forbidden_digit: Digit that cannot appear
        
        Returns:
            Count of numbers without forbidden digit
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started else 0
            
            state = (pos, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                if digit == forbidden_digit:
                    continue
                
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                result += dp(pos + 1, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, True, False)
    
    def count_numbers_with_at_most_k_digit(self, n: int, digit: int, k: int) -> int:
        """
        Count numbers up to n with at most k occurrences of a specific digit
        
        Args:
            n: Upper bound
            digit: Specific digit to count
            k: Maximum occurrences allowed
        
        Returns:
            Count of numbers with at most k occurrences of digit
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, count: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started else 0
            
            if count > k:
                return 0
            
            state = (pos, count, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for d in range(0, limit + 1):
                new_tight = tight and (d == limit)
                new_started = started or (d > 0)
                new_count = count + (1 if d == digit and new_started else 0)
                
                result += dp(pos + 1, new_count, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)
    
    def count_numbers_divisible_by_k(self, n: int, k: int) -> int:
        """
        Count numbers up to n that are divisible by k
        
        Args:
            n: Upper bound
            k: Divisor
        
        Returns:
            Count of numbers divisible by k
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, remainder: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started and remainder == 0 else 0
            
            state = (pos, remainder, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                if new_started:
                    new_remainder = (remainder * 10 + digit) % k
                else:
                    new_remainder = remainder
                
                result += dp(pos + 1, new_remainder, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)
    
    def count_palindromic_numbers(self, n: int) -> int:
        """
        Count palindromic numbers up to n
        
        Args:
            n: Upper bound
        
        Returns:
            Count of palindromic numbers
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, tight: bool, started: bool, palindrome_digits: List[int]) -> int:
            if pos == length:
                if not started:
                    return 0
                
                # Check if current number is palindrome
                return 1 if palindrome_digits == palindrome_digits[::-1] else 0
            
            state = (pos, tight, started, tuple(palindrome_digits))
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                new_palindrome = palindrome_digits[:]
                
                if new_started:
                    new_palindrome.append(digit)
                
                result += dp(pos + 1, new_tight, new_started, new_palindrome)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, True, False, [])
    
    def count_numbers_with_even_odd_constraint(self, n: int, 
                                             even_positions_even: bool = True) -> int:
        """
        Count numbers where even positions have even digits (or odd based on flag)
        
        Args:
            n: Upper bound
            even_positions_even: If True, even positions must have even digits
        
        Returns:
            Count of numbers satisfying constraint
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started else 0
            
            state = (pos, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                # Check position constraint (0-indexed, so even pos means 1st, 3rd, etc.)
                position_is_even = (pos % 2 == 0)
                digit_is_even = (digit % 2 == 0)
                
                # Apply constraint only for started numbers
                if started and even_positions_even:
                    if position_is_even and not digit_is_even:
                        continue
                elif started and not even_positions_even:
                    if position_is_even and digit_is_even:
                        continue
                
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                
                result += dp(pos + 1, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, True, False)

# ==================== ADVANCED DIGIT DP PROBLEMS ====================

class AdvancedDigitDP(DigitDP):
    """
    Advanced Digit DP problems with complex constraints
    """
    
    def count_numbers_with_decreasing_digits(self, n: int) -> int:
        """
        Count numbers up to n with non-increasing digits
        
        Args:
            n: Upper bound
        
        Returns:
            Count of numbers with non-increasing digits
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, last_digit: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started else 0
            
            state = (pos, last_digit, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                # Check decreasing constraint
                if started and digit > last_digit:
                    continue
                
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                new_last = digit if new_started else last_digit
                
                result += dp(pos + 1, new_last, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 9, True, False)
    
    def count_numbers_with_digit_pattern(self, n: int, pattern: str) -> int:
        """
        Count numbers up to n that contain a specific digit pattern
        
        Args:
            n: Upper bound
            pattern: String pattern to match (e.g., "123")
        
        Returns:
            Count of numbers containing the pattern
        """
        digits = self.get_digits(n)
        length = len(digits)
        pattern_len = len(pattern)
        
        def dp(pos: int, pattern_pos: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started and pattern_pos == pattern_len else 0
            
            state = (pos, pattern_pos, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                new_pattern_pos = pattern_pos
                
                if new_started:
                    if pattern_pos < pattern_len and str(digit) == pattern[pattern_pos]:
                        new_pattern_pos += 1
                    elif pattern_pos > 0:
                        # Reset pattern matching if current doesn't match
                        new_pattern_pos = 1 if str(digit) == pattern[0] else 0
                
                result += dp(pos + 1, new_pattern_pos, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)
    
    def count_numbers_with_balanced_sum(self, n: int) -> int:
        """
        Count numbers where sum of even-positioned digits equals sum of odd-positioned
        
        Args:
            n: Upper bound
        
        Returns:
            Count of numbers with balanced digit sum
        """
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, diff: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started and diff == 0 else 0
            
            state = (pos, diff, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                new_diff = diff
                
                if new_started:
                    # Even position (0-indexed): add to diff
                    # Odd position: subtract from diff
                    if pos % 2 == 0:
                        new_diff += digit
                    else:
                        new_diff -= digit
                
                result += dp(pos + 1, new_diff, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)
    
    def count_numbers_with_prime_digit_sum(self, n: int) -> int:
        """
        Count numbers up to n whose digit sum is prime
        
        Args:
            n: Upper bound
        
        Returns:
            Count of numbers with prime digit sum
        """
        def is_prime(num: int) -> bool:
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            
            for i in range(3, int(num ** 0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True
        
        digits = self.get_digits(n)
        length = len(digits)
        
        def dp(pos: int, digit_sum: int, tight: bool, started: bool) -> int:
            if pos == length:
                return 1 if started and is_prime(digit_sum) else 0
            
            state = (pos, digit_sum, tight, started)
            if state in self.memo:
                return self.memo[state]
            
            limit = digits[pos] if tight else 9
            result = 0
            
            for digit in range(0, limit + 1):
                new_tight = tight and (digit == limit)
                new_started = started or (digit > 0)
                new_sum = digit_sum + (digit if new_started else 0)
                
                result += dp(pos + 1, new_sum, new_tight, new_started)
            
            self.memo[state] = result
            return result
        
        self.clear_memo()
        return dp(0, 0, True, False)

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different Digit DP approaches"""
    print("=== Digit DP Performance Analysis ===\n")
    
    # Test different problem types
    unique_digits = UniqueDigitsProblems()
    digit_sum = DigitSumProblems()
    constrained = ConstrainedDigitProblems()
    
    test_numbers = [100, 1000, 10000]
    
    for n in test_numbers:
        print(f"Testing with n = {n}:")
        
        # Unique digits
        start_time = time.time()
        unique_count = unique_digits.count_unique_digits_up_to_n(n)
        time_unique = time.time() - start_time
        
        # Digit sum
        start_time = time.time()
        sum_digits = digit_sum.sum_of_digits_up_to_n(n)
        time_sum = time.time() - start_time
        
        # Constrained (no digit 5)
        start_time = time.time()
        no_five = constrained.count_numbers_without_digit(n, 5)
        time_constrained = time.time() - start_time
        
        print(f"  Unique digits: {unique_count} ({time_unique:.6f}s)")
        print(f"  Sum of digits: {sum_digits} ({time_sum:.6f}s)")
        print(f"  Without digit 5: {no_five} ({time_constrained:.6f}s)")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Digit DP Demo ===\n")
    
    # Unique Digits Problems
    print("1. Unique Digits Problems:")
    unique_digits = UniqueDigitsProblems()
    
    n = 2
    count_unique = unique_digits.count_numbers_with_unique_digits(n)
    print(f"  Count with unique digits (0 to 10^{n}-1): {count_unique}")
    
    up_to_100 = unique_digits.count_unique_digits_up_to_n(100)
    print(f"  Unique digits up to 100: {up_to_100}")
    
    range_count = unique_digits.count_unique_digits_in_range(10, 50)
    print(f"  Unique digits in range [10, 50]: {range_count}")
    
    sum_target = unique_digits.count_unique_digits_with_sum(3, 10)
    print(f"  3-digit unique numbers with digit sum 10: {sum_target}")
    print()
    
    # Digit Sum Problems
    print("2. Digit Sum Problems:")
    digit_sum = DigitSumProblems()
    
    sum_up_to = digit_sum.sum_of_digits_up_to_n(100)
    print(f"  Sum of all digits up to 100: {sum_up_to}")
    
    sum_range = digit_sum.sum_of_digits_in_range(10, 20)
    print(f"  Sum of digits in range [10, 20]: {sum_range}")
    
    count_with_sum = digit_sum.count_numbers_with_digit_sum(100, 10)
    print(f"  Numbers up to 100 with digit sum 10: {count_with_sum}")
    
    power_sum = digit_sum.sum_of_digit_powers(50, 2)
    print(f"  Sum of squared digits up to 50: {power_sum}")
    print()
    
    # Constrained Digit Problems
    print("3. Constrained Digit Problems:")
    constrained = ConstrainedDigitProblems()
    
    without_digit = constrained.count_numbers_without_digit(100, 7)
    print(f"  Numbers up to 100 without digit 7: {without_digit}")
    
    at_most_k = constrained.count_numbers_with_at_most_k_digit(100, 1, 2)
    print(f"  Numbers up to 100 with at most 2 occurrences of digit 1: {at_most_k}")
    
    divisible = constrained.count_numbers_divisible_by_k(100, 7)
    print(f"  Numbers up to 100 divisible by 7: {divisible}")
    
    palindromic = constrained.count_palindromic_numbers(100)
    print(f"  Palindromic numbers up to 100: {palindromic}")
    
    even_odd = constrained.count_numbers_with_even_odd_constraint(100)
    print(f"  Numbers with even digits at even positions: {even_odd}")
    print()
    
    # Advanced Digit DP
    print("4. Advanced Digit DP Problems:")
    advanced = AdvancedDigitDP()
    
    decreasing = advanced.count_numbers_with_decreasing_digits(100)
    print(f"  Numbers with non-increasing digits up to 100: {decreasing}")
    
    pattern_count = advanced.count_numbers_with_digit_pattern(1000, "12")
    print(f"  Numbers up to 1000 containing pattern '12': {pattern_count}")
    
    balanced = advanced.count_numbers_with_balanced_sum(100)
    print(f"  Numbers with balanced digit sum up to 100: {balanced}")
    
    prime_sum = advanced.count_numbers_with_prime_digit_sum(100)
    print(f"  Numbers with prime digit sum up to 100: {prime_sum}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Digit DP Pattern Recognition ===")
    print("Core Digit DP State Parameters:")
    print("  1. pos: Current position in number")
    print("  2. tight: Whether still bounded by upper limit")
    print("  3. started: Whether we've placed first non-zero digit")
    print("  4. Additional states based on problem requirements")
    
    print("\nCommon Additional States:")
    print("  1. mask: Bitmask for used digits (unique digits)")
    print("  2. sum: Current sum of digits")
    print("  3. remainder: Current number modulo some value")
    print("  4. count: Count of specific digits/patterns")
    print("  5. last_digit: For ordering constraints")
    
    print("\nDigit DP Framework:")
    print("  def dp(pos, tight, started, ...additional_states):")
    print("    if pos == length:")
    print("      return base_case")
    print("    ")
    print("    limit = digits[pos] if tight else 9")
    print("    result = 0")
    print("    ")
    print("    for digit in range(0, limit + 1):")
    print("      new_tight = tight and (digit == limit)")
    print("      new_started = started or (digit > 0)")
    print("      result += dp(pos + 1, new_tight, new_started, ...)")
    
    print("\nOptimization Techniques:")
    print("  1. Memoization with state tuples")
    print("  2. Mathematical shortcuts for simple cases")
    print("  3. Early termination for impossible states")
    print("  4. State compression when possible")
    
    print("\nReal-world Applications:")
    print("  1. Number theory and combinatorics")
    print("  2. Competitive programming contests")
    print("  3. Mathematical analysis and proofs")
    print("  4. Cryptography and number validation")
    print("  5. Statistical analysis of number properties")
    
    print("\nCommon Pitfalls:")
    print("  1. Forgetting to handle leading zeros correctly")
    print("  2. Incorrect tight boundary handling")
    print("  3. Missing base cases")
    print("  4. State explosion with too many parameters")
    print("  5. Not clearing memoization between different calls")
    
    print("\n=== Demo Complete ===") 