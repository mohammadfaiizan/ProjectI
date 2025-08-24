"""
402. Remove K Digits - Multiple Approaches
Difficulty: Medium

Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.
"""

from typing import List

class RemoveKDigits:
    """Multiple approaches to remove k digits to get smallest number"""
    
    def removeKdigits_stack_approach(self, num: str, k: int) -> str:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use stack to maintain increasing sequence of digits.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        to_remove = k
        
        for digit in num:
            # Remove larger digits from stack while we can
            while stack and stack[-1] > digit and to_remove > 0:
                stack.pop()
                to_remove -= 1
            
            stack.append(digit)
        
        # Remove remaining digits from the end if needed
        while to_remove > 0:
            stack.pop()
            to_remove -= 1
        
        # Build result and handle leading zeros
        result = ''.join(stack).lstrip('0')
        
        return result if result else '0'
    
    def removeKdigits_greedy_approach(self, num: str, k: int) -> str:
        """
        Approach 2: Greedy Approach
        
        Greedily remove the first digit that is larger than its next digit.
        
        Time: O(n * k), Space: O(n)
        """
        digits = list(num)
        
        for _ in range(k):
            # Find first digit that is larger than its next digit
            i = 0
            while i < len(digits) - 1 and digits[i] <= digits[i + 1]:
                i += 1
            
            # Remove the digit at position i
            digits.pop(i)
        
        # Handle leading zeros and empty result
        result = ''.join(digits).lstrip('0')
        return result if result else '0'
    
    def removeKdigits_recursive(self, num: str, k: int) -> str:
        """
        Approach 3: Recursive Approach
        
        Recursively find the best digit to keep at each position.
        
        Time: O(n * k), Space: O(k)
        """
        def helper(s: str, k: int) -> str:
            if k == 0:
                return s
            if k >= len(s):
                return '0'
            
            # Find the smallest digit in the first k+1 positions
            min_digit = min(s[:k+1])
            min_pos = s.index(min_digit)
            
            # Keep this digit and recursively solve for the rest
            remaining = s[min_pos + 1:]
            remaining_k = k - min_pos
            
            return min_digit + helper(remaining, remaining_k)
        
        result = helper(num, k).lstrip('0')
        return result if result else '0'
    
    def removeKdigits_dp_approach(self, num: str, k: int) -> str:
        """
        Approach 4: Dynamic Programming Approach
        
        Use DP to find optimal solution.
        
        Time: O(n * k), Space: O(n * k)
        """
        n = len(num)
        if k >= n:
            return '0'
        
        # dp[i][j] = smallest number using first i digits, removing j digits
        dp = {}
        
        def solve(pos: int, removed: int) -> str:
            if removed == k:
                return num[pos:]
            if pos == n:
                return '0' if removed < k else ''
            
            if (pos, removed) in dp:
                return dp[(pos, removed)]
            
            # Option 1: Remove current digit
            result1 = solve(pos + 1, removed + 1) if removed < k else None
            
            # Option 2: Keep current digit
            result2 = num[pos] + solve(pos + 1, removed)
            
            # Choose the smaller result
            if result1 is None:
                result = result2
            elif result2 == '':
                result = result1
            else:
                # Compare numerically (handle different lengths)
                if len(result1) < len(result2):
                    result = result1
                elif len(result1) > len(result2):
                    result = result2
                else:
                    result = min(result1, result2)
            
            dp[(pos, removed)] = result
            return result
        
        result = solve(0, 0).lstrip('0')
        return result if result else '0'
    
    def removeKdigits_sliding_window(self, num: str, k: int) -> str:
        """
        Approach 5: Sliding Window Approach
        
        Use sliding window to find optimal digits to keep.
        
        Time: O(n), Space: O(n)
        """
        n = len(num)
        if k >= n:
            return '0'
        
        result = []
        remaining_digits = n - k  # Number of digits to keep
        start = 0
        
        for i in range(remaining_digits):
            # Find the smallest digit in the valid window
            end = start + (k - (remaining_digits - i - 1)) + 1
            end = min(end, n)
            
            min_digit = min(num[start:end])
            min_pos = num.index(min_digit, start)
            
            result.append(min_digit)
            start = min_pos + 1
            k -= (min_pos - start + 1)
        
        result_str = ''.join(result).lstrip('0')
        return result_str if result_str else '0'
    
    def removeKdigits_priority_queue(self, num: str, k: int) -> str:
        """
        Approach 6: Priority Queue Approach
        
        Use priority queue to track removal candidates.
        
        Time: O(n log n), Space: O(n)
        """
        import heapq
        
        n = len(num)
        if k >= n:
            return '0'
        
        # Create list of (digit, position) pairs
        digits = [(int(num[i]), i) for i in range(n)]
        
        # Use a different strategy: build result by selecting smallest available digits
        used = [False] * n
        result = []
        remaining_k = k
        
        for pos in range(n - k):  # We need to select n-k digits
            # Find the smallest digit we can use at this position
            best_digit = '9'
            best_pos = -1
            
            # Look for the smallest digit in valid range
            for i in range(n):
                if used[i]:
                    continue
                
                # Check if we can use this digit (enough digits left to remove)
                digits_after = sum(1 for j in range(i + 1, n) if not used[j])
                positions_needed = (n - k) - pos - 1
                
                if digits_after >= positions_needed:
                    if num[i] < best_digit:
                        best_digit = num[i]
                        best_pos = i
            
            # Use the best digit found
            if best_pos != -1:
                result.append(best_digit)
                used[best_pos] = True
        
        result_str = ''.join(result).lstrip('0')
        return result_str if result_str else '0'


def test_remove_k_digits():
    """Test remove k digits algorithms"""
    solver = RemoveKDigits()
    
    test_cases = [
        ("1432219", 3, "1219", "Example 1"),
        ("10200", 1, "200", "Example 2"),
        ("10", 2, "0", "Remove all digits"),
        ("9", 1, "0", "Single digit removal"),
        ("112", 1, "11", "Remove first occurrence"),
        ("1234567890", 9, "0", "Remove almost all"),
        ("54321", 2, "321", "Decreasing sequence"),
        ("12345", 2, "123", "Increasing sequence"),
        ("100", 1, "0", "Leading zeros"),
        ("10001", 4, "0", "Multiple zeros"),
        ("432", 1, "32", "Remove largest"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.removeKdigits_stack_approach),
        ("Greedy Approach", solver.removeKdigits_greedy_approach),
        ("Recursive", solver.removeKdigits_recursive),
        ("Sliding Window", solver.removeKdigits_sliding_window),
    ]
    
    print("=== Testing Remove K Digits ===")
    
    for num, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Number: '{num}', k: {k}")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(num, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    num = "1432219"
    k = 3
    print(f"Number: '{num}', k: {k}")
    print("Goal: Remove 3 digits to get smallest possible number")
    
    stack = []
    to_remove = k
    
    for i, digit in enumerate(num):
        print(f"\nStep {i+1}: Processing digit '{digit}'")
        print(f"  Stack before: {stack}")
        print(f"  Remaining removals: {to_remove}")
        
        # Remove larger digits from stack
        removed = []
        while stack and stack[-1] > digit and to_remove > 0:
            removed_digit = stack.pop()
            removed.append(removed_digit)
            to_remove -= 1
        
        if removed:
            print(f"  Removed digits: {removed} (larger than '{digit}')")
            print(f"  Remaining removals: {to_remove}")
        
        stack.append(digit)
        print(f"  Stack after: {stack}")
    
    # Remove remaining digits from end if needed
    if to_remove > 0:
        print(f"\nRemoving {to_remove} digits from end:")
        for _ in range(to_remove):
            removed = stack.pop()
            print(f"  Removed '{removed}' from end")
    
    result = ''.join(stack).lstrip('0')
    result = result if result else '0'
    
    print(f"\nFinal stack: {stack}")
    print(f"Final result: '{result}'")


def demonstrate_greedy_concept():
    """Demonstrate greedy concept"""
    print("\n=== Greedy Concept Demonstration ===")
    
    print("Greedy strategy: Remove digits to make the number as small as possible")
    print("Key insight: Remove the first digit that is larger than its next digit")
    
    examples = [
        ("1432219", 3, "Remove 4, 3, 2 to get 1219"),
        ("54321", 2, "Remove 5, 4 to get 321"),
        ("12345", 2, "Remove 4, 5 to get 123"),
    ]
    
    for num, k, explanation in examples:
        print(f"\nExample: '{num}', k={k}")
        print(f"Strategy: {explanation}")
        
        # Show step by step
        digits = list(num)
        removals = []
        
        for removal in range(k):
            # Find first digit larger than next
            i = 0
            while i < len(digits) - 1 and digits[i] <= digits[i + 1]:
                i += 1
            
            removed = digits.pop(i)
            removals.append((removed, i))
            print(f"  Step {removal + 1}: Remove '{removed}' at position {i} -> {''.join(digits)}")
        
        result = ''.join(digits).lstrip('0') or '0'
        print(f"  Final result: '{result}'")


def visualize_digit_removal():
    """Visualize digit removal process"""
    print("\n=== Digit Removal Visualization ===")
    
    num = "1432219"
    k = 3
    print(f"Number: {num}")
    print(f"Remove: {k} digits")
    print()
    
    # Show all digits with positions
    print("Positions: " + " ".join(f"{i}" for i in range(len(num))))
    print("Digits:    " + " ".join(num))
    print()
    
    stack = []
    to_remove = k
    removed_positions = []
    
    for i, digit in enumerate(num):
        # Check what would be removed
        temp_stack = stack[:]
        temp_remove = to_remove
        temp_removed = []
        
        while temp_stack and temp_stack[-1] > digit and temp_remove > 0:
            temp_removed.append(temp_stack.pop())
            temp_remove -= 1
        
        if temp_removed:
            print(f"At position {i} (digit '{digit}'):")
            print(f"  Would remove: {temp_removed}")
            
            # Actually remove them
            while stack and stack[-1] > digit and to_remove > 0:
                stack.pop()
                to_remove -= 1
        
        stack.append(digit)
    
    # Remove from end if needed
    while to_remove > 0:
        stack.pop()
        to_remove -= 1
    
    result = ''.join(stack)
    print(f"\nKept digits: {' '.join(result)}")
    print(f"Final number: {result}")


def benchmark_remove_k_digits():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Approach", RemoveKDigits().removeKdigits_stack_approach),
        ("Greedy Approach", RemoveKDigits().removeKdigits_greedy_approach),
        ("Sliding Window", RemoveKDigits().removeKdigits_sliding_window),
    ]
    
    # Generate test cases
    def generate_number(length: int) -> str:
        return ''.join(str(random.randint(0, 9)) for _ in range(length))
    
    test_cases = [
        (100, 10),
        (1000, 100),
        (5000, 500),
    ]
    
    print("\n=== Remove K Digits Performance Benchmark ===")
    
    for length, k in test_cases:
        print(f"\n--- Number Length: {length}, k: {k} ---")
        
        num = generate_number(length)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(num, k)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result length: {len(result)}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = RemoveKDigits()
    
    edge_cases = [
        ("0", 1, "0", "Single zero"),
        ("1", 1, "0", "Single non-zero"),
        ("10", 1, "0", "Leading zero after removal"),
        ("100", 1, "0", "Multiple leading zeros"),
        ("1000", 3, "0", "All zeros after removal"),
        ("123", 3, "0", "Remove all digits"),
        ("123", 0, "123", "Remove zero digits"),
        ("000", 1, "0", "All zeros"),
        ("102", 1, "0", "Remove to get leading zero"),
        ("9876543210", 1, "876543210", "Remove largest digit"),
    ]
    
    for num, k, expected, description in edge_cases:
        try:
            result = solver.removeKdigits_stack_approach(num, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{num}', k={k} -> '{result}'")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def analyze_optimal_strategy():
    """Analyze the optimal strategy"""
    print("\n=== Optimal Strategy Analysis ===")
    
    print("Why the stack approach works:")
    print("1. We want to keep smaller digits as early as possible")
    print("2. If we see a smaller digit, we should remove larger digits before it")
    print("3. Stack maintains the current best prefix")
    print("4. Monotonic increasing stack ensures optimal ordering")
    
    print("\nExample analysis:")
    num = "1432219"
    k = 3
    print(f"Number: {num}, k: {k}")
    
    print("\nStep-by-step reasoning:")
    print("1. See '1': Keep it (smallest so far)")
    print("2. See '4': Keep it (larger than previous)")
    print("3. See '3': Remove '4', keep '3' (3 < 4)")
    print("4. See '2': Remove '3', keep '2' (2 < 3)")
    print("5. See '2': Keep it (same as previous)")
    print("6. See '1': Remove '2', keep '1' (1 < 2)")
    print("7. See '9': Keep it (larger than previous)")
    print("Result: '1219'")
    
    print("\nWhy this is optimal:")
    print("- We prioritize smaller digits in earlier positions")
    print("- Each removal makes the number smaller")
    print("- Greedy choice leads to global optimum")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ("1432219", 3),
        ("10200", 1),
        ("54321", 2),
        ("12345", 2),
    ]
    
    solver = RemoveKDigits()
    
    approaches = [
        ("Stack", solver.removeKdigits_stack_approach),
        ("Greedy", solver.removeKdigits_greedy_approach),
        ("Recursive", solver.removeKdigits_recursive),
        ("Sliding Window", solver.removeKdigits_sliding_window),
    ]
    
    for i, (num, k) in enumerate(test_cases):
        print(f"\nTest case {i+1}: '{num}', k={k}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(num, k)
                results[name] = result
                print(f"{name:15} | Result: '{result}'")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def demonstrate_leading_zeros():
    """Demonstrate handling of leading zeros"""
    print("\n=== Leading Zeros Handling ===")
    
    examples = [
        ("10200", 1, "Remove '1' -> '0200' -> '200'"),
        ("100", 1, "Remove '1' -> '00' -> '0'"),
        ("1000", 3, "Remove '1', '0', '0' -> '0'"),
        ("0123", 1, "Remove '0' -> '123'"),
    ]
    
    solver = RemoveKDigits()
    
    for num, k, explanation in examples:
        result = solver.removeKdigits_stack_approach(num, k)
        print(f"'{num}', k={k}: {explanation}")
        print(f"  Actual result: '{result}'")
        print()


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Each digit pushed/popped at most once"),
        ("Greedy Approach", "O(n * k)", "O(n)", "k iterations, each O(n)"),
        ("Recursive", "O(n * k)", "O(k)", "Recursive calls with string operations"),
        ("DP Approach", "O(n * k)", "O(n * k)", "DP table with string comparisons"),
        ("Sliding Window", "O(n)", "O(n)", "Single pass with window optimization"),
        ("Priority Queue", "O(n log n)", "O(n)", "Sorting and selection operations"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<12} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<12} | {notes}")


def demonstrate_monotonic_stack_property():
    """Demonstrate monotonic stack property"""
    print("\n=== Monotonic Stack Property ===")
    
    num = "54321"
    k = 2
    print(f"Processing '{num}' with k={k}")
    print("Stack maintains non-decreasing order:")
    
    stack = []
    to_remove = k
    
    for i, digit in enumerate(num):
        print(f"\nStep {i+1}: Processing '{digit}'")
        print(f"  Stack before: {stack}")
        
        # Show monotonic property before
        if len(stack) > 1:
            is_non_decreasing = all(stack[j] <= stack[j+1] for j in range(len(stack)-1))
            print(f"  Non-decreasing before: {'✓' if is_non_decreasing else '✗'}")
        
        # Remove violating elements
        removed = []
        while stack and stack[-1] > digit and to_remove > 0:
            removed.append(stack.pop())
            to_remove -= 1
        
        if removed:
            print(f"  Removed: {removed} (violate monotonic property)")
        
        stack.append(digit)
        print(f"  Stack after: {stack}")
        
        # Show monotonic property after
        if len(stack) > 1:
            is_non_decreasing = all(stack[j] <= stack[j+1] for j in range(len(stack)-1))
            print(f"  Non-decreasing after: {'✓' if is_non_decreasing else '✗'}")
    
    # Remove remaining from end
    while to_remove > 0:
        stack.pop()
        to_remove -= 1
    
    result = ''.join(stack)
    print(f"\nFinal result: '{result}'")


if __name__ == "__main__":
    test_remove_k_digits()
    demonstrate_greedy_concept()
    demonstrate_stack_approach()
    visualize_digit_removal()
    demonstrate_monotonic_stack_property()
    analyze_optimal_strategy()
    demonstrate_leading_zeros()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_remove_k_digits()

"""
Remove K Digits demonstrates monotonic stack applications for
numerical optimization, including greedy strategies and multiple
approaches for digit removal with lexicographical constraints.
"""
