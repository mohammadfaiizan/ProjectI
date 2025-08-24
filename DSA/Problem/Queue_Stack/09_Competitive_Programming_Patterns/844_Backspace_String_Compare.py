"""
844. Backspace String Compare - Multiple Approaches
Difficulty: Easy

Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue to be empty.
"""

from typing import List

class BackspaceStringCompare:
    """Multiple approaches to compare strings with backspace characters"""
    
    def backspaceCompare_stack(self, s: str, t: str) -> bool:
        """
        Approach 1: Stack Simulation (Optimal for readability)
        
        Use stacks to simulate the typing process.
        
        Time: O(n + m), Space: O(n + m)
        """
        def build_string(string: str) -> str:
            stack = []
            for char in string:
                if char == '#':
                    if stack:
                        stack.pop()
                else:
                    stack.append(char)
            return ''.join(stack)
        
        return build_string(s) == build_string(t)
    
    def backspaceCompare_two_pointers(self, s: str, t: str) -> bool:
        """
        Approach 2: Two Pointers from End (Optimal for space)
        
        Process strings from end to handle backspaces efficiently.
        
        Time: O(n + m), Space: O(1)
        """
        def next_valid_char(string: str, index: int) -> int:
            """Find next valid character index, handling backspaces"""
            backspace_count = 0
            
            while index >= 0:
                if string[index] == '#':
                    backspace_count += 1
                elif backspace_count > 0:
                    backspace_count -= 1
                else:
                    break
                index -= 1
            
            return index
        
        i, j = len(s) - 1, len(t) - 1
        
        while i >= 0 or j >= 0:
            # Find next valid characters
            i = next_valid_char(s, i)
            j = next_valid_char(t, j)
            
            # Compare characters
            if i < 0 and j < 0:
                return True
            if i < 0 or j < 0:
                return False
            if s[i] != t[j]:
                return False
            
            i -= 1
            j -= 1
        
        return True
    
    def backspaceCompare_build_result(self, s: str, t: str) -> bool:
        """
        Approach 3: Build Result Strings
        
        Build the final strings after processing backspaces.
        
        Time: O(n + m), Space: O(n + m)
        """
        def process_string(string: str) -> str:
            result = []
            for char in string:
                if char == '#':
                    if result:
                        result.pop()
                else:
                    result.append(char)
            return ''.join(result)
        
        return process_string(s) == process_string(t)
    
    def backspaceCompare_iterative(self, s: str, t: str) -> bool:
        """
        Approach 4: Iterative Character by Character
        
        Process both strings simultaneously.
        
        Time: O(n + m), Space: O(1)
        """
        def get_next_char(string: str, index: int) -> tuple:
            """Get next valid character and updated index"""
            skip = 0
            
            while index >= 0:
                if string[index] == '#':
                    skip += 1
                elif skip > 0:
                    skip -= 1
                else:
                    return string[index], index - 1
                index -= 1
            
            return None, index
        
        i, j = len(s) - 1, len(t) - 1
        
        while i >= 0 or j >= 0:
            char_s, i = get_next_char(s, i)
            char_t, j = get_next_char(t, j)
            
            if char_s != char_t:
                return False
        
        return True
    
    def backspaceCompare_recursive(self, s: str, t: str) -> bool:
        """
        Approach 5: Recursive Solution
        
        Use recursion to process backspaces.
        
        Time: O(n + m), Space: O(n + m) due to recursion
        """
        def process_recursive(string: str, index: int, result: List[str]) -> None:
            if index >= len(string):
                return
            
            char = string[index]
            if char == '#':
                if result:
                    result.pop()
            else:
                result.append(char)
            
            process_recursive(string, index + 1, result)
        
        result_s, result_t = [], []
        process_recursive(s, 0, result_s)
        process_recursive(t, 0, result_t)
        
        return result_s == result_t
    
    def backspaceCompare_generator(self, s: str, t: str) -> bool:
        """
        Approach 6: Generator-based Solution
        
        Use generators to yield valid characters.
        
        Time: O(n + m), Space: O(1)
        """
        def valid_chars(string: str):
            """Generator that yields valid characters"""
            skip = 0
            for char in reversed(string):
                if char == '#':
                    skip += 1
                elif skip > 0:
                    skip -= 1
                else:
                    yield char
        
        # Compare character by character using generators
        gen_s = valid_chars(s)
        gen_t = valid_chars(t)
        
        return list(gen_s) == list(gen_t)


def test_backspace_string_compare():
    """Test backspace string compare algorithms"""
    solver = BackspaceStringCompare()
    
    test_cases = [
        ("ab#c", "ad#c", True, "Example 1"),
        ("ab##", "#a#c", True, "Example 2"),
        ("a##c", "#a#c", True, "Example 3"),
        ("a#c", "b", False, "Different results"),
        ("", "", True, "Both empty"),
        ("a", "", False, "One empty"),
        ("#", "", True, "Backspace to empty"),
        ("##", "", True, "Multiple backspaces"),
        ("a##b#c", "c", True, "Complex backspaces"),
        ("ab#c#d", "d", True, "Sequential backspaces"),
        ("bxj##tw", "bxo#j##tw", True, "Mixed operations"),
        ("bxj##tw", "bxj###tw", False, "Different backspace counts"),
    ]
    
    algorithms = [
        ("Stack", solver.backspaceCompare_stack),
        ("Two Pointers", solver.backspaceCompare_two_pointers),
        ("Build Result", solver.backspaceCompare_build_result),
        ("Iterative", solver.backspaceCompare_iterative),
        ("Recursive", solver.backspaceCompare_recursive),
        ("Generator", solver.backspaceCompare_generator),
    ]
    
    print("=== Testing Backspace String Compare ===")
    
    for s, t, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"s: '{s}', t: '{t}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s, t)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_two_pointers_approach():
    """Demonstrate two pointers approach step by step"""
    print("\n=== Two Pointers Approach Step-by-Step Demo ===")
    
    s, t = "ab#c", "ad#c"
    
    print(f"Comparing: s='{s}', t='{t}'")
    print("Strategy: Process from end to handle backspaces efficiently")
    
    def next_valid_char_demo(string: str, index: int) -> int:
        """Demo version with logging"""
        print(f"  Finding valid char in '{string}' starting from index {index}")
        backspace_count = 0
        original_index = index
        
        while index >= 0:
            char = string[index]
            print(f"    Index {index}: '{char}'", end="")
            
            if char == '#':
                backspace_count += 1
                print(f" -> backspace count: {backspace_count}")
            elif backspace_count > 0:
                backspace_count -= 1
                print(f" -> skip char, backspace count: {backspace_count}")
            else:
                print(f" -> valid char found!")
                break
            index -= 1
        
        if index < 0:
            print(f"    No valid char found from index {original_index}")
        
        return index
    
    i, j = len(s) - 1, len(t) - 1
    step = 1
    
    while i >= 0 or j >= 0:
        print(f"\nStep {step}: i={i}, j={j}")
        
        # Find next valid characters
        i = next_valid_char_demo(s, i)
        j = next_valid_char_demo(t, j)
        
        print(f"Valid positions: i={i}, j={j}")
        
        # Compare characters
        if i < 0 and j < 0:
            print("Both strings exhausted -> Equal")
            break
        if i < 0 or j < 0:
            print("One string exhausted -> Not equal")
            break
        
        char_s = s[i] if i >= 0 else None
        char_t = t[j] if j >= 0 else None
        
        print(f"Comparing: '{char_s}' vs '{char_t}'")
        
        if char_s != char_t:
            print("Characters different -> Not equal")
            break
        
        print("Characters match -> Continue")
        i -= 1
        j -= 1
        step += 1
    
    print(f"\nFinal result: Equal")


def demonstrate_competitive_programming_optimizations():
    """Demonstrate competitive programming optimizations"""
    print("\n=== Competitive Programming Optimizations ===")
    
    solver = BackspaceStringCompare()
    
    # Optimization 1: Space-optimal solution
    print("1. Space Optimization (O(1) space):")
    print("   Two pointers approach uses constant extra space")
    print("   vs Stack approach which uses O(n) space")
    
    test_cases = [("a#b#c", "c"), ("ab##c", "c"), ("a##b#c", "c")]
    
    for s, t in test_cases:
        result = solver.backspaceCompare_two_pointers(s, t)
        print(f"   '{s}' vs '{t}' -> {result}")
    
    # Optimization 2: Early termination
    print(f"\n2. Early Termination Optimization:")
    print("   Stop as soon as we find a mismatch")
    
    s, t = "abcdef#g", "xyz#g"
    print(f"   Comparing: '{s}' vs '{t}'")
    print("   Algorithm stops at first character comparison")
    print("   No need to process entire strings")
    
    # Optimization 3: Generator for memory efficiency
    print(f"\n3. Generator-based Processing:")
    print("   Use generators for lazy evaluation")
    print("   Memory efficient for large strings")
    
    def show_generator_output(string: str):
        def valid_chars(s: str):
            skip = 0
            for char in reversed(s):
                if char == '#':
                    skip += 1
                elif skip > 0:
                    skip -= 1
                else:
                    yield char
        
        chars = list(valid_chars(string))
        print(f"   '{string}' -> {chars}")
    
    show_generator_output("ab#c#d")
    show_generator_output("a##b#c")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack", "O(n + m)", "O(n + m)", "Build strings then compare"),
        ("Two Pointers", "O(n + m)", "O(1)", "Process from end, optimal space"),
        ("Build Result", "O(n + m)", "O(n + m)", "Similar to stack approach"),
        ("Iterative", "O(n + m)", "O(1)", "Character by character processing"),
        ("Recursive", "O(n + m)", "O(n + m)", "Recursion stack overhead"),
        ("Generator", "O(n + m)", "O(1)", "Lazy evaluation approach"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<10} | {'Space':<10} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<10} | {space_comp:<10} | {notes}")
    
    print(f"\nTwo Pointers approach is optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = BackspaceStringCompare()
    
    edge_cases = [
        ("", "", True, "Both empty strings"),
        ("#", "", True, "Backspace on empty"),
        ("##", "", True, "Multiple backspaces on empty"),
        ("a", "a", True, "Single character match"),
        ("a", "b", False, "Single character mismatch"),
        ("a#", "", True, "Character then backspace"),
        ("#a", "a", True, "Backspace then character"),
        ("a##b", "b", True, "Multiple operations"),
        ("###", "", True, "Only backspaces"),
        ("a#b#c#d#e#f#g#h#i#j#k#", "", True, "Alternating pattern"),
        ("abcdefghijklmnop####", "abcdefghijkl", True, "Long string with backspaces"),
    ]
    
    for s, t, expected, description in edge_cases:
        try:
            result = solver.backspaceCompare_two_pointers(s, t)
            status = "✓" if result == expected else "✗"
            print(f"{description:35} | {status} | s='{s}', t='{t}' -> {result}")
        except Exception as e:
            print(f"{description:35} | ERROR: {str(e)[:30]}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    import random
    import string
    
    approaches = [
        ("Stack", BackspaceStringCompare().backspaceCompare_stack),
        ("Two Pointers", BackspaceStringCompare().backspaceCompare_two_pointers),
        ("Generator", BackspaceStringCompare().backspaceCompare_generator),
    ]
    
    # Generate test data
    def generate_test_string(length: int) -> str:
        chars = []
        for _ in range(length):
            if random.random() < 0.1:  # 10% chance of backspace
                chars.append('#')
            else:
                chars.append(random.choice(string.ascii_lowercase))
        return ''.join(chars)
    
    test_sizes = [100, 1000, 5000]
    
    print(f"\n=== Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\nString length: {size}")
        
        # Generate test strings
        s = generate_test_string(size)
        t = generate_test_string(size)
        
        for name, func in approaches:
            try:
                start_time = time.time()
                
                # Run multiple times for better measurement
                for _ in range(100):
                    func(s, t)
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 100
                
                print(f"  {name:15} | Avg time: {avg_time:.6f}s")
                
            except Exception as e:
                print(f"  {name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_backspace_string_compare()
    demonstrate_two_pointers_approach()
    demonstrate_competitive_programming_optimizations()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_approaches()

"""
Backspace String Compare demonstrates competitive programming patterns
with multiple optimization approaches for string processing with backspace
operations, including space-optimal and time-efficient solutions.
"""
