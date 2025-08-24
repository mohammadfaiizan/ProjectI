"""
1021. Remove Outermost Parentheses - Multiple Approaches
Difficulty: Easy

A valid parentheses string is either empty "", "(" + A + ")", or A + B, where A and B are valid parentheses strings, and + represents string concatenation.

For example, "", "()", "(())()", and "(()(()))" are all valid parentheses strings.

A valid parentheses string s is primitive if it is nonempty and there does not exist a way to split it into s = A + B, where A and B are nonempty valid parentheses strings.

Given a valid parentheses string s, consider its primitive decomposition: s = P1 + P2 + ... + Pk, where each Pi is a primitive valid parentheses string.

Return s after removing the outermost parentheses of every primitive string in the primitive decomposition of s.
"""

from typing import List

class RemoveOutermostParentheses:
    """Multiple approaches to remove outermost parentheses"""
    
    def removeOuterParentheses_stack_approach(self, s: str) -> str:
        """
        Approach 1: Stack-based Approach
        
        Use stack to track nesting level and identify outermost parentheses.
        
        Time: O(n), Space: O(n)
        """
        result = []
        stack = []
        
        for char in s:
            if char == '(':
                # Add to result only if not outermost (stack not empty)
                if stack:
                    result.append(char)
                stack.append(char)
            else:  # char == ')'
                stack.pop()
                # Add to result only if not outermost (stack not empty after pop)
                if stack:
                    result.append(char)
        
        return ''.join(result)
    
    def removeOuterParentheses_counter_approach(self, s: str) -> str:
        """
        Approach 2: Counter-based Approach
        
        Use counter to track nesting depth without explicit stack.
        
        Time: O(n), Space: O(n)
        """
        result = []
        depth = 0
        
        for char in s:
            if char == '(':
                # Add to result only if not outermost (depth > 0)
                if depth > 0:
                    result.append(char)
                depth += 1
            else:  # char == ')'
                depth -= 1
                # Add to result only if not outermost (depth > 0 after decrement)
                if depth > 0:
                    result.append(char)
        
        return ''.join(result)
    
    def removeOuterParentheses_primitive_split(self, s: str) -> str:
        """
        Approach 3: Primitive String Splitting
        
        Split into primitive strings first, then remove outer parentheses.
        
        Time: O(n), Space: O(n)
        """
        primitives = []
        depth = 0
        start = 0
        
        # Split into primitive strings
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            else:
                depth -= 1
            
            # When depth becomes 0, we found a complete primitive string
            if depth == 0:
                primitives.append(s[start:i+1])
                start = i + 1
        
        # Remove outermost parentheses from each primitive
        result = []
        for primitive in primitives:
            if len(primitive) > 2:  # More than just "()"
                result.append(primitive[1:-1])  # Remove first and last char
        
        return ''.join(result)
    
    def removeOuterParentheses_two_pointers(self, s: str) -> str:
        """
        Approach 4: Two Pointers Approach
        
        Use two pointers to identify primitive boundaries.
        
        Time: O(n), Space: O(n)
        """
        result = []
        left = 0
        
        while left < len(s):
            # Find the end of current primitive string
            depth = 0
            right = left
            
            while right < len(s):
                if s[right] == '(':
                    depth += 1
                else:
                    depth -= 1
                
                if depth == 0:
                    break
                right += 1
            
            # Add inner part of primitive (excluding outermost parentheses)
            if right > left + 1:  # More than just "()"
                result.append(s[left + 1:right])
            
            left = right + 1
        
        return ''.join(result)
    
    def removeOuterParentheses_recursive_approach(self, s: str) -> str:
        """
        Approach 5: Recursive Approach
        
        Use recursion to process nested structures.
        
        Time: O(n), Space: O(n)
        """
        def process(s: str, start: int) -> tuple:
            """Process string starting at index, return (result, next_index)"""
            if start >= len(s):
                return "", start
            
            result = []
            i = start
            depth = 0
            
            while i < len(s):
                char = s[i]
                
                if char == '(':
                    depth += 1
                    # Add to result only if not outermost
                    if depth > 1:
                        result.append(char)
                else:  # char == ')'
                    depth -= 1
                    # Add to result only if not outermost
                    if depth > 0:
                        result.append(char)
                    
                    # If depth becomes 0, we completed a primitive
                    if depth == 0:
                        break
                
                i += 1
            
            # Process remaining string recursively
            remaining_result, _ = process(s, i + 1)
            
            return ''.join(result) + remaining_result, len(s)
        
        result, _ = process(s, 0)
        return result
    
    def removeOuterParentheses_state_machine(self, s: str) -> str:
        """
        Approach 6: State Machine Approach
        
        Use state machine to track parentheses nesting.
        
        Time: O(n), Space: O(n)
        """
        class ParenthesesStateMachine:
            def __init__(self):
                self.depth = 0
                self.result = []
            
            def process_char(self, char: str) -> None:
                if char == '(':
                    if self.depth > 0:  # Not outermost
                        self.result.append(char)
                    self.depth += 1
                elif char == ')':
                    self.depth -= 1
                    if self.depth > 0:  # Not outermost
                        self.result.append(char)
            
            def get_result(self) -> str:
                return ''.join(self.result)
        
        machine = ParenthesesStateMachine()
        
        for char in s:
            machine.process_char(char)
        
        return machine.get_result()
    
    def removeOuterParentheses_optimized_single_pass(self, s: str) -> str:
        """
        Approach 7: Optimized Single Pass
        
        Single pass with minimal operations.
        
        Time: O(n), Space: O(n)
        """
        result = []
        opened = 0
        
        for char in s:
            if char == '(' and opened > 0:
                result.append(char)
            elif char == ')' and opened > 1:
                result.append(char)
            
            opened += 1 if char == '(' else -1
        
        return ''.join(result)


def test_remove_outermost_parentheses():
    """Test remove outermost parentheses algorithms"""
    solver = RemoveOutermostParentheses()
    
    test_cases = [
        ("(()())(())", "()()()", "Example 1"),
        ("(()())(())(()(()))", "()()()()(())", "Example 2"),
        ("()()", "", "Example 3"),
        ("((()))", "(())", "Single nested"),
        ("()", "", "Single primitive"),
        ("(())()", "()", "Two primitives"),
        ("((()()))", "(()())", "Deep nesting"),
        ("(()())(()())", "()()()()", "Multiple groups"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.removeOuterParentheses_stack_approach),
        ("Counter Approach", solver.removeOuterParentheses_counter_approach),
        ("Primitive Split", solver.removeOuterParentheses_primitive_split),
        ("Two Pointers", solver.removeOuterParentheses_two_pointers),
        ("Recursive Approach", solver.removeOuterParentheses_recursive_approach),
        ("State Machine", solver.removeOuterParentheses_state_machine),
        ("Optimized Single Pass", solver.removeOuterParentheses_optimized_single_pass),
    ]
    
    print("=== Testing Remove Outermost Parentheses ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input:    '{s}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_primitive_decomposition():
    """Demonstrate primitive decomposition concept"""
    print("\n=== Primitive Decomposition Demonstration ===")
    
    s = "(()())(())"
    print(f"Input string: {s}")
    print("Finding primitive decomposition:")
    
    primitives = []
    depth = 0
    start = 0
    
    for i, char in enumerate(s):
        print(f"Step {i+1}: char='{char}', depth before={depth}", end="")
        
        if char == '(':
            depth += 1
        else:
            depth -= 1
        
        print(f", depth after={depth}")
        
        if depth == 0:
            primitive = s[start:i+1]
            primitives.append(primitive)
            print(f"  -> Found primitive: '{primitive}'")
            start = i + 1
    
    print(f"\nPrimitive decomposition: {primitives}")
    
    # Remove outermost parentheses from each primitive
    print("Removing outermost parentheses:")
    result_parts = []
    
    for primitive in primitives:
        if len(primitive) > 2:
            inner = primitive[1:-1]
            result_parts.append(inner)
            print(f"  '{primitive}' -> '{inner}'")
        else:
            print(f"  '{primitive}' -> '' (empty after removing outer)")
    
    result = ''.join(result_parts)
    print(f"\nFinal result: '{result}'")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    s = "(()())(())"
    print(f"Processing: {s}")
    
    result = []
    stack = []
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        print(f"  Stack before: {stack}")
        print(f"  Result before: {''.join(result)}")
        
        if char == '(':
            if stack:  # Not outermost
                result.append(char)
                print(f"    Added '{char}' to result (not outermost)")
            else:
                print(f"    Skipped '{char}' (outermost opening)")
            stack.append(char)
        else:  # char == ')'
            stack.pop()
            if stack:  # Not outermost
                result.append(char)
                print(f"    Added '{char}' to result (not outermost)")
            else:
                print(f"    Skipped '{char}' (outermost closing)")
        
        print(f"  Stack after: {stack}")
        print(f"  Result after: {''.join(result)}")
    
    print(f"\nFinal result: {''.join(result)}")


def visualize_nesting_levels():
    """Visualize nesting levels"""
    print("\n=== Nesting Levels Visualization ===")
    
    s = "((()))(())"
    print(f"String: {s}")
    print("Nesting levels:")
    
    depth = 0
    levels = []
    
    for char in s:
        if char == '(':
            depth += 1
            levels.append(depth)
        else:
            levels.append(depth)
            depth -= 1
    
    # Print visualization
    print("Chars: ", end="")
    for char in s:
        print(f"{char:2}", end="")
    print()
    
    print("Depth: ", end="")
    for level in levels:
        print(f"{level:2}", end="")
    print()
    
    print("Outer: ", end="")
    for level in levels:
        print(f"{'*' if level == 1 else ' ':2}", end="")
    print()
    
    print("\n* marks outermost parentheses (to be removed)")


def benchmark_remove_outermost_parentheses():
    """Benchmark different approaches"""
    import time
    
    algorithms = [
        ("Stack Approach", RemoveOutermostParentheses().removeOuterParentheses_stack_approach),
        ("Counter Approach", RemoveOutermostParentheses().removeOuterParentheses_counter_approach),
        ("Primitive Split", RemoveOutermostParentheses().removeOuterParentheses_primitive_split),
        ("Optimized Single Pass", RemoveOutermostParentheses().removeOuterParentheses_optimized_single_pass),
    ]
    
    # Generate test strings of different sizes
    def generate_nested_string(depth: int, width: int) -> str:
        """Generate nested parentheses string"""
        if depth == 0:
            return "()"
        
        inner = generate_nested_string(depth - 1, width)
        return "(" + inner * width + ")"
    
    test_cases = [
        ("Small nested", generate_nested_string(3, 2)),
        ("Medium nested", generate_nested_string(4, 3)),
        ("Large flat", "()" * 1000),
    ]
    
    print("\n=== Remove Outermost Parentheses Performance Benchmark ===")
    
    for case_name, test_string in test_cases:
        print(f"\n--- {case_name} (length: {len(test_string)}) ---")
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                # Run multiple times for better measurement
                for _ in range(100):
                    result = alg_func(test_string)
                
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = RemoveOutermostParentheses()
    
    edge_cases = [
        ("()", "", "Single primitive pair"),
        ("(())", "()", "Single nested pair"),
        ("()()", "", "Two primitive pairs"),
        ("((()))", "(())", "Triple nested"),
        ("(()())", "()()", "Multiple inner pairs"),
        ("(())(())", "()()", "Two nested groups"),
        ("((()()))", "(()())", "Complex nesting"),
        ("(((())))", "(())", "Deep nesting"),
    ]
    
    for s, expected, description in edge_cases:
        try:
            result = solver.removeOuterParentheses_counter_approach(s)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{s}' -> '{result}'")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_string = "(()())(())(()(()))"
    
    solver = RemoveOutermostParentheses()
    
    approaches = [
        ("Stack", solver.removeOuterParentheses_stack_approach),
        ("Counter", solver.removeOuterParentheses_counter_approach),
        ("Primitive Split", solver.removeOuterParentheses_primitive_split),
        ("Two Pointers", solver.removeOuterParentheses_two_pointers),
        ("Optimized", solver.removeOuterParentheses_optimized_single_pass),
    ]
    
    print(f"Test string: {test_string}")
    
    results = {}
    
    for name, func in approaches:
        try:
            result = func(test_string)
            results[name] = result
            print(f"{name:15} | Result: '{result}'")
        except Exception as e:
            print(f"{name:15} | ERROR: {str(e)[:40]}")
    
    # Check consistency
    if results:
        first_result = list(results.values())[0]
        all_same = all(result == first_result for result in results.values())
        print(f"\nAll approaches agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Uses explicit stack"),
        ("Counter Approach", "O(n)", "O(n)", "Uses depth counter - most efficient"),
        ("Primitive Split", "O(n)", "O(n)", "Two-pass algorithm"),
        ("Two Pointers", "O(n)", "O(n)", "Multiple scans for primitives"),
        ("Recursive", "O(n)", "O(n)", "Recursion overhead"),
        ("State Machine", "O(n)", "O(n)", "Object-oriented approach"),
        ("Optimized Single Pass", "O(n)", "O(n)", "Minimal operations"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<8} | {space_comp:<8} | {notes}")


def demonstrate_optimization():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Demonstration ===")
    
    s = "(()())(())"
    print(f"Input: {s}")
    print("\nComparing counter vs stack approach:")
    
    # Counter approach (more efficient)
    print("\nCounter approach:")
    result1 = []
    depth = 0
    
    for i, char in enumerate(s):
        print(f"  Step {i+1}: '{char}', depth before: {depth}", end="")
        
        if char == '(':
            if depth > 0:
                result1.append(char)
                print(f", added to result", end="")
            depth += 1
        else:
            depth -= 1
            if depth > 0:
                result1.append(char)
                print(f", added to result", end="")
        
        print(f", depth after: {depth}")
    
    print(f"Result: {''.join(result1)}")
    
    # Stack approach (more memory)
    print("\nStack approach:")
    result2 = []
    stack = []
    
    for i, char in enumerate(s):
        print(f"  Step {i+1}: '{char}', stack size before: {len(stack)}", end="")
        
        if char == '(':
            if stack:
                result2.append(char)
                print(f", added to result", end="")
            stack.append(char)
        else:
            stack.pop()
            if stack:
                result2.append(char)
                print(f", added to result", end="")
        
        print(f", stack size after: {len(stack)}")
    
    print(f"Result: {''.join(result2)}")
    print(f"\nBoth approaches give same result: {'✓' if result1 == result2 else '✗'}")


if __name__ == "__main__":
    test_remove_outermost_parentheses()
    demonstrate_primitive_decomposition()
    demonstrate_stack_approach()
    visualize_nesting_levels()
    test_edge_cases()
    compare_approaches()
    demonstrate_optimization()
    analyze_time_complexity()
    benchmark_remove_outermost_parentheses()

"""
Remove Outermost Parentheses demonstrates stack-based string processing
for parentheses manipulation, including multiple approaches for tracking
nesting levels and identifying outermost parentheses efficiently.
"""
