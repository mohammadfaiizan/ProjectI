"""
1614. Maximum Nesting Depth of the Parentheses - Multiple Approaches
Difficulty: Easy

A string is a valid parentheses string (denoted VPS) if it meets one of the following:
- It is an empty string "", or a single character not equal to "(" or ")",
- It can be written as AB (A concatenated with B), where A and B are VPS's, or
- It can be written as (A), where A is a VPS.

We can similarly define the nesting depth depth(S) of any VPS S as follows:
- depth("") = 0
- depth(C) = 0, where C is a string with a single character not equal to "(" or ")".
- depth(A + B) = max(depth(A), depth(B)), where A and B are VPS's.
- depth("(" + A + ")") = 1 + depth(A), where A is a VPS.

For example, "", "()()", and "(()())" are VPS's (with nesting depths 0, 0, and 2), and ")(" and "(()" are not VPS's.

Given a VPS represented as string s, return the nesting depth of s.
"""

from typing import List

class MaximumNestingDepthOfParentheses:
    """Multiple approaches to find maximum nesting depth"""
    
    def maxDepth_stack_approach(self, s: str) -> int:
        """
        Approach 1: Stack-based Approach
        
        Use stack to track open parentheses.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        max_depth = 0
        
        for char in s:
            if char == '(':
                stack.append(char)
                max_depth = max(max_depth, len(stack))
            elif char == ')':
                if stack:
                    stack.pop()
        
        return max_depth
    
    def maxDepth_counter_approach(self, s: str) -> int:
        """
        Approach 2: Counter Approach (Optimal)
        
        Use counter to track current depth without stack.
        
        Time: O(n), Space: O(1)
        """
        current_depth = 0
        max_depth = 0
        
        for char in s:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth
    
    def maxDepth_recursive_approach(self, s: str) -> int:
        """
        Approach 3: Recursive Approach
        
        Use recursion to find maximum depth.
        
        Time: O(n), Space: O(n) due to recursion
        """
        def find_max_depth(s: str, index: int, current_depth: int) -> int:
            if index >= len(s):
                return 0
            
            if s[index] == '(':
                # Increase depth and continue
                remaining_depth = find_max_depth(s, index + 1, current_depth + 1)
                return max(current_depth + 1, remaining_depth)
            elif s[index] == ')':
                # Decrease depth and continue
                remaining_depth = find_max_depth(s, index + 1, current_depth - 1)
                return max(current_depth, remaining_depth)
            else:
                # Skip non-parentheses characters
                remaining_depth = find_max_depth(s, index + 1, current_depth)
                return max(current_depth, remaining_depth)
        
        return find_max_depth(s, 0, 0)
    
    def maxDepth_state_machine(self, s: str) -> int:
        """
        Approach 4: State Machine
        
        Use state machine to track parentheses state.
        
        Time: O(n), Space: O(1)
        """
        depth = 0
        max_depth = 0
        state = "NORMAL"  # NORMAL, INSIDE_PARENS
        
        for char in s:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
                state = "INSIDE_PARENS"
            elif char == ')':
                depth -= 1
                if depth == 0:
                    state = "NORMAL"
            # Ignore other characters
        
        return max_depth
    
    def maxDepth_functional_approach(self, s: str) -> int:
        """
        Approach 5: Functional Approach
        
        Use functional programming style with reduce.
        
        Time: O(n), Space: O(1)
        """
        from functools import reduce
        
        def process_char(acc, char):
            current_depth, max_depth = acc
            
            if char == '(':
                new_depth = current_depth + 1
                return (new_depth, max(max_depth, new_depth))
            elif char == ')':
                return (current_depth - 1, max_depth)
            else:
                return acc
        
        _, max_depth = reduce(process_char, s, (0, 0))
        return max_depth
    
    def maxDepth_two_pass(self, s: str) -> int:
        """
        Approach 6: Two-pass Approach
        
        First pass to validate, second pass to find depth.
        
        Time: O(n), Space: O(1)
        """
        # First pass: validate parentheses (optional for this problem)
        balance = 0
        for char in s:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                if balance < 0:
                    return 0  # Invalid parentheses
        
        # Second pass: find maximum depth
        current_depth = 0
        max_depth = 0
        
        for char in s:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth


def test_maximum_nesting_depth():
    """Test maximum nesting depth algorithms"""
    solver = MaximumNestingDepthOfParentheses()
    
    test_cases = [
        ("(1+(2*3)+((8)/4))+1", 3, "Example 1"),
        ("(1)+((2))+(((3)))", 3, "Example 2"),
        ("1+(2*3)/(2-1)", 1, "Example 3"),
        ("1", 0, "Example 4"),
        ("", 0, "Empty string"),
        ("()", 1, "Single pair"),
        ("(())", 2, "Nested pair"),
        ("()()", 1, "Adjacent pairs"),
        ("((()))", 3, "Triple nested"),
        ("(()())", 2, "Mixed nesting"),
        ("1+2*3", 0, "No parentheses"),
        ("(()(()))", 3, "Complex nesting"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.maxDepth_stack_approach),
        ("Counter Approach", solver.maxDepth_counter_approach),
        ("Recursive Approach", solver.maxDepth_recursive_approach),
        ("State Machine", solver.maxDepth_state_machine),
        ("Functional Approach", solver.maxDepth_functional_approach),
        ("Two Pass", solver.maxDepth_two_pass),
    ]
    
    print("=== Testing Maximum Nesting Depth of Parentheses ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_counter_approach():
    """Demonstrate counter approach step by step"""
    print("\n=== Counter Approach Step-by-Step Demo ===")
    
    s = "(1+(2*3)+((8)/4))+1"
    print(f"String: '{s}'")
    print("Strategy: Track current depth and maximum depth seen")
    
    current_depth = 0
    max_depth = 0
    
    print(f"\nStep-by-step processing:")
    
    for i, char in enumerate(s):
        old_depth = current_depth
        
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            print(f"  {i:2}: '{char}' -> depth: {old_depth} -> {current_depth}, max: {max_depth}")
        elif char == ')':
            current_depth -= 1
            print(f"  {i:2}: '{char}' -> depth: {old_depth} -> {current_depth}, max: {max_depth}")
        else:
            print(f"  {i:2}: '{char}' -> depth: {current_depth} (no change)")
    
    print(f"\nFinal maximum depth: {max_depth}")


def visualize_nesting_structure():
    """Visualize nesting structure"""
    print("\n=== Nesting Structure Visualization ===")
    
    expressions = [
        "(1+(2*3)+((8)/4))+1",
        "(()())",
        "((()))",
        "()()()"
    ]
    
    for expr in expressions:
        print(f"\nExpression: '{expr}'")
        
        # Show depth at each position
        depth = 0
        depth_trace = []
        
        for char in expr:
            if char == '(':
                depth += 1
            elif char == ')':
                depth_trace.append(depth)
                depth -= 1
            else:
                depth_trace.append(depth)
            
            if char != ')':
                depth_trace.append(depth)
        
        # Print with depth indicators
        print("Depth:   ", end="")
        depth = 0
        for char in expr:
            if char == '(':
                depth += 1
                print(f"{depth}", end="")
            elif char == ')':
                print(f"{depth}", end="")
                depth -= 1
            else:
                print(f"{depth}", end="")
        
        print()
        print(f"String:  {expr}")
        
        # Calculate max depth
        solver = MaximumNestingDepthOfParentheses()
        max_depth = solver.maxDepth_counter_approach(expr)
        print(f"Max depth: {max_depth}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Code complexity analysis
    print("1. Code Complexity Analysis:")
    code_snippets = [
        "if (a > 0) { if (b > 0) { return a + b; } }",
        "while (i < n) { for (j = 0; j < m; j++) { process(i, j); } }",
        "function f() { return g(h(i(x))); }"
    ]
    
    solver = MaximumNestingDepthOfParentheses()
    
    for snippet in code_snippets:
        # Extract parentheses for analysis
        parens_only = ''.join(c for c in snippet if c in '()')
        complexity = solver.maxDepth_counter_approach(parens_only)
        print(f"  Code: {snippet[:50]}...")
        print(f"  Parentheses: '{parens_only}'")
        print(f"  Nesting complexity: {complexity}")
    
    # Application 2: Mathematical expression complexity
    print(f"\n2. Mathematical Expression Complexity:")
    math_expressions = [
        "sin(cos(tan(x)))",
        "((a + b) * (c - d)) / ((e + f) * (g - h))",
        "log(exp(sqrt(x^2 + y^2)))"
    ]
    
    for expr in math_expressions:
        parens_only = ''.join(c for c in expr if c in '()')
        complexity = solver.maxDepth_counter_approach(parens_only)
        print(f"  Expression: {expr}")
        print(f"  Parentheses: '{parens_only}'")
        print(f"  Nesting depth: {complexity}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Uses stack to track parentheses"),
        ("Counter Approach", "O(n)", "O(1)", "Optimal space complexity"),
        ("Recursive Approach", "O(n)", "O(n)", "Recursion stack overhead"),
        ("State Machine", "O(n)", "O(1)", "State-based processing"),
        ("Functional Approach", "O(n)", "O(1)", "Functional programming style"),
        ("Two Pass", "O(n)", "O(1)", "Validation + depth calculation"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nCounter approach is optimal with O(n) time and O(1) space")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MaximumNestingDepthOfParentheses()
    
    edge_cases = [
        ("", 0, "Empty string"),
        ("abc", 0, "No parentheses"),
        ("(", 1, "Single open paren"),
        (")", 0, "Single close paren"),
        ("()", 1, "Single pair"),
        ("(())", 2, "Nested pair"),
        ("()()", 1, "Adjacent pairs"),
        ("((((", 4, "Multiple open"),
        ("))))", 0, "Multiple close"),
        ("(a(b(c)d)e)", 3, "With letters"),
        ("1+2*(3+4)", 1, "Math expression"),
        ("((())())", 3, "Mixed nesting"),
    ]
    
    for s, expected, description in edge_cases:
        try:
            result = solver.maxDepth_counter_approach(s)
            status = "✓" if result == expected else "✗"
            print(f"{description:20} | {status} | '{s}' -> {result}")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    
    approaches = [
        ("Stack", MaximumNestingDepthOfParentheses().maxDepth_stack_approach),
        ("Counter", MaximumNestingDepthOfParentheses().maxDepth_counter_approach),
        ("Recursive", MaximumNestingDepthOfParentheses().maxDepth_recursive_approach),
        ("State Machine", MaximumNestingDepthOfParentheses().maxDepth_state_machine),
    ]
    
    # Generate test string with deep nesting
    test_string = "(" * 1000 + "x" + ")" * 1000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Test string length: {len(test_string)} (depth: 1000)")
    
    for name, func in approaches:
        start_time = time.time()
        
        try:
            result = func(test_string)
            end_time = time.time()
            print(f"{name:15} | Time: {end_time - start_time:.4f}s | Result: {result}")
        except Exception as e:
            print(f"{name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_maximum_nesting_depth()
    demonstrate_counter_approach()
    visualize_nesting_structure()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_approaches()

"""
Maximum Nesting Depth of the Parentheses demonstrates expression parsing
fundamentals with multiple approaches for parentheses depth analysis,
including stack-based and space-optimized counter solutions.
"""
