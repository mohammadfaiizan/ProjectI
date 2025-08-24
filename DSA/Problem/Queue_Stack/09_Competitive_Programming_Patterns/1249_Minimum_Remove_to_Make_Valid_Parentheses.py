"""
1249. Minimum Remove to Make Valid Parentheses - Multiple Approaches
Difficulty: Medium

Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:
- It is the empty string, contains only lowercase characters, or
- It can be written as AB (A concatenated with B), where A and B are valid strings, or
- It can be written as (A), where A is a valid string.
"""

from typing import List

class MinimumRemoveToMakeValidParentheses:
    """Multiple approaches to make valid parentheses"""
    
    def minRemoveToMakeValid_stack(self, s: str) -> str:
        """
        Approach 1: Stack-based Solution (Optimal)
        
        Use stack to track unmatched parentheses and remove them.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        to_remove = set()
        
        # First pass: identify unmatched parentheses
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    stack.pop()  # Match found
                else:
                    to_remove.add(i)  # Unmatched ')'
        
        # Add unmatched '(' to removal set
        to_remove.update(stack)
        
        # Second pass: build result string
        result = []
        for i, char in enumerate(s):
            if i not in to_remove:
                result.append(char)
        
        return ''.join(result)
    
    def minRemoveToMakeValid_two_pass(self, s: str) -> str:
        """
        Approach 2: Two-Pass Solution
        
        First pass: remove invalid ')', Second pass: remove invalid '('.
        
        Time: O(n), Space: O(n)
        """
        # First pass: left to right, remove invalid ')'
        first_pass = []
        open_count = 0
        
        for char in s:
            if char == '(':
                first_pass.append(char)
                open_count += 1
            elif char == ')':
                if open_count > 0:
                    first_pass.append(char)
                    open_count -= 1
                # Skip invalid ')'
            else:
                first_pass.append(char)
        
        # Second pass: right to left, remove invalid '('
        result = []
        close_needed = 0
        
        for char in reversed(first_pass):
            if char == ')':
                result.append(char)
                close_needed += 1
            elif char == '(':
                if close_needed > 0:
                    result.append(char)
                    close_needed -= 1
                # Skip invalid '('
            else:
                result.append(char)
        
        return ''.join(reversed(result))
    
    def minRemoveToMakeValid_counter(self, s: str) -> str:
        """
        Approach 3: Counter-based Solution
        
        Use counters to track balance and remove invalid parentheses.
        
        Time: O(n), Space: O(n)
        """
        # Count total parentheses
        open_count = s.count('(')
        close_count = s.count(')')
        
        # Calculate how many to keep
        to_keep = min(open_count, close_count)
        
        result = []
        open_kept = 0
        close_kept = 0
        
        for char in s:
            if char == '(':
                if open_kept < to_keep:
                    result.append(char)
                    open_kept += 1
            elif char == ')':
                if close_kept < open_kept:
                    result.append(char)
                    close_kept += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def minRemoveToMakeValid_recursive(self, s: str) -> str:
        """
        Approach 4: Recursive Solution
        
        Use recursion to process string and maintain balance.
        
        Time: O(n), Space: O(n) due to recursion
        """
        def solve(index: int, open_count: int, current: List[str]) -> str:
            if index == len(s):
                return ''.join(current)
            
            char = s[index]
            
            if char == '(':
                # Try including this '('
                current.append(char)
                result = solve(index + 1, open_count + 1, current)
                if result:
                    return result
                current.pop()
                
                # Try skipping this '('
                return solve(index + 1, open_count, current)
            
            elif char == ')':
                # Try including this ')' if we have unmatched '('
                if open_count > 0:
                    current.append(char)
                    result = solve(index + 1, open_count - 1, current)
                    if result:
                        return result
                    current.pop()
                
                # Try skipping this ')'
                return solve(index + 1, open_count, current)
            
            else:
                # Regular character, always include
                current.append(char)
                return solve(index + 1, open_count, current)
        
        return solve(0, 0, [])
    
    def minRemoveToMakeValid_greedy(self, s: str) -> str:
        """
        Approach 5: Greedy Solution
        
        Greedily keep valid parentheses and remove invalid ones.
        
        Time: O(n), Space: O(n)
        """
        result = []
        balance = 0
        
        # First pass: handle ')' that don't have matching '('
        for char in s:
            if char == '(':
                result.append(char)
                balance += 1
            elif char == ')':
                if balance > 0:
                    result.append(char)
                    balance -= 1
                # Skip unmatched ')'
            else:
                result.append(char)
        
        # Second pass: remove extra '(' from the end
        final_result = []
        for char in reversed(result):
            if char == '(' and balance > 0:
                balance -= 1
                # Skip this '('
            else:
                final_result.append(char)
        
        return ''.join(reversed(final_result))


def test_minimum_remove_to_make_valid():
    """Test minimum remove to make valid parentheses algorithms"""
    solver = MinimumRemoveToMakeValidParentheses()
    
    test_cases = [
        ("()())", "()()"),
        ("(((", ""),
        ("())", "()"),
        ("((a))", "((a))"),
        ("(a))", "(a)"),
        ("((b", ""),
        ("a)b(c)d", "ab(c)d"),
        ("(a(b(c)d)", "a(b(c)d)"),
        ("", ""),
        ("abc", "abc"),
        ("(", ""),
        (")", ""),
        ("(()", "()"),
        ("())", "()"),
        ("(()(()", "(())"),
        ("((a((b))", "(a(b))"),
        ("()(()", "()()"),
    ]
    
    algorithms = [
        ("Stack", solver.minRemoveToMakeValid_stack),
        ("Two Pass", solver.minRemoveToMakeValid_two_pass),
        ("Counter", solver.minRemoveToMakeValid_counter),
        ("Recursive", solver.minRemoveToMakeValid_recursive),
        ("Greedy", solver.minRemoveToMakeValid_greedy),
    ]
    
    print("=== Testing Minimum Remove to Make Valid Parentheses ===")
    
    def is_valid_result(result: str, original: str) -> bool:
        """Check if result is a valid parentheses string derived from original"""
        # Check if result is subsequence of original
        i = j = 0
        while i < len(original) and j < len(result):
            if original[i] == result[j]:
                j += 1
            i += 1
        
        if j != len(result):
            return False
        
        # Check if parentheses are balanced
        balance = 0
        for char in result:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                if balance < 0:
                    return False
        
        return balance == 0
    
    for s, expected_pattern in test_cases:
        print(f"\n--- Input: '{s}' ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                is_valid = is_valid_result(result, s)
                status = "✓" if is_valid else "✗"
                print(f"{alg_name:15} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    s = "((a))"
    
    print(f"Input: '{s}'")
    print("Strategy: Use stack to track unmatched parentheses")
    
    stack = []
    to_remove = set()
    
    print(f"\nFirst pass - identify unmatched parentheses:")
    
    for i, char in enumerate(s):
        print(f"Step {i+1}: char='{char}' at index {i}")
        
        if char == '(':
            stack.append(i)
            print(f"  Push '(' index {i} to stack: {stack}")
        elif char == ')':
            if stack:
                matched = stack.pop()
                print(f"  Match ')' with '(' at index {matched}")
                print(f"  Stack after pop: {stack}")
            else:
                to_remove.add(i)
                print(f"  Unmatched ')', add index {i} to removal set")
        else:
            print(f"  Regular character, skip")
        
        print(f"  Current removal set: {to_remove}")
    
    # Add unmatched '(' to removal set
    print(f"\nUnmatched '(' in stack: {stack}")
    to_remove.update(stack)
    print(f"Final removal set: {to_remove}")
    
    # Build result
    result = []
    print(f"\nSecond pass - build result:")
    for i, char in enumerate(s):
        if i not in to_remove:
            result.append(char)
            print(f"  Keep '{char}' at index {i}")
        else:
            print(f"  Remove '{char}' at index {i}")
    
    final_result = ''.join(result)
    print(f"\nFinal result: '{final_result}'")


def demonstrate_two_pass_approach():
    """Demonstrate two-pass approach"""
    print("\n=== Two-Pass Approach Demo ===")
    
    s = "())"
    
    print(f"Input: '{s}'")
    print("Strategy: First pass removes invalid ')', second pass removes invalid '('")
    
    # First pass
    print(f"\nFirst pass (left to right) - remove invalid ')':")
    first_pass = []
    open_count = 0
    
    for i, char in enumerate(s):
        print(f"  Step {i+1}: char='{char}', open_count={open_count}")
        
        if char == '(':
            first_pass.append(char)
            open_count += 1
            print(f"    Add '(', open_count={open_count}")
        elif char == ')':
            if open_count > 0:
                first_pass.append(char)
                open_count -= 1
                print(f"    Add ')', open_count={open_count}")
            else:
                print(f"    Skip invalid ')'")
        else:
            first_pass.append(char)
            print(f"    Add regular char")
        
        print(f"    Current result: {''.join(first_pass)}")
    
    print(f"After first pass: '{''.join(first_pass)}'")
    
    # Second pass
    print(f"\nSecond pass (right to left) - remove invalid '(':")
    result = []
    close_needed = 0
    
    for i, char in enumerate(reversed(first_pass)):
        print(f"  Step {i+1}: char='{char}', close_needed={close_needed}")
        
        if char == ')':
            result.append(char)
            close_needed += 1
            print(f"    Add ')', close_needed={close_needed}")
        elif char == '(':
            if close_needed > 0:
                result.append(char)
                close_needed -= 1
                print(f"    Add '(', close_needed={close_needed}")
            else:
                print(f"    Skip invalid '('")
        else:
            result.append(char)
            print(f"    Add regular char")
    
    final_result = ''.join(reversed(result))
    print(f"\nFinal result: '{final_result}'")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = MinimumRemoveToMakeValidParentheses()
    
    # Pattern 1: Stack for matching
    print("1. Stack for Parentheses Matching:")
    print("   Use stack to track unmatched opening parentheses")
    print("   Mark positions of unmatched parentheses for removal")
    
    example1 = "((a))"
    result1 = solver.minRemoveToMakeValid_stack(example1)
    print(f"   '{example1}' -> '{result1}'")
    
    # Pattern 2: Two-pass processing
    print(f"\n2. Two-Pass Processing:")
    print("   First pass: left to right, handle one type of error")
    print("   Second pass: right to left, handle the other type")
    
    example2 = "())"
    result2 = solver.minRemoveToMakeValid_two_pass(example2)
    print(f"   '{example2}' -> '{result2}'")
    
    # Pattern 3: Balance tracking
    print(f"\n3. Balance Tracking:")
    print("   Maintain running balance of parentheses")
    print("   Remove characters that would make balance negative")
    
    # Pattern 4: Greedy approach
    print(f"\n4. Greedy Strategy:")
    print("   Keep parentheses that contribute to valid structure")
    print("   Remove excess parentheses greedily")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack", "O(n)", "O(n)", "Two passes, stack for tracking"),
        ("Two Pass", "O(n)", "O(n)", "Two linear passes"),
        ("Counter", "O(n)", "O(n)", "Count then process"),
        ("Recursive", "O(n)", "O(n)", "Recursion stack overhead"),
        ("Greedy", "O(n)", "O(n)", "Two passes with balance tracking"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nAll approaches are O(n) time, stack approach is most intuitive")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MinimumRemoveToMakeValidParentheses()
    
    edge_cases = [
        ("", "Empty string"),
        ("abc", "No parentheses"),
        ("(", "Single open"),
        (")", "Single close"),
        ("((", "Multiple opens"),
        ("))", "Multiple closes"),
        ("()", "Valid pair"),
        ("((()))", "Nested valid"),
        ("()())", "Mixed valid/invalid"),
        ("((())", "Unbalanced"),
        ("a(b)c", "With letters"),
        ("(a(b(c)d)e)", "Complex nested"),
        (")))((((", "All invalid"),
        ("())()", "Middle invalid"),
    ]
    
    for s, description in edge_cases:
        try:
            result = solver.minRemoveToMakeValid_stack(s)
            
            # Validate result
            balance = 0
            valid = True
            for char in result:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                    if balance < 0:
                        valid = False
                        break
            
            valid = valid and balance == 0
            status = "✓" if valid else "✗"
            
            print(f"{description:20} | {status} | '{s}' -> '{result}'")
        except Exception as e:
            print(f"{description:20} | ERROR: {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = MinimumRemoveToMakeValidParentheses()
    
    # Application 1: Code syntax fixing
    print("1. Code Syntax Fixing:")
    print("   Fix unbalanced parentheses in expressions")
    
    code_expressions = [
        "if (condition1 && (condition2)",
        "function(arg1, arg2))",
        "((array[index] + value)",
    ]
    
    for expr in code_expressions:
        fixed = solver.minRemoveToMakeValid_stack(expr)
        print(f"   '{expr}' -> '{fixed}'")
    
    # Application 2: Mathematical expression validation
    print(f"\n2. Mathematical Expression Validation:")
    print("   Clean up mathematical expressions with unbalanced parentheses")
    
    math_expressions = [
        "((x + y) * z",
        "a + (b * c))",
        "((a + b) * (c + d)",
    ]
    
    for expr in math_expressions:
        fixed = solver.minRemoveToMakeValid_stack(expr)
        print(f"   '{expr}' -> '{fixed}'")
    
    # Application 3: Text processing
    print(f"\n3. Text Processing:")
    print("   Clean up text with unbalanced parenthetical remarks")
    
    text_samples = [
        "This is a sentence (with a remark that is incomplete",
        "Another sentence) with a closing remark but no opening",
        "A sentence ((with nested remarks) that are unbalanced",
    ]
    
    for text in text_samples:
        cleaned = solver.minRemoveToMakeValid_stack(text)
        print(f"   '{text}'")
        print(f"   -> '{cleaned}'")


if __name__ == "__main__":
    test_minimum_remove_to_make_valid()
    demonstrate_stack_approach()
    demonstrate_two_pass_approach()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_real_world_applications()

"""
Minimum Remove to Make Valid Parentheses demonstrates competitive
programming patterns with stack-based matching, two-pass processing,
and balance tracking for parentheses validation and correction.
"""
