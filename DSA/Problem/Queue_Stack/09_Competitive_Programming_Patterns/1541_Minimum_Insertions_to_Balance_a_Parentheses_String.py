"""
1541. Minimum Insertions to Balance a Parentheses String - Multiple Approaches
Difficulty: Medium

Given a parentheses string s containing only the characters '(' and ')'. A parentheses string is balanced if:
- Any left parenthesis '(' must have a corresponding two consecutive right parenthesis '))'.
- Any left parenthesis '(' must come before its corresponding two consecutive right parenthesis '))'.

Return the minimum number of insertions required to make s balanced.
"""

from typing import List

class MinimumInsertionsToBalance:
    """Multiple approaches to balance parentheses with special rules"""
    
    def minInsertions_stack(self, s: str) -> int:
        """
        Approach 1: Stack-based Solution (Optimal)
        
        Use stack to track unmatched '(' and count insertions needed.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        insertions = 0
        i = 0
        
        while i < len(s):
            if s[i] == '(':
                stack.append('(')
                i += 1
            else:  # s[i] == ')'
                # Check if we have '))'
                if i + 1 < len(s) and s[i + 1] == ')':
                    # We have '))'
                    if stack:
                        stack.pop()  # Match with '('
                    else:
                        insertions += 1  # Need to insert '('
                    i += 2
                else:
                    # We have single ')', need another ')'
                    if stack:
                        stack.pop()  # Match with '('
                        insertions += 1  # Need to insert another ')'
                    else:
                        insertions += 2  # Need to insert '(' and ')'
                    i += 1
        
        # Each unmatched '(' needs '))'
        insertions += len(stack) * 2
        
        return insertions
    
    def minInsertions_counter(self, s: str) -> int:
        """
        Approach 2: Counter-based Solution
        
        Use counters to track balance and insertions.
        
        Time: O(n), Space: O(1)
        """
        insertions = 0
        open_needed = 0  # Number of '(' that need matching
        
        i = 0
        while i < len(s):
            if s[i] == '(':
                open_needed += 1
                i += 1
            else:  # s[i] == ')'
                # Check if we have '))'
                if i + 1 < len(s) and s[i + 1] == ')':
                    # We have '))'
                    if open_needed > 0:
                        open_needed -= 1  # Match with existing '('
                    else:
                        insertions += 1  # Need to insert '('
                    i += 2
                else:
                    # We have single ')', need another ')'
                    if open_needed > 0:
                        open_needed -= 1  # Match with existing '('
                        insertions += 1  # Need to insert another ')'
                    else:
                        insertions += 2  # Need to insert '(' and ')'
                    i += 1
        
        # Each remaining open_needed requires '))'
        insertions += open_needed * 2
        
        return insertions
    
    def minInsertions_state_machine(self, s: str) -> int:
        """
        Approach 3: State Machine Approach
        
        Use state machine to track current state and transitions.
        
        Time: O(n), Space: O(1)
        """
        insertions = 0
        balance = 0  # Number of unmatched '('
        
        i = 0
        while i < len(s):
            if s[i] == '(':
                balance += 1
                i += 1
            else:  # s[i] == ')'
                if i + 1 < len(s) and s[i + 1] == ')':
                    # Found '))'
                    if balance > 0:
                        balance -= 1
                    else:
                        insertions += 1  # Insert '('
                    i += 2
                else:
                    # Found single ')'
                    if balance > 0:
                        balance -= 1
                        insertions += 1  # Insert ')'
                    else:
                        insertions += 2  # Insert '(' and ')'
                    i += 1
        
        # Handle remaining unmatched '('
        insertions += balance * 2
        
        return insertions
    
    def minInsertions_greedy(self, s: str) -> int:
        """
        Approach 4: Greedy Approach
        
        Greedily process characters and maintain balance.
        
        Time: O(n), Space: O(1)
        """
        insertions = 0
        open_count = 0
        
        i = 0
        while i < len(s):
            char = s[i]
            
            if char == '(':
                open_count += 1
            else:  # char == ')'
                # Look ahead for another ')'
                if i + 1 < len(s) and s[i + 1] == ')':
                    # We have '))'
                    if open_count > 0:
                        open_count -= 1
                    else:
                        # No '(' to match, insert one
                        insertions += 1
                    i += 1  # Skip the next ')'
                else:
                    # Single ')', need to insert another ')'
                    if open_count > 0:
                        open_count -= 1
                        insertions += 1  # Insert ')'
                    else:
                        # No '(' to match, insert '(' and ')'
                        insertions += 2
            
            i += 1
        
        # Handle remaining unmatched '('
        insertions += open_count * 2
        
        return insertions
    
    def minInsertions_recursive(self, s: str) -> int:
        """
        Approach 5: Recursive Solution
        
        Use recursion to process string and calculate insertions.
        
        Time: O(n), Space: O(n) due to recursion
        """
        def solve(index: int, open_count: int) -> int:
            if index >= len(s):
                # End of string, need ')' for each unmatched '('
                return open_count * 2
            
            char = s[index]
            
            if char == '(':
                return solve(index + 1, open_count + 1)
            else:  # char == ')'
                # Check if next character is also ')'
                if index + 1 < len(s) and s[index + 1] == ')':
                    # We have '))'
                    if open_count > 0:
                        return solve(index + 2, open_count - 1)
                    else:
                        # Need to insert '('
                        return 1 + solve(index + 2, 0)
                else:
                    # Single ')'
                    if open_count > 0:
                        # Need to insert another ')'
                        return 1 + solve(index + 1, open_count - 1)
                    else:
                        # Need to insert '(' and ')'
                        return 2 + solve(index + 1, 0)
        
        return solve(0, 0)


def test_minimum_insertions_to_balance():
    """Test minimum insertions to balance algorithms"""
    solver = MinimumInsertionsToBalance()
    
    test_cases = [
        ("(()))", 1, "Example 1"),
        ("())", 0, "Example 2"),
        ("))()((", 3, "Example 3"),
        ("((((((", 12, "Only open"),
        ("))))))", 5, "Only close"),
        ("", 0, "Empty string"),
        ("(", 2, "Single open"),
        (")", 2, "Single close"),
        ("()", 1, "Single pair incomplete"),
        ("(()))", 1, "Almost balanced"),
        ("(()", 3, "Two opens one close"),
        ("())", 0, "Perfect balance"),
        ("(())", 2, "Nested incomplete"),
        ("))))", 4, "Multiple closes"),
        ("((()", 4, "Multiple opens"),
        ("()())", 1, "Multiple pairs"),
        ("()()", 2, "Two incomplete pairs"),
    ]
    
    algorithms = [
        ("Stack", solver.minInsertions_stack),
        ("Counter", solver.minInsertions_counter),
        ("State Machine", solver.minInsertions_state_machine),
        ("Greedy", solver.minInsertions_greedy),
        ("Recursive", solver.minInsertions_recursive),
    ]
    
    print("=== Testing Minimum Insertions to Balance ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: '{s}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    s = "(()))"
    
    print(f"Input: '{s}'")
    print("Rules: Each '(' needs '))', consecutive ')' count as one unit")
    print("Strategy: Use stack to track unmatched '(' and count insertions")
    
    stack = []
    insertions = 0
    i = 0
    
    print(f"\nStep-by-step processing:")
    
    while i < len(s):
        print(f"\nStep {i+1}: Processing '{s[i]}' at index {i}")
        print(f"  Current stack: {stack}")
        print(f"  Current insertions: {insertions}")
        
        if s[i] == '(':
            stack.append('(')
            print(f"  Added '(' to stack")
            i += 1
        else:  # s[i] == ')'
            # Check if we have '))'
            if i + 1 < len(s) and s[i + 1] == ')':
                print(f"  Found ')' at {i} and ')' at {i+1} -> '))'")
                if stack:
                    matched = stack.pop()
                    print(f"  Matched ')' with '(' from stack")
                else:
                    insertions += 1
                    print(f"  No '(' to match, need to insert '('")
                i += 2
            else:
                print(f"  Found single ')' at {i}")
                if stack:
                    matched = stack.pop()
                    insertions += 1
                    print(f"  Matched with '(' from stack, but need another ')'")
                else:
                    insertions += 2
                    print(f"  No '(' to match, need to insert '(' and ')'")
                i += 1
        
        print(f"  Stack after: {stack}")
        print(f"  Insertions after: {insertions}")
    
    # Handle remaining unmatched '('
    remaining = len(stack)
    if remaining > 0:
        print(f"\nUnmatched '(' in stack: {remaining}")
        print(f"Each needs '))', so add {remaining * 2} insertions")
        insertions += remaining * 2
    
    print(f"\nFinal result: {insertions} insertions needed")


def visualize_balancing_process():
    """Visualize the balancing process"""
    print("\n=== Balancing Process Visualization ===")
    
    test_cases = [
        ("(()))", "Need 1 insertion"),
        ("())", "Already balanced"),
        ("))()((", "Complex case"),
    ]
    
    solver = MinimumInsertionsToBalance()
    
    for s, description in test_cases:
        print(f"\n{description}: '{s}'")
        
        # Show what the balanced string might look like
        insertions = solver.minInsertions_stack(s)
        print(f"Minimum insertions needed: {insertions}")
        
        # Manual analysis for small cases
        if s == "(()))":
            print("Analysis:")
            print("  '(' at index 0 needs '))'")
            print("  '(' at index 1 needs '))'") 
            print("  ')' at index 2 and ')' at index 3 form ')' -> matches second '('")
            print("  ')' at index 4 is extra -> need to insert '(' before it")
            print("  Result: Insert 1 '(' -> '((()))'")
        
        elif s == "())":
            print("Analysis:")
            print("  '(' at index 0 needs '))'")
            print("  ')' at index 1 and ')' at index 2 form ')' -> matches '('")
            print("  Perfect balance, no insertions needed")
        
        elif s == "))()((":
            print("Analysis:")
            print("  ')' at 0,1 -> need '(' -> insert 1")
            print("  '(' at 2 matches ')' at 3, but need another ')' -> insert 1") 
            print("  '(' at 4,5 need ')' -> insert 4")
            print("  Total: 1 + 1 + 4 = 6 insertions")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = MinimumInsertionsToBalance()
    
    # Pattern 1: Modified parentheses matching
    print("1. Modified Parentheses Matching:")
    print("   Each '(' requires ')' instead of single ')'")
    print("   Consecutive ')' are treated as units")
    
    example1 = "(()))"
    result1 = solver.minInsertions_stack(example1)
    print(f"   '{example1}' -> {result1} insertions")
    
    # Pattern 2: Lookahead processing
    print(f"\n2. Lookahead Processing:")
    print("   Check next character when processing ')'")
    print("   Handle ')' differently based on what follows")
    
    # Pattern 3: Balance tracking
    print(f"\n3. Balance Tracking:")
    print("   Track unmatched '(' count")
    print("   Each unmatched '(' needs 2 insertions at the end")
    
    # Pattern 4: Greedy decision making
    print(f"\n4. Greedy Decision Making:")
    print("   Process characters left to right")
    print("   Make optimal local decisions")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack", "O(n)", "O(n)", "Stack for unmatched parentheses"),
        ("Counter", "O(n)", "O(1)", "Counter-based tracking"),
        ("State Machine", "O(n)", "O(1)", "State-based processing"),
        ("Greedy", "O(n)", "O(1)", "Greedy local decisions"),
        ("Recursive", "O(n)", "O(n)", "Recursion stack overhead"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nCounter and State Machine approaches are space-optimal")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MinimumInsertionsToBalance()
    
    edge_cases = [
        ("", 0, "Empty string"),
        ("(", 2, "Single open parenthesis"),
        (")", 2, "Single close parenthesis"),
        ("((", 4, "Two open parentheses"),
        ("))", 1, "Two close parentheses"),
        ("()", 1, "Incomplete pair"),
        ("())", 0, "Complete pair"),
        ("(()))", 1, "Nested with extra"),
        ("((()))", 0, "Perfectly nested"),
        ("))))", 4, "All closes"),
        ("((((", 8, "All opens"),
        ("()())", 1, "Multiple incomplete"),
        ("()()", 2, "Two incomplete pairs"),
        ("(()())", 2, "Complex nested"),
        (")()(", 3, "Mixed pattern"),
    ]
    
    for s, expected, description in edge_cases:
        try:
            result = solver.minInsertions_counter(s)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{s}' -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = MinimumInsertionsToBalance()
    
    # Application 1: Code completion systems
    print("1. Code Completion Systems:")
    print("   Auto-complete parentheses in code editors")
    print("   Special rules for different programming constructs")
    
    code_snippets = [
        "if (condition",
        "function(arg1, arg2",
        "array[index)",
    ]
    
    for snippet in code_snippets:
        # Simulate by treating as parentheses problem
        insertions = solver.minInsertions_counter(snippet.replace('[', '(').replace(']', ')'))
        print(f"   '{snippet}' needs {insertions} completions")
    
    # Application 2: Mathematical expression validation
    print(f"\n2. Mathematical Expression Validation:")
    print("   Validate and fix mathematical expressions")
    print("   Special rules for different bracket types")
    
    math_expressions = [
        "((x + y",
        "a + b))",
        "(x * (y + z",
    ]
    
    for expr in math_expressions:
        insertions = solver.minInsertions_counter(expr)
        print(f"   '{expr}' needs {insertions} fixes")
    
    # Application 3: Template processing
    print(f"\n3. Template Processing:")
    print("   Process template strings with special syntax")
    print("   Handle nested template constructs")
    
    templates = [
        "{{variable",
        "{{#each items}}",
        "{{/if}}}",
    ]
    
    for template in templates:
        # Simulate by converting to parentheses
        paren_version = template.replace('{', '(').replace('}', ')')
        insertions = solver.minInsertions_counter(paren_version)
        print(f"   '{template}' -> {insertions} fixes needed")


if __name__ == "__main__":
    test_minimum_insertions_to_balance()
    demonstrate_stack_approach()
    visualize_balancing_process()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_real_world_applications()

"""
Minimum Insertions to Balance a Parentheses String demonstrates
competitive programming patterns with modified parentheses matching,
lookahead processing, and balance tracking for special bracket rules.
"""
