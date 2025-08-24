"""
1190. Reverse Substrings Between Each Pair of Parentheses - Multiple Approaches
Difficulty: Medium

You are given a string s that consists of lower case English letters and brackets.

Reverse the strings in each pair of matching parentheses, starting from the innermost one.

Your result should not contain any brackets.
"""

from typing import List

class ReverseSubstringsBetweenParentheses:
    """Multiple approaches to reverse substrings between parentheses"""
    
    def reverseParentheses_stack_approach(self, s: str) -> str:
        """
        Approach 1: Stack Approach (Optimal)
        
        Use stack to handle nested parentheses and reversals.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for char in s:
            if char == ')':
                # Pop characters until we find '('
                temp = []
                while stack and stack[-1] != '(':
                    temp.append(stack.pop())
                
                # Remove the '('
                if stack:
                    stack.pop()
                
                # Add reversed characters back to stack
                stack.extend(temp)
            else:
                stack.append(char)
        
        # Filter out any remaining '(' characters and join
        return ''.join(char for char in stack if char != '(')
    
    def reverseParentheses_recursive(self, s: str) -> str:
        """
        Approach 2: Recursive Approach
        
        Use recursion to handle nested structures.
        
        Time: O(n), Space: O(n)
        """
        def parse(s: str, index: int) -> tuple:
            """Parse string and return (result, next_index)"""
            result = []
            
            while index < len(s):
                if s[index] == '(':
                    # Recursively parse inner content
                    inner_result, index = parse(s, index + 1)
                    # Reverse the inner content
                    result.append(inner_result[::-1])
                elif s[index] == ')':
                    # End of current group
                    return ''.join(result), index + 1
                else:
                    # Regular character
                    result.append(s[index])
                    index += 1
            
            return ''.join(result), index
        
        result, _ = parse(s, 0)
        return result
    
    def reverseParentheses_iterative_replacement(self, s: str) -> str:
        """
        Approach 3: Iterative Replacement
        
        Repeatedly find and replace innermost parentheses.
        
        Time: O(n²), Space: O(n)
        """
        while '(' in s:
            # Find innermost parentheses
            start = -1
            for i in range(len(s)):
                if s[i] == '(':
                    start = i
                elif s[i] == ')':
                    # Found matching closing parenthesis
                    # Reverse content between parentheses
                    content = s[start+1:i]
                    reversed_content = content[::-1]
                    
                    # Replace in string
                    s = s[:start] + reversed_content + s[i+1:]
                    break
        
        return s
    
    def reverseParentheses_wormhole_technique(self, s: str) -> str:
        """
        Approach 4: Wormhole Technique (Advanced)
        
        Use paired indices to simulate wormhole teleportation.
        
        Time: O(n), Space: O(n)
        """
        n = len(s)
        pair = [0] * n
        stack = []
        
        # Build pair mapping for parentheses
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                j = stack.pop()
                pair[i] = j
                pair[j] = i
        
        result = []
        i = 0
        direction = 1  # 1 for forward, -1 for backward
        
        while i < n:
            if s[i] in '()':
                # Jump to paired parenthesis and reverse direction
                i = pair[i]
                direction = -direction
            else:
                # Add character to result
                result.append(s[i])
            
            i += direction
        
        return ''.join(result)
    
    def reverseParentheses_two_pass(self, s: str) -> str:
        """
        Approach 5: Two Pass Approach
        
        First pass: identify parentheses pairs, Second pass: build result.
        
        Time: O(n), Space: O(n)
        """
        # First pass: find matching parentheses
        stack = []
        pairs = {}
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    left = stack.pop()
                    pairs[left] = i
                    pairs[i] = left
        
        # Second pass: build result with direction tracking
        result = []
        i = 0
        direction = 1
        
        while i < len(s):
            if s[i] in '()':
                # Jump to paired position and flip direction
                i = pairs[i]
                direction *= -1
            else:
                result.append(s[i])
            
            i += direction
        
        return ''.join(result)


def test_reverse_substrings_between_parentheses():
    """Test reverse substrings algorithms"""
    solver = ReverseSubstringsBetweenParentheses()
    
    test_cases = [
        ("(abcd)", "dcba", "Example 1"),
        ("(u(love)i)", "iloveu", "Example 2"),
        ("(ed(et(oc))el)", "leetcode", "Example 3"),
        ("a(bcdefghijkl(mno)p)q", "apmnolkjihgfedcbq", "Example 4"),
        ("", "", "Empty string"),
        ("abc", "abc", "No parentheses"),
        ("()", "", "Empty parentheses"),
        ("a(b)c", "abc", "Single character in parentheses"),
        ("((a))", "a", "Nested empty-like parentheses"),
        ("a(b(c)d)e", "adbce", "Nested parentheses"),
        ("(a(b(c(d)e)f)g)", "gdefcba", "Deep nesting"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.reverseParentheses_stack_approach),
        ("Recursive", solver.reverseParentheses_recursive),
        ("Iterative Replace", solver.reverseParentheses_iterative_replacement),
        ("Wormhole Technique", solver.reverseParentheses_wormhole_technique),
        ("Two Pass", solver.reverseParentheses_two_pass),
    ]
    
    print("=== Testing Reverse Substrings Between Parentheses ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: '{s}'")
        print(f"Expected: '{expected}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    s = "(u(love)i)"
    print(f"Input: '{s}'")
    print("Strategy: Use stack to collect characters and reverse on closing parenthesis")
    
    stack = []
    
    print(f"\nStep-by-step processing:")
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        print(f"  Stack before: {stack}")
        
        if char == ')':
            # Pop characters until we find '('
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())
            
            print(f"  Popped until '(': {temp}")
            
            # Remove the '('
            if stack:
                stack.pop()
                print(f"  Removed '('")
            
            # Add reversed characters back to stack
            stack.extend(temp)
            print(f"  Added reversed: {temp}")
        else:
            stack.append(char)
            print(f"  Added '{char}'")
        
        print(f"  Stack after: {stack}")
    
    # Build final result
    result = ''.join(char for char in stack if char != '(')
    print(f"\nFinal result: '{result}'")


def demonstrate_wormhole_technique():
    """Demonstrate wormhole technique step by step"""
    print("\n=== Wormhole Technique Step-by-Step Demo ===")
    
    s = "(u(love)i)"
    print(f"Input: '{s}'")
    print("Strategy: Use paired indices to simulate teleportation through parentheses")
    
    n = len(s)
    pair = [0] * n
    stack = []
    
    # Build pair mapping
    print(f"\nBuilding parentheses pairs:")
    for i in range(n):
        if s[i] == '(':
            stack.append(i)
            print(f"  Found '(' at index {i}")
        elif s[i] == ')':
            j = stack.pop()
            pair[i] = j
            pair[j] = i
            print(f"  Found ')' at index {i}, paired with '(' at index {j}")
    
    print(f"\nPair mapping: {pair}")
    
    # Traverse with wormhole jumps
    result = []
    i = 0
    direction = 1
    
    print(f"\nTraversal with wormhole jumps:")
    step = 1
    
    while i < n:
        print(f"  Step {step}: At index {i} ('{s[i]}'), direction: {direction}")
        
        if s[i] in '()':
            old_i = i
            i = pair[i]
            direction = -direction
            print(f"    Wormhole jump: {old_i} -> {i}, new direction: {direction}")
        else:
            result.append(s[i])
            print(f"    Added '{s[i]}' to result: {''.join(result)}")
        
        i += direction
        step += 1
    
    final_result = ''.join(result)
    print(f"\nFinal result: '{final_result}'")


def visualize_nested_processing():
    """Visualize nested parentheses processing"""
    print("\n=== Nested Parentheses Processing Visualization ===")
    
    s = "(ed(et(oc))el)"
    print(f"Input: '{s}'")
    print("Processing order (innermost first):")
    
    # Show the nested structure
    print(f"\nNested structure:")
    print(f"  Level 1: (ed(et(oc))el)")
    print(f"  Level 2:    (et(oc))")
    print(f"  Level 3:       (oc)")
    
    print(f"\nProcessing steps:")
    print(f"  Step 1: Process innermost '(oc)' -> 'co'")
    print(f"          Result: (ed(etco)el)")
    print(f"  Step 2: Process '(etco)' -> 'octe'")
    print(f"          Result: (edocteel)")
    print(f"  Step 3: Process '(edocteel)' -> 'leetcode'")
    print(f"          Final: leetcode")
    
    # Verify with actual implementation
    solver = ReverseSubstringsBetweenParentheses()
    result = solver.reverseParentheses_stack_approach(s)
    print(f"\nActual result: '{result}'")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = ReverseSubstringsBetweenParentheses()
    
    # Application 1: Text formatting with nested emphasis
    print("1. Text Formatting with Nested Emphasis:")
    text_examples = [
        ("Hello (world (beautiful) day)", "Text with nested emphasis"),
        ("The (quick (brown) fox)", "Nested descriptive text"),
        ("Code (review (important) notes)", "Documentation formatting"),
    ]
    
    for text, description in text_examples:
        result = solver.reverseParentheses_stack_approach(text)
        print(f"  {description}:")
        print(f"    Input:  '{text}'")
        print(f"    Output: '{result}'")
    
    # Application 2: Mathematical expression transformation
    print(f"\n2. Mathematical Expression Transformation:")
    expressions = [
        ("f(g(h(x)))", "Nested function calls"),
        ("a(b(c)d)e", "Algebraic expression"),
        ("sin(cos(tan(x)))", "Trigonometric functions"),
    ]
    
    for expr, description in expressions:
        result = solver.reverseParentheses_stack_approach(expr)
        print(f"  {description}:")
        print(f"    Original: '{expr}'")
        print(f"    Reversed: '{result}'")
    
    # Application 3: String obfuscation/deobfuscation
    print(f"\n3. String Obfuscation/Deobfuscation:")
    
    # Example of encoding a message
    original = "secret"
    # Manually create nested structure for demonstration
    encoded = f"({original[:3]}({original[3:]}))"
    decoded = solver.reverseParentheses_stack_approach(encoded)
    
    print(f"  Original message: '{original}'")
    print(f"  Encoded format:   '{encoded}'")
    print(f"  Decoded result:   '{decoded}'")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Single pass with stack"),
        ("Recursive", "O(n)", "O(n)", "Recursion with parsing"),
        ("Iterative Replace", "O(n²)", "O(n)", "Multiple string replacements"),
        ("Wormhole Technique", "O(n)", "O(n)", "Two-pass with teleportation"),
        ("Two Pass", "O(n)", "O(n)", "Pair mapping + traversal"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nStack, Wormhole, and Two Pass approaches are optimal")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = ReverseSubstringsBetweenParentheses()
    
    edge_cases = [
        ("", "", "Empty string"),
        ("a", "a", "Single character"),
        ("()", "", "Empty parentheses"),
        ("(a)", "a", "Single char in parentheses"),
        ("((a))", "a", "Double nested single char"),
        ("(())", "", "Nested empty parentheses"),
        ("a()b", "ab", "Empty parentheses between chars"),
        ("(ab)(cd)", "badc", "Adjacent parentheses groups"),
        ("((()))", "", "Multiple nested empty"),
        ("a(b(c(d(e)f)g)h)i", "ahgfedcbi", "Deep nesting"),
    ]
    
    for s, expected, description in edge_cases:
        try:
            result = solver.reverseParentheses_stack_approach(s)
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | '{s}' -> '{result}'")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    
    approaches = [
        ("Stack", ReverseSubstringsBetweenParentheses().reverseParentheses_stack_approach),
        ("Wormhole", ReverseSubstringsBetweenParentheses().reverseParentheses_wormhole_technique),
        ("Two Pass", ReverseSubstringsBetweenParentheses().reverseParentheses_two_pass),
        ("Iterative", ReverseSubstringsBetweenParentheses().reverseParentheses_iterative_replacement),
    ]
    
    # Generate complex nested string
    test_string = "a" + "(" * 100 + "b" * 100 + ")" * 100 + "c"
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Test string length: {len(test_string)} (100 levels of nesting)")
    
    for name, func in approaches:
        start_time = time.time()
        
        try:
            result = func(test_string)
            end_time = time.time()
            print(f"{name:15} | Time: {end_time - start_time:.4f}s | Result length: {len(result)}")
        except Exception as e:
            print(f"{name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_reverse_substrings_between_parentheses()
    demonstrate_stack_approach()
    demonstrate_wormhole_technique()
    visualize_nested_processing()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_approaches()

"""
Reverse Substrings Between Each Pair of Parentheses demonstrates advanced
string processing with nested structures, including stack-based parsing,
wormhole technique, and multiple approaches for parentheses handling.
"""
