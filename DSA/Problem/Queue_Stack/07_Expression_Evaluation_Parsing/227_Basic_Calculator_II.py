"""
227. Basic Calculator II - Multiple Approaches
Difficulty: Medium

Given a string s which represents an expression, evaluate this expression and return its value.

The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-2^31, 2^31 - 1].

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

The expression contains only non-negative integers, +, -, *, / operators and empty spaces. The expression does not contain parentheses.
"""

from typing import List

class BasicCalculatorII:
    """Multiple approaches to solve Basic Calculator II"""
    
    def calculate_stack_approach(self, s: str) -> int:
        """
        Approach 1: Stack Approach (Optimal)
        
        Use stack to handle operator precedence.
        
        Time: O(n), Space: O(n)
        """
        if not s:
            return 0
        
        stack = []
        num = 0
        operation = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                num = num * 10 + int(char)
            
            # Process operation when we hit an operator or reach end
            if char in '+-*/' or i == len(s) - 1:
                if operation == '+':
                    stack.append(num)
                elif operation == '-':
                    stack.append(-num)
                elif operation == '*':
                    stack.append(stack.pop() * num)
                elif operation == '/':
                    # Handle negative division (truncate toward zero)
                    prev = stack.pop()
                    stack.append(int(prev / num))
                
                # Reset for next number
                num = 0
                operation = char
        
        return sum(stack)
    
    def calculate_no_stack(self, s: str) -> int:
        """
        Approach 2: No Stack Approach (Space Optimized)
        
        Track only necessary values without using stack.
        
        Time: O(n), Space: O(1)
        """
        if not s:
            return 0
        
        result = 0
        prev_num = 0
        num = 0
        operation = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                num = num * 10 + int(char)
            
            if char in '+-*/' or i == len(s) - 1:
                if operation == '+':
                    result += prev_num
                    prev_num = num
                elif operation == '-':
                    result += prev_num
                    prev_num = -num
                elif operation == '*':
                    prev_num = prev_num * num
                elif operation == '/':
                    prev_num = int(prev_num / num)
                
                num = 0
                operation = char
        
        return result + prev_num
    
    def calculate_two_pass(self, s: str) -> int:
        """
        Approach 3: Two Pass Approach
        
        First pass: handle * and /, Second pass: handle + and -
        
        Time: O(n), Space: O(n)
        """
        # Remove spaces and tokenize
        tokens = []
        num = ''
        
        for char in s:
            if char.isdigit():
                num += char
            elif char in '+-*/':
                if num:
                    tokens.append(int(num))
                    num = ''
                tokens.append(char)
        
        if num:
            tokens.append(int(num))
        
        # First pass: handle * and /
        i = 0
        while i < len(tokens):
            if i > 0 and tokens[i] == '*':
                result = tokens[i-1] * tokens[i+1]
                tokens = tokens[:i-1] + [result] + tokens[i+2:]
                i -= 1
            elif i > 0 and tokens[i] == '/':
                result = int(tokens[i-1] / tokens[i+1])
                tokens = tokens[:i-1] + [result] + tokens[i+2:]
                i -= 1
            else:
                i += 1
        
        # Second pass: handle + and -
        result = tokens[0]
        i = 1
        
        while i < len(tokens):
            if tokens[i] == '+':
                result += tokens[i+1]
                i += 2
            elif tokens[i] == '-':
                result -= tokens[i+1]
                i += 2
            else:
                i += 1
        
        return result
    
    def calculate_recursive(self, s: str) -> int:
        """
        Approach 4: Recursive Approach
        
        Use recursion to handle operator precedence.
        
        Time: O(n), Space: O(n)
        """
        def parse_expression(s: str, index: int) -> tuple:
            """Parse expression and return (result, next_index)"""
            result = 0
            sign = 1
            
            while index < len(s):
                char = s[index]
                
                if char == ' ':
                    index += 1
                elif char.isdigit():
                    num, index = parse_number(s, index)
                    term, index = parse_term(s, index, num)
                    result += sign * term
                elif char == '+':
                    sign = 1
                    index += 1
                elif char == '-':
                    sign = -1
                    index += 1
                else:
                    break
            
            return result, index
        
        def parse_number(s: str, index: int) -> tuple:
            """Parse number and return (number, next_index)"""
            num = 0
            while index < len(s) and s[index].isdigit():
                num = num * 10 + int(s[index])
                index += 1
            return num, index
        
        def parse_term(s: str, index: int, num: int) -> tuple:
            """Parse term (handle * and /) and return (result, next_index)"""
            while index < len(s) and s[index] in '*/':
                op = s[index]
                index += 1
                
                # Skip spaces
                while index < len(s) and s[index] == ' ':
                    index += 1
                
                next_num, index = parse_number(s, index)
                
                if op == '*':
                    num *= next_num
                else:
                    num = int(num / next_num)
            
            return num, index
        
        result, _ = parse_expression(s, 0)
        return result
    
    def calculate_state_machine(self, s: str) -> int:
        """
        Approach 5: State Machine
        
        Use state machine to track parsing state.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        num = 0
        operation = '+'
        state = 'NUMBER'  # NUMBER, OPERATOR
        
        for char in s:
            if char == ' ':
                continue
            
            if state == 'NUMBER':
                if char.isdigit():
                    num = num * 10 + int(char)
                else:
                    # Process the number with previous operation
                    if operation == '+':
                        stack.append(num)
                    elif operation == '-':
                        stack.append(-num)
                    elif operation == '*':
                        stack.append(stack.pop() * num)
                    elif operation == '/':
                        stack.append(int(stack.pop() / num))
                    
                    # Transition to operator state
                    operation = char
                    num = 0
                    state = 'OPERATOR'
            
            elif state == 'OPERATOR':
                if char.isdigit():
                    num = num * 10 + int(char)
                    state = 'NUMBER'
        
        # Process final number
        if operation == '+':
            stack.append(num)
        elif operation == '-':
            stack.append(-num)
        elif operation == '*':
            stack.append(stack.pop() * num)
        elif operation == '/':
            stack.append(int(stack.pop() / num))
        
        return sum(stack)


def test_basic_calculator_ii():
    """Test Basic Calculator II algorithms"""
    solver = BasicCalculatorII()
    
    test_cases = [
        ("3+2*2", 7, "Example 1"),
        (" 3/2 ", 1, "Example 2"),
        (" 3+5 / 2 ", 5, "Example 3"),
        ("1-1+1", 1, "Multiple operations"),
        ("2*3-1", 5, "Multiplication and subtraction"),
        ("14/3*2", 8, "Division and multiplication"),
        ("1*2-3/4+5*6-7*8+9/10", -24, "Complex expression"),
        ("42", 42, "Single number"),
        ("0-1+2", 1, "Starting with zero"),
        ("100000/1/2/3/4/5+100000*2*3*4*5*6/7/8/9/10", 18236, "Large numbers"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.calculate_stack_approach),
        ("No Stack", solver.calculate_no_stack),
        ("Two Pass", solver.calculate_two_pass),
        ("Recursive", solver.calculate_recursive),
        ("State Machine", solver.calculate_state_machine),
    ]
    
    print("=== Testing Basic Calculator II ===")
    
    for expression, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Expression: '{expression}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(expression)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    expression = "3+2*2"
    print(f"Expression: '{expression}'")
    print("Strategy: Use stack to handle operator precedence")
    
    stack = []
    num = 0
    operation = '+'
    
    print(f"\nStep-by-step processing:")
    
    for i, char in enumerate(expression):
        print(f"\nStep {i+1}: Processing '{char}'")
        print(f"  Current num: {num}")
        print(f"  Current operation: '{operation}'")
        print(f"  Stack before: {stack}")
        
        if char.isdigit():
            num = num * 10 + int(char)
            print(f"  Updated num: {num}")
        
        if char in '+-*/' or i == len(expression) - 1:
            if operation == '+':
                stack.append(num)
                print(f"  Added {num} to stack")
            elif operation == '-':
                stack.append(-num)
                print(f"  Added {-num} to stack")
            elif operation == '*':
                prev = stack.pop()
                result = prev * num
                stack.append(result)
                print(f"  Multiplied {prev} * {num} = {result}")
            elif operation == '/':
                prev = stack.pop()
                result = int(prev / num)
                stack.append(result)
                print(f"  Divided {prev} / {num} = {result}")
            
            print(f"  Stack after: {stack}")
            
            if char in '+-*/':
                operation = char
                num = 0
                print(f"  New operation: '{operation}'")
    
    result = sum(stack)
    print(f"\nFinal result: sum({stack}) = {result}")


def demonstrate_operator_precedence():
    """Demonstrate operator precedence handling"""
    print("\n=== Operator Precedence Demonstration ===")
    
    expressions = [
        "2+3*4",
        "2*3+4", 
        "10-2*3",
        "20/4+2",
        "1+2*3+4"
    ]
    
    solver = BasicCalculatorII()
    
    print("How the stack approach handles precedence:")
    
    for expr in expressions:
        result = solver.calculate_stack_approach(expr)
        
        print(f"\nExpression: {expr}")
        print(f"Result: {result}")
        
        # Show evaluation order
        if expr == "2+3*4":
            print("  Evaluation: 2 + (3*4) = 2 + 12 = 14")
            print("  Stack operations: [2] -> [2, 12] -> sum = 14")
        elif expr == "2*3+4":
            print("  Evaluation: (2*3) + 4 = 6 + 4 = 10")
            print("  Stack operations: [6] -> [6, 4] -> sum = 10")


def visualize_calculation_process():
    """Visualize calculation process"""
    print("\n=== Calculation Process Visualization ===")
    
    expression = "14/3*2+1"
    
    print(f"Expression: {expression}")
    print("Processing order (left to right with precedence):")
    
    # Manual step-by-step breakdown
    steps = [
        ("14/3", "14/3 = 4 (integer division)"),
        ("4*2", "4*2 = 8"),
        ("8+1", "8+1 = 9")
    ]
    
    for step, explanation in steps:
        print(f"  {step}: {explanation}")
    
    solver = BasicCalculatorII()
    result = solver.calculate_stack_approach(expression)
    print(f"\nFinal result: {result}")


def demonstrate_edge_cases():
    """Demonstrate edge cases"""
    print("\n=== Edge Cases Demonstration ===")
    
    solver = BasicCalculatorII()
    
    edge_cases = [
        ("0", "Single zero"),
        ("42", "Single positive number"),
        ("1+0", "Addition with zero"),
        ("0*100", "Multiplication by zero"),
        ("5/1", "Division by one"),
        ("100/3", "Integer division with remainder"),
        ("-2+3", "Would be invalid (no unary minus)"),
        ("   3   +   2   ", "Extra spaces"),
    ]
    
    for expr, description in edge_cases:
        try:
            if expr == "-2+3":
                print(f"{description:30} | SKIP | {expr} (unary minus not supported)")
                continue
                
            result = solver.calculate_stack_approach(expr)
            print(f"{description:30} | ✓ | '{expr}' -> {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = BasicCalculatorII()
    
    # Application 1: Financial calculations
    print("1. Financial Calculations:")
    financial_expressions = [
        ("1000+500*12", "Initial amount + monthly savings for a year"),
        ("50000/12", "Annual salary to monthly"),
        ("100*108/100", "Price with 8% tax"),
        ("1000-1000*15/100", "Price after 15% discount"),
    ]
    
    for expr, description in financial_expressions:
        result = solver.calculate_stack_approach(expr)
        print(f"  {description}")
        print(f"    {expr} = {result}")
    
    # Application 2: Unit conversions
    print(f"\n2. Unit Conversions:")
    conversions = [
        ("100*1000", "Kilometers to meters (100 km)"),
        ("3600/60", "Seconds to minutes (3600 sec)"),
        ("212-32", "Fahrenheit to Celsius step 1 (212°F)"),
        ("180*5/9", "Fahrenheit to Celsius step 2"),
    ]
    
    for expr, description in conversions:
        result = solver.calculate_stack_approach(expr)
        print(f"  {description}")
        print(f"    {expr} = {result}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Single pass with stack"),
        ("No Stack", "O(n)", "O(1)", "Space optimized version"),
        ("Two Pass", "O(n)", "O(n)", "Separate passes for operators"),
        ("Recursive", "O(n)", "O(n)", "Recursion stack overhead"),
        ("State Machine", "O(n)", "O(n)", "State-based processing"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nNo Stack approach is optimal with O(1) space complexity")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    
    approaches = [
        ("Stack", BasicCalculatorII().calculate_stack_approach),
        ("No Stack", BasicCalculatorII().calculate_no_stack),
        ("Two Pass", BasicCalculatorII().calculate_two_pass),
        ("State Machine", BasicCalculatorII().calculate_state_machine),
    ]
    
    # Generate complex expression
    test_expr = "1+2*3+4*5+6*7+8*9+10*11+12*13+14*15+16*17+18*19+20*21"
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Test expression length: {len(test_expr)}")
    
    for name, func in approaches:
        start_time = time.time()
        
        try:
            # Run multiple times for better measurement
            for _ in range(1000):
                result = func(test_expr)
            
            end_time = time.time()
            print(f"{name:15} | Time: {end_time - start_time:.4f}s | Result: {result}")
        except Exception as e:
            print(f"{name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_basic_calculator_ii()
    demonstrate_stack_approach()
    demonstrate_operator_precedence()
    visualize_calculation_process()
    demonstrate_edge_cases()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    benchmark_approaches()

"""
Basic Calculator II demonstrates advanced expression evaluation with
operator precedence handling, including multiple optimization approaches
for arithmetic expression parsing without parentheses.
"""
