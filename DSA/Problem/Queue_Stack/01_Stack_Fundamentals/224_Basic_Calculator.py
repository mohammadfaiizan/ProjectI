"""
224. Basic Calculator - Multiple Approaches
Difficulty: Hard

Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

The expression string may contain open ( and closing parentheses ), the plus + or minus - operators, non-negative integers and spaces.
"""

from typing import List

class BasicCalculator:
    """Multiple approaches to implement basic calculator"""
    
    def calculate_stack_approach(self, s: str) -> int:
        """
        Approach 1: Stack-based Calculation
        
        Use stack to handle parentheses and operators.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        result = 0
        number = 0
        sign = 1  # 1 for positive, -1 for negative
        
        for char in s:
            if char.isdigit():
                number = number * 10 + int(char)
            elif char == '+':
                result += sign * number
                number = 0
                sign = 1
            elif char == '-':
                result += sign * number
                number = 0
                sign = -1
            elif char == '(':
                # Push current result and sign to stack
                stack.append(result)
                stack.append(sign)
                # Reset for new sub-expression
                result = 0
                sign = 1
            elif char == ')':
                # Complete current number
                result += sign * number
                number = 0
                
                # Pop sign and previous result
                result *= stack.pop()  # Apply sign
                result += stack.pop()  # Add previous result
        
        # Add the last number
        result += sign * number
        return result
    
    def calculate_recursive_approach(self, s: str) -> int:
        """
        Approach 2: Recursive Descent Parser
        
        Use recursion to handle nested expressions.
        
        Time: O(n), Space: O(n)
        """
        def parse_expression(index: int) -> tuple:
            """Parse expression starting at index, return (result, next_index)"""
            result = 0
            number = 0
            sign = 1
            
            while index < len(s):
                char = s[index]
                
                if char.isdigit():
                    number = number * 10 + int(char)
                elif char == '+':
                    result += sign * number
                    number = 0
                    sign = 1
                elif char == '-':
                    result += sign * number
                    number = 0
                    sign = -1
                elif char == '(':
                    # Recursively parse sub-expression
                    sub_result, next_index = parse_expression(index + 1)
                    number = sub_result
                    index = next_index
                elif char == ')':
                    # End of current sub-expression
                    result += sign * number
                    return result, index
                elif char == ' ':
                    pass  # Skip spaces
                
                index += 1
            
            # Add the last number
            result += sign * number
            return result, index
        
        result, _ = parse_expression(0)
        return result
    
    def calculate_iterative_parsing(self, s: str) -> int:
        """
        Approach 3: Iterative Parsing with State Machine
        
        Use state machine to parse expression.
        
        Time: O(n), Space: O(n)
        """
        def tokenize(s: str) -> List[str]:
            """Convert string to tokens"""
            tokens = []
            i = 0
            
            while i < len(s):
                if s[i].isspace():
                    i += 1
                elif s[i].isdigit():
                    # Parse number
                    num = ""
                    while i < len(s) and s[i].isdigit():
                        num += s[i]
                        i += 1
                    tokens.append(num)
                else:
                    # Operator or parenthesis
                    tokens.append(s[i])
                    i += 1
            
            return tokens
        
        def evaluate_tokens(tokens: List[str]) -> int:
            """Evaluate tokenized expression"""
            stack = []
            result = 0
            sign = 1
            
            for token in tokens:
                if token.isdigit():
                    result += sign * int(token)
                elif token == '+':
                    sign = 1
                elif token == '-':
                    sign = -1
                elif token == '(':
                    stack.append(result)
                    stack.append(sign)
                    result = 0
                    sign = 1
                elif token == ')':
                    result *= stack.pop()  # Apply sign
                    result += stack.pop()  # Add previous result
            
            return result
        
        tokens = tokenize(s)
        return evaluate_tokens(tokens)
    
    def calculate_two_stacks_approach(self, s: str) -> int:
        """
        Approach 4: Two Stacks (Numbers and Operators)
        
        Use separate stacks for numbers and operators.
        
        Time: O(n), Space: O(n)
        """
        def apply_operator(numbers: List[int], operators: List[str]) -> None:
            """Apply the top operator to top two numbers"""
            if len(numbers) >= 2 and operators:
                b = numbers.pop()
                a = numbers.pop()
                op = operators.pop()
                
                if op == '+':
                    numbers.append(a + b)
                elif op == '-':
                    numbers.append(a - b)
        
        numbers = []
        operators = []
        i = 0
        
        while i < len(s):
            if s[i].isspace():
                i += 1
            elif s[i].isdigit():
                # Parse number
                num = 0
                while i < len(s) and s[i].isdigit():
                    num = num * 10 + int(s[i])
                    i += 1
                numbers.append(num)
            elif s[i] in '+-':
                # Apply all pending operations
                while operators and operators[-1] != '(':
                    apply_operator(numbers, operators)
                operators.append(s[i])
                i += 1
            elif s[i] == '(':
                operators.append(s[i])
                i += 1
            elif s[i] == ')':
                # Apply operations until '('
                while operators and operators[-1] != '(':
                    apply_operator(numbers, operators)
                operators.pop()  # Remove '('
                i += 1
        
        # Apply remaining operations
        while operators:
            apply_operator(numbers, operators)
        
        return numbers[0] if numbers else 0
    
    def calculate_postfix_conversion(self, s: str) -> int:
        """
        Approach 5: Convert to Postfix and Evaluate
        
        Convert infix to postfix notation then evaluate.
        
        Time: O(n), Space: O(n)
        """
        def infix_to_postfix(s: str) -> List[str]:
            """Convert infix expression to postfix"""
            postfix = []
            stack = []
            i = 0
            
            while i < len(s):
                if s[i].isspace():
                    i += 1
                elif s[i].isdigit():
                    # Parse number
                    num = ""
                    while i < len(s) and s[i].isdigit():
                        num += s[i]
                        i += 1
                    postfix.append(num)
                elif s[i] in '+-':
                    # Pop operators with higher or equal precedence
                    while stack and stack[-1] in '+-':
                        postfix.append(stack.pop())
                    stack.append(s[i])
                    i += 1
                elif s[i] == '(':
                    stack.append(s[i])
                    i += 1
                elif s[i] == ')':
                    # Pop until '('
                    while stack and stack[-1] != '(':
                        postfix.append(stack.pop())
                    stack.pop()  # Remove '('
                    i += 1
            
            # Pop remaining operators
            while stack:
                postfix.append(stack.pop())
            
            return postfix
        
        def evaluate_postfix(postfix: List[str]) -> int:
            """Evaluate postfix expression"""
            stack = []
            
            for token in postfix:
                if token.isdigit():
                    stack.append(int(token))
                else:
                    b = stack.pop()
                    a = stack.pop()
                    
                    if token == '+':
                        stack.append(a + b)
                    elif token == '-':
                        stack.append(a - b)
            
            return stack[0]
        
        postfix = infix_to_postfix(s)
        return evaluate_postfix(postfix)
    
    def calculate_optimized_single_pass(self, s: str) -> int:
        """
        Approach 6: Optimized Single Pass
        
        Process expression in single pass with minimal stack usage.
        
        Time: O(n), Space: O(n)
        """
        result = 0
        number = 0
        sign = 1
        stack = [0]  # Stack to store intermediate results
        
        for char in s:
            if char.isdigit():
                number = number * 10 + int(char)
            elif char in '+-':
                stack[-1] += sign * number
                number = 0
                sign = 1 if char == '+' else -1
            elif char == '(':
                stack.append(sign)
                stack.append(0)
                sign = 1
            elif char == ')':
                stack[-1] += sign * number
                number = 0
                
                result = stack.pop()
                sign = stack.pop()
                stack[-1] += sign * result
        
        return stack[-1] + sign * number

def test_basic_calculator():
    """Test basic calculator algorithms"""
    solver = BasicCalculator()
    
    test_cases = [
        ("1 + 1", 2, "Simple addition"),
        (" 2-1 + 2 ", 3, "Mixed operations with spaces"),
        ("(1+(4+5+2)-3)+(6+8)", 23, "Complex parentheses"),
        ("1-(     -2)", 3, "Negative numbers"),
        ("2147483647", 2147483647, "Large number"),
        ("1-11", -10, "Negative result"),
        ("(7)-(0)+(4)", 11, "Parentheses around numbers"),
        ("1 + 2 * 3", 7, "No multiplication (treat as addition)"),  # This problem only has + and -
        ("((1+2))", 3, "Nested parentheses"),
        ("1-(5-2)", -2, "Nested subtraction"),
        ("0-1+2", 1, "Starting with zero"),
        ("   ", 0, "Only spaces"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.calculate_stack_approach),
        ("Recursive Approach", solver.calculate_recursive_approach),
        ("Iterative Parsing", solver.calculate_iterative_parsing),
        ("Two Stacks", solver.calculate_two_stacks_approach),
        ("Postfix Conversion", solver.calculate_postfix_conversion),
        ("Optimized Single Pass", solver.calculate_optimized_single_pass),
    ]
    
    print("=== Testing Basic Calculator ===")
    
    for expression, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Expression: '{expression}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(expression)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_stack_calculation():
    """Demonstrate stack-based calculation step by step"""
    print("\n=== Stack Calculation Step-by-Step Demo ===")
    
    s = "(1+(4+5+2)-3)+(6+8)"
    print(f"Calculating: '{s}'")
    
    stack = []
    result = 0
    number = 0
    sign = 1
    
    print(f"\nInitial state: result={result}, number={number}, sign={sign}, stack={stack}")
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing '{char}'")
        
        if char.isdigit():
            number = number * 10 + int(char)
            print(f"  -> Building number: {number}")
        elif char == '+':
            result += sign * number
            print(f"  -> Add {sign * number} to result: {result}")
            number = 0
            sign = 1
            print(f"  -> Reset: number={number}, sign={sign}")
        elif char == '-':
            result += sign * number
            print(f"  -> Add {sign * number} to result: {result}")
            number = 0
            sign = -1
            print(f"  -> Reset: number={number}, sign={sign}")
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            print(f"  -> Push result={result} and sign={sign} to stack")
            result = 0
            sign = 1
            print(f"  -> Reset for sub-expression: result={result}, sign={sign}")
        elif char == ')':
            result += sign * number
            print(f"  -> Complete sub-expression: result={result}")
            number = 0
            
            prev_sign = stack.pop()
            prev_result = stack.pop()
            result = prev_result + prev_sign * result
            print(f"  -> Pop sign={prev_sign}, prev_result={prev_result}")
            print(f"  -> New result: {prev_result} + {prev_sign} * {result//prev_sign if prev_sign != 0 else 0} = {result}")
        elif char == ' ':
            print(f"  -> Skip space")
        
        print(f"  -> State: result={result}, number={number}, sign={sign}, stack={stack}")
    
    # Add the last number
    result += sign * number
    print(f"\nFinal step: Add last number {sign * number}")
    print(f"Final result: {result}")

def benchmark_basic_calculator():
    """Benchmark different calculator approaches"""
    import time
    import random
    
    def generate_expression(length: int) -> str:
        """Generate random valid expression"""
        expr = []
        paren_depth = 0
        
        for i in range(length):
            if i == 0 or expr[-1] in '(+-':
                # Need a number
                if random.random() < 0.3 and paren_depth < 3:
                    expr.append('(')
                    paren_depth += 1
                else:
                    expr.append(str(random.randint(1, 100)))
            else:
                # Can add operator or close paren
                if paren_depth > 0 and random.random() < 0.2:
                    expr.append(')')
                    paren_depth -= 1
                elif random.random() < 0.5:
                    expr.append('+')
                else:
                    expr.append('-')
        
        # Close remaining parentheses
        while paren_depth > 0:
            expr.append(')')
            paren_depth -= 1
        
        return ''.join(expr)
    
    algorithms = [
        ("Stack Approach", BasicCalculator().calculate_stack_approach),
        ("Recursive Approach", BasicCalculator().calculate_recursive_approach),
        ("Optimized Single Pass", BasicCalculator().calculate_optimized_single_pass),
    ]
    
    expression_lengths = [50, 100, 200]
    
    print("\n=== Basic Calculator Performance Benchmark ===")
    
    for length in expression_lengths:
        print(f"\n--- Expression Length: ~{length} ---")
        test_expressions = [generate_expression(length) for _ in range(10)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            for expr in test_expressions:
                try:
                    result = alg_func(expr)
                except:
                    pass  # Skip errors for benchmark
            
            end_time = time.time()
            
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")

def test_edge_cases():
    """Test edge cases for basic calculator"""
    print("\n=== Testing Edge Cases ===")
    
    solver = BasicCalculator()
    
    edge_cases = [
        ("0", 0, "Single zero"),
        ("42", 42, "Single positive number"),
        ("-1", -1, "Single negative number"),
        ("1+2+3+4+5", 15, "Only additions"),
        ("10-2-3-1", 4, "Only subtractions"),
        ("((((1))))", 1, "Deep nesting"),
        ("1-(2-(3-(4-5)))", 3, "Deep nested subtraction"),
        ("   1   +   2   ", 3, "Many spaces"),
        ("()", 0, "Empty parentheses"),
        ("1+(2)", 3, "Unnecessary parentheses"),
    ]
    
    for expression, expected, description in edge_cases:
        try:
            result = solver.calculate_stack_approach(expression)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{expression}' = {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_basic_calculator()
    demonstrate_stack_calculation()
    test_edge_cases()
    benchmark_basic_calculator()

"""
Basic Calculator demonstrates advanced expression parsing techniques
including stack-based evaluation, recursive descent parsing, postfix
conversion, and optimized single-pass algorithms for mathematical expressions.
"""
