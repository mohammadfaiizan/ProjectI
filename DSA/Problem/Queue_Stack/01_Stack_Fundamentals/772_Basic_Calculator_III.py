"""
772. Basic Calculator III - Multiple Approaches
Difficulty: Hard

Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, '+', '-', '*', '/' operators, and open '(' and closing parentheses ')'. The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-2^31, 2^31 - 1].
"""

from typing import List, Union

class BasicCalculatorIII:
    """Multiple approaches to implement advanced calculator with all operations"""
    
    def calculate_stack_approach(self, s: str) -> int:
        """
        Approach 1: Stack-based Calculation with Precedence
        
        Use stack to handle operator precedence and parentheses.
        
        Time: O(n), Space: O(n)
        """
        def update(stack: List[int], num: int, op: str) -> None:
            """Update stack based on operator"""
            if op == '+':
                stack.append(num)
            elif op == '-':
                stack.append(-num)
            elif op == '*':
                stack.append(stack.pop() * num)
            elif op == '/':
                # Handle division with truncation toward zero
                prev = stack.pop()
                stack.append(int(prev / num))
        
        stack = []
        num = 0
        op = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                num = num * 10 + int(char)
            elif char == '(':
                # Recursively calculate expression in parentheses
                paren_count = 1
                j = i + 1
                while paren_count > 0:
                    if s[j] == '(':
                        paren_count += 1
                    elif s[j] == ')':
                        paren_count -= 1
                    j += 1
                
                # Recursively calculate sub-expression
                num = self.calculate_stack_approach(s[i+1:j-1])
                i = j - 1
            
            if char in '+-*/' or i == len(s) - 1:
                update(stack, num, op)
                op = char
                num = 0
        
        return sum(stack)
    
    def calculate_recursive_descent(self, s: str) -> int:
        """
        Approach 2: Recursive Descent Parser
        
        Use recursive descent parsing with proper precedence.
        
        Time: O(n), Space: O(n)
        """
        def parse_expression(index: int) -> tuple:
            """Parse expression, return (result, next_index)"""
            result, index = parse_term(index)
            
            while index < len(s) and s[index] in '+-':
                op = s[index]
                index += 1
                term_result, index = parse_term(index)
                
                if op == '+':
                    result += term_result
                else:  # op == '-'
                    result -= term_result
            
            return result, index
        
        def parse_term(index: int) -> tuple:
            """Parse term (handles * and /), return (result, next_index)"""
            result, index = parse_factor(index)
            
            while index < len(s) and s[index] in '*/':
                op = s[index]
                index += 1
                factor_result, index = parse_factor(index)
                
                if op == '*':
                    result *= factor_result
                else:  # op == '/'
                    result = int(result / factor_result)
            
            return result, index
        
        def parse_factor(index: int) -> tuple:
            """Parse factor (number or parenthesized expression)"""
            # Skip whitespace
            while index < len(s) and s[index] == ' ':
                index += 1
            
            if index < len(s) and s[index] == '(':
                # Parse parenthesized expression
                index += 1  # Skip '('
                result, index = parse_expression(index)
                index += 1  # Skip ')'
                return result, index
            else:
                # Parse number
                num = 0
                while index < len(s) and s[index].isdigit():
                    num = num * 10 + int(s[index])
                    index += 1
                return num, index
        
        result, _ = parse_expression(0)
        return result
    
    def calculate_shunting_yard(self, s: str) -> int:
        """
        Approach 3: Shunting Yard Algorithm
        
        Convert to postfix notation then evaluate.
        
        Time: O(n), Space: O(n)
        """
        def get_precedence(op: str) -> int:
            """Get operator precedence"""
            if op in '+-':
                return 1
            elif op in '*/':
                return 2
            return 0
        
        def infix_to_postfix(s: str) -> List[Union[int, str]]:
            """Convert infix to postfix notation"""
            output = []
            operator_stack = []
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
                    output.append(num)
                elif s[i] in '+-*/':
                    # Handle operators
                    while (operator_stack and 
                           operator_stack[-1] != '(' and
                           get_precedence(operator_stack[-1]) >= get_precedence(s[i])):
                        output.append(operator_stack.pop())
                    operator_stack.append(s[i])
                    i += 1
                elif s[i] == '(':
                    operator_stack.append(s[i])
                    i += 1
                elif s[i] == ')':
                    # Pop until '('
                    while operator_stack and operator_stack[-1] != '(':
                        output.append(operator_stack.pop())
                    operator_stack.pop()  # Remove '('
                    i += 1
            
            # Pop remaining operators
            while operator_stack:
                output.append(operator_stack.pop())
            
            return output
        
        def evaluate_postfix(postfix: List[Union[int, str]]) -> int:
            """Evaluate postfix expression"""
            stack = []
            
            for token in postfix:
                if isinstance(token, int):
                    stack.append(token)
                else:
                    b = stack.pop()
                    a = stack.pop()
                    
                    if token == '+':
                        stack.append(a + b)
                    elif token == '-':
                        stack.append(a - b)
                    elif token == '*':
                        stack.append(a * b)
                    elif token == '/':
                        stack.append(int(a / b))
            
            return stack[0]
        
        postfix = infix_to_postfix(s)
        return evaluate_postfix(postfix)
    
    def calculate_iterative_precedence(self, s: str) -> int:
        """
        Approach 4: Iterative with Precedence Handling
        
        Handle precedence iteratively without recursion.
        
        Time: O(n), Space: O(n)
        """
        def apply_operation(operands: List[int], operators: List[str]) -> None:
            """Apply the top operator to operands"""
            if len(operands) >= 2 and operators:
                b = operands.pop()
                a = operands.pop()
                op = operators.pop()
                
                if op == '+':
                    operands.append(a + b)
                elif op == '-':
                    operands.append(a - b)
                elif op == '*':
                    operands.append(a * b)
                elif op == '/':
                    operands.append(int(a / b))
        
        def get_precedence(op: str) -> int:
            if op in '+-':
                return 1
            elif op in '*/':
                return 2
            return 0
        
        operands = []
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
                operands.append(num)
            elif s[i] in '+-*/':
                # Apply operators with higher or equal precedence
                while (operators and 
                       operators[-1] != '(' and
                       get_precedence(operators[-1]) >= get_precedence(s[i])):
                    apply_operation(operands, operators)
                
                operators.append(s[i])
                i += 1
            elif s[i] == '(':
                operators.append(s[i])
                i += 1
            elif s[i] == ')':
                # Apply operations until '('
                while operators and operators[-1] != '(':
                    apply_operation(operands, operators)
                operators.pop()  # Remove '('
                i += 1
        
        # Apply remaining operations
        while operators:
            apply_operation(operands, operators)
        
        return operands[0] if operands else 0
    
    def calculate_optimized_stack(self, s: str) -> int:
        """
        Approach 5: Optimized Stack with Single Pass
        
        Optimized version with minimal stack operations.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        num = 0
        op = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                num = num * 10 + int(char)
            
            if char in '+-*/' or i == len(s) - 1:
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                elif op == '/':
                    # Handle division with proper truncation
                    prev = stack.pop()
                    stack.append(int(prev / num))
                
                op = char
                num = 0
            elif char == '(':
                # Find matching closing parenthesis
                paren_count = 1
                j = i + 1
                while paren_count > 0:
                    if s[j] == '(':
                        paren_count += 1
                    elif s[j] == ')':
                        paren_count -= 1
                    j += 1
                
                # Recursively evaluate sub-expression
                sub_result = self.calculate_optimized_stack(s[i+1:j-1])
                
                # Apply current operator to sub-result
                if op == '+':
                    stack.append(sub_result)
                elif op == '-':
                    stack.append(-sub_result)
                elif op == '*':
                    stack.append(stack.pop() * sub_result)
                elif op == '/':
                    prev = stack.pop()
                    stack.append(int(prev / sub_result))
                
                # Skip to after closing parenthesis
                i = j - 1
                num = 0
        
        return sum(stack)

def test_basic_calculator_iii():
    """Test advanced calculator algorithms"""
    solver = BasicCalculatorIII()
    
    test_cases = [
        ("1+1", 2, "Simple addition"),
        ("6-4/2", 4, "Mixed operations"),
        ("2*(5+5*2)/3+(6/2+8)", 21, "Complex expression"),
        ("(2+6*3+5-(3*14/7+2)*5)+3", -12, "Very complex"),
        ("0", 0, "Single zero"),
        ("3+2*2", 7, "Precedence test"),
        ("3/2", 1, "Integer division"),
        ("3+5/2", 5, "Mixed division"),
        ("14/3*2", 8, "Left associativity"),
        ("1*2-3/4+5*6-7*8+9/10", -24, "All operations"),
        ("(1)", 1, "Simple parentheses"),
        ("((2))", 2, "Nested parentheses"),
        ("2*3-1", 5, "Multiplication first"),
        ("1+2*3+4", 11, "Multiple precedence"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.calculate_stack_approach),
        ("Recursive Descent", solver.calculate_recursive_descent),
        ("Shunting Yard", solver.calculate_shunting_yard),
        ("Iterative Precedence", solver.calculate_iterative_precedence),
        ("Optimized Stack", solver.calculate_optimized_stack),
    ]
    
    print("=== Testing Basic Calculator III ===")
    
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

def demonstrate_shunting_yard():
    """Demonstrate Shunting Yard algorithm step by step"""
    print("\n=== Shunting Yard Algorithm Demo ===")
    
    expression = "3+4*2/(1-5)"
    print(f"Converting to postfix: '{expression}'")
    
    output = []
    operator_stack = []
    
    def get_precedence(op: str) -> int:
        if op in '+-':
            return 1
        elif op in '*/':
            return 2
        return 0
    
    i = 0
    step = 1
    
    while i < len(expression):
        char = expression[i]
        print(f"\nStep {step}: Processing '{char}'")
        
        if char.isdigit():
            output.append(int(char))
            print(f"  -> Add number {char} to output")
        elif char in '+-*/':
            print(f"  -> Processing operator '{char}' (precedence: {get_precedence(char)})")
            
            while (operator_stack and 
                   operator_stack[-1] != '(' and
                   get_precedence(operator_stack[-1]) >= get_precedence(char)):
                op = operator_stack.pop()
                output.append(op)
                print(f"    -> Pop '{op}' to output (higher/equal precedence)")
            
            operator_stack.append(char)
            print(f"    -> Push '{char}' to operator stack")
        elif char == '(':
            operator_stack.append(char)
            print(f"  -> Push '(' to operator stack")
        elif char == ')':
            print(f"  -> Processing ')' - pop until '('")
            while operator_stack and operator_stack[-1] != '(':
                op = operator_stack.pop()
                output.append(op)
                print(f"    -> Pop '{op}' to output")
            operator_stack.pop()  # Remove '('
            print(f"    -> Remove '(' from stack")
        
        print(f"  -> Output: {output}")
        print(f"  -> Operator stack: {operator_stack}")
        
        i += 1
        step += 1
    
    # Pop remaining operators
    print(f"\nFinal step: Pop remaining operators")
    while operator_stack:
        op = operator_stack.pop()
        output.append(op)
        print(f"  -> Pop '{op}' to output")
    
    print(f"\nPostfix expression: {output}")
    
    # Evaluate postfix
    print(f"\nEvaluating postfix:")
    stack = []
    
    for i, token in enumerate(output):
        print(f"Step {i+1}: Processing {token}")
        
        if isinstance(token, int):
            stack.append(token)
            print(f"  -> Push {token} to stack")
        else:
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                result = int(a / b)
            
            stack.append(result)
            print(f"  -> Calculate {a} {token} {b} = {result}")
        
        print(f"  -> Stack: {stack}")
    
    print(f"\nFinal result: {stack[0]}")

def benchmark_calculator_iii():
    """Benchmark different calculator approaches"""
    import time
    import random
    
    def generate_expression(length: int) -> str:
        """Generate random valid expression"""
        operators = ['+', '-', '*', '/']
        expr = [str(random.randint(1, 10))]
        
        for _ in range(length - 1):
            if random.random() < 0.2:  # Add parentheses
                expr.append('(')
                expr.append(str(random.randint(1, 10)))
                expr.append(random.choice(operators))
                expr.append(str(random.randint(1, 10)))
                expr.append(')')
            else:
                expr.append(random.choice(operators))
                expr.append(str(random.randint(1, 10)))
        
        return ''.join(expr)
    
    algorithms = [
        ("Stack Approach", BasicCalculatorIII().calculate_stack_approach),
        ("Shunting Yard", BasicCalculatorIII().calculate_shunting_yard),
        ("Iterative Precedence", BasicCalculatorIII().calculate_iterative_precedence),
    ]
    
    expression_lengths = [20, 50, 100]
    
    print("\n=== Calculator III Performance Benchmark ===")
    
    for length in expression_lengths:
        print(f"\n--- Expression Length: ~{length} ---")
        test_expressions = []
        
        for _ in range(5):
            try:
                expr = generate_expression(length)
                # Validate expression doesn't have division by zero
                if '/0' not in expr:
                    test_expressions.append(expr)
            except:
                pass
        
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
    """Test edge cases for calculator III"""
    print("\n=== Testing Edge Cases ===")
    
    solver = BasicCalculatorIII()
    
    edge_cases = [
        ("42", 42, "Single number"),
        ("1+2*3", 7, "Precedence"),
        ("(1+2)*3", 9, "Parentheses precedence"),
        ("6/3", 2, "Simple division"),
        ("7/3", 2, "Division truncation"),
        ("-7/3", -2, "Negative division truncation"),
        ("2*3*4", 24, "Multiple multiplication"),
        ("100/10/2", 5, "Left associative division"),
        ("((1))", 1, "Nested parentheses"),
        ("1*2+3*4", 14, "Mixed operations"),
    ]
    
    for expression, expected, description in edge_cases:
        try:
            result = solver.calculate_shunting_yard(expression)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | '{expression}' = {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_basic_calculator_iii()
    demonstrate_shunting_yard()
    test_edge_cases()
    benchmark_calculator_iii()

"""
Basic Calculator III demonstrates advanced expression parsing with full
operator precedence, including Shunting Yard algorithm, recursive descent
parsing, and optimized stack-based evaluation for complex mathematical expressions.
"""
