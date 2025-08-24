"""
Expression Evaluation Basic - Multiple Approaches
Difficulty: Easy

Implement basic expression evaluation using stack-based approaches.
Focus on fundamental expression parsing and evaluation techniques.
"""

from typing import List, Union
import operator

class BasicExpressionEvaluator:
    """Multiple approaches for basic expression evaluation"""
    
    def __init__(self):
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow
        }
        
        self.precedence = {
            '+': 1, '-': 1,
            '*': 2, '/': 2, '//': 2, '%': 2,
            '**': 3
        }
    
    def evaluate_postfix(self, expression: str) -> float:
        """
        Approach 1: Postfix Expression Evaluation
        
        Evaluate postfix (RPN) expression using stack.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        tokens = expression.split()
        
        for token in tokens:
            if token in self.operators:
                # Pop two operands
                if len(stack) < 2:
                    raise ValueError(f"Invalid expression: insufficient operands for {token}")
                
                b = stack.pop()
                a = stack.pop()
                
                # Apply operator
                result = self.operators[token](a, b)
                stack.append(result)
            else:
                # Operand - convert to number
                try:
                    stack.append(float(token))
                except ValueError:
                    raise ValueError(f"Invalid operand: {token}")
        
        if len(stack) != 1:
            raise ValueError("Invalid expression: too many operands")
        
        return stack[0]
    
    def evaluate_prefix(self, expression: str) -> float:
        """
        Approach 2: Prefix Expression Evaluation
        
        Evaluate prefix expression using stack.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        tokens = expression.split()
        
        # Process tokens from right to left
        for token in reversed(tokens):
            if token in self.operators:
                # Pop two operands
                if len(stack) < 2:
                    raise ValueError(f"Invalid expression: insufficient operands for {token}")
                
                a = stack.pop()
                b = stack.pop()
                
                # Apply operator
                result = self.operators[token](a, b)
                stack.append(result)
            else:
                # Operand - convert to number
                try:
                    stack.append(float(token))
                except ValueError:
                    raise ValueError(f"Invalid operand: {token}")
        
        if len(stack) != 1:
            raise ValueError("Invalid expression: too many operands")
        
        return stack[0]
    
    def infix_to_postfix(self, expression: str) -> str:
        """
        Approach 3: Infix to Postfix Conversion
        
        Convert infix expression to postfix using Shunting Yard algorithm.
        
        Time: O(n), Space: O(n)
        """
        output = []
        operator_stack = []
        tokens = self._tokenize(expression)
        
        for token in tokens:
            if self._is_number(token):
                output.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                # Pop operators until opening parenthesis
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                
                if not operator_stack:
                    raise ValueError("Mismatched parentheses")
                
                operator_stack.pop()  # Remove '('
            elif token in self.operators:
                # Pop operators with higher or equal precedence
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       operator_stack[-1] in self.precedence and
                       self.precedence[operator_stack[-1]] >= self.precedence[token]):
                    output.append(operator_stack.pop())
                
                operator_stack.append(token)
        
        # Pop remaining operators
        while operator_stack:
            if operator_stack[-1] in '()':
                raise ValueError("Mismatched parentheses")
            output.append(operator_stack.pop())
        
        return ' '.join(output)
    
    def evaluate_infix(self, expression: str) -> float:
        """
        Approach 4: Direct Infix Evaluation
        
        Evaluate infix expression directly using two stacks.
        
        Time: O(n), Space: O(n)
        """
        values = []  # Stack for operands
        operators = []  # Stack for operators
        tokens = self._tokenize(expression)
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if self._is_number(token):
                values.append(float(token))
            elif token == '(':
                operators.append(token)
            elif token == ')':
                # Evaluate until opening parenthesis
                while operators and operators[-1] != '(':
                    self._apply_operator(values, operators)
                
                if not operators:
                    raise ValueError("Mismatched parentheses")
                
                operators.pop()  # Remove '('
            elif token in self.operators:
                # Apply operators with higher or equal precedence
                while (operators and 
                       operators[-1] != '(' and
                       operators[-1] in self.precedence and
                       self.precedence[operators[-1]] >= self.precedence[token]):
                    self._apply_operator(values, operators)
                
                operators.append(token)
            
            i += 1
        
        # Apply remaining operators
        while operators:
            if operators[-1] in '()':
                raise ValueError("Mismatched parentheses")
            self._apply_operator(values, operators)
        
        if len(values) != 1:
            raise ValueError("Invalid expression")
        
        return values[0]
    
    def evaluate_simple(self, expression: str) -> float:
        """
        Approach 5: Simple Expression Evaluation (Python eval alternative)
        
        Evaluate simple expressions safely.
        
        Time: O(n), Space: O(1)
        """
        # Remove whitespace
        expression = expression.replace(' ', '')
        
        # Handle simple cases
        if not expression:
            return 0
        
        # Split by + and - (lowest precedence)
        result = 0
        current_term = ""
        current_sign = 1
        
        i = 0
        while i < len(expression):
            char = expression[i]
            
            if char in '+-':
                if current_term:
                    result += current_sign * self._evaluate_term(current_term)
                    current_term = ""
                
                current_sign = 1 if char == '+' else -1
            else:
                current_term += char
            
            i += 1
        
        # Add the last term
        if current_term:
            result += current_sign * self._evaluate_term(current_term)
        
        return result
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize expression into numbers and operators"""
        tokens = []
        current_token = ""
        
        for char in expression:
            if char.isspace():
                continue
            elif char.isdigit() or char == '.':
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                
                # Handle multi-character operators
                if char == '*' and tokens and tokens[-1] == '*':
                    tokens[-1] = '**'
                elif char == '/' and tokens and tokens[-1] == '/':
                    tokens[-1] = '//'
                else:
                    tokens.append(char)
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number"""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def _apply_operator(self, values: List[float], operators: List[str]) -> None:
        """Apply operator to top two values"""
        if len(values) < 2:
            raise ValueError("Invalid expression: insufficient operands")
        
        b = values.pop()
        a = values.pop()
        op = operators.pop()
        
        result = self.operators[op](a, b)
        values.append(result)
    
    def _evaluate_term(self, term: str) -> float:
        """Evaluate a term (handles * and /)"""
        # Split by * and /
        factors = []
        operators = []
        current_factor = ""
        
        for char in term:
            if char in '*/':
                if current_factor:
                    factors.append(float(current_factor))
                    current_factor = ""
                operators.append(char)
            else:
                current_factor += char
        
        if current_factor:
            factors.append(float(current_factor))
        
        # Evaluate left to right
        result = factors[0]
        for i, op in enumerate(operators):
            if op == '*':
                result *= factors[i + 1]
            elif op == '/':
                result /= factors[i + 1]
        
        return result


def test_expression_evaluator():
    """Test expression evaluation approaches"""
    evaluator = BasicExpressionEvaluator()
    
    test_cases = [
        # Postfix expressions
        ("3 4 +", 7, "postfix"),
        ("3 4 + 2 *", 14, "postfix"),
        ("15 7 1 1 + - / 3 * 2 1 1 + + -", 5, "postfix"),
        
        # Prefix expressions
        ("+ 3 4", 7, "prefix"),
        ("* + 3 4 2", 14, "prefix"),
        ("- * / 15 - 7 + 1 1 3 + 2 1", 5, "prefix"),
        
        # Infix expressions
        ("3 + 4", 7, "infix"),
        ("(3 + 4) * 2", 14, "infix"),
        ("3 + 4 * 2", 11, "infix"),
        ("(3 + 4) * (5 - 2)", 21, "infix"),
        
        # Simple expressions
        ("3+4", 7, "simple"),
        ("10-3", 7, "simple"),
        ("2*3+4", 10, "simple"),
    ]
    
    print("=== Testing Expression Evaluator ===")
    
    for expression, expected, eval_type in test_cases:
        print(f"\n--- {eval_type.upper()}: '{expression}' ---")
        print(f"Expected: {expected}")
        
        try:
            if eval_type == "postfix":
                result = evaluator.evaluate_postfix(expression)
            elif eval_type == "prefix":
                result = evaluator.evaluate_prefix(expression)
            elif eval_type == "infix":
                result = evaluator.evaluate_infix(expression)
            elif eval_type == "simple":
                result = evaluator.evaluate_simple(expression)
            
            status = "✓" if abs(result - expected) < 1e-9 else "✗"
            print(f"Result: {result} {status}")
            
        except Exception as e:
            print(f"Error: {str(e)}")


def demonstrate_infix_to_postfix():
    """Demonstrate infix to postfix conversion"""
    print("\n=== Infix to Postfix Conversion Demo ===")
    
    evaluator = BasicExpressionEvaluator()
    
    test_expressions = [
        "3 + 4",
        "3 + 4 * 2",
        "(3 + 4) * 2",
        "3 + 4 * 2 - 1",
        "(3 + 4) * (5 - 2)",
        "3 ** 2 + 4",
    ]
    
    for expr in test_expressions:
        try:
            postfix = evaluator.infix_to_postfix(expr)
            infix_result = evaluator.evaluate_infix(expr)
            postfix_result = evaluator.evaluate_postfix(postfix)
            
            print(f"Infix:   {expr}")
            print(f"Postfix: {postfix}")
            print(f"Result:  {infix_result} (infix) = {postfix_result} (postfix)")
            print()
            
        except Exception as e:
            print(f"Error with '{expr}': {str(e)}")


def demonstrate_step_by_step_evaluation():
    """Demonstrate step-by-step expression evaluation"""
    print("\n=== Step-by-Step Evaluation Demo ===")
    
    expression = "3 4 + 2 *"
    print(f"Evaluating postfix expression: '{expression}'")
    
    stack = []
    tokens = expression.split()
    
    for i, token in enumerate(tokens):
        print(f"\nStep {i+1}: Processing '{token}'")
        print(f"  Stack before: {stack}")
        
        if token in ['+', '-', '*', '/', '//', '%', '**']:
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = a / b
                
                stack.append(result)
                print(f"  Operation: {a} {token} {b} = {result}")
            else:
                print(f"  Error: Not enough operands for {token}")
                break
        else:
            # Number
            stack.append(float(token))
            print(f"  Pushed number: {token}")
        
        print(f"  Stack after: {stack}")
    
    print(f"\nFinal result: {stack[0] if stack else 'Error'}")


def test_error_handling():
    """Test error handling in expression evaluation"""
    print("\n=== Error Handling Tests ===")
    
    evaluator = BasicExpressionEvaluator()
    
    error_cases = [
        ("3 +", "postfix", "Insufficient operands"),
        ("3 4 + +", "postfix", "Too many operators"),
        ("3 4 5", "postfix", "Too many operands"),
        ("(3 + 4", "infix", "Mismatched parentheses"),
        ("3 + 4)", "infix", "Mismatched parentheses"),
        ("3 + + 4", "infix", "Invalid operator sequence"),
        ("", "simple", "Empty expression"),
        ("3 / 0", "infix", "Division by zero"),
    ]
    
    for expression, eval_type, expected_error in error_cases:
        print(f"\nTesting: '{expression}' ({eval_type})")
        print(f"Expected error: {expected_error}")
        
        try:
            if eval_type == "postfix":
                result = evaluator.evaluate_postfix(expression)
                print(f"Unexpected success: {result}")
            elif eval_type == "infix":
                result = evaluator.evaluate_infix(expression)
                print(f"Unexpected success: {result}")
            elif eval_type == "simple":
                result = evaluator.evaluate_simple(expression)
                print(f"Unexpected success: {result}")
                
        except Exception as e:
            print(f"Caught error: {type(e).__name__}: {str(e)}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    evaluator = BasicExpressionEvaluator()
    
    # Application 1: Calculator implementation
    print("1. Simple Calculator:")
    
    calculator_expressions = [
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "10 / 2 + 3",
        "2 ** 3 + 1",
    ]
    
    for expr in calculator_expressions:
        try:
            result = evaluator.evaluate_infix(expr)
            print(f"   {expr} = {result}")
        except Exception as e:
            print(f"   {expr} -> Error: {e}")
    
    # Application 2: Formula evaluation
    print(f"\n2. Formula Evaluation:")
    
    # Simple physics formulas (with substituted values)
    formulas = [
        ("Distance: d = v * t", "50 * 2", "v=50 m/s, t=2s"),
        ("Area of circle: A = π * r²", "3.14159 * 5 ** 2", "r=5"),
        ("Quadratic: ax² + bx + c", "2 * 3 ** 2 + 5 * 3 + 1", "a=2, b=5, c=1, x=3"),
    ]
    
    for description, expression, variables in formulas:
        try:
            result = evaluator.evaluate_infix(expression)
            print(f"   {description}")
            print(f"   Expression: {expression} ({variables})")
            print(f"   Result: {result}")
            print()
        except Exception as e:
            print(f"   Error in {description}: {e}")
    
    # Application 3: Configuration file evaluation
    print(f"3. Configuration Value Calculation:")
    
    config_expressions = [
        ("memory_limit", "1024 * 1024 * 512", "512 MB in bytes"),
        ("timeout", "30 * 60", "30 minutes in seconds"),
        ("max_connections", "100 * 2", "Double the base limit"),
    ]
    
    for name, expression, description in config_expressions:
        try:
            result = evaluator.evaluate_infix(expression)
            print(f"   {name}: {expression} = {result} ({description})")
        except Exception as e:
            print(f"   Error calculating {name}: {e}")


def benchmark_evaluation_methods():
    """Benchmark different evaluation methods"""
    print("\n=== Performance Benchmark ===")
    
    import time
    
    evaluator = BasicExpressionEvaluator()
    
    # Test expressions
    expressions = [
        ("3 4 +", "postfix"),
        ("+ 3 4", "prefix"),
        ("3 + 4", "infix"),
        ("3+4", "simple"),
    ]
    
    n_iterations = 10000
    
    for expression, eval_type in expressions:
        start_time = time.time()
        
        for _ in range(n_iterations):
            try:
                if eval_type == "postfix":
                    evaluator.evaluate_postfix(expression)
                elif eval_type == "prefix":
                    evaluator.evaluate_prefix(expression)
                elif eval_type == "infix":
                    evaluator.evaluate_infix(expression)
                elif eval_type == "simple":
                    evaluator.evaluate_simple(expression)
            except:
                pass
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_iterations
        
        print(f"{eval_type:10} '{expression:10}' | Avg time: {avg_time*1000000:.2f} μs")


if __name__ == "__main__":
    test_expression_evaluator()
    demonstrate_infix_to_postfix()
    demonstrate_step_by_step_evaluation()
    test_error_handling()
    demonstrate_real_world_applications()
    benchmark_evaluation_methods()

"""
Expression Evaluation Basic demonstrates fundamental expression parsing
and evaluation techniques using stack-based approaches for postfix, prefix,
and infix expressions with comprehensive error handling and real-world applications.
"""