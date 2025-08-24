"""
150. Evaluate Reverse Polish Notation - Multiple Approaches
Difficulty: Medium

Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, and /. Each operand may be an integer or another expression.

Note that division between two integers should truncate toward zero.

It is guaranteed that the given RPN expression is always valid. That means the expression would always evaluate to a result, and there will not be any division by zero operation.
"""

from typing import List, Union
import operator

class EvaluateRPN:
    """Multiple approaches to evaluate Reverse Polish Notation"""
    
    def evalRPN_stack_approach(self, tokens: List[str]) -> int:
        """
        Approach 1: Classic Stack Approach
        
        Use stack to evaluate RPN expression.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        operators = {'+', '-', '*', '/'}
        
        for token in tokens:
            if token in operators:
                # Pop two operands
                b = stack.pop()
                a = stack.pop()
                
                # Perform operation
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    # Truncate toward zero
                    result = int(a / b)
                
                stack.append(result)
            else:
                # Push operand
                stack.append(int(token))
        
        return stack[0]
    
    def evalRPN_operator_mapping(self, tokens: List[str]) -> int:
        """
        Approach 2: Operator Mapping with Lambda Functions
        
        Use dictionary mapping for operators.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        operations = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: int(a / b),  # Truncate toward zero
        }
        
        for token in tokens:
            if token in operations:
                b = stack.pop()
                a = stack.pop()
                result = operations[token](a, b)
                stack.append(result)
            else:
                stack.append(int(token))
        
        return stack[0]
    
    def evalRPN_operator_module(self, tokens: List[str]) -> int:
        """
        Approach 3: Using operator module
        
        Use Python's operator module for operations.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        operations = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': lambda a, b: int(operator.truediv(a, b)),
        }
        
        for token in tokens:
            if token in operations:
                b = stack.pop()
                a = stack.pop()
                result = operations[token](a, b)
                stack.append(result)
            else:
                stack.append(int(token))
        
        return stack[0]
    
    def evalRPN_recursive_approach(self, tokens: List[str]) -> int:
        """
        Approach 4: Recursive Approach
        
        Use recursion to evaluate RPN expression.
        
        Time: O(n), Space: O(n)
        """
        def evaluate(index: int) -> tuple:
            """Returns (result, next_index)"""
            token = tokens[index]
            
            if token in {'+', '-', '*', '/'}:
                # Get first operand
                a, next_idx = evaluate(index + 1)
                # Get second operand
                b, next_idx = evaluate(next_idx)
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = int(a / b)
                
                return result, next_idx
            else:
                # It's a number
                return int(token), index + 1
        
        # Reverse tokens for recursive evaluation
        tokens_reversed = tokens[::-1]
        result, _ = evaluate(0)
        return result
    
    def evalRPN_two_stacks(self, tokens: List[str]) -> int:
        """
        Approach 5: Two Stacks Approach
        
        Use separate stacks for operands and operators.
        
        Time: O(n), Space: O(n)
        """
        operand_stack = []
        operator_stack = []
        
        for token in tokens:
            if token in {'+', '-', '*', '/'}:
                operator_stack.append(token)
                
                # If we have at least 2 operands and 1 operator, evaluate
                if len(operand_stack) >= 2:
                    b = operand_stack.pop()
                    a = operand_stack.pop()
                    op = operator_stack.pop()
                    
                    if op == '+':
                        result = a + b
                    elif op == '-':
                        result = a - b
                    elif op == '*':
                        result = a * b
                    elif op == '/':
                        result = int(a / b)
                    
                    operand_stack.append(result)
            else:
                operand_stack.append(int(token))
        
        return operand_stack[0]
    
    def evalRPN_iterative_optimized(self, tokens: List[str]) -> int:
        """
        Approach 6: Iterative with Optimized Memory
        
        Optimize memory usage by reusing stack space.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for token in tokens:
            if token not in {'+', '-', '*', '/'}:
                stack.append(int(token))
            else:
                # Pop two operands and compute result
                second = stack.pop()
                first = stack.pop()
                
                if token == '+':
                    stack.append(first + second)
                elif token == '-':
                    stack.append(first - second)
                elif token == '*':
                    stack.append(first * second)
                else:  # token == '/'
                    # Handle division with truncation toward zero
                    stack.append(int(first / second))
        
        return stack[0]
    
    def evalRPN_functional_approach(self, tokens: List[str]) -> int:
        """
        Approach 7: Functional Programming Approach
        
        Use reduce and functional programming concepts.
        
        Time: O(n), Space: O(n)
        """
        from functools import reduce
        
        def process_token(stack, token):
            if token in {'+', '-', '*', '/'}:
                b = stack.pop()
                a = stack.pop()
                
                operations = {
                    '+': lambda x, y: x + y,
                    '-': lambda x, y: x - y,
                    '*': lambda x, y: x * y,
                    '/': lambda x, y: int(x / y),
                }
                
                result = operations[token](a, b)
                stack.append(result)
            else:
                stack.append(int(token))
            
            return stack
        
        result_stack = reduce(process_token, tokens, [])
        return result_stack[0]

def test_evaluate_rpn():
    """Test RPN evaluation algorithms"""
    solver = EvaluateRPN()
    
    test_cases = [
        (["2","1","+","3","*"], 9, "Simple expression: (2+1)*3"),
        (["4","13","5","/","+"], 6, "Division: 4+(13/5)"),
        (["10","6","9","3","+","-11","*","/","*","17","+","5","+"], 22, "Complex expression"),
        (["4","3","-"], 1, "Simple subtraction"),
        (["2","3","+"], 5, "Simple addition"),
        (["15","7","1","1","+","/","/","3","*","2","1","1","+","+","-"], 5, "Very complex"),
        (["-1","2","+"], 1, "Negative number"),
        (["3","11","+","5","-"], 9, "Multiple operations"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.evalRPN_stack_approach),
        ("Operator Mapping", solver.evalRPN_operator_mapping),
        ("Operator Module", solver.evalRPN_operator_module),
        ("Two Stacks", solver.evalRPN_two_stacks),
        ("Iterative Optimized", solver.evalRPN_iterative_optimized),
        ("Functional Approach", solver.evalRPN_functional_approach),
    ]
    
    print("=== Testing Evaluate RPN ===")
    
    for tokens, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Tokens: {tokens}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(tokens)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_rpn_evaluation():
    """Demonstrate step-by-step RPN evaluation"""
    print("\n=== RPN Evaluation Step-by-Step Demo ===")
    
    tokens = ["15", "7", "1", "1", "+", "/", "/", "3", "*", "2", "1", "1", "+", "+", "-"]
    print(f"Expression: {' '.join(tokens)}")
    
    stack = []
    
    for i, token in enumerate(tokens):
        if token in {'+', '-', '*', '/'}:
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
            print(f"Step {i+1:2d}: {token} -> pop {b}, {a} -> push {result} | Stack: {stack}")
        else:
            stack.append(int(token))
            print(f"Step {i+1:2d}: {token} -> push {token} | Stack: {stack}")
    
    print(f"\nFinal result: {stack[0]}")

def benchmark_rpn_evaluation():
    """Benchmark different RPN evaluation approaches"""
    import time
    import random
    
    # Generate test expressions
    def generate_rpn_expression(length: int) -> List[str]:
        """Generate random valid RPN expression"""
        tokens = []
        operand_count = 0
        
        for _ in range(length):
            if operand_count < 2 or random.random() < 0.7:
                # Add operand
                tokens.append(str(random.randint(1, 100)))
                operand_count += 1
            else:
                # Add operator
                tokens.append(random.choice(['+', '-', '*', '/']))
                operand_count -= 1  # Two operands become one result
        
        # Ensure we end with exactly one value
        while operand_count > 1:
            tokens.append(random.choice(['+', '-', '*', '/']))
            operand_count -= 1
        
        return tokens
    
    algorithms = [
        ("Stack Approach", EvaluateRPN().evalRPN_stack_approach),
        ("Operator Mapping", EvaluateRPN().evalRPN_operator_mapping),
        ("Iterative Optimized", EvaluateRPN().evalRPN_iterative_optimized),
    ]
    
    expression_lengths = [100, 500, 1000]
    
    print("\n=== RPN Evaluation Performance Benchmark ===")
    
    for length in expression_lengths:
        print(f"\n--- Expression Length: {length} ---")
        expressions = [generate_rpn_expression(length) for _ in range(10)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            for expr in expressions:
                try:
                    alg_func(expr)
                except:
                    pass  # Skip invalid expressions
            
            end_time = time.time()
            
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    test_evaluate_rpn()
    demonstrate_rpn_evaluation()
    benchmark_rpn_evaluation()

"""
Evaluate Reverse Polish Notation demonstrates multiple approaches
including stack-based evaluation, operator mapping, recursive methods,
and functional programming techniques for expression evaluation.
"""
