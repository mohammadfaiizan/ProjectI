"""
Stack Applications in Expression Evaluation and Parsing
=======================================================

Topics: Infix/Postfix/Prefix conversion, expression evaluation, parsing
Companies: Google, Amazon, Microsoft, Facebook, Apple, Compiler companies
Difficulty: Medium to Hard
Time Complexity: O(n) for most operations
Space Complexity: O(n) for stack storage
"""

from typing import List, Union, Dict, Optional
import re
import operator

class ExpressionEvaluator:
    
    def __init__(self):
        """Initialize with operator precedence and associativity"""
        self.precedence = {
            '+': 1, '-': 1,
            '*': 2, '/': 2, '%': 2,
            '^': 3, '**': 3,
            '(': 0, ')': 0
        }
        
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '^': operator.pow,
            '**': operator.pow
        }
        
        self.right_associative = {'^', '**'}
        self.evaluation_steps = []
    
    # ==========================================
    # 1. INFIX TO POSTFIX CONVERSION
    # ==========================================
    
    def infix_to_postfix(self, infix: str) -> str:
        """
        Convert infix expression to postfix using Shunting Yard algorithm
        
        Company: Google, Microsoft, Compiler design interviews
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: "3 + 4 * 2" → "3 4 2 * +"
        """
        stack = []
        postfix = []
        tokens = self.tokenize(infix)
        
        print(f"Converting infix to postfix: {infix}")
        print(f"Tokens: {tokens}")
        print("\nStep-by-step conversion:")
        
        for i, token in enumerate(tokens):
            print(f"\nStep {i+1}: Processing '{token}'")
            
            if self.is_operand(token):
                postfix.append(token)
                print(f"   Operand: Add '{token}' to output")
                print(f"   Output: {' '.join(postfix)}")
                
            elif token == '(':
                stack.append(token)
                print(f"   Left parenthesis: Push to stack")
                print(f"   Stack: {stack}")
                
            elif token == ')':
                print(f"   Right parenthesis: Pop until '('")
                while stack and stack[-1] != '(':
                    op = stack.pop()
                    postfix.append(op)
                    print(f"      Pop '{op}' to output")
                
                if stack and stack[-1] == '(':
                    stack.pop()  # Remove the '('
                    print(f"      Remove '(' from stack")
                
                print(f"   Output: {' '.join(postfix)}")
                print(f"   Stack: {stack}")
                
            else:  # Operator
                print(f"   Operator '{token}' (precedence: {self.precedence[token]})")
                
                while (stack and stack[-1] != '(' and
                       self.has_higher_precedence(stack[-1], token)):
                    op = stack.pop()
                    postfix.append(op)
                    print(f"      Pop higher precedence '{op}' to output")
                
                stack.append(token)
                print(f"      Push '{token}' to stack")
                print(f"   Output: {' '.join(postfix)}")
                print(f"   Stack: {stack}")
        
        # Pop remaining operators
        print(f"\nPopping remaining operators from stack:")
        while stack:
            op = stack.pop()
            postfix.append(op)
            print(f"   Pop '{op}' to output")
        
        result = ' '.join(postfix)
        print(f"\nFinal postfix expression: {result}")
        return result
    
    def has_higher_precedence(self, op1: str, op2: str) -> bool:
        """Check if op1 has higher precedence than op2"""
        prec1 = self.precedence.get(op1, 0)
        prec2 = self.precedence.get(op2, 0)
        
        if prec1 > prec2:
            return True
        elif prec1 == prec2:
            # Left associative operators
            return op2 not in self.right_associative
        else:
            return False
    
    # ==========================================
    # 2. INFIX TO PREFIX CONVERSION
    # ==========================================
    
    def infix_to_prefix(self, infix: str) -> str:
        """
        Convert infix expression to prefix notation
        
        Algorithm:
        1. Reverse the infix expression
        2. Replace '(' with ')' and vice versa
        3. Get postfix expression
        4. Reverse the postfix expression
        
        Time: O(n), Space: O(n)
        
        Example: "3 + 4 * 2" → "+ 3 * 4 2"
        """
        print(f"Converting infix to prefix: {infix}")
        
        # Step 1: Reverse the expression and swap parentheses
        tokens = self.tokenize(infix)
        reversed_tokens = []
        
        for token in reversed(tokens):
            if token == '(':
                reversed_tokens.append(')')
            elif token == ')':
                reversed_tokens.append('(')
            else:
                reversed_tokens.append(token)
        
        reversed_infix = ' '.join(reversed_tokens)
        print(f"Step 1: Reversed infix: {reversed_infix}")
        
        # Step 2: Get postfix of reversed expression
        postfix = self.infix_to_postfix_for_prefix(reversed_infix)
        print(f"Step 2: Postfix of reversed: {postfix}")
        
        # Step 3: Reverse the postfix to get prefix
        prefix_tokens = postfix.split()
        prefix = ' '.join(reversed(prefix_tokens))
        
        print(f"Step 3: Final prefix: {prefix}")
        return prefix
    
    def infix_to_postfix_for_prefix(self, infix: str) -> str:
        """Helper method for prefix conversion - modified precedence rules"""
        stack = []
        postfix = []
        tokens = infix.split()
        
        for token in tokens:
            if self.is_operand(token):
                postfix.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                if stack:
                    stack.pop()  # Remove '('
            else:  # Operator
                # For prefix, we need different associativity handling
                while (stack and stack[-1] != '(' and
                       self.precedence.get(stack[-1], 0) > self.precedence.get(token, 0)):
                    postfix.append(stack.pop())
                stack.append(token)
        
        while stack:
            postfix.append(stack.pop())
        
        return ' '.join(postfix)
    
    # ==========================================
    # 3. POSTFIX EXPRESSION EVALUATION
    # ==========================================
    
    def evaluate_postfix(self, postfix: str) -> float:
        """
        Evaluate postfix expression using stack
        
        Company: Amazon, Microsoft, Google
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: "3 4 2 * +" → 11
        """
        stack = []
        tokens = postfix.split()
        
        print(f"Evaluating postfix expression: {postfix}")
        print("\nStep-by-step evaluation:")
        
        for i, token in enumerate(tokens):
            print(f"\nStep {i+1}: Processing '{token}'")
            
            if self.is_operand(token):
                value = float(token)
                stack.append(value)
                print(f"   Operand: Push {value} to stack")
                print(f"   Stack: {stack}")
                
            else:  # Operator
                if len(stack) < 2:
                    raise ValueError(f"Insufficient operands for operator '{token}'")
                
                # Pop two operands (order matters!)
                operand2 = stack.pop()
                operand1 = stack.pop()
                
                print(f"   Operator '{token}': Pop {operand2} and {operand1}")
                
                # Perform operation
                if token in self.operators:
                    result = self.operators[token](operand1, operand2)
                else:
                    raise ValueError(f"Unknown operator: {token}")
                
                stack.append(result)
                print(f"   Calculate: {operand1} {token} {operand2} = {result}")
                print(f"   Stack: {stack}")
        
        if len(stack) != 1:
            raise ValueError("Invalid postfix expression")
        
        final_result = stack[0]
        print(f"\nFinal result: {final_result}")
        return final_result
    
    # ==========================================
    # 4. PREFIX EXPRESSION EVALUATION
    # ==========================================
    
    def evaluate_prefix(self, prefix: str) -> float:
        """
        Evaluate prefix expression using stack
        
        Algorithm: Process from right to left
        
        Time: O(n), Space: O(n)
        
        Example: "+ 3 * 4 2" → 11
        """
        stack = []
        tokens = prefix.split()
        
        print(f"Evaluating prefix expression: {prefix}")
        print("Processing from right to left:")
        
        for i, token in enumerate(reversed(tokens)):
            step_num = len(tokens) - i
            print(f"\nStep {step_num}: Processing '{token}'")
            
            if self.is_operand(token):
                value = float(token)
                stack.append(value)
                print(f"   Operand: Push {value} to stack")
                print(f"   Stack: {stack}")
                
            else:  # Operator
                if len(stack) < 2:
                    raise ValueError(f"Insufficient operands for operator '{token}'")
                
                # For prefix, first pop is first operand
                operand1 = stack.pop()
                operand2 = stack.pop()
                
                print(f"   Operator '{token}': Pop {operand1} and {operand2}")
                
                if token in self.operators:
                    result = self.operators[token](operand1, operand2)
                else:
                    raise ValueError(f"Unknown operator: {token}")
                
                stack.append(result)
                print(f"   Calculate: {operand1} {token} {operand2} = {result}")
                print(f"   Stack: {stack}")
        
        if len(stack) != 1:
            raise ValueError("Invalid prefix expression")
        
        final_result = stack[0]
        print(f"\nFinal result: {final_result}")
        return final_result
    
    # ==========================================
    # 5. INFIX EXPRESSION EVALUATION
    # ==========================================
    
    def evaluate_infix(self, infix: str) -> float:
        """
        Evaluate infix expression directly using two stacks
        
        Uses operator stack and operand stack
        
        Company: Calculator implementation, expression parsers
        Difficulty: Medium
        Time: O(n), Space: O(n)
        """
        operand_stack = []
        operator_stack = []
        tokens = self.tokenize(infix)
        
        print(f"Evaluating infix expression: {infix}")
        print(f"Tokens: {tokens}")
        print("\nUsing two stacks - operands and operators:")
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            print(f"\nStep {i+1}: Processing '{token}'")
            
            if self.is_operand(token):
                operand_stack.append(float(token))
                print(f"   Operand: Push {token} to operand stack")
                print(f"   Operand stack: {operand_stack}")
                
            elif token == '(':
                operator_stack.append(token)
                print(f"   Left parenthesis: Push to operator stack")
                print(f"   Operator stack: {operator_stack}")
                
            elif token == ')':
                print(f"   Right parenthesis: Evaluate until '('")
                while operator_stack and operator_stack[-1] != '(':
                    self.apply_operator(operand_stack, operator_stack)
                operator_stack.pop()  # Remove '('
                print(f"   After evaluation - Operand stack: {operand_stack}")
                print(f"   Operator stack: {operator_stack}")
                
            else:  # Operator
                print(f"   Operator '{token}' (precedence: {self.precedence[token]})")
                
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       self.has_higher_precedence(operator_stack[-1], token)):
                    print(f"      Applying higher precedence operator")
                    self.apply_operator(operand_stack, operator_stack)
                
                operator_stack.append(token)
                print(f"   Push '{token}' to operator stack")
                print(f"   Operand stack: {operand_stack}")
                print(f"   Operator stack: {operator_stack}")
            
            i += 1
        
        # Apply remaining operators
        print(f"\nApplying remaining operators:")
        while operator_stack:
            self.apply_operator(operand_stack, operator_stack)
        
        if len(operand_stack) != 1:
            raise ValueError("Invalid infix expression")
        
        result = operand_stack[0]
        print(f"\nFinal result: {result}")
        return result
    
    def apply_operator(self, operand_stack: List[float], operator_stack: List[str]) -> None:
        """Apply operator to top two operands"""
        if len(operand_stack) < 2 or not operator_stack:
            raise ValueError("Invalid expression")
        
        operator = operator_stack.pop()
        operand2 = operand_stack.pop()
        operand1 = operand_stack.pop()
        
        result = self.operators[operator](operand1, operand2)
        operand_stack.append(result)
        
        print(f"      Applied: {operand1} {operator} {operand2} = {result}")
        print(f"      Operand stack: {operand_stack}")
        print(f"      Operator stack: {operator_stack}")
    
    # ==========================================
    # 6. ADVANCED EXPRESSION PARSING
    # ==========================================
    
    def evaluate_with_variables(self, expression: str, variables: Dict[str, float]) -> float:
        """
        Evaluate expression with variables
        
        Company: Calculator apps, spreadsheet software
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: "x + y * 2" with {"x": 3, "y": 4} → 11
        """
        print(f"Evaluating expression with variables: {expression}")
        print(f"Variables: {variables}")
        
        # Replace variables with their values
        tokens = self.tokenize(expression)
        substituted_tokens = []
        
        for token in tokens:
            if token in variables:
                substituted_tokens.append(str(variables[token]))
                print(f"   Substituted '{token}' with {variables[token]}")
            else:
                substituted_tokens.append(token)
        
        substituted_expression = ' '.join(substituted_tokens)
        print(f"After substitution: {substituted_expression}")
        
        return self.evaluate_infix(substituted_expression)
    
    def parse_function_calls(self, expression: str) -> float:
        """
        Parse and evaluate expressions with function calls
        
        Supports: sin, cos, sqrt, abs, etc.
        
        Example: "sin(30) + sqrt(16)" → result
        """
        import math
        
        functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'abs': abs,
            'log': math.log,
            'exp': math.exp
        }
        
        print(f"Parsing expression with functions: {expression}")
        
        # Simple function call parsing (simplified implementation)
        # In practice, would use more sophisticated parsing
        
        # Replace function calls with their values
        import re
        
        def replace_function(match):
            func_name = match.group(1)
            arg = float(match.group(2))
            
            if func_name in functions:
                if func_name in ['sin', 'cos', 'tan']:
                    # Convert degrees to radians
                    arg = math.radians(arg)
                
                result = functions[func_name](arg)
                print(f"   Evaluated {func_name}({match.group(2)}) = {result}")
                return str(result)
            else:
                raise ValueError(f"Unknown function: {func_name}")
        
        # Pattern to match function(argument)
        pattern = r'(\w+)\(([^)]+)\)'
        processed = re.sub(pattern, replace_function, expression)
        
        print(f"After function evaluation: {processed}")
        return self.evaluate_infix(processed)
    
    # ==========================================
    # 7. UTILITY METHODS
    # ==========================================
    
    def tokenize(self, expression: str) -> List[str]:
        """Tokenize expression into operands and operators"""
        # Remove spaces and split into tokens
        expression = expression.replace(' ', '')
        tokens = []
        current_token = ""
        
        i = 0
        while i < len(expression):
            char = expression[i]
            
            if char.isdigit() or char == '.':
                current_token += char
            elif char.isalpha():  # Variable or function name
                current_token += char
            else:  # Operator or parenthesis
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                
                # Handle multi-character operators
                if char == '*' and i + 1 < len(expression) and expression[i + 1] == '*':
                    tokens.append('**')
                    i += 1
                else:
                    tokens.append(char)
            
            i += 1
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def is_operand(self, token: str) -> bool:
        """Check if token is an operand (number or variable)"""
        try:
            float(token)
            return True
        except ValueError:
            # Check if it's a variable (alphabetic)
            return token.isalpha()
    
    def validate_expression(self, expression: str) -> bool:
        """
        Validate if expression is syntactically correct
        
        Checks for:
        - Balanced parentheses
        - Valid operator placement
        - Valid operand format
        """
        tokens = self.tokenize(expression)
        paren_count = 0
        last_was_operator = True  # Start as if last was operator
        
        print(f"Validating expression: {expression}")
        
        for i, token in enumerate(tokens):
            if token == '(':
                paren_count += 1
                last_was_operator = True
            elif token == ')':
                paren_count -= 1
                if paren_count < 0:
                    print(f"   ✗ Unmatched closing parenthesis at position {i}")
                    return False
                last_was_operator = False
            elif self.is_operand(token):
                if not last_was_operator and i > 0:
                    print(f"   ✗ Two consecutive operands at position {i}")
                    return False
                last_was_operator = False
            elif token in self.operators:
                if last_was_operator and token not in ['+', '-']:
                    print(f"   ✗ Invalid operator placement at position {i}")
                    return False
                last_was_operator = True
            else:
                print(f"   ✗ Unknown token '{token}' at position {i}")
                return False
        
        if paren_count != 0:
            print(f"   ✗ Unmatched parentheses (count: {paren_count})")
            return False
        
        if last_was_operator:
            print(f"   ✗ Expression ends with operator")
            return False
        
        print(f"   ✓ Expression is valid")
        return True


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_expression_evaluation():
    """Demonstrate all expression evaluation capabilities"""
    print("=== EXPRESSION EVALUATION DEMONSTRATION ===\n")
    
    evaluator = ExpressionEvaluator()
    
    # 1. Infix to Postfix conversion
    print("=== INFIX TO POSTFIX CONVERSION ===")
    test_expressions = [
        "3 + 4 * 2",
        "( 3 + 4 ) * 2",
        "3 + 4 * 2 / ( 1 - 5 ) ^ 2",
        "a + b * c - d / e"
    ]
    
    for expr in test_expressions:
        postfix = evaluator.infix_to_postfix(expr)
        print()
    
    print("\n" + "="*60 + "\n")
    
    # 2. Infix to Prefix conversion
    print("=== INFIX TO PREFIX CONVERSION ===")
    for expr in test_expressions[:2]:  # Test first two
        prefix = evaluator.infix_to_prefix(expr)
        print()
    
    print("\n" + "="*60 + "\n")
    
    # 3. Postfix evaluation
    print("=== POSTFIX EXPRESSION EVALUATION ===")
    postfix_expressions = [
        "3 4 2 * +",
        "3 4 + 2 *",
        "15 7 1 1 + - / 3 * 2 1 1 + + -"
    ]
    
    for expr in postfix_expressions:
        try:
            result = evaluator.evaluate_postfix(expr)
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    print("\n" + "="*60 + "\n")
    
    # 4. Prefix evaluation
    print("=== PREFIX EXPRESSION EVALUATION ===")
    prefix_expressions = [
        "+ 3 * 4 2",
        "* + 3 4 2",
        "- + 15 / 7 + 1 1 * 3 + 2 + 1 1"
    ]
    
    for expr in prefix_expressions:
        try:
            result = evaluator.evaluate_prefix(expr)
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    print("\n" + "="*60 + "\n")
    
    # 5. Infix evaluation
    print("=== INFIX EXPRESSION EVALUATION ===")
    infix_expressions = [
        "3 + 4 * 2",
        "( 3 + 4 ) * 2",
        "10 - 2 * 3 + 8 / 4"
    ]
    
    for expr in infix_expressions:
        try:
            result = evaluator.evaluate_infix(expr)
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    print("\n" + "="*60 + "\n")
    
    # 6. Variables and functions
    print("=== ADVANCED FEATURES ===")
    
    print("1. Expression with variables:")
    try:
        result = evaluator.evaluate_with_variables("x + y * 2", {"x": 3, "y": 4})
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    print("2. Expression validation:")
    test_validation = [
        "3 + 4 * 2",      # Valid
        "3 + + 4",        # Invalid
        "( 3 + 4 )) * 2", # Invalid
        "3 + 4 *"         # Invalid
    ]
    
    for expr in test_validation:
        evaluator.validate_expression(expr)
        print()


if __name__ == "__main__":
    demonstrate_expression_evaluation()
    
    print("=== EXPRESSION EVALUATION MASTERY GUIDE ===")
    
    print("\n🎯 NOTATION TYPES:")
    print("• Infix: Operators between operands (3 + 4)")
    print("• Postfix (RPN): Operators after operands (3 4 +)")
    print("• Prefix (Polish): Operators before operands (+ 3 4)")
    
    print("\n📋 CONVERSION ALGORITHMS:")
    print("• Infix → Postfix: Shunting Yard algorithm")
    print("• Infix → Prefix: Reverse, convert to postfix, reverse")
    print("• Handle operator precedence and associativity")
    print("• Manage parentheses correctly")
    
    print("\n⚡ EVALUATION STRATEGIES:")
    print("• Postfix: Single stack, left to right")
    print("• Prefix: Single stack, right to left")
    print("• Infix: Two stacks (operands and operators)")
    print("• Handle edge cases and error conditions")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Tokenize expressions properly")
    print("• Define operator precedence clearly")
    print("• Handle associativity rules")
    print("• Validate expressions before evaluation")
    print("• Support parentheses and functions")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Calculator applications")
    print("• Spreadsheet formula evaluation")
    print("• Compiler expression parsing")
    print("• Mathematical software")
    print("• Scripting language interpreters")

