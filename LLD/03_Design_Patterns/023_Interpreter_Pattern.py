"""
INTERPRETER PATTERN - Behavioral Design Pattern
===============================================

Problem Statement:
Implement the Interpreter pattern to define a representation for a language's
grammar and provide an interpreter that uses the representation to interpret
sentences in the language:
- Grammar representation and parsing
- Domain-specific language (DSL) implementation
- Expression evaluation and interpretation
- Rule-based systems and business logic
- Query language and filter systems

Learning Objectives:
- Understand Interpreter vs Visitor pattern differences
- Implement grammar rules and abstract syntax trees
- Design domain-specific languages
- Handle context and variable scoping
- Create extensible interpretation systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import re
import json
import math
from datetime import datetime, timedelta
from enum import Enum


# ============================================================================
# INTERPRETER INTERFACE
# ============================================================================

class Context:
    """Context for interpretation with variables and functions."""
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.functions: Dict[str, Callable] = {}
        self.execution_stack: List[Dict[str, Any]] = []
        self.debug_mode = False
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set variable value."""
        self.variables[name] = value
        if self.debug_mode:
            print(f"Set variable: {name} = {value}")
    
    def get_variable(self, name: str) -> Any:
        """Get variable value."""
        if name in self.variables:
            return self.variables[name]
        else:
            raise NameError(f"Variable '{name}' is not defined")
    
    def has_variable(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self.variables
    
    def set_function(self, name: str, func: Callable) -> None:
        """Set function."""
        self.functions[name] = func
        if self.debug_mode:
            print(f"Registered function: {name}")
    
    def call_function(self, name: str, *args) -> Any:
        """Call function."""
        if name in self.functions:
            result = self.functions[name](*args)
            if self.debug_mode:
                print(f"Called function: {name}({args}) = {result}")
            return result
        else:
            raise NameError(f"Function '{name}' is not defined")
    
    def push_scope(self, scope_name: str = "anonymous") -> None:
        """Push new scope onto execution stack."""
        scope = {
            'name': scope_name,
            'variables': self.variables.copy(),
            'timestamp': datetime.now().isoformat()
        }
        self.execution_stack.append(scope)
        if self.debug_mode:
            print(f"Pushed scope: {scope_name}")
    
    def pop_scope(self) -> Optional[Dict[str, Any]]:
        """Pop scope from execution stack."""
        if self.execution_stack:
            scope = self.execution_stack.pop()
            self.variables = scope['variables']
            if self.debug_mode:
                print(f"Popped scope: {scope['name']}")
            return scope
        return None
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get context information."""
        return {
            'variables': dict(self.variables),
            'functions': list(self.functions.keys()),
            'stack_depth': len(self.execution_stack),
            'debug_mode': self.debug_mode
        }


class Expression(ABC):
    """Abstract expression interface."""
    
    @abstractmethod
    def interpret(self, context: Context) -> Any:
        """Interpret expression in given context."""
        pass
    
    @abstractmethod
    def get_expression_info(self) -> Dict[str, Any]:
        """Get expression information."""
        pass
    
    def __str__(self) -> str:
        """String representation of expression."""
        return f"{self.__class__.__name__}()"


# ============================================================================
# MATHEMATICAL EXPRESSION INTERPRETER
# ============================================================================

class NumberExpression(Expression):
    """Terminal expression for numbers."""
    
    def __init__(self, value: Union[int, float]):
        self.value = value
    
    def interpret(self, context: Context) -> Union[int, float]:
        """Return the number value."""
        return self.value
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'Number',
            'value': self.value,
            'is_terminal': True
        }
    
    def __str__(self) -> str:
        return str(self.value)


class VariableExpression(Expression):
    """Terminal expression for variables."""
    
    def __init__(self, name: str):
        self.name = name
    
    def interpret(self, context: Context) -> Any:
        """Get variable value from context."""
        return context.get_variable(self.name)
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'Variable',
            'name': self.name,
            'is_terminal': True
        }
    
    def __str__(self) -> str:
        return self.name


class BinaryOperationExpression(Expression):
    """Non-terminal expression for binary operations."""
    
    def __init__(self, left: Expression, operator: str, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right
    
    def interpret(self, context: Context) -> Union[int, float, bool]:
        """Interpret binary operation."""
        left_value = self.left.interpret(context)
        right_value = self.right.interpret(context)
        
        if self.operator == '+':
            return left_value + right_value
        elif self.operator == '-':
            return left_value - right_value
        elif self.operator == '*':
            return left_value * right_value
        elif self.operator == '/':
            if right_value == 0:
                raise ValueError("Division by zero")
            return left_value / right_value
        elif self.operator == '**' or self.operator == '^':
            return left_value ** right_value
        elif self.operator == '%':
            return left_value % right_value
        elif self.operator == '==':
            return left_value == right_value
        elif self.operator == '!=':
            return left_value != right_value
        elif self.operator == '<':
            return left_value < right_value
        elif self.operator == '<=':
            return left_value <= right_value
        elif self.operator == '>':
            return left_value > right_value
        elif self.operator == '>=':
            return left_value >= right_value
        elif self.operator == 'and':
            return left_value and right_value
        elif self.operator == 'or':
            return left_value or right_value
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'BinaryOperation',
            'operator': self.operator,
            'left': self.left.get_expression_info(),
            'right': self.right.get_expression_info(),
            'is_terminal': False
        }
    
    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"


class UnaryOperationExpression(Expression):
    """Non-terminal expression for unary operations."""
    
    def __init__(self, operator: str, operand: Expression):
        self.operator = operator
        self.operand = operand
    
    def interpret(self, context: Context) -> Union[int, float, bool]:
        """Interpret unary operation."""
        operand_value = self.operand.interpret(context)
        
        if self.operator == '-':
            return -operand_value
        elif self.operator == '+':
            return +operand_value
        elif self.operator == 'not':
            return not operand_value
        else:
            raise ValueError(f"Unknown unary operator: {self.operator}")
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'UnaryOperation',
            'operator': self.operator,
            'operand': self.operand.get_expression_info(),
            'is_terminal': False
        }
    
    def __str__(self) -> str:
        return f"{self.operator}{self.operand}"


class FunctionCallExpression(Expression):
    """Non-terminal expression for function calls."""
    
    def __init__(self, function_name: str, arguments: List[Expression]):
        self.function_name = function_name
        self.arguments = arguments
    
    def interpret(self, context: Context) -> Any:
        """Interpret function call."""
        # Evaluate arguments
        arg_values = [arg.interpret(context) for arg in self.arguments]
        
        # Call function
        return context.call_function(self.function_name, *arg_values)
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'FunctionCall',
            'function_name': self.function_name,
            'arguments': [arg.get_expression_info() for arg in self.arguments],
            'is_terminal': False
        }
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.function_name}({args_str})"


class AssignmentExpression(Expression):
    """Non-terminal expression for variable assignment."""
    
    def __init__(self, variable_name: str, value_expression: Expression):
        self.variable_name = variable_name
        self.value_expression = value_expression
    
    def interpret(self, context: Context) -> Any:
        """Interpret assignment."""
        value = self.value_expression.interpret(context)
        context.set_variable(self.variable_name, value)
        return value
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'Assignment',
            'variable_name': self.variable_name,
            'value_expression': self.value_expression.get_expression_info(),
            'is_terminal': False
        }
    
    def __str__(self) -> str:
        return f"{self.variable_name} = {self.value_expression}"


# ============================================================================
# SIMPLE EXPRESSION PARSER
# ============================================================================

class ExpressionParser:
    """Simple recursive descent parser for mathematical expressions."""
    
    def __init__(self):
        self.tokens = []
        self.current_token_index = 0
    
    def parse(self, expression_string: str) -> Expression:
        """Parse expression string into expression tree."""
        self.tokens = self._tokenize(expression_string)
        self.current_token_index = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
        
        result = self._parse_assignment()
        
        if self.current_token_index < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.current_token_index]}")
        
        return result
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize expression string."""
        # Simple tokenizer using regex
        token_pattern = r'(\d+\.?\d*|[a-zA-Z_][a-zA-Z0-9_]*|\*\*|==|!=|<=|>=|[+\-*/()=<>^%,]|\S)'
        tokens = re.findall(token_pattern, expression)
        
        # Filter out whitespace
        return [token for token in tokens if token.strip()]
    
    def _current_token(self) -> Optional[str]:
        """Get current token."""
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        return None
    
    def _consume_token(self, expected: str = None) -> str:
        """Consume current token."""
        if self.current_token_index >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.current_token_index]
        self.current_token_index += 1
        
        if expected and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")
        
        return token
    
    def _parse_assignment(self) -> Expression:
        """Parse assignment expression."""
        expr = self._parse_or()
        
        if self._current_token() == '=':
            if not isinstance(expr, VariableExpression):
                raise ValueError("Invalid assignment target")
            
            self._consume_token('=')
            value_expr = self._parse_assignment()
            return AssignmentExpression(expr.name, value_expr)
        
        return expr
    
    def _parse_or(self) -> Expression:
        """Parse logical OR expression."""
        expr = self._parse_and()
        
        while self._current_token() == 'or':
            operator = self._consume_token()
            right = self._parse_and()
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_and(self) -> Expression:
        """Parse logical AND expression."""
        expr = self._parse_equality()
        
        while self._current_token() == 'and':
            operator = self._consume_token()
            right = self._parse_equality()
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_equality(self) -> Expression:
        """Parse equality expression."""
        expr = self._parse_comparison()
        
        while self._current_token() in ['==', '!=']:
            operator = self._consume_token()
            right = self._parse_comparison()
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_comparison(self) -> Expression:
        """Parse comparison expression."""
        expr = self._parse_addition()
        
        while self._current_token() in ['<', '<=', '>', '>=']:
            operator = self._consume_token()
            right = self._parse_addition()
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_addition(self) -> Expression:
        """Parse addition/subtraction expression."""
        expr = self._parse_multiplication()
        
        while self._current_token() in ['+', '-']:
            operator = self._consume_token()
            right = self._parse_multiplication()
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_multiplication(self) -> Expression:
        """Parse multiplication/division expression."""
        expr = self._parse_power()
        
        while self._current_token() in ['*', '/', '%']:
            operator = self._consume_token()
            right = self._parse_power()
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_power(self) -> Expression:
        """Parse power expression."""
        expr = self._parse_unary()
        
        if self._current_token() in ['**', '^']:
            operator = self._consume_token()
            right = self._parse_power()  # Right associative
            expr = BinaryOperationExpression(expr, operator, right)
        
        return expr
    
    def _parse_unary(self) -> Expression:
        """Parse unary expression."""
        if self._current_token() in ['+', '-', 'not']:
            operator = self._consume_token()
            operand = self._parse_unary()
            return UnaryOperationExpression(operator, operand)
        
        return self._parse_primary()
    
    def _parse_primary(self) -> Expression:
        """Parse primary expression."""
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        # Number
        if re.match(r'\d+\.?\d*', token):
            self._consume_token()
            if '.' in token:
                return NumberExpression(float(token))
            else:
                return NumberExpression(int(token))
        
        # Variable or function call
        elif re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', token):
            name = self._consume_token()
            
            # Function call
            if self._current_token() == '(':
                self._consume_token('(')
                arguments = []
                
                if self._current_token() != ')':
                    arguments.append(self._parse_assignment())
                    
                    while self._current_token() == ',':
                        self._consume_token(',')
                        arguments.append(self._parse_assignment())
                
                self._consume_token(')')
                return FunctionCallExpression(name, arguments)
            
            # Variable
            else:
                return VariableExpression(name)
        
        # Parenthesized expression
        elif token == '(':
            self._consume_token('(')
            expr = self._parse_assignment()
            self._consume_token(')')
            return expr
        
        else:
            raise ValueError(f"Unexpected token: {token}")


# ============================================================================
# RULE-BASED SYSTEM INTERPRETER
# ============================================================================

class Rule:
    """Rule with condition and action."""
    
    def __init__(self, name: str, condition: Expression, action: Expression, priority: int = 0):
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority
        self.execution_count = 0
        self.last_executed = None
    
    def evaluate_condition(self, context: Context) -> bool:
        """Evaluate rule condition."""
        try:
            result = self.condition.interpret(context)
            return bool(result)
        except Exception as e:
            if context.debug_mode:
                print(f"Error evaluating condition for rule '{self.name}': {e}")
            return False
    
    def execute_action(self, context: Context) -> Any:
        """Execute rule action."""
        try:
            result = self.action.interpret(context)
            self.execution_count += 1
            self.last_executed = datetime.now()
            
            if context.debug_mode:
                print(f"Executed rule '{self.name}': {result}")
            
            return result
        except Exception as e:
            if context.debug_mode:
                print(f"Error executing action for rule '{self.name}': {e}")
            raise e
    
    def get_rule_info(self) -> Dict[str, Any]:
        """Get rule information."""
        return {
            'name': self.name,
            'priority': self.priority,
            'execution_count': self.execution_count,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None,
            'condition': str(self.condition),
            'action': str(self.action)
        }


class RuleEngine:
    """Rule-based system engine."""
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.context = Context()
        self.execution_log: List[Dict[str, Any]] = []
    
    def add_rule(self, rule: Rule) -> None:
        """Add rule to engine."""
        self.rules.append(rule)
        # Sort rules by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        print(f"Added rule: {rule.name} (priority: {rule.priority})")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                self.rules.remove(rule)
                print(f"Removed rule: {rule_name}")
                return True
        return False
    
    def execute_rules(self, max_iterations: int = 100) -> List[Dict[str, Any]]:
        """Execute all applicable rules."""
        executed_rules = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            rules_fired = False
            
            for rule in self.rules:
                if rule.evaluate_condition(self.context):
                    try:
                        result = rule.execute_action(self.context)
                        
                        execution_record = {
                            'iteration': iteration,
                            'rule_name': rule.name,
                            'result': result,
                            'timestamp': datetime.now().isoformat(),
                            'context_snapshot': self.context.get_context_info()
                        }
                        
                        executed_rules.append(execution_record)
                        self.execution_log.append(execution_record)
                        rules_fired = True
                        
                    except Exception as e:
                        error_record = {
                            'iteration': iteration,
                            'rule_name': rule.name,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        self.execution_log.append(error_record)
            
            # If no rules fired, we're done
            if not rules_fired:
                break
        
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached")
        
        return executed_rules
    
    def execute_single_rule(self, rule_name: str) -> Optional[Any]:
        """Execute specific rule if condition is met."""
        for rule in self.rules:
            if rule.name == rule_name:
                if rule.evaluate_condition(self.context):
                    return rule.execute_action(self.context)
                else:
                    print(f"Rule '{rule_name}' condition not met")
                    return None
        
        print(f"Rule '{rule_name}' not found")
        return None
    
    def get_applicable_rules(self) -> List[str]:
        """Get names of rules whose conditions are currently true."""
        applicable = []
        for rule in self.rules:
            if rule.evaluate_condition(self.context):
                applicable.append(rule.name)
        return applicable
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total_executions = sum(rule.execution_count for rule in self.rules)
        
        rule_stats = {}
        for rule in self.rules:
            rule_stats[rule.name] = {
                'executions': rule.execution_count,
                'priority': rule.priority,
                'last_executed': rule.last_executed.isoformat() if rule.last_executed else None
            }
        
        return {
            'total_rules': len(self.rules),
            'total_executions': total_executions,
            'execution_log_entries': len(self.execution_log),
            'rule_statistics': rule_stats,
            'context_variables': len(self.context.variables),
            'context_functions': len(self.context.functions)
        }


# ============================================================================
# QUERY LANGUAGE INTERPRETER
# ============================================================================

class QueryExpression(Expression):
    """Base class for query expressions."""
    
    @abstractmethod
    def filter_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on query."""
        pass


class FieldExpression(QueryExpression):
    """Expression for field access."""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def interpret(self, context: Context) -> str:
        """Return field name."""
        return self.field_name
    
    def filter_data(self, data: List[Dict[str, Any]]) -> List[Any]:
        """Extract field values from data."""
        return [item.get(self.field_name) for item in data]
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'Field',
            'field_name': self.field_name,
            'is_terminal': True
        }
    
    def __str__(self) -> str:
        return self.field_name


class ComparisonQueryExpression(QueryExpression):
    """Expression for field comparisons in queries."""
    
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value
    
    def interpret(self, context: Context) -> bool:
        """Interpret comparison (requires data context)."""
        # This would typically be used with data context
        return True
    
    def filter_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on comparison."""
        filtered = []
        
        for item in data:
            if self.field not in item:
                continue
            
            field_value = item[self.field]
            
            try:
                if self.operator == '==':
                    if field_value == self.value:
                        filtered.append(item)
                elif self.operator == '!=':
                    if field_value != self.value:
                        filtered.append(item)
                elif self.operator == '<':
                    if field_value < self.value:
                        filtered.append(item)
                elif self.operator == '<=':
                    if field_value <= self.value:
                        filtered.append(item)
                elif self.operator == '>':
                    if field_value > self.value:
                        filtered.append(item)
                elif self.operator == '>=':
                    if field_value >= self.value:
                        filtered.append(item)
                elif self.operator == 'contains':
                    if isinstance(field_value, str) and str(self.value) in field_value:
                        filtered.append(item)
                elif self.operator == 'startswith':
                    if isinstance(field_value, str) and field_value.startswith(str(self.value)):
                        filtered.append(item)
                elif self.operator == 'endswith':
                    if isinstance(field_value, str) and field_value.endswith(str(self.value)):
                        filtered.append(item)
            except (TypeError, ValueError):
                # Skip items where comparison fails
                continue
        
        return filtered
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'ComparisonQuery',
            'field': self.field,
            'operator': self.operator,
            'value': self.value,
            'is_terminal': True
        }
    
    def __str__(self) -> str:
        return f"{self.field} {self.operator} {self.value}"


class LogicalQueryExpression(QueryExpression):
    """Expression for logical operations in queries."""
    
    def __init__(self, left: QueryExpression, operator: str, right: QueryExpression):
        self.left = left
        self.operator = operator
        self.right = right
    
    def interpret(self, context: Context) -> bool:
        """Interpret logical operation."""
        return True  # Placeholder
    
    def filter_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on logical operation."""
        left_results = self.left.filter_data(data)
        right_results = self.right.filter_data(data)
        
        if self.operator == 'and':
            # Intersection
            left_ids = {id(item) for item in left_results}
            return [item for item in right_results if id(item) in left_ids]
        
        elif self.operator == 'or':
            # Union
            result_ids = set()
            results = []
            
            for item in left_results + right_results:
                item_id = id(item)
                if item_id not in result_ids:
                    result_ids.add(item_id)
                    results.append(item)
            
            return results
        
        else:
            raise ValueError(f"Unknown logical operator: {self.operator}")
    
    def get_expression_info(self) -> Dict[str, Any]:
        return {
            'type': 'LogicalQuery',
            'operator': self.operator,
            'left': self.left.get_expression_info(),
            'right': self.right.get_expression_info(),
            'is_terminal': False
        }
    
    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"


class QueryInterpreter:
    """Interpreter for simple query language."""
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
    
    def set_data(self, data: List[Dict[str, Any]]) -> None:
        """Set data to query."""
        self.data = data
        print(f"Loaded {len(data)} records for querying")
    
    def execute_query(self, query: QueryExpression) -> List[Dict[str, Any]]:
        """Execute query on data."""
        return query.filter_data(self.data)
    
    def parse_simple_query(self, query_string: str) -> QueryExpression:
        """Parse simple query string."""
        # Very basic query parser for demonstration
        # Format: "field operator value" or "query1 and/or query2"
        
        query_string = query_string.strip()
        
        # Check for logical operators
        if ' and ' in query_string:
            parts = query_string.split(' and ', 1)
            left = self.parse_simple_query(parts[0])
            right = self.parse_simple_query(parts[1])
            return LogicalQueryExpression(left, 'and', right)
        
        elif ' or ' in query_string:
            parts = query_string.split(' or ', 1)
            left = self.parse_simple_query(parts[0])
            right = self.parse_simple_query(parts[1])
            return LogicalQueryExpression(left, 'or', right)
        
        # Parse comparison
        operators = ['>=', '<=', '!=', '==', '>', '<', 'contains', 'startswith', 'endswith']
        
        for op in operators:
            if f' {op} ' in query_string:
                parts = query_string.split(f' {op} ', 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # Try to convert value to appropriate type
                    try:
                        if value_str.isdigit():
                            value = int(value_str)
                        elif value_str.replace('.', '').isdigit():
                            value = float(value_str)
                        elif value_str.lower() in ['true', 'false']:
                            value = value_str.lower() == 'true'
                        else:
                            # Remove quotes if present
                            value = value_str.strip('\'"')
                    except:
                        value = value_str.strip('\'"')
                    
                    return ComparisonQueryExpression(field, op, value)
        
        raise ValueError(f"Cannot parse query: {query_string}")


def demonstrate_interpreter_pattern():
    """
    Demonstrate Interpreter pattern implementations.
    """
    print("=== INTERPRETER PATTERN DEMONSTRATION ===\n")
    
    # 1. Mathematical Expression Interpreter
    print("1. MATHEMATICAL EXPRESSION INTERPRETER:")
    
    # Create context with variables and functions
    context = Context()
    context.set_variable('x', 5)
    context.set_variable('y', 3)
    context.set_variable('pi', 3.14159)
    
    # Add mathematical functions
    context.set_function('sin', math.sin)
    context.set_function('cos', math.cos)
    context.set_function('sqrt', math.sqrt)
    context.set_function('abs', abs)
    context.set_function('max', max)
    context.set_function('min', min)
    
    # Create parser
    parser = ExpressionParser()
    
    # Test expressions
    test_expressions = [
        "2 + 3 * 4",
        "x * y + 10",
        "(x + y) * 2",
        "x ** 2 + y ** 2",
        "sqrt(x ** 2 + y ** 2)",
        "max(x, y, 10)",
        "x > y and y > 0",
        "result = x * 2 + y"
    ]
    
    print("\n   Expression evaluation:")
    print("   " + "=" * 40)
    
    for expr_str in test_expressions:
        try:
            expression = parser.parse(expr_str)
            result = expression.interpret(context)
            print(f"   {expr_str} = {result}")
        except Exception as e:
            print(f"   {expr_str} -> Error: {e}")
    
    # Show context after expressions
    print(f"\n   Context after evaluation:")
    context_info = context.get_context_info()
    print(f"     Variables: {context_info['variables']}")
    print(f"     Functions: {context_info['functions']}")
    
    print()
    
    # 2. Rule-Based System
    print("2. RULE-BASED SYSTEM:")
    
    # Create rule engine
    rule_engine = RuleEngine()
    
    # Set up context for rules
    rule_engine.context.set_variable('temperature', 25)
    rule_engine.context.set_variable('humidity', 60)
    rule_engine.context.set_variable('fan_speed', 0)
    rule_engine.context.set_variable('ac_on', False)
    rule_engine.context.set_variable('heater_on', False)
    rule_engine.context.debug_mode = True
    
    # Create rules
    rules_data = [
        ("High Temperature", "temperature > 28", "ac_on = True", 10),
        ("Low Temperature", "temperature < 18", "heater_on = True", 10),
        ("High Humidity", "humidity > 70", "fan_speed = 3", 5),
        ("Medium Humidity", "humidity > 50 and humidity <= 70", "fan_speed = 2", 3),
        ("AC Running", "ac_on == True", "fan_speed = max(fan_speed, 2)", 8),
        ("Comfortable", "temperature >= 20 and temperature <= 26 and humidity <= 60", "fan_speed = 1", 1)
    ]
    
    print("\n   Creating rules:")
    for name, condition_str, action_str, priority in rules_data:
        try:
            condition = parser.parse(condition_str)
            action = parser.parse(action_str)
            rule = Rule(name, condition, action, priority)
            rule_engine.add_rule(rule)
        except Exception as e:
            print(f"   Error creating rule '{name}': {e}")
    
    # Test different scenarios
    scenarios = [
        ("Normal conditions", {'temperature': 22, 'humidity': 45}),
        ("Hot day", {'temperature': 32, 'humidity': 55}),
        ("Cold day", {'temperature': 15, 'humidity': 40}),
        ("Humid day", {'temperature': 25, 'humidity': 80})
    ]
    
    print(f"\n   Testing scenarios:")
    for scenario_name, variables in scenarios:
        print(f"\n   Scenario: {scenario_name}")
        print(f"   Initial: {variables}")
        
        # Set variables
        for var, value in variables.items():
            rule_engine.context.set_variable(var, value)
        
        # Reset control variables
        rule_engine.context.set_variable('fan_speed', 0)
        rule_engine.context.set_variable('ac_on', False)
        rule_engine.context.set_variable('heater_on', False)
        
        # Execute rules
        executed = rule_engine.execute_rules()
        
        # Show results
        final_context = rule_engine.context.get_context_info()
        print(f"   Final state:")
        print(f"     Fan speed: {final_context['variables']['fan_speed']}")
        print(f"     AC on: {final_context['variables']['ac_on']}")
        print(f"     Heater on: {final_context['variables']['heater_on']}")
        print(f"   Rules executed: {len(executed)}")
    
    # Show engine statistics
    stats = rule_engine.get_engine_statistics()
    print(f"\n   Rule engine statistics:")
    print(f"     Total rules: {stats['total_rules']}")
    print(f"     Total executions: {stats['total_executions']}")
    print(f"     Most executed rules:")
    
    sorted_rules = sorted(stats['rule_statistics'].items(), 
                         key=lambda x: x[1]['executions'], reverse=True)
    for rule_name, rule_stats in sorted_rules[:3]:
        print(f"       {rule_name}: {rule_stats['executions']} executions")
    
    print()
    
    # 3. Query Language Interpreter
    print("3. QUERY LANGUAGE INTERPRETER:")
    
    # Create sample data
    sample_data = [
        {'name': 'Alice', 'age': 30, 'department': 'Engineering', 'salary': 75000},
        {'name': 'Bob', 'age': 25, 'department': 'Marketing', 'salary': 55000},
        {'name': 'Charlie', 'age': 35, 'department': 'Engineering', 'salary': 85000},
        {'name': 'Diana', 'age': 28, 'department': 'Sales', 'salary': 60000},
        {'name': 'Eve', 'age': 32, 'department': 'Engineering', 'salary': 80000},
        {'name': 'Frank', 'age': 29, 'department': 'Marketing', 'salary': 58000}
    ]
    
    # Create query interpreter
    query_interpreter = QueryInterpreter()
    query_interpreter.set_data(sample_data)
    
    # Test queries
    test_queries = [
        "age > 30",
        "department == Engineering",
        "salary >= 70000",
        "age > 25 and department == Engineering",
        "salary < 60000 or age > 32",
        "name contains a",
        "department == Marketing and salary > 55000"
    ]
    
    print(f"\n   Query execution:")
    print("   " + "=" * 40)
    
    for query_str in test_queries:
        try:
            query = query_interpreter.parse_simple_query(query_str)
            results = query_interpreter.execute_query(query)
            
            print(f"\n   Query: {query_str}")
            print(f"   Results ({len(results)} records):")
            
            for result in results:
                print(f"     {result['name']}: age={result['age']}, "
                      f"dept={result['department']}, salary=${result['salary']}")
        
        except Exception as e:
            print(f"\n   Query: {query_str}")
            print(f"   Error: {e}")
    
    print()
    
    # 4. Complex Expression Trees
    print("4. COMPLEX EXPRESSION TREES:")
    
    # Build complex expression programmatically
    # Expression: (x + y) * sin(pi / 4) + max(x, y, 10)
    
    x_var = VariableExpression('x')
    y_var = VariableExpression('y')
    pi_var = VariableExpression('pi')
    
    # (x + y)
    sum_expr = BinaryOperationExpression(x_var, '+', y_var)
    
    # pi / 4
    pi_div_4 = BinaryOperationExpression(pi_var, '/', NumberExpression(4))
    
    # sin(pi / 4)
    sin_expr = FunctionCallExpression('sin', [pi_div_4])
    
    # (x + y) * sin(pi / 4)
    product_expr = BinaryOperationExpression(sum_expr, '*', sin_expr)
    
    # max(x, y, 10)
    max_expr = FunctionCallExpression('max', [x_var, y_var, NumberExpression(10)])
    
    # Final expression
    complex_expr = BinaryOperationExpression(product_expr, '+', max_expr)
    
    print(f"\n   Complex expression tree:")
    print(f"   Expression: {complex_expr}")
    
    # Evaluate with different variable values
    test_values = [
        {'x': 5, 'y': 3},
        {'x': 1, 'y': 8},
        {'x': 12, 'y': 7}
    ]
    
    for values in test_values:
        # Set variables
        for var, val in values.items():
            context.set_variable(var, val)
        
        result = complex_expr.interpret(context)
        print(f"   With x={values['x']}, y={values['y']}: {result:.4f}")
    
    # Show expression structure
    expr_info = complex_expr.get_expression_info()
    print(f"\n   Expression structure:")
    print(f"   {json.dumps(expr_info, indent=2)}")
    
    print()
    
    # 5. Domain-Specific Language (DSL)
    print("5. DOMAIN-SPECIFIC LANGUAGE (DSL):")
    
    # Simple configuration DSL
    class ConfigurationInterpreter:
        """Interpreter for configuration DSL."""
        
        def __init__(self):
            self.config = {}
        
        def interpret_config(self, config_lines: List[str]) -> Dict[str, Any]:
            """Interpret configuration lines."""
            self.config = {}
            
            for line_num, line in enumerate(config_lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                try:
                    self._interpret_line(line)
                except Exception as e:
                    print(f"   Error on line {line_num}: {e}")
            
            return self.config
        
        def _interpret_line(self, line: str) -> None:
            """Interpret single configuration line."""
            # Simple key = value syntax
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse value
                if value.startswith('"') and value.endswith('"'):
                    # String value
                    self.config[key] = value[1:-1]
                elif value.lower() in ['true', 'false']:
                    # Boolean value
                    self.config[key] = value.lower() == 'true'
                elif value.isdigit():
                    # Integer value
                    self.config[key] = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    # Float value
                    self.config[key] = float(value)
                elif value.startswith('[') and value.endswith(']'):
                    # List value
                    list_content = value[1:-1]
                    if list_content:
                        items = [item.strip().strip('"') for item in list_content.split(',')]
                        self.config[key] = items
                    else:
                        self.config[key] = []
                else:
                    # Default to string
                    self.config[key] = value
            else:
                raise ValueError(f"Invalid syntax: {line}")
    
    # Test configuration DSL
    config_text = [
        "# Database configuration",
        'host = "localhost"',
        "port = 5432",
        'database = "myapp"',
        "ssl_enabled = true",
        "connection_timeout = 30.5",
        'allowed_hosts = ["localhost", "127.0.0.1", "::1"]',
        "",
        "# Cache settings",
        "cache_enabled = true",
        "cache_ttl = 3600",
        'cache_type = "redis"'
    ]
    
    config_interpreter = ConfigurationInterpreter()
    
    print(f"\n   Configuration DSL interpretation:")
    print("   " + "=" * 40)
    
    print("   Input configuration:")
    for line in config_text:
        if line.strip():
            print(f"     {line}")
    
    result_config = config_interpreter.interpret_config(config_text)
    
    print(f"\n   Parsed configuration:")
    for key, value in result_config.items():
        print(f"     {key}: {value} ({type(value).__name__})")
    
    print()
    
    # 6. Interpreter Pattern Benefits
    print("6. INTERPRETER PATTERN BENEFITS:")
    print("   ✓ Grammar Representation: Clear representation of language grammar")
    print("   ✓ Extensibility: Easy to add new grammar rules and operations")
    print("   ✓ Flexibility: Can interpret different types of languages and expressions")
    print("   ✓ Composability: Complex expressions built from simple components")
    print("   ✓ Reusability: Expression trees can be reused and cached")
    print("   ✓ Separation of Concerns: Parsing separated from interpretation")
    print("   ✓ Domain-Specific Languages: Enables creation of specialized languages")
    print("   ✓ Rule-Based Systems: Natural fit for business rule engines")
    print("   ✓ Maintainability: Grammar changes localized to specific classes")
    print("   ✓ Testability: Individual expressions can be tested independently")
    print()
    
    print("=== INTERPRETER PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_interpreter_pattern()
