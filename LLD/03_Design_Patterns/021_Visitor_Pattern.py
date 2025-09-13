"""
VISITOR PATTERN - Behavioral Design Pattern
============================================

Problem Statement:
Implement the Visitor pattern to define new operations on a family of classes
without changing the classes themselves, by separating algorithms from the
object structure on which they operate:
- Operations on object hierarchies without modifying classes
- Double dispatch mechanism for type-specific behavior
- Extensible operations on complex object structures
- AST (Abstract Syntax Tree) processing and compilation
- Document processing and transformation systems

Learning Objectives:
- Understand Visitor vs Strategy pattern differences
- Implement double dispatch for type-specific operations
- Design extensible operation systems
- Handle complex object hierarchies
- Create maintainable and flexible processing systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type, Callable
import json
import math
from datetime import datetime
from enum import Enum


# ============================================================================
# VISITOR INTERFACE
# ============================================================================

class Visitor(ABC):
    """Abstract visitor interface."""
    
    @abstractmethod
    def get_visitor_name(self) -> str:
        """Get visitor name for identification."""
        pass
    
    def get_visitor_info(self) -> Dict[str, Any]:
        """Get visitor information."""
        return {
            'name': self.get_visitor_name(),
            'type': self.__class__.__name__,
            'operations': [method for method in dir(self) if method.startswith('visit_')]
        }


class Visitable(ABC):
    """Abstract visitable interface (Element)."""
    
    @abstractmethod
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor and delegate to appropriate visit method."""
        pass
    
    @abstractmethod
    def get_element_info(self) -> Dict[str, Any]:
        """Get element information."""
        pass


# ============================================================================
# MATHEMATICAL EXPRESSION SYSTEM
# ============================================================================

class Expression(Visitable):
    """Abstract expression class."""
    
    @abstractmethod
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        pass
    
    @abstractmethod
    def get_element_info(self) -> Dict[str, Any]:
        """Get expression information."""
        pass


class Number(Expression):
    """Number expression."""
    
    def __init__(self, value: float):
        self.value = value
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor using double dispatch."""
        if hasattr(visitor, 'visit_number'):
            return visitor.visit_number(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support Number")
    
    def get_element_info(self) -> Dict[str, Any]:
        """Get number information."""
        return {
            'type': 'Number',
            'value': self.value,
            'is_integer': self.value.is_integer() if isinstance(self.value, float) else True
        }


class BinaryOperation(Expression):
    """Binary operation expression."""
    
    def __init__(self, left: Expression, operator: str, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor using double dispatch."""
        if hasattr(visitor, 'visit_binary_operation'):
            return visitor.visit_binary_operation(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support BinaryOperation")
    
    def get_element_info(self) -> Dict[str, Any]:
        """Get binary operation information."""
        return {
            'type': 'BinaryOperation',
            'operator': self.operator,
            'left_type': self.left.__class__.__name__,
            'right_type': self.right.__class__.__name__
        }


class UnaryOperation(Expression):
    """Unary operation expression."""
    
    def __init__(self, operator: str, operand: Expression):
        self.operator = operator
        self.operand = operand
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor using double dispatch."""
        if hasattr(visitor, 'visit_unary_operation'):
            return visitor.visit_unary_operation(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support UnaryOperation")
    
    def get_element_info(self) -> Dict[str, Any]:
        """Get unary operation information."""
        return {
            'type': 'UnaryOperation',
            'operator': self.operator,
            'operand_type': self.operand.__class__.__name__
        }


class Variable(Expression):
    """Variable expression."""
    
    def __init__(self, name: str):
        self.name = name
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor using double dispatch."""
        if hasattr(visitor, 'visit_variable'):
            return visitor.visit_variable(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support Variable")
    
    def get_element_info(self) -> Dict[str, Any]:
        """Get variable information."""
        return {
            'type': 'Variable',
            'name': self.name
        }


class FunctionCall(Expression):
    """Function call expression."""
    
    def __init__(self, function_name: str, arguments: List[Expression]):
        self.function_name = function_name
        self.arguments = arguments
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor using double dispatch."""
        if hasattr(visitor, 'visit_function_call'):
            return visitor.visit_function_call(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support FunctionCall")
    
    def get_element_info(self) -> Dict[str, Any]:
        """Get function call information."""
        return {
            'type': 'FunctionCall',
            'function_name': self.function_name,
            'argument_count': len(self.arguments),
            'argument_types': [arg.__class__.__name__ for arg in self.arguments]
        }


# ============================================================================
# EXPRESSION VISITORS
# ============================================================================

class EvaluationVisitor(Visitor):
    """Visitor to evaluate mathematical expressions."""
    
    def __init__(self, variables: Dict[str, float] = None):
        self.variables = variables or {}
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'log': math.log,
            'abs': abs,
            'max': max,
            'min': min
        }
    
    def get_visitor_name(self) -> str:
        return "Expression Evaluator"
    
    def visit_number(self, number: Number) -> float:
        """Visit number node."""
        return number.value
    
    def visit_binary_operation(self, operation: BinaryOperation) -> float:
        """Visit binary operation node."""
        left_value = operation.left.accept(self)
        right_value = operation.right.accept(self)
        
        if operation.operator == '+':
            return left_value + right_value
        elif operation.operator == '-':
            return left_value - right_value
        elif operation.operator == '*':
            return left_value * right_value
        elif operation.operator == '/':
            if right_value == 0:
                raise ValueError("Division by zero")
            return left_value / right_value
        elif operation.operator == '**' or operation.operator == '^':
            return left_value ** right_value
        elif operation.operator == '%':
            return left_value % right_value
        else:
            raise ValueError(f"Unknown binary operator: {operation.operator}")
    
    def visit_unary_operation(self, operation: UnaryOperation) -> float:
        """Visit unary operation node."""
        operand_value = operation.operand.accept(self)
        
        if operation.operator == '-':
            return -operand_value
        elif operation.operator == '+':
            return operand_value
        else:
            raise ValueError(f"Unknown unary operator: {operation.operator}")
    
    def visit_variable(self, variable: Variable) -> float:
        """Visit variable node."""
        if variable.name in self.variables:
            return self.variables[variable.name]
        else:
            raise ValueError(f"Undefined variable: {variable.name}")
    
    def visit_function_call(self, function_call: FunctionCall) -> float:
        """Visit function call node."""
        if function_call.function_name not in self.functions:
            raise ValueError(f"Unknown function: {function_call.function_name}")
        
        # Evaluate arguments
        arg_values = [arg.accept(self) for arg in function_call.arguments]
        
        # Call function
        func = self.functions[function_call.function_name]
        try:
            return func(*arg_values)
        except Exception as e:
            raise ValueError(f"Error calling {function_call.function_name}: {e}")


class PrintVisitor(Visitor):
    """Visitor to print expressions in readable format."""
    
    def __init__(self, use_parentheses: bool = True):
        self.use_parentheses = use_parentheses
    
    def get_visitor_name(self) -> str:
        return "Expression Printer"
    
    def visit_number(self, number: Number) -> str:
        """Visit number node."""
        if number.value.is_integer():
            return str(int(number.value))
        else:
            return str(number.value)
    
    def visit_binary_operation(self, operation: BinaryOperation) -> str:
        """Visit binary operation node."""
        left_str = operation.left.accept(self)
        right_str = operation.right.accept(self)
        
        result = f"{left_str} {operation.operator} {right_str}"
        
        if self.use_parentheses:
            return f"({result})"
        else:
            return result
    
    def visit_unary_operation(self, operation: UnaryOperation) -> str:
        """Visit unary operation node."""
        operand_str = operation.operand.accept(self)
        return f"{operation.operator}{operand_str}"
    
    def visit_variable(self, variable: Variable) -> str:
        """Visit variable node."""
        return variable.name
    
    def visit_function_call(self, function_call: FunctionCall) -> str:
        """Visit function call node."""
        arg_strings = [arg.accept(self) for arg in function_call.arguments]
        args_str = ", ".join(arg_strings)
        return f"{function_call.function_name}({args_str})"


class DerivativeVisitor(Visitor):
    """Visitor to compute symbolic derivatives."""
    
    def __init__(self, variable: str = 'x'):
        self.variable = variable
    
    def get_visitor_name(self) -> str:
        return f"Derivative Calculator (d/d{self.variable})"
    
    def visit_number(self, number: Number) -> Expression:
        """Derivative of constant is zero."""
        return Number(0)
    
    def visit_binary_operation(self, operation: BinaryOperation) -> Expression:
        """Visit binary operation using differentiation rules."""
        if operation.operator == '+':
            # (f + g)' = f' + g'
            left_derivative = operation.left.accept(self)
            right_derivative = operation.right.accept(self)
            return BinaryOperation(left_derivative, '+', right_derivative)
        
        elif operation.operator == '-':
            # (f - g)' = f' - g'
            left_derivative = operation.left.accept(self)
            right_derivative = operation.right.accept(self)
            return BinaryOperation(left_derivative, '-', right_derivative)
        
        elif operation.operator == '*':
            # (f * g)' = f' * g + f * g'
            f = operation.left
            g = operation.right
            f_prime = f.accept(self)
            g_prime = g.accept(self)
            
            term1 = BinaryOperation(f_prime, '*', g)
            term2 = BinaryOperation(f, '*', g_prime)
            return BinaryOperation(term1, '+', term2)
        
        elif operation.operator == '/':
            # (f / g)' = (f' * g - f * g') / g^2
            f = operation.left
            g = operation.right
            f_prime = f.accept(self)
            g_prime = g.accept(self)
            
            numerator_term1 = BinaryOperation(f_prime, '*', g)
            numerator_term2 = BinaryOperation(f, '*', g_prime)
            numerator = BinaryOperation(numerator_term1, '-', numerator_term2)
            
            denominator = BinaryOperation(g, '**', Number(2))
            return BinaryOperation(numerator, '/', denominator)
        
        elif operation.operator == '**':
            # For x^n where n is constant: (x^n)' = n * x^(n-1)
            # For general case, use logarithmic differentiation
            base = operation.left
            exponent = operation.right
            
            # Simple case: base is variable, exponent is constant
            if (isinstance(base, Variable) and base.name == self.variable and 
                isinstance(exponent, Number)):
                if exponent.value == 0:
                    return Number(0)
                elif exponent.value == 1:
                    return Number(1)
                else:
                    coefficient = Number(exponent.value)
                    new_exponent = Number(exponent.value - 1)
                    power_term = BinaryOperation(base, '**', new_exponent)
                    return BinaryOperation(coefficient, '*', power_term)
            else:
                # General case: use chain rule and logarithmic differentiation
                # (f^g)' = f^g * (g' * ln(f) + g * f'/f)
                raise NotImplementedError("General power rule not implemented")
        
        else:
            raise ValueError(f"Derivative not implemented for operator: {operation.operator}")
    
    def visit_unary_operation(self, operation: UnaryOperation) -> Expression:
        """Visit unary operation."""
        if operation.operator == '-':
            # (-f)' = -f'
            operand_derivative = operation.operand.accept(self)
            return UnaryOperation('-', operand_derivative)
        elif operation.operator == '+':
            # (+f)' = f'
            return operation.operand.accept(self)
        else:
            raise ValueError(f"Derivative not implemented for unary operator: {operation.operator}")
    
    def visit_variable(self, variable: Variable) -> Expression:
        """Visit variable node."""
        if variable.name == self.variable:
            return Number(1)  # dx/dx = 1
        else:
            return Number(0)  # dy/dx = 0 where y is not x
    
    def visit_function_call(self, function_call: FunctionCall) -> Expression:
        """Visit function call using chain rule."""
        if len(function_call.arguments) != 1:
            raise NotImplementedError("Derivatives only implemented for single-argument functions")
        
        arg = function_call.arguments[0]
        arg_derivative = arg.accept(self)
        
        # Apply chain rule: (f(g(x)))' = f'(g(x)) * g'(x)
        if function_call.function_name == 'sin':
            # d/dx sin(u) = cos(u) * u'
            cos_term = FunctionCall('cos', [arg])
            return BinaryOperation(cos_term, '*', arg_derivative)
        
        elif function_call.function_name == 'cos':
            # d/dx cos(u) = -sin(u) * u'
            sin_term = FunctionCall('sin', [arg])
            neg_sin = UnaryOperation('-', sin_term)
            return BinaryOperation(neg_sin, '*', arg_derivative)
        
        elif function_call.function_name == 'tan':
            # d/dx tan(u) = sec^2(u) * u' = (1/cos^2(u)) * u'
            cos_term = FunctionCall('cos', [arg])
            cos_squared = BinaryOperation(cos_term, '**', Number(2))
            sec_squared = BinaryOperation(Number(1), '/', cos_squared)
            return BinaryOperation(sec_squared, '*', arg_derivative)
        
        elif function_call.function_name == 'log':
            # d/dx log(u) = (1/u) * u'
            reciprocal = BinaryOperation(Number(1), '/', arg)
            return BinaryOperation(reciprocal, '*', arg_derivative)
        
        elif function_call.function_name == 'sqrt':
            # d/dx sqrt(u) = (1/(2*sqrt(u))) * u'
            sqrt_term = FunctionCall('sqrt', [arg])
            denominator = BinaryOperation(Number(2), '*', sqrt_term)
            coefficient = BinaryOperation(Number(1), '/', denominator)
            return BinaryOperation(coefficient, '*', arg_derivative)
        
        else:
            raise NotImplementedError(f"Derivative not implemented for function: {function_call.function_name}")


class SimplificationVisitor(Visitor):
    """Visitor to simplify mathematical expressions."""
    
    def get_visitor_name(self) -> str:
        return "Expression Simplifier"
    
    def visit_number(self, number: Number) -> Expression:
        """Numbers are already simplified."""
        return number
    
    def visit_binary_operation(self, operation: BinaryOperation) -> Expression:
        """Simplify binary operations."""
        # First, simplify operands
        left = operation.left.accept(self)
        right = operation.right.accept(self)
        
        # If both operands are numbers, evaluate
        if isinstance(left, Number) and isinstance(right, Number):
            evaluator = EvaluationVisitor()
            result_value = BinaryOperation(left, operation.operator, right).accept(evaluator)
            return Number(result_value)
        
        # Simplification rules
        if operation.operator == '+':
            # x + 0 = x, 0 + x = x
            if isinstance(left, Number) and left.value == 0:
                return right
            if isinstance(right, Number) and right.value == 0:
                return left
        
        elif operation.operator == '-':
            # x - 0 = x
            if isinstance(right, Number) and right.value == 0:
                return left
            # x - x = 0 (simplified case)
            if self._expressions_equal(left, right):
                return Number(0)
        
        elif operation.operator == '*':
            # x * 0 = 0, 0 * x = 0
            if isinstance(left, Number) and left.value == 0:
                return Number(0)
            if isinstance(right, Number) and right.value == 0:
                return Number(0)
            # x * 1 = x, 1 * x = x
            if isinstance(left, Number) and left.value == 1:
                return right
            if isinstance(right, Number) and right.value == 1:
                return left
        
        elif operation.operator == '/':
            # x / 1 = x
            if isinstance(right, Number) and right.value == 1:
                return left
            # x / x = 1 (simplified case)
            if self._expressions_equal(left, right):
                return Number(1)
        
        elif operation.operator == '**':
            # x ^ 0 = 1
            if isinstance(right, Number) and right.value == 0:
                return Number(1)
            # x ^ 1 = x
            if isinstance(right, Number) and right.value == 1:
                return left
            # 1 ^ x = 1
            if isinstance(left, Number) and left.value == 1:
                return Number(1)
        
        # Return simplified operation
        return BinaryOperation(left, operation.operator, right)
    
    def visit_unary_operation(self, operation: UnaryOperation) -> Expression:
        """Simplify unary operations."""
        operand = operation.operand.accept(self)
        
        # If operand is a number, evaluate
        if isinstance(operand, Number):
            if operation.operator == '-':
                return Number(-operand.value)
            elif operation.operator == '+':
                return operand
        
        # Double negative: -(-x) = x
        if (operation.operator == '-' and isinstance(operand, UnaryOperation) and 
            operand.operator == '-'):
            return operand.operand
        
        return UnaryOperation(operation.operator, operand)
    
    def visit_variable(self, variable: Variable) -> Expression:
        """Variables are already simplified."""
        return variable
    
    def visit_function_call(self, function_call: FunctionCall) -> Expression:
        """Simplify function calls."""
        # Simplify arguments
        simplified_args = [arg.accept(self) for arg in function_call.arguments]
        
        # If all arguments are numbers, evaluate
        if all(isinstance(arg, Number) for arg in simplified_args):
            evaluator = EvaluationVisitor()
            result_value = FunctionCall(function_call.function_name, simplified_args).accept(evaluator)
            return Number(result_value)
        
        return FunctionCall(function_call.function_name, simplified_args)
    
    def _expressions_equal(self, expr1: Expression, expr2: Expression) -> bool:
        """Check if two expressions are structurally equal (simplified check)."""
        if type(expr1) != type(expr2):
            return False
        
        if isinstance(expr1, Number):
            return expr1.value == expr2.value
        elif isinstance(expr1, Variable):
            return expr1.name == expr2.name
        # Add more cases as needed
        
        return False


# ============================================================================
# DOCUMENT PROCESSING SYSTEM
# ============================================================================

class DocumentElement(Visitable):
    """Abstract document element."""
    
    @abstractmethod
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        pass
    
    @abstractmethod
    def get_element_info(self) -> Dict[str, Any]:
        """Get element information."""
        pass


class Paragraph(DocumentElement):
    """Paragraph element."""
    
    def __init__(self, text: str, style: str = "normal"):
        self.text = text
        self.style = style
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        if hasattr(visitor, 'visit_paragraph'):
            return visitor.visit_paragraph(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support Paragraph")
    
    def get_element_info(self) -> Dict[str, Any]:
        return {
            'type': 'Paragraph',
            'text_length': len(self.text),
            'style': self.style,
            'word_count': len(self.text.split())
        }


class Heading(DocumentElement):
    """Heading element."""
    
    def __init__(self, text: str, level: int = 1):
        self.text = text
        self.level = max(1, min(level, 6))  # HTML-style levels 1-6
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        if hasattr(visitor, 'visit_heading'):
            return visitor.visit_heading(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support Heading")
    
    def get_element_info(self) -> Dict[str, Any]:
        return {
            'type': 'Heading',
            'text': self.text,
            'level': self.level,
            'text_length': len(self.text)
        }


class List(DocumentElement):
    """List element."""
    
    def __init__(self, items: List[str], ordered: bool = False):
        self.items = items
        self.ordered = ordered
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        if hasattr(visitor, 'visit_list'):
            return visitor.visit_list(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support List")
    
    def get_element_info(self) -> Dict[str, Any]:
        return {
            'type': 'List',
            'item_count': len(self.items),
            'ordered': self.ordered,
            'total_text_length': sum(len(item) for item in self.items)
        }


class Table(DocumentElement):
    """Table element."""
    
    def __init__(self, headers: List[str], rows: List[List[str]]):
        self.headers = headers
        self.rows = rows
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        if hasattr(visitor, 'visit_table'):
            return visitor.visit_table(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support Table")
    
    def get_element_info(self) -> Dict[str, Any]:
        return {
            'type': 'Table',
            'column_count': len(self.headers),
            'row_count': len(self.rows),
            'headers': self.headers
        }


class Document(DocumentElement):
    """Document containing multiple elements."""
    
    def __init__(self, title: str, elements: List[DocumentElement]):
        self.title = title
        self.elements = elements
    
    def accept(self, visitor: Visitor) -> Any:
        """Accept visitor."""
        if hasattr(visitor, 'visit_document'):
            return visitor.visit_document(self)
        elif hasattr(visitor, 'visit'):
            return visitor.visit(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} doesn't support Document")
    
    def get_element_info(self) -> Dict[str, Any]:
        return {
            'type': 'Document',
            'title': self.title,
            'element_count': len(self.elements),
            'element_types': [elem.__class__.__name__ for elem in self.elements]
        }


# ============================================================================
# DOCUMENT VISITORS
# ============================================================================

class HTMLExportVisitor(Visitor):
    """Visitor to export document to HTML."""
    
    def __init__(self, include_css: bool = True):
        self.include_css = include_css
        self.html_parts = []
    
    def get_visitor_name(self) -> str:
        return "HTML Exporter"
    
    def visit_document(self, document: Document) -> str:
        """Visit document and generate complete HTML."""
        self.html_parts = []
        
        # HTML document structure
        self.html_parts.append("<!DOCTYPE html>")
        self.html_parts.append("<html>")
        self.html_parts.append("<head>")
        self.html_parts.append(f"<title>{document.title}</title>")
        
        if self.include_css:
            self.html_parts.append("<style>")
            self.html_parts.append("body { font-family: Arial, sans-serif; margin: 40px; }")
            self.html_parts.append("h1, h2, h3, h4, h5, h6 { color: #333; }")
            self.html_parts.append("table { border-collapse: collapse; width: 100%; }")
            self.html_parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            self.html_parts.append("th { background-color: #f2f2f2; }")
            self.html_parts.append("</style>")
        
        self.html_parts.append("</head>")
        self.html_parts.append("<body>")
        self.html_parts.append(f"<h1>{document.title}</h1>")
        
        # Process all elements
        for element in document.elements:
            element.accept(self)
        
        self.html_parts.append("</body>")
        self.html_parts.append("</html>")
        
        return "\n".join(self.html_parts)
    
    def visit_paragraph(self, paragraph: Paragraph) -> None:
        """Visit paragraph element."""
        if paragraph.style == "normal":
            self.html_parts.append(f"<p>{paragraph.text}</p>")
        else:
            self.html_parts.append(f"<p class='{paragraph.style}'>{paragraph.text}</p>")
    
    def visit_heading(self, heading: Heading) -> None:
        """Visit heading element."""
        self.html_parts.append(f"<h{heading.level}>{heading.text}</h{heading.level}>")
    
    def visit_list(self, list_element: List) -> None:
        """Visit list element."""
        tag = "ol" if list_element.ordered else "ul"
        self.html_parts.append(f"<{tag}>")
        
        for item in list_element.items:
            self.html_parts.append(f"<li>{item}</li>")
        
        self.html_parts.append(f"</{tag}>")
    
    def visit_table(self, table: Table) -> None:
        """Visit table element."""
        self.html_parts.append("<table>")
        
        # Headers
        self.html_parts.append("<thead>")
        self.html_parts.append("<tr>")
        for header in table.headers:
            self.html_parts.append(f"<th>{header}</th>")
        self.html_parts.append("</tr>")
        self.html_parts.append("</thead>")
        
        # Rows
        self.html_parts.append("<tbody>")
        for row in table.rows:
            self.html_parts.append("<tr>")
            for cell in row:
                self.html_parts.append(f"<td>{cell}</td>")
            self.html_parts.append("</tr>")
        self.html_parts.append("</tbody>")
        
        self.html_parts.append("</table>")


class MarkdownExportVisitor(Visitor):
    """Visitor to export document to Markdown."""
    
    def __init__(self):
        self.markdown_parts = []
    
    def get_visitor_name(self) -> str:
        return "Markdown Exporter"
    
    def visit_document(self, document: Document) -> str:
        """Visit document and generate Markdown."""
        self.markdown_parts = []
        
        # Document title
        self.markdown_parts.append(f"# {document.title}")
        self.markdown_parts.append("")
        
        # Process all elements
        for element in document.elements:
            element.accept(self)
            self.markdown_parts.append("")  # Add spacing between elements
        
        return "\n".join(self.markdown_parts)
    
    def visit_paragraph(self, paragraph: Paragraph) -> None:
        """Visit paragraph element."""
        self.markdown_parts.append(paragraph.text)
    
    def visit_heading(self, heading: Heading) -> None:
        """Visit heading element."""
        prefix = "#" * heading.level
        self.markdown_parts.append(f"{prefix} {heading.text}")
    
    def visit_list(self, list_element: List) -> None:
        """Visit list element."""
        for i, item in enumerate(list_element.items):
            if list_element.ordered:
                self.markdown_parts.append(f"{i+1}. {item}")
            else:
                self.markdown_parts.append(f"- {item}")
    
    def visit_table(self, table: Table) -> None:
        """Visit table element."""
        # Headers
        header_row = "| " + " | ".join(table.headers) + " |"
        self.markdown_parts.append(header_row)
        
        # Separator
        separator = "| " + " | ".join(["---"] * len(table.headers)) + " |"
        self.markdown_parts.append(separator)
        
        # Rows
        for row in table.rows:
            row_text = "| " + " | ".join(row) + " |"
            self.markdown_parts.append(row_text)


class WordCountVisitor(Visitor):
    """Visitor to count words in document."""
    
    def __init__(self):
        self.word_count = 0
        self.character_count = 0
        self.element_counts = {}
    
    def get_visitor_name(self) -> str:
        return "Word Counter"
    
    def visit_document(self, document: Document) -> Dict[str, Any]:
        """Visit document and count words."""
        self.word_count = 0
        self.character_count = 0
        self.element_counts = {}
        
        # Process all elements
        for element in document.elements:
            element.accept(self)
        
        return {
            'total_words': self.word_count,
            'total_characters': self.character_count,
            'element_counts': self.element_counts,
            'average_words_per_element': self.word_count / len(document.elements) if document.elements else 0
        }
    
    def visit_paragraph(self, paragraph: Paragraph) -> None:
        """Count words in paragraph."""
        words = len(paragraph.text.split())
        chars = len(paragraph.text)
        
        self.word_count += words
        self.character_count += chars
        self.element_counts['paragraphs'] = self.element_counts.get('paragraphs', 0) + 1
    
    def visit_heading(self, heading: Heading) -> None:
        """Count words in heading."""
        words = len(heading.text.split())
        chars = len(heading.text)
        
        self.word_count += words
        self.character_count += chars
        self.element_counts['headings'] = self.element_counts.get('headings', 0) + 1
    
    def visit_list(self, list_element: List) -> None:
        """Count words in list."""
        for item in list_element.items:
            words = len(item.split())
            chars = len(item)
            
            self.word_count += words
            self.character_count += chars
        
        self.element_counts['lists'] = self.element_counts.get('lists', 0) + 1
    
    def visit_table(self, table: Table) -> None:
        """Count words in table."""
        # Count header words
        for header in table.headers:
            words = len(header.split())
            chars = len(header)
            
            self.word_count += words
            self.character_count += chars
        
        # Count row words
        for row in table.rows:
            for cell in row:
                words = len(cell.split())
                chars = len(cell)
                
                self.word_count += words
                self.character_count += chars
        
        self.element_counts['tables'] = self.element_counts.get('tables', 0) + 1


class ValidationVisitor(Visitor):
    """Visitor to validate document structure and content."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.element_count = 0
    
    def get_visitor_name(self) -> str:
        return "Document Validator"
    
    def visit_document(self, document: Document) -> Dict[str, Any]:
        """Validate document structure."""
        self.errors = []
        self.warnings = []
        self.element_count = 0
        
        # Check document title
        if not document.title or not document.title.strip():
            self.errors.append("Document title is empty")
        
        # Check if document has content
        if not document.elements:
            self.errors.append("Document has no content elements")
        
        # Process all elements
        for element in document.elements:
            self.element_count += 1
            element.accept(self)
        
        # Check document structure
        if self.element_count > 0:
            heading_count = sum(1 for elem in document.elements if isinstance(elem, Heading))
            if heading_count == 0:
                self.warnings.append("Document has no headings - consider adding structure")
        
        return {
            'is_valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'elements_validated': self.element_count
        }
    
    def visit_paragraph(self, paragraph: Paragraph) -> None:
        """Validate paragraph."""
        if not paragraph.text or not paragraph.text.strip():
            self.errors.append(f"Empty paragraph found")
        
        if len(paragraph.text) > 1000:
            self.warnings.append(f"Very long paragraph ({len(paragraph.text)} characters) - consider breaking up")
    
    def visit_heading(self, heading: Heading) -> None:
        """Validate heading."""
        if not heading.text or not heading.text.strip():
            self.errors.append(f"Empty heading at level {heading.level}")
        
        if len(heading.text) > 100:
            self.warnings.append(f"Very long heading ({len(heading.text)} characters)")
    
    def visit_list(self, list_element: List) -> None:
        """Validate list."""
        if not list_element.items:
            self.errors.append("Empty list found")
        
        for i, item in enumerate(list_element.items):
            if not item or not item.strip():
                self.errors.append(f"Empty list item at position {i+1}")
    
    def visit_table(self, table: Table) -> None:
        """Validate table."""
        if not table.headers:
            self.errors.append("Table has no headers")
        
        if not table.rows:
            self.warnings.append("Table has no data rows")
        
        # Check row consistency
        expected_columns = len(table.headers)
        for i, row in enumerate(table.rows):
            if len(row) != expected_columns:
                self.errors.append(f"Table row {i+1} has {len(row)} columns, expected {expected_columns}")


def demonstrate_visitor_pattern():
    """
    Demonstrate Visitor pattern implementations.
    """
    print("=== VISITOR PATTERN DEMONSTRATION ===\n")
    
    # 1. Mathematical Expression System
    print("1. MATHEMATICAL EXPRESSION SYSTEM:")
    
    # Build expression: (x + 2) * (x - 1)
    x = Variable('x')
    two = Number(2)
    one = Number(1)
    
    x_plus_2 = BinaryOperation(x, '+', two)
    x_minus_1 = BinaryOperation(x, '-', one)
    expression = BinaryOperation(x_plus_2, '*', x_minus_1)
    
    print("\n   Expression operations:")
    print("   " + "=" * 40)
    
    # Print expression
    printer = PrintVisitor()
    expression_str = expression.accept(printer)
    print(f"   Expression: {expression_str}")
    
    # Evaluate expression with x = 3
    evaluator = EvaluationVisitor({'x': 3})
    result = expression.accept(evaluator)
    print(f"   Value at x=3: {result}")
    
    # Evaluate with different x values
    for x_val in [0, 1, 2, 5]:
        evaluator.variables['x'] = x_val
        result = expression.accept(evaluator)
        print(f"   Value at x={x_val}: {result}")
    
    # Compute derivative
    derivative_visitor = DerivativeVisitor('x')
    derivative = expression.accept(derivative_visitor)
    derivative_str = derivative.accept(printer)
    print(f"   Derivative: {derivative_str}")
    
    # Simplify derivative
    simplifier = SimplificationVisitor()
    simplified_derivative = derivative.accept(simplifier)
    simplified_str = simplified_derivative.accept(printer)
    print(f"   Simplified derivative: {simplified_str}")
    
    print()
    
    # 2. Complex Expression with Functions
    print("2. COMPLEX EXPRESSION WITH FUNCTIONS:")
    
    # Build expression: sin(x^2) + cos(2*x)
    x_squared = BinaryOperation(x, '**', Number(2))
    sin_x_squared = FunctionCall('sin', [x_squared])
    
    two_x = BinaryOperation(Number(2), '*', x)
    cos_2x = FunctionCall('cos', [two_x])
    
    complex_expr = BinaryOperation(sin_x_squared, '+', cos_2x)
    
    print("\n   Complex expression operations:")
    print("   " + "=" * 30)
    
    # Print complex expression
    complex_str = complex_expr.accept(printer)
    print(f"   Expression: {complex_str}")
    
    # Evaluate at different points
    evaluator = EvaluationVisitor({'x': 0})
    for x_val in [0, math.pi/4, math.pi/2, math.pi]:
        evaluator.variables['x'] = x_val
        result = complex_expr.accept(evaluator)
        print(f"   Value at x={x_val:.3f}: {result:.6f}")
    
    # Compute derivative of complex expression
    complex_derivative = complex_expr.accept(derivative_visitor)
    complex_deriv_str = complex_derivative.accept(printer)
    print(f"   Derivative: {complex_deriv_str}")
    
    print()
    
    # 3. Document Processing System
    print("3. DOCUMENT PROCESSING SYSTEM:")
    
    # Create document elements
    elements = [
        Heading("Introduction", 1),
        Paragraph("This document demonstrates the Visitor pattern in action. "
                 "The Visitor pattern allows us to define new operations on object hierarchies "
                 "without modifying the classes themselves."),
        
        Heading("Key Benefits", 2),
        List([
            "Separation of algorithms from object structure",
            "Easy addition of new operations",
            "Centralized operation logic",
            "Type-safe operations through double dispatch"
        ], ordered=False),
        
        Heading("Implementation Details", 2),
        Paragraph("The implementation involves creating visitor interfaces and concrete visitors "
                 "that implement specific operations on the object hierarchy."),
        
        Table(
            ["Pattern", "Type", "Complexity"],
            [
                ["Visitor", "Behavioral", "Medium"],
                ["Strategy", "Behavioral", "Low"],
                ["Observer", "Behavioral", "Low"],
                ["Command", "Behavioral", "Medium"]
            ]
        ),
        
        Heading("Conclusion", 2),
        Paragraph("The Visitor pattern is particularly useful when you have a stable object "
                 "hierarchy but need to add new operations frequently.")
    ]
    
    # Create document
    document = Document("Visitor Pattern Guide", elements)
    
    print("\n   Document processing:")
    print("   " + "=" * 40)
    
    # Word count analysis
    word_counter = WordCountVisitor()
    word_stats = document.accept(word_counter)
    print(f"   Word count analysis:")
    print(f"     Total words: {word_stats['total_words']}")
    print(f"     Total characters: {word_stats['total_characters']}")
    print(f"     Element counts: {word_stats['element_counts']}")
    print(f"     Average words per element: {word_stats['average_words_per_element']:.1f}")
    
    # Document validation
    validator = ValidationVisitor()
    validation_result = document.accept(validator)
    print(f"\n   Document validation:")
    print(f"     Valid: {validation_result['is_valid']}")
    print(f"     Errors: {len(validation_result['errors'])}")
    print(f"     Warnings: {len(validation_result['warnings'])}")
    
    if validation_result['warnings']:
        print("     Warning messages:")
        for warning in validation_result['warnings']:
            print(f"       - {warning}")
    
    # Export to HTML
    html_exporter = HTMLExportVisitor()
    html_content = document.accept(html_exporter)
    print(f"\n   HTML export: {len(html_content)} characters generated")
    print(f"   HTML preview (first 200 chars):")
    print(f"   {html_content[:200]}...")
    
    # Export to Markdown
    markdown_exporter = MarkdownExportVisitor()
    markdown_content = document.accept(markdown_exporter)
    print(f"\n   Markdown export: {len(markdown_content)} characters generated")
    print(f"   Markdown preview (first 300 chars):")
    print(f"   {markdown_content[:300]}...")
    
    print()
    
    # 4. Visitor Pattern with Different Object Types
    print("4. VISITOR PATTERN WITH DIFFERENT OBJECT TYPES:")
    
    # Create a mixed document with various elements
    mixed_elements = [
        Heading("Data Analysis Report", 1),
        Paragraph("This report contains various data visualizations and statistics."),
        
        Table(
            ["Metric", "Q1", "Q2", "Q3", "Q4"],
            [
                ["Revenue", "$100K", "$120K", "$110K", "$140K"],
                ["Users", "1,000", "1,200", "1,100", "1,400"],
                ["Growth", "10%", "20%", "15%", "25%"]
            ]
        ),
        
        Heading("Key Findings", 2),
        List([
            "Revenue increased by 40% year-over-year",
            "User base grew consistently each quarter",
            "Q4 showed the strongest performance"
        ], ordered=True),
        
        Paragraph("These results indicate strong market adoption and customer satisfaction.")
    ]
    
    mixed_document = Document("Q4 Analysis", mixed_elements)
    
    print("\n   Mixed document analysis:")
    print("   " + "=" * 40)
    
    # Analyze mixed document
    mixed_stats = mixed_document.accept(word_counter)
    print(f"   Mixed document statistics:")
    print(f"     Elements: {mixed_stats['elements_validated'] if 'elements_validated' in mixed_stats else len(mixed_elements)}")
    print(f"     Words: {mixed_stats['total_words']}")
    print(f"     Element breakdown: {mixed_stats['element_counts']}")
    
    # Validate mixed document
    mixed_validation = mixed_document.accept(validator)
    print(f"   Validation: {'✓ Valid' if mixed_validation['is_valid'] else '✗ Invalid'}")
    
    print()
    
    # 5. Custom Visitor Implementation
    print("5. CUSTOM VISITOR IMPLEMENTATION:")
    
    class StatisticsVisitor(Visitor):
        """Custom visitor to gather detailed document statistics."""
        
        def __init__(self):
            self.stats = {
                'total_elements': 0,
                'heading_levels': {},
                'longest_paragraph': 0,
                'table_cells': 0,
                'list_items': 0,
                'unique_words': set()
            }
        
        def get_visitor_name(self) -> str:
            return "Document Statistics Analyzer"
        
        def visit_document(self, document: Document) -> Dict[str, Any]:
            """Analyze document statistics."""
            self.stats = {
                'total_elements': 0,
                'heading_levels': {},
                'longest_paragraph': 0,
                'table_cells': 0,
                'list_items': 0,
                'unique_words': set()
            }
            
            for element in document.elements:
                self.stats['total_elements'] += 1
                element.accept(self)
            
            # Convert set to count for JSON serialization
            unique_word_count = len(self.stats['unique_words'])
            self.stats['unique_words'] = unique_word_count
            
            return self.stats
        
        def visit_paragraph(self, paragraph: Paragraph) -> None:
            """Analyze paragraph statistics."""
            words = paragraph.text.lower().split()
            self.stats['unique_words'].update(words)
            
            if len(paragraph.text) > self.stats['longest_paragraph']:
                self.stats['longest_paragraph'] = len(paragraph.text)
        
        def visit_heading(self, heading: Heading) -> None:
            """Analyze heading statistics."""
            level_key = f"h{heading.level}"
            self.stats['heading_levels'][level_key] = self.stats['heading_levels'].get(level_key, 0) + 1
            
            words = heading.text.lower().split()
            self.stats['unique_words'].update(words)
        
        def visit_list(self, list_element: List) -> None:
            """Analyze list statistics."""
            self.stats['list_items'] += len(list_element.items)
            
            for item in list_element.items:
                words = item.lower().split()
                self.stats['unique_words'].update(words)
        
        def visit_table(self, table: Table) -> None:
            """Analyze table statistics."""
            # Count header cells
            self.stats['table_cells'] += len(table.headers)
            
            # Count data cells
            for row in table.rows:
                self.stats['table_cells'] += len(row)
            
            # Add words from headers and cells
            for header in table.headers:
                words = header.lower().split()
                self.stats['unique_words'].update(words)
            
            for row in table.rows:
                for cell in row:
                    words = cell.lower().split()
                    self.stats['unique_words'].update(words)
    
    # Apply custom visitor
    stats_visitor = StatisticsVisitor()
    detailed_stats = document.accept(stats_visitor)
    
    print("\n   Custom statistics analysis:")
    print("   " + "=" * 40)
    print(f"   Total elements: {detailed_stats['total_elements']}")
    print(f"   Heading levels: {detailed_stats['heading_levels']}")
    print(f"   Longest paragraph: {detailed_stats['longest_paragraph']} characters")
    print(f"   Table cells: {detailed_stats['table_cells']}")
    print(f"   List items: {detailed_stats['list_items']}")
    print(f"   Unique words: {detailed_stats['unique_words']}")
    
    print()
    
    # 6. Visitor Pattern Benefits
    print("6. VISITOR PATTERN BENEFITS:")
    print("   ✓ Operation Extension: Easy to add new operations without modifying existing classes")
    print("   ✓ Separation of Concerns: Algorithms are separated from object structure")
    print("   ✓ Type Safety: Double dispatch ensures type-safe operations")
    print("   ✓ Centralized Logic: Related operations are grouped in visitor classes")
    print("   ✓ Maintainability: Changes to operations don't affect object hierarchy")
    print("   ✓ Reusability: Visitors can be reused across different object hierarchies")
    print("   ✓ Flexibility: Different visitors can implement same operations differently")
    print("   ✓ Extensibility: New element types can be added with corresponding visitor methods")
    print("   ✓ Clean Architecture: Promotes clean separation between data and operations")
    print("   ✓ Polymorphic Dispatch: Leverages polymorphism for elegant operation selection")
    print()
    
    print("=== VISITOR PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_visitor_pattern()
