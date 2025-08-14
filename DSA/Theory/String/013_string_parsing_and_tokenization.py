"""
String Parsing and Tokenization - Advanced Techniques
====================================================

Topics: Expression parsing, tokenization, lexical analysis
Companies: Google, Facebook, Microsoft, Amazon
Difficulty: Medium to Hard
"""

from typing import List, Dict, Set, Tuple, Optional, Union
import re
from collections import deque
from enum import Enum

class StringParsingTokenization:
    
    # ==========================================
    # 1. BASIC TOKENIZATION
    # ==========================================
    
    def simple_tokenize(self, text: str, delimiters: str = " \t\n") -> List[str]:
        """Simple tokenization by delimiters"""
        tokens = []
        current_token = ""
        
        for char in text:
            if char in delimiters:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def csv_parser(self, line: str, delimiter: str = ',', quote: str = '"') -> List[str]:
        """Parse CSV line handling quoted fields"""
        fields = []
        current_field = ""
        in_quotes = False
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == quote:
                if in_quotes and i + 1 < len(line) and line[i + 1] == quote:
                    # Escaped quote
                    current_field += quote
                    i += 1
                else:
                    in_quotes = not in_quotes
            elif char == delimiter and not in_quotes:
                fields.append(current_field)
                current_field = ""
            else:
                current_field += char
            
            i += 1
        
        fields.append(current_field)
        return fields
    
    # ==========================================
    # 2. EXPRESSION PARSING
    # ==========================================
    
    class TokenType(Enum):
        NUMBER = "NUMBER"
        OPERATOR = "OPERATOR"
        LPAREN = "LPAREN"
        RPAREN = "RPAREN"
        VARIABLE = "VARIABLE"
        EOF = "EOF"
    
    class Token:
        def __init__(self, type_: 'TokenType', value: str):
            self.type = type_
            self.value = value
        
        def __repr__(self):
            return f"Token({self.type}, {self.value})"
    
    def tokenize_expression(self, expr: str) -> List['Token']:
        """Tokenize mathematical expression"""
        tokens = []
        i = 0
        
        while i < len(expr):
            if expr[i].isspace():
                i += 1
            elif expr[i].isdigit():
                # Parse number
                num = ""
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num += expr[i]
                    i += 1
                tokens.append(self.Token(self.TokenType.NUMBER, num))
            elif expr[i] in "+-*/^%":
                tokens.append(self.Token(self.TokenType.OPERATOR, expr[i]))
                i += 1
            elif expr[i] == '(':
                tokens.append(self.Token(self.TokenType.LPAREN, expr[i]))
                i += 1
            elif expr[i] == ')':
                tokens.append(self.Token(self.TokenType.RPAREN, expr[i]))
                i += 1
            elif expr[i].isalpha():
                # Parse variable
                var = ""
                while i < len(expr) and (expr[i].isalnum() or expr[i] == '_'):
                    var += expr[i]
                    i += 1
                tokens.append(self.Token(self.TokenType.VARIABLE, var))
            else:
                i += 1
        
        tokens.append(self.Token(self.TokenType.EOF, ""))
        return tokens
    
    def infix_to_postfix(self, expression: str) -> str:
        """Convert infix to postfix notation (Shunting Yard Algorithm)"""
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, '%': 2}
        right_associative = {'^'}
        
        output = []
        operators = []
        
        tokens = self.tokenize_expression(expression)
        
        for token in tokens:
            if token.type == self.TokenType.NUMBER or token.type == self.TokenType.VARIABLE:
                output.append(token.value)
            elif token.type == self.TokenType.OPERATOR:
                while (operators and 
                       operators[-1] != '(' and
                       ((token.value not in right_associative and 
                         precedence.get(operators[-1], 0) >= precedence.get(token.value, 0)) or
                        (token.value in right_associative and 
                         precedence.get(operators[-1], 0) > precedence.get(token.value, 0)))):
                    output.append(operators.pop())
                operators.append(token.value)
            elif token.type == self.TokenType.LPAREN:
                operators.append(token.value)
            elif token.type == self.TokenType.RPAREN:
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                if operators:
                    operators.pop()  # Remove '('
        
        while operators:
            output.append(operators.pop())
        
        return ' '.join(output)
    
    def evaluate_postfix(self, postfix: str) -> float:
        """Evaluate postfix expression"""
        stack = []
        tokens = postfix.split()
        
        for token in tokens:
            if token.replace('.', '').replace('-', '').isdigit():
                stack.append(float(token))
            elif token in "+-*/%^":
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    
                    if token == '+':
                        stack.append(a + b)
                    elif token == '-':
                        stack.append(a - b)
                    elif token == '*':
                        stack.append(a * b)
                    elif token == '/':
                        stack.append(a / b if b != 0 else float('inf'))
                    elif token == '%':
                        stack.append(a % b if b != 0 else float('inf'))
                    elif token == '^':
                        stack.append(a ** b)
        
        return stack[0] if stack else 0
    
    # ==========================================
    # 3. STRING PARSING ALGORITHMS
    # ==========================================
    
    def parse_json_simple(self, json_str: str) -> Union[dict, list, str, float, bool, None]:
        """Simple JSON parser"""
        json_str = json_str.strip()
        self.pos = 0
        self.text = json_str
        
        def parse_value():
            self.skip_whitespace()
            
            if self.pos >= len(self.text):
                return None
            
            char = self.text[self.pos]
            
            if char == '"':
                return self.parse_string()
            elif char == '{':
                return self.parse_object()
            elif char == '[':
                return self.parse_array()
            elif char == 't':
                return self.parse_true()
            elif char == 'f':
                return self.parse_false()
            elif char == 'n':
                return self.parse_null()
            elif char.isdigit() or char == '-':
                return self.parse_number()
            
            return None
        
        def parse_string():
            self.pos += 1  # Skip opening quote
            start = self.pos
            
            while self.pos < len(self.text) and self.text[self.pos] != '"':
                if self.text[self.pos] == '\\':
                    self.pos += 2  # Skip escaped character
                else:
                    self.pos += 1
            
            result = self.text[start:self.pos]
            self.pos += 1  # Skip closing quote
            return result
        
        def parse_object():
            obj = {}
            self.pos += 1  # Skip '{'
            self.skip_whitespace()
            
            while self.pos < len(self.text) and self.text[self.pos] != '}':
                # Parse key
                key = parse_value()
                self.skip_whitespace()
                
                if self.pos < len(self.text) and self.text[self.pos] == ':':
                    self.pos += 1
                    self.skip_whitespace()
                    value = parse_value()
                    obj[key] = value
                
                self.skip_whitespace()
                if self.pos < len(self.text) and self.text[self.pos] == ',':
                    self.pos += 1
                    self.skip_whitespace()
            
            if self.pos < len(self.text):
                self.pos += 1  # Skip '}'
            
            return obj
        
        def parse_array():
            arr = []
            self.pos += 1  # Skip '['
            self.skip_whitespace()
            
            while self.pos < len(self.text) and self.text[self.pos] != ']':
                value = parse_value()
                arr.append(value)
                self.skip_whitespace()
                
                if self.pos < len(self.text) and self.text[self.pos] == ',':
                    self.pos += 1
                    self.skip_whitespace()
            
            if self.pos < len(self.text):
                self.pos += 1  # Skip ']'
            
            return arr
        
        def parse_number():
            start = self.pos
            if self.text[self.pos] == '-':
                self.pos += 1
            
            while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                self.pos += 1
            
            return float(self.text[start:self.pos])
        
        def parse_true():
            self.pos += 4
            return True
        
        def parse_false():
            self.pos += 5
            return False
        
        def parse_null():
            self.pos += 4
            return None
        
        self.skip_whitespace = lambda: None
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        
        def skip_whitespace():
            while self.pos < len(self.text) and self.text[self.pos].isspace():
                self.pos += 1
        
        self.skip_whitespace = skip_whitespace
        return parse_value()
    
    # ==========================================
    # 4. URL AND PATH PARSING
    # ==========================================
    
    def parse_url(self, url: str) -> Dict[str, str]:
        """Parse URL into components"""
        result = {
            'scheme': '',
            'host': '',
            'port': '',
            'path': '',
            'query': '',
            'fragment': ''
        }
        
        # Fragment
        if '#' in url:
            url, result['fragment'] = url.rsplit('#', 1)
        
        # Query
        if '?' in url:
            url, result['query'] = url.rsplit('?', 1)
        
        # Scheme
        if '://' in url:
            result['scheme'], url = url.split('://', 1)
        
        # Host and port
        if '/' in url:
            host_port, result['path'] = url.split('/', 1)
            result['path'] = '/' + result['path']
        else:
            host_port = url
        
        if ':' in host_port:
            result['host'], result['port'] = host_port.rsplit(':', 1)
        else:
            result['host'] = host_port
        
        return result
    
    def parse_query_string(self, query: str) -> Dict[str, List[str]]:
        """Parse query string into parameters"""
        params = {}
        
        if not query:
            return params
        
        for param in query.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                key = key.replace('+', ' ')
                value = value.replace('+', ' ')
                
                if key not in params:
                    params[key] = []
                params[key].append(value)
        
        return params
    
    # ==========================================
    # 5. LEXICAL ANALYSIS
    # ==========================================
    
    def lex_analyze(self, code: str) -> List[Tuple[str, str]]:
        """Simple lexical analysis for programming language"""
        keywords = {'if', 'else', 'while', 'for', 'return', 'function', 'var', 'let', 'const'}
        operators = {'+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||'}
        
        tokens = []
        i = 0
        
        while i < len(code):
            if code[i].isspace():
                i += 1
            elif code[i].isalpha() or code[i] == '_':
                # Identifier or keyword
                start = i
                while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                    i += 1
                
                word = code[start:i]
                token_type = 'KEYWORD' if word in keywords else 'IDENTIFIER'
                tokens.append((token_type, word))
            
            elif code[i].isdigit():
                # Number
                start = i
                while i < len(code) and (code[i].isdigit() or code[i] == '.'):
                    i += 1
                tokens.append(('NUMBER', code[start:i]))
            
            elif code[i] == '"' or code[i] == "'":
                # String literal
                quote = code[i]
                i += 1
                start = i
                while i < len(code) and code[i] != quote:
                    if code[i] == '\\':
                        i += 2
                    else:
                        i += 1
                tokens.append(('STRING', code[start:i]))
                i += 1  # Skip closing quote
            
            elif code[i:i+2] in operators:
                tokens.append(('OPERATOR', code[i:i+2]))
                i += 2
            
            elif code[i] in operators:
                tokens.append(('OPERATOR', code[i]))
                i += 1
            
            elif code[i] in '(){}[]':
                tokens.append(('DELIMITER', code[i]))
                i += 1
            
            elif code[i] in ',;':
                tokens.append(('PUNCTUATION', code[i]))
                i += 1
            
            else:
                i += 1
        
        return tokens

# Test Examples
def run_examples():
    spt = StringParsingTokenization()
    
    print("=== STRING PARSING AND TOKENIZATION EXAMPLES ===\n")
    
    # Basic tokenization
    print("1. BASIC TOKENIZATION:")
    tokens = spt.simple_tokenize("Hello world python")
    print(f"Tokens: {tokens}")
    
    csv_fields = spt.csv_parser('name,"John Doe",30,"San Francisco, CA"')
    print(f"CSV fields: {csv_fields}")
    
    # Expression parsing
    print("\n2. EXPRESSION PARSING:")
    expr = "3 + 4 * 2 / (1 - 5) ^ 2"
    postfix = spt.infix_to_postfix(expr)
    result = spt.evaluate_postfix(postfix)
    print(f"Infix: {expr}")
    print(f"Postfix: {postfix}")
    print(f"Result: {result}")
    
    # JSON parsing
    print("\n3. JSON PARSING:")
    json_str = '{"name": "John", "age": 30, "hobbies": ["reading", "coding"]}'
    parsed = spt.parse_json_simple(json_str)
    print(f"Parsed JSON: {parsed}")
    
    # URL parsing
    print("\n4. URL PARSING:")
    url = "https://example.com:8080/path/to/resource?param1=value1&param2=value2#section"
    parsed_url = spt.parse_url(url)
    print(f"Parsed URL: {parsed_url}")
    
    query_params = spt.parse_query_string("name=John&age=30&skills=python&skills=java")
    print(f"Query params: {query_params}")
    
    # Lexical analysis
    print("\n5. LEXICAL ANALYSIS:")
    code = 'function add(a, b) { return a + b; }'
    tokens = spt.lex_analyze(code)
    print(f"Lexical tokens: {tokens}")

if __name__ == "__main__":
    run_examples() 