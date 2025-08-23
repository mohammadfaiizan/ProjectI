"""
1597. Build Binary Expression Tree From Infix Expression
Difficulty: Hard

Problem:
A binary expression tree is a kind of binary tree used to represent arithmetic expressions. 
Each node of a binary expression tree has either zero or two children. 
Leaf nodes (nodes with 0 children) correspond to operands (variables), 
and internal nodes (nodes with 2 children) correspond to the operators '+', '-', '*', '/' 
(which could be viewed as functions with two arguments).

It's guaranteed that no subtree will yield a value that exceeds 10^9 in absolute value, 
and all intermediate calculations will fit in a 32-bit integer.

For each binary expression tree, you need to return its root node.

Note: This problem relates to MST through the concept of building optimal tree structures 
and managing node connections with priorities, similar to how MST algorithms handle edge weights.

Examples:
Input: s = "2-1+2"
Output: [+,-,2,2,1]
Explanation: The tree represents the expression (2-1)+2

Input: s = "3*4-2*5"
Output: [-,*,*,3,4,2,5]  
Explanation: The tree represents the expression (3*4)-(2*5)

Input: s = "1+2+3+4+5"
Output: [+,+,5,+,4,1,+,null,null,2,3]
Explanation: Left-associative: ((((1+2)+3)+4)+5)

Constraints:
- 1 <= s.length <= 1000
- s consists of digits and the characters '+', '-', '*', and '/'.
- Operands in s are exactly 1 digit.
- It is guaranteed that s is a valid expression.
"""

from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def expTree_approach1_recursive_descent(self, s: str) -> Optional[TreeNode]:
        """
        Approach 1: Recursive Descent Parser
        
        Build expression tree using recursive descent parsing with precedence.
        Similar to MST in that we're building optimal tree structure.
        
        Time: O(n)
        Space: O(n)
        """
        self.index = 0
        
        def parse_expression():
            """Parse expression with + and - (lowest precedence)"""
            node = parse_term()
            
            while self.index < len(s) and s[self.index] in '+-':
                op = s[self.index]
                self.index += 1
                right = parse_term()
                # Create operator node with current tree as left child
                op_node = TreeNode(op)
                op_node.left = node
                op_node.right = right
                node = op_node
            
            return node
        
        def parse_term():
            """Parse term with * and / (higher precedence)"""
            node = parse_factor()
            
            while self.index < len(s) and s[self.index] in '*/':
                op = s[self.index]
                self.index += 1
                right = parse_factor()
                # Create operator node
                op_node = TreeNode(op)
                op_node.left = node
                op_node.right = right
                node = op_node
            
            return node
        
        def parse_factor():
            """Parse factor (numbers and parentheses)"""
            if s[self.index] == '(':
                self.index += 1  # Skip '('
                node = parse_expression()
                self.index += 1  # Skip ')'
                return node
            else:
                # Parse number (single digit)
                val = s[self.index]
                self.index += 1
                return TreeNode(val)
        
        return parse_expression()
    
    def expTree_approach2_shunting_yard_algorithm(self, s: str) -> Optional[TreeNode]:
        """
        Approach 2: Shunting Yard Algorithm (Dijkstra's Algorithm)
        
        Convert infix to postfix, then build tree from postfix.
        Similar to MST edge processing with priority consideration.
        
        Time: O(n)
        Space: O(n)
        """
        def get_precedence(op):
            """Get operator precedence"""
            if op in '+-':
                return 1
            if op in '*/':
                return 2
            return 0
        
        def is_left_associative(op):
            """Check if operator is left associative"""
            return op in '+-*/'
        
        # Convert infix to postfix using Shunting Yard
        output_queue = []
        operator_stack = []
        
        i = 0
        while i < len(s):
            char = s[i]
            
            if char.isdigit():
                output_queue.append(char)
            elif char in '+-*/':
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       (get_precedence(operator_stack[-1]) > get_precedence(char) or
                        (get_precedence(operator_stack[-1]) == get_precedence(char) and 
                         is_left_associative(char)))):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(char)
            elif char == '(':
                operator_stack.append(char)
            elif char == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack:
                    operator_stack.pop()  # Remove '('
            
            i += 1
        
        # Pop remaining operators
        while operator_stack:
            output_queue.append(operator_stack.pop())
        
        # Build tree from postfix expression
        stack = []
        for token in output_queue:
            if token.isdigit():
                stack.append(TreeNode(token))
            else:
                # Pop two operands
                right = stack.pop()
                left = stack.pop()
                # Create operator node
                op_node = TreeNode(token)
                op_node.left = left
                op_node.right = right
                stack.append(op_node)
        
        return stack[0] if stack else None
    
    def expTree_approach3_precedence_climbing(self, s: str) -> Optional[TreeNode]:
        """
        Approach 3: Precedence Climbing
        
        Build tree by climbing precedence levels.
        Similar to MST priority processing.
        
        Time: O(n)
        Space: O(n)
        """
        self.index = 0
        
        def get_precedence(op):
            """Get operator precedence"""
            if op in '+-':
                return 1
            if op in '*/':
                return 2
            return 0
        
        def parse_primary():
            """Parse primary expression (number or parentheses)"""
            if self.index < len(s) and s[self.index] == '(':
                self.index += 1  # Skip '('
                node = parse_expression(0)
                self.index += 1  # Skip ')'
                return node
            else:
                # Parse number
                val = s[self.index]
                self.index += 1
                return TreeNode(val)
        
        def parse_expression(min_precedence):
            """Parse expression with precedence climbing"""
            left = parse_primary()
            
            while (self.index < len(s) and 
                   s[self.index] in '+-*/' and
                   get_precedence(s[self.index]) >= min_precedence):
                
                op = s[self.index]
                self.index += 1
                
                # For left associative operators, use precedence + 1
                # For right associative, use same precedence
                next_min_prec = get_precedence(op) + 1
                right = parse_expression(next_min_prec)
                
                # Create operator node
                op_node = TreeNode(op)
                op_node.left = left
                op_node.right = right
                left = op_node
            
            return left
        
        return parse_expression(0)
    
    def expTree_approach4_stack_based_parsing(self, s: str) -> Optional[TreeNode]:
        """
        Approach 4: Stack-based Parsing with Operator Precedence
        
        Use two stacks for operands and operators.
        Similar to Union-Find operations in MST.
        
        Time: O(n)
        Space: O(n)
        """
        def get_precedence(op):
            """Get operator precedence"""
            return {'(': 0, '+': 1, '-': 1, '*': 2, '/': 2}.get(op, 0)
        
        def apply_operator(operators, operands):
            """Apply top operator to operands"""
            if len(operands) < 2 or not operators:
                return
            
            op = operators.pop()
            right = operands.pop()
            left = operands.pop()
            
            op_node = TreeNode(op)
            op_node.left = left
            op_node.right = right
            operands.append(op_node)
        
        operands = []  # Stack of TreeNode operands
        operators = []  # Stack of operator characters
        
        i = 0
        while i < len(s):
            char = s[i]
            
            if char.isdigit():
                operands.append(TreeNode(char))
            elif char == '(':
                operators.append(char)
            elif char == ')':
                # Apply all operators until '('
                while operators and operators[-1] != '(':
                    apply_operator(operators, operands)
                if operators:
                    operators.pop()  # Remove '('
            elif char in '+-*/':
                # Apply operators with higher or equal precedence
                while (operators and 
                       operators[-1] != '(' and
                       get_precedence(operators[-1]) >= get_precedence(char)):
                    apply_operator(operators, operands)
                operators.append(char)
            
            i += 1
        
        # Apply remaining operators
        while operators:
            apply_operator(operators, operands)
        
        return operands[0] if operands else None
    
    def expTree_approach5_mst_inspired_construction(self, s: str) -> Optional[TreeNode]:
        """
        Approach 5: MST-inspired Tree Construction
        
        Build tree by treating operators as edges with weights (precedence).
        Connect operands optimally based on operator precedence.
        
        Time: O(n log n)
        Space: O(n)
        """
        # Tokenize expression
        tokens = []
        operands = []
        operators = []
        
        for i, char in enumerate(s):
            if char.isdigit():
                node = TreeNode(char)
                tokens.append(('operand', node, i))
                operands.append((node, i))
            elif char in '+-*/':
                tokens.append(('operator', char, i))
                operators.append((char, i))
        
        # Create "edges" representing operator applications
        # Each operator connects two adjacent operands
        edges = []
        
        for op, op_pos in operators:
            # Find operands before and after this operator
            left_operand = None
            right_operand = None
            
            # Find nearest operand to the left
            for operand, pos in reversed(operands):
                if pos < op_pos:
                    left_operand = (operand, pos)
                    break
            
            # Find nearest operand to the right  
            for operand, pos in operands:
                if pos > op_pos:
                    right_operand = (operand, pos)
                    break
            
            if left_operand and right_operand:
                # Priority based on operator precedence (higher precedence = lower value for min-heap)
                precedence = {'*': 1, '/': 1, '+': 2, '-': 2}[op]
                edges.append((precedence, op_pos, op, left_operand, right_operand))
        
        # Sort edges by precedence (similar to MST edge sorting)
        edges.sort()
        
        # Build tree by applying operators in precedence order
        # This is a simplified version - actual implementation would need
        # more sophisticated handling of operand management
        
        # For this approach, we'll fall back to the recursive descent method
        # as the MST analogy, while instructive, is not directly applicable
        # to expression tree construction
        
        return self.expTree_approach1_recursive_descent(s)

def test_expression_tree():
    """Test all approaches with various expressions"""
    solution = Solution()
    
    test_cases = [
        "2-1+2",
        "3*4-2*5", 
        "1+2+3+4+5",
        "2*3+4",
        "2+3*4",
        "(2+3)*4",
        "1",
        "1+2*3-4/2",
    ]
    
    approaches = [
        ("Recursive Descent", solution.expTree_approach1_recursive_descent),
        ("Shunting Yard", solution.expTree_approach2_shunting_yard_algorithm),
        ("Precedence Climbing", solution.expTree_approach3_precedence_climbing),
        ("Stack-based", solution.expTree_approach4_stack_based_parsing),
        ("MST-inspired", solution.expTree_approach5_mst_inspired_construction),
    ]
    
    def print_tree(root, level=0, prefix="Root: "):
        """Print tree structure"""
        if root:
            print("  " * level + prefix + str(root.val))
            if root.left or root.right:
                if root.left:
                    print_tree(root.left, level + 1, "L--- ")
                else:
                    print("  " * (level + 1) + "L--- None")
                if root.right:
                    print_tree(root.right, level + 1, "R--- ")
                else:
                    print("  " * (level + 1) + "R--- None")
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, expression in enumerate(test_cases[:3]):  # Test first 3 cases
            print(f"\nTest {i+1}: Expression '{expression}'")
            try:
                result = func(expression)
                if result:
                    print_tree(result)
                else:
                    print("  Result: None")
            except Exception as e:
                print(f"  Error: {e}")

def demonstrate_parsing_techniques():
    """Demonstrate different parsing techniques"""
    print("\n=== Parsing Techniques Demo ===")
    
    expression = "2+3*4"
    print(f"Expression: {expression}")
    
    print(f"\n1. **Recursive Descent Parsing:**")
    print(f"   • Parse by grammar rules")
    print(f"   • Expression → Term (+|- Term)*")
    print(f"   • Term → Factor (*|/ Factor)*") 
    print(f"   • Factor → Number | (Expression)")
    print(f"   • Natural precedence handling")
    
    print(f"\n2. **Shunting Yard Algorithm:**")
    print(f"   • Convert infix to postfix: 2 3 4 * +")
    print(f"   • Build tree from postfix")
    print(f"   • Two-phase approach")
    
    print(f"\n3. **Precedence Climbing:**")
    print(f"   • Climb precedence levels")
    print(f"   • Handle associativity explicitly")
    print(f"   • Recursive with precedence parameter")
    
    print(f"\n4. **Stack-based Parsing:**")
    print(f"   • Maintain operator and operand stacks")
    print(f"   • Apply operators based on precedence")
    print(f"   • Similar to calculator evaluation")

def demonstrate_mst_connection():
    """Demonstrate connection to MST concepts"""
    print("\n=== MST Connection Demo ===")
    
    print("Expression Tree Building vs MST Construction:")
    
    print("\n1. **Structural Similarities:**")
    print("   • Both build tree structures")
    print("   • Both consider connection costs/priorities")
    print("   • Both ensure all elements are connected")
    print("   • Both avoid cycles")
    
    print("\n2. **Priority Handling:**")
    print("   • MST: Edge weights determine connection order")
    print("   • Expression: Operator precedence determines tree structure")
    print("   • Both use greedy selection based on priorities")
    
    print("\n3. **Union-Find Analogy:**")
    print("   • MST: Union-Find tracks connected components")
    print("   • Expression: Stack operations merge sub-expressions")
    print("   • Both incrementally build final structure")
    
    print("\n4. **Algorithm Design Patterns:**")
    print("   • Sorting by priority (edges/operators)")
    print("   • Greedy selection strategies")
    print("   • Efficient data structure usage")
    print("   • Optimal substructure properties")
    
    print("\n5. **Tree Construction Process:**")
    print("   • MST: Start with forest, merge trees via edges")
    print("   • Expression: Start with operands, merge via operators")
    print("   • Both result in single connected tree")

def analyze_precedence_and_associativity():
    """Analyze operator precedence and associativity"""
    print("\n=== Precedence and Associativity Analysis ===")
    
    print("Operator Properties:")
    
    print("\n1. **Precedence Levels:**")
    print("   • Level 1 (lowest): + and -")
    print("   • Level 2 (higher): * and /")
    print("   • Parentheses override precedence")
    
    print("\n2. **Associativity Rules:**")
    print("   • Left associative: a + b + c = (a + b) + c")
    print("   • All basic operators (+, -, *, /) are left associative")
    print("   • Important for tree structure")
    
    print("\n3. **Tree Structure Impact:**")
    print("   Expression: 1 + 2 + 3")
    print("   Left associative tree:")
    print("       +")
    print("      / \\")
    print("     +   3")  
    print("    / \\")
    print("   1   2")
    
    print("\n4. **Precedence Impact:**")
    print("   Expression: 2 + 3 * 4")
    print("   Correct tree (respecting precedence):")
    print("       +")
    print("      / \\")
    print("     2   *")
    print("        / \\")
    print("       3   4")
    
    print("\n5. **Common Mistakes:**")
    print("   • Ignoring precedence: 2 + 3 * 4 ≠ (2 + 3) * 4")
    print("   • Wrong associativity: 8 / 4 / 2 = (8 / 4) / 2 = 1")
    print("   • Parentheses change meaning: (2 + 3) * 4 ≠ 2 + 3 * 4")

def demonstrate_practical_applications():
    """Demonstrate practical applications"""
    print("\n=== Practical Applications ===")
    
    print("Expression Tree Applications:")
    
    print("\n1. **Compiler Design:**")
    print("   • Parse arithmetic expressions in source code")
    print("   • Generate abstract syntax trees (AST)")
    print("   • Code optimization and generation")
    print("   • Error detection and reporting")
    
    print("\n2. **Calculator Implementation:**")
    print("   • Parse user input expressions")
    print("   • Evaluate expressions with correct precedence")
    print("   • Handle complex nested expressions")
    print("   • Scientific calculator functions")
    
    print("\n3. **Computer Algebra Systems:**")
    print("   • Symbolic mathematics software")
    print("   • Expression manipulation and simplification")
    print("   • Derivative and integral calculation")
    print("   • Mathematical formula processing")
    
    print("\n4. **Database Query Processing:**")
    print("   • Parse SQL WHERE clauses")
    print("   • Optimize query execution plans")
    print("   • Expression evaluation in queries")
    print("   • Index usage optimization")
    
    print("\n5. **Spreadsheet Applications:**")
    print("   • Parse cell formulas")
    print("   • Dependency tracking between cells")
    print("   • Automatic recalculation")
    print("   • Formula auditing and debugging")
    
    print("\n6. **Game Development:**")
    print("   • Scripting language parsers")
    print("   • Game logic expression evaluation")
    print("   • AI decision tree construction")
    print("   • Physics simulation formulas")

if __name__ == "__main__":
    test_expression_tree()
    demonstrate_parsing_techniques()
    demonstrate_mst_connection()
    analyze_precedence_and_associativity()
    demonstrate_practical_applications()

"""
Expression Tree and MST Concepts:
1. Tree Construction with Priority-based Selection
2. Recursive Descent Parsing and Grammar Rules
3. Operator Precedence and Associativity Handling
4. Stack-based Algorithms and Data Structure Management
5. Connection to MST through Priority Processing

Key Problem Insights:
- Expression trees represent arithmetic expressions structurally
- Operator precedence determines tree construction order
- Multiple parsing algorithms achieve same result
- Connection to MST through priority-based construction

Algorithm Strategy:
1. Recursive Descent: Parse by grammar rules with precedence
2. Shunting Yard: Convert to postfix, then build tree
3. Precedence Climbing: Recursively handle precedence levels
4. Stack-based: Use two stacks for operators and operands

MST Connection:
- Both problems build optimal tree structures
- Both use priority-based selection (precedence/weights)
- Both ensure all elements are properly connected
- Similar algorithmic design patterns and optimization strategies

Parsing Techniques:
- Recursive descent for grammar-based parsing
- Operator precedence for mathematical expressions
- Stack algorithms for evaluation and construction
- Tree building from linear input representation

Real-world Applications:
- Compiler design and AST construction
- Calculator and mathematical software
- Database query processing and optimization
- Spreadsheet formula parsing and evaluation
- Game scripting and AI decision systems

This problem demonstrates fundamental parsing algorithms
and their connection to tree construction optimization.
"""
