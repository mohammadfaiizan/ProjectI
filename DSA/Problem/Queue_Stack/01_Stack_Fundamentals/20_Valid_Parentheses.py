"""
20. Valid Parentheses - Multiple Approaches
Difficulty: Easy

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.
"""

from typing import Dict, List

class ValidParentheses:
    """Multiple approaches to validate parentheses"""
    
    def isValid_stack_approach(self, s: str) -> bool:
        """
        Approach 1: Classic Stack Approach
        
        Use stack to track opening brackets and match with closing ones.
        
        Time: O(n), Space: O(n)
        """
        if not s:
            return True
        
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:  # Closing bracket
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:  # Opening bracket
                stack.append(char)
        
        return len(stack) == 0
    
    def isValid_optimized_stack(self, s: str) -> bool:
        """
        Approach 2: Optimized Stack with Early Termination
        
        Optimize with early termination and length check.
        
        Time: O(n), Space: O(n)
        """
        # Odd length strings can't be valid
        if len(s) % 2 == 1:
            return False
        
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in s:
            if char in pairs:  # Opening bracket
                stack.append(char)
            else:  # Closing bracket
                if not stack or pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def isValid_counter_approach(self, s: str) -> bool:
        """
        Approach 3: Counter-based Approach (Limited to single type)
        
        Use counters for validation (works only for single bracket type).
        
        Time: O(n), Space: O(1)
        """
        # This approach only works if all brackets are of the same type
        # For demonstration purposes, let's assume only parentheses
        if not all(c in '()' for c in s):
            return self.isValid_stack_approach(s)  # Fallback to stack
        
        counter = 0
        for char in s:
            if char == '(':
                counter += 1
            elif char == ')':
                counter -= 1
                if counter < 0:  # More closing than opening
                    return False
        
        return counter == 0
    
    def isValid_replace_approach(self, s: str) -> bool:
        """
        Approach 4: String Replace Approach
        
        Repeatedly remove valid pairs until string is empty or no more pairs.
        
        Time: O(n²), Space: O(n)
        """
        while '()' in s or '[]' in s or '{}' in s:
            s = s.replace('()', '').replace('[]', '').replace('{}', '')
        
        return s == ''
    
    def isValid_recursive_approach(self, s: str) -> bool:
        """
        Approach 5: Recursive Approach
        
        Use recursion to validate nested structures.
        
        Time: O(n), Space: O(n)
        """
        def validate(s: str, index: int, stack: List[str]) -> bool:
            if index == len(s):
                return len(stack) == 0
            
            char = s[index]
            
            if char in '([{':
                stack.append(char)
                return validate(s, index + 1, stack)
            else:  # Closing bracket
                if not stack:
                    return False
                
                last_open = stack.pop()
                pairs = {'(': ')', '[': ']', '{': '}'}
                
                if pairs[last_open] != char:
                    return False
                
                return validate(s, index + 1, stack)
        
        return validate(s, 0, [])
    
    def isValid_state_machine(self, s: str) -> bool:
        """
        Approach 6: State Machine Approach
        
        Model validation as a state machine.
        
        Time: O(n), Space: O(n)
        """
        class StateMachine:
            def __init__(self):
                self.stack = []
                self.pairs = {')': '(', '}': '{', ']': '['}
            
            def process_char(self, char: str) -> bool:
                if char in '([{':
                    self.stack.append(char)
                    return True
                elif char in ')]}':
                    if not self.stack or self.stack.pop() != self.pairs[char]:
                        return False
                    return True
                return False  # Invalid character
            
            def is_valid_state(self) -> bool:
                return len(self.stack) == 0
        
        machine = StateMachine()
        
        for char in s:
            if not machine.process_char(char):
                return False
        
        return machine.is_valid_state()

def test_valid_parentheses():
    """Test valid parentheses algorithms"""
    solver = ValidParentheses()
    
    test_cases = [
        ("()", True, "Simple parentheses"),
        ("()[]{}", True, "Multiple types"),
        ("(]", False, "Mismatched brackets"),
        ("([)]", False, "Wrong order"),
        ("{[]}", True, "Nested brackets"),
        ("", True, "Empty string"),
        ("((", False, "Only opening"),
        ("))", False, "Only closing"),
        ("(())", True, "Nested parentheses"),
        ("([{}])", True, "Complex nesting"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.isValid_stack_approach),
        ("Optimized Stack", solver.isValid_optimized_stack),
        ("Counter Approach", solver.isValid_counter_approach),
        ("Replace Approach", solver.isValid_replace_approach),
        ("Recursive Approach", solver.isValid_recursive_approach),
        ("State Machine", solver.isValid_state_machine),
    ]
    
    print("=== Testing Valid Parentheses ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Input: '{s}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_valid_parentheses()

"""
Valid Parentheses demonstrates fundamental stack operations
for bracket matching, state machine design, and string
validation algorithms with multiple optimization techniques.
"""
