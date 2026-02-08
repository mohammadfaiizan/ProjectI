"""
Problem: Arithmetic Expression Conversion (Infix/Prefix/Postfix)
URL: https://www.geeksforgeeks.org/arithmetic-expression-evalution/

Problem Statement:
Implement all 6 conversions: infix-to-postfix, infix-to-prefix, prefix-to-infix, postfix-to-infix, prefix-to-postfix, postfix-to-prefix.

Sample Input/Output:
Input: "A+B*C"
Output (Infix to Postfix): "ABC*+"
Output (Infix to Prefix): "+A*BC"
"""


class Solution:
    def Precedence(self, op):
        """
        Get operator precedence.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if op == '^':
            return 3
        if op == '*' or op == '/':
            return 2
        if op == '+' or op == '-':
            return 1
        return 0

    def Is_Operator(self, c):
        """
        Check if character is operator.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return c == '+' or c == '-' or c == '*' or c == '/' or c == '^'

    def Infix_To_Postfix_Stack(self, infix):
        """
        Convert infix to postfix using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        postfix = ""
        for c in infix:
            if c.isalnum():
                postfix += c
            elif c == '(':
                st.append(c)
            elif c == ')':
                while st and st[-1] != '(':
                    postfix += st.pop()
                st.pop()
            elif self.Is_Operator(c):
                while st and st[-1] != '(' and self.Precedence(st[-1]) >= self.Precedence(c):
                    postfix += st.pop()
                st.append(c)
        while st:
            postfix += st.pop()
        return postfix

    def Infix_To_Prefix_Stack(self, infix):
        """
        Convert infix to prefix using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        infix = infix[::-1]
        infix_list = list(infix)
        for i in range(len(infix_list)):
            if infix_list[i] == '(':
                infix_list[i] = ')'
            elif infix_list[i] == ')':
                infix_list[i] = '('
        infix = "".join(infix_list)
        
        st = []
        prefix = ""
        for c in infix:
            if c.isalnum():
                prefix += c
            elif c == '(':
                st.append(c)
            elif c == ')':
                while st and st[-1] != '(':
                    prefix += st.pop()
                st.pop()
            elif self.Is_Operator(c):
                while st and st[-1] != '(' and self.Precedence(st[-1]) > self.Precedence(c):
                    prefix += st.pop()
                st.append(c)
        while st:
            prefix += st.pop()
        return prefix[::-1]

    def Prefix_To_Infix_Stack(self, prefix):
        """
        Convert prefix to infix using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        prefix = prefix[::-1]
        for c in prefix:
            if c.isalnum():
                st.append(c)
            elif self.Is_Operator(c):
                op1 = st.pop()
                op2 = st.pop()
                temp = "(" + op1 + c + op2 + ")"
                st.append(temp)
        return st[0]

    def Postfix_To_Infix_Stack(self, postfix):
        """
        Convert postfix to infix using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for c in postfix:
            if c.isalnum():
                st.append(c)
            elif self.Is_Operator(c):
                op2 = st.pop()
                op1 = st.pop()
                temp = "(" + op1 + c + op2 + ")"
                st.append(temp)
        return st[0]

    def Prefix_To_Postfix_Stack(self, prefix):
        """
        Convert prefix to postfix using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        prefix = prefix[::-1]
        for c in prefix:
            if c.isalnum():
                st.append(c)
            elif self.Is_Operator(c):
                op1 = st.pop()
                op2 = st.pop()
                temp = op1 + op2 + c
                st.append(temp)
        return st[0]

    def Postfix_To_Prefix_Stack(self, postfix):
        """
        Convert postfix to prefix using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for c in postfix:
            if c.isalnum():
                st.append(c)
            elif self.Is_Operator(c):
                op2 = st.pop()
                op1 = st.pop()
                temp = c + op1 + op2
                st.append(temp)
        return st[0]


def Test_Expression_Conversion():
    solution = Solution()
    
    print("=== Infix to Postfix ===")
    print(f"A+B*C -> {solution.Infix_To_Postfix_Stack('A+B*C')}")
    print(f"(A+B)*C -> {solution.Infix_To_Postfix_Stack('(A+B)*C')}")
    print(f"A+B*(C-D) -> {solution.Infix_To_Postfix_Stack('A+B*(C-D)')}")
    
    print("\n=== Infix to Prefix ===")
    print(f"A+B*C -> {solution.Infix_To_Prefix_Stack('A+B*C')}")
    print(f"(A+B)*C -> {solution.Infix_To_Prefix_Stack('(A+B)*C')}")
    
    print("\n=== Prefix to Infix ===")
    print(f"+A*BC -> {solution.Prefix_To_Infix_Stack('+A*BC')}")
    print(f"*+ABC -> {solution.Prefix_To_Infix_Stack('*+ABC')}")
    
    print("\n=== Postfix to Infix ===")
    print(f"ABC*+ -> {solution.Postfix_To_Infix_Stack('ABC*+')}")
    print(f"AB+C* -> {solution.Postfix_To_Infix_Stack('AB+C*')}")
    
    print("\n=== Prefix to Postfix ===")
    print(f"+A*BC -> {solution.Prefix_To_Postfix_Stack('+A*BC')}")
    print(f"*+ABC -> {solution.Prefix_To_Postfix_Stack('*+ABC')}")
    
    print("\n=== Postfix to Prefix ===")
    print(f"ABC*+ -> {solution.Postfix_To_Prefix_Stack('ABC*+')}")
    print(f"AB+C* -> {solution.Postfix_To_Prefix_Stack('AB+C*')}")


if __name__ == "__main__":
    Test_Expression_Conversion()
